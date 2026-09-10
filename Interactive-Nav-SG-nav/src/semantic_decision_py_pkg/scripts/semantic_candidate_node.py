#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

from semantic_decision_py_pkg.behavior_candidates import (
    BEHAVIOR_SCAN,
    BehaviorCandidate,
    CandidateGenerator,
    CandidateGeneratorConfig,
)
from semantic_decision_py_pkg.model_policy import compact_graph
from semantic_decision_py_pkg.frontier_terminal_contract import (
    summarize_frontier_filtering,
)
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311
from semantic_decision_py_pkg.startup_scan_lifecycle import StartupScanLifecycle
from semantic_decision_py_pkg.occupancy_path import (
    OccupancyTraversal,
    STATUS_OCCUPIED,
    STATUS_NO_PATH,
    STATUS_OUT_OF_BOUNDS,
    STATUS_SEARCH_LIMIT,
    STATUS_UNAVAILABLE,
    footprint_occupancy_status,
    grid_path_status,
    segment_occupancy_status,
)

patch_roslogging_findcaller_for_py311()

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String


def _unresolved_interaction_target_count(graph: dict) -> int:
    """Count public graph targets that still require interaction.

    An empty candidate list may only mean that a remembered target is out of
    view or temporarily on cooldown.  Keep that distinct from true mission
    exhaustion, while excluding executor-confirmed unavailable targets.
    """

    count = 0
    for node in list((graph or {}).get("nodes") or []):
        interaction = node.get("interaction") or {}
        requires_interaction = interaction.get(
            "requires_interaction", node.get("requires_interaction")
        )
        if not bool(requires_interaction):
            continue
        state = str(
            interaction.get("state", node.get("interaction_state")) or "unknown"
        ).casefold()
        capability = str(
            interaction.get("capability", node.get("interaction_capability")) or ""
        ).casefold()
        if state in {"open", "opened", "succeeded", "satisfied"}:
            continue
        if capability in {"blocked", "unavailable", "unsupported", "locked"}:
            continue
        count += 1
    return count


class SemanticCandidateNode:
    def __init__(self) -> None:
        rospy.init_node("semantic_candidate_node")
        topics = rospy.get_param("~topics", {}) or {}
        config = rospy.get_param("~candidate", {}) or {}
        policy_config = rospy.get_param("~policy", {}) or {}
        ablation_config = rospy.get_param("~ablation", {}) or {}
        configured_backend = str(policy_config.get("backend", "rule")).casefold()
        module2 = str(ablation_config.get("module2", "")).casefold()
        module1 = str(ablation_config.get("module1", "")).casefold()
        effective_policy_backend = (
            "model"
            if module2 == "mllm_score"
            else "rule"
            if module2 == "rule_cost"
            else configured_backend
        )
        unknown_portal_default = config.get("portal_unknown_default_interact")
        if unknown_portal_default is None:
            # Rule-only behavior: a restricted-GT portal remains an opaque
            # geometry observation, so the executor (not a hidden joint
            # state) resolves an unknown portal by attempting interaction.
            # Full MLLM runs keep their visual-evidence gate by default.
            unknown_portal_default = effective_policy_backend != "model"
        drawer_pre_action_mllm = config.get("drawer_pre_action_mllm")
        if drawer_pre_action_mllm is None:
            # Only the MLLM lane owns an image-derived drawer-front contract.
            # Rule baselines retain their existing explicit configuration rather
            # than silently depending on a disabled M1 worker.
            drawer_pre_action_mllm = module1 == "dynamic_mllm"
        container_pre_action_mllm = config.get("container_pre_action_mllm")
        if container_pre_action_mllm is None:
            # A model-run container needs a causally later visual front check
            # at the arrived ring pose.  Rule baselines keep their explicit
            # geometry/oracle configuration and do not depend on M1.
            container_pre_action_mllm = module1 == "dynamic_mllm"
        scan_config = rospy.get_param("~startup_scan", {}) or {}
        self.startup_scan_enabled = bool(scan_config.get("enabled", False))
        self.startup_scan_angle_rad = float(scan_config.get("angle_rad", 2.0 * 3.141592653589793))
        self.startup_scan_angular_speed_rad_s = float(
            scan_config.get("angular_speed_rad_s", 1.25)
        )
        self.startup_scan_control_dt_s = max(
            1e-3, float(scan_config.get("control_dt_s", 0.2))
        )
        self.startup_scan_timeout_s = max(0.0, float(scan_config.get("timeout_s", 15.0)))
        self.startup_scan_max_control_steps = max(
            1, int(scan_config.get("max_control_steps", 40))
        )
        self.startup_scan_lifecycle = StartupScanLifecycle(
            enabled=self.startup_scan_enabled
        )
        self.generator = CandidateGenerator(
            CandidateGeneratorConfig(
                max_frontier_candidates=int(config.get("max_frontier_candidates", 12)),
                interaction_types=tuple(
                    config.get("interaction_types", ["portal", "container"])
                ),
                interaction_semantic_types=tuple(
                    config.get("interaction_semantic_types", [])
                ),
                container_require_same_room=bool(
                    config.get("container_require_same_room", False)
                ),
                container_allow_connected_room=bool(
                    config.get("container_allow_connected_room", False)
                ),
                max_state_age_sec=float(config.get("max_state_age_sec", 300.0)),
                min_state_confidence=float(config.get("min_state_confidence", 0.5)),
                portal_require_attribute_ready=bool(
                    config.get("portal_require_attribute_ready", False)
                ),
                portal_allow_unknown_state=bool(
                    config.get("portal_allow_unknown_state", True)
                ),
                portal_unknown_default_interact=bool(unknown_portal_default),
                remembered_portal_reobservation_enabled=bool(
                    config.get("remembered_portal_reobservation_enabled", False)
                ),
                portal_standoff_m=float(config.get("portal_standoff_m", 1.0)),
                portal_obstacle_clearance_m=float(
                    config.get("portal_obstacle_clearance_m", 0.30)
                ),
                portal_obstacle_push_step_m=float(
                    config.get("portal_obstacle_push_step_m", 0.10)
                ),
                portal_obstacle_push_max_m=float(
                    config.get("portal_obstacle_push_max_m", 0.60)
                ),
                portal_approach_standoff_offsets_m=tuple(
                    # Negative offsets intentionally request closer portal
                    # stances; _approach_candidates clamps the resulting
                    # absolute standoff, not the offset itself.
                    float(value)
                    for value in config.get(
                        "portal_approach_standoff_offsets_m", [0.0, 0.25, 0.50]
                    )
                ),
                portal_approach_tangent_offsets_m=tuple(
                    float(value)
                    for value in config.get(
                        "portal_approach_tangent_offsets_m", [0.0, 0.20, -0.20]
                    )
                ),
                portal_approach_yaw_offsets_rad=tuple(
                    float(value)
                    for value in config.get(
                        "portal_approach_yaw_offsets_rad", [0.0]
                    )
                ),
                portal_opposite_side_fallback_enabled=bool(
                    config.get("portal_opposite_side_fallback_enabled", True)
                ),
                portal_require_reference_yaw=bool(
                    config.get("portal_require_reference_yaw", False)
                ),
                portal_side_hysteresis_m=max(
                    0.0, float(config.get("portal_side_hysteresis_m", 0.0))
                ),
                portal_traversal_distance_m=float(
                    config.get("portal_traversal_distance_m", 0.8)
                ),
                portal_traversal_max_start_distance_m=float(
                    config.get("portal_traversal_max_start_distance_m", 2.0)
                ),
                portal_traversal_completion_margin_m=float(
                    config.get("portal_traversal_completion_margin_m", 0.35)
                ),
                container_standoff_m=float(config.get("container_standoff_m", 1.0)),
                fridge_standoff_m=(
                    float(config["fridge_standoff_m"])
                    if config.get("fridge_standoff_m") is not None
                    else None
                ),
                fridge_multiview_angular_scale=max(
                    0.0,
                    min(1.0, float(config.get("fridge_multiview_angular_scale", 1.0))),
                ),
                container_m1_face_selection_enabled=bool(
                    config.get("container_m1_face_selection_enabled", False)
                ),
                drawer_pre_action_mllm=bool(drawer_pre_action_mllm),
                drawer_pre_action_observation_max_attempts=max(
                    1,
                    int(
                        config.get(
                            "drawer_pre_action_observation_max_attempts", 2
                        )
                    ),
                ),
                container_pre_action_mllm=bool(container_pre_action_mllm),
                fridge_direct_interaction_on_arrival=bool(
                    config.get("fridge_direct_interaction_on_arrival", False)
                ),
                container_pre_action_observation_max_attempts=max(
                    1,
                    int(
                        config.get(
                            "container_pre_action_observation_max_attempts", 4
                        )
                    ),
                ),
                drawer_standoff_m=(
                    float(config["drawer_standoff_m"])
                    if config.get("drawer_standoff_m") is not None
                    else None
                ),
                container_action_standoff_m=(
                    float(config["container_action_standoff_m"])
                    if config.get("container_action_standoff_m") is not None
                    else None
                ),
                drawer_action_standoff_m=(
                    float(config["drawer_action_standoff_m"])
                    if config.get("drawer_action_standoff_m") is not None
                    else None
                ),
                fridge_action_standoff_m=(
                    float(config["fridge_action_standoff_m"])
                    if config.get("fridge_action_standoff_m") is not None
                    else None
                ),
                container_m1_capture_standoff_m=(
                    float(config["container_m1_capture_standoff_m"])
                    if config.get("container_m1_capture_standoff_m") is not None
                    else None
                ),
                drawer_m1_capture_standoff_m=(
                    float(config["drawer_m1_capture_standoff_m"])
                    if config.get("drawer_m1_capture_standoff_m") is not None
                    else None
                ),
                fridge_m1_capture_standoff_m=(
                    float(config["fridge_m1_capture_standoff_m"])
                    if config.get("fridge_m1_capture_standoff_m") is not None
                    else None
                ),
                container_observation_standoff_m=float(
                    config.get("container_observation_standoff_m", 0.7)
                ),
                drawer_observation_standoff_m=float(
                    config.get("drawer_observation_standoff_m", 0.65)
                ),
                fridge_observation_standoff_m=float(
                    config.get("fridge_observation_standoff_m", 0.8)
                ),
                container_navigation_anchor_outer_offset_m=(
                    float(config["container_navigation_anchor_outer_offset_m"])
                    if config.get("container_navigation_anchor_outer_offset_m")
                    is not None
                    else None
                ),
                container_navigation_anchor_ring_count=(
                    int(config["container_navigation_anchor_ring_count"])
                    if config.get("container_navigation_anchor_ring_count")
                    is not None
                    else None
                ),
                container_navigation_anchor_tangent_offset_m=(
                    float(config["container_navigation_anchor_tangent_offset_m"])
                    if config.get("container_navigation_anchor_tangent_offset_m")
                    is not None
                    else None
                ),
                container_anchor_shared_pose_enabled=bool(
                    config.get("container_anchor_shared_pose_enabled", False)
                ),
                drawer_navigation_anchor_aabb_fan_enabled=bool(
                    config.get("drawer_navigation_anchor_aabb_fan_enabled", False)
                ),
                drawer_navigation_anchor_fan_clearances_m=tuple(
                    float(value)
                    for value in config.get(
                        "drawer_navigation_anchor_fan_clearances_m",
                        [0.50, 0.85, 1.20],
                    )
                ),
                drawer_navigation_anchor_fan_angles_deg=tuple(
                    float(value)
                    for value in config.get(
                        "drawer_navigation_anchor_fan_angles_deg",
                        [-30.0, -15.0, 0.0, 15.0, 30.0],
                    )
                ),
                fridge_navigation_anchor_aabb_fan_enabled=bool(
                    config.get("fridge_navigation_anchor_aabb_fan_enabled", False)
                ),
                fridge_navigation_anchor_fan_clearances_m=tuple(
                    float(value)
                    for value in config.get(
                        "fridge_navigation_anchor_fan_clearances_m",
                        [1.15, 1.35, 1.55],
                    )
                ),
                fridge_navigation_anchor_fan_angles_deg=tuple(
                    float(value)
                    for value in config.get(
                        "fridge_navigation_anchor_fan_angles_deg",
                        [-15.0, 0.0, 15.0],
                    )
                ),
                container_safe_staging_outer_offset_m=float(
                    config.get("container_safe_staging_outer_offset_m", 0.35)
                ),
                container_safe_staging_ring_count=max(
                    1,
                    min(
                        4,
                        int(config.get("container_safe_staging_ring_count", 3)),
                    ),
                ),
                container_safe_staging_tangent_offset_m=max(
                    0.0,
                    float(
                        config.get("container_safe_staging_tangent_offset_m", 0.0)
                    ),
                ),
                container_two_stage_observation_max_attempts=max(
                    1,
                    int(
                        config.get(
                            "container_two_stage_observation_max_attempts", 12
                        )
                    ),
                ),
                container_m1_same_pose_samples_per_view=max(
                    1,
                    int(
                        config.get(
                            "container_m1_same_pose_samples_per_view", 2
                        )
                    ),
                ),
                container_m1_max_viewpoints=max(
                    1, int(config.get("container_m1_max_viewpoints", 4))
                ),
                container_m1_max_total_requests=max(
                    1, int(config.get("container_m1_max_total_requests", 8))
                ),
                container_safe_staging_arrival_tolerance_m=float(
                    config.get("container_safe_staging_arrival_tolerance_m", 0.30)
                ),
                container_interaction_ready_distance_m=float(
                    config.get("container_interaction_ready_distance_m", 0.18)
                ),
                container_action_lateral_offset_m=max(
                    0.0,
                    float(config.get("container_action_lateral_offset_m", 0.22)),
                ),
                container_m1_front_axis_from_capture=bool(
                    config.get("container_m1_front_axis_from_capture", True)
                ),
                interaction_safety_margin_m=float(
                    config.get("interaction_safety_margin_m", 0.0)
                ),
                interaction_ready_distance_m=float(
                    config.get("interaction_ready_distance_m", 0.45)
                ),
                interaction_ready_yaw_tolerance_rad=max(
                    0.05,
                    float(config.get("interaction_ready_yaw_tolerance_rad", 0.55)),
                ),
                require_current_visibility=bool(
                    config.get("require_current_visibility", False)
                ),
                portal_require_current_visibility=bool(
                    config.get("portal_require_current_visibility", False)
                ),
                portal_min_visible_pixels=int(
                    config.get("portal_min_visible_pixels", 128)
                ),
                portal_min_visible_fraction=float(
                    config.get("portal_min_visible_fraction", 0.2)
                ),
                target_standoff_m=float(config.get("target_standoff_m", 1.0)),
                target_max_state_age_sec=float(
                    config.get("target_max_state_age_sec", 300.0)
                ),
                target_require_current_visibility=bool(
                    config.get("target_require_current_visibility", False)
                ),
                target_require_same_room=bool(
                    config.get("target_require_same_room", False)
                ),
                target_allow_connected_room=bool(
                    config.get("target_allow_connected_room", False)
                ),
                target_require_visibility_verification=bool(
                    config.get("target_require_visibility_verification", True)
                ),
                target_min_visible_pixels=int(
                    config.get("target_min_visible_pixels", 16)
                ),
                target_min_visible_fraction=float(
                    config.get("target_min_visible_fraction", 0.2)
                ),
                target_min_consecutive_observations=int(
                    config.get("target_min_consecutive_observations", 2)
                ),
                target_arrival_tolerance_m=float(
                    config.get("target_arrival_tolerance_m", 0.35)
                ),
            )
        )
        self.explorer_status: dict = {}
        self.explorer_proposal_stream: dict = {}
        self.has_proposal_stream = False
        self.graph: dict = {}
        self.target_context: dict = dict(rospy.get_param("~target", {}) or {})
        self._target_lock = threading.Lock()
        self.target_revision = 0
        # ROS callbacks and the one-second candidate timer run on separate
        # callback threads in a multi-threaded spinner.  Keep the map/odom
        # payload and its timing metadata as one coherent snapshot so a map
        # from one instant cannot be paired with a pose from another instant.
        self._occupancy_state_lock = threading.RLock()
        self.robot_xy: tuple[float, float] | None = None
        self.occupancy_grid: OccupancyGrid | None = None
        self.robot_xy_stamp_s: float | None = None
        self.robot_xy_received_at: float | None = None
        self.occupancy_grid_stamp_s: float | None = None
        self.occupancy_grid_received_at: float | None = None
        self.portal_goal_require_known_free = bool(
            config.get("portal_goal_require_known_free", False)
        )
        self.portal_goal_known_free_radius_m = max(
            0.0, float(config.get("portal_goal_known_free_radius_m", 0.30))
        )
        self.portal_goal_occupied_threshold = max(
            1, int(config.get("portal_goal_occupied_threshold", 50))
        )
        self.filtered_portal_candidate_ids: list[str] = []
        # This is a deliberately cheap candidate-level guard before the
        # executor's authoritative move_base/make_plan check.  It is disabled
        # by default so simulator/replay behavior is unchanged; the physical
        # override enables it.  Unknown cells are retained, while explicitly
        # occupied or out-of-bounds samples fail closed.
        self.candidate_occupancy_preflight_enabled = bool(
            config.get("candidate_occupancy_preflight_enabled", False)
        )
        self.candidate_occupancy_reject_occupied = bool(
            config.get("candidate_occupancy_reject_occupied", True)
        )
        self.candidate_occupancy_sample_step_m = max(
            0.02, float(config.get("candidate_occupancy_sample_step_m", 0.10))
        )
        self.candidate_occupancy_start_ignore_m = max(
            0.0, float(config.get("candidate_occupancy_start_ignore_m", 0.35))
        )
        self.candidate_occupancy_endpoint_radius_m = max(
            0.0, float(config.get("candidate_occupancy_endpoint_radius_m", 0.25))
        )
        self.candidate_occupancy_threshold = max(
            1, int(config.get("candidate_occupancy_threshold", 50))
        )
        # OCC preflight is only authoritative when the map and pose belong to
        # roughly the same capture interval.  A stale/mismatched pair is
        # deliberately treated as unknown and candidates are retained for the
        # executor's authoritative planner.  ``<= 0`` disables this guard for
        # replay/debug configurations that do not carry reliable timestamps.
        self.candidate_occupancy_max_pair_age_s = max(
            0.0,
            float(config.get("candidate_occupancy_max_pair_age_s", 1.0)),
        )
        # Check the whole route with the robot footprint, including apparently
        # free centre-lines through narrow passages. A shared Dijkstra search
        # avoids repeated expansions and a ``make_plan`` RPC per anchor. It is
        # disabled by default for simulator compatibility and enabled by the
        # physical override.
        self.candidate_occupancy_use_astar = bool(
            config.get("candidate_occupancy_use_astar", False)
        )
        self.candidate_occupancy_astar_max_expansions = max(
            100,
            int(config.get("candidate_occupancy_astar_max_expansions", 12000)),
        )
        self.filtered_occupancy_candidate_ids: list[str] = []
        self.occupancy_preflight_timing: dict[str, object] = {
            "enabled": self.candidate_occupancy_preflight_enabled,
            "map_available": False,
            "robot_available": False,
            "pair_available": False,
            "pair_accepted": False,
            "pair_reason": "missing_map_or_odom",
            "pair_age_s": None,
            "source_stamp_delta_s": None,
            "receipt_delta_s": None,
            "latest_age_s": None,
            "max_pair_age_s": self.candidate_occupancy_max_pair_age_s,
            "elapsed_ms": 0.0,
            "candidate_count_checked": 0,
            "candidate_count_rejected": 0,
            "option_count_checked": 0,
            "option_count_rejected": 0,
            "unknown_option_count": 0,
            "astar_option_count": 0,
            "astar_reachable_count": 0,
            "astar_no_path_count": 0,
            "astar_search_limit_count": 0,
        }
        self.sequence = 0
        self.publisher = rospy.Publisher(
            topics.get("candidates", "/semantic_decision/candidates"),
            String,
            queue_size=1,
            latch=True,
        )
        rospy.Subscriber(
            topics.get("explorer_status", "/explore_py/status"),
            String,
            self._explorer_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get("explorer_proposals", "/explore_py/proposals"),
            String,
            self._proposal_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get("unified_graph", "/semantic_mapping/unified_graph"),
            String,
            self._graph_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get("target_context", "/semantic_decision/target"),
            String,
            self._target_callback,
            queue_size=2,
        )
        rospy.Subscriber(
            topics.get("odom", "/odom"), Odometry, self._odom_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.get("occupancy_grid", "/struct_mapping/occ_map"),
            OccupancyGrid,
            self._occupancy_callback,
            queue_size=1,
        )
        if self.startup_scan_enabled:
            rospy.Subscriber(
                topics.get("behavior_feedback", "/semantic_decision/behavior_feedback"),
                String,
                self._startup_scan_feedback_callback,
                queue_size=10,
            )
        self.timer = rospy.Timer(rospy.Duration(1.0), self._publish)

    def _explorer_callback(self, message: String) -> None:
        if self.has_proposal_stream:
            return
        try:
            self.explorer_status = json.loads(message.data)
        except json.JSONDecodeError:
            return

    def _proposal_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        self.explorer_proposal_stream = payload
        self.has_proposal_stream = True

    def _graph_callback(self, message: String) -> None:
        try:
            graph = json.loads(message.data)
        except json.JSONDecodeError:
            return
        episode_id = str(graph.get("episode_id") or "")
        self.graph = graph
        # The first graph id can arrive after the map-ready signal.  Bind it to
        # the existing SCAN instance; only a later different non-empty id is a
        # new episode that must create a new mandatory scan.
        self.startup_scan_lifecycle.observe_episode(episode_id)

    def _startup_scan_candidate_id(self) -> str:
        return self.startup_scan_lifecycle.candidate_id

    def _startup_scan_candidate(self) -> BehaviorCandidate:
        return BehaviorCandidate(
            candidate_id=self._startup_scan_candidate_id(),
            behavior_type=BEHAVIOR_SCAN,
            source="mandatory_startup_scan",
            target_id="startup_scan",
            target_name="Initial 360-degree scan",
            features={"priority": 1.0, "information_gain": 1.0},
            metadata={
                "mandatory_startup_scan": True,
                "startup_scan_instance_id": self._startup_scan_candidate_id(),
                "startup_scan_bound_episode_id": (
                    self.startup_scan_lifecycle.bound_episode_id
                ),
                "target_angle_rad": self.startup_scan_angle_rad,
                "angular_speed_rad_s": self.startup_scan_angular_speed_rad_s,
                "control_dt_s": self.startup_scan_control_dt_s,
                "timeout_s": self.startup_scan_timeout_s,
                "max_control_steps": self.startup_scan_max_control_steps,
            },
        )

    def _startup_scan_feedback_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if str(payload.get("behavior_type") or "").upper() != BEHAVIOR_SCAN:
            return
        self.startup_scan_lifecycle.record_feedback(
            payload.get("candidate_id"),
            payload.get("status"),
            payload.get("detail") or {},
        )

    @staticmethod
    def _message_stamp_s(message: object) -> float | None:
        """Return a finite positive ROS header stamp when one is available."""

        try:
            header = getattr(message, "header", None)
            stamp = getattr(header, "stamp", None)
            if stamp is None:
                return None
            to_sec = getattr(stamp, "to_sec", None)
            if callable(to_sec):
                stamp = to_sec()
            value = float(stamp)
        except (AttributeError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0.0 else None

    def _occupancy_state_snapshot(self) -> dict[str, object]:
        """Capture OCC/odom values and timing metadata atomically."""

        lock = getattr(self, "_occupancy_state_lock", None)
        if lock is None:
            return {
                "robot_xy": getattr(self, "robot_xy", None),
                "occupancy_grid": getattr(self, "occupancy_grid", None),
                "robot_xy_stamp_s": getattr(self, "robot_xy_stamp_s", None),
                "robot_xy_received_at": getattr(
                    self, "robot_xy_received_at", None
                ),
                "occupancy_grid_stamp_s": getattr(
                    self, "occupancy_grid_stamp_s", None
                ),
                "occupancy_grid_received_at": getattr(
                    self, "occupancy_grid_received_at", None
                ),
            }
        with lock:
            return {
                "robot_xy": self.robot_xy,
                "occupancy_grid": self.occupancy_grid,
                "robot_xy_stamp_s": self.robot_xy_stamp_s,
                "robot_xy_received_at": self.robot_xy_received_at,
                "occupancy_grid_stamp_s": self.occupancy_grid_stamp_s,
                "occupancy_grid_received_at": self.occupancy_grid_received_at,
            }

    def _occupancy_pair_status(
        self,
        *,
        state: dict[str, object] | None = None,
        now: float | None = None,
    ) -> dict[str, object]:
        """Check whether the latest OCC and odom can safely be paired.

        ``pair_age_s`` is the largest of source-stamp skew, callback-receipt
        skew, and age of either receipt.  If the ROS source stamp is absent,
        receipt timing is used instead.  A failed check is intentionally
        non-destructive: callers must retain candidates and defer the final
        decision to the authoritative planner.
        """

        state = state or self._occupancy_state_snapshot()
        grid = state.get("occupancy_grid")
        robot_xy = state.get("robot_xy")
        result: dict[str, object] = {
            "available": bool(grid is not None and robot_xy is not None),
            "paired": False,
            "reason": "missing_map_or_odom",
            "pair_age_s": None,
            "source_stamp_delta_s": None,
            "receipt_delta_s": None,
            "latest_age_s": None,
        }
        if grid is None or robot_xy is None:
            return result
        max_age = max(
            0.0,
            float(getattr(self, "candidate_occupancy_max_pair_age_s", 1.0)),
        )
        if max_age <= 0.0:
            # Explicitly disabled pairing is useful for replay data that has
            # no trustworthy source or receipt timestamps.
            result.update({"paired": True, "reason": "age_check_disabled"})
            return result
        occupancy_received = state.get("occupancy_grid_received_at")
        odom_received = state.get("robot_xy_received_at")
        if occupancy_received is None or odom_received is None:
            result["reason"] = "missing_receipt_timestamps"
            return result
        try:
            occupancy_received = float(occupancy_received)
            odom_received = float(odom_received)
        except (TypeError, ValueError):
            result["reason"] = "invalid_receipt_timestamps"
            return result
        if not all(math.isfinite(value) for value in (occupancy_received, odom_received)):
            result["reason"] = "invalid_receipt_timestamps"
            return result
        now_value = time.monotonic() if now is None else float(now)
        if not math.isfinite(now_value):
            now_value = time.monotonic()
        receipt_delta = abs(occupancy_received - odom_received)
        latest_age = max(
            0.0,
            now_value - occupancy_received,
            now_value - odom_received,
        )
        result["receipt_delta_s"] = receipt_delta
        result["latest_age_s"] = latest_age

        occupancy_stamp = state.get("occupancy_grid_stamp_s")
        odom_stamp = state.get("robot_xy_stamp_s")
        source_delta: float | None = None
        if occupancy_stamp is not None and odom_stamp is not None:
            try:
                occupancy_stamp = float(occupancy_stamp)
                odom_stamp = float(odom_stamp)
            except (TypeError, ValueError):
                occupancy_stamp = odom_stamp = None
            if (
                occupancy_stamp is not None
                and odom_stamp is not None
                and math.isfinite(occupancy_stamp)
                and math.isfinite(odom_stamp)
            ):
                source_delta = abs(occupancy_stamp - odom_stamp)
        result["source_stamp_delta_s"] = source_delta
        pair_age = max(
            value for value in (receipt_delta, latest_age, source_delta or 0.0)
        )
        result["pair_age_s"] = pair_age
        if pair_age > max_age:
            result["reason"] = "pair_age_exceeded"
            return result
        result.update({"paired": True, "reason": "paired"})
        return result

    @staticmethod
    def _grid_shape_and_data(grid: object) -> tuple[int, int, object] | None:
        """Validate OccupancyGrid dimensions before indexing its data array."""

        try:
            info = getattr(grid, "info")
            width = int(getattr(info, "width"))
            height = int(getattr(info, "height"))
            data = getattr(grid, "data")
            expected = width * height
            if width <= 0 or height <= 0 or expected <= 0 or len(data) < expected:
                return None
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        return width, height, data

    def _target_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            with self._target_lock:
                if payload != self.target_context:
                    self.target_context = payload
                    self.target_revision += 1

    def _target_snapshot(self) -> tuple[int, dict]:
        with self._target_lock:
            # Callbacks replace this decoded JSON object; they never mutate it.
            return self.target_revision, dict(self.target_context)

    def _publish_target_snapshot(self, payload: dict, revision: int) -> bool:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._target_lock:
            if revision != self.target_revision:
                return False
            self.publisher.publish(String(data=encoded))
            return True

    def _odom_callback(self, message: Odometry) -> None:
        robot_xy = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
        )
        received_at = time.monotonic()
        stamp_s = self._message_stamp_s(message)
        with self._occupancy_state_lock:
            self.robot_xy = robot_xy
            self.robot_xy_stamp_s = stamp_s
            self.robot_xy_received_at = received_at

    def _occupancy_callback(self, message: OccupancyGrid) -> None:
        received_at = time.monotonic()
        stamp_s = self._message_stamp_s(message)
        with self._occupancy_state_lock:
            self.occupancy_grid = message
            self.occupancy_grid_stamp_s = stamp_s
            self.occupancy_grid_received_at = received_at

    def _goal_is_known_free(self, goal: list[float]) -> bool:
        state = self._occupancy_state_snapshot()
        grid = state.get("occupancy_grid")
        if grid is None or len(goal or []) < 2:
            return False
        shape = self._grid_shape_and_data(grid)
        if shape is None:
            # A partially received/truncated OccupancyGrid must never make the
            # candidate timer raise IndexError.  Treat it as unknown instead.
            return False
        width, height, data = shape
        try:
            origin = grid.info.origin
            q = origin.orientation
            origin_x = float(origin.position.x)
            origin_y = float(origin.position.y)
            qx = float(q.x)
            qy = float(q.y)
            qz = float(q.z)
            qw = float(q.w)
            goal_x = float(goal[0])
            goal_y = float(goal[1])
        except AttributeError:
            return False
        except (TypeError, ValueError):
            return False
        if not all(
            math.isfinite(value)
            for value in (origin_x, origin_y, qx, qy, qz, qw, goal_x, goal_y)
        ):
            return False
        origin_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        dx = goal_x - origin_x
        dy = goal_y - origin_y
        cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        try:
            resolution = max(float(grid.info.resolution), 1e-6)
        except (AttributeError, TypeError, ValueError):
            return False
        center_x = int(math.floor(local_x / resolution))
        center_y = int(math.floor(local_y / resolution))
        radius_cells = int(math.ceil(self.portal_goal_known_free_radius_m / resolution))
        for offset_y in range(-radius_cells, radius_cells + 1):
            for offset_x in range(-radius_cells, radius_cells + 1):
                if math.hypot(offset_x, offset_y) * resolution > self.portal_goal_known_free_radius_m + 1e-9:
                    continue
                grid_x, grid_y = center_x + offset_x, center_y + offset_y
                if not (0 <= grid_x < width and 0 <= grid_y < height):
                    return False
                try:
                    value = int(data[grid_y * width + grid_x])
                except (IndexError, TypeError, ValueError):
                    return False
                if value < 0 or value >= self.portal_goal_occupied_threshold:
                    return False
        return True

    def _filter_portal_goals_by_occupancy(
        self, candidates: list[BehaviorCandidate]
    ) -> list[BehaviorCandidate]:
        self.filtered_portal_candidate_ids = []
        pair_state = self._occupancy_state_snapshot()
        occupancy_grid = pair_state.get("occupancy_grid")
        if occupancy_grid is None:
            return candidates
        if self._grid_shape_and_data(occupancy_grid) is None:
            # An incomplete map is unavailable, not evidence that every
            # portal stance is unsafe.  Leave the candidate for move_base.
            return candidates
        # Obstacle pushing and known-free checks are hard filters.  Do not
        # apply either to a map that cannot be causally paired with odom; the
        # authoritative move_base planner will re-check the candidate later.
        if not self._occupancy_pair_status(state=pair_state).get("paired", False):
            return candidates
        kept = []
        for candidate in candidates:
            metadata = candidate.metadata or {}
            if candidate.behavior_type != "INTERACT" or metadata.get("node_type") != "portal":
                kept.append(candidate)
                continue
            goals = list(metadata.get("goal_xyyaw_candidates") or [])
            labels = list(metadata.get("interaction_approach_pose_labels") or [])
            valid_pairs = []
            push_distances = []
            for index, goal in enumerate(goals):
                adjusted_goal, pushed_m = self._push_portal_goal_from_obstacle(
                    goal,
                    metadata,
                    occupancy_grid=occupancy_grid,
                )
                if self.portal_goal_require_known_free and not self._goal_is_known_free(
                    adjusted_goal
                ):
                    continue
                valid_pairs.append(
                    (
                        adjusted_goal,
                        labels[index] if index < len(labels) else "portal_source_side",
                    )
                )
                push_distances.append(float(pushed_m))
            if not valid_pairs:
                self.filtered_portal_candidate_ids.append(candidate.candidate_id)
                continue
            valid_goals = [list(pair[0]) for pair in valid_pairs]
            valid_labels = [pair[1] for pair in valid_pairs]
            candidate.goal_xyyaw = list(valid_goals[0])
            metadata["goal_xyyaw_candidates"] = valid_goals
            metadata["interaction_approach_pose_labels"] = valid_labels
            metadata["portal_obstacle_push_m"] = (
                push_distances[0] if push_distances else 0.0
            )
            metadata["portal_obstacle_push_applied"] = bool(
                any(distance > 1e-6 for distance in push_distances)
            )
            metadata["portal_goal_known_free"] = bool(self.portal_goal_require_known_free)
            candidate.metadata = metadata
            if candidate.interaction_command is not None:
                candidate.interaction_command["interaction_approach_pose_xyyaw"] = list(
                    valid_goals[0]
                )
                candidate.interaction_command["interaction_approach_pose_labels"] = valid_labels
            kept.append(candidate)
        return kept

    @staticmethod
    def _finite_goal_option(goal: object) -> list[float] | None:
        """Normalize a candidate goal without making malformed data unsafe."""

        values = list(goal or []) if isinstance(goal, (list, tuple)) else []
        if len(values) < 2:
            return None
        try:
            normalized = [float(values[0]), float(values[1])]
            normalized.append(float(values[2]) if len(values) >= 3 else 0.0)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in normalized):
            return None
        return normalized

    def _filter_candidate_goals_by_occupancy(
        self, candidates: list[BehaviorCandidate]
    ) -> list[BehaviorCandidate]:
        """Check anchor endpoints and routes against one coherent OCC receipt.

        This guard intentionally runs before the executor's synchronous
        ``move_base/make_plan`` call.  It only rejects a candidate when the
        latest map explicitly reports an unsafe footprint or no connecting
        route. With grid search disabled, only the legacy straight-ray check
        is used. Unknown cells are retained and are reported in
        candidate metadata, because an incomplete map must not make the public
        candidate pool disappear.  A missing map/odom is also retained and
        leaves the authoritative planner as the final safety check.
        """

        started = time.perf_counter()
        pair_state = self._occupancy_state_snapshot()
        occupancy_grid = pair_state.get("occupancy_grid")
        robot_xy = pair_state.get("robot_xy")
        pair_status = self._occupancy_pair_status(state=pair_state)
        self.filtered_occupancy_candidate_ids = []
        timing: dict[str, object] = {
            "enabled": bool(self.candidate_occupancy_preflight_enabled),
            "map_available": occupancy_grid is not None,
            "robot_available": robot_xy is not None,
            "pair_available": bool(pair_status.get("available", False)),
            "pair_accepted": bool(pair_status.get("paired", False)),
            "pair_reason": str(pair_status.get("reason") or ""),
            "pair_age_s": pair_status.get("pair_age_s"),
            "source_stamp_delta_s": pair_status.get("source_stamp_delta_s"),
            "receipt_delta_s": pair_status.get("receipt_delta_s"),
            "latest_age_s": pair_status.get("latest_age_s"),
            "max_pair_age_s": float(
                getattr(self, "candidate_occupancy_max_pair_age_s", 1.0)
            ),
            "elapsed_ms": 0.0,
            "candidate_count_checked": 0,
            "candidate_count_rejected": 0,
            "option_count_checked": 0,
            "option_count_rejected": 0,
            "unknown_option_count": 0,
            "astar_option_count": 0,
            "astar_reachable_count": 0,
            "astar_no_path_count": 0,
            "astar_search_limit_count": 0,
        }

        def finish(result: list[BehaviorCandidate]) -> list[BehaviorCandidate]:
            timing["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            self.occupancy_preflight_timing = timing
            return result

        if not self.candidate_occupancy_preflight_enabled:
            return finish(candidates)
        if occupancy_grid is None or robot_xy is None:
            # Startup/map refresh races are deliberately non-destructive.  A
            # later publish tick will retry with the newest map and odometry.
            return finish(candidates)
        if not bool(pair_status.get("paired", False)):
            # Never fail closed on a stale/mixed OCC+odom pair.  Keeping the
            # candidate lets the executor's make_plan/costmap check decide
            # once a coherent pair is available.
            return finish(candidates)

        kept: list[BehaviorCandidate] = []
        traversal = OccupancyTraversal(
            occupancy_grid,
            robot_radius_m=self.candidate_occupancy_endpoint_radius_m,
            occupied_threshold=self.candidate_occupancy_threshold,
        ) if self.candidate_occupancy_use_astar else None
        hard_statuses = {STATUS_OCCUPIED, STATUS_OUT_OF_BOUNDS, STATUS_NO_PATH}
        for candidate in candidates:
            behavior_type = str(candidate.behavior_type or "").upper()
            if behavior_type not in {"EXPLORE", "INTERACT", "NAVIGATE"}:
                kept.append(candidate)
                continue

            primary = self._finite_goal_option(candidate.goal_xyyaw)
            metadata = dict(candidate.metadata or {})
            raw_options = list(metadata.get("goal_xyyaw_candidates") or [])
            options: list[list[float]] = []
            if primary is not None:
                options.append(primary)
            for raw_option in raw_options:
                option = self._finite_goal_option(raw_option)
                if option is None:
                    continue
                if any(
                    math.hypot(option[0] - prior[0], option[1] - prior[1]) <= 1e-6
                    and abs(option[2] - prior[2]) <= 1e-6
                    for prior in options
                ):
                    continue
                options.append(option)
            if not options:
                kept.append(candidate)
                continue

            timing["candidate_count_checked"] = int(
                timing["candidate_count_checked"]
            ) + 1
            option_checks: list[dict[str, object]] = []
            valid_indices: list[int] = []
            rejected_indices: list[int] = []
            for option_index, option in enumerate(options):
                endpoint_status = (
                    traversal.point_state(option[0], option[1])
                    if traversal is not None
                    else footprint_occupancy_status(
                        occupancy_grid,
                        option[0],
                        option[1],
                        radius_m=self.candidate_occupancy_endpoint_radius_m,
                        occupied_threshold=self.candidate_occupancy_threshold,
                    )
                )
                path_status = segment_occupancy_status(
                    occupancy_grid,
                    robot_xy,
                    option,
                    sample_step_m=self.candidate_occupancy_sample_step_m,
                    start_ignore_m=self.candidate_occupancy_start_ignore_m,
                    occupied_threshold=self.candidate_occupancy_threshold,
                )
                path_state = str(path_status.get("status") or STATUS_UNAVAILABLE)
                path_unknown = bool(path_status.get("unknown", False))
                astar_status: dict[str, object] | None = None
                # A free centre-line is not proof of clearance for the body.
                # Share inflated-cell results across every anchor in this map
                # snapshot and reuse a source-rooted search across goals.
                if (
                    self.candidate_occupancy_use_astar
                    and endpoint_status not in hard_statuses
                ):
                    astar_status = grid_path_status(
                        occupancy_grid,
                        robot_xy,
                        option,
                        robot_radius_m=self.candidate_occupancy_endpoint_radius_m,
                        occupied_threshold=self.candidate_occupancy_threshold,
                        unknown_is_blocked=False,
                        max_expansions=self.candidate_occupancy_astar_max_expansions,
                        traversal=traversal,
                    )
                    timing["astar_option_count"] = int(
                        timing["astar_option_count"]
                    ) + 1
                    astar_state = str(astar_status.get("status") or "")
                    path_state = astar_state or STATUS_UNAVAILABLE
                    path_unknown = bool(astar_status.get("unknown", False))
                    if bool(astar_status.get("reachable")):
                        timing["astar_reachable_count"] = int(
                            timing["astar_reachable_count"]
                        ) + 1
                        path_state = str(astar_status.get("status") or "free")
                    elif astar_state == STATUS_NO_PATH:
                        timing["astar_no_path_count"] = int(
                            timing["astar_no_path_count"]
                        ) + 1
                    elif astar_state == STATUS_SEARCH_LIMIT:
                        timing["astar_search_limit_count"] = int(
                            timing["astar_search_limit_count"]
                        ) + 1
                        # Search-limit is inconclusive, not a proof of a
                        # collision. Preserve the candidate for the
                        # authoritative planner and expose the uncertainty.
                        path_state = STATUS_SEARCH_LIMIT
                        path_unknown = True
                hard_status = (
                    endpoint_status
                    if endpoint_status in hard_statuses
                    else path_state
                    if path_state in hard_statuses
                    else ""
                )
                unknown = endpoint_status == "unknown" or path_unknown
                if unknown:
                    timing["unknown_option_count"] = int(
                        timing["unknown_option_count"]
                    ) + 1
                hard_rejected = bool(hard_status) and bool(
                    self.candidate_occupancy_reject_occupied
                )
                timing["option_count_checked"] = int(
                    timing["option_count_checked"]
                ) + 1
                if hard_rejected:
                    rejected_indices.append(option_index)
                    timing["option_count_rejected"] = int(
                        timing["option_count_rejected"]
                    ) + 1
                else:
                    valid_indices.append(option_index)
                option_checks.append(
                    {
                        "index": option_index,
                        "goal_xyyaw": list(option),
                        "endpoint_status": endpoint_status,
                        "path_status": path_state,
                        "path_sample_count": int(path_status.get("sample_count", 0) or 0),
                        "path_unknown": path_unknown,
                        "astar": astar_status,
                        "hard_status": hard_status or None,
                        "rejected": hard_rejected,
                    }
                )

            if traversal is not None:
                timing["collision_cells_checked"] = len(traversal.states)
                timing["collision_cache_hits"] = traversal.cache_hits
                timing["endpoint_footprints_checked"] = len(traversal.point_states)
                timing["endpoint_footprint_cache_hits"] = traversal.point_cache_hits
                timing["search_method"] = "shared_dijkstra"
                timing["search_expanded_count"] = sum(
                    search.expanded for search in traversal.searches.values()
                )
            metadata["occupancy_preflight"] = {
                "enabled": True,
                "endpoint_radius_m": self.candidate_occupancy_endpoint_radius_m,
                "sample_step_m": self.candidate_occupancy_sample_step_m,
                "start_ignore_m": self.candidate_occupancy_start_ignore_m,
                "occupied_threshold": self.candidate_occupancy_threshold,
                "options": option_checks,
            }
            metadata["occupancy_preflight_valid_option_indices"] = list(valid_indices)
            metadata["occupancy_preflight_rejected_option_indices"] = list(
                rejected_indices
            )
            if valid_indices:
                selected_index = valid_indices[0]
                selected_goal = list(options[selected_index])
                candidate.goal_xyyaw = selected_goal
                metadata["occupancy_preflight_selected_option_index"] = selected_index
                metadata["occupancy_preflight_selected_goal_xyyaw"] = list(
                    selected_goal
                )
                # For a simple portal/target candidate the command pose follows
                # the selected safe fallback.  Two-stage containers retain
                # their immutable staging/action index mappings; their executor
                # will use the preflight trace as an additional diagnostic and
                # run the normal make_plan check for each mapped pose.
                if (
                    candidate.interaction_command is not None
                    and str(metadata.get("node_type") or "").casefold()
                    != "container"
                ):
                    interaction_command = dict(candidate.interaction_command)
                    interaction_command["interaction_approach_pose_xyyaw"] = list(
                        selected_goal
                    )
                    candidate.interaction_command = interaction_command
                candidate.metadata = metadata
                kept.append(candidate)
                continue

            # No option is usable only when the map explicitly hard-failed all
            # options.  An unavailable/unknown-only map never reaches here,
            # preserving the candidate for the authoritative planner.
            if rejected_indices and self.candidate_occupancy_reject_occupied:
                metadata["occupancy_preflight_status"] = "rejected_all_hard"
                candidate.metadata = metadata
                self.filtered_occupancy_candidate_ids.append(
                    str(candidate.candidate_id)
                )
                timing["candidate_count_rejected"] = int(
                    timing["candidate_count_rejected"]
                ) + 1
                continue
            metadata["occupancy_preflight_status"] = "retained"
            candidate.metadata = metadata
            kept.append(candidate)
        return finish(kept)

    def _nearest_occupied_vector(
        self,
        goal: list[float],
        search_radius_m: float,
        *,
        occupancy_grid: object | None = None,
    ) -> tuple[float, float, float]:
        """Return distance and outward vector from nearest occupied cell.

        The vector is expressed in world XY and points from the occupied cell
        toward the goal.  For a wall this is the local obstacle normal, which
        is the correct direction for moving an interaction stance away from
        the obstacle.  Returning the vector (rather than only its length)
        avoids using the door centre as a proxy for the obstacle normal when
        the door frame, inflation layer, or a neighbouring object is what
        actually blocks the generated pose.
        """

        grid = (
            occupancy_grid
            if occupancy_grid is not None
            else self._occupancy_state_snapshot().get("occupancy_grid")
        )
        shape = self._grid_shape_and_data(grid)
        if shape is None or len(goal or []) < 2:
            return math.inf, 0.0, 0.0
        width, height, data = shape
        origin = grid.info.origin
        q = origin.orientation
        origin_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        resolution = max(float(grid.info.resolution), 1e-6)
        dx = float(goal[0]) - float(origin.position.x)
        dy = float(goal[1]) - float(origin.position.y)
        cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        center_x = int(math.floor(local_x / resolution))
        center_y = int(math.floor(local_y / resolution))
        radius_cells = int(math.ceil(max(0.0, search_radius_m) / resolution))
        nearest = math.inf
        nearest_dx = nearest_dy = 0.0
        for offset_y in range(-radius_cells, radius_cells + 1):
            for offset_x in range(-radius_cells, radius_cells + 1):
                grid_x, grid_y = center_x + offset_x, center_y + offset_y
                if not (0 <= grid_x < width and 0 <= grid_y < height):
                    continue
                try:
                    value = int(data[grid_y * width + grid_x])
                except (IndexError, TypeError, ValueError):
                    return math.inf, 0.0, 0.0
                if value < self.portal_goal_occupied_threshold:
                    continue
                cell_local_x = (grid_x + 0.5) * resolution
                cell_local_y = (grid_y + 0.5) * resolution
                world_x = float(origin.position.x) + cos_yaw * cell_local_x - sin_yaw * cell_local_y
                world_y = float(origin.position.y) + sin_yaw * cell_local_x + cos_yaw * cell_local_y
                delta_x = float(goal[0]) - world_x
                delta_y = float(goal[1]) - world_y
                distance = math.hypot(delta_x, delta_y)
                if distance < nearest:
                    nearest = distance
                    nearest_dx, nearest_dy = delta_x, delta_y
        return nearest, nearest_dx, nearest_dy

    def _occupied_clearance_m(
        self,
        goal: list[float],
        search_radius_m: float,
        *,
        occupancy_grid: object | None = None,
    ) -> float:
        """Distance from a goal to the nearest occupied costmap cell center."""

        return self._nearest_occupied_vector(
            goal, search_radius_m, occupancy_grid=occupancy_grid
        )[0]

    def _push_portal_goal_from_obstacle(
        self,
        goal: list[float],
        metadata: dict[str, Any],
        *,
        occupancy_grid: object | None = None,
    ) -> tuple[list[float], float]:
        """Push a portal goal away along its door/obstacle normal if needed."""

        required = max(0.0, float(self.generator.config.portal_obstacle_clearance_m))
        step = max(0.01, float(self.generator.config.portal_obstacle_push_step_m))
        max_push = max(0.0, float(self.generator.config.portal_obstacle_push_max_m))
        grid = (
            occupancy_grid
            if occupancy_grid is not None
            else self._occupancy_state_snapshot().get("occupancy_grid")
        )
        if grid is None or required <= 0.0 or max_push <= 0.0:
            return list(goal), 0.0
        search_radius = required + max_push + 0.5
        clearance, normal_x, normal_y = self._nearest_occupied_vector(
            goal, search_radius, occupancy_grid=grid
        )
        if clearance >= required:
            return list(goal), 0.0

        # Use the nearest occupied cell's outward normal.  The door OBB
        # normal is only a fallback when the costmap contains no usable
        # occupied cell vector (for example while the map is being updated).
        away_x, away_y = normal_x, normal_y
        norm = math.hypot(away_x, away_y)
        if norm <= 1e-6:
            center = list(
                metadata.get("portal_clearance_aabb_center_xy")
                or metadata.get("portal_aabb_center_xy")
                or []
            )
            if len(center) >= 2:
                away_x = float(goal[0]) - float(center[0])
                away_y = float(goal[1]) - float(center[1])
                norm = math.hypot(away_x, away_y)
            if norm <= 1e-6:
                yaw = float(goal[2]) if len(goal) >= 3 else 0.0
                away_x, away_y, norm = -math.cos(yaw), -math.sin(yaw), 1.0
        away_x /= norm
        away_y /= norm
        pushed = 0.0
        best = list(goal)
        while pushed < max_push - 1e-9:
            pushed = min(max_push, pushed + step)
            trial = list(goal)
            trial[0] += away_x * pushed
            trial[1] += away_y * pushed
            best = trial
            trial_clearance, _, _ = self._nearest_occupied_vector(
                trial, search_radius, occupancy_grid=grid
            )
            if trial_clearance >= required:
                return trial, pushed
        return best, pushed

    def _publish(self, _event) -> None:
        target_revision, target_context = self._target_snapshot()
        state = self._occupancy_state_snapshot()
        robot_xy = state.get("robot_xy")
        explorer_input = (
            self.explorer_proposal_stream
            if self.has_proposal_stream
            else self.explorer_status
        )
        ready = bool(explorer_input.get("ready", False))
        candidates = self.generator.generate(
            explorer_input, self.graph, robot_xy, target_context
        )
        candidates = self._filter_portal_goals_by_occupancy(candidates)
        candidates = self._filter_candidate_goals_by_occupancy(candidates)
        episode_id = str(
            self.graph.get("episode_id")
            or self.startup_scan_lifecycle.bound_episode_id
        )
        # No behavior may dispatch before map readiness.  Once ready, the
        # mandatory SCAN has a stable instance id even if the graph's episode
        # id has not arrived yet.  A failed scan deliberately exposes no
        # ordinary candidates, so failure can never masquerade as completion.
        if self.startup_scan_enabled:
            if not ready:
                candidates = []
            elif self.startup_scan_lifecycle.should_publish_scan(ready):
                candidates = [self._startup_scan_candidate()]
            elif self.startup_scan_lifecycle.blocks_regular_candidates():
                candidates = []
        navigation_frontiers = [
            candidate for candidate in candidates if candidate.behavior_type == "EXPLORE"
        ]
        interaction_frontiers = [
            candidate
            for candidate in candidates
            if candidate.behavior_type == "INTERACT"
            and not bool(
                (candidate.metadata or {}).get("interaction_group_already_explored")
            )
        ]
        # Compatibility field consumed by mission completion: once SCAN moves
        # to the semantic executor, it must reflect executor feedback rather
        # than ExplorePy's deliberately disabled legacy initial_spin.
        initial_scan_complete = (
            self.startup_scan_lifecycle.state == "COMPLETE"
            if self.startup_scan_enabled
            else bool(explorer_input.get("initial_scan_complete", True))
        )
        explorer_state = explorer_input.get("state") or {}
        active_navigation_frontier = bool(
            explorer_input.get("active_proposal_id")
            or explorer_state.get("active_goal")
        )
        frontier_filtering = summarize_frontier_filtering(
            explorer_input.get("frontier_debug")
        )
        retryable_filtered_frontier = bool(
            frontier_filtering["filtered_frontier_retryable"]
            and not active_navigation_frontier
            and not navigation_frontiers
        )
        unresolved_interaction_target_count = _unresolved_interaction_target_count(
            self.graph
        )
        connected_unknown_area_present = bool(
            int(frontier_filtering.get("raw_frontier_material_cluster_count", 0) or 0)
            > 0
        )
        navigation_frontier_exhausted = bool(
            ready
            and initial_scan_complete
            and not active_navigation_frontier
            and not navigation_frontiers
            and not retryable_filtered_frontier
            and not connected_unknown_area_present
        )
        interaction_frontier_exhausted = bool(
            not interaction_frontiers and unresolved_interaction_target_count == 0
        )
        combined_frontier_exhausted = bool(
            navigation_frontier_exhausted and interaction_frontier_exhausted
        )
        self.sequence += 1
        payload = {
            "schema_version": 1,
            "sequence": self.sequence,
            "timestamp": time.time(),
            "episode_id": episode_id,
            "graph_revision": self.graph.get("graph_revision", 0),
            "robot_xy": list(robot_xy) if robot_xy is not None else None,
            "target_context": target_context,
            "target_revision": target_revision,
            "exploration_context": {
                "ready": ready,
                "initial_scan_complete": initial_scan_complete,
                "startup_scan_enabled": self.startup_scan_enabled,
                "startup_scan_state": self.startup_scan_lifecycle.state,
                "startup_scan_failed": self.startup_scan_lifecycle.state == "FAILED",
                "startup_scan_failure_detail": dict(self.startup_scan_lifecycle.failure_detail),
                "frontier_exhausted": combined_frontier_exhausted,
                "navigation_frontier_exhausted": navigation_frontier_exhausted,
                "navigation_frontier_count": len(navigation_frontiers),
                "interaction_frontier_exhausted": interaction_frontier_exhausted,
                "interaction_frontier_count": len(interaction_frontiers),
                "unresolved_interaction_target_count": (
                    unresolved_interaction_target_count
                ),
                "connected_unknown_area_present": connected_unknown_area_present,
                "combined_frontier_count": len(navigation_frontiers)
                + len(interaction_frontiers),
                "occupancy_filtered_portal_candidate_ids": list(
                    self.filtered_portal_candidate_ids
                ),
                "occupancy_filtered_candidate_ids": list(
                    self.filtered_occupancy_candidate_ids
                ),
                "occupancy_preflight": dict(self.occupancy_preflight_timing),
                "source_frontier_exhausted": bool(
                    explorer_input.get("frontier_exhausted", False)
                ),
                "proposal_count": int(explorer_input.get("proposal_count", 0) or 0),
                **frontier_filtering,
                "map_resolution": float(
                    explorer_input.get("map_resolution", 0.0) or 0.0
                ),
                "observation_step": self.graph.get("capture_step"),
                "source": "explore_py_proposals"
                if self.has_proposal_stream
                else "explore_py_status_compatibility",
            },
            "graph_context": compact_graph(self.graph),
            "occupancy_preflight": dict(self.occupancy_preflight_timing),
            "candidate_count": len(candidates),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
        self._publish_target_snapshot(payload, target_revision)


if __name__ == "__main__":
    SemanticCandidateNode()
    rospy.spin()
