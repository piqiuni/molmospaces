#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time

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
        self.robot_xy: tuple[float, float] | None = None
        self.occupancy_grid: OccupancyGrid | None = None
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

    def _target_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.target_context = payload

    def _odom_callback(self, message: Odometry) -> None:
        self.robot_xy = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
        )

    def _occupancy_callback(self, message: OccupancyGrid) -> None:
        self.occupancy_grid = message

    def _goal_is_known_free(self, goal: list[float]) -> bool:
        grid = self.occupancy_grid
        if grid is None or len(goal or []) < 2:
            return False
        origin = grid.info.origin
        q = origin.orientation
        origin_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        dx = float(goal[0]) - float(origin.position.x)
        dy = float(goal[1]) - float(origin.position.y)
        cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        resolution = max(float(grid.info.resolution), 1e-6)
        center_x = int(math.floor(local_x / resolution))
        center_y = int(math.floor(local_y / resolution))
        radius_cells = int(math.ceil(self.portal_goal_known_free_radius_m / resolution))
        width, height = int(grid.info.width), int(grid.info.height)
        for offset_y in range(-radius_cells, radius_cells + 1):
            for offset_x in range(-radius_cells, radius_cells + 1):
                if math.hypot(offset_x, offset_y) * resolution > self.portal_goal_known_free_radius_m + 1e-9:
                    continue
                grid_x, grid_y = center_x + offset_x, center_y + offset_y
                if not (0 <= grid_x < width and 0 <= grid_y < height):
                    return False
                value = int(grid.data[grid_y * width + grid_x])
                if value < 0 or value >= self.portal_goal_occupied_threshold:
                    return False
        return True

    def _filter_portal_goals_by_occupancy(
        self, candidates: list[BehaviorCandidate]
    ) -> list[BehaviorCandidate]:
        self.filtered_portal_candidate_ids = []
        if not self.portal_goal_require_known_free and self.occupancy_grid is None:
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
                    goal, metadata
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

    def _nearest_occupied_vector(
        self, goal: list[float], search_radius_m: float
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

        grid = self.occupancy_grid
        if grid is None or len(goal or []) < 2:
            return math.inf, 0.0, 0.0
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
        width, height = int(grid.info.width), int(grid.info.height)
        nearest = math.inf
        nearest_dx = nearest_dy = 0.0
        for offset_y in range(-radius_cells, radius_cells + 1):
            for offset_x in range(-radius_cells, radius_cells + 1):
                grid_x, grid_y = center_x + offset_x, center_y + offset_y
                if not (0 <= grid_x < width and 0 <= grid_y < height):
                    continue
                value = int(grid.data[grid_y * width + grid_x])
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

    def _occupied_clearance_m(self, goal: list[float], search_radius_m: float) -> float:
        """Distance from a goal to the nearest occupied costmap cell center."""

        return self._nearest_occupied_vector(goal, search_radius_m)[0]

    def _push_portal_goal_from_obstacle(
        self, goal: list[float], metadata: dict[str, Any]
    ) -> tuple[list[float], float]:
        """Push a portal goal away along its door/obstacle normal if needed."""

        required = max(0.0, float(self.generator.config.portal_obstacle_clearance_m))
        step = max(0.01, float(self.generator.config.portal_obstacle_push_step_m))
        max_push = max(0.0, float(self.generator.config.portal_obstacle_push_max_m))
        if self.occupancy_grid is None or required <= 0.0 or max_push <= 0.0:
            return list(goal), 0.0
        search_radius = required + max_push + 0.5
        clearance, normal_x, normal_y = self._nearest_occupied_vector(
            goal, search_radius
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
                trial, search_radius
            )
            if trial_clearance >= required:
                return trial, pushed
        return best, pushed

    def _publish(self, _event) -> None:
        explorer_input = (
            self.explorer_proposal_stream
            if self.has_proposal_stream
            else self.explorer_status
        )
        ready = bool(explorer_input.get("ready", False))
        candidates = self.generator.generate(
            explorer_input, self.graph, self.robot_xy, self.target_context
        )
        candidates = self._filter_portal_goals_by_occupancy(candidates)
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
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
            "target_context": dict(self.target_context),
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
            "candidate_count": len(candidates),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
        self.publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )


if __name__ == "__main__":
    SemanticCandidateNode()
    rospy.spin()
