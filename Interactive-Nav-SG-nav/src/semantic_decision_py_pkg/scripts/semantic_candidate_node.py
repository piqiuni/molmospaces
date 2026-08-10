#!/usr/bin/env python3
from __future__ import annotations

import json
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
from nav_msgs.msg import Odometry
from std_msgs.msg import String


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
                portal_standoff_m=float(config.get("portal_standoff_m", 1.0)),
                portal_traversal_distance_m=float(
                    config.get("portal_traversal_distance_m", 0.9)
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
                container_observation_standoff_m=float(
                    config.get("container_observation_standoff_m", 0.7)
                ),
                drawer_observation_standoff_m=float(
                    config.get("drawer_observation_standoff_m", 0.65)
                ),
                fridge_observation_standoff_m=float(
                    config.get("fridge_observation_standoff_m", 0.8)
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
                container_safe_staging_arrival_tolerance_m=float(
                    config.get("container_safe_staging_arrival_tolerance_m", 0.30)
                ),
                container_interaction_ready_distance_m=float(
                    config.get("container_interaction_ready_distance_m", 0.18)
                ),
                interaction_safety_margin_m=float(
                    config.get("interaction_safety_margin_m", 0.0)
                ),
                interaction_ready_distance_m=float(
                    config.get("interaction_ready_distance_m", 0.45)
                ),
                require_current_visibility=bool(
                    config.get("require_current_visibility", False)
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
        navigation_frontier_exhausted = bool(
            ready
            and initial_scan_complete
            and not active_navigation_frontier
            and not navigation_frontiers
            and not retryable_filtered_frontier
        )
        interaction_frontier_exhausted = not interaction_frontiers
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
                "combined_frontier_count": len(navigation_frontiers)
                + len(interaction_frontiers),
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
