#!/usr/bin/env python3
from __future__ import annotations

import json
import time

from semantic_decision_py_pkg.behavior_candidates import (
    CandidateGenerator,
    CandidateGeneratorConfig,
)
from semantic_decision_py_pkg.model_policy import compact_graph
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311

patch_roslogging_findcaller_for_py311()

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class SemanticCandidateNode:
    def __init__(self) -> None:
        rospy.init_node("semantic_candidate_node")
        topics = rospy.get_param("~topics", {}) or {}
        config = rospy.get_param("~candidate", {}) or {}
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
                portal_standoff_m=float(config.get("portal_standoff_m", 1.0)),
                container_standoff_m=float(config.get("container_standoff_m", 1.0)),
                drawer_standoff_m=float(config.get("drawer_standoff_m", 0.60)),
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
                open_fraction_threshold=float(
                    config.get("open_fraction_threshold", 0.67)
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
            self.graph = json.loads(message.data)
        except json.JSONDecodeError:
            return

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
        candidates = self.generator.generate(
            explorer_input, self.graph, self.robot_xy, self.target_context
        )
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
        ready = bool(explorer_input.get("ready", False))
        initial_scan_complete = bool(
            explorer_input.get("initial_scan_complete", True)
        )
        explorer_state = explorer_input.get("state") or {}
        active_navigation_frontier = bool(
            explorer_input.get("active_proposal_id")
            or explorer_state.get("active_goal")
        )
        navigation_frontier_exhausted = bool(
            ready
            and initial_scan_complete
            and not active_navigation_frontier
            and not navigation_frontiers
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
            "episode_id": self.graph.get("episode_id", ""),
            "graph_revision": self.graph.get("graph_revision", 0),
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
            "target_context": dict(self.target_context),
            "exploration_context": {
                "ready": ready,
                "initial_scan_complete": initial_scan_complete,
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
