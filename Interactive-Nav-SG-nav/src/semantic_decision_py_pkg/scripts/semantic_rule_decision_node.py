#!/usr/bin/env python3
from __future__ import annotations

import json
import time

from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311
from semantic_decision_py_pkg.rule_policy import RulePolicy, RulePolicyConfig

patch_roslogging_findcaller_for_py311()

import rospy
from std_msgs.msg import String


class SemanticRuleDecisionNode:
    def __init__(self) -> None:
        rospy.init_node("semantic_rule_decision_node")
        topics = rospy.get_param("~topics", {}) or {}
        config = rospy.get_param("~policy", {}) or {}
        self.policy = RulePolicy(
            RulePolicyConfig(
                exploration_gain_weight=float(config.get("exploration_gain_weight", 1.0)),
                visibility_gain_weight=float(config.get("visibility_gain_weight", 0.8)),
                semantic_gain_weight=float(config.get("semantic_gain_weight", 0.6)),
                confidence_weight=float(config.get("confidence_weight", 0.25)),
                priority_weight=float(config.get("priority_weight", 0.45)),
                distance_cost_weight=float(config.get("distance_cost_weight", 0.45)),
                distance_normalization_m=float(config.get("distance_normalization_m", 6.0)),
                interaction_cost_weight=float(config.get("interaction_cost_weight", 0.30)),
                staleness_cost_weight=float(config.get("staleness_cost_weight", 0.35)),
                portal_bonus=float(config.get("portal_bonus", 0.55)),
                container_bonus=float(config.get("container_bonus", 0.15)),
                continuity_bonus=float(config.get("continuity_bonus", 0.25)),
                minimum_score=float(config.get("minimum_score", -1e9)),
            )
        )
        self.failure_cooldown_s = float(config.get("failure_cooldown_s", 45.0))
        self.success_cooldown_s = float(config.get("success_cooldown_s", 5.0))
        self.latest_candidates_payload: dict = {}
        self.active_candidate_id = ""
        self.active_decision_id = ""
        self.active_behavior_type = ""
        self.minimum_candidate_sequence = 0
        self.cooldown_until: dict[str, float] = {}
        self.decision_index = 0
        self.selected_pub = rospy.Publisher(
            topics.get("selected_behavior", "/semantic_decision/selected_behavior"),
            String,
            queue_size=1,
            latch=True,
        )
        self.trace_pub = rospy.Publisher(
            "/semantic_decision/decision_trace", String, queue_size=1, latch=True
        )
        rospy.Subscriber(
            topics.get("candidates", "/semantic_decision/candidates"),
            String,
            self._candidate_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            topics.get("behavior_feedback", "/semantic_decision/behavior_feedback"),
            String,
            self._feedback_callback,
            queue_size=10,
        )
        self.timer = rospy.Timer(rospy.Duration(0.5), self._tick)

    def _candidate_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        episode_id = str(payload.get("episode_id") or "")
        previous_episode = str(self.latest_candidates_payload.get("episode_id") or "")
        if episode_id and previous_episode and episode_id != previous_episode:
            self.active_candidate_id = ""
            self.active_decision_id = ""
            self.active_behavior_type = ""
            self.minimum_candidate_sequence = 0
            self.cooldown_until.clear()
        self.latest_candidates_payload = payload

    def _feedback_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        candidate_id = str(payload.get("candidate_id") or "")
        decision_id = str(payload.get("decision_id") or "")
        if decision_id and self.active_decision_id and decision_id != self.active_decision_id:
            return
        status = str(payload.get("status") or "")
        if status not in {"SUCCEEDED", "FAILED", "CANCELED", "REJECTED"}:
            return
        if candidate_id:
            cooldown_s = (
                self.success_cooldown_s if status == "SUCCEEDED" else self.failure_cooldown_s
            )
            self.cooldown_until[candidate_id] = time.monotonic() + cooldown_s
        if status == "SUCCEEDED" and self.active_behavior_type == "INTERACT":
            self.minimum_candidate_sequence = max(
                self.minimum_candidate_sequence,
                int(self.latest_candidates_payload.get("sequence", 0) or 0) + 1,
            )
        self.active_candidate_id = ""
        self.active_decision_id = ""
        self.active_behavior_type = ""

    def _tick(self, _event) -> None:
        if self.active_candidate_id:
            return
        candidate_sequence = int(self.latest_candidates_payload.get("sequence", 0) or 0)
        if candidate_sequence < self.minimum_candidate_sequence:
            return
        self.minimum_candidate_sequence = 0
        now = time.monotonic()
        candidates = []
        for payload in self.latest_candidates_payload.get("candidates") or []:
            candidate_id = str(payload.get("candidate_id") or "")
            if now < self.cooldown_until.get(candidate_id, 0.0):
                continue
            candidates.append(BehaviorCandidate(**payload))
        scored = [self.policy.score(candidate) for candidate in candidates]
        eligible = [
            candidate
            for candidate in scored
            if candidate.score >= self.policy.config.minimum_score
        ]
        selected = min(
            eligible,
            key=lambda candidate: (-candidate.score, candidate.candidate_id),
            default=None,
        )
        trace = {
            "timestamp": time.time(),
            "episode_id": self.latest_candidates_payload.get("episode_id", ""),
            "graph_revision": self.latest_candidates_payload.get("graph_revision", 0),
            "active_candidate_id": self.active_candidate_id,
            "ranked_candidates": [
                candidate.to_dict()
                for candidate in sorted(scored, key=lambda item: (-item.score, item.candidate_id))
            ],
        }
        self.trace_pub.publish(
            String(data=json.dumps(trace, ensure_ascii=False, separators=(",", ":")))
        )
        if selected is None:
            return
        self.decision_index += 1
        decision_id = f"decision_{self.decision_index:06d}"
        selection = selected.to_dict()
        selection.update(
            {
                "decision_id": decision_id,
                "selected_at": time.time(),
                "episode_id": self.latest_candidates_payload.get("episode_id", ""),
                "graph_revision": self.latest_candidates_payload.get("graph_revision", 0),
                "policy_mode": "rule_cost_v1",
            }
        )
        self.active_candidate_id = selected.candidate_id
        self.active_decision_id = decision_id
        self.active_behavior_type = selected.behavior_type
        self.selected_pub.publish(
            String(data=json.dumps(selection, ensure_ascii=False, separators=(",", ":")))
        )


if __name__ == "__main__":
    SemanticRuleDecisionNode()
    rospy.spin()
