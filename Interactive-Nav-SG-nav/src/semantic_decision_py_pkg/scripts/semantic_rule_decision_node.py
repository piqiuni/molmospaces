#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time

from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.env_config import apply_model_env_overrides, load_env_file
from semantic_decision_py_pkg.mission_completion import (
    MissionCompletionConfig,
    MissionCompletionTracker,
    TargetMissionTracker,
)
from semantic_decision_py_pkg.model_policy import (
    ModelCircuitBreaker,
    ModelPolicyClient,
    ModelPolicyConfig,
)
from semantic_decision_py_pkg.ros_compat import patch_roslogging_findcaller_for_py311
from semantic_decision_py_pkg.rule_policy import (
    RulePolicy,
    RulePolicyConfig,
    progressive_failure_cooldown,
)
from semantic_mllm_py_pkg.ablation import AblationConfig

patch_roslogging_findcaller_for_py311()

import rospy
from std_msgs.msg import String


class SemanticRuleDecisionNode:
    def __init__(self) -> None:
        load_env_file(os.environ.get("SEMANTIC_DECISION_ENV_FILE"))
        rospy.init_node("semantic_rule_decision_node")
        topics = rospy.get_param("~topics", {}) or {}
        config = rospy.get_param("~policy", {}) or {}
        mission_config = rospy.get_param("~mission", {}) or {}
        ablation_config = rospy.get_param("~ablation", {}) or {}
        self.ablation = AblationConfig(
            module1=str(ablation_config.get("module1", "dynamic_rule")),
            module2=str(ablation_config.get("module2", "rule_cost")),
            module3=str(ablation_config.get("module3", "rule_verified")),
        )
        model_config = apply_model_env_overrides(rospy.get_param("~model", {}) or {})
        completion_config = rospy.get_param("~completion", {}) or {}
        self.policy = RulePolicy(
            RulePolicyConfig(
                exploration_gain_weight=float(config.get("exploration_gain_weight", 1.0)),
                visibility_gain_weight=float(config.get("visibility_gain_weight", 0.8)),
                semantic_gain_weight=float(config.get("semantic_gain_weight", 0.6)),
                target_relevance_weight=float(config.get("target_relevance_weight", 3.0)),
                confidence_weight=float(config.get("confidence_weight", 0.25)),
                priority_weight=float(config.get("priority_weight", 0.45)),
                distance_cost_weight=float(config.get("distance_cost_weight", 0.45)),
                distance_normalization_m=float(config.get("distance_normalization_m", 6.0)),
                interaction_cost_weight=float(config.get("interaction_cost_weight", 0.30)),
                staleness_cost_weight=float(config.get("staleness_cost_weight", 0.35)),
                portal_bonus=float(config.get("portal_bonus", 0.55)),
                container_bonus=float(config.get("container_bonus", 0.15)),
                continuity_bonus=float(config.get("continuity_bonus", 0.25)),
                nearby_interaction_radius_m=float(
                    config.get("nearby_interaction_radius_m", 1.5)
                ),
                nearby_interaction_bonus=float(
                    config.get("nearby_interaction_bonus", 1.5)
                ),
                interaction_priority_bonus=float(
                    config.get("interaction_priority_bonus", 0.0)
                ),
                minimum_score=float(config.get("minimum_score", -1e9)),
            )
        )
        self.failure_cooldown_s = float(config.get("failure_cooldown_s", 30.0))
        configured_failure_schedule = config.get("failure_cooldown_schedule_s")
        self.failure_cooldown_schedule_s = tuple(
            float(value)
            for value in (
                configured_failure_schedule
                if isinstance(configured_failure_schedule, (list, tuple))
                else [self.failure_cooldown_s]
            )
        )
        self.interaction_target_failure_cooldown_s = float(
            config.get("interaction_target_failure_cooldown_s", self.failure_cooldown_s)
        )
        self.success_cooldown_s = float(config.get("success_cooldown_s", 5.0))
        self.failure_retry_delay_s = float(config.get("failure_retry_delay_s", 2.0))
        self.mission_mode = self._normalize_mission_mode(
            mission_config.get("mode", "semantic_interaction_exploration")
        )
        configured_backend = str(config.get("backend", "rule")).casefold()
        self.policy_backend = (
            "model"
            if self.ablation.module2 == "mllm_score"
            else "rule"
            if self.ablation.module2 == "rule_cost"
            else configured_backend
        )
        self.model_policy = ModelPolicyClient(
            ModelPolicyConfig(
                mode=str(model_config.get("mode", "disabled")),
                command=str(model_config.get("command", "")),
                endpoint=str(model_config.get("endpoint", "")),
                api_key_env=str(model_config.get("api_key_env", "OPENAI_API_KEY")),
                model=str(model_config.get("model", "qwen3.6-35b-a3b")),
                protocol=str(model_config.get("protocol", "openai_chat")),
                timeout_s=float(model_config.get("timeout_s", 3.0)),
                temperature=float(model_config.get("temperature", 0.0)),
                max_tokens=int(model_config.get("max_tokens", 128)),
                reasoning_effort=str(model_config.get("reasoning_effort", "off")),
                image_detail=str(model_config.get("image_detail", "low")),
                max_graph_nodes=int(model_config.get("max_graph_nodes", 80)),
                max_graph_edges=int(model_config.get("max_graph_edges", 160)),
                metrics_path=str(model_config.get("metrics_path", "")),
            )
        )
        self.model_circuit_breaker = ModelCircuitBreaker(
            consecutive_timeout_limit=int(
                model_config.get("consecutive_timeout_limit", 2)
            ),
            cooldown_s=float(model_config.get("timeout_cooldown_s", 60.0)),
        )
        self.completion_tracker = MissionCompletionTracker(
            MissionCompletionConfig(
                empty_candidate_confirmations=int(
                    completion_config.get("empty_candidate_confirmations", 3)
                ),
                empty_candidate_min_steps=int(
                    completion_config.get("empty_candidate_min_steps", 50)
                ),
                stagnation_failure_limit=int(
                    completion_config.get("stagnation_failure_limit", 0)
                ),
            )
        )
        self.target_mission = TargetMissionTracker()
        self.latest_candidates_payload: dict = {}
        self.active_candidate_id = ""
        self.active_decision_id = ""
        self.active_behavior_type = ""
        self.minimum_candidate_sequence = 0
        self.next_decision_time = 0.0
        self.goal_complete = False
        self.target_goal_complete = False
        self.target_context: dict = {}
        self.active_target_goal = False
        self.cooldown_until: dict[str, float] = {}
        self.failure_counts: dict[str, int] = {}
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
        self.goal_status_pub = rospy.Publisher(
            topics.get("goal_status", "/semantic_decision/goal_status"),
            String,
            queue_size=2,
            latch=True,
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
            self.next_decision_time = 0.0
            self.goal_complete = False
            self.target_goal_complete = False
            self.active_target_goal = False
            self.target_mission.reset()
            self.cooldown_until.clear()
            self.failure_counts.clear()
            self.model_circuit_breaker = ModelCircuitBreaker(
                consecutive_timeout_limit=self.model_circuit_breaker.consecutive_timeout_limit,
                cooldown_s=self.model_circuit_breaker.cooldown_s,
            )
            self.completion_tracker.reset()
        target_context = payload.get("target_context") or {}
        target_key = json.dumps(target_context, ensure_ascii=False, sort_keys=True)
        previous_target_key = json.dumps(
            self.target_context, ensure_ascii=False, sort_keys=True
        )
        if target_key != previous_target_key:
            self.target_context = dict(target_context)
            self.goal_complete = False
            self.target_goal_complete = False
            self.target_mission.reset()
            self._publish_goal_status(
                "ACTIVE" if target_context.get("enabled") else "DISABLED"
            )
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
        self.completion_tracker.note_feedback(payload)
        if candidate_id:
            if status == "SUCCEEDED":
                self.failure_counts.pop(candidate_id, None)
                cooldown_s = self.success_cooldown_s
            else:
                failure_count = self.failure_counts.get(candidate_id, 0) + 1
                self.failure_counts[candidate_id] = failure_count
                cooldown_s = progressive_failure_cooldown(
                    self.failure_cooldown_schedule_s,
                    failure_count,
                )
            self.cooldown_until[candidate_id] = time.monotonic() + cooldown_s
        if status != "SUCCEEDED" and self.active_behavior_type == "INTERACT":
            target_id = self._interaction_target_id(candidate_id)
            if target_id:
                self.cooldown_until[target_id] = time.monotonic() + max(
                    0.0, self.interaction_target_failure_cooldown_s
                )
        if status != "SUCCEEDED":
            self.next_decision_time = time.monotonic() + self.failure_retry_delay_s
        if status == "SUCCEEDED" and self.active_behavior_type == "INTERACT":
            self.minimum_candidate_sequence = max(
                self.minimum_candidate_sequence,
                int(self.latest_candidates_payload.get("sequence", 0) or 0) + 1,
            )
        active_behavior_type = self.active_behavior_type
        target_interaction_succeeded = (
            status == "SUCCEEDED"
            and active_behavior_type == "INTERACT"
            and self.target_mission.matches_target_interaction(
                target_context=self.target_context,
                feedback=payload,
                candidates=self.latest_candidates_payload.get("candidates") or [],
            )
        )
        if status == "SUCCEEDED" and (
            self.active_target_goal or target_interaction_succeeded
        ):
            transition = self.target_mission.on_behavior_succeeded(
                behavior_type=active_behavior_type,
                active_target_goal=True,
                target_context=self.target_context,
                feedback=payload,
                next_candidate_sequence=int(
                    self.latest_candidates_payload.get("sequence", 0) or 0
                )
                + 1,
            )
            if transition["phase"] == "target_reached":
                self.minimum_candidate_sequence = max(
                    self.minimum_candidate_sequence,
                    int(transition["minimum_candidate_sequence"]),
                )
                self._publish_goal_status(
                    "TARGET_REACHED",
                    detail=transition["detail"],
                )
            elif transition["phase"] == "container_opened":
                self._publish_goal_status(
                    "TARGET_CONTAINER_INTERACTED",
                    detail=transition["detail"],
                )
            elif transition["phase"] == "complete":
                self.target_goal_complete = True
                detail = dict(transition["detail"])
                detail["reason"] = "target_goal_succeeded"
                if target_interaction_succeeded and not self.active_target_goal:
                    detail["target_interaction_source"] = "autonomous_interaction"
                self._publish_goal_status("SUCCEEDED", detail=detail)
                if self.mission_mode == "semantic_interaction_object_goal":
                    self.goal_complete = True
        self.active_candidate_id = ""
        self.active_decision_id = ""
        self.active_behavior_type = ""
        self.active_target_goal = False

    def _tick(self, _event) -> None:
        if self.active_candidate_id:
            return
        if time.monotonic() < self.next_decision_time:
            return
        if self.goal_complete:
            return
        candidate_sequence = int(self.latest_candidates_payload.get("sequence", 0) or 0)
        if candidate_sequence < self.minimum_candidate_sequence:
            return
        self.minimum_candidate_sequence = 0
        if self.mission_mode == "semantic_interaction_object_goal" and self.target_goal_complete:
            self.goal_complete = True
            return
        if self.completion_tracker.update(
            self.latest_candidates_payload,
            has_active_behavior=bool(self.active_candidate_id),
            target_enabled=bool(self.target_context.get("enabled")),
        ):
            exploration_context = (
                self.latest_candidates_payload.get("exploration_context") or {}
            )
            self.goal_complete = True
            self._publish_goal_status(
                "EXPLORATION_EXHAUSTED",
                detail={
                    "reason": self.completion_tracker.reason
                    or "navigation_and_interaction_frontiers_exhausted",
                    "completion_confirmations": self.completion_tracker.confirmations,
                    "candidate_sequence": candidate_sequence,
                    "target_goal_succeeded": self.target_goal_complete,
                    "navigation_frontier_count": int(
                        exploration_context.get("navigation_frontier_count", 0) or 0
                    ),
                    "interaction_frontier_count": int(
                        exploration_context.get("interaction_frontier_count", 0)
                        or 0
                    ),
                },
            )
            return
        now = time.monotonic()
        candidates = []
        for payload in self.latest_candidates_payload.get("candidates") or []:
            candidate_id = str(payload.get("candidate_id") or "")
            target_cooldown_key = self._interaction_target_id(candidate_id)
            if (
                now < self.cooldown_until.get(candidate_id, 0.0)
                or (target_cooldown_key and now < self.cooldown_until.get(target_cooldown_key, 0.0))
            ):
                continue
            metadata = payload.get("metadata") or {}
            if self.target_goal_complete and (
                bool(metadata.get("target_goal"))
                or bool(metadata.get("target_match"))
            ):
                continue
            candidates.append(BehaviorCandidate(**payload))
        scored = [self.policy.score(candidate) for candidate in candidates]
        eligible = [
            candidate
            for candidate in scored
            if candidate.score >= self.policy.config.minimum_score
        ]
        eligible = self.target_mission.filter_candidates(eligible)
        selected = min(
            eligible,
            key=lambda candidate: (-candidate.score, candidate.candidate_id),
            default=None,
        )
        model_circuit_open = not self.model_circuit_breaker.allow_request(now)
        if self.policy_backend == "model" and not model_circuit_open:
            model_selected = self.model_policy.select(
                eligible,
                target_context=self.latest_candidates_payload.get("target_context") or {},
                graph=self.latest_candidates_payload.get("graph_context") or {},
                robot_context={
                    "robot_xy": self.latest_candidates_payload.get("robot_xy"),
                    "exploration_context": self.latest_candidates_payload.get(
                        "exploration_context"
                    ),
                    "active_candidate_id": self.active_candidate_id,
                },
                metrics_context={
                    "episode_id": self.latest_candidates_payload.get("episode_id", ""),
                    "graph_revision": self.latest_candidates_payload.get("graph_revision", 0),
                    "candidate_sequence": candidate_sequence,
                    "candidate_ids": [candidate.candidate_id for candidate in eligible],
                },
            )
            if model_selected is not None:
                selected = model_selected
                self.model_circuit_breaker.record_success()
            elif self.model_policy.last_error:
                self.model_circuit_breaker.record_failure(self.model_policy.last_error, now)
        elif self.policy_backend == "model":
            self.model_policy.last_error = "model_circuit_open_after_consecutive_timeouts"
            self.model_policy.last_result_source = "rule_fallback_circuit_open"
            self.model_policy.last_metrics = {}
        trace = {
            "timestamp": time.time(),
            "episode_id": self.latest_candidates_payload.get("episode_id", ""),
            "graph_revision": self.latest_candidates_payload.get("graph_revision", 0),
            "active_candidate_id": self.active_candidate_id,
            "policy_backend": self.policy_backend,
            "ablation": self.ablation.to_dict(),
            "model_error": self.model_policy.last_error,
            "model_result_source": self.model_policy.last_result_source,
            "model_metrics": dict(self.model_policy.last_metrics),
            "model_circuit_open": model_circuit_open,
            "model_circuit_open_until": self.model_circuit_breaker.open_until,
            "model_consecutive_timeouts": self.model_circuit_breaker.consecutive_timeouts,
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
                "policy_mode": (
                    f"model_{self.model_policy.config.mode}"
                    if self.policy_backend == "model"
                    else "rule_cost_v1"
                ),
            }
        )
        self.active_candidate_id = selected.candidate_id
        self.active_decision_id = decision_id
        self.active_behavior_type = selected.behavior_type
        self.active_target_goal = bool((selected.metadata or {}).get("target_goal")) or (
            self.target_mission.pending_interaction is not None
            and selected.behavior_type == "INTERACT"
        )
        self.selected_pub.publish(
            String(data=json.dumps(selection, ensure_ascii=False, separators=(",", ":")))
        )

    @staticmethod
    def _interaction_target_id(candidate_id: str) -> str:
        parts = str(candidate_id or "").split(":", 2)
        if len(parts) >= 2 and parts[0] == "interaction" and parts[1]:
            return f"interaction_target:{parts[1]}"
        return ""

    def _publish_goal_status(self, status: str, detail: dict | None = None) -> None:
        payload = {
            "status": status,
            "mission_mode": self.mission_mode,
            "target_context": dict(self.target_context),
            "decision_id": self.active_decision_id,
            "timestamp": time.time(),
            "detail": detail or {},
        }
        self.goal_status_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    @staticmethod
    def _normalize_mission_mode(value: object) -> str:
        normalized = str(value or "").strip().casefold().replace("-", "_")
        if normalized in {"object_goal", "semantic_interaction_object_goal"}:
            return "semantic_interaction_object_goal"
        return "semantic_interaction_exploration"


if __name__ == "__main__":
    SemanticRuleDecisionNode()
    rospy.spin()
