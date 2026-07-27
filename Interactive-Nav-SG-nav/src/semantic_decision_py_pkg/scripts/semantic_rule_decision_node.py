#!/usr/bin/env python3
from __future__ import annotations

import copy
from collections import deque
import json
import os
import threading
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
    candidate_group_id,
    compact_candidate_groups,
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
        self.repeat_guard_enabled = bool(config.get("repeat_guard_enabled", True))
        self.repeat_guard_low_gain_limit = max(
            1, int(config.get("repeat_guard_low_gain_limit", 2))
        )
        self.repeat_guard_min_frontier_shrink_m = max(
            0.0, float(config.get("repeat_guard_min_frontier_shrink_m", 0.15))
        )
        self.mission_mode = self._normalize_mission_mode(
            mission_config.get("mode", "semantic_interaction_exploration")
        )
        self.post_container_target_wait_s = max(
            0.0, float(mission_config.get("post_container_target_wait_s", 5.0))
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
                max_tokens=int(model_config.get("max_tokens", 96)),
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
        self.state_lock = threading.RLock()
        self.latest_candidates_payload: dict = {}
        self.decision_in_flight = False
        self.decision_history = deque(maxlen=32)
        self.group_history: dict[str, dict] = {}
        self.last_selected_group_id = ""
        self.active_candidate_id = ""
        self.active_decision_id = ""
        self.active_behavior_type = ""
        self.minimum_candidate_sequence = 0
        self.next_decision_time = 0.0
        self.goal_complete = False
        self.target_goal_complete = False
        self.target_context: dict = {}
        self.active_target_goal = False
        self.target_observation_wait_until = 0.0
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
        with self.state_lock:
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
                self.target_observation_wait_until = 0.0
                self.target_mission.reset()
                self.cooldown_until.clear()
                self.failure_counts.clear()
                self.decision_history.clear()
                self.group_history.clear()
                self.last_selected_group_id = ""
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
                self.target_observation_wait_until = 0.0
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
        with self.state_lock:
            self._handle_feedback(payload)

    def _handle_feedback(self, payload: dict) -> None:
        candidate_id = str(payload.get("candidate_id") or "")
        decision_id = str(payload.get("decision_id") or "")
        if decision_id and self.active_decision_id and decision_id != self.active_decision_id:
            return
        status = str(payload.get("status") or "")
        if status not in {"SUCCEEDED", "FAILED", "CANCELED", "REJECTED"}:
            return
        self._record_decision_result(payload)
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
        if status == "SUCCEEDED":
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
                self.target_observation_wait_until = (
                    time.monotonic() + self.post_container_target_wait_s
                )
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
        with self.state_lock:
            if self.active_candidate_id or self.decision_in_flight:
                return
            if time.monotonic() < self.next_decision_time:
                return
            if self.goal_complete:
                return
            candidate_snapshot = copy.deepcopy(self.latest_candidates_payload)
            self.decision_in_flight = True
        try:
            self._decide_from_snapshot(candidate_snapshot)
        finally:
            with self.state_lock:
                self.decision_in_flight = False

    def _decide_from_snapshot(self, candidate_snapshot: dict) -> None:
        candidate_sequence = int(candidate_snapshot.get("sequence", 0) or 0)
        if candidate_sequence < self.minimum_candidate_sequence:
            return
        self.minimum_candidate_sequence = 0
        if self.mission_mode == "semantic_interaction_object_goal" and self.target_goal_complete:
            self.goal_complete = True
            return
        if self.completion_tracker.update(
            candidate_snapshot,
            has_active_behavior=bool(self.active_candidate_id),
            target_enabled=bool(self.target_context.get("enabled")),
        ):
            exploration_context = (
                candidate_snapshot.get("exploration_context") or {}
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
        for payload in candidate_snapshot.get("candidates") or []:
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
        target_eligible = [
            candidate
            for candidate in eligible
            if bool((candidate.metadata or {}).get("target_goal"))
            or bool((candidate.metadata or {}).get("target_match"))
        ]
        waiting_for_target_observation = bool(
            self.mission_mode == "semantic_interaction_object_goal"
            and self.target_observation_wait_until > now
            and not target_eligible
        )
        if self.mission_mode == "semantic_interaction_object_goal" and target_eligible:
            eligible = target_eligible
            self.target_observation_wait_until = 0.0
        selected = min(
            eligible,
            key=lambda candidate: (-candidate.score, candidate.candidate_id),
            default=None,
        )
        if waiting_for_target_observation:
            selected = None
        model_circuit_open = not self.model_circuit_breaker.allow_request(now)
        decision_history, group_history = self._history_context(candidate_snapshot)
        projected_groups, _ = compact_candidate_groups(
            eligible,
            candidate_snapshot.get("graph_context") or {},
            decision_history=group_history,
        )
        self.model_policy.last_candidate_groups = projected_groups
        self.model_policy.last_ranking_ids = []
        self.model_policy.last_selected_group_id = ""
        self.model_policy.last_reason = ""
        self.model_policy.last_confidence = ""
        model_selected_group_id = ""
        selection_override_reason = (
            "waiting_for_stable_target_after_container"
            if waiting_for_target_observation
            else ""
        )
        if (
            self.policy_backend == "model"
            and not model_circuit_open
            and not waiting_for_target_observation
        ):
            model_selected = self.model_policy.select(
                eligible,
                target_context=candidate_snapshot.get("target_context") or {},
                graph=candidate_snapshot.get("graph_context") or {},
                robot_context={
                    "robot_xy": candidate_snapshot.get("robot_xy"),
                    "exploration_context": candidate_snapshot.get(
                        "exploration_context"
                    ),
                    "active_candidate_id": self.active_candidate_id,
                    "decision_history": decision_history,
                    "group_history": group_history,
                },
                metrics_context={
                    "episode_id": candidate_snapshot.get("episode_id", ""),
                    "graph_revision": candidate_snapshot.get("graph_revision", 0),
                    "candidate_sequence": candidate_sequence,
                    "candidate_ids": [candidate.candidate_id for candidate in eligible],
                },
            )
            model_selected_group_id = self.model_policy.last_selected_group_id
            if model_selected is not None:
                selected = model_selected
                self.model_circuit_breaker.record_success()
            elif self.model_policy.last_error:
                self.model_circuit_breaker.record_failure(self.model_policy.last_error, now)
        elif self.policy_backend == "model":
            self.model_policy.last_error = "model_circuit_open_after_consecutive_timeouts"
            self.model_policy.last_result_source = "rule_fallback_circuit_open"
            self.model_policy.last_metrics = {}
        if not waiting_for_target_observation:
            selected, selection_override_reason = self._apply_repeat_guard(
                selected,
                eligible,
                candidate_snapshot.get("graph_context") or {},
            )
        with self.state_lock:
            latest_sequence = int(self.latest_candidates_payload.get("sequence", 0) or 0)
            latest_revision = int(
                self.latest_candidates_payload.get("graph_revision", 0) or 0
            )
            latest_candidate_ids = {
                str(item.get("candidate_id") or "")
                for item in self.latest_candidates_payload.get("candidates") or []
            }
        stale_selected = bool(
            selected is not None
            and latest_sequence != candidate_sequence
            and selected.candidate_id not in latest_candidate_ids
        )
        trace = {
            "timestamp": time.time(),
            "episode_id": candidate_snapshot.get("episode_id", ""),
            "input_graph_revision": candidate_snapshot.get("graph_revision", 0),
            "input_candidate_sequence": candidate_sequence,
            "publish_graph_revision": latest_revision,
            "publish_candidate_sequence": latest_sequence,
            "active_candidate_id": self.active_candidate_id,
            "policy_backend": self.policy_backend,
            "ablation": self.ablation.to_dict(),
            "model_error": self.model_policy.last_error,
            "model_result_source": self.model_policy.last_result_source,
            "model_metrics": dict(self.model_policy.last_metrics),
            "model_circuit_open": model_circuit_open,
            "model_circuit_open_until": self.model_circuit_breaker.open_until,
            "model_consecutive_timeouts": self.model_circuit_breaker.consecutive_timeouts,
            "model_ranked_group_ids": list(self.model_policy.last_ranking_ids),
            "model_selected_group_id": model_selected_group_id,
            "model_reason": self.model_policy.last_reason,
            "model_confidence": self.model_policy.last_confidence,
            "candidate_groups": list(self.model_policy.last_candidate_groups),
            "executed_candidate_id": selected.candidate_id if selected is not None else "",
            "executed_group_id": (
                candidate_group_id(selected, candidate_snapshot.get("graph_context") or {})
                if selected is not None
                else ""
            ),
            "selection_override_reason": selection_override_reason,
            "stale_selected_candidate": stale_selected,
            "recent_decisions": decision_history,
            "ranked_candidates": [
                candidate.to_dict()
                for candidate in sorted(scored, key=lambda item: (-item.score, item.candidate_id))
            ],
        }
        self.trace_pub.publish(
            String(data=json.dumps(trace, ensure_ascii=False, separators=(",", ":")))
        )
        if selected is None or stale_selected:
            return
        with self.state_lock:
            self.decision_index += 1
            decision_id = f"decision_{self.decision_index:06d}"
            executed_group_id = candidate_group_id(
                selected, candidate_snapshot.get("graph_context") or {}
            )
            selection = selected.to_dict()
            selection.update(
                {
                    "decision_id": decision_id,
                    "selected_at": time.time(),
                    "episode_id": candidate_snapshot.get("episode_id", ""),
                    "graph_revision": candidate_snapshot.get("graph_revision", 0),
                    "candidate_sequence": candidate_sequence,
                    "model_selected_group_id": model_selected_group_id,
                    "model_ranked_group_ids": list(self.model_policy.last_ranking_ids),
                    "model_reason": self.model_policy.last_reason,
                    "model_confidence": self.model_policy.last_confidence,
                    "executed_group_id": executed_group_id,
                    "selection_override_reason": selection_override_reason,
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
            self._record_decision_selection(
                decision_id,
                selected,
                executed_group_id,
                candidate_snapshot,
                model_selected_group_id,
                selection_override_reason,
            )
        self.selected_pub.publish(
            String(data=json.dumps(selection, ensure_ascii=False, separators=(",", ":")))
        )

    def _history_context(self, candidate_snapshot: dict) -> tuple[list[dict], list[dict]]:
        self._refresh_history_metrics(candidate_snapshot)
        observation_step = int(
            (candidate_snapshot.get("exploration_context") or {}).get(
                "observation_step", 0
            )
            or 0
        )
        with self.state_lock:
            recent = [
                {
                    "group_id": str(entry.get("group_id") or ""),
                    "candidate_id": str(entry.get("candidate_id") or ""),
                    "behavior_type": str(entry.get("behavior_type") or ""),
                    "result": str(entry.get("result") or "PENDING"),
                    "steps_ago": max(
                        0, observation_step - int(entry.get("observation_step", 0) or 0)
                    ),
                    "frontier_length_delta_m": round(
                        float(entry.get("frontier_length_delta_m", 0.0) or 0.0), 2
                    ),
                }
                for entry in list(self.decision_history)[-8:]
            ]
            groups = []
            for group_id, stats in sorted(self.group_history.items()):
                groups.append(
                    {
                        "group_id": group_id,
                        "selection_count": int(stats.get("selection_count", 0) or 0),
                        "consecutive_selection_count": int(
                            stats.get("consecutive_selection_count", 0) or 0
                        ),
                        "last_selected_steps_ago": max(
                            0,
                            observation_step
                            - int(stats.get("last_selected_step", 0) or 0),
                        ),
                        "last_result": str(stats.get("last_result") or "UNKNOWN"),
                        "last_frontier_length_delta_m": float(
                            stats.get("last_frontier_length_delta_m", 0.0) or 0.0
                        ),
                        "low_gain_repeat_count": int(
                            stats.get("low_gain_repeat_count", 0) or 0
                        ),
                    }
                )
        return recent, groups

    @staticmethod
    def _group_frontier_length(
        group_id: str, candidates: list[BehaviorCandidate], graph: dict
    ) -> float:
        projected, _ = compact_candidate_groups(candidates, graph)
        for item in projected:
            if str(item.get("id") or "") == group_id:
                return float(item.get("frontier_length_m", 0.0) or 0.0)
        return 0.0

    def _record_decision_selection(
        self,
        decision_id: str,
        selected: BehaviorCandidate,
        group_id: str,
        candidate_snapshot: dict,
        model_selected_group_id: str,
        override_reason: str,
    ) -> None:
        candidates = [
            BehaviorCandidate(**item)
            for item in candidate_snapshot.get("candidates") or []
        ]
        observation_step = int(
            (candidate_snapshot.get("exploration_context") or {}).get(
                "observation_step", 0
            )
            or 0
        )
        stats = self.group_history.setdefault(group_id, {})
        stats["selection_count"] = int(stats.get("selection_count", 0) or 0) + 1
        stats["consecutive_selection_count"] = (
            int(stats.get("consecutive_selection_count", 0) or 0) + 1
            if self.last_selected_group_id == group_id
            else 1
        )
        stats["last_selected_step"] = observation_step
        stats["last_result"] = "PENDING"
        self.last_selected_group_id = group_id
        self.decision_history.append(
            {
                "decision_id": decision_id,
                "group_id": group_id,
                "candidate_id": selected.candidate_id,
                "behavior_type": selected.behavior_type,
                "observation_step": observation_step,
                "frontier_length_before_m": self._group_frontier_length(
                    group_id,
                    candidates,
                    candidate_snapshot.get("graph_context") or {},
                ),
                "model_selected_group_id": model_selected_group_id,
                "override_reason": override_reason,
                "result": "PENDING",
                "frontier_metrics_evaluated": False,
            }
        )

    def _record_decision_result(self, payload: dict) -> None:
        decision_id = str(payload.get("decision_id") or "")
        entry = next(
            (
                item
                for item in reversed(self.decision_history)
                if str(item.get("decision_id") or "") == decision_id
            ),
            None,
        )
        if entry is None:
            return
        status = str(payload.get("status") or "UNKNOWN")
        entry["result"] = status
        group_id = str(entry.get("group_id") or "")
        entry["evaluate_after_sequence"] = (
            int(self.latest_candidates_payload.get("sequence", 0) or 0) + 1
        )
        stats = self.group_history.setdefault(group_id, {})
        stats["last_result"] = status

    def _refresh_history_metrics(self, candidate_snapshot: dict) -> None:
        sequence = int(candidate_snapshot.get("sequence", 0) or 0)
        candidates = [
            BehaviorCandidate(**item)
            for item in candidate_snapshot.get("candidates") or []
        ]
        graph = candidate_snapshot.get("graph_context") or {}
        with self.state_lock:
            for entry in self.decision_history:
                if bool(entry.get("frontier_metrics_evaluated")):
                    continue
                if str(entry.get("result") or "PENDING") == "PENDING":
                    continue
                if sequence < int(entry.get("evaluate_after_sequence", 0) or 0):
                    continue
                group_id = str(entry.get("group_id") or "")
                current_length = self._group_frontier_length(
                    group_id, candidates, graph
                )
                before_length = float(
                    entry.get("frontier_length_before_m", 0.0) or 0.0
                )
                delta = current_length - before_length
                entry["frontier_length_after_m"] = current_length
                entry["frontier_length_delta_m"] = delta
                entry["frontier_metrics_evaluated"] = True
                stats = self.group_history.setdefault(group_id, {})
                stats["last_frontier_length_delta_m"] = delta
                if (
                    str(entry.get("behavior_type") or "").upper() == "EXPLORE"
                    and str(entry.get("result") or "") == "SUCCEEDED"
                    and delta > -self.repeat_guard_min_frontier_shrink_m
                ):
                    stats["low_gain_repeat_count"] = int(
                        stats.get("low_gain_repeat_count", 0) or 0
                    ) + 1
                elif str(entry.get("result") or "") == "SUCCEEDED":
                    stats["low_gain_repeat_count"] = 0

    def _apply_repeat_guard(
        self,
        selected: BehaviorCandidate | None,
        eligible: list[BehaviorCandidate],
        graph: dict,
    ) -> tuple[BehaviorCandidate | None, str]:
        if (
            not self.repeat_guard_enabled
            or selected is None
            or str(selected.behavior_type or "").upper() != "EXPLORE"
        ):
            return selected, ""
        selected_group = candidate_group_id(selected, graph)
        with self.state_lock:
            stats = dict(self.group_history.get(selected_group) or {})
        if int(stats.get("low_gain_repeat_count", 0) or 0) < self.repeat_guard_low_gain_limit:
            return selected, ""
        projected, groups = compact_candidate_groups(eligible, graph)
        lengths = {
            str(item.get("id") or ""): float(item.get("frontier_length_m", 0.0) or 0.0)
            for item in projected
        }
        alternatives = [
            group_id
            for group_id in self.model_policy.last_ranking_ids
            if group_id != selected_group and group_id in groups
        ]
        alternatives.extend(
            group_id
            for group_id in groups
            if group_id != selected_group and group_id not in alternatives
        )
        if not alternatives:
            return selected, ""
        best_alternative = max(
            alternatives,
            key=lambda group_id: (
                lengths.get(group_id, 0.0),
                groups[group_id][0].score,
            ),
        )
        if lengths.get(selected_group, 0.0) >= lengths.get(best_alternative, 0.0):
            return selected, ""
        return groups[best_alternative][0], "repeat_guard_low_frontier_shrink"

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
