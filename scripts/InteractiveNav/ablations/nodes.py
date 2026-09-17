"""Subclass factories keep ROS imports and all interventions opt-in."""

import json
from .perception import PostActionVisualGuard

from .policies import (
    FlatCandidateCurator,
    FlatMemoryModelPolicy,
    CuratedGreedyPolicy,
    NearestInteractionPolicy,
    without_outcome_continuations,
)


def decision_node_class(base, variant):
    if variant == "full":
        return base
    if variant not in {"no_interaction_graph", "no_task_decision", "no_outcome_update"}:
        raise ValueError(variant)

    class AblationDecisionNode(base):
        def __init__(self):
            self._ablation_ready = False
            self._perception_guard = PostActionVisualGuard()
            super().__init__()
            with self.state_lock:
                if variant == "no_interaction_graph":
                    self.model_policy = FlatMemoryModelPolicy(self.model_policy.config)
                    self.candidate_curator = FlatCandidateCurator(self.candidate_curator.config)
                    self.policy = NearestInteractionPolicy()
                elif variant == "no_task_decision":
                    # Keep Full's curation, priority gates, fallback, and latest
                    # snapshot validation. Replace only the model-lane selector.
                    self.model_policy = CuratedGreedyPolicy(self.model_policy.config)
                self._ablation_ready = True
            if variant == "no_outcome_update":
                import rospy
                from std_msgs.msg import String
                topics = rospy.get_param("~topics", {})
                self._perception_result_sub = rospy.Subscriber(
                    topics.get("interaction_result", "/semantic_mapping/interaction_result"),
                    String, self._perception_result_callback, queue_size=20)

        def _perception_result_callback(self, message):
            try:
                payload = json.loads(message.data)
            except (ValueError, TypeError):
                return
            if not isinstance(payload, dict):
                return
            with self.state_lock:
                self._perception_guard.record(
                    payload, self._observation_step(self.latest_candidates_payload))

        def _tick(self, event):
            if self._ablation_ready:
                return super()._tick(event)

        def _handle_feedback(self, payload):
            # The native subscriber already holds state_lock. Preserve action lifecycle,
            # failure cooldowns and target arrival; remove only the semantic continuation.
            if (variant == "no_outcome_update"
                    and self.active_behavior_type == "INTERACT"
                    and payload.get("status") == "SUCCEEDED"
                    and (not payload.get("decision_id") or not self.active_decision_id
                         or payload["decision_id"] == self.active_decision_id)):
                active = self.active_interaction_candidate or {}
                if not (active.get("metadata") or {}).get("observation_only_reobserve"):
                    # Close the cross-topic race: feedback can arrive before the
                    # physical-result subscriber, but selection must already wait.
                    self._perception_guard.record(
                        {**(active.get("interaction_command") or {}),
                         "node_id": active.get("target_id"),
                         **(payload.get("detail") or {})},
                        self._observation_step(self.latest_candidates_payload))
            result = super()._handle_feedback(payload)
            if variant in {"no_outcome_update", "no_interaction_graph"}:
                self.pending_post_interaction_traversal = {}
                self.post_interaction_refresh_gate.clear()
                self.interaction_outcome_beliefs.clear()
            return result

        def _decide_from_snapshot(self, snapshot):
            if variant == "no_outcome_update":
                snapshot = self._perception_guard.apply(without_outcome_continuations(snapshot))
            return super()._decide_from_snapshot(snapshot)

        def _eligible_candidates_from_snapshot(self, snapshot, **kwargs):
            # Native selection also revalidates a newer snapshot after model inference.
            if variant == "no_outcome_update":
                snapshot = self._perception_guard.apply(without_outcome_continuations(snapshot))
            return super()._eligible_candidates_from_snapshot(snapshot, **kwargs)

    return AblationDecisionNode


def mapping_node_class(base, variant):
    if variant != "no_outcome_update":
        raise ValueError("Only no_outcome_update replaces the mapping node")

    class PerceptionOnlyMappingNode(base):
        def interaction_command_callback(self, msg):
            # Do not create the optimistic planning overlay for a suppressed result.
            pass

        def interaction_result_callback(self, msg):
            # Executor and decision subscribers still receive the same result topic.
            # Sensor/OCC/attribute callbacks are inherited without modification.
            pass

    return PerceptionOnlyMappingNode
