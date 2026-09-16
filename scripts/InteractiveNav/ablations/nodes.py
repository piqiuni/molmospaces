"""Subclass factories keep ROS imports and all interventions opt-in."""

from .policies import (
    FlatCandidateCurator,
    FlatMemoryModelPolicy,
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
            super().__init__()
            with self.state_lock:
                if variant == "no_interaction_graph":
                    self.model_policy = FlatMemoryModelPolicy(self.model_policy.config)
                    self.candidate_curator = FlatCandidateCurator(self.candidate_curator.config)
                    self.policy = NearestInteractionPolicy()
                elif variant == "no_task_decision":
                    self.policy_backend = "rule"
                    self.policy = NearestInteractionPolicy()
                self._ablation_ready = True

        def _tick(self, event):
            if self._ablation_ready:
                return super()._tick(event)

        def _handle_feedback(self, payload):
            # The native subscriber already holds state_lock. Preserve action lifecycle,
            # failure cooldowns and target arrival; remove only the semantic continuation.
            result = super()._handle_feedback(payload)
            if variant == "no_outcome_update":
                self.pending_post_interaction_traversal = {}
                self.post_interaction_refresh_gate.clear()
                self.interaction_outcome_beliefs.clear()
            return result

        def _decide_from_snapshot(self, snapshot):
            if variant == "no_outcome_update":
                snapshot = without_outcome_continuations(snapshot)
            return super()._decide_from_snapshot(snapshot)

        def _eligible_candidates_from_snapshot(self, snapshot, **kwargs):
            # Native selection also revalidates a newer snapshot after model inference.
            if variant == "no_outcome_update":
                snapshot = without_outcome_continuations(snapshot)
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
