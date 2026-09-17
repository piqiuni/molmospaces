"""Recover through-door goals from perception without an action-result history."""

from copy import deepcopy

from semantic_decision_py_pkg.behavior_candidates import CandidateGenerator


class PerceptionOnlyCandidateGenerator(CandidateGenerator):
    def generate(self, explorer_status, graph, *args, **kwargs):
        candidates = super().generate(explorer_status, graph, *args, **kwargs)
        nodes = {str(n.get("id") or ""): n for n in (graph or {}).get("nodes", [])}
        for candidate in candidates:
            node = nodes.get(candidate.target_id, {})
            attrs = node.get("attributes") or {}
            try:
                step = int(attrs.get("attribute_observation_frame_index"))
            except (TypeError, ValueError):
                step = -1
            candidate.metadata.update(
                perception_visual_step=step,
                perception_visual_ready=(attrs.get("attribute_status") == "ready"
                    and (node.get("type") != "portal"
                         or attrs.get("portal_state_consensus_accepted") is True)),
            )
        return candidates

    def _portal_traversal_candidates(self, graph, robot_xy):
        observed = deepcopy(graph)
        sensor_open_ids = set()
        for node in observed.get("nodes", []):
            if node.get("type") != "portal":
                continue
            interaction = node.get("interaction") or {}
            interaction.pop("operation_history", None)
            if interaction.get("state") == "open":
                # Reuse the native OCC/visual-consensus gate and geometry. This
                # temporary label never enters the graph or M1's state cache.
                interaction["state"] = "static_open"
                sensor_open_ids.add(str(node.get("id") or ""))
        candidates = super()._portal_traversal_candidates(observed, robot_xy)
        for candidate in candidates:
            if candidate.target_id in sensor_open_ids:
                candidate.source = "perception_confirmed_portal"
                candidate.metadata.update(state="open", static_open_occ_confirmed=False,
                                          perception_open_occ_confirmed=True)
        return candidates


def candidate_node_class(base):
    class PerceptionOnlyCandidateNode(base):
        def __init__(self):
            self._ablation_ready = False
            super().__init__()
            self.generator = PerceptionOnlyCandidateGenerator(self.generator.config)
            self._ablation_ready = True

        def _publish(self, event):
            if self._ablation_ready:
                return super()._publish(event)

    return PerceptionOnlyCandidateNode
