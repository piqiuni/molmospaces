"""Stateless object memory at the planning boundary; local execution stays guarded."""

from copy import deepcopy
from dataclasses import replace

from semantic_decision_py_pkg.behavior_candidates import CandidateGenerator


OBJECT_FIELDS = {
    "id", "type", "label", "name", "semantic_name", "confidence", "centroid",
    "aabb_center", "aabb_extent", "aabb_size", "is_currently_visible", "last_seen",
    "first_seen", "observation_count", "source_object_id", "source_object_name",
}
VISUAL_FIELDS = {
    "visible_pixels", "visible_fraction", "consecutive_observations", "frame_index",
    "observation_evidence", "source_object_name", "instance_id", "semantic_name",
}


def flat_objects(graph):
    """Never retain room/containment edges or open/closed/scanned/result fields."""
    result = {key: deepcopy(graph[key]) for key in
              ("episode_id", "graph_revision", "capture_step", "stamp_sec") if key in graph}
    result.update(nodes=[], edges=[])
    for node in graph.get("nodes", []):
        if node.get("type") in {"room", "floor", "building", "scene", "robot"}:
            continue
        item = {key: deepcopy(value) for key, value in node.items() if key in OBJECT_FIELDS}
        # observation_evidence can carry inferred state, so retain only visual counts.
        item["attributes"] = {key: deepcopy(value) for key, value in
                              (node.get("attributes") or {}).items()
                              if key in VISUAL_FIELDS - {"observation_evidence"}}
        result["nodes"].append(item)
    return result


def strip_relational_candidate(candidate):
    """Remove planning annotations, retaining geometry and execution preconditions."""
    for payload in (candidate.features, candidate.metadata):
        for key in list(payload):
            if ("room" in key or "contain" in key and key.startswith(("containing_", "target_"))
                    or key in {"state", "state_age_sec", "state_age_ratio", "expected_effect", "interaction_effect",
                               "requires_interaction", "traversable", "operation_history",
                               "drawer_scan_completed", "target_relevance", "connectivity_gain"}):
                payload.pop(key, None)
    candidate.source = "flat_object_memory"
    return candidate


class FlatObjectCandidateGenerator(CandidateGenerator):
    def __init__(self, config):
        super().__init__(replace(
            config, portal_unknown_default_interact=True,
            portal_require_attribute_ready=False, min_state_confidence=0.0,
            remembered_portal_reobservation_enabled=False,
        ))

    def generate(self, explorer_status, graph, robot_xy, target_context=None,
                 room_segment_grid=None, **kwargs):
        memory = flat_objects(graph or {})
        # These are operation proposals, not inferred physical states. Every
        # interaction is routed through the original executor's fresh-view gate.
        for node in memory["nodes"]:
            if node.get("type") in {"portal", "container"}:
                node["interaction"] = {"is_interactable": True, "state": "unknown",
                                       "requires_interaction": True}
        candidates = super().generate(explorer_status, memory, robot_xy, target_context,
                                      None, **kwargs)
        for candidate in candidates:
            strip_relational_candidate(candidate)
            if candidate.behavior_type == "INTERACT":
                candidate.metadata.update(observation_required=True, reobserve=True,
                                          observation_reason="flat_memory_fresh_view_required")
        return candidates

    @classmethod
    def _target_navigation_anchor(cls, node, graph, nodes_by_id):
        return node

    def _portal_traversal_candidates(self, graph, robot_xy):
        return []

    @staticmethod
    def _attach_portal_child_room_context(candidates, graph):
        pass

    @classmethod
    def _attach_frontier_room_context(cls, candidates, graph, robot_xy):
        pass

    def _attach_spatial_context(self, candidates, graph):
        pass


def candidate_node_class(base):
    class FlatObjectCandidateNode(base):
        def __init__(self):
            self._ablation_ready = False
            super().__init__()
            self.generator = FlatObjectCandidateGenerator(self.generator.config)
            self.graph = flat_objects(self.graph)
            self._ablation_ready = True

        def _graph_callback(self, message):
            import json
            try:
                graph = flat_objects(json.loads(message.data))
            except (ValueError, TypeError):
                return
            self.graph = graph
            self.startup_scan_lifecycle.observe_episode(str(graph.get("episode_id") or ""))

        def _publish(self, event):
            if self._ablation_ready:
                return super()._publish(event)

    return FlatObjectCandidateNode
