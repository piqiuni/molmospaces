"""Decision-only interventions. Executable candidates retain their geometry."""

from copy import deepcopy
import math

from semantic_decision_py_pkg.candidate_curator import (
    CandidateCurationResult,
    CandidateCurator,
    candidate_history_key,
)
from semantic_decision_py_pkg.model_policy import (
    ModelPolicyClient,
    build_subgoal_selection_response_schema,
    compact_candidate,
    compact_target_context,
)


def distance(candidate):
    try:
        value = float(candidate.features.get("distance_m", math.inf))
    except (TypeError, ValueError):
        return math.inf
    return max(0.0, value) if math.isfinite(value) else math.inf


class NearestInteractionPolicy:
    """Task-independent search, after the shared scan/visible-goal gates."""

    @staticmethod
    def key(candidate):
        if candidate.metadata.get("target_goal") and candidate.metadata.get("target_reliably_observed"):
            priority = 0
        elif candidate.behavior_type == "SCAN":
            priority = 1
        elif candidate.behavior_type == "INTERACT":
            priority = 2
        else:
            priority = 3
        return priority, distance(candidate), candidate.candidate_id

    def score(self, candidate, current_candidate_id=""):
        priority, value, _ = self.key(candidate)
        cost = value / (1.0 + value) if math.isfinite(value) else 1.0
        candidate.score = -float(priority) - cost
        candidate.score_terms = {"ablation_priority": -priority, "ablation_distance_cost": -cost}
        return candidate

    def select(self, candidates, current_candidate_id=""):
        candidates = list(candidates)
        for candidate in candidates:
            self.score(candidate)
        return min(candidates, key=self.key) if candidates else None


class FlatCandidateCurator(CandidateCurator):
    """Keep native eligibility/cooldowns; replace relational ranking by distance."""

    def curate(self, candidates, *, graph=None, history_by_key=None, observation_step=0,
               target_context=None, entered_room_ids=None):
        accepted, rejected = self.filter_candidates(
            candidates, graph=graph, history_by_key=history_by_key,
            observation_step=observation_step,
        )
        quotas = {"NAVIGATE": self.config.navigate_quota,
                  "INTERACT": self.config.interaction_quota,
                  "EXPLORE": self.config.explore_quota}
        ranked = {kind: sorted((c for c in accepted if c.behavior_type == kind),
                               key=lambda c: (distance(c), c.candidate_id))
                  for kind in quotas}
        selected = [c for kind, pool in ranked.items() for c in pool[:max(0, quotas[kind])]]
        selected = selected[:max(0, self.config.candidate_top_k)]
        selected_ids = {c.candidate_id for c in selected}
        return CandidateCurationResult(
            candidates=selected,
            rejected=rejected,
            omitted={c.candidate_id: "ablation_distance_quota" for c in accepted
                     if c.candidate_id not in selected_ids},
            history_key_by_id={c.candidate_id: candidate_history_key(
                c, graph or {}, self.config.region_size_m) for c in accepted},
            ranked_ids_by_type={kind: [c.candidate_id for c in pool] for kind, pool in ranked.items()},
        )


class FlatMemoryModelPolicy(ModelPolicyClient):
    """Same model and response validation, without explicit graph reasoning."""

    def __init__(self, config):
        if config.selection_granularity.casefold() != "candidate":
            raise ValueError("no_interaction_graph requires candidate selection granularity")
        super().__init__(config)

    def select(self, candidates, target_context=None, graph=None, robot_context=None, metrics_context=None):
        # Prevent native fallback/guard paths from reintroducing graph-derived scores.
        context = {key: value for key, value in (robot_context or {}).items()
                   if key not in {"candidate_pre_scores", "candidate_pre_score_terms",
                                  "candidate_decision_hints", "room_frontier_lengths"}}
        return super().select(candidates, target_context, graph, context, metrics_context)

    def build_request(self, candidates, target_context, graph, robot_context=None):
        candidate_keys = {"id", "action", "subject_id", "subject_type", "subject_name",
                          "subject_semantic_type", "distance_m", "state",
                          "unknown_component_area_m2", "expected_visible_unknown_area_m2"}
        options = [{key: deepcopy(value) for key, value in compact_candidate(c).items()
                    if key in candidate_keys} for c in candidates]
        self.last_candidate_groups = options
        node_keys = {"id", "type", "label", "name", "centroid", "is_currently_visible",
                     "state_age_sec", "interaction_state"}
        memory = [{key: deepcopy(value) for key, value in node.items() if key in node_keys}
                  for node in graph.get("nodes", []) if node.get("type") not in {"room", "floor", "building"}]
        history_keys = {"candidate_id", "behavior_type", "target_id", "status", "result"}
        history = [{key: deepcopy(value) for key, value in row.items()
                    if key in history_keys and isinstance(value, (str, int, float, bool))}
                   for row in (robot_context or {}).get("decision_history", [])[-8:]]
        return {
            "schema_version": 4,
            "instruction": (
                "Find the requested target using the observed flat object memory and executable actions. "
                "Rank up to three CURRENT candidate IDs unchanged. Compare object semantics, observed "
                "state, distance and exploration gain. Avoid repeated completed or failed actions when "
                "alternatives exist. Never invent observations or select historical IDs. Return only "
                '{"ranked_ids":[...],"reason":"...","confidence":"..."}, ranked_ids first. '
                "Allowed reason: TARGET_VISIBLE, REVEAL_TARGET_CONTAINER, UNLOCK_ROUTE, EXPLORE_TARGET_ROOM, "
                "INFORMATION_GAIN, RECOVERY_DIVERSIFICATION, DISTANCE_TIEBREAK, NO_SEMANTIC_PREFERENCE. "
                "Allowed confidence: low, medium, high. No extra keys or prose."
            ),
            "mission": compact_target_context(target_context),
            "object_memory": memory[:self.config.max_graph_nodes],
            "recent_decisions": history,
            "candidates": options,
        }

    def _request_http(self, payload, metrics_context=None):
        # The native HTTP adapter has a fixed graph-context allowlist, so explicitly
        # transmit flat memory here instead of silently losing it at serialization.
        response = self._mllm_client.request_json(
            role="subgoal_selection",
            instruction=payload["instruction"],
            context={key: payload[key] for key in
                     ("mission", "object_memory", "candidates", "recent_decisions")},
            response_schema=build_subgoal_selection_response_schema(payload),
            timeout_s=self.config.timeout_s,
            max_tokens=self.config.max_tokens,
            metrics_context=metrics_context,
        )
        self.last_metrics = response.metrics()
        if response.error:
            raise ValueError(response.error)
        if not isinstance(response.payload, dict):
            raise ValueError("model HTTP response must be a JSON object")
        return response.payload


def without_outcome_continuations(snapshot):
    """Ordinary OCC-confirmed static doorway traversal remains available."""
    result = deepcopy(snapshot)
    result["candidates"] = [c for c in result.get("candidates", [])
                            if not c.get("metadata", {}).get("post_interaction_traversal")
                            or c.get("metadata", {}).get("static_open_occ_confirmed")]
    result["candidate_count"] = len(result["candidates"])
    return result
