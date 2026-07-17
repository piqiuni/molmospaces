from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .behavior_candidates import BehaviorCandidate


@dataclass
class RulePolicyConfig:
    exploration_gain_weight: float = 1.0
    visibility_gain_weight: float = 0.8
    semantic_gain_weight: float = 0.6
    confidence_weight: float = 0.25
    priority_weight: float = 0.45
    distance_cost_weight: float = 0.45
    distance_normalization_m: float = 6.0
    interaction_cost_weight: float = 0.30
    staleness_cost_weight: float = 0.35
    portal_bonus: float = 0.55
    container_bonus: float = 0.15
    continuity_bonus: float = 0.25
    minimum_score: float = -1e9


class RulePolicy:
    def __init__(self, config: RulePolicyConfig | None = None) -> None:
        self.config = config or RulePolicyConfig()

    def score(
        self, candidate: BehaviorCandidate, current_candidate_id: str = ""
    ) -> BehaviorCandidate:
        features = candidate.features
        distance_ratio = max(0.0, float(features.get("distance_m", 0.0))) / max(
            self.config.distance_normalization_m, 1e-6
        )
        terms = {
            "exploration_gain": self.config.exploration_gain_weight
            * float(features.get("exploration_gain", 0.0)),
            "visibility_gain": self.config.visibility_gain_weight
            * float(features.get("visibility_gain", 0.0)),
            "semantic_gain": self.config.semantic_gain_weight
            * float(features.get("semantic_gain", 0.0)),
            "confidence": self.config.confidence_weight
            * float(features.get("confidence", 0.0)),
            "priority": self.config.priority_weight
            * float(features.get("priority", 0.0)),
            "distance_cost": -self.config.distance_cost_weight * distance_ratio,
            "interaction_cost": -self.config.interaction_cost_weight
            * float(features.get("interaction_cost", 0.0)),
            "staleness_cost": -self.config.staleness_cost_weight
            * float(features.get("state_age_ratio", 0.0)),
            "type_bonus": self._type_bonus(candidate),
            "continuity": (
                self.config.continuity_bonus
                if current_candidate_id and candidate.candidate_id == current_candidate_id
                else 0.0
            ),
        }
        candidate.score_terms = terms
        candidate.score = sum(terms.values())
        return candidate

    def select(
        self,
        candidates: Iterable[BehaviorCandidate],
        current_candidate_id: str = "",
    ) -> BehaviorCandidate | None:
        scored = [self.score(candidate, current_candidate_id) for candidate in candidates]
        scored = [candidate for candidate in scored if candidate.score >= self.config.minimum_score]
        if not scored:
            return None
        scored.sort(key=lambda candidate: (-candidate.score, candidate.candidate_id))
        return scored[0]

    def _type_bonus(self, candidate: BehaviorCandidate) -> float:
        node_type = str(candidate.metadata.get("node_type") or "")
        if node_type == "portal":
            return self.config.portal_bonus
        if node_type == "container":
            return self.config.container_bonus
        return 0.0
