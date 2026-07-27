from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable

from .behavior_candidates import BehaviorCandidate


SUPPORTED_BEHAVIOR_TYPES = ("NAVIGATE", "INTERACT", "EXPLORE")


@dataclass
class CandidateCuratorConfig:
    candidate_top_k: int = 8
    navigate_quota: int = 1
    interaction_quota: int = 3
    explore_quota: int = 4
    max_frontiers_per_room: int = 2
    max_candidates_per_region: int = 1
    region_size_m: float = 1.0
    repeat_guard_low_gain_limit: int = 2
    explore_min_visible_gain_ratio: float = 0.25
    goal_position_tolerance_m: float = 0.35
    goal_yaw_tolerance_rad: float = 0.50


@dataclass
class CandidateCurationResult:
    candidates: list[BehaviorCandidate]
    rejected: dict[str, str] = field(default_factory=dict)
    omitted: dict[str, str] = field(default_factory=dict)
    quality_by_id: dict[str, float] = field(default_factory=dict)
    quality_terms_by_id: dict[str, dict[str, float]] = field(default_factory=dict)
    history_key_by_id: dict[str, str] = field(default_factory=dict)
    ranked_ids_by_type: dict[str, list[str]] = field(default_factory=dict)
    mandatory_ids: list[str] = field(default_factory=list)

    def trace(self) -> dict[str, Any]:
        return {
            "selected_ids": [candidate.candidate_id for candidate in self.candidates],
            "mandatory_ids": list(self.mandatory_ids),
            "ranked_ids_by_type": {
                key: list(value) for key, value in self.ranked_ids_by_type.items()
            },
            "quality_by_id": {
                key: round(float(value), 4)
                for key, value in self.quality_by_id.items()
            },
            "quality_terms_by_id": {
                candidate_id: {
                    key: round(float(value), 4) for key, value in terms.items()
                }
                for candidate_id, terms in self.quality_terms_by_id.items()
            },
            "history_key_by_id": dict(self.history_key_by_id),
            "rejected": dict(self.rejected),
            "omitted": dict(self.omitted),
        }


@dataclass
class CandidateValidationResult:
    valid: bool
    reason: str
    candidate: BehaviorCandidate | None = None


def _normalized_behavior_type(candidate: BehaviorCandidate) -> str:
    return str(candidate.behavior_type or "").strip().upper()


def _candidate_action(candidate: BehaviorCandidate) -> str:
    behavior_type = _normalized_behavior_type(candidate)
    if behavior_type == "INTERACT":
        return str((candidate.interaction_command or {}).get("action") or "open").casefold()
    if behavior_type == "NAVIGATE":
        return "navigate"
    return "explore"


def _room_node_id(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    return text if text.startswith("room_") else f"room_{text}"


def candidate_room_id(candidate: BehaviorCandidate, graph: dict[str, Any]) -> str:
    metadata = candidate.metadata or {}
    room_id = _room_node_id(metadata.get("target_room_id") or metadata.get("room_id"))
    if room_id:
        return room_id
    goal = list(candidate.goal_xyyaw or [])
    if len(goal) < 2:
        return "unknown"
    closest_id = "unknown"
    closest_distance_sq = float("inf")
    for node in graph.get("nodes") or []:
        if str(node.get("type") or "").casefold() != "room":
            continue
        centroid = list(node.get("centroid") or [])
        if len(centroid) < 2:
            continue
        distance_sq = (float(centroid[0]) - float(goal[0])) ** 2 + (
            float(centroid[1]) - float(goal[1])
        ) ** 2
        if distance_sq < closest_distance_sq:
            closest_distance_sq = distance_sq
            closest_id = str(node.get("id") or "unknown")
    return closest_id


def candidate_history_key(
    candidate: BehaviorCandidate,
    graph: dict[str, Any] | None = None,
    region_size_m: float = 1.0,
) -> str:
    behavior_type = _normalized_behavior_type(candidate)
    metadata = candidate.metadata or {}
    if behavior_type == "EXPLORE":
        point = list(metadata.get("frontier_point") or candidate.goal_xyyaw or [])
        if len(point) < 2:
            return f"explore_region:{candidate.candidate_id}"
        size = max(0.10, float(region_size_m))
        region_x = int(round(float(point[0]) / size))
        region_y = int(round(float(point[1]) / size))
        room_id = candidate_room_id(candidate, graph or {})
        return f"explore_region:{room_id}:{region_x}:{region_y}"
    if behavior_type == "INTERACT":
        group_id = str(
            (candidate.interaction_command or {}).get("interaction_group_id")
            or metadata.get("interaction_group_id")
            or "default"
        )
        return (
            f"interaction:{candidate.target_id or candidate.candidate_id}:"
            f"{_candidate_action(candidate)}:{group_id}"
        )
    return f"navigate:{candidate.target_id or candidate.candidate_id}"


def candidate_semantic_signature(candidate: BehaviorCandidate) -> dict[str, Any]:
    metadata = candidate.metadata or {}
    interaction = candidate.interaction_command or {}
    return {
        "candidate_id": candidate.candidate_id,
        "behavior_type": _normalized_behavior_type(candidate),
        "target_id": str(candidate.target_id or ""),
        "action": _candidate_action(candidate),
        "interaction_group_id": str(
            interaction.get("interaction_group_id")
            or metadata.get("interaction_group_id")
            or ""
        ),
        "expected_state": str(interaction.get("expected_state") or ""),
        "node_type": str(metadata.get("node_type") or ""),
    }


def candidate_fingerprint(candidate: BehaviorCandidate) -> str:
    payload = candidate_semantic_signature(candidate)
    goal = list(candidate.goal_xyyaw or [])
    payload["goal_xyyaw"] = [round(float(value), 3) for value in goal[:3]]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _goal_is_finite(candidate: BehaviorCandidate) -> bool:
    goal = list(candidate.goal_xyyaw or [])
    return len(goal) >= 2 and all(math.isfinite(float(value)) for value in goal[:3])


def candidate_rejection_reason(candidate: BehaviorCandidate) -> str:
    candidate_id = str(candidate.candidate_id or "")
    behavior_type = _normalized_behavior_type(candidate)
    metadata = candidate.metadata or {}
    if not candidate_id:
        return "missing_candidate_id"
    if behavior_type not in SUPPORTED_BEHAVIOR_TYPES:
        return "unsupported_behavior_type"
    if not _goal_is_finite(candidate):
        return "missing_or_invalid_goal_pose"
    if metadata.get("hard_constraints_passed") is False:
        return "hard_constraints_failed"
    for key in ("reachable", "path_reachable", "approach_reachable"):
        if metadata.get(key) is False:
            return f"{key}_false"
    if bool(metadata.get("interaction_group_already_explored")):
        return "interaction_group_already_explored"
    if bool(metadata.get("interaction_group_previously_failed")):
        return "interaction_group_previously_failed"
    if behavior_type == "INTERACT":
        action = _candidate_action(candidate)
        state = str(metadata.get("state") or "unknown").casefold()
        if action == "open" and state in {"open", "static_open", "completed"}:
            return "interaction_action_already_satisfied"
        if action == "close" and state in {"closed", "static_closed"}:
            return "interaction_action_already_satisfied"
    return ""


def validate_candidate_update(
    selected: BehaviorCandidate,
    latest_candidates: Iterable[BehaviorCandidate],
    *,
    position_tolerance_m: float = 0.35,
    yaw_tolerance_rad: float = 0.50,
) -> CandidateValidationResult:
    latest = next(
        (
            candidate
            for candidate in latest_candidates
            if candidate.candidate_id == selected.candidate_id
        ),
        None,
    )
    if latest is None:
        return CandidateValidationResult(False, "candidate_missing")
    rejection = candidate_rejection_reason(latest)
    if rejection:
        return CandidateValidationResult(False, rejection)
    if candidate_semantic_signature(latest) != candidate_semantic_signature(selected):
        return CandidateValidationResult(False, "candidate_semantics_changed")
    old_goal = list(selected.goal_xyyaw or [])
    new_goal = list(latest.goal_xyyaw or [])
    displacement = math.hypot(
        float(new_goal[0]) - float(old_goal[0]),
        float(new_goal[1]) - float(old_goal[1]),
    )
    if displacement > max(0.0, float(position_tolerance_m)):
        return CandidateValidationResult(False, "candidate_goal_moved")
    if len(old_goal) >= 3 and len(new_goal) >= 3:
        yaw_delta = abs(
            math.atan2(
                math.sin(float(new_goal[2]) - float(old_goal[2])),
                math.cos(float(new_goal[2]) - float(old_goal[2])),
            )
        )
        if yaw_delta > max(0.0, float(yaw_tolerance_rad)):
            return CandidateValidationResult(False, "candidate_goal_yaw_changed")
    return CandidateValidationResult(True, "candidate_content_compatible", latest)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _relative_scores(
    candidates: list[BehaviorCandidate],
    getter,
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    values = {candidate.candidate_id: float(getter(candidate)) for candidate in candidates}
    unique = sorted(set(values.values()))
    if len(unique) <= 1:
        return {candidate_id: 1.0 for candidate_id in values}
    rank = {value: index / float(len(unique) - 1) for index, value in enumerate(unique)}
    return {
        candidate_id: rank[value] if higher_is_better else 1.0 - rank[value]
        for candidate_id, value in values.items()
    }


class CandidateCurator:
    def __init__(self, config: CandidateCuratorConfig | None = None) -> None:
        self.config = config or CandidateCuratorConfig()

    def filter_candidates(
        self,
        candidates: Iterable[BehaviorCandidate],
        *,
        graph: dict[str, Any] | None = None,
        history_by_key: dict[str, dict[str, Any]] | None = None,
        observation_step: int = 0,
    ) -> tuple[list[BehaviorCandidate], dict[str, str]]:
        accepted = []
        rejected: dict[str, str] = {}
        seen_ids: set[str] = set()
        history_by_key = history_by_key or {}
        for candidate in candidates:
            candidate_id = str(candidate.candidate_id or "")
            if candidate_id in seen_ids:
                rejected[candidate_id] = "duplicate_candidate_id"
                continue
            seen_ids.add(candidate_id)
            reason = candidate_rejection_reason(candidate)
            history_key = candidate_history_key(
                candidate,
                graph or {},
                self.config.region_size_m,
            )
            history = history_by_key.get(history_key) or {}
            if not reason and int(history.get("cooldown_until_step", 0) or 0) > int(
                observation_step
            ):
                reason = "history_region_cooldown"
            if reason:
                rejected[candidate_id or f"missing_id_{len(rejected)}"] = reason
                continue
            accepted.append(candidate)
        return accepted, rejected

    def curate(
        self,
        candidates: Iterable[BehaviorCandidate],
        *,
        graph: dict[str, Any] | None = None,
        history_by_key: dict[str, dict[str, Any]] | None = None,
        observation_step: int = 0,
    ) -> CandidateCurationResult:
        graph = graph or {}
        history_by_key = history_by_key or {}
        accepted, rejected = self.filter_candidates(
            candidates,
            graph=graph,
            history_by_key=history_by_key,
            observation_step=observation_step,
        )
        history_key_by_id = {
            candidate.candidate_id: candidate_history_key(
                candidate, graph, self.config.region_size_m
            )
            for candidate in accepted
        }
        pools = {
            behavior_type: [
                candidate
                for candidate in accepted
                if _normalized_behavior_type(candidate) == behavior_type
            ]
            for behavior_type in SUPPORTED_BEHAVIOR_TYPES
        }
        explore_pool = pools["EXPLORE"]
        expected_visible_area_by_id = {
            candidate.candidate_id: max(
                0.0,
                float(
                    (candidate.metadata or {}).get(
                        "expected_visible_unknown_area_m2", 0.0
                    )
                    or 0.0
                ),
            )
            for candidate in explore_pool
        }
        max_expected_visible_area = max(
            expected_visible_area_by_id.values(), default=0.0
        )
        omitted: dict[str, str] = {}
        if max_expected_visible_area > 0.0:
            minimum_visible_area = max_expected_visible_area * max(
                0.0,
                min(1.0, float(self.config.explore_min_visible_gain_ratio)),
            )
            visible_gain_pool = [
                candidate
                for candidate in explore_pool
                if expected_visible_area_by_id[candidate.candidate_id]
                + 1e-9
                >= minimum_visible_area
            ]
            if visible_gain_pool:
                for candidate in explore_pool:
                    if candidate not in visible_gain_pool:
                        omitted[candidate.candidate_id] = (
                            "low_expected_visible_gain_suppressed"
                        )
                explore_pool = visible_gain_pool
                pools["EXPLORE"] = visible_gain_pool
        non_repeated_explore = [
            candidate
            for candidate in explore_pool
            if int(
                (
                    history_by_key.get(history_key_by_id[candidate.candidate_id]) or {}
                ).get("low_gain_repeat_count", 0)
                or 0
            )
            < max(1, int(self.config.repeat_guard_low_gain_limit))
        ]
        if non_repeated_explore:
            for candidate in explore_pool:
                if candidate not in non_repeated_explore:
                    omitted[candidate.candidate_id] = "history_low_gain_suppressed"
            pools["EXPLORE"] = non_repeated_explore

        quality_by_id: dict[str, float] = {}
        quality_terms_by_id: dict[str, dict[str, float]] = {}
        for behavior_type, pool in pools.items():
            scores, terms = self._score_pool(
                behavior_type,
                pool,
                history_by_key,
                history_key_by_id,
            )
            quality_by_id.update(scores)
            quality_terms_by_id.update(terms)
        ranked = {
            behavior_type: sorted(
                pool,
                key=lambda candidate: (
                    -quality_by_id.get(candidate.candidate_id, 0.0),
                    candidate.candidate_id,
                ),
            )
            for behavior_type, pool in pools.items()
        }
        ranked_ids_by_type = {
            behavior_type: [candidate.candidate_id for candidate in pool]
            for behavior_type, pool in ranked.items()
        }

        mandatory = sorted(
            [
                candidate
                for candidate in accepted
                if bool((candidate.metadata or {}).get("target_goal"))
                or bool((candidate.metadata or {}).get("target_match"))
            ],
            key=lambda candidate: (
                0
                if bool((candidate.metadata or {}).get("target_goal"))
                else 1,
                candidate.candidate_id,
            ),
        )
        selected: list[BehaviorCandidate] = []
        selected_ids: set[str] = set()
        selected_history_keys: set[str] = set()
        room_frontier_counts: dict[str, int] = {}
        top_k = max(1, int(self.config.candidate_top_k))

        def add(candidate: BehaviorCandidate, *, mandatory_candidate: bool = False) -> bool:
            if len(selected) >= top_k or candidate.candidate_id in selected_ids:
                return False
            behavior_type = _normalized_behavior_type(candidate)
            history_key = history_key_by_id[candidate.candidate_id]
            if not mandatory_candidate and behavior_type == "EXPLORE":
                if (
                    int(self.config.max_candidates_per_region) > 0
                    and history_key in selected_history_keys
                ):
                    return False
                room_id = candidate_room_id(candidate, graph)
                if (
                    int(self.config.max_frontiers_per_room) > 0
                    and room_frontier_counts.get(room_id, 0)
                    >= int(self.config.max_frontiers_per_room)
                ):
                    return False
                room_frontier_counts[room_id] = room_frontier_counts.get(room_id, 0) + 1
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
            selected_history_keys.add(history_key)
            return True

        for candidate in mandatory:
            add(candidate, mandatory_candidate=True)

        quotas = {
            "NAVIGATE": max(0, int(self.config.navigate_quota)),
            "INTERACT": max(0, int(self.config.interaction_quota)),
            "EXPLORE": max(0, int(self.config.explore_quota)),
        }
        for behavior_type in SUPPORTED_BEHAVIOR_TYPES:
            selected_count = sum(
                _normalized_behavior_type(candidate) == behavior_type
                for candidate in selected
            )
            for candidate in ranked[behavior_type]:
                if selected_count >= quotas[behavior_type] or len(selected) >= top_k:
                    break
                if add(candidate):
                    selected_count += 1

        cursor = {behavior_type: 0 for behavior_type in SUPPORTED_BEHAVIOR_TYPES}
        fill_order = ("INTERACT", "EXPLORE", "NAVIGATE")
        while len(selected) < top_k:
            added = False
            for behavior_type in fill_order:
                pool = ranked[behavior_type]
                while cursor[behavior_type] < len(pool):
                    candidate = pool[cursor[behavior_type]]
                    cursor[behavior_type] += 1
                    if add(candidate):
                        added = True
                        break
                if len(selected) >= top_k:
                    break
            if not added:
                break

        for candidate in accepted:
            if candidate.candidate_id not in selected_ids and candidate.candidate_id not in omitted:
                omitted[candidate.candidate_id] = "top_k_or_diversity_limit"
        return CandidateCurationResult(
            candidates=selected,
            rejected=rejected,
            omitted=omitted,
            quality_by_id=quality_by_id,
            quality_terms_by_id=quality_terms_by_id,
            history_key_by_id=history_key_by_id,
            ranked_ids_by_type=ranked_ids_by_type,
            mandatory_ids=[candidate.candidate_id for candidate in mandatory],
        )

    def _score_pool(
        self,
        behavior_type: str,
        pool: list[BehaviorCandidate],
        history_by_key: dict[str, dict[str, Any]],
        history_key_by_id: dict[str, str],
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        if not pool:
            return {}, {}
        distance_rank = _relative_scores(
            pool,
            lambda candidate: max(
                0.0,
                float(
                    candidate.features.get(
                        "path_length_m",
                        candidate.features.get("distance_m", 0.0),
                    )
                    or 0.0
                ),
            ),
            higher_is_better=False,
        )
        history_novelty = {
            candidate.candidate_id: 1.0
            / (
                1.0
                + float(
                    (
                        history_by_key.get(history_key_by_id[candidate.candidate_id])
                        or {}
                    ).get("selection_count", 0)
                    or 0
                )
            )
            for candidate in pool
        }
        if behavior_type == "EXPLORE":
            information_rank = _relative_scores(
                pool,
                lambda candidate: max(
                    0.0, float(candidate.features.get("exploration_gain", 0.0) or 0.0)
                ),
                higher_is_better=True,
            )
            stability_rank = _relative_scores(
                pool,
                lambda candidate: max(
                    0.0,
                    float(
                        (candidate.metadata or {}).get(
                            "stable_sequence_count",
                            (candidate.metadata or {}).get("candidate_age_sequences", 1.0),
                        )
                        or 0.0
                    ),
                ),
                higher_is_better=True,
            )
            terms = {
                candidate.candidate_id: {
                    "information_gain_rank": 0.45
                    * information_rank[candidate.candidate_id],
                    "path_efficiency_rank": 0.25
                    * distance_rank[candidate.candidate_id],
                    "spatial_novelty": 0.20
                    * history_novelty[candidate.candidate_id],
                    "frontier_stability": 0.10
                    * stability_rank[candidate.candidate_id],
                }
                for candidate in pool
            }
        elif behavior_type == "INTERACT":
            cost_rank = _relative_scores(
                pool,
                lambda candidate: max(
                    0.0, float(candidate.features.get("interaction_cost", 0.0) or 0.0)
                ),
                higher_is_better=False,
            )
            terms = {
                candidate.candidate_id: {
                    "approach_efficiency": 0.30
                    * distance_rank[candidate.candidate_id],
                    "state_confidence": 0.20
                    * _clamp01(candidate.features.get("confidence", 0.0)),
                    "structural_gain": 0.25 * self._structural_gain(candidate),
                    "cost_efficiency": 0.15 * cost_rank[candidate.candidate_id],
                    "interaction_novelty": 0.10
                    * history_novelty[candidate.candidate_id],
                }
                for candidate in pool
            }
        else:
            terms = {
                candidate.candidate_id: {
                    "target_confidence": 0.40
                    * _clamp01(candidate.features.get("confidence", 0.0)),
                    "path_efficiency_rank": 0.35
                    * distance_rank[candidate.candidate_id],
                    "goal_pose_quality": 0.25
                    * (1.0 if len(candidate.goal_xyyaw or []) >= 3 else 0.5),
                }
                for candidate in pool
            }
        scores = {
            candidate_id: sum(candidate_terms.values())
            for candidate_id, candidate_terms in terms.items()
        }
        return scores, terms

    @staticmethod
    def _structural_gain(candidate: BehaviorCandidate) -> float:
        metadata = candidate.metadata or {}
        node_type = str(metadata.get("node_type") or "").casefold()
        effect = str(metadata.get("expected_effect") or "").casefold()
        state = str(metadata.get("state") or "unknown").casefold()
        if node_type == "portal":
            connected_rooms = list(metadata.get("connected_room_ids") or [])
            return 1.0 if len(connected_rooms) >= 2 or "access" in effect else 0.75
        if node_type == "container" and state not in {"open", "static_open", "completed"}:
            return 1.0 if "reveal" in effect or "content" in effect else 0.75
        return 0.5 if effect else 0.0
