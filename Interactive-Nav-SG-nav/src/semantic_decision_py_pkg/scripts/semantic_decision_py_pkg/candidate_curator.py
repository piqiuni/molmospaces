from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
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
    # Keep the visibility term stronger than the room-novelty term: a clearly
    # larger observed unknown region should not lose merely because it is in
    # a room that was seen earlier.
    explore_visible_unknown_area_weight: float = 0.50
    explore_new_room_bonus: float = 0.15
    goal_position_tolerance_m: float = 0.35
    goal_yaw_tolerance_rad: float = 0.50
    # In object-goal mode an observed container whose semantics contradict the
    # requested object is noise for the MLLM.  Keep this opt-in at the curator
    # boundary (rather than the generator) so exploration can still retain the
    # observation as a fallback when no better proposal exists.
    suppress_semantic_container_mismatch: bool = True


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
    decision_hint_by_id: dict[str, str] = field(default_factory=dict)

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
            "decision_hint_by_id": dict(self.decision_hint_by_id),
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


def _explicit_candidate_room_id(candidate: BehaviorCandidate) -> str:
    """Return an observation-provided room id, never a geometric fallback."""
    metadata = candidate.metadata or {}
    room_id = _room_node_id(metadata.get("target_room_id") or metadata.get("room_id"))
    return "" if room_id in {"", "unknown", "room_unknown"} else room_id


def candidate_room_id(candidate: BehaviorCandidate, graph: dict[str, Any]) -> str:
    room_id = _explicit_candidate_room_id(candidate)
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


def _tokens(value: Any) -> set[str]:
    """Return small semantic tokens without adding a model-side dependency."""
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(value or "").casefold())
        if token
    }


# These are deliberately broad, conservative priors.  They are not ground
# truth and are only used to remove an obvious contradiction (for example a
# dresser candidate for a potato); unknown objects/containers remain eligible.
_FOOD_TOKENS = {
    "apple",
    "banana",
    "bread",
    "carrot",
    "cheese",
    "chicken",
    "cookie",
    "drink",
    "food",
    "fruit",
    "juice",
    "milk",
    "orange",
    "potato",
    "rice",
    "snack",
    "soda",
    "tomato",
    "vegetable",
    "water",
    "wine",
}
_PERSONAL_TOKENS = {
    "alarmclock",
    "book",
    "clock",
    "key",
    "keys",
    "laptop",
    "notebook",
    "pen",
    "pencil",
    "phone",
    "remote",
    "sock",
    "toy",
    "watch",
}
_KITCHEN_ITEM_TOKENS = {
    "bottle",
    "bowl",
    "cup",
    "fork",
    "kettle",
    "knife",
    "mug",
    "pan",
    "plate",
    "pot",
    "spoon",
}
_KITCHEN_CONTAINER_TOKENS = {
    "cabinet",
    "cupboard",
    "drawer",
    "fridge",
    "pantry",
    "refrigerator",
    "shelf",
    "storage",
}
_PERSONAL_CONTAINER_TOKENS = {
    "cabinet",
    "desk",
    "drawer",
    "dresser",
    "nightstand",
    "shelf",
    "wardrobe",
}
_STRONGLY_PERSONAL_CONTAINERS = {"dresser", "nightstand", "wardrobe", "desk"}
_STRONGLY_KITCHEN_CONTAINERS = {"fridge", "refrigerator", "oven", "dishwasher"}


def _target_tokens(target_context: dict[str, Any] | None) -> set[str]:
    target_context = target_context or {}
    values = list(target_context.get("object_labels") or [])
    values.extend(
        target_context.get(key)
        for key in ("target_name", "target_object", "object_label")
        if target_context.get(key)
    )
    result: set[str] = set()
    for value in values:
        result.update(_tokens(value))
    return result


def _container_tokens(candidate: BehaviorCandidate) -> set[str]:
    metadata = candidate.metadata or {}
    values = [
        candidate.target_name,
        metadata.get("semantic_name"),
        metadata.get("category"),
        metadata.get("source_object_name"),
    ]
    result: set[str] = set()
    for value in values:
        result.update(_tokens(value))
    return result


def _contains_semantic_term(tokens: set[str], vocabulary: set[str]) -> bool:
    return any(
        term == token or term in token or token in term
        for token in tokens
        for term in vocabulary
    )


def _semantic_container_compatibility(
    candidate: BehaviorCandidate,
    target_context: dict[str, Any] | None,
) -> tuple[str, float]:
    """Classify an interaction container against the *public* goal text.

    Returns ``(reason, compatibility)`` where compatibility is 1 for a
    plausible match, 0 for an explicit contradiction, and 0.5 when no safe
    semantic conclusion can be made.  This intentionally never consults
    hidden simulator state or target-room metadata.
    """
    if _normalized_behavior_type(candidate) != "INTERACT":
        return "not_container", 0.5
    metadata = candidate.metadata or {}
    if str(metadata.get("node_type") or "").casefold() != "container":
        return "not_container", 0.5
    if bool(metadata.get("target_match")):
        return "explicit_target_match", 1.0
    target_context = target_context or {}
    if not bool(target_context.get("enabled")):
        return "target_disabled", 0.5
    target = _target_tokens(target_context)
    containers = _container_tokens(candidate)
    if not target or not containers:
        return "semantic_unknown", 0.5
    is_food = _contains_semantic_term(target, _FOOD_TOKENS)
    is_kitchen_item = _contains_semantic_term(target, _KITCHEN_ITEM_TOKENS)
    is_kitchen_target = is_food or is_kitchen_item
    is_personal = _contains_semantic_term(target, _PERSONAL_TOKENS)
    is_kitchen_container = _contains_semantic_term(
        containers, _KITCHEN_CONTAINER_TOKENS
    )
    is_personal_container = _contains_semantic_term(
        containers, _PERSONAL_CONTAINER_TOKENS
    )
    if is_kitchen_target and is_personal_container and not is_kitchen_container:
        return "target_container_semantic_mismatch", 0.0
    if is_personal and is_kitchen_container and not is_personal_container:
        return "target_container_semantic_mismatch", 0.0
    if is_kitchen_target and is_kitchen_container:
        return "plausible_kitchen_container", 1.0
    if is_personal and is_personal_container:
        return "plausible_personal_container", 1.0
    # A strongly typed contradiction is safer than a weak lexical overlap.
    if is_kitchen_target and _contains_semantic_term(
        containers, _STRONGLY_PERSONAL_CONTAINERS
    ):
        return "target_container_semantic_mismatch", 0.0
    if is_personal and _contains_semantic_term(
        containers, _STRONGLY_KITCHEN_CONTAINERS
    ):
        return "target_container_semantic_mismatch", 0.0
    return "semantic_unknown", 0.5


def _normalized_room_id(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    return text if text.startswith("room_") else f"room_{text}"


def _candidate_room(value: Any) -> str:
    return _normalized_room_id(value)


def _portal_endpoints(candidate: BehaviorCandidate) -> set[str]:
    metadata = candidate.metadata or {}
    return {
        _normalized_room_id(value)
        for value in metadata.get("connected_room_ids") or []
        if value not in (None, "")
    }


def _topology_hints(
    candidates: list[BehaviorCandidate],
    graph: dict[str, Any],
    target_context: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, str]]:
    """Compute deterministic route/mission priors before asking the MLLM.

    The graph is observation-derived.  Closed portal edges are treated as
    *potential* edges for ranking only; execution still goes through the
    normal interaction policy.  This lets the first portal on an observed
    route outrank a nearby but irrelevant container without leaking GT.
    """
    candidates = list(candidates)
    target_rooms: set[str] = set()
    current_rooms: set[str] = set()
    for candidate in candidates:
        metadata = candidate.metadata or {}
        if metadata.get("target_goal") or metadata.get("target_match"):
            room = _candidate_room(metadata.get("target_room_id") or metadata.get("room_id"))
            if room:
                target_rooms.add(room)
        room = _candidate_room(metadata.get("robot_room_id"))
        if room:
            current_rooms.add(room)

    # Include observed portal nodes not currently represented as candidates so
    # route distance reflects the whole semantic graph.
    portal_endpoints_by_id: dict[str, set[str]] = {}
    for node in graph.get("nodes") or []:
        if str(node.get("type") or "").casefold() != "portal":
            continue
        attributes = node.get("attributes") or {}
        values = node.get("connected_room_ids")
        if values is None:
            values = attributes.get("connected_room_ids") or []
        endpoints = {
            _normalized_room_id(value) for value in values if value not in (None, "")
        }
        if endpoints:
            portal_endpoints_by_id[str(node.get("id") or "")] = endpoints
    for candidate in candidates:
        if _normalized_behavior_type(candidate) == "INTERACT":
            node_type = str((candidate.metadata or {}).get("node_type") or "").casefold()
            if node_type == "portal":
                endpoints = _portal_endpoints(candidate)
                if endpoints:
                    portal_endpoints_by_id.setdefault(candidate.target_id, endpoints)

    adjacency: dict[str, set[str]] = {}
    for endpoints in portal_endpoints_by_id.values():
        values = sorted(endpoints)
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)

    # Reverse BFS from all currently observed target rooms.  If no target room
    # is observed yet, local portal hints still provide useful ordering.
    distance_to_target: dict[str, int] = {}
    queue = list(sorted(target_rooms))
    for room in queue:
        distance_to_target[room] = 0
    while queue:
        room = queue.pop(0)
        for neighbor in sorted(adjacency.get(room, set())):
            if neighbor in distance_to_target:
                continue
            distance_to_target[neighbor] = distance_to_target[room] + 1
            queue.append(neighbor)

    scores: dict[str, float] = {}
    terms: dict[str, dict[str, float]] = {}
    hints: dict[str, str] = {}
    for candidate in candidates:
        metadata = candidate.metadata or {}
        behavior_type = _normalized_behavior_type(candidate)
        hint = ""
        topology_bonus = 0.0
        if bool(metadata.get("target_goal")):
            hint = "TARGET_GOAL"
            topology_bonus = 1.25
        elif bool(metadata.get("post_interaction_traversal")):
            # Once a portal interaction succeeds, traversing through its
            # opening is the state-transition continuation.  It must outrank
            # a newly visible but merely plausible container nearby.
            hint = "POST_INTERACTION_TRAVERSE"
            topology_bonus = 1.50
        elif behavior_type == "INTERACT" and bool(metadata.get("target_match")):
            hint = "TARGET_CONTAINER"
            topology_bonus = 0.95
        elif (
            behavior_type == "INTERACT"
            and str(metadata.get("node_type") or "").casefold() == "portal"
        ):
            endpoints = portal_endpoints_by_id.get(
                candidate.target_id
            ) or _portal_endpoints(candidate)
            local = bool(current_rooms & endpoints)
            critical = False
            if local and distance_to_target:
                for room in current_rooms & endpoints:
                    current_distance = distance_to_target.get(room)
                    if current_distance is None:
                        continue
                    for neighbor in endpoints - {room}:
                        if (
                            distance_to_target.get(neighbor, current_distance + 1)
                            < current_distance
                        ):
                            critical = True
                            break
                    if critical:
                        break
            if critical:
                hint = "NEXT_ROUTE_PORTAL"
                topology_bonus = 1.35
            elif local:
                hint = "LOCAL_ROUTE_PORTAL"
                topology_bonus = 0.45
            elif endpoints:
                hint = "REMOTE_PORTAL"
                topology_bonus = 0.15
        elif behavior_type == "EXPLORE":
            room = _candidate_room(
                metadata.get("target_room_id") or metadata.get("room_id")
            )
            if room and room in target_rooms:
                hint = "TARGET_ROOM_FRONTIER"
                topology_bonus = 0.55
        compatibility_reason, compatibility = _semantic_container_compatibility(
            candidate, target_context
        )
        if (
            behavior_type == "INTERACT"
            and str(metadata.get("node_type") or "").casefold() == "container"
        ):
            if compatibility >= 1.0 and not hint:
                hint = "PLAUSIBLE_TARGET_CONTAINER"
                topology_bonus += 0.40
            elif compatibility_reason == "semantic_unknown" and not hint:
                hint = "UNKNOWN_CONTAINER"
        if topology_bonus:
            terms[candidate.candidate_id] = {"topology_priority": topology_bonus}
            scores[candidate.candidate_id] = topology_bonus
        else:
            terms[candidate.candidate_id] = {}
            scores[candidate.candidate_id] = 0.0
        if hint:
            hints[candidate.candidate_id] = hint
    return scores, terms, hints


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


def _expected_visible_unknown_area(candidate: BehaviorCandidate) -> float:
    try:
        return max(
            0.0,
            float(
                (candidate.metadata or {}).get(
                    "expected_visible_unknown_area_m2", 0.0
                )
                or 0.0
            ),
        )
    except (TypeError, ValueError):
        return 0.0


def _known_room_id(value: Any) -> str:
    room_id = _room_node_id(value)
    return "" if room_id in {"", "unknown", "room_unknown"} else room_id


def _history_explore_room_id(history_key: str) -> str:
    parts = str(history_key or "").split(":", 3)
    if len(parts) < 3 or parts[0] != "explore_region":
        return ""
    return _known_room_id(parts[1])


def _history_records_room_visit(history: dict[str, Any]) -> bool:
    for field in ("selection_count", "visit_count"):
        try:
            if float(history.get(field, 0) or 0) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    # The node writes this field together with selection_count.  Keeping this
    # fallback makes the curator compatible with persisted history from older
    # runs that may not contain the counter yet (step zero is still a visit).
    return "last_selected_step" in history


def _visited_explore_room_ids(
    history_by_key: dict[str, dict[str, Any]],
) -> set[str]:
    visited: set[str] = set()
    for history_key, history in history_by_key.items():
        if not isinstance(history, dict) or not _history_records_room_visit(history):
            continue
        room_id = _history_explore_room_id(history_key)
        if room_id:
            visited.add(room_id)
    return visited


def _new_room_frontier_ids(
    candidates: Iterable[BehaviorCandidate],
    graph: dict[str, Any],
    visited_room_ids: set[str],
) -> set[str]:
    """Find eligible frontiers in observed rooms that have not been visited.

    A candidate reaches this helper only after normal curator validation.  The
    explicit false checks below additionally keep a room novelty bonus off a
    candidate whose local reachability metadata became unavailable.  Room
    identity comes from the candidate when present, otherwise the already-used
    observation graph association in :func:`candidate_room_id`; no simulator
    or benchmark ground truth is consulted.
    """
    result: set[str] = set()
    for candidate in candidates:
        if _normalized_behavior_type(candidate) != "EXPLORE":
            continue
        metadata = candidate.metadata or {}
        if metadata.get("room_reachable") is False or any(
            metadata.get(key) is False
            for key in ("reachable", "path_reachable", "approach_reachable")
        ):
            continue
        room_id = _known_room_id(candidate_room_id(candidate, graph))
        if not room_id or room_id in visited_room_ids:
            continue
        robot_room_id = _known_room_id(metadata.get("robot_room_id"))
        if robot_room_id and room_id == robot_room_id:
            continue
        result.add(candidate.candidate_id)
    return result


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
        target_context: dict[str, Any] | None = None,
    ) -> CandidateCurationResult:
        graph = graph or {}
        history_by_key = history_by_key or {}
        accepted, rejected = self.filter_candidates(
            candidates,
            graph=graph,
            history_by_key=history_by_key,
            observation_step=observation_step,
        )
        omitted: dict[str, str] = {}
        if (
            self.config.suppress_semantic_container_mismatch
            and bool((target_context or {}).get("enabled"))
            and accepted
        ):
            incompatible_ids = {
                candidate.candidate_id
                for candidate in accepted
                if _semantic_container_compatibility(candidate, target_context)[1] <= 0.0
            }
            # Keep an incompatible container as a last-resort fallback only
            # when it is literally the sole proposal.  In normal mixed tasks
            # a portal/frontier is available and the noisy candidate is hidden
            # from the model request while remaining in the raw graph.
            if incompatible_ids and any(
                candidate.candidate_id not in incompatible_ids for candidate in accepted
            ):
                accepted = [
                    candidate
                    for candidate in accepted
                    if candidate.candidate_id not in incompatible_ids
                ]
                omitted.update(
                    {
                        candidate_id: "target_container_semantic_mismatch"
                        for candidate_id in incompatible_ids
                    }
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
            candidate.candidate_id: _expected_visible_unknown_area(candidate)
            for candidate in explore_pool
        }
        max_expected_visible_area = max(
            expected_visible_area_by_id.values(), default=0.0
        )
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

        visited_explore_room_ids = _visited_explore_room_ids(history_by_key)
        new_room_frontier_ids = _new_room_frontier_ids(
            pools["EXPLORE"],
            graph,
            visited_explore_room_ids,
        )
        quality_by_id: dict[str, float] = {}
        quality_terms_by_id: dict[str, dict[str, float]] = {}
        for behavior_type, pool in pools.items():
            scores, terms = self._score_pool(
                behavior_type,
                pool,
                history_by_key,
                history_key_by_id,
                new_room_frontier_ids,
            )
            quality_by_id.update(scores)
            quality_terms_by_id.update(terms)
        topology_scores, topology_terms, decision_hint_by_id = _topology_hints(
            accepted,
            graph,
            target_context,
        )
        for candidate in accepted:
            candidate_id = candidate.candidate_id
            if candidate_id not in quality_terms_by_id:
                quality_terms_by_id[candidate_id] = {}
            quality_terms_by_id[candidate_id].update(
                topology_terms.get(candidate_id) or {}
            )
            quality_by_id[candidate_id] = float(quality_by_id.get(candidate_id, 0.0)) + float(
                topology_scores.get(candidate_id, 0.0)
            )
        for candidate_id in sorted(new_room_frontier_ids):
            decision_hint_by_id.setdefault(candidate_id, "NEW_ROOM_FRONTIER")
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
            decision_hint_by_id=decision_hint_by_id,
        )

    def _score_pool(
        self,
        behavior_type: str,
        pool: list[BehaviorCandidate],
        history_by_key: dict[str, dict[str, Any]],
        history_key_by_id: dict[str, str],
        new_room_frontier_ids: set[str],
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
            expected_visible_area_by_id = {
                candidate.candidate_id: _expected_visible_unknown_area(candidate)
                for candidate in pool
            }
            max_expected_visible_area = max(
                expected_visible_area_by_id.values(), default=0.0
            )
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
                    "visible_unknown_area_priority": max(
                        0.0,
                        float(self.config.explore_visible_unknown_area_weight),
                    )
                    * (
                        expected_visible_area_by_id[candidate.candidate_id]
                        / max_expected_visible_area
                        if max_expected_visible_area > 0.0
                        else 0.0
                    ),
                    "unvisited_room_frontier_bonus": max(
                        0.0, float(self.config.explore_new_room_bonus)
                    )
                    if candidate.candidate_id in new_room_frontier_ids
                    else 0.0,
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
