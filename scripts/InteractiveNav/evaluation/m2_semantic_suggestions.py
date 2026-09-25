"""Opt-in, observation-only context suggestions for offline M2 experiments.

These helpers do not rank, filter or rewrite candidates.  Historical execution
outcomes are evidence, not authoritative state for a newly offered action.
Only the already frozen recent-decision window is used for historical facts.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any, Mapping


STAGE_HINTS = frozenset({
    "POST_INTERACTION_TRAVERSE", "POST_INTERACTION_SCAN", "NEXT_ROUTE_PORTAL",
    "TARGET_GOAL", "TARGET_VISIBLE", "CONTAINER_STAGE_OBSERVE",
})
_IN_PROGRESS = frozenset({"STARTED", "RUNNING", "PENDING", "ACCEPTED", "IN_PROGRESS"})
_SUCCEEDED = frozenset({"SUCCEEDED", "SUCCESS", "COMPLETED"})
_FAILED = frozenset({"FAILED", "FAILURE", "ABORTED", "REJECTED", "TIMED_OUT", "TIMEOUT"})
_EVENT_KEYS = ("source_interaction_event_id", "interaction_event_id")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _xy(value: Any) -> list[float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    point = [_number(item) for item in value[:2]]
    return point if all(item is not None for item in point) else None


def _frame(value: Any) -> str | None:
    return str(value).lstrip("/") if value else None


def _result_class(entry: Mapping[str, Any]) -> str:
    status = str(entry.get("result") or entry.get("status") or "UNKNOWN").upper()
    if status in _IN_PROGRESS:
        return "in_progress"
    if status in _SUCCEEDED:
        return "succeeded"
    if status in _FAILED:
        return "failed"
    if status in {"CANCELED", "CANCELLED", "PREEMPTED"}:
        return "cancelled"
    return "unknown"


def _event_id(value: Mapping[str, Any]) -> str | None:
    metadata = value.get("metadata") or {}
    for key in _EVENT_KEYS:
        found = value.get(key) or metadata.get(key)
        if found not in (None, ""):
            return str(found)
    # decision_id, candidate_id and group_id are not operation-event identity.
    return None


def _history(request: Mapping[str, Any], raw_record: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    cutoff = _number(((raw_record or {}).get("reconstruction") or {}).get("request_cutoff"))
    seen: dict[tuple[str, Any], dict[str, Any]] = {}
    for index, raw in enumerate(request.get("recent_decisions") or []):
        if not isinstance(raw, Mapping):
            continue
        entry = dict(raw)
        if cutoff is not None:
            stamps = [_number(entry.get(key)) for key in ("selected_at", "feedback_timestamp")]
            if any(stamp is not None and stamp > cutoff for stamp in stamps):
                continue
        decision_id = entry.get("decision_id")
        identity = ("decision_id", str(decision_id)) if decision_id else ("window_index", index)
        seen[identity] = entry
    return list(seen.values())


def _raw_candidates(raw_record: Mapping[str, Any] | None) -> tuple[dict[str, dict[str, Any]], str]:
    if not raw_record:
        return {}, "raw_snapshot_not_supplied"
    reconstruction = raw_record.get("reconstruction") or {}
    cutoff = _number(reconstruction.get("request_cutoff"))
    stamp = _number(reconstruction.get("semantic_timestamp"))
    if cutoff is None or stamp is None or stamp > cutoff:
        return {}, "raw_snapshot_timestamp_not_verified"
    if reconstruction.get("candidate_exact_sequence_and_revision") is not True:
        return {}, "raw_candidate_snapshot_not_exact"
    return {
        str(item["candidate_id"]): dict(item)
        for item in raw_record.get("raw_candidates") or []
        if isinstance(item, Mapping) and item.get("candidate_id")
    }, "exact_candidate_snapshot_before_request_cutoff"


def _history_reference(entry: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    if entry.get("decision_id"):
        return {"decision_id": entry["decision_id"]}
    indices = [index for index, raw in enumerate(request.get("recent_decisions") or []) if raw == entry]
    return {"recent_decisions_index": indices[-1]}


def _compact_action(entry: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        **_history_reference(entry, request),
        "step": entry.get("step", entry.get("observation_step")),
        "result": entry.get("result", entry.get("status", "UNKNOWN")),
        "result_class": _result_class(entry),
    }
    if _event_id(entry):
        result["source_interaction_event_id"] = _event_id(entry)
    return result


def build_stage_facts(
    request: Mapping[str, Any], raw_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize stage evidence without inferring pending/completed state."""
    history = _history(request, raw_record)
    raw_by_id, raw_source = _raw_candidates(raw_record)
    current_stages = []
    conflicts = []
    for candidate in request.get("candidates") or []:
        hint = str(candidate.get("decision_hint") or "")
        if hint not in STAGE_HINTS:
            continue
        candidate_id = str(candidate.get("id") or "")
        target_id = str(candidate.get("subject_id") or "")
        raw = raw_by_id.get(candidate_id, {})
        current_event = _event_id(candidate) or _event_id(raw)
        related = []
        for entry in history:
            same_candidate = str(entry.get("candidate_id") or "") == candidate_id
            same_target = bool(target_id) and str(entry.get("target_id") or "") == target_id
            if not same_candidate and not same_target:
                continue
            historical_event = _event_id(entry)
            identity = (
                "same_event" if current_event == historical_event else "different_event"
            ) if current_event and historical_event else "unknown"
            fact = _compact_action(entry, request)
            fact.update(related_by="same_candidate_id" if same_candidate else "same_target_id", event_identity_match=identity)
            related.append(fact)
            if same_candidate and _result_class(entry) == "succeeded":
                conflicts.append({
                    "candidate_id": candidate_id,
                    "history_decision_id": entry.get("decision_id"),
                    "kind": "prior_same_candidate_success_with_current_stage_hint",
                    "event_identity_match": identity,
                    "interpretation": (
                        "same_event_success_and_current_hint_require_reconciliation"
                        if identity == "same_event" else
                        "different_events_do_not_establish_repeated_completion"
                        if identity == "different_event" else
                        "unknown_event_identity_do_not_infer_current_completion"
                    ),
                })
        current_stages.append({
            "candidate_id": candidate_id,
            "current_hint": hint,
            "target_id": target_id or None,
            "room_id": candidate.get("room_id"),
            "source_interaction_event_id": current_event,
            "current_event_identity": "recorded" if current_event else "unknown",
            "completion_state": "not_inferred",
            "related_history": related,
        })
    return {
        "schema_version": "m2_stage_facts_v2",
        "history_scope": "frozen_recent_decisions_only",
        "history_references": "decision_id or zero-based recent_decisions_index; full action/target/reason remain in recent_decisions",
        "history_decision_count": len(history),
        "history_result_counts": dict(Counter(_result_class(entry) for entry in history)),
        "raw_candidate_source": raw_source,
        "current_stage_candidates": current_stages,
        "recent_interaction_results": [
            _compact_action(entry, request) for entry in history
            if str(entry.get("behavior_type") or "").upper() in {"INTERACT", "SCAN"}
        ],
        "prior_success_current_hint_conflicts": conflicts,
        "interpretation_limits": [
            "A current stage hint is not an authoritative pending flag.",
            "A prior success for the same candidate or target does not prove the current event completed.",
            "Missing event identity remains unknown; STARTED is not failure.",
        ],
    }


def _history_goal(entry: Mapping[str, Any]) -> tuple[list[float] | None, str | None]:
    metadata = entry.get("metadata") or {}
    return _xy(entry.get("goal_xy")), _frame(
        entry.get("goal_frame_id") or metadata.get("goal_frame_id")
        or entry.get("frame_id") or metadata.get("frame_id")
    )


def _current_goal(
    candidate: Mapping[str, Any], raw: Mapping[str, Any],
) -> tuple[list[float] | None, str | None, str]:
    point = _xy(candidate.get("goal_xy"))
    frame = _frame(candidate.get("goal_frame_id") or candidate.get("frame_id"))
    if point is not None:
        return point, frame, "frozen_candidate_goal_xy"
    metadata = raw.get("metadata") or {}
    return _xy(raw.get("goal_xyyaw")), _frame(
        raw.get("goal_frame_id") or metadata.get("goal_frame_id") or metadata.get("frame_id")
    ), "timestamp_bounded_raw_candidate_goal_xyyaw" if raw else "not_recorded"


def _explicit_region(candidate: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[str | None, str | None]:
    for source_name, source in (("candidate", candidate), ("raw_candidate.metadata", raw.get("metadata") or {})):
        for key in ("history_key", "region_key", "region_id"):
            if source.get(key) not in (None, ""):
                return str(source[key]), source_name + "." + key
    return None, None


def build_region_history_facts(
    request: Mapping[str, Any], raw_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Group recorded region keys; never substitute a room for region identity."""
    history = _history(request, raw_record)
    raw_by_id, raw_source = _raw_candidates(raw_record)
    grouped: dict[str, list[dict[str, Any]]] = {}
    unassigned = 0
    for entry in history:
        key = str(entry.get("history_key") or "")
        if key.startswith("explore_region:"):
            grouped.setdefault(key, []).append(entry)
        elif str(entry.get("behavior_type") or "").upper() == "EXPLORE":
            unassigned += 1
    regions = []
    for key, entries in grouped.items():
        counts = Counter(_result_class(entry) for entry in entries)
        gains = []
        for entry in entries:
            measurements = {
                name: entry[name] for name in ("frontier_shrink_m", "frontier_length_delta_m", "exploration_gain")
                if name in entry and _number(entry[name]) is not None
            }
            if measurements:
                gains.append({**_history_reference(entry, request), **measurements})
        last = entries[-1]
        regions.append({
            "region_key": key,
            "selection_count": len(entries),
            "successful_execution_count": counts["succeeded"],
            "failed_execution_count": counts["failed"],
            "in_progress_count": counts["in_progress"],
            "cancelled_count": counts["cancelled"],
            "unknown_result_count": counts["unknown"],
            "last_result": last.get("result", last.get("status", "UNKNOWN")),
            "last_step": last.get("step", last.get("observation_step")),
            "history_refs": [_history_reference(entry, request) for entry in entries],
            "gain_measurements": gains if gains else None,
        })
    associations = []
    candidate_count = missing_region_count = missing_geometry_count = 0
    for candidate in request.get("candidates") or []:
        if candidate.get("action") != "explore" and candidate.get("subject_type") != "frontier":
            continue
        candidate_count += 1
        candidate_id = str(candidate.get("id") or "")
        raw = raw_by_id.get(candidate_id, {})
        region, region_source = _explicit_region(candidate, raw)
        point, frame, coordinate_source = _current_goal(candidate, raw)
        distances = []
        for key, entries in grouped.items():
            comparable = []
            for entry in entries:
                historical_point, historical_frame = _history_goal(entry)
                if point is None or not frame or frame != historical_frame or historical_point is None:
                    continue
                distance = math.dist(point, historical_point)
                comparable.append((distance, entry))
            if comparable:
                distance, closest = min(comparable, key=lambda pair: pair[0])
                distances.append({
                    "historical_region_key": key,
                    "nearest_recorded_goal_distance_m": round(distance, 4),
                    "same_recorded_goal_xy": distance <= 1e-6,
                    "historical_decision_id": closest.get("decision_id"),
                    "coordinate_frame_id": frame,
                    "region_equivalence_inferred": False,
                })
        missing_region_count += not bool(region)
        missing_geometry_count += not bool(distances)
        if not region and not distances:
            continue
        association = {"candidate_id": candidate_id}
        if region:
            association.update(
                explicit_region_key=region, region_identity_source=region_source,
                matching_historical_region_key=region if region in grouped else None,
            )
        if distances:
            association.update(
                current_goal_xy=point, current_goal_frame_id=frame,
                coordinate_source=coordinate_source,
                distances_to_historical_regions=distances,
            )
        associations.append(association)
    return {
        "schema_version": "m2_region_history_facts_v2",
        "history_scope": "frozen_recent_decisions_only",
        "history_references": "decision_id or zero-based recent_decisions_index; ordered results/goals/frames/rooms remain in recent_decisions",
        "history_decision_count": len(history),
        "raw_candidate_source": raw_source,
        "regions": regions,
        "exploration_decisions_without_region_key": unassigned,
        "current_candidate_region_facts": associations,
        "missing_information": {
            "exploration_candidate_count": candidate_count,
            "candidates_without_explicit_region": missing_region_count,
            "candidates_without_comparable_same_frame_history": missing_geometry_count,
            "omitted_candidate_fields": "unknown, not negative evidence; only additional known associations are listed",
            "physical_region_visit_counts": "not_observed",
        },
        "interpretation_limits": [
            "Selection and successful execution counts are not physical region visit counts.",
            "Same room or nearby coordinates do not prove the same region.",
            "No region size or region key is inferred from coordinates.",
            "Distances require explicit matching current-goal and historical-goal frames.",
            "Missing gain observations remain unknown, not zero.",
        ],
    }


def enrich_request(
    request: Mapping[str, Any], raw_record: Mapping[str, Any] | None = None,
    *, stage: bool = False, regions: bool = False,
) -> dict[str, Any]:
    """Add experiment-only robot fields; preserve every pre-existing field."""
    result = deepcopy(dict(request))
    robot = result.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("request.robot must be an object")
    additions = {}
    if stage:
        additions["stage_facts"] = build_stage_facts(request, raw_record)
    if regions:
        additions["region_history_facts"] = build_region_history_facts(request, raw_record)
    conflict = set(additions).intersection(robot)
    if conflict:
        raise ValueError("Refusing to overwrite existing suggestion fields: " + ", ".join(sorted(conflict)))
    robot.update(additions)
    return result
