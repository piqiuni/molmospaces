#!/usr/bin/env python3
"""Freeze prediction-blind, public-fact proxy labels for M2 context ablations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


VERSION = "public_context_rubric_v1"
STAGES = {"TARGET_GOAL": 0, "POST_INTERACTION_TRAVERSE": 1, "NEXT_ROUTE_PORTAL": 2}
SUCCEEDED = {"SUCCEEDED", "SUCCESS", "COMPLETED"}
FAILED = {"FAILED", "FAILURE", "ABORTED", "TIMEOUT", "TIMED_OUT"}
SMALL_CONTAINERS = {
    "cabinet", "cupboard", "drawer", "dresser", "nightstand", "refrigerator",
    "fridge", "microwave", "oven", "wardrobe", "chestofdrawers",
}
FOOD_TARGETS = {"apple", "banana", "egg", "lettuce", "potato", "tomato", "bread", "milk"}
ROOM_ALIASES = {"livingroom": "livingroom", "diningroom": "diningroom", "bathroom": "bathroom"}

RUBRIC = """# M2 public-context rubric v1 — prediction-blind context ablation

These are deterministic, agent-reviewed **policy-proxy labels**, not human or
simulator ground truth. The author has seen earlier prompts and public examples;
this is prediction-blind, **not prompt-blind**. Neither current predictions nor
historical M2 answers are read. Candidate pre_score, room_target_affinity,
room_object_reasoning, PLAUSIBLE_TARGET_CONTAINER and target-plausibility flags
are not answer keys and are never consulted.

## Frozen inputs and split

Use the supplied union of full-pool and legacy-pool candidate IDs and public
context. Copy the existing split assignment by source.attempt; never repartition
after inspecting labels or results. A common label set is used for every arm.
No acceptable ID in a legacy arm is a candidate-recall loss, not a reason to
change the label or silently exclude the case. If source attempts are absent
from the fixed split, generation fails. The manifest hashes inputs, split,
generator, rubric and annotations.

## Rules, in priority order

1. Reject only explicit invalid public actions: an open request whose current
   state is already open/completed, or an exact previously successful open
   without a later explicit closed state. A bed, toilet or bathtub cannot be
   searched for inside an observed small household cabinet/fridge/drawer.
   An explicit target-object match is exempt from this containment rule.
2. Read at most the last 30 prior decisions and past room-entry events. Ignore
   entries after the source cutoff step/time; mask an outcome observed only
   after that cutoff. Match candidate_id exactly: aggregate spatial-region
   history is not evidence that this exact action failed. Defer exact failures
   only if a fresh non-forbidden alternative exists. A later pending retry
   supersedes an earlier result; outcomes not known at the request are unknown.
3. Prefer a currently declared executable TARGET_GOAL,
   POST_INTERACTION_TRAVERSE, or NEXT_ROUTE_PORTAL, in that order. These are
   public stage contracts, not verified counterfactual simulator success.
4. Otherwise apply a deliberately limited exploration proxy: join room types
   and confidence (>=0.85) from graph rooms and same-room candidate observations.
   Normalize aliases but do not consult affinity scores or producer priors.
   Explicit target labels supply only these common-sense room families:
   toilet/bathtub→bathroom; bed→bedroom; listed foods→kitchen/diningroom.
   When a compatible high-confidence room exists, defer confidently incompatible
   frontier rooms; unknown room types remain possible. Prefer explicitly
   unentered rooms over known entered rooms, leaving unknown visitation possible.
   Among remaining frontiers accept visible-unknown-area estimates within 10%
   of the maximum. Missing area is unknown, never silently zero, and such
   candidates remain possible. Distance is not treated as a global optimum.
5. Other non-forbidden, non-deferred portals/containers/navigation actions remain
   acceptable alongside the preferred frontier: public partial maps cannot
   establish their relative target-discovery value. There may be many answers.
6. Contradictory high-confidence types for the same room, contradictory room
   visit evidence, or contradictory robot-current-room fields make the case
   unscored unless an explicit stage contract determines the choice. If all
   candidates are acceptable, or none can be justified, mark unscored. A
   single-candidate union is separately reported, never primary discrimination.

## Metrics and limitations

Primary: discriminative_public_rubric_top1_acceptance. Its denominator is fixed
by main_score_eligible across all arms, including model/transport/schema errors
and candidate-recall loss. Report candidate-recall coverage, exposed forbidden
top-1 rate, stage acceptance, single-candidate validity, ambiguous/contradictory
counts and data-quality categories separately. A missing reasonable candidate
is upstream pool loss; a wrong choice with reasonable candidates available is
selection loss. Neither is an episode-success measurement.

This replay contains partial histories and reconstructed public snapshots.
Known geometry does not prove reachability or exact room shape. Visiting a room
does not prove fully searching it. Selection/dispatch does not prove task success.
These labels cannot establish navigation SR, globally optimal actions, or causal
closed-loop improvement. Publish a new rubric version for genuine defects;
never tune frozen labels in response to model predictions.
"""


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _room(value: Any) -> str:
    if value in (None, "", "unknown", "room_unknown"):
        return ""
    text = str(value)
    return text if text.startswith("room_") else f"room_{text}"


def _first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    return next((_number(mapping[key]) for key in keys if _number(mapping.get(key)) is not None), None)


def _past_events(
    events: list[dict[str, Any]], source: dict[str, Any], *, decisions: bool
) -> tuple[list[dict[str, Any]], set[str]]:
    cutoff_step = _first_number(source, ("cutoff_step", "observation_step", "recorder_step", "step"))
    cutoff_time = _first_number(source, ("cutoff_timestamp", "request_timestamp", "timestamp"))
    if cutoff_time is None:
        cutoff_time = _number((source.get("reconstruction") or {}).get("request_cutoff"))
    flags: set[str] = set()
    if events and cutoff_step is None and cutoff_time is None:
        flags.add("history_cutoff_unverifiable")
    past = []
    for raw in events:
        if not isinstance(raw, dict):
            flags.add("malformed_history_record")
            continue
        event = dict(raw)
        step = _first_number(event, ("step", "observation_step", "entry_step"))
        stamp = _first_number(event, ("available_timestamp", "timestamp", "selected_at", "selection_timestamp", "entry_timestamp"))
        if (cutoff_step is not None and step is not None and step > cutoff_step) or (
            cutoff_time is not None and stamp is not None and stamp > cutoff_time
        ):
            flags.add("future_history_rejected")
            continue
        result_step = _first_number(event, ("result_observation_step", "result_step", "outcome_step"))
        result_time = _first_number(event, ("feedback_timestamp", "result_timestamp", "outcome_timestamp"))
        if decisions and (
            (cutoff_step is not None and result_step is not None and result_step > cutoff_step)
            or (cutoff_time is not None and result_time is not None and result_time > cutoff_time)
        ):
            event["result"] = "PENDING"
            event["failure_reason"] = ""
            flags.add("future_outcome_masked")
        past.append(event)
    # The context contract is chronological. Sort known steps stably so a
    # malformed input order cannot turn an older failure into the latest result.
    past.sort(key=lambda event: _first_number(event, ("step", "observation_step", "entry_step")) or 0.0)
    return (past[-30:] if decisions else past), flags


def _target_types(request: dict[str, Any]) -> tuple[set[str], set[str]]:
    target = (request.get("mission") or {}).get("target") or {}
    labels = {_normalized(value) for value in [target.get("name"), *(target.get("labels") or [])] if value}
    plausible_rooms: set[str] = set()
    if labels & {"toilet", "bathtub"}:
        plausible_rooms.add("bathroom")
    if "bed" in labels:
        plausible_rooms.add("bedroom")
    if labels & FOOD_TARGETS:
        plausible_rooms.update({"kitchen", "diningroom"})
    return labels, plausible_rooms


def _room_evidence(
    request: dict[str, Any], candidates: list[dict[str, Any]], visits: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    robot = request.get("robot") or {}
    graph = request.get("graph") or {}
    current = {_room(value) for value in (robot.get("current_room"), graph.get("current_room")) if _room(value)}
    conflicts = ["robot_current_room_conflict"] if len(current) > 1 else []
    visited = {_room(item.get("room_id")) for item in visits if _room(item.get("room_id"))} | current
    # Legacy entered_rooms is a set, not a temporal trajectory, but does prove entry.
    visited.update(_room(value) for value in robot.get("entered_rooms") or [] if _room(value))
    room_ids = {_room(c.get("room_id")) for c in candidates if _room(c.get("room_id"))}
    result: dict[str, dict[str, Any]] = {}
    for room_id in sorted(room_ids):
        nodes = [node for node in graph.get("rooms") or [] if _room(node.get("id")) == room_id]
        peers = [candidate for candidate in candidates if _room(candidate.get("room_id")) == room_id]
        typed = [(node.get("type"), node.get("room_attribute_confidence", node.get("type_confidence"))) for node in nodes]
        typed.extend((peer.get("room_attribute"), peer.get("room_attribute_confidence")) for peer in peers)
        high_types = {
            ROOM_ALIASES.get(_normalized(kind), _normalized(kind))
            for kind, confidence in typed
            if kind and not str(kind).startswith("room_") and _normalized(kind) != "unknown"
            and (_number(confidence) or 0.0) >= 0.85
        }
        states = set()
        if room_id in visited or any(peer.get("room_status") in {"entered_room", "current_room"} for peer in peers):
            states.add("entered")
        if any(peer.get("room_status") == "unentered_new_room" for peer in peers):
            states.add("unentered")
        conflict = len(high_types) > 1 or len(states) > 1
        if conflict:
            conflicts.append(room_id)
        result[room_id] = {
            "high_confidence_types": sorted(high_types),
            "visit_state": next(iter(states)) if len(states) == 1 else "unknown",
            "conflict": conflict,
        }
    return result, conflicts


def annotate(case: dict[str, Any], split: str) -> dict[str, Any]:
    request = case["request"]
    source = case["source"]
    candidates = list(request.get("candidates") or [])
    ids = [str(candidate.get("id") or "") for candidate in candidates]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Missing/duplicate candidate IDs in {case['case_id']}")
    decisions, flags = _past_events(list(request.get("recent_decisions") or []), source, decisions=True)
    visits, visit_flags = _past_events(list((request.get("robot") or {}).get("room_visit_history") or []), source, decisions=False)
    flags.update(visit_flags)
    if not decisions:
        flags.add("no_prior_decisions")
    if not visits:
        flags.add("no_room_visit_history")
    for key in ("initial_xy", "current_xy"):
        if not (request.get("robot") or {}).get(key):
            flags.add(f"missing_robot_{key}")
    latest_by_id = {str(event.get("candidate_id")): event for event in decisions if event.get("candidate_id")}
    target_types, plausible_rooms = _target_types(request)
    room_evidence, conflicts = _room_evidence(request, candidates, visits)
    if conflicts:
        flags.add("conflicting_public_room_evidence")
    forbidden: dict[str, str] = {}
    deferred: dict[str, str] = {}
    exact_history = {}
    for candidate in candidates:
        cid = candidate["id"]
        state = str(candidate.get("state") or "").casefold()
        event = latest_by_id.get(cid) or {}
        result = str(event.get("result") or "").upper()
        if event:
            exact_history[cid] = {"result": result, "step": event.get("step", event.get("observation_step")), "failure_reason": event.get("failure_reason", "")}
        if candidate.get("action") == "open" and (
            state in {"open", "static_open", "completed"}
            or (result in SUCCEEDED and state not in {"closed", "static_closed"})
        ):
            forbidden[cid] = "Already-open state or exact successful open without a newer closed state."
        if (
            candidate.get("subject_type") == "container"
            and not candidate.get("explicit_target_match")
            and target_types & {"toilet", "bed", "bathtub"}
            and _normalized(candidate.get("subject_semantic_type")) in SMALL_CONTAINERS
        ):
            forbidden[cid] = "Large fixture target cannot be searched for inside this small household container."
        if result in FAILED:
            deferred[cid] = "Exact candidate failed in prior decision history; prefer a fresh alternative when available."
    eligible = [candidate for candidate in candidates if candidate["id"] not in forbidden]
    fresh = [candidate for candidate in eligible if candidate["id"] not in deferred]
    if fresh:
        eligible = fresh
    stages = [candidate for candidate in eligible if candidate.get("decision_hint") in STAGES]
    evidence = []
    acceptable: list[str] = []
    rule = "insufficient_public_evidence"
    confidence = "low"
    if stages:
        priority = min(STAGES[candidate["decision_hint"]] for candidate in stages)
        acceptable = [candidate["id"] for candidate in stages if STAGES[candidate["decision_hint"]] == priority]
        rule, confidence = "explicit_stage_progression", "high"
        evidence.append("Declared public stage contract; no current exact failure with a fresh alternative.")
    elif eligible:
        frontiers = [candidate for candidate in eligible if candidate.get("subject_type") == "frontier"]
        compatible = [candidate for candidate in frontiers if plausible_rooms.intersection(room_evidence.get(_room(candidate.get("room_id")), {}).get("high_confidence_types") or [])]
        if compatible:
            frontiers = [candidate for candidate in frontiers if (
                not room_evidence.get(_room(candidate.get("room_id")), {}).get("high_confidence_types")
                or candidate in compatible
            )]
            evidence.append("Compatible high-confidence room exists; retain compatible/unknown rooms, not contradictory types.")
        if any(room_evidence.get(_room(candidate.get("room_id")), {}).get("visit_state") == "unentered" for candidate in frontiers):
            frontiers = [candidate for candidate in frontiers if room_evidence.get(_room(candidate.get("room_id")), {}).get("visit_state") != "entered"]
            evidence.append("An explicitly unentered room is available; retain unentered/unknown visitation.")
        areas = {candidate["id"]: _number(candidate.get("expected_visible_unknown_area_m2")) for candidate in frontiers}
        known_areas = [area for area in areas.values() if area is not None and area >= 0.0]
        maximum = max(known_areas, default=0.0)
        acceptable = [candidate["id"] for candidate in frontiers if areas[candidate["id"]] is None or areas[candidate["id"]] >= maximum * 0.9]
        acceptable.extend(candidate["id"] for candidate in eligible if candidate.get("subject_type") != "frontier")
        rule, confidence = "search_progress_proxy", "medium"
        evidence.append("Visible-unknown area within 10% of maximum; unknown areas and other possible actions remain acceptable.")
        if any(area is None for area in areas.values()):
            flags.add("missing_frontier_visible_area")
    acceptable = sorted(set(acceptable))
    single = len(candidates) == 1
    discriminating = bool(acceptable) and len(acceptable) < len(candidates) and not single
    if conflicts and rule != "explicit_stage_progression":
        discriminating = False
        evidence.append("Conflicting room observations/visit evidence: excluded conservatively.")
    return {
        "case_id": case["case_id"], "split": split, "source_attempt": source["attempt"],
        "annotation_version": VERSION, "annotation_origin": "prediction-blind rule-derived public-fact proxy; not human ground truth",
        "status": "single_candidate" if single else "rule_derived" if discriminating else "unscored",
        "acceptable_top1_ids": acceptable if discriminating or single else [],
        "forbidden_ids": sorted(forbidden), "forbidden_reasons": forbidden,
        "deprioritized_ids": sorted(deferred), "deprioritized_reasons": deferred,
        "candidate_count": len(candidates), "candidate_ids": sorted(ids),
        "single_candidate": single, "main_score_eligible": discriminating,
        "rubric_rule": rule, "confidence": confidence,
        "label_kind": "public_stage_contract" if rule == "explicit_stage_progression" else "policy_rubric_proxy",
        "evidence": evidence, "resolved_room_evidence": room_evidence,
        "public_evidence_conflicts": conflicts, "exact_prior_candidate_outcomes": exact_history,
        "history_records_used": len(decisions), "room_visit_records_used": len(visits),
        "data_quality_flags": sorted(flags),
        "limitations": "No counterfactual simulator outcome or hidden target data; acceptance is not episode SR.",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(inputs: Path, split_path: Path, output_dir: Path) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = split["episode_assignment"]
    cases = []
    for line in inputs.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            # Do not read embedded references, prior model answers or annotations.
            cases.append({key: raw[key] for key in ("case_id", "source", "request")})
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    annotations = [annotate(case, assignments[case["source"]["attempt"]]) for case in cases]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in ("annotations.jsonl", "rubric.md", "manifest.json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Frozen label files already exist; choose a new output directory/version")
    paths["annotations.jsonl"].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in annotations), encoding="utf-8")
    paths["rubric.md"].write_text(RUBRIC, encoding="utf-8")
    manifest = {
        "schema_version": "m2_context_label_manifest_v1", "annotation_version": VERSION,
        "inputs_sha256": _sha(inputs), "split_sha256": _sha(split_path),
        "annotations_sha256": _sha(paths["annotations.jsonl"]), "rubric_sha256": _sha(paths["rubric.md"]),
        "generator_sha256": _sha(Path(__file__)), "case_count": len(cases),
        "episode_count": len({case["source"]["attempt"] for case in cases}),
        "by_split": {name: {
            "cases": sum(row["split"] == name for row in annotations),
            "main_score_eligible": sum(row["split"] == name and row["main_score_eligible"] for row in annotations),
            "statuses": dict(Counter(row["status"] for row in annotations if row["split"] == name)),
            "rules": dict(Counter(row["rubric_rule"] for row in annotations if row["split"] == name)),
        } for name in ("dev", "holdout")},
        "data_quality_counts": dict(Counter(flag for row in annotations for flag in row["data_quality_flags"])),
        "blindness": {"prediction_blind": True, "prompt_blind": False, "historical_response_used": False, "pre_score_or_affinity_used": False},
        "metric_name": "discriminative_public_rubric_top1_acceptance",
        "candidate_recall_policy": "Common union labels; missing every acceptable ID counts as upstream recall loss, not denominator exclusion.",
        "not_claimed": ["human-ground-truth accuracy", "navigation success rate", "causal closed-loop improvement"],
    }
    paths["manifest.json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.inputs, args.split, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
