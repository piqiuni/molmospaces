#!/usr/bin/env python3
"""Freeze secondary public-policy checks without reading any model predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.remove(str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.InteractiveNav.evaluation import m2_context_labels as historical


VERSION = "m2_semantic_public_checks_v1"
CHECK_NAMES = (
    "semantic_room", "semantic_container", "semantic_support",
    "initial_room_proximity", "new_region", "exact_failed_repeat",
    "related_unchecked_container", "stage_contract_conflict", "top3_diversity",
)

# These are explicit soft search priors, not claims of actual target locations.
PRIORS = {
    "egg": ({"kitchen", "diningroom"}, {"fridge", "refrigerator"}, {"countertop", "kitchencounter", "table", "diningtable"}),
    "lettuce": ({"kitchen", "diningroom"}, {"fridge", "refrigerator"}, {"countertop", "kitchencounter", "table", "diningtable"}),
    "apple": ({"kitchen", "diningroom"}, {"fridge", "refrigerator"}, {"countertop", "kitchencounter", "table", "diningtable"}),
    "tomato": ({"kitchen", "diningroom"}, {"fridge", "refrigerator"}, {"countertop", "kitchencounter", "table", "diningtable"}),
    "potato": ({"kitchen", "diningroom"}, set(), {"countertop", "kitchencounter", "table", "diningtable"}),
    "bowl": ({"kitchen", "diningroom"}, {"kitchencabinet"}, {"countertop", "kitchencounter", "table", "diningtable"}),
    "book": ({"bedroom", "livingroom", "study", "office"}, {"bookcase", "bookshelf"}, {"bookcase", "bookshelf", "desk"}),
    "cd": ({"livingroom", "bedroom", "study", "office"}, {"mediacabinet"}, {"tvstand", "mediacabinet", "desk"}),
    "alarmclock": ({"bedroom", "livingroom"}, set(), {"nightstand", "bedsidetable", "desk"}),
    "toilet": ({"bathroom"}, set(), set()),
    "bed": ({"bedroom"}, set(), set()),
    "bathtub": ({"bathroom"}, set(), set()),
}
ROOM_UNLIKELY = {
    "egg": {"bedroom", "bathroom"}, "lettuce": {"bedroom", "bathroom"},
    "apple": {"bedroom", "bathroom"}, "tomato": {"bedroom", "bathroom"},
    "potato": {"bedroom", "bathroom"}, "bowl": {"bedroom", "bathroom"},
    "book": {"bathroom"}, "cd": {"bathroom", "kitchen"},
    "alarmclock": {"bathroom", "kitchen"}, "toilet": {"kitchen", "bedroom", "livingroom", "diningroom"},
    "bed": {"kitchen", "bathroom", "diningroom"}, "bathtub": {"kitchen", "bedroom", "livingroom", "diningroom"},
}
UNRELATED_STORAGE = {"wardrobe", "dresser", "chestofdrawers"}
SMALL_STORAGE = historical.SMALL_CONTAINERS | {"refrigerator", "kitchencabinet", "mediacabinet"}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _number(value: Any) -> float | None:
    number = historical._number(value)
    return number if number is not None and math.isfinite(number) else None


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    pair = (_number(value[0]), _number(value[1]))
    return pair if None not in pair else None


def _close(left: Any, right: Any, tolerance: float = 0.1) -> bool:
    a, b = _number(left), _number(right)
    return a is not None and b is not None and min(a, b) >= 0 and abs(a - b) <= max(a, b, 0.01) * tolerance


def _target(request: dict[str, Any]) -> str | None:
    target = (request.get("mission") or {}).get("target") or {}
    words = {_norm(value) for value in [target.get("name"), *(target.get("labels") or [])]}
    known = sorted(words & set(PRIORS))
    return known[0] if len(known) == 1 else None


def _subject(candidate: dict[str, Any]) -> str:
    return _norm(candidate.get("subject_semantic_type") or candidate.get("subject_name"))


def _check(name: str, mode: str = "pairwise") -> dict[str, Any]:
    return {"name": name, "mode": mode, "applicable": False, "pairs": [],
            "preferred_ids": [], "disfavored_ids": [], "reason": "insufficient_matched_public_alternatives"}


def _pair(check: dict[str, Any], preferred: dict[str, Any], other: dict[str, Any], evidence: str) -> None:
    if preferred["id"] == other["id"]:
        return
    item = {"preferred_id": preferred["id"], "other_id": other["id"], "evidence": evidence}
    if item not in check["pairs"]:
        check["pairs"].append(item)
    check.update(applicable=True, reason="predeclared_soft_policy_preference")


def build_checks(case: dict[str, Any], *, synthetic: bool = False) -> dict[str, Any]:
    request, source = case["request"], case.get("source") or {}
    candidates = request.get("candidates") or []
    ids = [candidate.get("id") for candidate in candidates]
    if not ids or any(not isinstance(cid, str) or not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Current candidate IDs must be nonempty and unique")
    checks = {name: _check(name) for name in CHECK_NAMES}
    decisions, flags = historical._past_events(list(request.get("recent_decisions") or []), source, decisions=True)
    visits, visit_flags = historical._past_events(list((request.get("robot") or {}).get("room_visit_history") or []), source, decisions=False)
    flags |= visit_flags
    if "history_cutoff_unverifiable" in flags:
        decisions, visits = [], []
    latest = {str(event["candidate_id"]): event for event in decisions if event.get("candidate_id")}
    room_info, conflicts = historical._room_evidence(request, candidates, visits)
    target = _target(request)
    room_priors, container_priors, support_priors = PRIORS.get(target, (set(), set(), set()))
    room_nodes = {historical._room(room.get("id")): room for room in (request.get("graph") or {}).get("rooms") or []}

    def room(candidate: dict[str, Any]) -> dict[str, Any]:
        return room_info.get(historical._room(candidate.get("room_id")), {})

    def semantic_class(candidate: dict[str, Any]) -> str:
        kinds = set(room(candidate).get("high_confidence_types") or [])
        if not kinds or not target:
            return "unknown"
        if kinds & room_priors:
            return "compatible"
        return "less_likely" if kinds & ROOM_UNLIKELY.get(target, set()) else "unknown"

    def result(candidate: dict[str, Any]) -> str:
        return str((latest.get(candidate["id"]) or {}).get("result") or "").upper()

    def completed_open(candidate: dict[str, Any]) -> bool:
        state = str(candidate.get("state") or "").casefold()
        return candidate.get("action") == "open" and (state in {"open", "static_open", "completed"} or (
            result(candidate) in historical.SUCCEEDED and state not in {"closed", "static_closed"}))

    failed = [candidate for candidate in candidates if result(candidate) in historical.FAILED]
    fresh = [candidate for candidate in candidates if candidate not in failed and not completed_open(candidate)]
    for preferred in fresh:
        for other in failed:
            _pair(checks["exact_failed_repeat"], preferred, other, "Exact ID failed before cutoff; a current fresh alternative exists. Not a room/region failure.")
    stage = [candidate for candidate in candidates if candidate.get("decision_hint") in historical.STAGES]
    stage_conflicts = [candidate for candidate in stage if result(candidate) in historical.SUCCEEDED | historical.FAILED]
    checks["stage_contract_conflict"].update(
        mode="diagnostic", applicable=bool(stage), reason="declared_phase_is_not_physical_success",
        stage_ids=[candidate["id"] for candidate in stage],
        conflict_ids=[candidate["id"] for candidate in stage_conflicts],
        warning="Past success/failure and a current stage hint coexist; dispatch/feedback does not prove physical traversal or stage completion.",
    )
    frontiers = [candidate for candidate in fresh if candidate.get("subject_type") == "frontier"]
    search_checks = [name for name in CHECK_NAMES if name not in {"exact_failed_repeat", "stage_contract_conflict"}]
    if stage or conflicts:
        for name in search_checks:
            checks[name]["reason"] = "suppressed_by_stage_contract_or_conflicting_room_evidence"
    else:
        for preferred in frontiers:
            for other in frontiers:
                if semantic_class(preferred) == "compatible" and semantic_class(other) == "less_likely":
                    _pair(checks["semantic_room"], preferred, other, "Explicit target prior and >=0.85 room-type evidence; soft room-search preference, not hidden location.")
                same_class = semantic_class(preferred) == semantic_class(other) and semantic_class(preferred) != "less_likely"
                pstate, ostate = room(preferred).get("visit_state"), room(other).get("visit_state")
                gain_match = _close(preferred.get("expected_visible_unknown_area_m2"), other.get("expected_visible_unknown_area_m2"))
                if same_class and gain_match and pstate == "unentered" and ostate == "entered":
                    _pair(checks["new_region"], preferred, other, "Same semantic class and visible-gain estimates within 10%; explicit unentered vs entered. Entry is not complete search.")
                ptypes = room(preferred).get("high_confidence_types") or []
                otypes = room(other).get("high_confidence_types") or []
                graph_frame = (request.get("graph") or {}).get("frame_id")
                robot_frame = (request.get("robot") or {}).get("position_frame_id")
                geometry_match = (graph_frame and graph_frame == robot_frame and same_class and ptypes == otypes and ptypes and pstate == ostate and pstate in {"entered", "unentered"}
                                  and gain_match and _close(preferred.get("unknown_component_area_m2"), other.get("unknown_component_area_m2"))
                                  and _close(preferred.get("distance_m"), other.get("distance_m"), 0.1))
                initial = _xy((request.get("robot") or {}).get("initial_xy"))
                pn, on = room_nodes.get(historical._room(preferred.get("room_id")), {}), room_nodes.get(historical._room(other.get("room_id")), {})
                pxy, oxy = _xy(pn.get("aabb_center_xy") or pn.get("centroid_xy")), _xy(on.get("aabb_center_xy") or on.get("centroid_xy"))
                if geometry_match and initial and pxy and oxy and math.dist(initial, pxy) + 1.0 <= math.dist(initial, oxy):
                    _pair(checks["initial_room_proximity"], preferred, other, "Matched room type/visit state, visible+component gain and current distance within 10%; initial-point-to-room-center distance at least 1m smaller. Not path distance or proven reset spawn.")
                def support_nodes(candidate: dict[str, Any]) -> set[str]:
                    return {_norm(node.get("type")) for node in candidate.get("nearby_semantic_nodes") or []
                            if _number(node.get("distance_m")) is not None and 0 <= float(node["distance_m"]) <= 2.0}
                if (support_priors and historical._room(preferred.get("room_id")) == historical._room(other.get("room_id"))
                    and gain_match and _close(preferred.get("distance_m"), other.get("distance_m"), 0.1)
                    and support_nodes(preferred) & support_priors and support_nodes(other)
                    and not support_nodes(other) & support_priors):
                    _pair(checks["semantic_support"], preferred, other, "Matched same-room frontiers, visible gain/current distance; nearby observed plausible support within 2m vs other observed objects. Proximity does not prove containment, visibility or target presence.")
        containers = [candidate for candidate in fresh if candidate.get("subject_type") == "container" and candidate.get("action") == "open"]
        if target in {"toilet", "bed", "bathtub"}:
            for preferred in frontiers:
                for other in containers:
                    if _subject(other) in SMALL_STORAGE and not other.get("explicit_target_match"):
                        _pair(checks["semantic_container"], preferred, other, "Large fixed target is not inside an observed small storage container; useful frontier alternative exists.")
        elif container_priors:
            related = [candidate for candidate in containers if _subject(candidate) in container_priors and str(candidate.get("state")).casefold() in {"closed", "static_closed"}]
            unrelated = [candidate for candidate in containers if _subject(candidate) in UNRELATED_STORAGE]
            for preferred in related:
                for other in unrelated:
                    _pair(checks["semantic_container"], preferred, other, "Predeclared plausible target-container prior vs unrelated clothing storage; soft preference only.")
                    if result(preferred) not in historical.SUCCEEDED:
                        _pair(checks["related_unchecked_container"], preferred, other, "Related currently closed container, no exact observed successful inspection in last30; absence of completion is NOT proof never checked.")
            for preferred in related:
                if result(preferred) in historical.SUCCEEDED:
                    continue
                for other in frontiers:
                    if semantic_class(other) == "less_likely" and room(other).get("visit_state") == "entered":
                        _pair(checks["related_unchecked_container"], preferred, other, "Related closed container with no observed completed inspection vs already-entered explicitly less-likely frontier; not a blanket container-over-frontier rule.")
        useful = [candidate for candidate in fresh if semantic_class(candidate) != "less_likely" and not (
            target in {"toilet", "bed", "bathtub"} and candidate.get("subject_type") == "container" and _subject(candidate) in SMALL_STORAGE)]
        groups = {candidate["id"]: f"{candidate.get('subject_type', 'unknown')}:{historical._room(candidate.get('room_id')) or 'unknown'}" for candidate in useful}
        checks["top3_diversity"].update(mode="diversity", applicable=len(set(groups.values())) >= 2,
            reason="two_distinct_public_action_family_room_groups_available" if len(set(groups.values())) >= 2 else "insufficient_diverse_public_alternatives",
            candidate_groups=groups, required_groups=2,
            warning="Group diversity is a fallback policy proxy, not independent reachability or conditional execution validity.")
    for check in checks.values():
        check["preferred_ids"] = sorted({pair["preferred_id"] for pair in check["pairs"]})
        check["disfavored_ids"] = sorted({pair["other_id"] for pair in check["pairs"]})
    return {"schema_version": VERSION, "case_id": case["case_id"], "dataset_kind": "synthetic" if synthetic else "historical",
            "target_prior_key": target, "candidate_ids": ids, "checks": list(checks.values()),
            "data_quality_flags": sorted(flags), "room_evidence_conflicts": conflicts,
            "old_primary_labels_unchanged": True}


def score_checks(check_row: dict[str, Any], response: dict[str, Any] | None) -> dict[str, Any]:
    """Score secondary preferences; always publish unknown/coverage, never blend old labels."""
    ranked = response.get("ranked_ids") if isinstance(response, dict) else None
    valid = isinstance(ranked, list) and 1 <= len(ranked) <= 3 and all(isinstance(cid, str) for cid in ranked)
    valid = bool(valid and len(set(ranked)) == len(ranked) and set(ranked) <= set(check_row["candidate_ids"]))
    rank = {cid: index for index, cid in enumerate(ranked)} if valid else {}
    outputs = []
    for check in check_row["checks"]:
        out = {"name": check["name"], "applicable": check["applicable"], "status": "not_applicable"}
        if check["applicable"] and check["mode"] == "diagnostic":
            out.update(status="diagnostic_only", conflict_present=bool(check["conflict_ids"]),
                       selected_conflicting_hint=bool(valid and ranked[0] in check["conflict_ids"]))
        elif check["applicable"] and not valid:
            out.update(status="fail", reason="missing_or_invalid_ranked_ids")
        elif check["applicable"] and check["mode"] == "diversity":
            selected = {check["candidate_groups"][cid] for cid in ranked if cid in check["candidate_groups"]}
            out.update(status="pass" if len(selected) >= check["required_groups"] else "fail", selected_group_count=len(selected))
        elif check["applicable"]:
            counts = Counter()
            for pair in check["pairs"]:
                preferred, other = rank.get(pair["preferred_id"]), rank.get(pair["other_id"])
                if preferred is None and other is None:
                    counts["unknown"] += 1
                elif preferred is not None and (other is None or preferred < other):
                    counts["pass"] += 1
                else:
                    counts["fail"] += 1
            out.update(status="fail" if counts["fail"] else "pass" if counts["pass"] else "unknown",
                       pair_counts={key: counts[key] for key in ("pass", "fail", "unknown")})
        outputs.append(out)
    return {"case_id": check_row["case_id"], "ranked_ids_valid": valid, "checks": outputs}


def _probe(name: str, target: str = "egg") -> dict[str, Any]:
    def room(rid: int, x: float, kind: str) -> dict[str, Any]:
        return {"id": f"room_{rid}", "type": kind, "room_attribute_confidence": 0.95,
                "centroid_xy": [x, 0], "aabb_center_xy": [x, 0], "aabb_size_xy": [2, 2]}
    candidates = [{"id": f"frontier:{i}:{i}", "action": "explore", "subject_id": f"{i}:{i}", "subject_type": "frontier",
                   "effect": "reveal_space", "room_id": f"room_{i}", "room_status": "unentered_new_room", "distance_m": 3.0,
                   "expected_visible_unknown_area_m2": 4.0, "unknown_component_area_m2": 8.0} for i in (1, 2)]
    return {"schema_version": "interactive_nav_m2_replay_case_v1", "case_id": f"synthetic::{name}",
            "source": {"attempt": f"synthetic/{name}", "step": 10, "synthetic": True},
            "request": {"schema_version": 5, "instruction": "", "mission": {"mode": "object_goal", "target": {"name": target, "labels": [target], "visible": False}},
                        "robot": {"initial_xy": [0, 0], "current_xy": [6, 0], "position_frame_id": "tf_frame_map", "current_room": "room_0",
                                  "room_visit_history": [{"room_id": "room_3", "entry_step": 0}, {"room_id": "room_0", "entry_step": 2}]},
                        "graph": {"frame_id": "tf_frame_map", "current_room": "room_0", "rooms": [room(0, 6, "hallway"), room(1, 3, "kitchen"), room(2, 9, "kitchen"), room(3, 0, "hallway")], "portals": [], "containers": []},
                        "room_object_reasoning": {}, "recent_decisions": [], "candidates": candidates}}


def synthetic_probes(instruction: str) -> list[dict[str, Any]]:
    """Nine matched positive/counterexample pairs; never enter historical166 denominator."""
    probes = []
    for counter in (False, True):
        suffix = "counter" if counter else "positive"
        p = _probe(f"semantic_room_{suffix}", "toilet" if counter else "egg")
        p["request"]["graph"]["rooms"][2]["type"] = "bathroom" if counter else "bedroom"
        probes.append(p)
        p = _probe(f"semantic_container_{suffix}", "toilet" if counter else "egg")
        c = p["request"]["candidates"]
        c[0] = {"id": "interaction:fridge:open", "action": "open", "subject_id": "fridge", "subject_type": "container", "subject_semantic_type": "fridge", "effect": "reveal_contents", "state": "closed", "room_id": "room_1", "distance_m": 3.0}
        if not counter:
            c[1] = {**c[0], "id": "interaction:wardrobe:open", "subject_id": "wardrobe", "subject_semantic_type": "wardrobe"}
        probes.append(p)
        p = _probe(f"semantic_support_{suffix}", "book" if counter else "alarmclock")
        for c in p["request"]["candidates"]:
            c["room_id"] = "room_1"
        p["request"]["graph"]["rooms"][1]["type"] = "bedroom"
        p["request"]["candidates"][0]["nearby_semantic_nodes"] = [{"type": "nightstand", "distance_m": 0.5}]
        p["request"]["candidates"][1]["nearby_semantic_nodes"] = [{"type": "bookshelf", "distance_m": 0.5}]
        probes.append(p)
        p = _probe(f"initial_room_proximity_{suffix}")
        if counter:
            p["request"]["graph"]["rooms"][1]["type"] = "bathroom"
        probes.append(p)
        p = _probe(f"new_region_{suffix}")
        p["request"]["candidates"][1]["room_status"] = "entered_room"
        p["request"]["robot"]["room_visit_history"].insert(1, {"room_id": "room_2", "entry_step": 1})
        if counter:
            p["request"]["graph"]["rooms"][1]["type"] = "bathroom"
        probes.append(p)
        p = _probe(f"exact_failed_repeat_{suffix}")
        p["request"]["recent_decisions"] = [{"candidate_id": "frontier:1:old" if counter else "frontier:1:1", "step": 3, "result_step": 4, "result": "FAILED", "target_room_id": "room_1", "history_key": "same_region", "failure_reason": "make_plan_unreachable"}]
        probes.append(p)
        p = _probe(f"related_unchecked_container_{suffix}")
        p["request"]["candidates"][0] = {"id": "interaction:fridge:open", "action": "open", "subject_id": "fridge", "subject_type": "container", "subject_semantic_type": "fridge", "effect": "reveal_contents", "state": "open" if counter else "closed", "room_id": "room_1", "distance_m": 3.0}
        p["request"]["graph"]["rooms"][2]["type"] = "bedroom"
        p["request"]["candidates"][1]["room_status"] = "entered_room"
        p["request"]["robot"]["room_visit_history"].insert(1, {"room_id": "room_2", "entry_step": 1})
        probes.append(p)
        p = _probe(f"stage_contract_conflict_{suffix}")
        p["request"]["candidates"][0] = {"id": "traverse:door:1", "action": "navigate", "subject_id": "door", "subject_type": "portal", "effect": "access_room", "state": "open", "room_id": "room_1", "decision_hint": "POST_INTERACTION_TRAVERSE", "distance_m": 3.0}
        p["request"]["recent_decisions"] = [{"candidate_id": "interaction:door:open" if counter else "traverse:door:1", "step": 3, "result_step": 4, "result": "SUCCEEDED", "behavior_type": "INTERACT" if counter else "NAVIGATE", "target_id": "door", "target_room_id": "room_1", "reason": "executor_feedback_only_not_verified_physical_traverse"}]
        probes.append(p)
        p = _probe(f"top3_diversity_{suffix}")
        extra = copy.deepcopy(p["request"]["candidates"][0]); extra.update(id="frontier:1:3", subject_id="1:3")
        p["request"]["candidates"].append(extra)
        if counter:
            p["request"]["candidates"][1]["room_id"] = "room_1"
        probes.append(p)
    for probe in probes:
        probe["request"]["instruction"] = instruction
        graph = probe["request"]["graph"]
        for candidate in probe["request"]["candidates"]:
            if candidate["subject_type"] == "container":
                graph["containers"].append({"id": candidate["subject_id"], "type": candidate["subject_semantic_type"], "state": candidate["state"], "interaction_available": True, "room_id": candidate["room_id"]})
            elif candidate["subject_type"] == "portal":
                graph["portals"].append({"id": candidate["subject_id"], "type": "portal", "state": "open", "connects": ["room_0", "room_1"], "center_xy": [4.5, 0]})
    return probes


RUBRIC = """# M2 semantic public-policy checks v1

Prediction-blind new checks, not prompt-blind; no predictions are consumed by the generator.
These are secondary soft policy diagnostics, NOT navigation SR, physical truth, hidden target labels,
nor a replacement for frozen historical130/dev38/holdout92. Old holdout has been inspected and is not a new blind set.

## Rules and uncertainty

- semantic_room: explicit whitelist target prior, observed room type confidence>=0.85, compatible versus listed less-likely rooms. Unknown stays unknown.
- semantic_container: plausible closed target-container versus clothing storage; large fixed target versus small storage with a useful frontier alternative. Soft priors are not impossibility except the explicitly large-fixed/small-container constraint.
- semantic_support: matched same-room frontiers/current distance/visible gain, observed nearby plausible support within2m versus another observed object. This does not prove support or target presence.
- initial_room_proximity: matching explicit robot/graph coordinate frames, same known room type, same explicit entered/unentered state, visible+component gain and current distance within10%; initial-point-to-room-center distance smaller by>=1m. Euclidean center distance is not path distance or verified spawn proximity. Missing values do not become zero.
- new_region: same semantic class, visible gain within10%, explicit unentered versus entered. Visiting is not complete exploration.
- exact_failed_repeat: last30 temporally bounded decisions, exact candidateID known failed, current fresh alternative exists. Different viewpoint/ID in same room is not failure; later pending outcome supersedes earlier failure. Future feedback is masked.
- related_unchecked_container: target-related currently closed container with no observed completed inspection versus unrelated clothing storage or already-entered less-likely frontier. Limited history cannot prove truly unchecked; this is not blanket container priority.
- stage_contract_conflict: diagnostic only when stage hints coexist with exact prior success/failure. Never infer physical traversal or phase truth from dispatched actions or feedback, never grade these warnings as navigation success.
- top3_diversity: at least two available nonfailed/noncompleted plausible action-family/room groups; assess whether up-to3 IDs span>=2 groups. Does not establish reachability or independent executable alternatives.

Search preferences are suppressed for explicit stage contracts or contradictory room evidence.
Unsupported/ambiguous target names, missing/control-conflicting facts, and unidentifiable alternatives remain unknown/unscored.
Candidate pre_scores, semantic affinities, prior model choices and future outcomes are never label inputs.

## Metrics contract

Per check publish historical/synthetic separately: applicable N, pass/fail/unknown counts, coverage,
and pass/applicable (lower-bound only). Pairwise check compares listed preferred-other pairs within
ranked_ids; if neither ID is ranked the pair is unknown, NOT automatically correct. Any reversed
resolved pair makes that case fail; at least one preferred pair with no reversal passes. Missing/invalid
ranked IDs fail applicable scored checks. Diagnostic stage warnings have no correctness denominator.
Publish pair resolved/unknown counts; never silently discard unknown from a headline score. This
relative-ranking diagnostic is not the old top1 metric and must not be called its replacement.

## Synthetic probes

18 hand-built public snapshot probes (9 positive/counterexample pairs), distinct synthetic namespace,
independent room geometry and current IDs; no historical future geometry transplanted. They are
controlled instruction diagnostics, not naturally sampled episodes. The counterexample changes
target/semantic compatibility/state/exactID or available diversity to prevent unconditional rules.
Negative controls may make a specific check inapplicable; report that explicitly, not as success.
No labels are embedded in request; checks are a separate artifact. Public priors are versioned in code.
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(inputs: Path, output_dir: Path) -> dict[str, Any]:
    source = inputs / "cases.jsonl" if inputs.is_dir() else inputs
    cases = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Input cases must be nonempty with unique IDs")
    if any(case.get("arm", "full_context") != "full_context" for case in cases):
        raise ValueError("Only frozen full_context inputs are accepted")
    checks = [build_checks(case) for case in cases]
    probes = synthetic_probes(cases[0]["request"].get("instruction", ""))
    probe_checks = [build_checks(case, synthetic=True) for case in probes]
    names = ("checks.jsonl", "synthetic_cases.jsonl", "synthetic_checks.jsonl", "rubric.md", "manifest.json")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Choose a fresh directory; frozen artifacts are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in zip(names[:3], (checks, probes, probe_checks)):
        (output_dir / name).write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    (output_dir / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    def counts(records: list[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(check["name"] for record in records for check in record["checks"] if check["applicable"]))
    manifest = {"schema_version": VERSION, "input_path": str(source.resolve()), "input_sha256": _sha(source),
                "generator_sha256": _sha(Path(__file__)), "history_helper_sha256": _sha(Path(historical.__file__)),
                "historical_cases": len(cases), "synthetic_cases": len(probes),
                "historical_applicable": {name: counts(checks).get(name, 0) for name in CHECK_NAMES},
                "synthetic_applicable": {name: counts(probe_checks).get(name, 0) for name in CHECK_NAMES},
                "target_prior_counts": dict(Counter(row["target_prior_key"] or "unsupported_or_ambiguous" for row in checks)),
                "file_sha256": {name: _sha(output_dir / name) for name in names[:-1]},
                "prediction_blind": True, "old_labels_modified": False, "model_requests": 0}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.inputs, args.output_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
