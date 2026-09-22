from copy import deepcopy

import pytest

from scripts.InteractiveNav.evaluation.m2_replay_eval import audit_public_request
from scripts.InteractiveNav.evaluation.m2_semantic_suggestions import (
    build_region_history_facts,
    build_stage_facts,
    enrich_request,
)


def _request(history=None, candidates=None):
    return {
        "schema_version": 5, "instruction": "fixed prompt", "mission": {"mode": "object_goal"},
        "robot": {"position_frame_id": "map", "current_xy": [0, 0]},
        "recent_decisions": history or [], "graph": {"frame_id": "map"},
        "room_object_reasoning": {},
        "candidates": candidates or [{"id": "traverse:p", "subject_id": "p", "action": "navigate",
                                       "decision_hint": "POST_INTERACTION_TRAVERSE"}],
    }


def _record(raw_candidates=None):
    return {
        "raw_candidates": raw_candidates or [{"candidate_id": "traverse:p", "metadata": {"source_interaction_event_id": "event_new"}}],
        "reconstruction": {"request_cutoff": 100, "semantic_timestamp": 99,
                           "candidate_exact_sequence_and_revision": True},
    }


def _history(decision="d1", candidate="traverse:p", result="SUCCEEDED", **extra):
    return {"decision_id": decision, "candidate_id": candidate, "target_id": "p",
            "behavior_type": "NAVIGATE", "step": 10, "result": result, **extra}


def test_same_candidate_prior_success_does_not_complete_unknown_current_event():
    result = build_stage_facts(_request([_history()]), _record())
    stage = result["current_stage_candidates"][0]
    assert stage["source_interaction_event_id"] == "event_new"
    assert stage["completion_state"] == "not_inferred"
    assert stage["related_history"][0]["event_identity_match"] == "unknown"
    assert result["prior_success_current_hint_conflicts"][0]["interpretation"] == "unknown_event_identity_do_not_infer_current_completion"


def test_different_events_remain_distinct_and_started_is_not_failure():
    history = [_history(source_interaction_event_id="event_old"),
               _history("d2", result="STARTED", success=False, step=20)]
    result = build_stage_facts(_request(history), _record())
    assert result["prior_success_current_hint_conflicts"][0]["event_identity_match"] == "different_event"
    assert result["history_result_counts"] == {"succeeded": 1, "in_progress": 1}
    assert result["current_stage_candidates"][0]["completion_state"] == "not_inferred"


def test_open_success_is_related_evidence_not_traversal_completion_conflict():
    history = [_history(candidate="interaction:p:open", behavior_type="INTERACT")]
    result = build_stage_facts(_request(history), _record())
    assert result["current_stage_candidates"][0]["related_history"][0]["related_by"] == "same_target_id"
    assert result["prior_success_current_hint_conflicts"] == []


def test_unverified_raw_timestamp_cannot_supply_event_identity():
    raw = _record()
    raw["reconstruction"]["semantic_timestamp"] = 101
    result = build_stage_facts(_request([_history()]), raw)
    assert result["current_stage_candidates"][0]["current_event_identity"] == "unknown"
    assert result["raw_candidate_source"] == "raw_snapshot_timestamp_not_verified"


def test_history_is_only_the_frozen_window_and_deduplicates_decisions():
    current = _history()
    raw = _record()
    raw["robot_context"] = {"decision_history": [_history("old1"), _history("old2")]}
    result = build_stage_facts(_request([current, current]), raw)
    assert result["history_decision_count"] == 1


def _region_request(frame=None):
    frame_fields = {"goal_frame_id": frame} if frame else {}
    history = [
        _history(candidate="frontier:old1", result="FAILED", behavior_type="EXPLORE",
                 history_key="explore_region:room_1:1:1", target_room_id="room_1", goal_xy=[1, 1], **frame_fields),
        _history("d2", candidate="frontier:old2", result="STARTED", behavior_type="EXPLORE",
                 history_key="explore_region:room_1:9:9", target_room_id="room_1", goal_xy=[9, 9], **frame_fields),
    ]
    candidates = [{"id": "frontier:new", "action": "explore", "subject_type": "frontier", "room_id": "room_1"}]
    return _request(history, candidates)


def test_same_room_does_not_merge_regions_and_missing_gain_or_visits_stay_unknown():
    result = build_region_history_facts(_region_request())
    assert len(result["regions"]) == 2
    assert result["regions"][0]["failed_execution_count"] == 1
    assert result["regions"][1]["failed_execution_count"] == 0
    assert result["regions"][1]["in_progress_count"] == 1
    assert all(region["gain_measurements"] is None for region in result["regions"])
    assert result["missing_information"]["physical_region_visit_counts"] == "not_observed"
    assert result["current_candidate_region_facts"] == []
    assert result["missing_information"]["candidates_without_explicit_region"] == 1


def test_current_robot_frame_does_not_supply_missing_historical_goal_frame():
    raw = _record([{"candidate_id": "frontier:new", "goal_xyyaw": [1, 1, 0], "metadata": {"frame_id": "map"}}])
    result = build_region_history_facts(_region_request(), raw)
    assert result["current_candidate_region_facts"] == []
    assert result["missing_information"]["candidates_without_comparable_same_frame_history"] == 1


def test_explicit_matching_frames_allow_distance_but_not_region_equivalence():
    raw = _record([{"candidate_id": "frontier:new", "goal_xyyaw": [1, 1, 0], "metadata": {"frame_id": "map"}}])
    result = build_region_history_facts(_region_request("map"), raw)
    candidate = result["current_candidate_region_facts"][0]
    assert candidate["current_goal_xy"] == [1.0, 1.0]
    assert candidate["distances_to_historical_regions"][0]["nearest_recorded_goal_distance_m"] == 0
    assert candidate["distances_to_historical_regions"][0]["same_recorded_goal_xy"] is True
    assert all(item["region_equivalence_inferred"] is False for item in candidate["distances_to_historical_regions"])
    assert "matching_historical_region_key" not in candidate


def test_mismatched_frames_do_not_produce_distances():
    raw = _record([{"candidate_id": "frontier:new", "goal_xyyaw": [1, 1, 0], "metadata": {"frame_id": "odom"}}])
    result = build_region_history_facts(_region_request("map"), raw)
    assert result["current_candidate_region_facts"] == []
    assert result["missing_information"]["candidates_without_comparable_same_frame_history"] == 1


def test_explicit_region_key_can_match_without_geometric_guess():
    raw = _record([{"candidate_id": "frontier:new", "metadata": {"history_key": "explore_region:room_1:1:1"}}])
    result = build_region_history_facts(_region_request(), raw)
    assert result["current_candidate_region_facts"][0]["matching_historical_region_key"] == "explore_region:room_1:1:1"


def test_combined_suggestions_only_add_robot_fields_and_preserve_candidates():
    request = _region_request()
    original = deepcopy(request)
    augmented = enrich_request(request, _record(), stage=True, regions=True)
    assert request == original
    assert augmented["candidates"] == original["candidates"]
    assert set(augmented) == set(original)
    assert set(augmented["robot"]) == set(original["robot"]) | {"stage_facts", "region_history_facts"}
    augmented["robot"].pop("stage_facts")
    augmented["robot"].pop("region_history_facts")
    assert augmented == original
    audit_public_request(enrich_request(request, stage=True, regions=True))


def test_existing_suggestion_fields_are_never_overwritten():
    request = _request()
    request["robot"]["stage_facts"] = {"existing": True}
    with pytest.raises(ValueError, match="overwrite"):
        enrich_request(request, stage=True)


def test_compact_regions_reference_unmodified_history_and_preserve_real_gain():
    request = _region_request()
    request["recent_decisions"][0]["frontier_shrink_m"] = 0.25
    original = deepcopy(request)
    result = enrich_request(request, regions=True)
    region = result["robot"]["region_history_facts"]["regions"][0]
    assert region["history_refs"] == [{"decision_id": "d1"}]
    assert region["gain_measurements"] == [{"decision_id": "d1", "frontier_shrink_m": 0.25}]
    assert "result_sequence" not in region and "recorded_goals" not in region
    assert result["recent_decisions"] == original["recent_decisions"]
    assert result["candidates"] == original["candidates"]


def test_history_without_decision_id_has_resolvable_window_reference():
    request = _region_request()
    request["recent_decisions"][0].pop("decision_id")
    result = build_region_history_facts(request)
    assert result["regions"][0]["history_refs"] == [{"recent_decisions_index": 0}]
