from __future__ import annotations

import copy
import json

import pytest

from scripts.InteractiveNav.evaluation import m2_context_labels as labels


def _case():
    return {
        "case_id": "case-1",
        "source": {"attempt": "episode_0001/attempt_001", "cutoff_step": 100, "cutoff_timestamp": 1000},
        "request": {
            "mission": {"target": {"name": "toilet", "labels": ["toilet", "potty"]}},
            "robot": {
                "current_room": "room_1", "initial_xy": [0, 0], "current_xy": [1, 1],
                "room_visit_history": [{"room_id": "room_1", "entry_step": 1}],
            },
            "graph": {"rooms": [{"id": "room_1", "type": "unknown"}], "current_room": "room_1"},
            "candidates": [
                {"id": "frontier:large", "subject_type": "frontier", "room_id": "room_1", "action": "explore", "expected_visible_unknown_area_m2": 20},
                {"id": "frontier:small", "subject_type": "frontier", "room_id": "room_1", "action": "explore", "expected_visible_unknown_area_m2": 5},
                {"id": "interaction:fridge", "subject_type": "container", "subject_semantic_type": "refrigerator", "action": "open", "state": "closed"},
            ],
            "recent_decisions": [],
        },
    }


def test_labels_ignore_affinity_scores_reasoning_and_prior_predictions():
    case = _case()
    expected = labels.annotate(case, "holdout")
    case["reference"] = {"response": {"ranked_ids": ["interaction:fridge"]}}
    case["request"]["room_object_reasoning"] = {"target": {"plausible_room_types": ["kitchen"]}}
    for candidate in case["request"]["candidates"]:
        candidate.update({
            "pre_score": 99, "pre_score_terms": {"magic": 99},
            "room_target_affinity": 1, "decision_hint": "PLAUSIBLE_TARGET_CONTAINER",
        })
    assert labels.annotate(case, "holdout") == expected
    assert expected["acceptable_top1_ids"] == ["frontier:large"]
    assert expected["forbidden_ids"] == ["interaction:fridge"]


def test_only_exact_candidate_history_failure_is_deferred():
    case = _case()
    case["request"]["candidates"][0]["history"] = {"last_result": "FAILED", "low_gain_repeat_count": 9}
    assert labels.annotate(case, "dev")["acceptable_top1_ids"] == ["frontier:large"]
    case["request"]["recent_decisions"] = [{"candidate_id": "frontier:large", "step": 90, "result": "FAILED"}]
    result = labels.annotate(case, "dev")
    assert result["acceptable_top1_ids"] == ["frontier:small"]
    assert result["deprioritized_ids"] == ["frontier:large"]


def test_future_decisions_room_visits_and_results_are_not_used():
    case = _case()
    case["request"]["recent_decisions"] = [
        {"candidate_id": "frontier:large", "step": 101, "result": "FAILED"},
        {"candidate_id": "frontier:large", "step": 90, "result": "FAILED", "result_step": 101},
    ]
    case["request"]["robot"]["room_visit_history"].append({"room_id": "room_2", "entry_step": 110})
    result = labels.annotate(case, "dev")
    assert result["acceptable_top1_ids"] == ["frontier:large"]
    assert result["deprioritized_ids"] == []
    assert result["history_records_used"] == 1
    assert result["room_visit_records_used"] == 1
    assert {"future_history_rejected", "future_outcome_masked"} <= set(result["data_quality_flags"])


def test_history_is_bounded_to_thirty_and_latest_retry_supersedes_failure():
    case = _case()
    case["request"]["recent_decisions"] = [
        {"candidate_id": "frontier:large", "step": 1, "result": "FAILED"},
        *[{"candidate_id": f"other:{step}", "step": step, "result": "SUCCEEDED"} for step in range(2, 32)],
    ]
    result = labels.annotate(case, "dev")
    assert result["history_records_used"] == 30
    assert result["acceptable_top1_ids"] == ["frontier:large"]
    case["request"]["recent_decisions"] = [
        {"candidate_id": "frontier:large", "step": 80, "result": "FAILED"},
        {"candidate_id": "frontier:large", "step": 90, "result": "PENDING"},
    ]
    assert labels.annotate(case, "dev")["deprioritized_ids"] == []


def test_room_history_distinguishes_visited_from_unentered_and_conflict():
    case = _case()
    small = case["request"]["candidates"][1]
    small.update({"room_id": "room_2", "room_status": "unentered_new_room"})
    assert labels.annotate(case, "dev")["acceptable_top1_ids"] == ["frontier:small"]
    case["request"]["robot"]["room_visit_history"].append({"room_id": "room_2", "entry_step": 50})
    result = labels.annotate(case, "dev")
    assert result["status"] == "unscored"
    assert result["acceptable_top1_ids"] == []
    assert "room_2" in result["public_evidence_conflicts"]


def test_contradictory_room_types_unscored_except_explicit_stage():
    case = _case()
    case["request"]["graph"]["rooms"][0].update({"type": "bathroom", "room_attribute_confidence": 0.95})
    case["request"]["candidates"][0].update({"room_attribute": "kitchen", "room_attribute_confidence": 0.95})
    assert labels.annotate(case, "dev")["status"] == "unscored"
    case["request"]["candidates"][1]["decision_hint"] = "POST_INTERACTION_TRAVERSE"
    result = labels.annotate(case, "dev")
    assert result["acceptable_top1_ids"] == ["frontier:small"]
    assert result["label_kind"] == "public_stage_contract"


def test_successful_open_can_be_reopened_after_explicit_closed_state():
    case = _case()
    case["request"]["mission"]["target"] = {"name": "apple"}
    case["request"]["recent_decisions"] = [{"candidate_id": "interaction:fridge", "step": 90, "result": "SUCCEEDED"}]
    assert "interaction:fridge" not in labels.annotate(case, "dev")["forbidden_ids"]
    case["request"]["candidates"][2]["state"] = "unknown"
    assert "interaction:fridge" in labels.annotate(case, "dev")["forbidden_ids"]


def test_missing_area_stays_unknown_and_multiple_answers_are_allowed():
    case = _case()
    del case["request"]["candidates"][1]["expected_visible_unknown_area_m2"]
    result = labels.annotate(case, "holdout")
    assert result["acceptable_top1_ids"] == ["frontier:large", "frontier:small"]
    assert "missing_frontier_visible_area" in result["data_quality_flags"]
    case["request"]["candidates"] = case["request"]["candidates"][:2]
    assert labels.annotate(case, "holdout")["status"] == "unscored"


def test_generate_keeps_fixed_split_and_refuses_overwrite(tmp_path):
    case = _case()
    other = copy.deepcopy(case)
    other["case_id"] = "case-2"
    other["source"]["attempt"] = "episode_0002/attempt_001"
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("\n".join(json.dumps(row) for row in (case, other)) + "\n")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"episode_assignment": {case["source"]["attempt"]: "holdout", other["source"]["attempt"]: "dev"}}))
    output = tmp_path / "labels"
    manifest = labels.generate(inputs, split, output)
    annotations = [json.loads(line) for line in (output / "annotations.jsonl").read_text().splitlines()]
    assert [row["split"] for row in annotations] == ["holdout", "dev"]
    assert manifest["blindness"]["prediction_blind"]
    assert not manifest["blindness"]["prompt_blind"]
    with pytest.raises(FileExistsError):
        labels.generate(inputs, split, output)


def test_missing_split_attempt_and_duplicate_candidate_ids_are_rejected(tmp_path):
    case = _case()
    case["request"]["candidates"].append(case["request"]["candidates"][0])
    with pytest.raises(ValueError, match="candidate IDs"):
        labels.annotate(case, "dev")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(json.dumps(_case()) + "\n")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"episode_assignment": {}}))
    with pytest.raises(KeyError):
        labels.generate(inputs, split, tmp_path / "labels")
