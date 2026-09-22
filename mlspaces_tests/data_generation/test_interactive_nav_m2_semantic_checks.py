import copy
import json

import pytest

from scripts.InteractiveNav.evaluation import m2_semantic_checks as checks


def indexed(row):
    return {item["name"]: item for item in row["checks"]}


def probe(name):
    return next(row for row in checks.synthetic_probes("frozen instruction") if row["case_id"] == f"synthetic::{name}")


def test_probes_are_public_separate_and_match_current_ids():
    probes = checks.synthetic_probes("frozen instruction")
    assert len(probes) == len({row["case_id"] for row in probes}) == 18
    for row in probes:
        assert row["source"]["synthetic"]
        assert row["request"]["instruction"] == "frozen instruction"
        assert not {"checks", "acceptable_top1_ids", "preferred_ids", "reference", "predictions"} & set(row["request"])
        frozen = checks.build_checks(row, synthetic=True)
        assert frozen["dataset_kind"] == "synthetic"
        current = set(frozen["candidate_ids"])
        for check in frozen["checks"]:
            assert set(check["preferred_ids"] + check["disfavored_ids"]) <= current


def test_initial_nearness_is_controlled_not_blanket_semantic_override():
    positive = indexed(checks.build_checks(probe("initial_room_proximity_positive")))
    counter = indexed(checks.build_checks(probe("initial_room_proximity_counter")))
    assert positive["initial_room_proximity"]["preferred_ids"] == ["frontier:1:1"]
    assert not counter["initial_room_proximity"]["applicable"]
    assert counter["semantic_room"]["preferred_ids"] == ["frontier:2:2"]
    missing = probe("initial_room_proximity_positive")
    missing["request"]["candidates"][0].pop("unknown_component_area_m2")
    assert not indexed(checks.build_checks(missing))["initial_room_proximity"]["applicable"]
    wrong_frame = probe("initial_room_proximity_positive")
    wrong_frame["request"]["robot"]["position_frame_id"] = "odom_untransformed"
    assert not indexed(checks.build_checks(wrong_frame))["initial_room_proximity"]["applicable"]


def test_novelty_controls_semantics_and_exact_failure_uses_known_cutoff():
    positive = indexed(checks.build_checks(probe("new_region_positive")))
    counter = indexed(checks.build_checks(probe("new_region_counter")))
    assert positive["new_region"]["preferred_ids"] == ["frontier:1:1"]
    assert not counter["new_region"]["applicable"]
    assert counter["semantic_room"]["preferred_ids"] == ["frontier:2:2"]
    failed = probe("exact_failed_repeat_positive")
    assert indexed(checks.build_checks(failed))["exact_failed_repeat"]["disfavored_ids"] == ["frontier:1:1"]
    assert not indexed(checks.build_checks(probe("exact_failed_repeat_counter")))["exact_failed_repeat"]["applicable"]
    failed["request"]["recent_decisions"][0]["result_step"] = 11
    assert not indexed(checks.build_checks(failed))["exact_failed_repeat"]["applicable"]
    failed["request"]["recent_decisions"][0]["result_step"] = 4
    failed["request"]["recent_decisions"].append({"candidate_id": "frontier:1:1", "step": 5, "result": "PENDING"})
    assert not indexed(checks.build_checks(failed))["exact_failed_repeat"]["applicable"]


def test_container_and_support_priors_have_counterexamples():
    positive = indexed(checks.build_checks(probe("semantic_container_positive")))
    counter = indexed(checks.build_checks(probe("semantic_container_counter")))
    assert positive["semantic_container"]["preferred_ids"] == ["interaction:fridge:open"]
    assert counter["semantic_container"]["disfavored_ids"] == ["interaction:fridge:open"]
    assert indexed(checks.build_checks(probe("semantic_support_positive")))["semantic_support"]["preferred_ids"] == ["frontier:1:1"]
    assert indexed(checks.build_checks(probe("semantic_support_counter")))["semantic_support"]["preferred_ids"] == ["frontier:2:2"]
    assert indexed(checks.build_checks(probe("related_unchecked_container_positive")))["related_unchecked_container"]["applicable"]
    assert not indexed(checks.build_checks(probe("related_unchecked_container_counter")))["related_unchecked_container"]["applicable"]
    reclosed = probe("semantic_container_positive")
    reclosed["request"]["recent_decisions"] = [{"candidate_id": "interaction:fridge:open", "step": 1, "result_step": 2, "result": "SUCCEEDED"}]
    assert not indexed(checks.build_checks(reclosed))["related_unchecked_container"]["applicable"]


def test_stage_conflict_is_diagnostic_not_a_physical_success_grade():
    case = probe("stage_contract_conflict_positive")
    frozen = checks.build_checks(case)
    by_name = indexed(frozen)
    assert by_name["stage_contract_conflict"]["conflict_ids"] == ["traverse:door:1"]
    assert not by_name["initial_room_proximity"]["applicable"]
    scored = indexed(checks.score_checks(frozen, {"ranked_ids": ["traverse:door:1"]}))
    assert scored["stage_contract_conflict"]["status"] == "diagnostic_only"
    assert scored["stage_contract_conflict"]["selected_conflicting_hint"]
    counter = indexed(checks.build_checks(probe("stage_contract_conflict_counter")))
    assert counter["stage_contract_conflict"]["conflict_ids"] == []


def test_unknown_and_schema_failures_do_not_become_success():
    case = probe("semantic_room_positive")
    third = copy.deepcopy(case["request"]["candidates"][0])
    third.update(id="frontier:3:3", room_id="room_3", room_status="entered_room")
    case["request"]["candidates"].append(third)
    frozen = checks.build_checks(case)
    for response in [None, {"ranked_ids": ["invented"]}, {"ranked_ids": ["frontier:1:1", "frontier:1:1"]}, {"ranked_ids": [{}]}]:
        assert indexed(checks.score_checks(frozen, response))["semantic_room"]["status"] == "fail"
    assert indexed(checks.score_checks(frozen, {"ranked_ids": ["frontier:3:3"]}))["semantic_room"]["status"] == "unknown"
    assert indexed(checks.score_checks(frozen, {"ranked_ids": ["frontier:2:2", "frontier:1:1"]}))["semantic_room"]["status"] == "fail"
    assert indexed(checks.score_checks(frozen, {"ranked_ids": ["frontier:1:1", "frontier:2:2"]}))["semantic_room"]["status"] == "pass"


def test_top3_diversity_and_conflicting_room_evidence():
    frozen = checks.build_checks(probe("top3_diversity_positive"))
    assert indexed(checks.score_checks(frozen, {"ranked_ids": ["frontier:1:1", "frontier:1:3"]}))["top3_diversity"]["status"] == "fail"
    assert indexed(checks.score_checks(frozen, {"ranked_ids": ["frontier:1:1", "frontier:2:2"]}))["top3_diversity"]["status"] == "pass"
    assert not indexed(checks.build_checks(probe("top3_diversity_counter")))["top3_diversity"]["applicable"]
    case = probe("semantic_room_positive")
    case["request"]["candidates"][0].update(room_attribute="bathroom", room_attribute_confidence=0.99)
    assert not indexed(checks.build_checks(case))["semantic_room"]["applicable"]


def test_generate_freezes_only_secondary_artifacts(tmp_path):
    inputs = tmp_path / "cases.jsonl"
    case = probe("semantic_room_positive")
    case.update(case_id="full_context::historical-example", arm="full_context")
    inputs.write_text(json.dumps(case) + "\n")
    output = tmp_path / "frozen"
    manifest = checks.generate(inputs, output)
    assert manifest["historical_cases"] == 1
    assert manifest["synthetic_cases"] == 18
    assert manifest["model_requests"] == 0
    assert not manifest["old_labels_modified"]
    assert manifest["historical_applicable"]["semantic_room"] == 1
    assert all(checks._sha(output / name) == sha for name, sha in manifest["file_sha256"].items())
    with pytest.raises(FileExistsError):
        checks.generate(inputs, output)
