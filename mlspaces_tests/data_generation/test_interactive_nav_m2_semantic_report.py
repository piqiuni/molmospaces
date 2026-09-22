from copy import deepcopy
import json

import pytest

from scripts.InteractiveNav.evaluation import m2_replay_eval as replay
from scripts.InteractiveNav.evaluation.m2_semantic_report import build_report, generate, render_markdown


def _response(top1="a"):
    return {"ranked_ids": [top1], "reason": sorted(replay.ALLOWED_REASON_CODES)[0],
            "confidence": sorted(replay.ALLOWED_CONFIDENCE_CODES)[0]}


def _fixtures(arms=("b", "p1", "p2", "p6", "s1_stage")):
    cases, predictions, historical, synthetic = [], [], [], []
    labels = [{"case_id": base, "split": split, "main_score_eligible": eligible,
               "acceptable_top1_ids": ["a"], "forbidden_ids": ["b"], "single_candidate": not eligible}
              for base, split, eligible in (("h1", "dev", True), ("h2", "holdout", True), ("h3", "holdout", False))]
    for dataset, bases in (("historical", ("h1", "h2", "h3")), ("synthetic", ("s1",))):
        for base in bases:
            source_id = f"{dataset}::{base}"
            check_row = {"case_id": source_id, "candidate_ids": ["a", "b", "c"], "checks": [
                {"name": "semantic_room", "mode": "pairwise", "applicable": True,
                 "pairs": [{"preferred_id": "a", "other_id": "b"}]},
                {"name": "semantic_support", "mode": "pairwise", "applicable": False, "pairs": []},
                {"name": "stage_contract_conflict", "mode": "diagnostic", "applicable": True,
                 "conflict_ids": ["a"]},
            ]}
            (historical if dataset == "historical" else synthetic).append(check_row)
            for arm in arms:
                for repeat in (1, 2):
                    case_id = f"{arm}::r{repeat}::{source_id}"
                    case = {"case_id": case_id, "base_case_id": base, "source_case_id": source_id,
                            "dataset": dataset, "arm": arm, "repeat": repeat,
                            "source": {"reconstruction": {"strict_version_aligned": base == "h1"}},
                            "request": {"candidates": [{"id": candidate_id} for candidate_id in ("a", "b", "c")]},
                            "input_tokens_estimate": 50}
                    cases.append(case)
                    top1 = "c" if dataset == "synthetic" else "a"
                    if arm == "b" and base == "h1" and repeat == 2:
                        top1 = "b"
                    if arm == "p2" and base == "h2" and repeat == 1:
                        top1 = "b"
                    predictions.append({"case_id": case_id, "response": _response(top1), "prompt_tokens": 100,
                                        "total_tokens": 105, "latency_s": 2.0, "total_latency_s": 2.2,
                                        "retry_count": 1, "attempts": [{"prompt_tokens": 40}, {"prompt_tokens": 60}]})
    return cases, predictions, labels, historical, synthetic


def _summary(report, arm="b", repeat="all"):
    return report["by_arm"][arm]["by_repeat"][repeat]["historical"]["all"]["all"]


def test_pooled_primary_keeps_both_repeats_and_synthetic_has_separate_denominator():
    report = build_report(*_fixtures())
    assert _summary(report)["main_acceptable_top1"] == {"numerator": 3, "denominator": 4, "rate": 0.75}
    assert _summary(report, repeat="1")["main_acceptable_top1"]["denominator"] == 2
    assert _summary(report, repeat="2")["main_acceptable_top1"]["numerator"] == 1
    synthetic = report["by_arm"]["b"]["by_repeat"]["all"]["synthetic"]
    assert "main_acceptable_top1" not in synthetic
    assert synthetic["cases"] == 2
    assert _summary(report)["cases"] == 6


def test_paired_key_includes_repeat_and_extra_controls_are_reported():
    report = build_report(*_fixtures())
    pairs = {(item["target"], item["control"]): item for item in report["paired_comparisons"]}
    assert {("p2", "p1"), ("s1_stage", "p6"), ("p2", "b")} <= set(pairs)
    result = pairs[("p2", "b")]["by_repeat"]["all"]["historical_primary"]["all"]["all"]
    assert result["denominator"] == 4
    assert result["target_only_accepted"] == result["control_only_accepted"] == 1
    assert result["target_win_cases"] == [{"repeat": 2, "base_case_id": "h1"}]
    assert result["target_loss_cases"] == [{"repeat": 1, "base_case_id": "h2"}]


@pytest.mark.parametrize("fault", ["schema", "transport", "missing"])
def test_invalid_complete_schema_or_transport_or_missing_cannot_pass_checks(fault):
    fixtures = list(_fixtures(("b",)))
    prediction = fixtures[1][0]
    if fault == "schema":
        prediction["response"]["reason"] = "not_a_valid_reason"
    elif fault == "transport":
        prediction["transport_error"] = "mock_transport_failure"
    else:
        fixtures[1].pop(0)
    report = build_report(*fixtures)
    row = next(row for row in report["case_scores"] if row["base_case_id"] == "h1" and row["repeat"] == 1)
    assert row["valid"] is False
    assert row["accepted"] is False
    assert row["strategy_checks"][0]["status"] == "fail"
    assert _summary(report)["main_acceptable_top1"]["denominator"] == 4


def test_unknown_and_zero_support_are_not_passes_and_stage_is_not_scored():
    report = build_report(*_fixtures())
    checks = report["by_arm"]["b"]["by_repeat"]["all"]["synthetic"]["strategy_checks"]
    assert checks["semantic_room"]["unknown"] == 2
    assert checks["semantic_room"]["pass_over_applicable"]["denominator"] == 2
    assert checks["semantic_room"]["pass_over_applicable"]["numerator"] == 0
    assert checks["semantic_room"]["known_only_pass"]["denominator"] == 0
    assert checks["semantic_support"]["applicable"] == 0
    assert checks["semantic_support"]["pass_over_applicable"]["rate"] is None
    assert checks["stage_contract_conflict"]["pass_over_applicable"] is None
    assert checks["stage_contract_conflict"]["diagnostic_only"] == 2
    assert report["check_support"]["historical"]["applicable_per_round"]["semantic_support"] == 0


def test_nonprimary_invalid_schema_does_not_change_primary_denominator():
    fixtures = list(_fixtures(("b",)))
    item = next(item for item in fixtures[1] if item["case_id"].endswith("historical::h3"))
    item["response"]["ranked_ids"] = ["a", "a"]
    report = build_report(*fixtures)
    assert _summary(report)["schema_error_count"] == 1
    assert _summary(report)["main_acceptable_top1"]["numerator"] == 3
    assert _summary(report)["main_acceptable_top1"]["denominator"] == 4


def test_stability_excludes_invalid_empty_agreement_and_keeps_accuracy_distinct():
    fixtures = list(_fixtures(("b",)))
    for prediction in fixtures[1]:
        if prediction["case_id"].endswith("historical::h3"):
            prediction["response"] = None
    report = build_report(*fixtures)
    stats = report["by_arm"]["b"]["stability"]["historical"]
    assert stats["all_repeats_valid"] == 2
    assert stats["any_missing_or_invalid_repeat"] == 1
    assert stats["top1_agreement_given_all_valid"]["denominator"] == 2
    assert stats["top1_agreement_given_all_valid"]["numerator"] == 1
    assert stats["primary_all_repeats_accepted"]["numerator"] == 1
    assert stats["primary_at_least_one_repeat_accepted"]["numerator"] == 2


def test_input_tokens_use_last_attempt_but_consumption_includes_retries():
    report = build_report(*_fixtures(("b",)))
    stats = _summary(report)
    assert stats["distributions"]["input_tokens_actual"]["mean"] == 60
    assert stats["prompt_tokens_all_attempts"] == 600
    assert stats["total_tokens_all_attempts"] == 630
    assert stats["retry_count"] == 6


@pytest.mark.parametrize("fault", ["duplicate_prediction", "unknown_prediction", "incomplete_matrix", "changed_candidates"])
def test_rejects_nonpaired_or_ambiguous_inputs(fault):
    fixtures = list(_fixtures())
    if fault == "duplicate_prediction":
        fixtures[1].append(deepcopy(fixtures[1][0]))
    elif fault == "unknown_prediction":
        fixtures[1].append({"case_id": "not_in_frozen_inputs"})
    elif fault == "incomplete_matrix":
        removed = fixtures[0].pop()
        fixtures[1] = [item for item in fixtures[1] if item["case_id"] != removed["case_id"]]
    else:
        fixtures[0][0]["request"]["candidates"].reverse()
    with pytest.raises(ValueError):
        build_report(*fixtures)


def test_markdown_states_scope_limits_and_checks_have_applicable_denominators():
    report = build_report(*_fixtures(("b",)))
    markdown = render_markdown(report)
    assert "不是导航成功率" in markdown
    assert "Synthetic 独立策略检查" in markdown
    assert "Unknown" in markdown and "不计分" in markdown
    assert "4" in markdown
    assert "not pure prompt ablations" in markdown


def test_generate_writes_fresh_report_without_modifying_inputs(tmp_path):
    inputs, run, output = tmp_path / "inputs", tmp_path / "run", tmp_path / "report"
    inputs.mkdir()
    run.mkdir()
    cases, predictions, labels, historical, synthetic = _fixtures(("b",))
    for path, rows in ((inputs / "cases.jsonl", cases), (run / "predictions.jsonl", predictions),
                       (inputs / "base_annotations.jsonl", labels), (inputs / "checks.jsonl", historical),
                       (inputs / "synthetic_checks.jsonl", synthetic)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (inputs / "manifest.json").write_text(json.dumps({"arms": ["b"], "repeats": 2}))
    (run / "manifest.json").write_text("{}")
    before = {path: path.read_bytes() for path in inputs.iterdir()}
    report = generate(inputs, run, output)
    assert (output / "report.md").exists()
    assert json.loads((output / "report.json").read_text())["planned_requests"] == 8
    assert report["predictions_received"] == 8
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(FileExistsError):
        generate(inputs, run, output)


def test_frozen_hash_mismatch_prevents_scoring(tmp_path):
    inputs, run = tmp_path / "inputs", tmp_path / "run"
    inputs.mkdir()
    run.mkdir()
    (inputs / "manifest.json").write_text(json.dumps({"generated_artifact_sha256": {"cases.jsonl": "wrong"}}))
    (inputs / "cases.jsonl").write_text("[]")
    (run / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        generate(inputs, run, tmp_path / "report")
