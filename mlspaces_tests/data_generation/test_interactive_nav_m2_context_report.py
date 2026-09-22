from __future__ import annotations

import json

import pytest

from scripts.InteractiveNav.evaluation import m2_context_report as report


def _case(arm, base, *, strict, candidate_ids=("good", "bad")):
    return {
        "case_id": f"{arm}::{base}", "base_case_id": base, "arm": arm,
        "source": {"reconstruction": {"strict_version_aligned": strict}},
        "input_tokens_estimate": 50,
        "request": {"candidates": [{"id": identity} for identity in candidate_ids]},
    }


def _prediction(case, top1):
    return {
        "case_id": case["case_id"],
        "response": {"ranked_ids": [top1], "reason": "INFORMATION_GAIN", "confidence": "high"},
        "prompt_tokens": 100, "total_tokens": 110, "retry_count": 1,
        "attempts": [{"prompt_tokens": 50}, {"prompt_tokens": 50}],
        "latency_s": 2, "total_latency_s": 3,
    }


def _label(base, split):
    return {
        "case_id": base, "split": split, "main_score_eligible": True,
        "acceptable_top1_ids": ["good"], "forbidden_ids": ["bad"],
        "label_kind": "policy_rubric_proxy", "single_candidate": False,
    }


def test_fixed_denominators_include_missing_prediction_and_candidate_recall_loss(tmp_path):
    inputs, run = tmp_path / "inputs", tmp_path / "run"
    inputs.mkdir()
    run.mkdir()
    cases = [
        _case("full_context", "one", strict=True),
        _case("legacy_candidate_pool", "one", strict=True, candidate_ids=("bad",)),
        _case("full_context", "two", strict=False),
        _case("legacy_candidate_pool", "two", strict=False),
    ]
    predictions = [_prediction(cases[0], "good"), _prediction(cases[1], "bad"), _prediction(cases[3], "good")]
    (inputs / "cases.jsonl").write_text("\n".join(json.dumps(row) for row in cases) + "\n")
    (inputs / "manifest.json").write_text("{}")
    (run / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in predictions) + "\n")
    (run / "manifest.json").write_text("{}")
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text("\n".join(json.dumps(row) for row in (_label("one", "dev"), _label("two", "holdout"))) + "\n")
    result = report.generate(inputs, run, annotations, tmp_path / "report")
    full = result["by_arm"]["full_context"]["all"]["all"]
    legacy = result["by_arm"]["legacy_candidate_pool"]["all"]["all"]
    assert full["main_acceptable_top1"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert full["missing_predictions"] == 1
    assert legacy["candidate_recall"]["denominator"] == 2
    assert legacy["upstream_candidate_recall_losses"] == 1
    assert legacy["forbidden_top1_exposed"]["numerator"] == 1
    assert full["distributions"]["input_tokens_actual"]["p50"] == 50  # Not 100 summed over retries.
    pair = result["paired_full_vs_ablation"]["legacy_candidate_pool"]["all"]["all"]
    assert (pair["full_only_accepted"], pair["ablation_only_accepted"], pair["net_full_gain"]) == (1, 1, 0)
    assert result["by_arm"]["full_context"]["holdout"]["lagged_or_incomplete"]["main_acceptable_top1"]["denominator"] == 1
    assert (tmp_path / "report" / "report.md").exists()


def test_transport_error_or_bad_schema_never_count_as_correct():
    case = _case("full_context", "one", strict=True)
    annotation = _label("one", "dev")
    prediction = _prediction(case, "good")
    prediction["transport_error"] = "timeout"
    row = report._score_case(case, prediction, annotation)
    assert not row["accepted"] and not row["valid"]
    prediction.pop("transport_error")
    prediction["response"]["ranked_ids"] = ["unlisted"]
    row = report._score_case(case, prediction, annotation)
    assert not row["accepted"] and row["schema_error"]
    stat = report.summarize([row])
    assert stat["main_acceptable_top1"]["denominator"] == 1
    assert stat["schema_error_count"] == 1


def test_paired_summary_refuses_different_denominators():
    full = [{"base_case_id": "one", "main_score_eligible": True, "accepted": True}]
    with pytest.raises(ValueError, match="fixed eligible case"):
        report.paired(full, [])


def test_guard_shadow_uses_production_guard_without_replacing_raw_score():
    case = _case("full_context", "one", strict=True)
    row = {
        "case_id": case["case_id"], "base_case_id": "one", "arm": "full_context",
        "valid": True, "top1": "bad", "accepted": False, "main_score_eligible": True,
    }
    diagnostics = {"one": {"pools": {"all_actions_12_frontiers": {
        "quality_by_id": {"good": 2.0, "bad": 0.0},
        "decision_hint_by_id": {"good": "NEXT_ROUTE_PORTAL"},
    }}}}
    shadow = report.guard_shadow([row], {case["case_id"]: case}, diagnostics, {"one": _label("one", "dev")})
    assert shadow["by_arm"]["full_context"]["main_recovered"] == 1
    assert shadow["changed_cases"][0]["shadow_top1"] == "good"
    assert row["top1"] == "bad" and not row["accepted"]
