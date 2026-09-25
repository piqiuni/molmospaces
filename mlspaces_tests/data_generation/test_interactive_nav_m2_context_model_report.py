from __future__ import annotations

import json

import pytest

from scripts.InteractiveNav.evaluation import m2_context_model_report as models


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    cases, labels = [], []
    for base, split in (("one", "dev"), ("two", "holdout")):
        cases.append({
            "case_id": f"full_context::{base}", "base_case_id": base, "arm": "full_context",
            "source": {"reconstruction": {"strict_version_aligned": base == "one"}},
            "request": {"candidates": [{"id": "good"}, {"id": "bad"}]},
            "input_tokens_estimate": 100,
        })
        labels.append({
            "case_id": base, "split": split, "main_score_eligible": True,
            "acceptable_top1_ids": ["good"], "forbidden_ids": ["bad"],
            "label_kind": "public_stage_contract" if base == "one" else "policy_rubric_proxy",
            "single_candidate": False,
        })
    _write_rows(inputs / "cases.jsonl", cases)
    _write_rows(inputs / "base_annotations.jsonl", labels)
    (inputs / "manifest.json").write_text("{}")
    return inputs, cases


def _prediction(case, top1, *, latency=3, wait=0):
    return {
        "case_id": case["case_id"], "base_case_id": case["base_case_id"],
        "response": {"ranked_ids": [top1], "reason": "INFORMATION_GAIN", "confidence": "high"},
        "latency_s": latency, "total_latency_s": latency + wait, "rate_limit_wait_s": wait,
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
        "attempts": [{"prompt_tokens": 100, "completion_tokens": 20}],
    }


def test_common_scoring_and_qwen_pairing_keep_errors_and_split_denominators(tmp_path):
    inputs, cases = _fixture(tmp_path)
    qwen_path, gpt_path = tmp_path / "qwen.jsonl", tmp_path / "gpt.jsonl"
    qwen = [_prediction(cases[0], "good"), _prediction(cases[1], "bad")]
    extra = dict(qwen[0], case_id="no_geometry::one", arm="no_geometry")
    _write_rows(qwen_path, [*qwen, extra])
    gpt = [_prediction(cases[0], "good", latency=5, wait=5), _prediction(cases[1], "good", latency=5, wait=5)]
    gpt[0].update(response=None, transport_error="HTTP 502")
    _write_rows(gpt_path, gpt)
    result = models.generate(inputs, {"qwen": qwen_path, "gpt": gpt_path}, tmp_path / "report")
    assert result["run_checks"]["qwen"]["ignored_other_qwen_arms"] == 1
    assert result["fixed_primary_by_split"] == {"dev": 1, "holdout": 1}
    stats = result["by_model"]["gpt"]["all"]["all"]
    assert stats["main_acceptable_top1"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert stats["transport_error_count"] == 1
    assert stats["stage_acceptance"]["denominator"] == 1
    assert stats["distributions"]["latency_s"]["mean"] == 5
    assert stats["distributions"]["total_latency_s"]["mean"] == 10
    assert stats["distributions"]["rate_limit_wait_s"]["mean"] == 5
    pair = result["paired_vs_qwen"]["gpt"]["all"]["all"]
    assert pair["reference_only_accepted"] == pair["model_only_accepted"] == 1
    assert pair["net_model_gain"] == 0
    assert (tmp_path / "report" / "report.md").exists()


def test_incomplete_runs_do_not_write_partial_report(tmp_path):
    inputs, cases = _fixture(tmp_path)
    qwen_path, remote_path = tmp_path / "qwen.jsonl", tmp_path / "remote.jsonl"
    _write_rows(qwen_path, [_prediction(case, "good") for case in cases])
    _write_rows(remote_path, [_prediction(cases[0], "good")])
    with pytest.raises(ValueError, match="Incomplete run"):
        models.generate(inputs, {"qwen": qwen_path, "gemini": remote_path}, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_remote_cannot_silently_mix_other_arms_and_duplicates_are_rejected(tmp_path):
    _, cases = _fixture(tmp_path)
    case_map = {case["case_id"]: case for case in cases}
    path = tmp_path / "predictions.jsonl"
    predictions = [_prediction(case, "good") for case in cases]
    _write_rows(path, [*predictions, dict(predictions[0], case_id="no_geometry::one", arm="no_geometry")])
    with pytest.raises(ValueError, match="Unexpected prediction"):
        models._complete_predictions(path, case_map, allow_qwen_ablations=False)
    _write_rows(path, [*predictions, predictions[0]])
    with pytest.raises(ValueError, match="duplicate"):
        models._complete_predictions(path, case_map, allow_qwen_ablations=True)


def test_namespaced_annotation_fallback_preserves_labels_and_input_arm_check(tmp_path):
    inputs, cases = _fixture(tmp_path)
    labels = [json.loads(line) for line in (inputs / "base_annotations.jsonl").read_text().splitlines()]
    # A separate fixture exercises the fallback without deleting artifacts.
    other = tmp_path / "namespaced_inputs"
    other.mkdir()
    _write_rows(other / "cases.jsonl", cases)
    _write_rows(other / "annotations.jsonl", [dict(label, case_id=f"full_context::{label['case_id']}", base_case_id=label["case_id"]) for label in labels])
    _, by_base, _ = models._frozen_cases(other)
    assert set(by_base) == {"one", "two"}
    assert by_base["one"]["acceptable_top1_ids"] == ["good"]
    cases[0]["arm"] = "compact_control"
    _write_rows(other / "cases.jsonl", cases)
    with pytest.raises(ValueError, match="only full_context"):
        models._frozen_cases(other)
