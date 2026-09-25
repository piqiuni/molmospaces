from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.InteractiveNav.evaluation import m2_context_ablation as context
from scripts.InteractiveNav.evaluation import m2_replay_eval as replay
from scripts.InteractiveNav.evaluation import m2_semantic_experiment as experiment
from scripts.InteractiveNav.evaluation.m2_semantic_checks import synthetic_probes


def test_all_arms_preserve_candidate_facts_and_input():
    request = synthetic_probes("baseline")[0]["request"]
    original = deepcopy(request)
    prompts = {arm: arm for arm in experiment.ARMS[:7]}
    for arm in experiment.ARMS:
        result = experiment.variant_request(request, arm, prompts)
        assert result["candidates"] == original["candidates"]
        assert result["instruction"] == experiment.instruction_for(arm, prompts)
        extra = set(result["robot"]) - set(original["robot"])
        assert extra <= {"stage_facts", "region_history_facts"}
        if arm == "s_all":
            assert len(extra) == 2
            wire = json.dumps(context.wire_messages(result))
            assert "stage_facts" in wire and "region_history_facts" in wire
        for key in extra:
            del result["robot"][key]
        result["instruction"] = original["instruction"]
        assert result == original
    assert request == original


def test_suggestions_have_the_same_p6_parent():
    prompts = {arm: arm for arm in experiment.ARMS[:7]}
    for arm in experiment.ARMS[7:]:
        assert experiment.instruction_for(arm, prompts).startswith("p6\n\n")
    with pytest.raises(ValueError, match="Unknown arm"):
        experiment.instruction_for("unknown", prompts)


def test_prompt_manifest_hashes_and_baseline_unchanged():
    root = Path(__file__).resolve().parents[2]
    folder = root / "scripts/InteractiveNav/configs/semantic_decision/prompts/m2_semantic_20260922"
    manifest = json.loads((folder / "manifest.json").read_text())
    for arm in experiment.ARMS[1:7]:
        row = manifest["prompts"][arm]
        path = folder / row["file"]
        assert experiment.sha(path) == row["file_sha256"]
        assert "/no_think" not in path.read_text()


def test_run_rejects_existing_directory(tmp_path):
    with pytest.raises(FileExistsError, match="fresh run"):
        experiment.run(SimpleNamespace(inputs=str(tmp_path), output_dir=str(tmp_path)))


def test_prepare_rejects_existing_directory(tmp_path):
    args = SimpleNamespace(inputs=str(tmp_path), checks=str(tmp_path), prompt_dir=str(tmp_path), output_dir=str(tmp_path))
    with pytest.raises(FileExistsError, match="fresh output"):
        experiment.prepare(args)


def test_run_shared_pool_metadata_and_artifact_guard(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    original = synthetic_probes("baseline")[0]
    cases = [{**original, "case_id": f"b::r{repeat}::historical::base", "base_case_id": "base",
              "source_case_id": "full_context::base", "dataset": "historical", "arm": "b", "repeat": repeat}
             for repeat in (1, 2)]
    replay._write_jsonl(inputs / "cases.jsonl", cases)
    replay._write_jsonl(inputs / "base_annotations.jsonl", [{"case_id": "base", "acceptable_top1_ids": ["frontier:1:1"]}])
    frozen = {"cases_file_sha256": experiment.sha(inputs / "cases.jsonl"), "cases_sha256": context.digest(cases),
              "max_output_tokens": 1536, "generated_artifact_sha256": {
                  "base_annotations.jsonl": experiment.sha(inputs / "base_annotations.jsonl")}}
    context.write_json(inputs / "manifest.json", frozen)
    calls = []

    def fake_transport(args, output):
        assert args.temperature == 0 and args.reasoning_effort == "off"
        assert args.max_tokens == 1536

        def request(payload, case_id):
            assert "acceptable_top1_ids" not in json.dumps(payload)
            calls.append(case_id)
            return {"ranked_ids": ["frontier:1:1"], "reason": "INFORMATION_GAIN", "confidence": "medium"}, {}
        return request

    monkeypatch.setattr(replay, "build_mllm_request_function", fake_transport)
    args = SimpleNamespace(inputs=str(inputs), output_dir=str(tmp_path / "run"), endpoint="http://unused", model="qwen",
                           concurrency=2, timeout_s=120, max_retries=1, max_tokens=1536)
    manifest = experiment.run(args)
    rows = list(replay._jsonl_rows(tmp_path / "run/predictions.jsonl"))
    assert manifest["completed_requests"] == 2 and manifest["errors"] == {}
    assert len(set(calls)) == 2 and {row["repeat"] for row in rows} == {1, 2}
    assert all(row["base_case_id"] == "base" and row["dataset"] == "historical" for row in rows)
    (inputs / "base_annotations.jsonl").write_text("changed")
    args.output_dir = str(tmp_path / "rejected_run")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        experiment.run(args)
    assert not Path(args.output_dir).exists()
