from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.InteractiveNav.evaluation import m2_context_ablation as ablation
from scripts.InteractiveNav.evaluation import m2_replay_eval as replay


def sample_request():
    return {
        "schema_version": 4, "instruction": "Frozen instruction",
        "mission": {"mode": "object_goal", "target": {"name": "toilet"}},
        "robot": {"current_xy": [1, 2], "initial_xy": [0, 0],
                  "current_room": "room_1", "room_visit_history": [{"room_id": "room_1", "entry_xy": [0, 0]}]},
        "graph": {"rooms": [{"id": "room_1", "centroid_xy": [2, 2], "aabb_size_xy": [5, 5],
                              "eligible_frontier_length_m": 4.0, "anchor_objects": ["tv"]}],
                  "portals": [{"id": "door_1", "center_xy": [3, 1]}]},
        "room_object_reasoning": {},
        "candidates": [{"id": "frontier:1", "expected_visible_unknown_area_m2": 9.0}],
        "recent_decisions": [{"candidate_id": "old", "result": "FAILED", "goal_xy": [3, 3]}],
    }


def test_ablations_preserve_prompt_pool_and_source():
    request = sample_request()
    before = deepcopy(request)
    for arm in ablation.ARM_NAMES:
        variant = ablation.context_variant(request, arm)
        assert variant["instruction"] == before["instruction"]
        assert variant["candidates"] == before["candidates"]
    assert request == before


def test_geometry_ablation_removes_geometry_from_history_too():
    variant = ablation.context_variant(sample_request(), "no_geometry")
    assert "current_xy" not in variant["robot"]
    assert "goal_xy" not in variant["recent_decisions"][0]
    assert "center_xy" not in variant["graph"]["portals"][0]
    assert variant["robot"]["current_room"] == "room_1"


def test_frontier_ablation_removes_only_room_aggregate():
    variant = ablation.context_variant(sample_request(), "no_room_frontiers")
    assert "eligible_frontier_length_m" not in variant["graph"]["rooms"][0]
    assert variant["candidates"][0]["expected_visible_unknown_area_m2"] == 9
    assert variant["recent_decisions"]


def test_history_and_route_are_separate_factors():
    request = sample_request()
    assert ablation.context_variant(request, "no_recent_decisions")["robot"]["room_visit_history"]
    assert ablation.context_variant(request, "no_room_route")["recent_decisions"]


def test_token_count_messages_match_production_transport(monkeypatch, tmp_path):
    symbols = replay._runtime_symbols()
    captured = {}

    def capture(self, endpoint, payload, headers, **kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"ranked_ids":["frontier:1"],"reason":"INFORMATION_GAIN","confidence":"high"}'}}]}, ""

    monkeypatch.setattr(symbols["MLLMClient"], "_request_openai_chat_stream", capture)
    config = SimpleNamespace(mode="http", endpoint="http://unused", api_key_env="", model="qwen",
                             protocol="openai_chat", command="", timeout_s=10,
                             temperature=0, max_tokens=1536, reasoning_effort="off")
    request = sample_request()
    response, _ = replay.build_mllm_request_function(config, tmp_path, metrics_path="")(request, "test")
    assert response is not None
    assert captured["messages"] == ablation.wire_messages(request)


def frozen_export_inputs(tmp_path):
    inputs = tmp_path / "frozen"
    inputs.mkdir()
    request = sample_request()
    cases = [{"schema_version": replay.CASE_SCHEMA_VERSION,
              "case_id": f"{arm}::case1", "base_case_id": "case1", "arm": arm,
              "request": request, "request_sha256": ablation.digest(request)}
             for arm in ("full_context", "compact_control")]
    labels = [{"case_id": "case1", "main_score_eligible": True, "split": "holdout",
               "acceptable_top1_ids": ["frontier:1"], "forbidden_ids": []}]
    replay._write_jsonl(inputs / "cases.jsonl", cases)
    replay._write_jsonl(inputs / "labels.jsonl", labels)
    return inputs, cases, labels


def test_export_arm_keeps_requests_and_only_namespaces_labels(tmp_path):
    inputs, cases, labels = frozen_export_inputs(tmp_path)
    before = (inputs / "cases.jsonl").read_bytes()
    output = tmp_path / "remote"
    args = SimpleNamespace(inputs=str(inputs), annotations=str(inputs / "labels.jsonl"),
                           arm="full_context", output_dir=str(output))
    manifest = ablation.export_arm(args)
    assert manifest["case_count"] == 1
    assert replay.load_cases(output / "cases.jsonl") == cases[:1]
    exported = replay.load_annotations(output / "annotations.jsonl")["full_context::case1"]
    assert exported == {**labels[0], "case_id": "full_context::case1", "base_case_id": "case1"}
    assert replay.load_annotations(output / "base_annotations.jsonl") == {"case1": labels[0]}
    assert (inputs / "cases.jsonl").read_bytes() == before
    with pytest.raises(FileExistsError):
        ablation.export_arm(args)


def test_export_arm_refuses_tampered_request_before_writing(tmp_path):
    inputs, cases, _ = frozen_export_inputs(tmp_path)
    cases[0]["request"] = {**cases[0]["request"], "instruction": "Changed"}
    replay._write_jsonl(inputs / "cases.jsonl", cases)
    output = tmp_path / "remote"
    with pytest.raises(ValueError, match="hash mismatch"):
        ablation.export_arm(SimpleNamespace(inputs=str(inputs), annotations=str(inputs / "labels.jsonl"),
                                           arm="full_context", output_dir=str(output)))
    assert not output.exists()
