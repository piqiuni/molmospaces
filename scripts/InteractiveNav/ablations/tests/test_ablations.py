from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from ablations import VARIANTS
from ablations.launch import native_runner, render_artifacts, runner_path, write_artifacts
from ablations.nodes import decision_node_class, mapping_node_class
from ablations.policies import (
    FlatCandidateCurator, FlatMemoryModelPolicy, NearestInteractionPolicy,
    without_outcome_continuations,
)
from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.candidate_curator import CandidateCuratorConfig
from semantic_decision_py_pkg.model_policy import ModelPolicyConfig
import run_benchmark_ablation as launcher
import run_benchmark_eval as baseline

REPO = Path(__file__).resolve().parents[4]


def candidate(name="door", distance=2.0, kind="INTERACT", **metadata):
    return BehaviorCandidate(
        candidate_id=name, behavior_type=kind, source="test", target_id=name,
        target_name=name, goal_xyyaw=[distance, 0., 0.],
        interaction_command={"action": "open", "node_id": name} if kind == "INTERACT" else None,
        features={"distance_m": distance, "target_relevance": 1.0},
        metadata={"node_type": "portal", "state": "closed", **metadata},
    )


def test_flat_request_removes_relations_but_preserves_object_semantics_and_executor():
    obj = candidate(connected_room_ids=["room_a", "room_b"], target_room_id="room_b",
                    interaction_effect="access_room", semantic_name="door", decision_hint="NEXT_ROUTE_PORTAL")
    graph = {"nodes": [{"id": "room_a", "type": "room", "label": "bedroom"},
                       {"id": "door", "type": "portal", "label": "door", "centroid": [1, 2, 0],
                        "interaction_state": "closed", "connected_room_ids": ["room_a", "room_b"]}],
             "edges": [{"src_id": "room_a", "dst_id": "room_b", "relation": "connects"}]}
    original = deepcopy((obj, graph))
    policy = FlatMemoryModelPolicy(ModelPolicyConfig())
    payload = policy.build_request([obj], {"enabled": True, "object_label": "apple"}, graph,
                                   {"candidate_pre_scores": {"door": 100},
                                    "decision_history": [{"candidate_id": "old", "status": "FAILED",
                                                          "room_id": "room_a", "result": {"room_id": "room_b"}}]})
    assert payload["mission"]["target"]["name"] == "apple"
    assert "interaction_state" not in payload["object_memory"][0]
    assert "state" not in payload["candidates"][0]
    assert payload["candidates"][0]["subject_semantic_type"] == "door"
    forbidden = {"graph", "edges", "effect", "connected_room_ids", "room_id", "room_object_reasoning",
                 "pre_score", "decision_hint", "candidate_pre_scores", "target_room_id"}

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()

    assert not keys(payload) & forbidden
    assert (obj, graph) == original


def test_flat_selection_uses_model_choice_without_graph_score_override(monkeypatch):
    policy = FlatMemoryModelPolicy(ModelPolicyConfig())
    near, far = candidate("near", 1), candidate("far", 5)
    received = []

    def request(payload, **kwargs):
        received.append(payload)
        return {"ranked_ids": ["far"], "reason": "INFORMATION_GAIN", "confidence": "high"}

    monkeypatch.setattr(policy, "_request", request)
    chosen = policy.select([near, far], graph={}, robot_context={
        "candidate_pre_scores": {"near": 9999, "far": -9999},
        "candidate_decision_hints": {"near": "NEXT_ROUTE_PORTAL"}})
    assert chosen is far
    assert chosen.interaction_command == {"action": "open", "node_id": "far"}
    assert received and not policy.last_pre_score_guard


def test_flat_memory_reaches_http_client_with_native_response_schema(monkeypatch):
    policy = FlatMemoryModelPolicy(ModelPolicyConfig(mode="http"))
    received = []

    def request_json(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(payload={"ranked_ids": ["door"], "reason": "UNLOCK_ROUTE", "confidence": "high"},
                               error="", metrics=lambda: {"request_count": 1})

    monkeypatch.setattr(policy._mllm_client, "request_json", request_json)
    chosen = policy.select([candidate()], graph={"nodes": [{"id": "door", "type": "portal", "label": "door"}]})
    assert chosen.candidate_id == "door"
    assert received[0]["context"]["object_memory"][0]["label"] == "door"
    assert "graph" not in received[0]["context"]
    assert received[0]["response_schema"]
    assert policy.last_metrics["request_count"] == 1
    assert policy.last_metrics["attempts"][0]["error"] == ""


def test_flat_curator_keeps_safety_and_quota_without_relational_ranking():
    curator = FlatCandidateCurator(CandidateCuratorConfig(interaction_quota=1))
    near, far = candidate("near", 1), candidate("far", 4, decision_hint="NEXT_ROUTE_PORTAL")
    unsafe = candidate("unsafe", .1)
    unsafe.goal_xyyaw = [float("nan"), 0, 0]
    before = deepcopy([near, far])
    result = curator.curate([far, unsafe, near], target_context={"enabled": True, "object_label": "apple"})
    assert result.candidates == [near]
    assert "unsafe" in result.rejected
    assert result.quality_by_id == result.decision_hint_by_id == {}
    assert [near, far] == before


def test_nearest_search_ignores_semantic_relevance_and_keeps_observed_goal():
    policy = NearestInteractionPolicy()
    near, far = candidate("near", 1), candidate("far", 5)
    near.features["target_relevance"] = -1000
    far.features["target_relevance"] = 1000
    frontier = candidate("frontier", .1, kind="EXPLORE")
    assert policy.select([far, frontier, near]) is near
    goal = candidate("goal", 10, kind="NAVIGATE", target_goal=True, target_reliably_observed=True)
    assert policy.select([far, near, goal]) is goal
    assert policy.select([]) is None


def test_nearest_ties_and_invalid_distances_are_deterministic():
    a, b = candidate("a", 1), candidate("b", 1)
    invalid = candidate("invalid", float("nan"))
    assert NearestInteractionPolicy().select([invalid, b, a]) is a


def test_outcome_filter_keeps_observed_static_doorway_and_does_not_mutate():
    event = candidate("event", post_interaction_traversal=True).to_dict()
    static = candidate("static", post_interaction_traversal=True, static_open_occ_confirmed=True).to_dict()
    snapshot = {"candidates": [event, static, candidate().to_dict()], "candidate_count": 3}
    before = deepcopy(snapshot)
    result = without_outcome_continuations(snapshot)
    assert [c["candidate_id"] for c in result["candidates"]] == ["static", "door"]
    assert result["candidate_count"] == 2 and snapshot == before


def test_mapping_only_suppresses_action_events_and_inherits_perception():
    class Mapper:
        def __init__(self):
            self.events = []

        def interaction_command_callback(self, msg):
            self.events.append("command")

        def interaction_result_callback(self, msg):
            self.events.append("result")

        def occupancy_callback(self, msg):
            self.events.append("occupancy")

        def gt_observation_callback(self, msg):
            self.events.append("observation")

    node = mapping_node_class(Mapper, "no_outcome_update")()
    node.interaction_command_callback({})
    node.interaction_result_callback({})
    node.occupancy_callback({})
    node.gt_observation_callback({})
    assert node.events == ["occupancy", "observation"]
    assert type(node).occupancy_callback is Mapper.occupancy_callback


@pytest.mark.parametrize("variant", VARIANTS)
def test_generated_launch_uses_only_explicit_wrappers_and_valid_shell(tmp_path, variant):
    artifacts = render_artifacts(REPO, tmp_path, variant, "/usr/bin/python3")
    if variant == "full":
        assert artifacts == {}
        assert runner_path(REPO, tmp_path, variant) == native_runner(REPO)
        assert decision_node_class(object, variant) is object
        return
    write_artifacts(artifacts)
    subprocess.run(["bash", "-n", str(tmp_path / "runner.sh")], check=True)
    nav = ET.fromstring(artifacts[tmp_path / "nav.launch"])
    assert any(inc.get("file") == str(tmp_path / "semantic_decision.launch") for inc in nav.iter("include"))
    decision = ET.fromstring(artifacts[tmp_path / "semantic_decision.launch"])
    wrapped = [node.get("type") for node in decision.iter("node") if node.get("launch-prefix")]
    expected = {"semantic_rule_decision_node.py"}
    if variant in {"no_interaction_graph", "no_outcome_update"}:
        expected.add("semantic_candidate_node.py")
    assert set(wrapped) == expected
    assert (tmp_path / "semantic_mapping_py.launch" in artifacts) == (variant == "no_outcome_update")
    assert "adapter_sha256=" in artifacts[tmp_path / "runner.sh"]
    assert f"SCRIPT_DIR={REPO}/scripts/InteractiveNav" in artifacts[tmp_path / "runner.sh"]
    assert 'module2: "mllm_score"' in artifacts[tmp_path / "runner.sh"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_dry_run_reuses_baseline_budget_and_writes_nothing(tmp_path, capsys, variant):
    output = tmp_path / "unused"
    assert launcher.main(["--variant", variant, "--workers", "1", "--max-steps", "20",
                          "--episode-indices", "10", "--output-dir", str(output), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    command, indices = baseline.build_command(payload["config"], output)
    assert indices == [10]
    if variant != "full":
        command += ["--runner", str(output / "ablation/runner.sh")]
    assert payload["command"] == command
    assert payload["config"]["paper_cost_budget"] == 30.0
    assert not output.exists()


def test_invalid_profile_and_incompatible_selection_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        render_artifacts(REPO, tmp_path, "misspelled", "/usr/bin/python3")
    with pytest.raises(ValueError):
        FlatMemoryModelPolicy(ModelPolicyConfig(selection_granularity="room"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_launcher_records_effective_variant_and_uses_owned_cleanup(tmp_path, monkeypatch, variant):
    launched, cleaned = [], []
    process = SimpleNamespace(poll=lambda: 0, wait=lambda **kwargs: 0, returncode=0)

    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return process

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def bind(self, *_):
            pass

    monkeypatch.setattr(launcher.socket, "socket", Socket)
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda args, **kwargs: "test-head\n" if "rev-parse" in args else "")
    monkeypatch.setattr(baseline, "check_mujoco_gpu_inventory", lambda config: ["0", "1"])
    monkeypatch.setattr(launcher, "check_model_endpoints", lambda config: None)
    monkeypatch.setattr(baseline, "cleanup", lambda owner, force: cleaned.append(owner))
    monkeypatch.setattr(baseline, "progress", lambda *_: "test progress")
    monkeypatch.setattr(baseline, "completion_report", lambda *a, **k: "test complete")
    output = tmp_path / "run"
    assert launcher.main(["--variant", variant, "--workers", "1", "--episode-indices", "10",
                          "--output-dir", str(output)]) == 0
    manifest = json.loads((output / "ablation_manifest.json").read_text())
    assert manifest["variant"] == variant and manifest["git_head"] == "test-head"
    assert cleaned == [str(output)]
    command, kwargs = launched[0]
    assert command == manifest["command"]
    assert kwargs["env"][baseline.OWNER_KEY] == str(output)
    assert kwargs["env"]["PAPER_COST_BUDGET"] == "30.0"
    assert Path(kwargs["env"]["TMPDIR"]).is_relative_to(output)
    assert ("--runner" in command) == (variant != "full")
    if variant != "full":
        assert Path(command[command.index("--runner") + 1]).is_file()
