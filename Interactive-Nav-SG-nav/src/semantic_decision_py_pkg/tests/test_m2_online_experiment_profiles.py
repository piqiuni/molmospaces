from copy import deepcopy
import hashlib

import pytest

from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.env_config import apply_model_env_overrides
from semantic_decision_py_pkg.model_policy import ModelPolicyClient, ModelPolicyConfig
from semantic_mllm_py_pkg.client import MLLMResponse


def _inputs():
    graph = {
        "frame_id": "map",
        "nodes": [
            {"id": "room_1", "type": "room", "label": "bedroom",
             "centroid": [5, 0, 0], "aabb_center": [0, 0, 0], "aabb_size": [4, 4, 3],
             "attributes": {"room_attribute": "bedroom", "room_attribute_confidence": 0.95}},
            {"id": "room_2", "type": "room", "label": "kitchen",
             "centroid": [0, 0, 0], "aabb_center": [10, 0, 0], "aabb_size": [4, 4, 3]},
            {"id": "bed_1", "type": "object", "label": "bed", "room_id": 1,
             "centroid": [1, 0, 0], "aabb_size": [2, 1, 1], "is_currently_visible": False},
            {"id": "refrigerator_1", "type": "container", "label": "refrigerator",
             "room_id": 2, "centroid": [10, 0, 0],
             "interaction_state": "closed", "is_interactable": True},
        ],
    }
    candidate = BehaviorCandidate(
        candidate_id="frontier:1", behavior_type="EXPLORE", source="test",
        target_id="frontier_1", target_name="frontier", features={"distance_m": 2.0},
        metadata={"room_id": 2, "robot_room_id": 2,
                  "room_status": "unentered_high_confidence_mismatch",
                  "room_target_affinity": -0.5, "room_target_affinity_reason": "kitchen_vs_bed",
                  "nearby_semantic_nodes": [{"type": "bed", "distance_m": 1, "visible": False}]},
    )
    robot = {
        "robot_xy": [10, 0], "position_frame_id": "odom",
        "robot_graph_xy": [0, 0], "robot_graph_frame_id": "map",
        "initial_xy": [0, 0], "initial_position_source": "first_observed_graph_pose",
        "room_visit_history": [{"room_id": 1}, {"room_id": 2}, {"room_id": 1}],
        "entered_room_ids": [2, 1, 2],
        "decision_history": [{"candidate_id": str(step), "step": step} for step in range(35)],
        "candidate_pre_scores": {"frontier:1": 0.75},
        "candidate_pre_score_terms": {"frontier:1": {"distance": 0.25}},
        "candidate_decision_hints": {"frontier:1": "NEW_ROOM_FRONTIER_HIGH_CONFIDENCE_MISMATCH"},
    }
    return [candidate], {"enabled": True, "target_name": "bed"}, graph, robot


def test_default_public_profile_is_unchanged_and_keeps_30_events():
    default = ModelPolicyClient().build_request(*_inputs())
    explicit = ModelPolicyClient(ModelPolicyConfig(context_profile="public_facts_v2", recent_decision_limit=30)).build_request(*_inputs())
    assert default == explicit
    assert len(default["recent_decisions"]) == 30
    assert default["recent_decisions"][0]["step"] == 5
    assert default["robot"]["current_room"] == "room_1"
    assert default["robot"]["current_xy"] == [0, 0]
    assert default["room_object_reasoning"] == {}
    assert "pre_score" not in default["candidates"][0]
    assert "visible" not in default["candidates"][0]["nearby_semantic_nodes"][0]


def test_historical_profile_restores_legacy_context_without_mutating_execution_inputs():
    args = _inputs()
    before = deepcopy(args)
    client = ModelPolicyClient(ModelPolicyConfig(context_profile="historical_compat_v1", recent_decision_limit=8))
    result = client.build_request(*args)
    assert args == before
    assert result["robot"] == {"current_room": "room_1", "entered_rooms": ["room_1", "room_2"]}
    assert result["graph"]["current_room"] == "room_1"
    assert len(result["recent_decisions"]) == 8
    assert result["recent_decisions"][0]["step"] == 27
    room = result["graph"]["rooms"][0]
    assert room["anchor_objects"] == [{"type": "bed", "visible": False}]
    assert "centroid_xy" not in room and "eligible_frontier_length_m" not in room
    assert "center_xy" not in result["graph"]["containers"][0]
    assert result["graph"]["containers"][0]["state"] == "closed"
    assert result["room_object_reasoning"]
    candidate = result["candidates"][0]
    assert candidate["id"] == "frontier:1"
    assert candidate["pre_score"] == 0.75
    assert candidate["room_target_affinity"] == -0.5
    assert candidate["decision_hint"] == "NEW_ROOM_FRONTIER_HIGH_CONFIDENCE_MISMATCH"
    assert candidate["nearby_semantic_nodes"][0]["visible"] is False
    assert candidate["robot_room_id"] == "room_1"


@pytest.mark.parametrize("profile", ["public_facts_v2", "historical_compat_v1"])
def test_profile_never_falls_back_to_untransformed_odom_when_tf_missing(profile):
    args = _inputs()
    args[3]["robot_graph_xy"] = None
    result = ModelPolicyClient(ModelPolicyConfig(context_profile=profile)).build_request(*args)
    assert "current_room" not in result["robot"]
    assert "current_room" not in result["graph"]
    assert "robot_room_id" not in result["candidates"][0]


def test_legacy_truncation_is_local_to_model_projection():
    args = _inputs()
    args[2]["nodes"] = [{"id": f"small_{i}", "type": "object", "label": "small"} for i in range(80)] + args[2]["nodes"]
    historical = ModelPolicyClient(ModelPolicyConfig(context_profile="historical_compat_v1")).build_request(*args)
    public = ModelPolicyClient().build_request(*args)
    assert historical["graph"]["rooms"] == []
    assert historical["graph"]["current_room"] == "room_1"
    assert len(public["graph"]["rooms"]) == 2
    assert len(args[2]["nodes"]) == 84
    assert [c["id"] for c in historical["candidates"]] == [c["id"] for c in public["candidates"]]


@pytest.mark.parametrize("limit", [0, 8, 30])
def test_recent_history_limit_is_explicit(limit):
    result = ModelPolicyClient(ModelPolicyConfig(recent_decision_limit=limit)).build_request(*_inputs())
    assert len(result["recent_decisions"]) == limit


def test_prompt_file_is_loaded_once_and_sha_is_logged_not_in_model_context(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("  Frozen instruction\n", encoding="utf-8")
    original = prompt.read_bytes()
    client = ModelPolicyClient(ModelPolicyConfig(mode="http", prompt_file=str(prompt)))
    prompt.write_text("Changed after initialization", encoding="utf-8")
    payload = client.build_request(*_inputs())
    assert payload["instruction"] == "Frozen instruction"
    assert set(payload) == {"schema_version", "instruction", "mission", "robot", "recent_decisions", "graph", "room_object_reasoning", "candidates"}
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return MLLMResponse(payload={"ranked_ids": ["frontier:1"], "reason": "INFORMATION_GAIN", "confidence": "low"}, latency_s=0.001)

    monkeypatch.setattr(client._mllm_client, "request_json", fake_request)
    client._request(payload)
    meta = captured["metrics_context"]
    assert meta["m2_context_profile"] == "public_facts_v2"
    assert meta["m2_recent_decision_limit"] == 30
    assert meta["m2_prompt_sha256"] == hashlib.sha256(b"Frozen instruction").hexdigest()
    assert meta["m2_prompt_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert client.last_metrics["m2_prompt_sha256"] == meta["m2_prompt_sha256"]


@pytest.mark.parametrize("config", [
    ModelPolicyConfig(context_profile="typo"),
    ModelPolicyConfig(recent_decision_limit=-1),
    ModelPolicyConfig(recent_decision_limit=31),
    ModelPolicyConfig(context_profile="historical_compat_v1", selection_granularity="room"),
])
def test_invalid_profile_configuration_fails_before_any_request(config):
    with pytest.raises(ValueError):
        ModelPolicyClient(config)


def test_missing_or_empty_prompt_fails_before_any_request(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelPolicyClient(ModelPolicyConfig(prompt_file=str(tmp_path / "missing.txt")))
    empty = tmp_path / "empty.txt"
    empty.write_text("\n ")
    with pytest.raises(ValueError, match="empty"):
        ModelPolicyClient(ModelPolicyConfig(prompt_file=str(empty)))


def test_experiment_environment_contract(monkeypatch):
    values = {
        "CONTEXT_PROFILE": "historical_compat_v1", "PROMPT_FILE": "/home/ldl/prompt.txt",
        "RECENT_DECISION_LIMIT": "8", "CANDIDATE_POOL_MODE": "legacy",
        "CANDIDATE_TOP_K": "8", "CANDIDATE_MAX_FRONTIER_CANDIDATES": "12",
    }
    for key, value in values.items():
        monkeypatch.setenv("SEMANTIC_M2_" + key, value)
    config = apply_model_env_overrides({})
    assert config["context_profile"] == "historical_compat_v1"
    assert config["prompt_file"] == "/home/ldl/prompt.txt"
    assert config["recent_decision_limit"] == 8
    assert config["candidate_pool_mode"] == "legacy"
    assert config["candidate_top_k"] == 8
    assert config["candidate_max_frontier_candidates"] == 12


def test_invalid_experiment_numeric_environment_fails_fast(monkeypatch):
    monkeypatch.setenv("SEMANTIC_M2_RECENT_DECISION_LIMIT", "eight")
    with pytest.raises(ValueError, match="SEMANTIC_M2_RECENT_DECISION_LIMIT"):
        apply_model_env_overrides({})


def test_legacy_projection_keeps_portal_redaction_and_potential_room_facts():
    args = _inputs()
    args[2]["nodes"].extend([
        {"id": "portal_doorframe_private_asset", "type": "portal", "label": "private_asset_name",
         "centroid": [2, 0, 0], "attributes": {"private_target_position": [9, 9], "connected_room_ids": [1, 3]},
         "interaction": {"state": "open", "is_interactable": True}},
        {"id": "room_3", "type": "room", "label": "unobserved_portal_room",
         "attributes": {"is_potential_room": True, "observed_free_space": False,
                        "source_portal_id": "portal_doorframe_private_asset", "private_target_position": [9, 9]}},
    ])
    result = ModelPolicyClient(ModelPolicyConfig(context_profile="historical_compat_v1")).build_request(*args)
    import json
    serialized = json.dumps(result)
    assert "private_asset" not in serialized
    assert "private_target_position" not in serialized
    portal = result["graph"]["portals"][0]
    room = next(room for room in result["graph"]["rooms"] if room["id"] == "room_3")
    assert portal["state"] == "open"
    assert room["potential_room"] is True and room["observed_free_space"] is False
    assert room["source_portal_id"] == portal["id"]
