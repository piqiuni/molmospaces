import json

import pytest

from semantic_mllm_py_pkg import client as client_module
from semantic_mllm_py_pkg.ablation import AblationConfig
from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig
from semantic_mllm_py_pkg.schemas import (
    validate_attribute_patch,
    validate_skill_action,
    validate_skill_plan,
    validate_subgoal_selection,
    validate_visual_interaction_plan,
    validate_visual_verification,
)


def test_ablation_modes_are_independent() -> None:
    config = AblationConfig("static_semantic", "mllm_score", "direct_atomic")
    assert config.to_dict() == {
        "module1": "static_semantic",
        "module2": "mllm_score",
        "module3": "direct_atomic",
    }
    assert config.uses_mllm


def test_schema_validation_rejects_unknown_candidate() -> None:
    with pytest.raises(ValueError):
        validate_subgoal_selection({"candidate_id": "bad"}, {"good"})


def test_subgoal_selection_accepts_ranked_ids() -> None:
    result = validate_subgoal_selection(
        {
            "ranked_ids": ["door", "frontier", "door"],
            "reason": "target_room",
            "confidence": "high",
        },
        {"door", "frontier"},
    )

    assert result["candidate_id"] == "door"
    assert result["ranked_ids"] == ["door", "frontier"]
    assert result["reason"] == "TARGET_ROOM"
    assert result["confidence"] == "high"


def test_role_schemas_normalize_outputs() -> None:
    attribute = validate_attribute_patch(
        {
            "object_id": "drawer_1",
            "interactable": True,
            "interaction_class": "container",
            "interaction_parts": [{"part_id": "top", "type": "drawer", "state": "closed"}],
            "confidence": 0.8,
        }
    )
    assert attribute["interaction_parts"][0]["part_id"] == "top"
    skill = validate_skill_plan(
        {"subactions": [{"skill": "open_part", "part_id": "top"}]}, "drawer_1"
    )
    assert skill["subactions"][0]["skill"] == "open_part"
    assert validate_skill_action(
        {"part_id": "top", "action": "open"}, {"top"}
    ) == {"part_id": "top", "action": "open"}
    verification = validate_visual_verification({"success": True})
    assert verification["success"] is True


def test_attribute_patch_uses_part_confidence_when_global_confidence_is_omitted() -> None:
    attribute = validate_attribute_patch(
        {
            "object_id": "fridge_1",
            "interactable": True,
            "interaction_class": "container",
            "coarse_state": "closed",
            "interaction_parts": [
                {
                    "part_id": "handle_1",
                    "type": "handle",
                    "state": "closed",
                    "handle_visible": True,
                    "confidence": 0.9,
                }
            ],
        }
    )

    assert attribute["confidence"] == 0.9


def test_mock_client_returns_role_payload() -> None:
    client = MLLMClient(MLLMClientConfig(mode="mock", model="mock"))
    response = client.request_json(
        role="subgoal_selection",
        instruction="select",
        context={"candidates": [{"candidate_id": "candidate_1"}]},
    )
    assert response.error == ""
    assert response.payload == {"candidate_id": "candidate_1"}
    assert response.tps >= 0.0


def test_visual_interaction_plan_normalizes_and_sorts_drawer_regions() -> None:
    plan = validate_visual_interaction_plan(
        {
            "target_type": "drawer",
            "action": "open",
            "operation_method": "pull",
            "open_regions": [
                {"center": [0.55, 0.78], "confidence": 1.4},
                {"x": 0.52, "y": 0.21, "confidence": 0.8},
                {"center": [0.53, 0.22], "confidence": 0.6},
                {"center": [2.0, 0.5], "confidence": 0.9},
            ],
            "confidence": 0.9,
        },
        expected_target_type="drawer_container",
    )

    assert plan["target_type"] == "drawer_container"
    assert plan["action"] == "scan"
    assert plan["operation_method"] == "pull"
    assert plan["open_regions"] == [
        {"center": [0.52, 0.21], "confidence": 0.8},
        {"center": [0.55, 0.78], "confidence": 1.0},
    ]


def test_visual_interaction_plan_uses_expected_type_when_image_is_ambiguous() -> None:
    plan = validate_visual_interaction_plan(
        {
            "target_type": "unknown",
            "operation_method": "unknown",
            "open_regions": [],
        },
        expected_target_type="door",
    )

    assert plan["target_type"] == "door"
    assert plan["operation_method"] == "unknown"


def test_openai_base_endpoint_is_resolved() -> None:
    client = MLLMClient(MLLMClientConfig(endpoint="http://localhost:8317/v1"))
    assert client._resolved_endpoint("openai_chat") == (
        "http://localhost:8317/v1/chat/completions"
    )


def test_reasoning_off_is_explicit_and_raw_http_response_is_retained(monkeypatch) -> None:
    captured = {}
    raw_response = (
        '{"output":[{"type":"message","content":[{"type":"output_text",'
        '"text":"{\\"candidate_id\\":\\"candidate_1\\"}"}]}],'
        '"usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8,'
        '"output_tokens_details":{"reasoning_tokens":0}}}'
    )

    class FakeResponse:
        def read(self):
            return raw_response.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request_object, timeout):
        captured["payload"] = json.loads(request_object.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(client_module.request, "urlopen", fake_urlopen)
    client = MLLMClient(
        MLLMClientConfig(
            mode="http",
            endpoint="http://localhost:8317/v1",
            model="vision-model",
            protocol="openai_responses",
            reasoning_effort="off",
        )
    )
    response = client.request_json(
        role="subgoal_selection",
        instruction="select",
        context={"candidates": [{"candidate_id": "candidate_1"}]},
    )

    assert captured["payload"]["reasoning"] == {"effort": "none"}
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "/no_think" in captured["payload"]["input"][0]["content"][0]["text"]
    assert response.payload == {"candidate_id": "candidate_1"}
    assert response.raw_http_response == raw_response
    assert response.usage["output_tokens"] == 5


def test_invalid_http_json_keeps_raw_text_and_usage(monkeypatch) -> None:
    raw_response = json.dumps(
        {
            "choices": [{"message": {"content": '{"part_id":"drawer_1"'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }
    )

    class FakeResponse:
        def read(self):
            return raw_response.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(client_module.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    response = MLLMClient(
        MLLMClientConfig(mode="http", endpoint="http://localhost:8317/v1")
    ).request_json(role="skill_planning", instruction="plan", context={})

    assert response.payload is None
    assert "invalid JSON response" in response.error
    assert response.raw_text == '{"part_id":"drawer_1"'
    assert response.completion_tokens == 5


def test_openai_chat_reasoning_off_uses_enable_thinking(monkeypatch) -> None:
    captured = {}
    raw_response = (
        '{"choices":[{"message":{"content":"{\\"candidate_id\\":\\"candidate_1\\"}"}}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}'
    )

    class FakeResponse:
        def read(self):
            return raw_response.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request_object, timeout):
        captured["payload"] = json.loads(request_object.data.decode("utf-8"))
        captured["endpoint"] = request_object.full_url
        return FakeResponse()

    monkeypatch.setattr(client_module.request, "urlopen", fake_urlopen)
    client = MLLMClient(
        MLLMClientConfig(
            mode="http",
            endpoint="http://localhost:8317/v1",
            model="vision-model",
            protocol="openai_chat",
            reasoning_effort="off",
        )
    )
    response = client.request_json(
        role="subgoal_selection",
        instruction="select",
        context={"candidates": [{"candidate_id": "candidate_1"}]},
    )

    assert captured["endpoint"] == "http://localhost:8317/v1/chat/completions"
    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["reasoning_effort"] == "none"
    assert "/no_think" in captured["payload"]["messages"][1]["content"][0]["text"]
    assert response.payload == {"candidate_id": "candidate_1"}
