import asyncio
import json

import pytest

from semantic_mllm_py_pkg import client as client_module
from semantic_mllm_py_pkg.ablation import AblationConfig
from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig
from semantic_mllm_py_pkg.schemas import (
    build_attribute_patch_response_schema,
    build_visual_interaction_plan_response_schema,
    build_visual_verification_response_schema,
    validate_attribute_patch,
    validate_room_attribute_patch,
    validate_skill_action,
    validate_skill_plan,
    validate_subgoal_selection,
    validate_visual_interaction_plan,
    validate_visual_verification,
)


def _patch_openai_stream(monkeypatch, raw_response: str, captured: dict | None = None):
    captured = {} if captured is None else captured

    def fake_stream(self, endpoint, body_payload, headers, *, timeout_s):
        captured["endpoint"] = endpoint
        captured["payload"] = dict(body_payload)
        captured["headers"] = dict(headers)
        captured["timeout"] = timeout_s
        return json.loads(raw_response), raw_response

    monkeypatch.setattr(MLLMClient, "_request_openai_chat_stream", fake_stream)
    return captured


def test_openai_chat_stream_timeout_closes_response(monkeypatch) -> None:
    events = {}

    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"{"}}]}'
            await asyncio.sleep(10.0)

        async def aclose(self):
            events["response_closed"] = True

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            events["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            events["client_closed"] = True

        def build_request(self, method, endpoint, **kwargs):
            events["request"] = {"method": method, "endpoint": endpoint, **kwargs}
            return object()

        async def send(self, _request, stream=False):
            events["stream"] = stream
            return FakeResponse()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    response = MLLMClient(
        MLLMClientConfig(
            mode="http",
            endpoint="http://127.0.0.1:8317/v1",
            timeout_s=0.05,
        )
    ).request_json(role="subgoal_selection", instruction="select", context={})

    assert response.payload is None
    assert response.error == "timed out"
    assert events["stream"] is True
    assert events["request"]["json"]["stream"] is True
    assert events["request"]["headers"]["Accept"] == "text/event-stream"
    assert events["response_closed"] is True
    assert events["client_closed"] is True


def test_openai_chat_stream_assembles_content_and_usage(monkeypatch) -> None:
    events = {}

    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"{\\"candidate_id\\":"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"\\"candidate_1\\"}"}}]}'
            yield (
                'data: {"choices":[],"usage":{"prompt_tokens":3,'
                '"completion_tokens":5,"total_tokens":8}}'
            )
            yield "data: [DONE]"
            await asyncio.sleep(10.0)

        async def aclose(self):
            events["response_closed"] = True

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            events["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            events["client_closed"] = True

        def build_request(self, method, endpoint, **kwargs):
            events["request"] = {"method": method, "endpoint": endpoint, **kwargs}
            return object()

        async def send(self, _request, stream=False):
            events["stream"] = stream
            return FakeResponse()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    response = MLLMClient(
        MLLMClientConfig(
            mode="http",
            endpoint="http://127.0.0.1:8317/v1",
            timeout_s=1.0,
        )
    ).request_json(role="subgoal_selection", instruction="select", context={})

    assert response.payload == {"candidate_id": "candidate_1"}
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 5
    assert response.total_tokens == 8
    assert events["stream"] is True
    assert events["response_closed"] is True
    assert events["client_closed"] is True


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


def test_attribute_patch_normalizes_model_interaction_labels() -> None:
    attribute = validate_attribute_patch(
        {
            "object_id": "door_1",
            "interactable": True,
            "interaction_class": "Door",
            "coarse_state": " Static Open ",
        }
    )
    assert attribute["interaction_class"] == "portal"
    assert attribute["coarse_state"] == "static_open"

    closed = validate_attribute_patch(
        {
            "object_id": "door_2",
            "interactable": True,
            "interaction_class": "PORTAL",
            "coarse_state": "Closed",
        }
    )
    assert closed["coarse_state"] == "closed"


def test_attribute_patch_keeps_visual_portal_morphology_separate_from_state() -> None:
    attribute = validate_attribute_patch(
        {
            "object_id": "door_3",
            "interactable": False,
            "interaction_class": "portal",
            "coarse_state": "unknown",
            "portal_morphology": {"door_leaf": "No Leaf", "confidence": 1.2},
            "portal_aperture_evidence": {
                "open_aperture": "Clear Gap",
                "confidence": 1.2,
            },
        }
    )

    assert attribute["portal_morphology"] == {
        "door_leaf": "absent",
        "confidence": 1.0,
    }
    assert attribute["portal_aperture_evidence"] == {
        "open_aperture": "visible",
        "confidence": 1.0,
    }


def test_attribute_patch_response_schema_binds_target_and_bounds_portal_evidence() -> None:
    schema = build_attribute_patch_response_schema("door_0001")
    body = schema["schema"]
    properties = body["properties"]

    assert schema["name"] == "attribute_inference"
    assert schema["strict"] is True
    assert body["additionalProperties"] is False
    assert properties["object_id"] == {
        "type": "string",
        "enum": ["door_0001"],
    }
    assert properties["portal_morphology"]["type"] == ["object", "null"]
    assert properties["portal_morphology"]["properties"]["door_leaf"]["enum"] == [
        "absent",
        "present",
        "unknown",
    ]
    assert properties["portal_aperture_evidence"]["properties"][
        "open_aperture"
    ]["enum"] == ["visible", "not_visible", "unknown"]
    assert properties["interaction_parts"]["maxItems"] == 1
    assert set(body["required"]) == {
        "object_id",
        "interactable",
        "interaction_class",
        "coarse_state",
        "portal_morphology",
        "portal_aperture_evidence",
        "view_state",
        "view_state_confidence",
        "front_surface_visible",
        "front_surface_confidence",
        "approach_ready",
        "needs_reobserve",
        "interaction_parts",
        "action_regions",
        "confidence",
    }
    with pytest.raises(ValueError, match="object_id"):
        build_attribute_patch_response_schema("")

    non_portal = validate_attribute_patch(
        {
            "object_id": "fridge_1",
            "interactable": True,
            "interaction_class": "container",
            "coarse_state": "closed",
            "portal_morphology": None,
            "portal_aperture_evidence": None,
            "interaction_parts": [],
            "confidence": 0.9,
        }
    )
    assert "portal_morphology" not in non_portal
    assert "portal_aperture_evidence" not in non_portal


def test_attribute_patch_front_view_fields_are_conservatively_normalized() -> None:
    side_view = validate_attribute_patch(
        {
            "object_id": "target",
            "interactable": True,
            "interaction_class": "container",
            "coarse_state": "closed",
            "view_state": "side",
            "view_state_confidence": 0.9,
            "front_surface_visible": True,
            "front_surface_confidence": 0.9,
            "approach_ready": True,
            "needs_reobserve": False,
        }
    )
    assert side_view["view_state"] == "side_or_back"
    assert side_view["approach_ready"] is False
    assert side_view["needs_reobserve"] is True

    front_view = validate_attribute_patch(
        {
            "object_id": "target",
            "interactable": True,
            "interaction_class": "container",
            "coarse_state": "closed",
            "view_state": "front",
            "view_state_confidence": 0.9,
            "front_surface_visible": True,
            "front_surface_confidence": 0.9,
            "approach_ready": True,
            "needs_reobserve": False,
        }
    )
    assert front_view["approach_ready"] is True
    assert front_view["needs_reobserve"] is False


def test_room_attribute_patch_is_separate_from_object_patch() -> None:
    room = validate_room_attribute_patch(
        {
            "room_id": "7",
            "room_attribute": "Kitchen",
            "confidence": 1.2,
            "evidence_object_ids": ["object_stove_1", 3],
        }
    )

    assert room == {
        "room_id": 7,
        "room_attribute": "kitchen",
        "confidence": 1.0,
        "evidence_object_ids": ["object_stove_1", "3"],
    }


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


def test_container_side_view_requires_reposition_and_drops_guessed_handle() -> None:
    plan = validate_visual_interaction_plan(
        {
            "target_type": "other_container",
            "action": "open",
            "operation_method": "pull",
            "view_state": "side_or_back",
            "approach_ready": True,
            "reposition_required": False,
            "open_regions": [{"center": [0.8, 0.5], "confidence": 0.9}],
            "confidence": 0.9,
            "reason": "side view",
        },
        expected_target_type="other_container",
    )

    assert plan["approach_ready"] is False
    assert plan["reposition_required"] is True
    assert plan["operation_method"] == "unknown"
    assert plan["open_regions"] == []


def test_visual_interaction_plan_schema_requires_frontality_contract() -> None:
    schema = build_visual_interaction_plan_response_schema()
    assert schema["name"] == "visual_interaction_plan"
    assert schema["strict"] is True
    assert schema["schema"]["required"] == [
        "target_type",
        "action",
        "operation_method",
        "view_state",
        "approach_ready",
        "reposition_required",
        "open_regions",
        "confidence",
        "reason",
    ]


def test_visual_verification_schema_is_strict_and_bounded() -> None:
    schema = build_visual_verification_response_schema()
    body = schema["schema"]
    properties = body["properties"]

    assert schema["name"] == "visual_verification"
    assert schema["strict"] is True
    assert body["additionalProperties"] is False
    assert body["required"] == [
        "success",
        "confidence",
        "reason",
        "observed_states",
        "new_contents_visible",
        "retry_action",
    ]
    assert properties["reason"]["maxLength"] == 96
    assert properties["observed_states"]["additionalProperties"] is False
    assert properties["observed_states"]["required"] == [
        "target_state",
        "visible_change",
    ]
    assert properties["retry_action"]["enum"] == [
        "none",
        "retry",
        "reposition",
        "rescan",
    ]


def test_visual_verification_schema_is_sent_as_openai_json_schema(monkeypatch) -> None:
    captured = {}
    raw_response = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"success":true,"confidence":0.9,"reason":"door open",'
                            '"observed_states":{"target_state":"open",'
                            '"visible_change":"yes"},"new_contents_visible":false,'
                            '"retry_action":"none"}'
                        )
                    }
                }
            ]
        }
    )

    _patch_openai_stream(monkeypatch, raw_response, captured)
    schema = build_visual_verification_response_schema()
    response = MLLMClient(
        MLLMClientConfig(
            mode="http",
            endpoint="http://localhost:8317/v1",
            protocol="openai_chat",
        )
    ).request_json(
        role="visual_verification",
        instruction="verify",
        context={"target": {}},
        response_schema=schema,
        max_tokens=256,
    )

    assert captured["payload"]["max_tokens"] == 256
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert captured["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": schema,
    }
    assert response.payload == {
        "success": True,
        "confidence": 0.9,
        "reason": "door open",
        "observed_states": {"target_state": "open", "visible_change": "yes"},
        "new_contents_visible": False,
        "retry_action": "none",
    }


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

    _patch_openai_stream(monkeypatch, raw_response)
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

    _patch_openai_stream(monkeypatch, raw_response, captured)
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


def test_openai_chat_uses_json_schema_when_supplied(monkeypatch) -> None:
    captured = {}
    raw_response = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"ranked_ids":["frontier:a"],"reason":"INFORMATION_GAIN","confidence":"high"}'
                    }
                }
            ]
        }
    )

    _patch_openai_stream(monkeypatch, raw_response, captured)
    schema = {
        "name": "subgoal_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ranked_ids": {"type": "array"},
                "reason": {"type": "string"},
                "confidence": {"type": "string"},
            },
            "required": ["ranked_ids", "reason", "confidence"],
        },
    }
    response = MLLMClient(
        MLLMClientConfig(
            mode="http",
            endpoint="http://localhost:8317/v1",
            protocol="openai_chat",
        )
    ).request_json(
        role="subgoal_selection",
        instruction="select",
        context={"candidates": [{"id": "frontier:a"}]},
        response_schema=schema,
    )

    assert captured["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": schema,
    }
    assert response.payload["ranked_ids"] == ["frontier:a"]
