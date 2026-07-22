import pytest

from semantic_mllm_py_pkg.ablation import AblationConfig
from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig
from semantic_mllm_py_pkg.schemas import (
    validate_attribute_patch,
    validate_skill_plan,
    validate_subgoal_selection,
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
    verification = validate_visual_verification({"success": True})
    assert verification["success"] is True


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


def test_openai_base_endpoint_is_resolved() -> None:
    client = MLLMClient(MLLMClientConfig(endpoint="http://localhost:8317/v1"))
    assert client._resolved_endpoint("openai_chat") == (
        "http://localhost:8317/v1/chat/completions"
    )
