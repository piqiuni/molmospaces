from semantic_mllm_py_pkg.schemas import validate_visual_verification


def test_visual_verification_accepts_list_observed_states() -> None:
    result = validate_visual_verification(
        {"success": True, "observed_states": ["door_open", "contents_visible"]}
    )
    assert result["success"] is True
    assert result["observed_states"]["observation_0"] == "door_open"
