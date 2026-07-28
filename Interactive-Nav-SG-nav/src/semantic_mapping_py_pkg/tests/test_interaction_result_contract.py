from semantic_mapping_py_pkg.interaction_result_contract import (
    merge_interaction_result_with_command,
    take_pending_interaction_command,
)


def test_opaque_result_recovers_only_public_command_metadata() -> None:
    command = {
        "command_id": "decision_1:door_open",
        "decision_id": "decision_1",
        "candidate_id": "interaction:portal_1:open",
        "event_id": "interaction_1",
        "node_id": "portal_1",
        "object_id": "obj_1",
        "node_type": "portal",
        "action": "open",
        "interaction_mode": "open_close",
        "expected_state": "open",
        "approach_goal_xyyaw": [-1.0, 0.0, 0.0],
        "joint_infos": [{"joint_name": "private_hinge"}],
    }
    result = {
        "command_id": "decision_1:door_open",
        "success": True,
        "source": "evaluator_object_skill",
    }

    merged = merge_interaction_result_with_command(result, command)

    assert merged == {
        **result,
        "decision_id": "decision_1",
        "candidate_id": "interaction:portal_1:open",
        "event_id": "interaction_1",
        "node_id": "portal_1",
        "object_id": "obj_1",
        "node_type": "portal",
        "action": "open",
        "interaction_mode": "open_close",
        "expected_state": "open",
        "approach_goal_xyyaw": [-1.0, 0.0, 0.0],
    }
    assert "joint_infos" not in merged


def test_result_fields_take_precedence_over_command_metadata() -> None:
    merged = merge_interaction_result_with_command(
        {
            "command_id": "command_1",
            "node_id": "portal_result",
            "action": "close",
            "success": True,
        },
        {
            "command_id": "command_1",
            "node_id": "portal_command",
            "action": "open",
            "expected_state": "open",
        },
    )

    assert merged["node_id"] == "portal_result"
    assert merged["action"] == "close"
    assert merged["expected_state"] == "open"


def test_pending_command_matches_command_id_when_event_ids_differ() -> None:
    pending = {
        "decision_1:door_open": {
            "command_id": "decision_1:door_open",
            "event_id": "decision_interaction_001",
            "node_id": "portal_1",
        }
    }

    command = take_pending_interaction_command(
        pending,
        {
            "command_id": "decision_1:door_open",
            "event_id": "object_skill_001",
            "node_id": "portal_1",
        },
    )

    assert command is not None
    assert command["event_id"] == "decision_interaction_001"
    assert pending == {}


def test_pending_command_falls_back_to_latest_matching_node() -> None:
    pending = {
        "old": {"command_id": "old", "node_id": "portal_1"},
        "new": {"command_id": "new", "node_id": "portal_1"},
    }

    command = take_pending_interaction_command(
        pending,
        {"event_id": "opaque_result", "node_id": "portal_1"},
    )

    assert command == {"command_id": "new", "node_id": "portal_1"}
    assert list(pending) == ["old"]
