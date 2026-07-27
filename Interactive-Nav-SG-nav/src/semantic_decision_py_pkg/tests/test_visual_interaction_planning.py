from __future__ import annotations

import sys
from pathlib import Path


DECISION_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(DECISION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DECISION_SCRIPTS))


from semantic_decision_py_pkg.visual_interaction_planning import (
    action_for_opaque_open_contract,
    candidate_with_visual_operation_plan,
    infer_visual_interaction_target_type,
)


def test_dresser_visual_plan_becomes_joint_free_drawer_scan() -> None:
    candidate = {
        "target_name": "chestofdrawers_asset",
        "interaction_command": {
            "action": "open",
            "joint_names": ["legacy_joint"],
            "interaction_groups": [
                {"group_id": "legacy", "joint_names": ["legacy_joint"]}
            ],
        },
    }
    node = {"type": "container", "name": "Dresser"}
    plan = {
        "target_type": "drawer_container",
        "action": "scan",
        "operation_method": "pull",
        "open_regions": [
            {"center": [0.5, 0.2], "confidence": 0.9},
            {"center": [0.5, 0.8], "confidence": 0.8},
        ],
    }

    assert infer_visual_interaction_target_type(candidate, node) == "drawer_container"
    planned = candidate_with_visual_operation_plan(candidate, plan)
    command = planned["interaction_command"]
    assert command["action"] == "scan"
    assert command["sequence_type"] == "drawer_scan"
    assert command["open_regions"] == plan["open_regions"]
    assert "joint_names" not in command
    assert "interaction_groups" not in command
    assert "view_profile" not in command


def test_portal_visual_plan_preserves_atomic_open_and_method() -> None:
    candidate = {
        "target_name": "double_door_root",
        "interaction_command": {"action": "open"},
    }
    node = {"type": "portal", "name": "Door"}
    plan = {
        "target_type": "door",
        "action": "open",
        "operation_method": "double_hinged",
        "open_regions": [],
    }

    assert infer_visual_interaction_target_type(candidate, node) == "door"
    command = candidate_with_visual_operation_plan(candidate, plan)["interaction_command"]
    assert command["action"] == "open"
    assert command["operation_method"] == "double_hinged"
    assert command.get("sequence_type", "") == ""


def test_opaque_open_contract_clamps_only_when_explicitly_enabled() -> None:
    assert action_for_opaque_open_contract("scan", enabled=False) == "scan"
    assert action_for_opaque_open_contract("scan", enabled=True) == "open"
