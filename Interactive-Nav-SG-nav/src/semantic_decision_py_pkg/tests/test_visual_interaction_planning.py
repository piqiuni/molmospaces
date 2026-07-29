from __future__ import annotations

import sys
from pathlib import Path


DECISION_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(DECISION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DECISION_SCRIPTS))


from semantic_decision_py_pkg.visual_interaction_planning import (
    action_for_opaque_open_contract,
    candidate_with_direct_drawer_scan,
    candidate_with_visual_operation_plan,
    current_visible_bbox_2d,
    current_visible_bbox_capture_step,
    fresh_direct_drawer_scan_candidate,
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


def test_current_visible_drawer_uses_public_container_box_for_direct_scan() -> None:
    candidate = {
        "target_name": "dresser_asset",
        "interaction_command": {"action": "open", "joint_names": ["private_joint"]},
    }
    node = {
        "type": "container",
        "name": "Dresser",
        "is_currently_visible": True,
        "attributes": {"projected_bbox_2d": [50, 30, 10, 90]},
    }

    assert current_visible_bbox_2d(node) == [10.0, 30.0, 50.0, 90.0]
    planned = candidate_with_direct_drawer_scan(candidate, node, capture_step=17)

    assert planned is not None
    command = planned["interaction_command"]
    assert command["action"] == "scan"
    assert command["sequence_type"] == "drawer_scan"
    assert command["open_regions"] == []
    assert command["drawer_container_bbox_2d"] == [10.0, 30.0, 50.0, 90.0]
    assert command["drawer_container_capture_step"] == 17
    assert "joint_names" not in command


def test_direct_drawer_scan_requires_a_current_valid_box() -> None:
    candidate = {"interaction_command": {"action": "open"}}
    assert candidate_with_direct_drawer_scan(
        candidate,
        {"is_currently_visible": False, "bbox_2d": [0, 0, 20, 20]},
        capture_step=4,
    ) is None
    assert candidate_with_direct_drawer_scan(
        candidate,
        {"is_currently_visible": True, "bbox_2d": [0, 0, 0, 20]},
        capture_step=4,
    ) is None
    assert candidate_with_direct_drawer_scan(
        candidate,
        {"is_currently_visible": True, "bbox_2d": [0, 0, 20, 20]},
        capture_step=None,
    ) is None


def test_direct_drawer_scan_uses_only_a_post_arrival_current_public_frame() -> None:
    candidate = {
        "target_id": "container_obj_000016",
        "interaction_command": {"action": "open"},
    }
    node = {
        "id": "container_obj_000016",
        "is_currently_visible": True,
        "attributes": {
            "bbox_2d": [10, 20, 110, 160],
            "last_observation_frame_index": 42,
        },
    }

    assert current_visible_bbox_capture_step(node) == 42
    planned, reason = fresh_direct_drawer_scan_candidate(
        candidate,
        node,
        graph_capture_step=42,
        graph_revision=19,
        minimum_graph_capture_step=41,
        minimum_graph_revision=18,
        rgb_image_sequence=53,
        minimum_rgb_image_sequence=52,
        rgb_capture_step=42,
        minimum_rgb_capture_step=41,
    )

    assert reason == "ready"
    assert planned is not None
    assert planned["interaction_command"]["drawer_container_capture_step"] == 42
    assert planned["interaction_command"]["drawer_container_bbox_2d"] == [
        10.0,
        20.0,
        110.0,
        160.0,
    ]


def test_direct_drawer_scan_rejects_stale_or_nonmatching_public_bbox_frame() -> None:
    candidate = {"interaction_command": {"action": "open"}}
    node = {
        "is_currently_visible": True,
        "attributes": {
            "bbox_2d": [10, 20, 110, 160],
            "last_observation_frame_index": 41,
        },
    }
    kwargs = {
        "graph_capture_step": 42,
        "graph_revision": 19,
        "minimum_graph_capture_step": 41,
        "minimum_graph_revision": 18,
        "rgb_image_sequence": 53,
        "minimum_rgb_image_sequence": 52,
        "rgb_capture_step": 42,
        "minimum_rgb_capture_step": 41,
    }

    planned, reason = fresh_direct_drawer_scan_candidate(candidate, node, **kwargs)
    assert planned is None
    assert reason == "target_not_observed_in_current_capture"

    node["attributes"]["last_observation_frame_index"] = 42
    planned, reason = fresh_direct_drawer_scan_candidate(
        candidate,
        node,
        **{**kwargs, "graph_capture_step": 41},
    )
    assert planned is None
    assert reason == "graph_capture_not_fresh"

    planned, reason = fresh_direct_drawer_scan_candidate(
        candidate,
        node,
        **{**kwargs, "rgb_image_sequence": 52},
    )
    assert planned is None
    assert reason == "rgb_image_not_fresh"


def test_model_drawer_plan_cannot_inherit_a_direct_scan_box() -> None:
    candidate = {
        "interaction_command": {"drawer_container_bbox_2d": [1, 2, 3, 4]},
    }
    planned = candidate_with_visual_operation_plan(
        candidate,
        {
            "target_type": "drawer_container",
            "action": "scan",
            "operation_method": "pull",
            "open_regions": [],
        },
    )
    assert "drawer_container_bbox_2d" not in planned["interaction_command"]
