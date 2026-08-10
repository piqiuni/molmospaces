from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
MAPPING_SCRIPTS = SRC_ROOT / "semantic_mapping_py_pkg" / "scripts"
DECISION_SCRIPTS = SRC_ROOT / "semantic_decision_py_pkg" / "scripts"
for path in (MAPPING_SCRIPTS, DECISION_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from semantic_decision_py_pkg.behavior_candidates import (
    CandidateGenerator,
    CandidateGeneratorConfig,
)
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


def test_minimal_gt_unknown_portal_defaults_to_rule_interaction_only() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            {
                "id": "double_door_root",
                "name": "Door",
                "bbox_2d": [0, 0, 19, 19],
                "segmentation": {
                    "rows": [index // 20 for index in range(400)],
                    "cols": [index % 20 for index in range(400)],
                },
                "box_3d": {
                    "center": [2.0, 0.0, 1.0],
                    "size": [0.2, 1.0, 2.0],
                    "frame_id": "world",
                },
            }
        ],
        source_mode="realtime_gt_observation",
    )

    graph = store.as_graph_dict()
    rule_candidates = CandidateGenerator(
        CandidateGeneratorConfig(portal_unknown_default_interact=True)
    ).generate(
        {"initial_scan_complete": True},
        graph,
        robot_xy=(0.0, 0.0),
    )
    interaction = next(
        candidate
        for candidate in rule_candidates
        if candidate.behavior_type == "INTERACT"
    )

    portal = next(
        node for node in graph["nodes"] if node["type"] == "portal"
    )
    public_id = portal["attributes"]["instance_id"]
    assert public_id != "double_door_root"
    assert public_id.startswith("door_")
    command = interaction.interaction_command or {}
    assert command["object_id"] == public_id
    assert command["action"] == "open"
    assert interaction.target_name == public_id
    assert command["node_id"] == public_id
    assert command["object_id"].startswith("door_")

    # The MLLM lane keeps the visual-evidence gate.  It does not infer a
    # closed/open state from restricted-GT geometry alone.
    model_candidates = CandidateGenerator(
        CandidateGeneratorConfig(portal_unknown_default_interact=False)
    ).generate(
        {"initial_scan_complete": True},
        graph,
        robot_xy=(0.0, 0.0),
    )
    assert not any(candidate.behavior_type == "INTERACT" for candidate in model_candidates)

    assert store.apply_attribute_patch(
        {
            "object_id": public_id,
            "attribute_status": "ready",
            "interactable": True,
            "interaction_class": "portal",
            "coarse_state": "closed",
            "confidence": 0.9,
            "source": "mllm_attribute_inference",
        }
    )
    candidates = CandidateGenerator(
        CandidateGeneratorConfig(portal_unknown_default_interact=False)
    ).generate(
        {"initial_scan_complete": True},
        store.as_graph_dict(),
        robot_xy=(0.0, 0.0),
    )
    interaction = next(candidate for candidate in candidates if candidate.behavior_type == "INTERACT")
    command = interaction.interaction_command or {}
    assert command["object_id"] == public_id
    assert command["action"] == "open"
    assert set(command) == {
        "node_id",
        "object_id",
        "action",
        "interaction_mode",
        "expected_state",
        "interaction_approach_pose_xyyaw",
        "interaction_approach_axis_xy",
        "interaction_approach_pose_labels",
        "interaction_ready_distance_m",
        "interaction_ready_yaw_tolerance_rad",
    }

    assert "joint_names" not in command
    assert "close_other_joint_names" not in command
    assert "close_other_joints" not in command
    assert interaction.target_name == public_id


def test_rule_portal_state_changes_only_after_public_executor_feedback() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            {
                "id": "door_x_private_root",
                "name": "Door",
                "bbox_2d": [0, 0, 19, 19],
                "segmentation": {
                    "rows": [index // 20 for index in range(400)],
                    "cols": [index % 20 for index in range(400)],
                },
                "box_3d": {
                    "center": [2.0, 0.0, 1.0],
                    "size": [0.2, 1.0, 2.0],
                    "frame_id": "world",
                },
            }
        ],
        source_mode="realtime_gt_observation",
    )
    graph = store.as_graph_dict()
    portal = next(node for node in graph["nodes"] if node["type"] == "portal")
    public_id = portal["attributes"]["instance_id"]
    assert portal["interaction"]["state"] == "unknown"
    assert portal["interaction"]["state_source"] == "unobserved"

    # A public executor result is the first rule-lane state evidence.  It
    # carries the opaque public ID, never the simulator's door body name.
    assert store.update_interaction_result(
        {
            "object_id": public_id,
            "node_id": portal["id"],
            "action": "open",
            "success": False,
            "state": "blocked",
            "interaction_capability": "blocked",
            "interactable": False,
            "source": "force_interaction_capability_check",
        }
    )
    updated = next(
        node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal"
    )
    assert updated["interaction"]["state"] == "blocked"
    assert updated["interaction"]["state_source"] == "force_interaction_capability_check"
    assert updated["interaction"]["is_interactable"] is False
    assert updated["interaction"]["requires_interaction"] is False
    assert "door_x" not in str(updated).casefold()

    candidates = CandidateGenerator(
        CandidateGeneratorConfig(portal_unknown_default_interact=True)
    ).generate(
        {"initial_scan_complete": True},
        store.as_graph_dict(),
        robot_xy=(0.0, 0.0),
    )
    assert not any(candidate.behavior_type == "INTERACT" for candidate in candidates)
