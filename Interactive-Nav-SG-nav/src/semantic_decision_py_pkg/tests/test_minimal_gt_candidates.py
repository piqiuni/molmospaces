from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
MAPPING_SCRIPTS = SRC_ROOT / "semantic_mapping_py_pkg" / "scripts"
DECISION_SCRIPTS = SRC_ROOT / "semantic_decision_py_pkg" / "scripts"
for path in (MAPPING_SCRIPTS, DECISION_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from semantic_decision_py_pkg.behavior_candidates import CandidateGenerator
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


def test_minimal_gt_portal_generates_id_only_interaction_command() -> None:
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

    candidates = CandidateGenerator().generate(
        {"initial_scan_complete": True},
        store.as_graph_dict(),
        robot_xy=(0.0, 0.0),
    )
    interaction = next(candidate for candidate in candidates if candidate.behavior_type == "INTERACT")
    command = interaction.interaction_command or {}
    assert command["object_id"] == "double_door_root"
    assert command["action"] == "open"
    assert set(command) == {
        "node_id",
        "object_id",
        "action",
        "interaction_mode",
        "expected_state",
    }

    assert "joint_names" not in command
    assert "close_other_joint_names" not in command
    assert "close_other_joints" not in command
