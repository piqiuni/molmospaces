"""Cross-package contract test for V3 restricted GT and semantic object goals."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_SOURCE_ROOT = REPOSITORY_ROOT / "Interactive-Nav-SG-nav" / "src"
for package_scripts in (
    SEMANTIC_SOURCE_ROOT / "semantic_mapping_py_pkg" / "scripts",
    SEMANTIC_SOURCE_ROOT / "semantic_decision_py_pkg" / "scripts",
):
    if str(package_scripts) not in sys.path:
        sys.path.insert(0, str(package_scripts))


from scripts.InteractiveNav.evaluation.ros_object_goal_adapter import (
    adapt_restricted_gt_frame_for_semantic_mapping,
    adapt_restricted_gt_frame_for_legacy_mapping,
    adapt_strict_perception_payload_for_semantic_mapping,
    validate_semantic_minimal_perception_payload,
)
from semantic_decision_py_pkg.behavior_candidates import CandidateGenerator
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


def test_v3_restricted_frame_generates_an_opaque_semantic_open_command() -> None:
    frame = {
        "protocol_version": "interactive_nav_v3_restricted_gt_v1",
        "episode_id": "episode_000001",
        "episode_reset": False,
        "frame_index": 7,
        "observations": [
            {
                "instance_id": "obj_000001",
                "name": "door",
                "bbox_2d_xyxy": [0, 0, 3, 3],
                "mask_rle": {"size": [4, 4], "counts": [0, 16]},
                "bbox_3d": {
                    "center": [2.0, 0.0, 1.0],
                    "size": [0.2, 1.0, 2.0],
                    "frame_id": "world",
                },
            }
        ],
    }

    wire = adapt_restricted_gt_frame_for_semantic_mapping(frame, stamp_sec=1.0)
    validate_semantic_minimal_perception_payload(wire)
    observation = wire["observations"][0]
    assert set(observation) == {"id", "name", "bbox_2d", "mask_rle", "box_3d"}

    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        wire["observations"], stamp=1.0, source_mode="realtime_gt_observation"
    )
    graph = store.as_graph_dict(stamp=1.0)
    portal = next(node for node in graph["nodes"] if node["type"] == "portal")
    assert portal["confidence"] == 1.0
    assert portal["interaction"]["is_interactable"] is True

    candidates = CandidateGenerator().generate(
        {"initial_scan_complete": True}, graph, robot_xy=(0.0, 0.0)
    )
    interaction = next(
        candidate for candidate in candidates if candidate.behavior_type == "INTERACT"
    )
    assert interaction.interaction_command == {
        "node_id": "portal_obj_000001",
        "object_id": "obj_000001",
        "action": "open",
        "interaction_mode": "open_close",
        "expected_state": "open",
    }


def test_strict_detector_payload_uses_the_same_compact_semantic_wire_schema() -> None:
    wire = adapt_strict_perception_payload_for_semantic_mapping(
        {
            "schema_version": 1,
            "episode_id": "episode_000001",
            "episode_reset": False,
            "capture_step": 2,
            "stamp_sec": 2.0,
            "observations": [
                {
                    "id": "obj_000001",
                    "name": "cabinet",
                    "bbox_2d_xyxy": [0, 0, 1, 1],
                    "segmentation_rle": {"size": [2, 2], "counts": [0, 4]},
                    "box3d_center": [1.0, 0.0, 1.0],
                    "box3d_size": [1.0, 1.0, 2.0],
                }
            ],
        }
    )

    validate_semantic_minimal_perception_payload(wire)
    assert wire["observations"] == [
        {
            "id": "obj_000001",
            "name": "cabinet",
            "bbox_2d": [0, 0, 1, 1],
            "mask_rle": {"size": [2, 2], "counts": [0, 4]},
            "box_3d": {
                "center": [1.0, 0.0, 1.0],
                "size": [1.0, 1.0, 2.0],
                "frame_id": "world",
            },
        }
    ]


def test_legacy_adapter_remains_mask_rle_compatible_without_dense_decode() -> None:
    frame = {
        "protocol_version": "interactive_nav_v3_restricted_gt_v1",
        "episode_id": "episode_000001",
        "episode_reset": False,
        "frame_index": 1,
        "observations": [
            {
                "instance_id": "obj_000001",
                "name": "door",
                "bbox_2d_xyxy": [0, 0, 1, 1],
                "mask_rle": {"size": [2, 2], "counts": [0, 4]},
                "bbox_3d": {
                    "center": [0.0, 0.0, 1.0],
                    "size": [0.2, 1.0, 2.0],
                    "frame_id": "world",
                },
            }
        ],
    }

    legacy = adapt_restricted_gt_frame_for_legacy_mapping(frame, stamp_sec=1.0)

    assert legacy["observations"][0]["visible_pixels"] == 4
