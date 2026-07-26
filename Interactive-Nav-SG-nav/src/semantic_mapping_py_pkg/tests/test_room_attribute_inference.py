from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.room_inference_backends import WeightedRoomAttributeInferencer


def test_weighted_object_types_choose_room_and_keep_evidence() -> None:
    inferencer = WeightedRoomAttributeInferencer(
        {
            "kitchen": {"stove": 1.0, "refrigerator": 0.9},
            "livingroom": {"sofa": 1.0, "tv": 0.7},
        }
    )

    result = inferencer.infer(
        [
            {"node_id": "sofa_1", "semantic_name": "sofa", "confidence": 1.0},
            {"node_id": "tv_1", "semantic_name": "tv", "confidence": 0.8},
        ]
    )

    assert result["room_attribute"] == "livingroom"
    assert result["confidence"] > 0.5
    assert result["scores"]["livingroom"] > 0.0
    assert result["evidence"][0]["object_label"] == "sofa"


def test_graph_store_infers_room_attribute_without_room_id_semantic_prior() -> None:
    store = InteractionGraphStore(
        scene_id="house_7",
        room_id_to_name={},
        object_room_priors={
            "kitchen": {"stove": 1.0, "refrigerator": 0.9},
            "bedroom": {"bed": 1.0},
        },
    )
    store.update_observations(
        [
            {
                "instance_id": "stove_1",
                "semantic_name": "stove",
                "category": "stove",
                "confidence": 1.0,
                "room_id": 0,
                "position": [1.0, 1.0, 0.5],
                "aabb_center": [1.0, 1.0, 0.5],
                "aabb_size": [0.5, 0.5, 1.0],
            }
        ],
        source_mode="realtime_gt_observation",
    )

    room = next(node for node in store.as_graph_dict()["nodes"] if node["type"] == "room")
    assert room["label"] == "room_0"
    assert room["attributes"]["room_attribute"] == "kitchen"
    assert room["attributes"]["room_attribute_source"] == "weighted_object_types"
    assert room["attributes"]["room_attribute_evidence"][0]["object_label"] == "stove"


def test_container_exposes_inferred_children_after_geometry_match() -> None:
    store = InteractionGraphStore(scene_id="house_7")
    store.update_observations(
        [
            {
                "instance_id": "fridge_1",
                "semantic_name": "fridge",
                "confidence": 1.0,
                "room_id": 1,
                "is_receptacle": True,
                "is_articulable": True,
                "position": [1.0, 1.0, 1.0],
                "aabb_center": [1.0, 1.0, 1.0],
                "aabb_size": [1.0, 1.0, 2.0],
            },
            {
                "instance_id": "apple_1",
                "semantic_name": "apple",
                "confidence": 1.0,
                "room_id": 1,
                "position": [1.0, 1.0, 1.5],
                "aabb_center": [1.0, 1.0, 1.5],
                "aabb_size": [0.1, 0.1, 0.1],
            },
        ],
        source_mode="realtime_gt_observation",
    )

    container = next(
        node
        for node in store.as_graph_dict()["nodes"]
        if node["type"] == "container"
    )
    assert container["attributes"]["inferred_child_count"] == 1
    assert container["attributes"]["inferred_child_ids"] == ["object_apple_1"]
