from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


from semantic_mapping_py_pkg.graph_rules import normalize_observation
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


def minimal_observation(
    instance_id: str,
    name: str,
    center: list[float],
    size: list[float],
    pixels: int = 100,
) -> dict:
    side = max(1, int(pixels**0.5))
    rows = [index // side for index in range(pixels)]
    cols = [index % side for index in range(pixels)]
    return {
        "id": instance_id,
        "name": name,
        "bbox_2d": [0, 0, side - 1, side - 1],
        "segmentation": {"rows": rows, "cols": cols},
        "box_3d": {"center": center, "size": size, "frame_id": "world"},
    }


def test_minimal_gt_is_normalized_from_only_allowed_fields() -> None:
    normalized = normalize_observation(
        minimal_observation("double_door_root", "Door", [2.0, 1.0, 1.0], [0.2, 1.0, 2.0])
    )

    assert normalized["instance_id"] == "double_door_root"
    assert normalized["semantic_name"] == "door"
    assert normalized["aabb_center"] == [2.0, 1.0, 1.0]
    assert normalized["aabb_size"] == [0.2, 1.0, 2.0]
    assert normalized["visible_pixels"] == 100
    assert normalized["visible_fraction"] == 1.0
    assert normalized["minimal_gt_observation"] is True


def test_compact_minimal_gt_uses_explicit_visible_pixel_count() -> None:
    observation = minimal_observation(
        "apple_1", "Apple", [1.0, 2.0, 0.5], [0.1, 0.1, 0.1], pixels=1
    )
    observation.pop("segmentation")
    observation["bbox_2d"] = [0, 0, 9, 9]
    observation["visible_pixels"] = 64

    normalized = normalize_observation(observation)

    assert normalized["visible_pixels"] == 64
    assert normalized["visible_fraction"] == 0.64
    assert normalized["segmentation"] is None


def test_compact_minimal_gt_uses_explicit_visibility_summary() -> None:
    observation = minimal_observation(
        "apple_1", "Apple", [1.0, 2.0, 0.8], [0.1, 0.1, 0.1]
    )
    observation.pop("segmentation")
    observation["visible_pixels"] = 37
    observation["visible_fraction"] = 0.74

    normalized = normalize_observation(observation)

    assert normalized["visible_pixels"] == 37
    assert normalized["visible_fraction"] == 0.74


def test_minimal_gt_adapter_discards_legacy_graph_metadata() -> None:
    raw = minimal_observation(
        "chair_1", "Chair", [1.0, 2.0, 0.5], [0.5, 0.5, 1.0]
    )
    raw.update(
        {
            "semantic_name": "door",
            "category": "Door",
            "instance_id": "wrong_instance",
            "position": [99.0, 99.0, 99.0],
            "aabb_center": [99.0, 99.0, 99.0],
            "aabb_size": [9.0, 9.0, 9.0],
            "confidence": 0.01,
            "visible_pixels": 1,
            "parent": "cabinet_root",
            "children": ["chair_child"],
            "is_receptacle": True,
            "is_pickup_candidate": True,
            "is_articulable": True,
            "is_door": True,
            "is_movable_door": True,
            "joint_type": "hinge",
            "joint_range": [0.0, 1.0],
            "joint_value": 1.0,
            "joint_infos": [{"joint_name": "forbidden_hinge"}],
            "room_id": 7,
            "connected_room_ids": [7, 8],
        }
    )

    normalized = normalize_observation(raw)

    assert normalized["instance_id"] == "chair_1"
    assert normalized["semantic_name"] == "chair"
    assert normalized["category"] == "chair"
    assert normalized["position"] == [1.0, 2.0, 0.5]
    assert normalized["aabb_center"] == [1.0, 2.0, 0.5]
    assert normalized["aabb_size"] == [0.5, 0.5, 1.0]
    assert normalized["viz_aabb_center"] == [1.0, 2.0, 0.5]
    assert normalized["viz_aabb_size"] == [0.5, 0.5, 1.0]
    assert normalized["confidence"] == 1.0
    assert normalized["visible_pixels"] == 100
    assert normalized["parent"] is None
    assert normalized["children"] == []
    assert normalized["is_receptacle"] is False
    assert normalized["is_pickup_candidate"] is False
    assert normalized["is_articulable"] is False
    assert normalized["is_door"] is False
    assert normalized["is_movable_door"] is False
    assert normalized["joint_type"] == "none"
    assert normalized["joint_range"] == [0.0, 0.0]
    assert normalized["joint_value"] is None
    assert normalized["joint_infos"] == []
    assert normalized["room_id"] is None
    assert normalized["connected_room_ids"] == []


def test_graph_ignores_legacy_flags_parent_and_joint_state() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    chair = minimal_observation(
        "chair_1", "Chair", [1.0, 2.0, 0.5], [0.5, 0.5, 1.0]
    )
    chair.update(
        {
            "parent": "cabinet_root",
            "is_articulable": True,
            "is_door": True,
            "joint_type": "hinge",
            "joint_range": [0.0, 1.0],
            "joint_value": 1.0,
        }
    )
    door = minimal_observation(
        "door_1", "Door", [2.0, 2.0, 1.0], [0.2, 1.0, 2.0]
    )
    door.update(
        {
            "parent": "wall_root",
            "is_door": False,
            "is_articulable": False,
            "joint_type": "hinge",
            "joint_range": [0.0, 1.0],
            "joint_value": 1.0,
        }
    )

    store.update_observations(
        [chair, door], stamp=1.0, source_mode="realtime_gt_observation"
    )
    graph = store.as_graph_dict(stamp=1.0)
    chair_node = next(node for node in graph["nodes"] if node["id"] == "object_chair_1")
    door_node = next(node for node in graph["nodes"] if node["id"] == "portal_door_1")

    assert chair_node["type"] == "object"
    assert chair_node["interaction"]["is_interactable"] is False
    assert door_node["type"] == "portal"
    assert door_node["interaction"]["state"] == "unknown"
    assert door_node["interaction"]["requires_interaction"] is True
    forbidden = {"parent", "is_door", "is_articulable", "joint_infos"}
    assert forbidden.isdisjoint(chair_node["attributes"])
    assert forbidden.isdisjoint(door_node["attributes"])


def test_minimal_gt_builds_interactive_portal_without_joint_metadata() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    observation = minimal_observation(
        "double_door_root", "Door", [2.0, 1.0, 1.0], [0.2, 1.0, 2.0]
    )
    store.update_observations(
        [observation], stamp=1.0, source_mode="realtime_gt_observation"
    )
    store.update_observations(
        [observation], stamp=2.0, source_mode="realtime_gt_observation"
    )

    portal = next(
        node for node in store.as_graph_dict(stamp=2.0)["nodes"] if node["type"] == "portal"
    )
    assert portal["id"] == "portal_double_door_root"
    assert portal["interaction"]["is_interactable"] is True
    assert portal["interaction"]["interaction_mode"] == "open_close"
    assert portal["interaction"]["state"] == "unknown"
    assert portal["interaction"]["requires_interaction"] is True
    assert portal["attributes"]["consecutive_observations"] == 2
    forbidden = {
        "joint_infos",
        "observation_evidence",
        "asset_id",
        "object_id",
        "parent",
        "children",
        "is_articulable",
        "is_movable_door",
        "orientation",
        "interaction_approach_axis_xy",
    }
    assert forbidden.isdisjoint(portal["attributes"])


def test_minimal_gt_visibility_streak_is_computed_in_graph_store() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    observation = minimal_observation("fridge_1", "Fridge", [1.0, 1.0, 1.0], [1.0, 1.0, 2.0])
    store.update_observations([observation], source_mode="realtime_gt_observation")
    store.update_observations([], source_mode="realtime_gt_observation")
    store.update_observations([observation], source_mode="realtime_gt_observation")

    container = next(
        node for node in store.as_graph_dict()["nodes"] if node["type"] == "container"
    )
    assert container["attributes"]["consecutive_observations"] == 1
    assert container["attributes"]["max_consecutive_observations"] == 1


def test_minimal_gt_container_relation_is_inferred_from_3d_boxes() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    fridge = minimal_observation("fridge_1", "Fridge", [1.0, 1.0, 1.0], [1.0, 1.0, 2.0])
    can = minimal_observation("can_1", "Soda Can", [1.0, 1.0, 1.0], [0.05, 0.05, 0.1])
    store.update_observations(
        [fridge, can], source_mode="realtime_gt_observation"
    )

    graph = store.as_graph_dict()
    assert any(
        edge["src_id"] == "container_fridge_1"
        and edge["relation"] == "contains"
        and edge["dst_id"] == "object_can_1"
        for edge in graph["edges"]
    )


def test_executor_semantic_result_updates_graph_by_object_id() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    observation = minimal_observation(
        "double_door_root", "Door", [2.0, 1.0, 1.0], [0.2, 1.0, 2.0]
    )
    store.update_observations([observation], source_mode="realtime_gt_observation")
    assert store.update_interaction_result(
        {
            "object_id": "double_door_root",
            "state": "open",
            "success": True,
            "verification_source": "executor_state_verification",
        }
    )

    portal = next(
        node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal"
    )
    assert portal["interaction"]["state"] == "open"
    assert portal["interaction"]["completed_interaction_groups"] == []
    assert "joint_infos" not in portal["attributes"]
    assert "joint_interaction_states" not in portal["interaction"]
    assert "joint_open_fractions" not in portal["interaction"]


def test_joint_only_executor_result_cannot_infer_graph_state() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            minimal_observation(
                "door_1", "Door", [2.0, 1.0, 1.0], [0.2, 1.0, 2.0]
            )
        ],
        source_mode="realtime_gt_observation",
    )

    assert store.update_interaction_result(
        {
            "node_id": "portal_door_1",
            "success": True,
            "joint_infos": [
                {
                    "joint_name": "hinge_1",
                    "joint_type": "hinge",
                    "joint_range": [0.0, 1.0],
                    "joint_value": 1.0,
                }
            ],
            "verification_source": "mujoco_joint_readback",
        }
    )

    portal = next(
        node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal"
    )
    assert portal["interaction"]["state"] == "unknown"
    assert portal["interaction"]["state_source"] == "semantic_graph_default"
    assert "joint_infos" not in portal["attributes"]


def test_minimal_gt_id_routes_mllm_attribute_patch() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            minimal_observation(
                "double_door_root",
                "Door",
                [2.0, 1.0, 1.0],
                [0.2, 1.0, 2.0],
            )
        ],
        stamp=1.0,
        source_mode="realtime_gt_observation",
    )
    assert store.apply_attribute_patch(
        {
            "object_id": "double_door_root",
            "attribute_status": "ready",
            "interactable": True,
            "interaction_class": "portal",
            "coarse_state": "closed",
            "expected_effect": "unlock_connectivity",
            "confidence": 0.9,
            "interaction_parts": [],
            "source": "mllm_attribute_inference",
        },
        stamp=2.0,
    )

    portal = next(
        node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal"
    )
    assert portal["interaction"]["state"] == "closed"
    assert portal["interaction"]["state_source"] == "mllm_attribute_inference"
    assert portal["attributes"]["attribute_status"] == "ready"
