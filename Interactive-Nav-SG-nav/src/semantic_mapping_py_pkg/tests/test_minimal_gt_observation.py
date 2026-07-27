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


def test_executor_joint_readback_is_reduced_to_semantic_result() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    observation = minimal_observation(
        "double_door_root", "Door", [2.0, 1.0, 1.0], [0.2, 1.0, 2.0]
    )
    store.update_observations([observation], source_mode="realtime_gt_observation")
    assert store.update_interaction_result(
        {
            "node_id": "portal_double_door_root",
            "interaction_group_id": "all_joints",
            "state": "open",
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
    assert portal["interaction"]["state"] == "open"
    assert portal["interaction"]["completed_interaction_groups"] == ["all_joints"]
    assert "joint_infos" not in portal["attributes"]
    assert "joint_interaction_states" not in portal["interaction"]
    assert "joint_open_fractions" not in portal["interaction"]


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
