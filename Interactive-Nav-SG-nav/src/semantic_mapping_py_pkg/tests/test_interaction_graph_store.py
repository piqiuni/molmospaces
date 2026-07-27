from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.gt_observation_provider import build_gt_observation_batches
from semantic_mapping_py_pkg.graph_rules import observation_from_detection
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore


def observation(**kwargs):
    base = {
        "observation_id": "obs",
        "instance_id": "",
        "semantic_name": "object",
        "category": "object",
        "confidence": 1.0,
        "position": [0.0, 0.0, 0.0],
        "aabb_center": [0.0, 0.0, 0.0],
        "aabb_size": [0.1, 0.1, 0.1],
        "room_id": None,
        "parent": None,
        "children": [],
        "is_receptacle": False,
        "is_pickup_candidate": False,
        "is_articulable": False,
        "is_door": False,
        "is_movable_door": False,
        "joint_type": "none",
        "joint_range": [0.0, 0.0],
        "joint_value": None,
        "source": "test",
        "name": kwargs.get("instance_id", kwargs.get("semantic_name", "object")),
    }
    base.update(kwargs)
    return base


def test_incremental_room_object_graph_growth():
    store = InteractionGraphStore(scene_id="test_scene")
    batches = [
        [observation(observation_id="obs1", instance_id="obj_1", semantic_name="apple", room_id=1, position=[1, 1, 0.8])],
        [observation(observation_id="obs2", instance_id="obj_2", semantic_name="cup", room_id=1, position=[2, 1, 0.8])],
    ]
    store.update_observations(batches[0], source_mode="gt_replay")
    first_graph = store.as_graph_dict()
    assert any(node["id"] == "room_1" for node in first_graph["nodes"])
    assert len(first_graph["nodes"]) == 3
    scene = next(node for node in first_graph["nodes"] if node["type"] == "scene")
    room = next(node for node in first_graph["nodes"] if node["type"] == "room")
    assert room["parent_id"] == scene["id"]
    assert any(
        edge["src_id"] == scene["id"] and edge["relation"] == "has_room" and edge["dst_id"] == room["id"]
        for edge in first_graph["edges"]
    )

    store.update_observations(batches[1], source_mode="gt_replay")
    second_graph = store.as_graph_dict()
    assert len(second_graph["nodes"]) == 4
    assert len(second_graph["views"]["navigation_view"]["hints"]) >= 3


def test_portal_connects_two_rooms_and_navigation_hint():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                observation_id="door1",
                instance_id="door_1",
                semantic_name="door",
                category="Door",
                room_id=1,
                connected_room_ids=[1, 2],
                is_door=True,
                is_movable_door=True,
                is_articulable=True,
                joint_type="hinge",
                joint_range=[0.0, 1.57],
                joint_value=0.0,
                position=[0.0, 0.0, 1.0],
                aabb_center=[0.0, 0.0, 1.0],
                aabb_size=[0.9, 0.1, 2.0],
            )
        ],
        source_mode="gt_replay",
    )
    graph = store.as_graph_dict()
    relations = {(edge["src_id"], edge["relation"], edge["dst_id"]) for edge in graph["edges"]}
    assert ("portal_door_1", "connects", "room_1") in relations
    assert ("portal_door_1", "connects", "room_2") in relations
    hint_types = {hint["type"] for hint in graph["views"]["navigation_view"]["hints"]}
    assert "interactive_portal" in hint_types


def test_support_and_container_hierarchy_assignment():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                observation_id="support",
                instance_id="table_1",
                semantic_name="table",
                room_id=1,
                is_receptacle=True,
                position=[1.0, 1.0, 0.4],
                aabb_center=[1.0, 1.0, 0.4],
                aabb_size=[1.0, 1.0, 0.8],
            ),
            observation(
                observation_id="obj",
                instance_id="apple_1",
                semantic_name="apple",
                room_id=1,
                position=[1.0, 1.0, 0.82],
                aabb_center=[1.0, 1.0, 0.82],
                aabb_size=[0.1, 0.1, 0.1],
            ),
            observation(
                observation_id="container",
                instance_id="fridge_1",
                semantic_name="fridge",
                room_id=1,
                is_receptacle=True,
                is_articulable=True,
                joint_type="hinge",
                joint_range=[0.0, 1.0],
                joint_value=0.0,
                position=[3.0, 0.0, 1.0],
                aabb_center=[3.0, 0.0, 1.0],
                aabb_size=[1.0, 1.0, 2.0],
            ),
            observation(
                observation_id="milk",
                instance_id="milk_1",
                semantic_name="milk",
                room_id=1,
                position=[3.0, 0.0, 1.0],
                aabb_center=[3.0, 0.0, 1.0],
                aabb_size=[0.1, 0.1, 0.2],
            ),
        ]
    )
    graph = store.as_graph_dict()
    relations = {(edge["src_id"], edge["relation"], edge["dst_id"]) for edge in graph["edges"]}
    assert ("support_table_1", "supports", "object_apple_1") in relations
    assert ("container_fridge_1", "contains", "object_milk_1") in relations


def test_object_above_container_is_not_inferred_as_internal_content():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="dresser_1",
                semantic_name="dresser",
                is_receptacle=True,
                is_articulable=True,
                position=[0.0, 0.0, 0.35],
                aabb_center=[0.0, 0.0, 0.35],
                aabb_size=[0.4, 0.7, 0.7],
            ),
            observation(
                instance_id="phone_1",
                semantic_name="cellphone",
                position=[0.0, 0.0, 0.71],
                aabb_center=[0.0, 0.0, 0.71],
                aabb_size=[0.1, 0.05, 0.02],
            ),
        ]
    )

    relations = {
        (edge["src_id"], edge["relation"], edge["dst_id"])
        for edge in store.as_graph_dict()["edges"]
    }
    assert ("container_dresser_1", "contains", "object_phone_1") not in relations


def test_open_drawer_does_not_attach_visible_non_pickup_plant_by_proximity():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="drawer_1",
                semantic_name="drawer",
                is_receptacle=True,
                is_articulable=True,
                joint_type="slide",
                joint_range=[0.0, 0.4],
                joint_value=0.4,
                position=[0.0, 0.0, 0.4],
                aabb_center=[0.0, 0.0, 0.4],
                aabb_size=[0.8, 0.6, 0.5],
            ),
            observation(
                instance_id="houseplant_1",
                semantic_name="houseplant",
                position=[0.15, 0.0, 0.45],
                aabb_center=[0.15, 0.0, 0.45],
                aabb_size=[0.25, 0.25, 0.60],
                is_pickup_candidate=False,
            ),
        ],
        source_mode="realtime_gt_observation",
    )

    relations = {
        (edge["src_id"], edge["relation"], edge["dst_id"])
        for edge in store.as_graph_dict()["edges"]
    }
    assert ("container_drawer_1", "contains", "object_houseplant_1") not in relations


def test_multi_drawer_joint_metadata_is_not_stored_or_grouped():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="dresser_1",
                semantic_name="dresser",
                is_receptacle=True,
                is_articulable=True,
                joint_type="slide",
                joint_range=[0.0, 0.4],
                joint_value=0.0,
                joint_infos=[
                    {
                        "joint_name": "drawer_top",
                        "joint_type": "slide",
                        "joint_range": [0.0, 0.4],
                        "joint_value": 0.0,
                    },
                    {
                        "joint_name": "drawer_bottom",
                        "joint_type": "slide",
                        "joint_range": [0.0, 0.4],
                        "joint_value": 0.0,
                    },
                ],
            )
        ]
    )

    node = next(
        node
        for node in store.as_graph_dict()["nodes"]
        if node["id"] == "container_dresser_1"
    )
    assert node["interaction"]["interaction_mode"] == "slide"
    assert node["interaction"]["state"] == "unknown"
    assert "joint_infos" not in node["attributes"]
    assert "interaction_groups" not in node["attributes"]


def test_fridge_joint_metadata_does_not_define_graph_interaction_groups():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="fridge_1",
                semantic_name="fridge",
                category="Fridge",
                is_receptacle=True,
                is_articulable=True,
                joint_infos=[
                    {
                        "joint_name": "fridge_door_left",
                        "joint_type": "hinge",
                        "joint_range": [0.0, 1.57],
                        "joint_value": 0.0,
                    },
                    {
                        "joint_name": "fridge_inner_slide",
                        "joint_type": "slide",
                        "joint_range": [0.0, 0.4],
                        "joint_value": 0.0,
                    },
                ],
            )
        ]
    )

    node = next(
        node
        for node in store.as_graph_dict()["nodes"]
        if node["id"] == "container_fridge_1"
    )
    assert node["interaction"]["interaction_mode"] == "open_close"
    assert node["interaction"]["state"] == "unknown"
    assert "joint_infos" not in node["attributes"]
    assert "interaction_groups" not in node["attributes"]


def test_toilet_receptacle_metadata_does_not_make_it_a_container():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="toilet_1",
                semantic_name="toilet",
                is_receptacle=True,
                is_articulable=True,
            )
        ]
    )

    node = next(
        node for node in store.as_graph_dict()["nodes"] if node["id"] == "object_toilet_1"
    )
    assert node["type"] == "object"


def test_existing_instance_updates_instead_of_duplication():
    store = InteractionGraphStore(scene_id="test_scene")
    obs = observation(
        observation_id="obs1",
        instance_id="apple_1",
        semantic_name="apple",
        room_id=1,
        position=[0.0, 0.0, 0.8],
    )
    store.update_observations([obs])
    obs2 = dict(obs)
    obs2["observation_id"] = "obs2"
    obs2["position"] = [0.05, 0.02, 0.8]
    store.update_observations([obs2])
    nodes = [node for node in store.as_graph_dict()["nodes"] if node["type"] == "object"]
    assert len(nodes) == 1
    assert nodes[0]["observation_count"] == 2


def test_object_store_confirms_same_label_despite_box_size_jitter():
    store = ObjectMapStore(match_distance=0.5, min_confirmations=2, size_match_ratio=0.7)
    store.update(
        [
            {
                "semantic_class": "window",
                "confidence": 0.8,
                "world_position": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_center": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_size": {"x": 0.03, "y": 0.32, "z": 0.38},
            }
        ],
        stamp=1.0,
    )
    store.update(
        [
            {
                "semantic_class": "window",
                "confidence": 0.9,
                "world_position": {"x": 1.02, "y": 2.01, "z": 0.8},
                "world_box3d_center": {"x": 1.02, "y": 2.01, "z": 0.8},
                "world_box3d_size": {"x": 0.12, "y": 0.28, "z": 0.41},
            }
        ],
        stamp=2.0,
    )

    tracked = store.as_tracked_detections()
    assert len(tracked) == 1
    assert tracked[0]["observation_count"] == 2
    assert tracked[0]["world_box3d_size"] == {"x": 0.07050000000000001, "y": 0.30200000000000005, "z": 0.3935}


def test_object_store_rejects_large_box_outlier_from_stable_box():
    store = ObjectMapStore(match_distance=0.5, min_confirmations=2, size_match_ratio=0.7)
    store.update(
        [
            {
                "semantic_class": "chair",
                "confidence": 0.8,
                "world_position": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_center": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_size": {"x": 0.3, "y": 0.3, "z": 0.8},
            }
        ],
        stamp=1.0,
    )
    store.update(
        [
            {
                "semantic_class": "chair",
                "confidence": 0.9,
                "world_position": {"x": 1.01, "y": 2.01, "z": 0.8},
                "world_box3d_center": {"x": 1.01, "y": 2.01, "z": 0.8},
                "world_box3d_size": {"x": 3.0, "y": 3.0, "z": 3.0},
            }
        ],
        stamp=2.0,
    )

    tracked = store.as_tracked_detections()
    assert len(tracked) == 1
    assert tracked[0]["observation_count"] == 2
    assert tracked[0]["world_box3d_size"] == {"x": 0.3, "y": 0.3, "z": 0.8}
    assert tracked[0]["latest_box3d_size"] == {"x": 3.0, "y": 3.0, "z": 3.0}


def test_object_store_can_expose_tentative_tracks_for_graph():
    store = ObjectMapStore(match_distance=0.5, min_confirmations=2, size_match_ratio=0.7)
    store.update(
        [
            {
                "semantic_class": "bottle",
                "confidence": 0.75,
                "world_position": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_center": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_size": {"x": 0.08, "y": 0.08, "z": 0.24},
            }
        ],
        stamp=1.0,
    )

    assert store.as_tracked_detections() == []
    tentative = store.as_tracked_detections(min_observations=1, confirmed_only=False)
    assert len(tentative) == 1
    assert tentative[0]["semantic_class"] == "bottle"
    assert tentative[0]["observation_count"] == 1


def test_object_store_merges_overlapping_different_labels():
    store = ObjectMapStore(match_distance=0.5, min_confirmations=2, size_match_ratio=0.7)
    store.update(
        [
            {
                "semantic_class": "window",
                "confidence": 0.8,
                "world_position": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_center": {"x": 1.0, "y": 2.0, "z": 0.8},
                "world_box3d_size": {"x": 0.2, "y": 0.4, "z": 0.5},
            }
        ],
        stamp=1.0,
    )
    store.update(
        [
            {
                "semantic_class": "curtain",
                "confidence": 0.9,
                "world_position": {"x": 1.02, "y": 2.01, "z": 0.8},
                "world_box3d_center": {"x": 1.02, "y": 2.01, "z": 0.8},
                "world_box3d_size": {"x": 0.21, "y": 0.39, "z": 0.49},
            }
        ],
        stamp=2.0,
    )

    tracked = store.as_tracked_detections()
    assert len(tracked) == 1
    assert tracked[0]["semantic_class"] == "curtain"
    assert tracked[0]["candidate_labels"] == ["curtain", "window"]
    assert tracked[0]["label_votes"]["window"] == 0.8
    assert tracked[0]["label_votes"]["curtain"] == 0.9


def test_detection_observation_keeps_latest_visual_box():
    obs = observation_from_detection(
        {
            "semantic_class": "window",
            "confidence": 0.9,
            "world_position": {"x": 1.0, "y": 2.0, "z": 0.8},
            "world_box3d_center": {"x": 1.0, "y": 2.0, "z": 0.8},
            "world_box3d_size": {"x": 0.12, "y": 0.28, "z": 0.41},
            "viz_aabb_center": {"x": 1.1, "y": 2.1, "z": 0.9},
            "viz_aabb_size": {"x": 0.2, "y": 0.3, "z": 0.4},
        },
        observation_id="det_0001",
    )

    assert obs["aabb_size"] == [0.12, 0.28, 0.41]
    assert obs["viz_aabb_center"] == [1.1, 2.1, 0.9]
    assert obs["viz_aabb_size"] == [0.2, 0.3, 0.4]


def test_json_serializable_and_gt_batches():
    records = [
        {
            "name": "apple_1",
            "object_id": "apple_1",
            "category": "apple",
            "room_id": 1,
            "position": [0.0, 0.0, 0.8],
            "aabb_center": [0.0, 0.0, 0.8],
            "aabb_size": [0.1, 0.1, 0.1],
            "is_receptacle": False,
            "is_pickup_candidate": False,
            "is_articulable": False,
            "is_door": False,
            "is_movable_door": False,
            "joint_infos": [],
        },
        {
            "name": "door_1",
            "object_id": "door_1",
            "category": "Door",
            "room_id": 1,
            "position": [1.0, 0.0, 1.0],
            "aabb_center": [1.0, 0.0, 1.0],
            "aabb_size": [0.9, 0.1, 2.0],
            "is_receptacle": False,
            "is_pickup_candidate": False,
            "is_articulable": True,
            "is_door": True,
            "is_movable_door": True,
            "joint_infos": [{"joint_type": "hinge", "joint_range": [0.0, 1.57], "joint_value": 0.0}],
            "connected_room_ids": [1, 2],
        },
    ]
    batches = build_gt_observation_batches(records, num_batches=2, shuffle=False, seed=0)
    assert len(batches) == 2
    store = InteractionGraphStore(scene_id="test_scene")
    for batch in batches:
        store.update_observations(batch, source_mode="gt_replay")
    payload = store.as_graph_dict()
    json.dumps(payload)
    json.dumps(store.as_navigation_hints())


class FakeGridInfo:
    width = 8
    height = 8
    resolution = 1.0

    class Origin:
        class Position:
            x = 0.0
            y = 0.0

        position = Position()

    origin = Origin()


def test_realtime_visibility_age_and_episode_reset():
    store = InteractionGraphStore(scene_id="test_scene")
    store.reset("episode_000001", source_mode="realtime_gt_observation")
    door = observation(
        instance_id="gt_000001",
        semantic_name="door",
        is_door=True,
        is_articulable=True,
        joint_type="hinge",
        joint_range=[0.0, 1.0],
        joint_value=0.0,
    )
    store.update_observations([door], stamp=10.0, source_mode="realtime_gt_observation")
    store.update_observations([], stamp=12.0, source_mode="realtime_gt_observation")
    graph = store.as_graph_dict(stamp=13.0)
    portal = next(node for node in graph["nodes"] if node["type"] == "portal")
    assert portal["is_currently_visible"] is False
    assert portal["state_age_sec"] == 3.0
    assert portal["interaction"]["state"] == "unknown"
    assert graph["episode_id"] == "episode_000001"
    assert graph["graph_revision"] == 2

    store.reset("episode_000002", source_mode="realtime_gt_observation")
    graph = store.as_graph_dict(stamp=14.0)
    assert graph["episode_id"] == "episode_000002"
    assert graph["graph_revision"] == 0
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["type"] == "scene"
    assert graph["nodes"][0]["attributes"]["episode_id"] == "episode_000002"


def test_portal_room_connections_are_inferred_from_room_ring():
    scene_data = []
    for _y in range(FakeGridInfo.height):
        scene_data.extend([1, 1, 1, 1, 2, 2, 2, 2])
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_room_grid(FakeGridInfo(), scene_data, [100] * len(scene_data))
    store.update_observations(
        [
            observation(
                instance_id="gt_000001",
                semantic_name="door",
                is_door=True,
                is_articulable=True,
                joint_type="hinge",
                joint_range=[0.0, 1.0],
                joint_value=0.0,
                position=[4.0, 4.0, 1.0],
                aabb_center=[4.0, 4.0, 1.0],
                aabb_size=[0.2, 1.0, 2.0],
            )
        ],
        source_mode="realtime_gt_observation",
    )
    graph = store.as_graph_dict()
    portal = next(node for node in graph["nodes"] if node["type"] == "portal")
    assert set(portal["attributes"]["connected_room_ids"]) == {1, 2}
    assert portal["attributes"]["connectivity_status"] == "connected"
    connects = [edge for edge in graph["edges"] if edge["src_id"] == portal["id"] and edge["relation"] == "connects"]
    assert {edge["dst_id"] for edge in connects} == {"room_1", "room_2"}
    assert all(edge["attributes"]["traversable"] is False for edge in connects)
    assert all(edge["attributes"]["requires_interaction"] is True for edge in connects)


def test_portal_room_ring_does_not_reach_distant_room():
    class WideGridInfo(FakeGridInfo):
        width = 12

    scene_data = []
    for _y in range(WideGridInfo.height):
        scene_data.extend([2] * 7 + [-1] + [1] * 4)
    store = InteractionGraphStore(
        scene_id="test_scene", portal_room_max_radius_m=1.0
    )
    store.update_room_grid(WideGridInfo(), scene_data, [100] * len(scene_data))
    store.update_observations(
        [
            observation(
                instance_id="door_near_room_2",
                semantic_name="door",
                is_door=True,
                is_articulable=True,
                joint_type="hinge",
                joint_range=[0.0, 1.0],
                joint_value=0.0,
                position=[5.0, 4.0, 1.0],
                aabb_center=[5.0, 4.0, 1.0],
                aabb_size=[1.0, 2.0, 2.0],
            )
        ],
        source_mode="realtime_gt_observation",
    )

    portal = next(
        node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal"
    )
    assert portal["attributes"]["connected_room_ids"] == [2]
    assert portal["attributes"]["connectivity_status"] == "partial"


def test_container_room_assignment_uses_nearest_segment_ring():
    scene_data = [1] * (FakeGridInfo.width * FakeGridInfo.height)
    for y in range(3, 5):
        for x in range(3, 5):
            scene_data[y * FakeGridInfo.width + x] = -1
    store = InteractionGraphStore(
        scene_id="test_scene",
        object_room_search_margin_m=0.75,
    )
    store.update_room_grid(FakeGridInfo(), scene_data, [100] * len(scene_data))
    store.update_observations(
        [
            observation(
                instance_id="fridge_ring",
                semantic_name="fridge",
                category="Fridge",
                is_receptacle=True,
                is_articulable=True,
                position=[4.0, 4.0, 1.0],
                aabb_center=[4.0, 4.0, 1.0],
                aabb_size=[2.0, 2.0, 2.0],
            )
        ]
    )

    fridge = next(
        node
        for node in store.as_graph_dict()["nodes"]
        if node["id"] == "container_fridge_ring"
    )
    assert fridge["room_id"] == 1
    assert fridge["parent_id"] == "room_1"


def test_container_relation_persists_while_child_is_unobserved_and_clears_after_move():
    store = InteractionGraphStore(scene_id="test_scene")
    container = observation(
        instance_id="gt_000001",
        semantic_name="fridge",
        room_id=1,
        is_receptacle=True,
        is_articulable=True,
        joint_type="hinge",
        joint_range=[0.0, 1.0],
        joint_value=1.0,
        position=[2.0, 2.0, 1.0],
        aabb_center=[2.0, 2.0, 1.0],
        aabb_size=[1.0, 1.0, 2.0],
    )
    child = observation(
        instance_id="gt_000002",
        semantic_name="milk",
        room_id=1,
        position=[2.0, 2.0, 1.0],
        aabb_center=[2.0, 2.0, 1.0],
        aabb_size=[0.1, 0.1, 0.2],
    )
    store.update_observations([container, child], source_mode="realtime_gt_observation")
    store.update_observations([container], source_mode="realtime_gt_observation")
    relations = {(edge["src_id"], edge["relation"], edge["dst_id"]) for edge in store.as_graph_dict()["edges"]}
    assert ("container_gt_000001", "contains", "object_gt_000002") in relations

    moved_child = dict(child)
    moved_child["position"] = [4.0, 4.0, 1.0]
    moved_child["aabb_center"] = [4.0, 4.0, 1.0]
    store.update_observations([moved_child], source_mode="realtime_gt_observation")
    relations = {(edge["src_id"], edge["relation"], edge["dst_id"]) for edge in store.as_graph_dict()["edges"]}
    assert ("container_gt_000001", "contains", "object_gt_000002") not in relations


def test_interaction_result_updates_planner_fields():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="gt_000001",
                semantic_name="door",
                is_door=True,
                is_articulable=True,
                joint_type="hinge",
                joint_range=[0.0, 1.0],
                joint_value=0.0,
            )
        ],
        source_mode="realtime_gt_observation",
    )
    assert store.update_interaction_result(
        {
            "instance_id": "gt_000001",
            "state": "open",
            "source": "oracle_interaction",
            "approach_goal_xyyaw": [1.0, 2.0, 0.5],
        },
        stamp=20.0,
    )
    portal = next(node for node in store.as_graph_dict(stamp=20.0)["nodes"] if node["type"] == "portal")
    assert portal["interaction"]["state"] == "open"
    assert portal["interaction"]["traversable"] is True
    assert portal["interaction"]["requires_interaction"] is False
    assert portal["interaction"]["operation_history"] == [
        {
            "event_id": "interaction_000001",
            "action": "unknown",
            "timestamp": 20.0,
            "pre_state": "unknown",
            "post_state": "open",
            "success": True,
            "execution_cost": 1.0,
            "verification_source": "oracle_interaction",
            "approach_goal_xyyaw": [1.0, 2.0, 0.5],
        }
    ]
    assert "joints" not in portal["attributes"]
    assert "primary_joint_name" not in portal["attributes"]
    assert "observation_evidence" not in portal["attributes"]
    connects_edges = [
        edge for edge in store.as_graph_dict(stamp=20.0)["edges"]
        if edge["relation"] == "connects" and edge["src_id"] == portal["id"]
    ]
    assert all(edge["attributes"]["traversable"] is True for edge in connects_edges)

    store.update_observations(
        [
            observation(
                instance_id="gt_000001",
                semantic_name="door",
                is_door=True,
                is_articulable=True,
                joint_type="hinge",
                joint_range=[0.0, 1.0],
                joint_value=1.0,
            )
        ],
        stamp=21.0,
        source_mode="realtime_gt_observation",
    )
    portal = next(node for node in store.as_graph_dict(stamp=21.0)["nodes"] if node["type"] == "portal")
    assert len(portal["interaction"]["operation_history"]) == 1


def test_non_articulated_portal_feedback_persists_static_capability() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    doorframe = observation(
        instance_id="doorframe_static_1",
        semantic_name="doorframe",
        is_door=False,
        is_articulable=False,
    )
    doorframe["source_object_name"] = "doorframe_static_1"
    store.update_observations(
        [doorframe], source_mode="realtime_gt_observation", stamp=1.0
    )
    assert store.update_interaction_result(
        {
            "source_object_name": "doorframe_static_1",
            "event_id": "static_feedback",
            "state": "static",
            "success": False,
            "reason": "non_articulated",
            "interaction_capability": "static",
            "interactable": False,
        },
        stamp=2.0,
    )
    store.update_observations(
        [doorframe], source_mode="realtime_gt_observation", stamp=3.0
    )
    assert store.apply_attribute_patch(
        {
            "object_id": "doorframe_static_1",
            "attribute_status": "ready",
            "interactable": True,
            "interaction_class": "portal",
            "coarse_state": "closed",
            "confidence": 0.95,
            "interaction_parts": [],
            "source": "mllm_attribute_inference",
        },
        stamp=4.0,
    )

    graph = store.as_graph_dict(stamp=4.0)
    portal = next(
        node for node in graph["nodes"]
        if node["id"] == "portal_doorframe_static_1"
    )
    assert portal["type"] == "portal"
    assert portal["interaction"]["state"] == "static"
    assert portal["interaction"]["is_interactable"] is False
    assert portal["interaction"]["requires_interaction"] is False
    assert portal["interaction"]["interaction_mode"] == "none"
    assert portal["interaction"]["capability"] == "static"
    navigation_hint = next(
        hint for hint in graph["views"]["navigation_view"]["hints"]
        if hint["node_id"] == "portal_doorframe_static_1"
    )
    assert navigation_hint["requires_interaction"] is False


def test_interaction_result_can_match_source_name_and_is_idempotent_by_event_id():
    store = InteractionGraphStore(scene_id="test_scene")
    closed = observation(
        instance_id="gt_000001",
        semantic_name="door",
        is_door=True,
        is_articulable=True,
        is_movable_door=True,
        joint_type="hinge",
        joint_range=[0.0, 1.0],
        joint_value=0.0,
    )
    closed["source_object_name"] = "double_door_root"
    store.update_observations([closed], source_mode="realtime_gt_observation")
    result = {
        "source_object_name": "double_door_root",
        "event_id": "phase_04_close",
        "state": "closed",
        "joint_infos": [
            {
                "joint_name": "left_hinge",
                "joint_type": "hinge",
                "joint_range": [0.0, 1.0],
                "joint_value": 0.0,
            }
        ],
        "source": "direct_joint_readback",
    }
    assert store.update_interaction_result(result, stamp=30.0)
    assert store.update_interaction_result(result, stamp=31.0)
    stale_open = dict(closed)
    stale_open["joint_value"] = 1.0
    stale_open["joint_infos"] = [
        {
            "joint_name": "left_hinge",
            "joint_type": "hinge",
            "joint_range": [0.0, 1.0],
            "joint_value": 1.0,
        }
    ]
    store.update_observations([stale_open], stamp=32.0, source_mode="realtime_gt_observation")
    portal = next(node for node in store.as_graph_dict(stamp=31.0)["nodes"] if node["type"] == "portal")
    assert portal["interaction"]["state"] == "closed"
    assert [event["event_id"] for event in portal["interaction"]["operation_history"]] == [
        "phase_04_close"
    ]


def test_portal_state_ignores_joint_delta_until_semantic_result_arrives():
    store = InteractionGraphStore(scene_id="test_scene")
    closed = observation(
        instance_id="negative_hinge_door",
        semantic_name="door",
        is_door=True,
        is_articulable=True,
        is_movable_door=True,
        joint_type="hinge",
        joint_range=[-1.0, 0.0],
        joint_value=0.0,
    )
    store.update_observations([closed], source_mode="realtime_gt_observation")

    ajar = dict(closed)
    ajar["joint_value"] = -0.4
    store.update_observations([ajar], source_mode="realtime_gt_observation")
    portal = next(node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal")
    assert portal["interaction"]["state"] == "unknown"
    assert portal["interaction"]["traversable"] is False
    assert portal["interaction"]["requires_interaction"] is True

    opened = dict(closed)
    opened["joint_value"] = -0.8
    store.update_observations([opened], source_mode="realtime_gt_observation")
    portal = next(node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal")
    assert portal["interaction"]["state"] == "unknown"
    assert portal["interaction"]["traversable"] is False
    assert "open_fraction" not in portal["interaction"]


def test_late_attribute_patch_does_not_override_interaction_state():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [observation(instance_id="gt_000001", semantic_name="door", is_door=True)],
        source_mode="realtime_gt_observation",
        stamp=1.0,
    )
    assert store.update_interaction_result(
        {
            "instance_id": "gt_000001",
            "state": "open",
            "source": "direct_joint_readback",
            "event_id": "interaction_001",
        },
        stamp=10.0,
    )
    assert store.apply_attribute_patch(
        {
            "object_id": "gt_000001",
            "attribute_status": "ready",
            "interactable": True,
            "interaction_class": "portal",
            "coarse_state": "closed",
            "confidence": 0.9,
            "interaction_parts": [],
            "source": "mllm_attribute_inference",
        },
        stamp=5.0,
    )
    portal = next(node for node in store.as_graph_dict(stamp=11.0)["nodes"] if node["type"] == "portal")
    assert portal["interaction"]["state"] == "open"
    assert portal["attributes"]["attribute_status"] == "ready"


def test_newer_attribute_patch_does_not_override_verified_interaction_state():
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [observation(instance_id="gt_000001", semantic_name="door", is_door=True)],
        source_mode="realtime_gt_observation",
        stamp=1.0,
    )
    assert store.update_interaction_result(
        {
            "instance_id": "gt_000001",
            "state": "open",
            "source": "executor_state_verification",
            "event_id": "interaction_001",
        },
        stamp=10.0,
    )
    assert store.apply_attribute_patch(
        {
            "object_id": "gt_000001",
            "attribute_status": "ready",
            "interactable": True,
            "interaction_class": "container",
            "coarse_state": "closed",
            "confidence": 0.95,
            "interaction_parts": [],
            "source": "mllm_attribute_inference",
        },
        stamp=12.0,
    )

    portal = next(
        node
        for node in store.as_graph_dict(stamp=12.0)["nodes"]
        if node["id"] == "portal_gt_000001"
    )
    assert portal["type"] == "portal"
    assert portal["interaction"]["state"] == "open"
    assert portal["interaction"]["traversable"] is True
    assert portal["interaction"]["requires_interaction"] is False
    assert portal["attributes"]["attribute_status"] == "ready"


def test_joint_values_do_not_make_plain_box_interactable_or_open():
    store = InteractionGraphStore(scene_id="test_scene")
    box = observation(
        instance_id="open_box",
        semantic_name="box",
        is_receptacle=True,
        is_articulable=True,
        joint_type="hinge",
        joint_range=[-2.8, 0.0],
        joint_value=-2.8,
    )
    box["joint_infos"] = [
        {
            "joint_name": "negative_flap",
            "joint_type": "hinge",
            "joint_range": [-2.8, 0.0],
            "joint_value": -2.8,
        },
        {
            "joint_name": "positive_flap",
            "joint_type": "hinge",
            "joint_range": [0.0, 2.2],
            "joint_value": 2.2,
        },
    ]
    store.update_observations([box], source_mode="realtime_gt_observation")

    node = next(node for node in store.as_graph_dict()["nodes"] if node["type"] == "container")
    assert node["interaction"]["state"] == "unknown"
    assert node["interaction"]["is_interactable"] is False
    assert node["interaction"]["requires_interaction"] is False


def test_room_merge_retires_secondary_room_and_migrates_children() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [observation(instance_id="cup_1", semantic_name="cup", room_id=2)]
    )
    info = type("Info", (), {"width": 2, "resolution": 1.0})()
    info.origin = type("Origin", (), {"position": type("Position", (), {"x": 0.0, "y": 0.0})()})()

    store.update_room_grid(
        info,
        [1, 2],
        [100, 100],
        room_merges={2: 1},
        geometry_stability_frames=1,
    )
    graph = store.as_graph_dict()
    cup = next(node for node in graph["nodes"] if node["id"] == "object_cup_1")
    room_two = next(node for node in graph["nodes"] if node["id"] == "room_2")
    assert cup["room_id"] == 1
    assert room_two["attributes"]["active"] is False
    assert not any(
        edge["src_id"] == "scene_test_scene" and edge["dst_id"] == "room_2"
        for edge in graph["edges"]
    )


def test_declared_parent_does_not_override_geometric_container_relation() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="dresser_1",
                semantic_name="dresser",
                room_id=1,
                is_receptacle=True,
                is_articulable=True,
                aabb_center=[1.0, 1.0, 0.5],
                aabb_size=[1.0, 1.0, 1.0],
                name="dresser_root",
            ),
            observation(
                instance_id="pencil_1",
                semantic_name="pencil",
                room_id=2,
                parent="dresser_root",
                aabb_center=[1.8, 1.0, 0.5],
                aabb_size=[0.05, 0.05, 0.05],
            ),
        ]
    )
    relations = {
        (edge["src_id"], edge["relation"], edge["dst_id"])
        for edge in store.as_graph_dict()["edges"]
    }
    assert ("container_dresser_1", "contains", "object_pencil_1") not in relations


def test_open_drawer_does_not_contain_object_resting_on_top() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="dresser_1",
                semantic_name="dresser",
                is_receptacle=True,
                is_articulable=True,
                joint_type="slide",
                joint_range=[0.0, 1.0],
                joint_value=1.0,
                aabb_center=[1.0, 1.0, 0.35],
                aabb_size=[0.6, 0.5, 0.7],
            ),
            observation(
                instance_id="pencil_1",
                semantic_name="pencil",
                position=[1.0, 1.0, 0.705],
                aabb_center=[1.0, 1.0, 0.705],
                aabb_size=[0.02, 0.15, 0.02],
            ),
        ],
        source_mode="realtime_gt_observation",
    )
    relations = {
        (edge["src_id"], edge["relation"], edge["dst_id"])
        for edge in store.as_graph_dict()["edges"]
    }
    assert ("container_dresser_1", "contains", "object_pencil_1") not in relations


def test_open_drawer_live_aabb_contains_and_retains_hidden_content() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    drawer = observation(
        instance_id="drawer_1",
        semantic_name="drawer",
        is_receptacle=True,
        is_articulable=True,
        joint_type="slide",
        joint_range=[0.0, 1.0],
        joint_value=1.0,
        aabb_center=[1.0, 1.3, 0.5],
        aabb_size=[1.0, 1.6, 1.0],
    )
    pencil = observation(
        instance_id="pencil_1",
        semantic_name="pencil",
        position=[1.0, 1.85, 0.5],
        aabb_center=[1.0, 1.85, 0.5],
        aabb_size=[0.05, 0.05, 0.05],
    )
    store.update_observations([drawer, pencil], source_mode="realtime_gt_observation")
    assert any(
        edge["src_id"] == "container_drawer_1"
        and edge["relation"] == "contains"
        and edge["dst_id"] == "object_pencil_1"
        for edge in store.as_graph_dict()["edges"]
    )
    drawer["joint_value"] = 0.0
    store.update_observations([drawer], source_mode="realtime_gt_observation")
    assert any(
        edge["src_id"] == "container_drawer_1"
        and edge["relation"] == "contains"
        and edge["dst_id"] == "object_pencil_1"
        for edge in store.as_graph_dict()["edges"]
    )


def test_house2_dresser_does_not_contain_pencil_on_top() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    store.update_observations(
        [
            observation(
                instance_id="dresser_1",
                semantic_name="dresser",
                is_receptacle=True,
                is_articulable=True,
                joint_type="slide",
                joint_range=[0.0, 0.213],
                joint_value=0.213,
                aabb_center=[2.905994996767497, 0.2881095037307979, 0.3492059886203346],
                aabb_size=[0.6197699904441833, 0.5209470056486065, 0.6984999723109293],
            ),
            observation(
                instance_id="pencil_1",
                semantic_name="pencil",
                position=[3.0611600037425593, 0.14212999982194482, 0.7051813530875322],
                aabb_center=[3.061191923838525, 0.1397641560319149, 0.7049824030053934],
                aabb_size=[0.00872475984339438, 0.18825023621644954, 0.008829840175350157],
            ),
        ],
        source_mode="realtime_gt_observation",
    )
    relations = {
        (edge["src_id"], edge["relation"], edge["dst_id"])
        for edge in store.as_graph_dict()["edges"]
    }
    assert ("container_dresser_1", "contains", "object_pencil_1") not in relations


def test_container_semantic_interaction_state_survives_later_observations() -> None:

    store = InteractionGraphStore(scene_id="test_scene")
    dresser = observation(
        instance_id="dresser_1",
        semantic_name="dresser",
        is_receptacle=True,
        is_articulable=True,
        joint_type="slide",
        joint_range=[0.0, 0.4],
        joint_value=0.0,
        joint_infos=[
            {"joint_name": "top", "joint_type": "slide", "joint_range": [0.0, 0.4], "joint_value": 0.0},
            {"joint_name": "bottom", "joint_type": "slide", "joint_range": [0.0, 0.4], "joint_value": 0.0},
        ],
    )
    store.update_observations([dresser], source_mode="realtime_gt_observation")
    node_id = "container_dresser_1"
    assert store.update_interaction_result(
        {
            "node_id": node_id,
            "event_id": "open_top",
            "interaction_group_id": "region:top",
            "post_state": "ajar",
            "success": True,
        }
    )
    store.update_observations([dresser], source_mode="realtime_gt_observation")
    node = next(node for node in store.as_graph_dict()["nodes"] if node["id"] == node_id)
    assert node["interaction"]["completed_interaction_groups"] == ["region:top"]

    assert "joint_interaction_states" not in node["interaction"]
    assert node["interaction"]["state"] == "ajar"

    assert store.update_interaction_result(
        {
            "node_id": node_id,
            "event_id": "open_bottom",
            "interaction_group_id": "region:bottom",
            "post_state": "open",
            "success": True,
        }
    )
    node = next(node for node in store.as_graph_dict()["nodes"] if node["id"] == node_id)
    assert node["interaction"]["completed_interaction_groups"] == [
        "region:bottom",
        "region:top",
    ]
    assert "all_interaction_groups_completed" not in node["interaction"]

    assert node["interaction"]["state"] == "open"


def test_room_geometry_does_not_shrink_after_confirmed_observation() -> None:
    store = InteractionGraphStore(scene_id="test_scene")
    info = type("Info", (), {"width": 3, "resolution": 1.0})()
    info.origin = type("Origin", (), {"position": type("Position", (), {"x": 0.0, "y": 0.0})()})()
    store.update_room_grid(info, [1, 1, -1], [100, 100, 0], geometry_stability_frames=1)
    initial = next(node for node in store.as_graph_dict()["nodes"] if node["id"] == "room_1")
    initial_center = list(initial["aabb_center"])
    initial_size = list(initial["aabb_size"])
    store.update_room_grid(info, [1, -1, -1], [100, 0, 0], geometry_stability_frames=1)
    updated = next(node for node in store.as_graph_dict()["nodes"] if node["id"] == "room_1")
    assert updated["aabb_center"] == initial_center
    assert updated["aabb_size"] == initial_size


def test_attribute_patch_sets_semantic_state_and_preserves_last_seen():
    store = InteractionGraphStore(scene_id="test_scene")
    door = observation(
        instance_id="gt_000001",
        semantic_name="door",
        is_door=True,
        is_articulable=True,
        joint_type="hinge",
        joint_range=[0.0, 1.0],
        joint_value=0.0,
        frame_index=7,
    )
    store.update_observations([door], stamp=10.0, source_mode="realtime_gt_observation")
    assert store.apply_attribute_patch(
        {
            "object_id": "gt_000001",
            "attribute_status": "ready",
            "request_sequence": 3,
            "observation_frame_index": 7,
            "interactable": True,
            "interaction_class": "portal",
            "coarse_state": "open",
            "confidence": 0.9,
            "interaction_parts": [{"part_id": "door", "type": "door"}],
        },
        stamp=20.0,
    )

    portal = next(
        node for node in store.as_graph_dict(stamp=21.0)["nodes"] if node["type"] == "portal"
    )
    assert portal["interaction"]["state"] == "open"
    assert portal["interaction"]["state_source"] == "mllm_attribute_inference"
    assert "observation_evidence" not in portal["attributes"]
    assert portal["last_seen"] == 10.0
    assert portal["attributes"]["attribute_updated_at"] == 20.0


def test_superseded_attribute_frame_is_marked_stale_without_state_change():
    store = InteractionGraphStore(scene_id="test_scene")
    door = observation(
        instance_id="gt_000001",
        semantic_name="door",
        is_door=True,
        is_articulable=True,
        joint_type="hinge",
        joint_range=[0.0, 1.0],
        joint_value=0.0,
        frame_index=8,
    )
    store.update_observations([door], stamp=8.0, source_mode="realtime_gt_observation")
    reopened = dict(door)
    reopened["joint_value"] = 1.0
    reopened["frame_index"] = 9
    store.update_observations([reopened], stamp=9.0, source_mode="realtime_gt_observation")
    assert store.update_interaction_result(
        {
            "instance_id": "gt_000001",
            "event_id": "executor_opened_door",
            "state": "open",
            "source": "executor_verification",
        },
        stamp=10.0,
    )

    assert store.apply_attribute_patch(
        {
            "object_id": "gt_000001",
            "attribute_status": "ready",
            "request_sequence": 4,
            "observation_frame_index": 8,
            "interactable": True,
            "interaction_class": "portal",
            "coarse_state": "closed",
            "confidence": 0.9,
        },
        stamp=12.0,
    )
    portal = next(
        node for node in store.as_graph_dict(stamp=12.0)["nodes"] if node["type"] == "portal"
    )
    assert portal["attributes"]["attribute_status"] == "stale"
    assert portal["attributes"]["attribute_error"] == "observation_version_superseded"
    assert portal["interaction"]["state"] == "open"


def test_configured_interaction_geometry_is_recorded_for_minimal_gt_node() -> None:
    source_name = "fridge_house7"
    store = InteractionGraphStore(
        scene_id="test_scene",
        interaction_geometry_overrides={
            source_name: {
                "interaction_approach_axis_xy": [1.0, 0.0],
                "interaction_approach_pose_xyyaw": [8.25, 1.05, 3.141593],
                "interaction_reference_aabb_center": [7.09, 1.05, 0.76],
                "interaction_reference_aabb_size": [0.82, 0.88, 1.53],
                "source": "test_calibration",
            }
        },
    )
    store.update_observations(
        [
            observation(
                instance_id=source_name,
                source_object_name=source_name,
                semantic_name="fridge",
                category="fridge",
                is_receptacle=True,
                is_articulable=True,
                minimal_gt_observation=True,
            )
        ],
        source_mode="realtime_gt_observation",
    )

    node = next(
        item
        for item in store.as_graph_dict()["nodes"]
        if item["type"] == "container"
    )
    assert node["attributes"]["interaction_approach_axis_xy"] == [1.0, 0.0]
    assert node["attributes"]["interaction_approach_pose_xyyaw"] == [
        8.25,
        1.05,
        3.141593,
    ]
    assert node["attributes"]["interaction_geometry_source"] == "test_calibration"
