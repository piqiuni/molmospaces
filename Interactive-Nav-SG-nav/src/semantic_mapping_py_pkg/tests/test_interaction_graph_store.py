from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.gt_observation_provider import build_gt_observation_batches
from semantic_mapping_py_pkg.graph_rules import infer_interaction_state, observation_from_detection
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
    assert portal["interaction"]["state"] == "closed"
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
        {"instance_id": "gt_000001", "state": "open", "source": "oracle_interaction"},
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
            "pre_state": "closed",
            "post_state": "open",
            "success": True,
            "execution_cost": 1.0,
            "verification_source": "oracle_interaction",
        }
    ]
    assert "joints" not in portal["attributes"]
    assert "primary_joint_name" not in portal["attributes"]
    assert "observation_evidence" in portal["attributes"]
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


def test_portal_state_uses_joint_delta_and_ajar_is_not_traversable():
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
    assert portal["interaction"]["state"] == "ajar"
    assert portal["interaction"]["traversable"] is False
    assert portal["interaction"]["requires_interaction"] is True

    opened = dict(closed)
    opened["joint_value"] = -0.8
    store.update_observations([opened], source_mode="realtime_gt_observation")
    portal = next(node for node in store.as_graph_dict()["nodes"] if node["type"] == "portal")
    assert portal["interaction"]["state"] == "open"
    assert portal["interaction"]["traversable"] is True
    assert portal["interaction"]["open_fraction"] == 0.8


def test_negative_joint_range_uses_zero_as_closed_endpoint():
    observation_data = {"is_movable_door": True}
    assert infer_interaction_state(
        "container", "hinge", [-2.8, 0.0], -2.8, observation_data
    ) == "open"
    assert infer_interaction_state(
        "container", "hinge", [-2.8, 0.0], 0.0, observation_data
    ) == "closed"


def test_initially_open_multi_joint_box_does_not_require_interaction():
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
    assert node["interaction"]["state"] == "open"
    assert node["interaction"]["requires_interaction"] is False
