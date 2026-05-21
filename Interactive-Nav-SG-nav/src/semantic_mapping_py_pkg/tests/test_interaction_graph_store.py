from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.gt_observation_provider import build_gt_observation_batches
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


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
    assert len(first_graph["nodes"]) == 2

    store.update_observations(batches[1], source_mode="gt_replay")
    second_graph = store.as_graph_dict()
    assert len(second_graph["nodes"]) == 3
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
