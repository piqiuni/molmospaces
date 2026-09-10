"""Parent lifetime under occlusion and changing room/container geometry."""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


def make_scene():
    store = InteractionGraphStore(scene_id="hierarchy_test")
    drawer = {
        "instance_id": "drawer_1", "semantic_name": "drawer", "room_id": 1,
        "is_receptacle": True, "is_articulable": True,
        "position": [1, 1, 0.5], "aabb_center": [1, 1, 0.5],
        "aabb_size": [1, 1, 1],
    }
    pencil = {
        "instance_id": "pencil_1", "semantic_name": "pencil", "room_id": 1,
        "position": [1, 1.35, 0.5], "aabb_center": [1, 1.35, 0.5],
        "aabb_size": [0.05, 0.05, 0.05],
    }
    store.update_observations([drawer, pencil], stamp=1.0, source_mode="realtime_gt_observation")
    assert store.nodes["object_pencil_1"].parent_id == "container_drawer_1"
    return store, drawer, pencil


def has_containment(store):
    return any(edge.src_id == "container_drawer_1" and edge.relation == "contains"
               and edge.dst_id == "object_pencil_1" for edge in store.edges.values())


def test_hidden_content_keeps_parent_when_closed_drawer_box_shrinks():
    store, drawer, pencil = make_scene()
    closed_drawer = {**drawer, "aabb_size": [1, 0.2, 1]}
    store.update_observations([closed_drawer], stamp=2.0, source_mode="realtime_gt_observation")
    assert not store.nodes["object_pencil_1"].is_currently_visible
    assert store.nodes["object_pencil_1"].parent_id == "container_drawer_1"
    assert has_containment(store)
    store.update_observations([closed_drawer], stamp=3.0, source_mode="realtime_gt_observation")
    assert has_containment(store)
    # A fresh visible observation outside the current box may revoke the relation.
    store.update_observations([closed_drawer, pencil], stamp=4.0, source_mode="realtime_gt_observation")
    assert not has_containment(store)
    assert store.nodes["object_pencil_1"].parent_id == "room_1"


def test_orphan_parent_does_not_point_to_retired_room():
    store, _, _ = make_scene()
    room = store.nodes["room_1"]
    room.attributes["active"] = False
    store.nodes["container_drawer_1"].type = "object"
    store._rebuild_relations(now=2.0)
    obj = store.nodes["object_pencil_1"]
    assert obj.parent_id == store._ensure_scene_node().id
    assert not has_containment(store)
    assert not room.attributes["active"]


def test_deleted_container_cannot_retain_hidden_child():
    store, _, _ = make_scene()
    store.nodes["object_pencil_1"].is_currently_visible = False
    del store.nodes["container_drawer_1"]
    store._rebuild_relations(now=2.0)
    assert store.nodes["object_pencil_1"].parent_id == "room_1"
    assert not has_containment(store)


def test_new_overlapping_container_does_not_claim_hidden_content():
    store, drawer, _ = make_scene()
    closed_drawer = {**drawer, "aabb_size": [1, 0.2, 1]}
    neighbor = {**drawer, "instance_id": "drawer_2"}
    store.update_observations([closed_drawer, neighbor], stamp=2.0, source_mode="realtime_gt_observation")
    assert store.nodes["object_pencil_1"].parent_id == "container_drawer_1"
    assert has_containment(store)


def test_parent_cache_detects_subcentimeter_boundary_crossing():
    store, _, _ = make_scene()
    obj = store.nodes["object_pencil_1"]
    obj.aabb_size = [0.1, 0.1, 0.1]
    obj.aabb_center = obj.centroid = [1.459, 1.0, 0.5]
    store._rebuild_relations(now=2.0)
    assert has_containment(store)
    obj.aabb_center = obj.centroid = [1.461, 1.0, 0.5]
    store._rebuild_relations(now=3.0)
    assert not has_containment(store)


def test_parent_cache_retains_one_version_per_object():
    store, _, _ = make_scene()
    obj = store.nodes["object_pencil_1"]
    for index in range(300):
        obj.aabb_center = obj.centroid = [1 + index * 0.01, 1.0, 0.5]
        store._rebuild_relations(now=2.0 + index)
    assert len(store._parent_relation_cache) == 1
    del store.nodes[obj.id]
    store._rebuild_relations(now=400.0)
    assert not store._parent_relation_cache
