import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


def confirmed_portal():
    store = InteractionGraphStore(scene_id="provisional_geometry")
    detection = {"instance_id": "door1", "semantic_name": "door", "category": "door",
                 "room_id": 1, "position": [2, 0, 1], "aabb_center": [2, 0, 1],
                 "aabb_size": [0.1, 0.9, 2]}
    for stamp in (1.0, 2.0):
        store.update_observations([detection], stamp=stamp, source_mode="detector_online")
    portal = store.nodes["portal_door1"]
    store.apply_attribute_patch({"object_id": portal.id, "attribute_status": "ready",
                                 "confidence": 0.95, "source": "mllm_attribute_inference",
                                 "observed_object_name": "door", "interaction_class": "portal",
                                 "interactable": True, "coarse_state": "closed"}, stamp=2.1)
    return store, portal


def assert_local_geometry(store, room, side=1):
    assert all(math.isfinite(value) for value in room.aabb_center + room.aabb_size)
    assert room.aabb_center[:2] == pytest.approx([2 + side * store.portal_child_room_offset_m, 0])
    assert max(room.aabb_size[:2]) < 3
    assert room.attributes["observed_free_space"] is False
    assert room.attributes["cell_count"] == 0


def test_forced_room_is_local_and_never_scans_occ_for_its_synthetic_id(monkeypatch):
    store, portal = confirmed_portal()
    monkeypatch.setattr(store, "_default_room_center", lambda *args: pytest.fail("full-grid center fallback"))
    monkeypatch.setattr(store, "_default_room_size", lambda *args: pytest.fail("full-grid size fallback"))
    room = store.ensure_provisional_room_for_portal(portal)
    assert_local_geometry(store, room)
    assert portal.attributes["portal_child_room_id"] == room.room_id


@pytest.mark.parametrize("approach_x, side", [(1, 1), (3, -1)])
def test_opening_reuses_forced_room_identity(approach_x, side):
    store, portal = confirmed_portal()
    room = store.ensure_provisional_room_for_portal(portal)
    store.update_interaction_result({"node_id": portal.id, "action": "open", "success": True,
                                     "approach_goal_xyyaw": [approach_x, 0, 0]}, stamp=3.0)
    assert portal.attributes["portal_child_room_id"] == room.room_id
    assert [node.id for node in store.nodes.values() if node.attributes.get("is_potential_room")] == [room.id]
    assert_local_geometry(store, room, side=side)
    assert store.ensure_provisional_room_for_portal(portal).id == room.id
    assert_local_geometry(store, room, side=side)


def test_legacy_id_coordinate_room_is_repaired_in_place():
    store, portal = confirmed_portal()
    room = store._ensure_room_node(1000000)
    room.attributes.update({"is_potential_room": True, "parent_portal_id": portal.id})
    portal.attributes["potential_room_ids"] = [room.room_id]
    repaired = store.ensure_provisional_room_for_portal(portal)
    assert repaired.id == room.id
    assert_local_geometry(store, repaired)


def test_rejected_portal_removes_numeric_room_references():
    store, portal = confirmed_portal()
    room = store.ensure_provisional_room_for_portal(portal)
    store.apply_attribute_patch({"object_id": portal.id, "attribute_status": "ready", "confidence": 0.95,
                                 "source": "mllm_attribute_inference", "observed_object_name": "wall_panel",
                                 "interaction_class": "none", "interactable": False}, stamp=3.0)
    store.prune_unqualified_provisional_rooms()
    assert room.id not in store.nodes
    assert not portal.attributes.get("potential_room_ids")
    assert not portal.attributes.get("portal_child_room_id")
