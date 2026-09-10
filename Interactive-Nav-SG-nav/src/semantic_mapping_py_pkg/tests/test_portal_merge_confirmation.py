"""Track handoff must not delete the confirmed semantic doorway identity."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore


def _confirmed():
    store = InteractionGraphStore(scene_id="portal_merge")
    observation = {"instance_id": "door", "semantic_name": "door", "category": "door",
                   "room_id": 1, "position": [2, 0, 1], "aabb_center": [2, 0, 1],
                   "aabb_size": [.1, .9, 2.]}
    for stamp in (1., 2.):
        store.update_observations([observation], stamp=stamp, source_mode="detector_online")
    portal = store.nodes["portal_door"]
    assert store.apply_attribute_patch({"object_id": portal.id, "attribute_status": "ready",
        "confidence": .95, "source": "mllm_attribute_inference", "observed_object_name": "door",
        "interaction_class": "portal", "interactable": True, "coarse_state": "closed"}, stamp=2.1)
    return store, portal


@pytest.mark.parametrize("status", ["ready", "pending", "failed", "stale"])
def test_confirmed_identity_survives_more_frequent_unconfirmed_track(status):
    store, confirmed = _confirmed()
    room = store.ensure_provisional_room_for_portal(confirmed)
    if status != "ready":
        assert store.apply_attribute_patch({"object_id": confirmed.id,
            "attribute_status": status, "source": "mllm_attribute_inference"}, stamp=3.)
    before = copy.deepcopy(confirmed.attributes)
    new_track = copy.deepcopy(confirmed)
    new_track.id = "portal_new_track"
    new_track.observation_count = 20
    new_track.attributes = {key: value for key, value in new_track.attributes.items()
        if key.startswith("interaction_reference_")}
    new_track.attributes["instance_id"] = "new_track"
    confirmed.is_currently_visible = False
    new_track.is_currently_visible = True
    store.nodes[new_track.id] = new_track
    store._merge_duplicate_portal_nodes()
    assert confirmed.id in store.nodes
    assert new_track.id not in store.nodes
    assert confirmed.attributes["attribute_status"] == status
    assert confirmed.attributes["attribute_last_ready"] == before["attribute_last_ready"]
    assert confirmed.attributes["persistent_semantic_node"] is True
    assert confirmed.observation_count == 20  # Keep stronger detector evidence as well.
    assert confirmed.is_currently_visible is True
    store.prune_unqualified_provisional_rooms()
    assert room.id in store.nodes
    assert room.attributes["source_portal_id"] == confirmed.id


def test_unconfirmed_duplicates_keep_detector_ranking_and_do_not_invent_confirmation():
    store, first = _confirmed()
    first.attributes = {key: value for key, value in first.attributes.items()
                        if key.startswith("interaction_reference_")}
    second = copy.deepcopy(first)
    second.id = "portal_more_observations"
    second.observation_count = 10
    store.nodes[second.id] = second
    store._merge_duplicate_portal_nodes()
    assert first.id not in store.nodes
    assert second.id in store.nodes
    assert not second.attributes.get("persistent_semantic_node")
    assert "attribute_last_ready" not in second.attributes


def test_m1_refresh_result_still_routes_to_surviving_public_id():
    store, confirmed = _confirmed()
    assert store.apply_attribute_patch({"object_id": confirmed.id,
        "attribute_status": "pending", "source": "mllm_attribute_inference"}, stamp=3.)
    duplicate = copy.deepcopy(confirmed)
    duplicate.id = "portal_unconfirmed"
    duplicate.observation_count = 30
    duplicate.attributes = {key: value for key, value in duplicate.attributes.items()
                            if key.startswith("interaction_reference_")}
    store.nodes[duplicate.id] = duplicate
    store._merge_duplicate_portal_nodes()
    assert store.apply_attribute_patch({"object_id": confirmed.id, "attribute_status": "ready",
        "confidence": .96, "source": "mllm_attribute_inference", "observed_object_name": "door",
        "interaction_class": "portal", "interactable": True, "coarse_state": "open",
        "portal_aperture_evidence": {"open_aperture": "visible", "confidence": .96}}, stamp=4.)
    assert store.nodes[confirmed.id].attributes["attribute_status"] == "ready"
    assert store.nodes[confirmed.id].interaction["state"] == "open"


def test_unqualified_ready_duplicate_cannot_replace_pending_confirmed_state():
    store, confirmed = _confirmed()
    assert store.apply_attribute_patch({"object_id": confirmed.id,
        "attribute_status": "pending", "source": "mllm_attribute_inference"}, stamp=3.)
    before_attrs = copy.deepcopy(confirmed.attributes)
    before_interaction = copy.deepcopy(confirmed.interaction)
    duplicate = copy.deepcopy(confirmed)
    duplicate.id = "portal_weak_ready"
    duplicate.observation_count = 30
    duplicate.attributes.pop("attribute_last_ready", None)
    duplicate.attributes.update(attribute_status="ready", attribute_confidence=.1)
    duplicate.interaction["state"] = "open"
    store.nodes[duplicate.id] = duplicate

    store._merge_duplicate_portal_nodes()

    assert confirmed.id in store.nodes
    assert duplicate.id not in store.nodes
    assert confirmed.attributes["attribute_status"] == "pending"
    assert confirmed.attributes["attribute_confidence"] == before_attrs["attribute_confidence"]
    assert confirmed.attributes["attribute_last_ready"] == before_attrs["attribute_last_ready"]
    assert confirmed.interaction == before_interaction
