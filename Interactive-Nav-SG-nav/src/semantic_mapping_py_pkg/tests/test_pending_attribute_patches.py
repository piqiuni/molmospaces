from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.pending_attribute_patches import PendingAttributePatchCache


def observation(instance_id):
    return {
        "observation_id": f"obs_{instance_id}",
        "instance_id": instance_id,
        "semantic_name": "door",
        "category": "door",
        "confidence": 1.0,
        "position": [0.0, 0.0, 1.0],
        "aabb_center": [0.0, 0.0, 1.0],
        "aabb_size": [0.9, 0.1, 2.0],
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
        "name": instance_id,
    }


def ready_patch(object_id, request_sequence, state="closed"):
    return {
        "object_id": object_id,
        "attribute_status": "ready",
        "request_sequence": request_sequence,
        "interactable": True,
        "interaction_class": "portal",
        "coarse_state": state,
        "interaction_parts": [],
        "confidence": 0.9,
        "source": "mllm_attribute_inference",
    }


def graph_node(store, object_id):
    return next(
        node
        for node in store.nodes.values()
        if node.attributes.get("instance_id") == object_id
    )


def test_patch_before_node_creation_is_replayed_for_same_episode():
    store = InteractionGraphStore(scene_id="test_scene")
    store.reset(episode_id="episode_1", source_mode="realtime_gt_observation")
    cache = PendingAttributePatchCache(max_entries=8)
    cache.update_lifecycle(
        episode_id="episode_1", episode_generation=1, episode_active=True
    )

    assert not cache.apply_or_store(
        store,
        ready_patch("door_1", 4),
        episode_id="episode_1",
        episode_generation=1,
        stamp=2.0,
    )
    assert len(cache) == 1

    store.update_observations(
        [observation("door_1")],
        stamp=3.0,
        source_mode="realtime_gt_observation",
    )
    assert cache.replay(store)
    assert len(cache) == 0
    node = graph_node(store, "door_1")
    assert node.attributes["attribute_status"] == "ready"
    assert node.attributes["attribute_request_sequence"] == 4
    assert node.interaction["state"] == "closed"


def test_cache_keeps_only_latest_request_per_episode_object():
    store = InteractionGraphStore(scene_id="test_scene")
    store.reset(episode_id="episode_1", source_mode="realtime_gt_observation")
    cache = PendingAttributePatchCache(max_entries=8)
    cache.update_lifecycle(
        episode_id="episode_1", episode_generation=1, episode_active=True
    )

    cache.apply_or_store(
        store,
        ready_patch("door_1", 5, state="open"),
        episode_id="episode_1",
        episode_generation=1,
    )
    cache.apply_or_store(
        store,
        ready_patch("door_1", 4, state="closed"),
        episode_id="episode_1",
        episode_generation=1,
    )
    assert len(cache) == 1

    store.update_observations(
        [observation("door_1")], source_mode="realtime_gt_observation"
    )
    assert cache.replay(store)
    node = graph_node(store, "door_1")
    assert node.attributes["attribute_request_sequence"] == 5
    assert node.interaction["state"] == "open"


def test_cache_is_bounded_by_object_count():
    store = InteractionGraphStore(scene_id="test_scene")
    store.reset(episode_id="episode_1", source_mode="realtime_gt_observation")
    cache = PendingAttributePatchCache(max_entries=2)
    cache.update_lifecycle(
        episode_id="episode_1", episode_generation=1, episode_active=True
    )

    for index in range(3):
        cache.apply_or_store(
            store,
            ready_patch(f"door_{index}", index + 1),
            episode_id="episode_1",
            episode_generation=1,
        )

    assert len(cache) == 2
    assert cache.pending_keys() == [
        (1, "episode_1", "door_1"),
        (1, "episode_1", "door_2"),
    ]


def test_future_generation_waits_for_matching_graph_and_stale_patch_is_rejected():
    store = InteractionGraphStore(scene_id="test_scene")
    store.reset(episode_id="episode_1", source_mode="realtime_gt_observation")
    store.update_observations(
        [observation("shared_door")], source_mode="realtime_gt_observation"
    )
    cache = PendingAttributePatchCache(max_entries=8)
    cache.update_lifecycle(
        episode_id="episode_1", episode_generation=1, episode_active=True
    )

    assert not cache.apply_or_store(
        store,
        ready_patch("shared_door", 8),
        episode_id="episode_2",
        episode_generation=2,
    )
    assert len(cache) == 1
    assert "attribute_status" not in graph_node(store, "shared_door").attributes

    cache.update_lifecycle(
        episode_id="episode_2", episode_generation=2, episode_active=True
    )
    assert not cache.replay(store)
    store.reset(episode_id="episode_2", source_mode="realtime_gt_observation")
    store.update_observations(
        [observation("shared_door")], source_mode="realtime_gt_observation"
    )
    assert cache.replay(store)
    assert graph_node(store, "shared_door").attributes["attribute_request_sequence"] == 8

    assert not cache.apply_or_store(
        store,
        ready_patch("shared_door", 99),
        episode_id="episode_1",
        episode_generation=1,
    )
    assert len(cache) == 0
    assert graph_node(store, "shared_door").attributes["attribute_request_sequence"] == 8


def test_inactive_lifecycle_clears_pending_patches():
    store = InteractionGraphStore(scene_id="test_scene")
    store.reset(episode_id="episode_1", source_mode="realtime_gt_observation")
    cache = PendingAttributePatchCache(max_entries=8)
    cache.update_lifecycle(
        episode_id="episode_1", episode_generation=1, episode_active=True
    )
    cache.apply_or_store(
        store,
        ready_patch("door_1", 1),
        episode_id="episode_1",
        episode_generation=1,
    )
    assert len(cache) == 1

    cache.update_lifecycle(episode_generation=1, episode_active=False)
    assert len(cache) == 0
    assert not cache.apply_or_store(
        store,
        ready_patch("door_1", 2),
        episode_id="episode_1",
        episode_generation=1,
    )
