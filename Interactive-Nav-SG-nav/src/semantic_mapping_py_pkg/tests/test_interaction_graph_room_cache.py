from __future__ import annotations

import pathlib
import sys


PACKAGE_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))

from semantic_mapping_py_pkg import interaction_graph_store as graph_module  # noqa: E402
from semantic_mapping_py_pkg.interaction_graph_store import (  # noqa: E402
    InteractionGraphStore,
)


class _GridInfo:
    width = 8
    height = 8
    resolution = 1.0

    class _Origin:
        class _Position:
            x = 0.0
            y = 0.0

        position = _Position()

    origin = _Origin()


def test_portal_room_probe_reuses_same_room_grid_epoch(monkeypatch):
    calls = {"count": 0}
    original = graph_module.world_to_grid

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_module, "world_to_grid", counted)
    scene_data = [1, 1, 1, 1, 2, 2, 2, 2] * _GridInfo.height
    store = InteractionGraphStore(scene_id="cache_test")
    store.update_room_grid(_GridInfo(), scene_data, [100] * len(scene_data))
    observation = {
        "instance_id": "door_cache",
        "semantic_name": "door",
        "is_door": True,
        "position": [4.0, 4.0, 1.0],
        "aabb_center": [4.0, 4.0, 1.0],
        "aabb_size": [0.2, 1.0, 2.0],
        "yaw": 1.5707963267948966,
    }
    store.update_observations([observation], source_mode="detector_online")
    first_count = calls["count"]
    store.update_observations(
        [{**observation, "frame_index": 2}], source_mode="detector_online"
    )
    second_count = calls["count"] - first_count

    assert first_count > 0
    assert second_count == 0
    assert len(store._portal_room_ids_cache) == 1

    changed_scene_data = list(scene_data)
    changed_scene_data[0] = 2
    store.update_room_grid(
        _GridInfo(), changed_scene_data, [100] * len(changed_scene_data)
    )
    store.update_observations(
        [{**observation, "frame_index": 3}], source_mode="detector_online"
    )
    assert calls["count"] > first_count


def test_first_room_grid_statistic_reuses_reduced_geometry(monkeypatch):
    """A new room must not rescan the full grid for throwaway defaults."""

    store = InteractionGraphStore(scene_id="first_room_geometry")

    def unexpected_default(*_args, **_kwargs):
        raise AssertionError("room-grid statistic should provide initial geometry")

    monkeypatch.setattr(store, "_default_room_center", unexpected_default)
    monkeypatch.setattr(store, "_default_room_size", unexpected_default)

    scene_data = [1] * (_GridInfo.width * _GridInfo.height)
    store.update_room_grid(_GridInfo(), scene_data, [100] * len(scene_data))

    room = store.nodes["room_1"]
    # Grid cells are represented by their centers: [0.5, 7.5] in both axes.
    assert room.aabb_center == [4.0, 4.0, 0.1]
    assert room.aabb_size == [8.0, 8.0, 0.2]


def test_detector_reuse_does_not_count_as_new_room_grid_observation():
    """YOLO refreshes must not satisfy the room two-frame gate."""
    scene_data = [1] * (_GridInfo.width * _GridInfo.height)
    store = InteractionGraphStore(scene_id="room_count_test")
    store.update_room_grid(_GridInfo(), scene_data, [100] * len(scene_data))
    room = store.nodes["room_1"]
    assert room.attributes["room_observation_count"] == 1

    observation = {
        "instance_id": "chair_1",
        "semantic_name": "chair",
        "position": [2.0, 2.0, 0.5],
        "aabb_center": [2.0, 2.0, 0.5],
        "aabb_size": [0.4, 0.4, 1.0],
    }
    store.update_observations([observation], stamp=1.0, source_mode="detector_online")
    store.update_observations([{**observation, "frame_index": 2}], stamp=2.0, source_mode="detector_online")
    assert room.attributes["room_observation_count"] == 1

    # A second accepted room grid, even with identical cell content, is the
    # event that advances the room temporal gate.
    store.update_room_grid(_GridInfo(), list(scene_data), [100] * len(scene_data))
    assert room.attributes["room_observation_count"] == 2


def test_spatial_track_index_preserves_no_id_association():
    """Unidentified nearby boxes still reuse one track without a full scan."""

    store = InteractionGraphStore(scene_id="spatial_track_test", match_distance=0.5)
    first = {
        "semantic_name": "chair",
        "position": [1.0, 1.0, 0.5],
        "aabb_center": [1.0, 1.0, 0.5],
        "aabb_size": [0.4, 0.4, 1.0],
    }
    second = {**first, "position": [1.15, 1.05, 0.5], "aabb_center": [1.15, 1.05, 0.5]}
    store.update_observations([first], stamp=1.0, source_mode="detector_online")
    store.update_observations([second], stamp=2.0, source_mode="detector_online")

    object_nodes = [node for node in store.nodes.values() if node.type == "object"]
    assert len(object_nodes) == 1
    assert object_nodes[0].observation_count == 2


def test_spatial_track_index_does_not_merge_far_same_label_boxes():
    store = InteractionGraphStore(scene_id="spatial_far_test", match_distance=0.5)
    observations = [
        {
            "semantic_name": "chair",
            "position": [float(index) * 2.0, 0.0, 0.5],
            "aabb_center": [float(index) * 2.0, 0.0, 0.5],
            "aabb_size": [0.4, 0.4, 1.0],
        }
        for index in range(8)
    ]
    store.update_observations(observations, stamp=1.0, source_mode="detector_online")
    assert len([node for node in store.nodes.values() if node.type == "object"]) == 8


def test_parent_relation_cache_reuses_and_invalidates_geometry(monkeypatch):
    store = InteractionGraphStore(scene_id="parent_cache_test")
    support = {
        "instance_id": "table_1",
        "semantic_name": "table",
        "position": [0.0, 0.0, 0.4],
        "aabb_center": [0.0, 0.0, 0.4],
        "aabb_size": [1.0, 1.0, 0.8],
        "room_id": 1,
    }
    object_detection = {
        "instance_id": "apple_1",
        "semantic_name": "apple",
        "position": [0.0, 0.0, 0.85],
        "aabb_center": [0.0, 0.0, 0.85],
        "aabb_size": [0.1, 0.1, 0.1],
        "room_id": 1,
    }
    calls = {"count": 0}
    original = store._find_parent_node

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_find_parent_node", counted)
    store.update_observations([support, object_detection], stamp=1.0)
    first_count = calls["count"]
    store.update_observations(
        [support, object_detection], stamp=2.0, source_mode="detector_online"
    )
    assert first_count > 0
    assert calls["count"] == first_count

    moved_support = {
        **support,
        "position": [2.0, 0.0, 0.4],
        "aabb_center": [2.0, 0.0, 0.4],
    }
    store.update_observations(
        [moved_support, object_detection], stamp=3.0, source_mode="detector_online"
    )
    assert calls["count"] > first_count
