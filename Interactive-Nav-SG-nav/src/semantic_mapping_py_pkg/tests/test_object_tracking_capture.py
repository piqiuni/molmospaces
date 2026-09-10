import copy
import math

import pytest

from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore


def detection(**extra):
    return {
        "semantic_class": "fridge", "confidence": 0.9, "instance_id": "fridge-1",
        "world_position": [1.0, 2.0, 1.0], "world_box3d_center": [1.0, 2.0, 1.0],
        "world_box3d_size": [0.7, 0.6, 1.8], "yaw": 0.1,
        **extra,
    }


def test_multiple_boxes_in_one_capture_cannot_confirm_one_track():
    store = ObjectMapStore(min_confirmations=2)
    store.update([detection()] * 10, stamp=1.0)
    assert len(store.objects) == 1
    assert store.objects[0]["observation_count"] == 1
    assert store.objects[0]["hit_streak"] == 1
    assert not store.as_tracked_detections()
    store.update([detection()], stamp=2.0)
    assert store.objects[0]["observation_count"] == 2
    assert len(store.as_tracked_detections()) == 1


def test_repeated_or_older_capture_cannot_confirm_move_or_expire_tracks():
    store = ObjectMapStore(min_confirmations=2)
    store.update([detection()], stamp=10.0)
    before = copy.deepcopy(store.objects)
    for stamp in (10.0, 9.0, float("nan"), float("inf")):
        assert store.update([detection(world_position=[9.0, 9.0, 9.0])], stamp) is False
        assert store.update([], stamp) is False
        assert store.objects == before
    store.update([detection()], stamp=11.0)
    assert store.objects[0]["is_confirmed"]
    store.update([], stamp=12.0)
    assert store.objects[0]["miss_streak"] == 1
    store.update([], stamp=12.0)
    assert store.objects[0]["miss_streak"] == 1


@pytest.mark.parametrize("extra", [
    {"world_position": [float("nan"), 0.0, 1.0]},
    {"world_box3d_center": [1.0, float("inf"), 1.0]},
    {"world_box3d_size": [-1.0, 1.0, 1.0]},
    {"world_box3d_size": [float("inf"), 1.0, 1.0]},
    {"yaw": float("nan")}, {"confidence": float("inf")},
    {"confidence": "invalid"},
    {"world_position": ["bad", 2.0, 1.0]},
    {"world_position": [1.0, 2.0]},
    {"world_position": {"x": 1.0}},
    {"world_box3d_center": "bad"},
    {"world_box3d_size": {"x": 1.0, "y": "bad", "z": 1.0}},
    {"world_box3d_center": None, "aabb_center": [float("nan"), 2.0, 1.0]},
    {"world_box3d_size": None, "aabb_size": [1.0, -1.0, 1.0]},
])
def test_invalid_geometry_cannot_poison_history_or_stop_other_tracks(extra):
    store = ObjectMapStore(min_confirmations=2)
    store.update([detection()], stamp=1.0)
    before = copy.deepcopy(store.objects[0])
    store.update([detection(**extra), detection(instance_id="second", world_position=[4.0, 2.0, 1.0],
                                            world_box3d_center=[4.0, 2.0, 1.0])], stamp=2.0)
    first = store.objects[0]
    for field in ("coord_history", "yaw_history", "aabb_center", "aabb_size", "observation_count"):
        assert first[field] == before[field]
    assert len(store.objects) == 2
    assert all(math.isfinite(value) for obj in store.objects for value in obj["coord"])


def test_positionless_detection_cannot_create_an_origin_track():
    store = ObjectMapStore()
    store.update([{"semantic_class": "fridge", "confidence": 0.9, "bbox": [1, 2, 30, 40]}], stamp=1.0)
    assert not store.objects


@pytest.mark.parametrize("center_key,size_key", [
    ("world_box3d_center", "world_box3d_size"), ("aabb_center", "aabb_size"),
])
def test_box_only_observation_uses_measured_center_not_origin(center_key, size_key):
    store = ObjectMapStore(min_confirmations=1)
    store.update([{"semantic_class": "fridge", "confidence": 0.9,
                   center_key: [3.0, 4.0, 1.0], size_key: [0.5, 0.6, 1.8]}], stamp=1.0)
    obj = store.objects[0]
    assert obj["coord"] == obj["aabb_center"] == [3.0, 4.0, 1.0]
    assert obj["aabb_size"] == [0.5, 0.6, 1.8]


def test_existing_episode_reset_allows_capture_clock_restart():
    store = ObjectMapStore()
    store.update([detection()], stamp=100.0)
    store.objects = []
    store.next_id = 1
    assert store.update([detection()], stamp=1.0)
    assert store.objects[0]["observation_count"] == 1


def test_explicit_reset_discards_names_indexes_and_capture_watermark():
    store = ObjectMapStore(min_confirmations=3, match_distance=0.7)
    store.update([detection(instance_id="track_0001")], stamp=100.0)
    store.set_m1_canonical_label("track_0001", "water_dispenser")
    store.reset()
    assert not store.objects and not store.m1_canonical_labels
    assert store._last_update_stamp is None
    assert store._match_identity_index is None
    assert not store._match_object_index
    assert store.min_confirmations == 3 and store.match_distance == 0.7
    store.update([detection(instance_id="track_0001")], stamp=1.0)
    assert store.objects[0]["semantic_name"] == "fridge"
    assert store.objects[0]["observation_count"] == 1


def test_accepted_name_survives_raw_detections_without_tracker_ids():
    store = ObjectMapStore()
    raw = detection(instance_id="")
    store.update([raw], stamp=1.0)
    store.update([raw], stamp=2.0)
    store.set_m1_canonical_label("track_0001", "water_dispenser")
    assert store.objects[0]["semantic_name"] == "water_dispenser"
    for stamp in (3.0, 4.0, 5.0):
        store.update([raw], stamp=stamp)
    assert len(store.objects) == 1
    assert store.as_tracked_detections()[0]["semantic_class"] == "water_dispenser"
