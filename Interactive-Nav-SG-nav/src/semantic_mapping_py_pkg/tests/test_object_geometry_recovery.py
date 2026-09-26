"""Physical RGB-D budgets are not missed detections or permanent box locks."""

import pytest

from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore


def detection(**extra):
    return {
        "semantic_class": "cabinet", "instance_id": "cabinet_1", "confidence": 0.8,
        "world_position": [2.0, 1.0, 1.0], "world_box3d_center": [2.0, 1.0, 1.0],
        "world_box3d_size": [0.8, 0.6, 1.8], "bbox": [100, 100, 300, 400],
        "depth_valid_points": 128, **extra,
    }


def deferred(**extra):
    return {"semantic_class": "cabinet", "bbox": [100, 100, 300, 400],
            "geometry_skipped": True, "geometry_skip_reason": "geometry_budget_ms", **extra}


def physical_store(**extra):
    return ObjectMapStore(min_confirmations=2, confirmed_geometry_reacquisition=True,
                          stable_box_recovery_confirmations=3, **extra)


def current(store):
    return store.as_tracked_detections(min_observations=2, currently_observed_only=True)


def bottle(x, z, bbox):
    return detection(semantic_class="bottle", instance_id="",
                     world_position=[x, 1.0, z], world_box3d_center=[x, 1.0, z],
                     world_box3d_size=[0.08, 0.08, 0.20], bbox=bbox)


def test_xyz_size_matching_keeps_adjacent_and_stacked_bottles():
    store = physical_store(size_aware_xyz_matching=True)
    boxes = [[0, 0, 20, 50], [25, 0, 45, 50], [0, 60, 20, 110]]
    for frame in range(3):
        shift = frame * .002
        items = [bottle(2 + shift, .5, boxes[0]),
                 bottle(2.12 + shift, .5, boxes[1]),
                 bottle(2 + shift, .8, boxes[2])]
        store.update(items[::-1] if frame % 2 else items, 1 + frame * .1)
    assert len(current(store)) == 3
    assert all(item["observation_count"] == 3 for item in current(store))


def test_same_frame_distinct_boxes_do_not_disappear_when_nearest_track_used():
    store = physical_store(size_aware_xyz_matching=True)
    for frame in range(3):
        store.update([bottle(2, .5, [0, 0, 20, 50]),
                      bottle(2.03, .5, [25, 0, 45, 50])], 1 + frame * .1)
    assert len(current(store)) == 2


def test_true_duplicate_box_does_not_confirm_twice():
    store = physical_store(size_aware_xyz_matching=True)
    item = bottle(2, .5, [0, 0, 20, 50])
    store.update([item, item], 1)
    assert len(store.objects) == 1
    assert store.objects[0]["observation_count"] == 1


def test_visible_content_inside_closed_container_is_geometrically_eligible():
    from types import SimpleNamespace
    from semantic_mapping_py_pkg.interaction_graph_store import (
        _container_contains, _is_plausible_container_content,
    )
    container = SimpleNamespace(aabb_center=[0, 0, .8], aabb_size=[.7, .6, 1.6])
    for z in (.3, .6, .9):
        obj = SimpleNamespace(label="bottle", name="bottle", aabb_center=[0, 0, z],
                              aabb_size=[.08, .08, .2])
        assert _container_contains(obj, container)
        assert _is_plausible_container_content(obj, container)


@pytest.mark.parametrize("counts,expected", [
    ({"mask_area": 150000}, 150000),
    ({"mask_area": 150000, "visible_pixels": 120000}, 120000),
    ({"mask_area": 0}, 0),
    ({}, 3000),
])
def test_sampled_mask_does_not_replace_true_visible_area(counts, expected):
    store = physical_store()
    item = detection(semantic_class="fridge", bbox=[0, 0, 400, 700],
                     mask={"rows": [10] * 3000, "cols": [10] * 3000}, **counts)
    store.update([item], 1.0)
    store.update([item], 1.1)
    tracked = current(store)[0]
    assert tracked["visible_pixels"] == expected
    assert tracked["visible_fraction"] == pytest.approx(expected / 280000)


def test_three_supported_measurements_can_replace_a_bad_initial_huge_box():
    store = physical_store()
    huge = detection(world_box3d_size=[6.0, 3.0, 2.0])
    store.update([huge], 1.0)
    store.update([huge], 1.1)
    store.update([detection()], 1.2)
    store.update([detection()], 1.3)
    assert store.objects[0]["aabb_size"] == [6.0, 3.0, 2.0]
    store.update([detection()], 1.4)
    assert len(store.objects) == 1
    assert store.objects[0]["aabb_size"] == pytest.approx([0.8, 0.6, 1.8])
    assert store.objects[0]["box_recovery_count"] == 1
    assert current(store)[0]["world_box3d_size"] == {"x": 0.8, "y": 0.6, "z": 1.8}


def test_single_geometry_outlier_does_not_replace_good_geometry():
    store = physical_store()
    store.update([detection()], 1.0)
    store.update([detection()], 1.1)
    store.update([detection(world_box3d_size=[6.0, 3.0, 2.0])], 1.2)
    assert store.objects[0]["aabb_size"] == pytest.approx([0.8, 0.6, 1.8])
    store.update([detection()], 1.3)
    assert not store.objects[0].get("box_recovery_window")
    assert not store.objects[0].get("box_recovery_count")


@pytest.mark.parametrize("weak", [{"confidence": 0.1}, {"depth_valid_points": 10}])
def test_weak_repeated_measurements_do_not_reset_the_stable_box(weak):
    store = physical_store()
    huge = detection(world_box3d_size=[6.0, 3.0, 2.0])
    store.update([huge], 1.0)
    store.update([huge], 1.1)
    for index in range(10):
        store.update([detection(**weak)], 1.2 + index * 0.1)
    assert store.objects[0]["aabb_size"] == [6.0, 3.0, 2.0]


def test_disagreeing_boxes_and_real_misses_break_recovery_consensus():
    store = physical_store()
    store.update([detection(world_box3d_size=[6.0, 3.0, 2.0])], 1.0)
    store.update([detection()], 1.1)
    store.update([detection()], 1.2)
    store.update([detection(world_box3d_size=[1.5, 1.0, 1.8])], 1.3)
    assert len(store.objects[0]["box_recovery_window"]) == 1
    store.update([], 1.4)
    assert not store.objects[0].get("box_recovery_window")
    assert store.objects[0]["aabb_size"] == [6.0, 3.0, 2.0]


def test_budget_omission_does_not_add_observations_or_publish_old_geometry():
    store = physical_store()
    store.update([detection()], 1.0)
    store.update([detection()], 1.1)
    store.update([], 1.2, geometry_deferred_detections=[deferred()])
    obj = store.objects[0]
    assert obj["hit_streak"] == 2
    assert obj["observation_count"] == 2
    assert obj["last_seen"] == 1.1
    assert not current(store)
    store.update([detection()], 1.3)
    assert current(store)[0]["observation_count"] == 3


def test_a_deferred_detection_alone_cannot_create_or_confirm_a_track():
    store = physical_store()
    store.update([deferred()], 1.0)
    assert not store.objects
    store.update([detection()], 1.1)
    store.update([deferred()], 1.2)
    assert store.objects[0]["observation_count"] == 1
    assert not store.objects[0]["is_confirmed"]
    assert not current(store)
    store.update([detection()], 1.3)
    assert len(current(store)) == 1


@pytest.mark.parametrize("wrong", [
    {"geometry_skip_reason": "implausibly_large_box"},
    {"geometry_skip_reason": "missing_capture_pose"},
    {"semantic_class": "chair"}, {"bbox": [400, 400, 500, 600]},
    {"bbox": ["invalid", 0, 10, 10]},
])
def test_invalid_or_unrelated_skips_are_not_treated_as_budget_continuity(wrong):
    store = physical_store()
    store.update([detection()], 1.0)
    store.update([], 1.1, geometry_deferred_detections=[deferred(**wrong)])
    assert store.objects[0]["hit_streak"] == 0
    store.update([detection()], 1.2)
    assert not current(store)


def test_confirmed_object_reacquires_on_one_new_geometry_without_reconfirming():
    store = physical_store()
    store.update([detection()], 1.0)
    store.update([detection()], 1.1)
    store.update([], 1.2)
    store.update([detection(world_box3d_size=[0.9, 0.7, 1.9])], 1.3)
    assert store.objects[0]["hit_streak"] == 1
    assert len(current(store)) == 1
    assert current(store)[0]["world_box3d_size"]["x"] > 0.8


def test_first_admission_still_requires_two_uninterrupted_observations():
    store = physical_store()
    store.update([detection()], 1.0)
    store.update([], 1.1)
    store.update([detection()], 1.2)
    assert store.objects[0]["observation_count"] == 2
    assert not current(store)
    store.update([detection()], 1.3)
    assert len(current(store)) == 1


def test_budget_receipts_do_not_refresh_the_unconfirmed_track_ttl():
    store = physical_store(stale_after_sec=1.0)
    store.update([detection()], 1.0)
    for index in range(12):
        store.update([], 1.1 + index * 0.1, geometry_deferred_detections=[deferred()])
    assert not store.objects


def test_mapper_passes_budget_hints_without_publishing_old_world_boxes():
    import json
    import threading
    from types import SimpleNamespace

    pytest.importorskip("rospy")
    from semantic_mapping_node import SemanticMappingNode

    node = object.__new__(SemanticMappingNode)
    node.enable_object_mapping = True
    node.lock = threading.RLock()
    node.object_store = physical_store()
    node.graph_min_observations = 2
    graph_receipts, published = [], []
    node.graph_store = SimpleNamespace(
        nodes={}, update_observations=lambda values, **kwargs: graph_receipts.append(values)
    )
    node.tracked_detections_pub = SimpleNamespace(publish=published.append)
    node._record_component_timing = lambda *args: None
    node._update_room_portal_hints = lambda *args, **kwargs: False

    def receipt(stamp, values, hints=None):
        node._process_object_message(SimpleNamespace(data=json.dumps({
            "stamp_sec": stamp, "detections": values,
            "geometry_deferred_detections": hints or [],
        })))

    receipt(1.0, [detection()])
    receipt(1.1, [detection()])
    receipt(1.2, [], [deferred()])
    assert node.object_store.objects[0]["hit_streak"] == 2
    assert graph_receipts[-1] == []
    assert json.loads(published[-1].data)["detections"] == []
    receipt(1.3, [detection()])
    assert len(graph_receipts[-1]) == 1
    assert json.loads(published[-1].data)["detections"][0]["observation_count"] == 3
