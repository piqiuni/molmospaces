from dataclasses import replace
from io import BytesIO
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_decision_py_pkg.navigation_clearance import ArrivalClearanceGrid


def message(values=None):
    values = np.zeros((20, 20), dtype=np.int8) if values is None else values
    return SimpleNamespace(
        data=values.ravel(), header=SimpleNamespace(frame_id="map"),
        info=SimpleNamespace(width=values.shape[1], height=values.shape[0], resolution=.1,
            origin=SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
                                   orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))))


def ready(values=None):
    grid = ArrivalClearanceGrid.from_message(message(values), costmap=True)
    assert grid.components.shape == grid.obstacle_distance.shape
    return grid


@pytest.mark.parametrize("before,after,components_shared,distance_shared", [
    (0, 0, True, True), (0, 42, True, True), (99, 98, False, True),
    (100, 99, True, False), (-1, 100, True, True), (-1, 0, False, False),
    (0, 100, False, False),
])
@pytest.mark.parametrize("incremental", [False, True])
def test_reuse_requires_exact_respective_masks(before, after, components_shared, distance_shared, incremental):
    values = np.zeros((20, 20), dtype=np.int8)
    values[10, 10] = before
    old = ready(values)
    values[10, 10] = after
    if incremental:
        new = old.updated(SimpleNamespace(x=10, y=10, width=1, height=1, data=[after]))
    else:
        new = ArrivalClearanceGrid.from_message(message(values), costmap=True, previous=old)
    assert new is not old and not np.shares_memory(old.values, new.values)
    assert old.values[10, 10] == before and new.values[10, 10] == after
    assert (old.components is new.components) is components_shared
    assert (old.obstacle_distance is new.obstacle_distance) is distance_shared
    reference = ready(values)
    assert np.array_equal(reference.components, new.components)
    assert np.array_equal(reference.obstacle_distance, new.obstacle_distance)
    assert new.check((1.05, 1.05), .15) == reference.check((1.05, 1.05), .15)
    assert new.reachable((.5, .5), (1.05, 1.05)) == reference.reachable((.5, .5), (1.05, 1.05))
    assert not new.values.flags.writeable
    assert not new.components.flags.writeable
    assert not new.obstacle_distance.flags.writeable


@pytest.mark.parametrize("field,value", [("resolution", .05), ("origin_x", 1.),
    ("origin_y", 1.), ("origin_yaw", .1), ("frame_id", "odom"),
    ("occupied_threshold", 50), ("inscribed_threshold", None)])
def test_geometry_and_threshold_changes_never_share(field, value):
    old = ready()
    new = replace(old, **{field: value})._reuse_preprocessing(old)
    assert "components" not in new.__dict__
    assert "obstacle_distance" not in new.__dict__
    assert old.components is not new.components


def test_shape_change_never_shares():
    old = ready()
    new = ArrivalClearanceGrid.from_message(message(np.zeros((10, 40), dtype=np.int8)),
                                            costmap=True, previous=old)
    assert "components" not in new.__dict__
    assert new.components.shape == (10, 40)


def test_uncomputed_cache_stays_lazy():
    old = ArrivalClearanceGrid.from_message(message(), costmap=True)
    new = ArrivalClearanceGrid.from_message(message(), costmap=True, previous=old)
    for grid in (old, new):
        assert "components" not in grid.__dict__ and "obstacle_distance" not in grid.__dict__


def test_mutable_direct_input_cannot_share_stale_cache():
    old = replace(ready(), values=np.zeros((20, 20), dtype=np.int16))
    assert old.components[0, 0] != 0
    old.values[0, 0] = 100
    new = ArrivalClearanceGrid.from_message(message(old.values), costmap=True, previous=old)
    assert old.components is not new.components
    assert new.components[0, 0] == 0


def test_candidate_callback_refreshes_receipt_time_and_current_values(monkeypatch):
    pytest.importorskip("rospy")
    import semantic_candidate_node as module
    node = object.__new__(module.SemanticCandidateNode)
    node.clearance_lock = threading.RLock()
    node.clearance_maps = {"global": (ready(), 1.)}
    old = node.clearance_maps["global"][0]
    monkeypatch.setattr(module.time, "monotonic", lambda: 23.)
    current = message()
    current.data[25] = 42
    node._clearance_map_callback(current, "global")
    new, received_at = node.clearance_maps["global"]
    assert received_at == 23. and new.values.ravel()[25] == 42
    assert new.components is old.components
    node._clearance_map_callback(SimpleNamespace(), "global")
    assert "global" not in node.clearance_maps


def test_numpy_incremental_wire_and_snapshot_update():
    pytest.importorskip("rospy")
    from map_msgs.msg import OccupancyGridUpdate
    from rospy.numpy_msg import numpy_msg
    original = OccupancyGridUpdate(x=3, y=4, width=3, height=1, data=[-1, 99, 100])
    wire = BytesIO()
    original.serialize(wire)
    decoded = numpy_msg(OccupancyGridUpdate)().deserialize(wire.getvalue())
    encoded = BytesIO()
    decoded.serialize(encoded)
    assert encoded.getvalue() == wire.getvalue()
    old = ready()
    actual, expected = old.updated(decoded), old.updated(original)
    assert np.array_equal(actual.values, expected.values)
    assert np.array_equal(actual.components, expected.components)
    assert np.array_equal(actual.obstacle_distance, expected.obstacle_distance)


def test_numpy_local_map_recovery_does_not_use_array_truthiness():
    pytest.importorskip("rospy")
    from semantic_behavior_executor import SemanticBehaviorExecutor
    node = object.__new__(SemanticBehaviorExecutor)
    node.lock = threading.RLock()
    node._latest_occupancy = message()
    node.map_frame = "map"
    node._current_pose = lambda _frame: (1., 1., 0.)
    node.stuck_recovery_robot_radius_m = .25
    node.stuck_recovery_safety_margin_m = .05
    node.stuck_recovery_unknown_is_blocked = True
    node._latest_occupancy_received_at = time.monotonic()
    node.rear_goal_local_costmap_max_age_s = .75
    snapshot, detail = node._fresh_rear_local_costmap_snapshot()
    assert snapshot is node._latest_occupancy
    assert detail["costmap_age_s"] < .75
    clear, _ = node._container_map_clearance(snapshot, "map", [1., 1., 0.], .15, costmap=True)
    assert clear
    assert node._safe_recovery_distance((1., 1., 0.), 1., .2) > 0
    assert not node._escape_nearest_obstacle("test")


@pytest.mark.parametrize("data", [None, [], np.array([], dtype=np.int8)])
def test_empty_numpy_local_map_remains_unavailable(data):
    pytest.importorskip("rospy")
    from semantic_behavior_executor import SemanticBehaviorExecutor
    node = object.__new__(SemanticBehaviorExecutor)
    node.lock = threading.RLock()
    node._latest_occupancy = message()
    node._latest_occupancy.data = data
    snapshot, detail = node._fresh_rear_local_costmap_snapshot()
    assert snapshot is None and detail["reason"] == "rear_goal_local_costmap_unavailable"
