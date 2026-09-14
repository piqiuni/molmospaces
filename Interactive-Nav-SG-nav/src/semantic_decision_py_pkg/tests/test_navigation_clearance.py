import math
import time
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_decision_py_pkg.navigation_clearance import ArrivalClearanceGrid


def make_grid(*, costmap=False):
    return ArrivalClearanceGrid(np.zeros((100, 100), dtype=np.int16), 0.1,
                                -5., -5., 0., "map", 100 if costmap else 50,
                                99 if costmap else None)


@pytest.mark.parametrize("obstacle", [-1, 50, 100])
@pytest.mark.parametrize("tolerance", [0., 0.05, 0.15, 0.25])
def test_clearance_excludes_arrival_tolerance_but_keeps_footprint_margin(obstacle, tolerance):
    grid = make_grid()
    grid.values[50, 54] = obstacle
    detail = grid.check((0., 0.), tolerance)
    assert detail["clear"]
    assert detail["clearance_radius_m"] == pytest.approx(.25 + .05 + .1/math.sqrt(2))
    assert detail["arrival_region_radius_m"] == 0.
    grid.values[50, 53] = obstacle
    assert grid.check((0., 0.), tolerance)["reason"] == "arrival_region_blocked"


def test_local_inscribed_cost_is_not_inflated_twice():
    grid = make_grid(costmap=True)
    grid.values[50, 51] = 99
    assert grid.check((0., 0.), 0.15)["clear"]
    grid.values[50, 50] = 99
    assert grid.check((0., 0.), 0.15)["reason"] == "center_blocked"


def test_cell_half_diagonal_and_rotated_origin():
    base = make_grid()
    base.values[50, 53] = 100
    rotated = ArrivalClearanceGrid(base.values, 0.1, 5., -5., math.pi/2, "map")
    assert not base.check((0., 0.), 0.)["clear"]
    assert not rotated.check((0., 0.), 0.)["clear"]
    assert base.check((4.8, 0.), 0.)["reason"] == "outside_map_window"


def test_candidate_clearance_uses_global_only_outside_local_window():
    pytest.importorskip("rospy")
    from semantic_candidate_node import SemanticCandidateNode
    node = object.__new__(SemanticCandidateNode)
    node.navigation_clearance_enabled = True
    node.clearance_lock = threading.RLock()
    node.map_frame = "map"
    node.clearance_robot_radius_m, node.clearance_safety_margin_m = 0.25, 0.05
    node.clearance_map_max_age_s = 0.75
    local = make_grid(costmap=True)
    planning = ArrivalClearanceGrid(np.zeros((300, 300)), .1, -15., -15., 0., "map")
    node.clearance_maps = {"local": (local, time.monotonic()), "planning": (planning, time.monotonic())}
    assert node._make_clearance_check()((10., 0., 0.), .15)["costmap_source"] == "planning"
    local.values[50, 50] = 100
    assert not node._make_clearance_check()((0., 0., 0.), .15)["clear"]
    node.clearance_maps["planning"] = (planning, time.monotonic()-2)
    assert not node._make_clearance_check()((10., 0., 0.), .15)["clear"]


@pytest.mark.parametrize("resolution", [0., -1., math.nan, math.inf])
def test_invalid_message_geometry(resolution):
    message = SimpleNamespace(info=SimpleNamespace(width=2, height=2, resolution=resolution))
    with pytest.raises(ValueError):
        ArrivalClearanceGrid.from_message(message)


@pytest.mark.parametrize("barrier", [-1, 99, 100])
def test_reachable_uses_inflated_four_connected_free_space(barrier):
    grid = make_grid(costmap=True)
    grid.values[:, 50] = barrier
    assert grid.check((1., 0.), .25)["clear"]
    assert grid.reachable((-1., 0.), (1., 0.))["reason"] == "path_disconnected"
    assert grid.reachable((-1., 0.), (-2., 0.))["clear"]
    assert grid.reachable((-1., 0.), (10., 0.))["reason"] == "path_outside_map"


def test_diagonal_contact_is_not_a_navigable_corridor():
    grid = ArrivalClearanceGrid(np.array([[0, 99], [99, 0]]), 1., 0., 0., 0., "map", 100, 99)
    assert not grid.reachable((.5, .5), (1.5, 1.5))["clear"]


def test_global_incremental_update_invalidates_connectivity_without_mutating_snapshot():
    grid = make_grid(costmap=True)
    assert grid.reachable((-1., 0.), (1., 0.))["clear"]
    update = SimpleNamespace(x=50, y=0, width=1, height=100, data=[99]*100)
    changed = grid.updated(update)
    assert not changed.reachable((-1., 0.), (1., 0.))["clear"]
    assert grid.reachable((-1., 0.), (1., 0.))["clear"]
    assert not changed.values.flags.writeable
    with pytest.raises(ValueError):
        grid.updated(SimpleNamespace(x=100, y=0, width=1, height=100, data=[99]*100))


def test_wall_adjacent_frontier_moves_without_reducing_arrival_tolerance():
    grid = make_grid()
    grid.values[:, 53] = 100
    goal, targets = [0., 0., 0.], [[0., 1.]]
    assert not grid.check(goal, .25)["clear"]
    alternatives = grid.recovery_goals(goal, .25, targets)
    assert alternatives
    for point in alternatives:
        assert grid.check(point, .25)["clear"]
        assert math.dist(point[:2], goal[:2]) <= .8
        assert grid.visible(point, targets[0])
    # Keep the search on this side of the wall to exercise visibility rejection.
    assert grid.recovery_goals(goal, .25, [[1., 1.]], radius_m=.6) == []


def test_recovery_goal_clearance_does_not_depend_on_arrival_tolerance():
    grid = make_grid()
    grid.values[:, 53] = 100
    goal, targets = [0., 0., 0.], [[0., 1.]]
    expected = grid.recovery_goals(goal, 0., targets)
    assert expected
    for tolerance in (.05, .15, .25):
        assert grid.recovery_goals(goal, tolerance, targets) == expected


def test_candidate_global_connectivity_is_checked_after_local_clearance():
    pytest.importorskip("rospy")
    from semantic_candidate_node import SemanticCandidateNode
    node = object.__new__(SemanticCandidateNode)
    node.navigation_clearance_enabled = True
    node.clearance_lock = threading.RLock()
    node.map_frame = node.robot_frame = "map"
    node.robot_xy = (-1., 0.)
    node.clearance_robot_radius_m, node.clearance_safety_margin_m = .25, .05
    node.clearance_map_max_age_s = .75
    global_grid = make_grid(costmap=True)
    global_grid.values[:, 50] = 99
    node.clearance_maps = {"local": (make_grid(costmap=True), time.monotonic()),
                           "global": (global_grid, time.monotonic())}
    assert node._make_clearance_check()((1., 0., 0.), .25)["reason"] == "path_disconnected"
    node.clearance_maps["global"] = (global_grid, time.monotonic()-2.)
    assert node._make_clearance_check()((1., 0., 0.), .25)["clear"]
