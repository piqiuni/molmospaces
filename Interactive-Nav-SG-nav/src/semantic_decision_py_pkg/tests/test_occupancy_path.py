from types import SimpleNamespace
import math

from semantic_decision_py_pkg.occupancy_path import (
    OccupancyTraversal,
    STATUS_FREE,
    STATUS_OCCUPIED,
    STATUS_OUT_OF_BOUNDS,
    STATUS_UNKNOWN,
    footprint_occupancy_status,
    grid_path_status,
    point_occupancy_status,
    segment_occupancy_status,
)
import semantic_decision_py_pkg.occupancy_path as occupancy_path
import pytest


def _grid(width, height, data, *, resolution=0.1, origin=(0.0, 0.0)):
    position = SimpleNamespace(x=float(origin[0]), y=float(origin[1]), z=0.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    pose = SimpleNamespace(position=position, orientation=orientation)
    info = SimpleNamespace(
        width=int(width), height=int(height), resolution=float(resolution), origin=pose
    )
    return SimpleNamespace(info=info, data=list(data))


@pytest.mark.parametrize("yaw", [0.0, 1.2])
def test_footprint_tests_obstacle_cell_area_at_actual_subcell_goal(yaw):
    data = [0] * 100
    data[5 * 10 + 4] = 100
    grid = _grid(10, 10, data, origin=(5.0, -3.0))
    grid.info.origin.orientation.z = math.sin(yaw / 2)
    grid.info.origin.orientation.w = math.cos(yaw / 2)

    def world(x, y):
        return [5 + math.cos(yaw) * x - math.sin(yaw) * y,
                -3 + math.sin(yaw) * x + math.cos(yaw) * y]
    # Goal lies near the upper-right corner of (2,2). Its radius reaches
    # the lower-left corner of obstacle (4,5), but not that cell's centre.
    assert footprint_occupancy_status(grid, *world(0.29, 0.29), radius_m=0.25) == STATUS_OCCUPIED
    result = grid_path_status(grid, world(0.35, 0.25), world(0.29, 0.29), robot_radius_m=0.25)
    assert not result["reachable"]
    assert result["status"] == STATUS_OCCUPIED


def test_shared_footprint_does_not_round_radius_up_to_a_whole_cell():
    data = [0] * 121
    data[5 * 11 + 7] = 100
    grid = _grid(11, 11, data)
    # From centre (5,5), the occupied square starts 0.15 m away.
    traversal = OccupancyTraversal(grid, robot_radius_m=0.12)
    assert traversal.cell_state((5, 5)) == STATUS_FREE
    assert footprint_occupancy_status(grid, 0.55, 0.55, radius_m=0.12) == STATUS_FREE


@pytest.mark.parametrize("radius", [0.0, 0.05, 0.12, 0.25])
def test_shared_and_endpoint_footprints_agree_at_cell_centres(radius):
    data = [0] * 121
    data[5 * 11 + 7] = 100
    grid = _grid(11, 11, data)
    traversal = OccupancyTraversal(grid, robot_radius_m=radius)
    for y in range(11):
        for x in range(11):
            assert traversal.cell_state((x, y)) == footprint_occupancy_status(
                grid, (x + 0.5) * 0.1, (y + 0.5) * 0.1, radius_m=radius)


def test_endpoint_preflight_and_search_share_exact_point_check(monkeypatch):
    grid = _grid(20, 20, [0] * 400)
    traversal = OccupancyTraversal(grid, robot_radius_m=0.12)
    original = occupancy_path._footprint_status_context
    calls = []

    def counted(*args):
        calls.append(args[1:3])
        return original(*args)

    monkeypatch.setattr(occupancy_path, "_footprint_status_context", counted)
    assert traversal.point_state(1.29, 1.29) == STATUS_FREE
    result = grid_path_status(grid, [0.35, 0.35], [1.29, 1.29], robot_radius_m=0.12, traversal=traversal)
    assert result["reachable"]
    assert calls == [(1.29, 1.29)]
    assert traversal.point_cache_hits == 1
    # A new map batch must not reuse the old receipt's free result.
    grid.data[12 * 20 + 12] = 100
    assert OccupancyTraversal(grid, robot_radius_m=0.12).point_state(1.29, 1.29) == STATUS_OCCUPIED


def test_free_segment_and_endpoint_are_safe():
    grid = _grid(20, 4, [0] * 80)

    assert point_occupancy_status(grid, 0.25, 0.15) == STATUS_FREE
    assert footprint_occupancy_status(grid, 0.25, 0.15, radius_m=0.0) == STATUS_FREE
    result = segment_occupancy_status(
        grid, (0.15, 0.15), (1.45, 0.15), sample_step_m=0.1
    )

    assert result["status"] == STATUS_FREE
    assert result["occupied"] is False
    assert result["out_of_bounds"] is False
    assert result["sample_count"] > 1


def test_search_skips_footprints_on_non_improving_return_edges(monkeypatch):
    grid = _grid(20, 1, [0] * 20)
    traversal = OccupancyTraversal(grid)
    original = traversal.cell_state
    ordinary_checks = []

    def counted(cell, *, start=False):
        if not start:
            ordinary_checks.append(cell)
        return original(cell, start=start)

    monkeypatch.setattr(traversal, "cell_state", counted)
    result = grid_path_status(grid, [0.15, 0.05], [1.55, 0.05],
                              traversal=traversal, allow_diagonal=False)
    assert result["status"] == STATUS_FREE
    assert result["path_cost_m"] == pytest.approx(1.4)
    assert result["path_cell_count"] == 15
    # No ordinary classification on a return edge to the settled start.
    assert (1, 0) not in ordinary_checks


def test_unknown_segment_is_retained_not_rejected():
    data = [0] * 80
    # A complete unknown row is enough to exercise both endpoint and path
    # handling without requiring ROS message classes.
    for index in range(20 + 4, 20 + 8):
        data[index] = -1
    grid = _grid(20, 4, data)

    assert point_occupancy_status(grid, 0.45, 0.15) == STATUS_UNKNOWN
    result = segment_occupancy_status(
        grid, (0.15, 0.15), (0.75, 0.15), sample_step_m=0.1
    )

    assert result["status"] == STATUS_UNKNOWN
    assert result["unknown"] is True
    assert result["occupied"] is False
    assert result["out_of_bounds"] is False


def test_known_occupied_segment_fails_closed():
    data = [0] * 80
    data[2 * 20 + 8] = 100
    grid = _grid(20, 4, data)

    result = segment_occupancy_status(
        grid, (0.15, 0.25), (1.45, 0.25), sample_step_m=0.1
    )

    assert result["status"] == STATUS_OCCUPIED
    assert result["occupied"] is True
    assert result["out_of_bounds"] is False
    assert result["first_occupied_distance_m"] > 0.0


def test_out_of_bounds_endpoint_fails_closed():
    grid = _grid(10, 10, [0] * 100)

    assert point_occupancy_status(grid, 1.05, 0.15) == STATUS_OUT_OF_BOUNDS
    result = segment_occupancy_status(
        grid, (0.15, 0.15), (1.05, 0.15), sample_step_m=0.1
    )

    assert result["status"] == STATUS_OUT_OF_BOUNDS
    assert result["out_of_bounds"] is True


def test_hot_path_decodes_grid_geometry_once(monkeypatch):
    """Segment/A* queries must not reparse ROS geometry for every cell."""

    grid = _grid(40, 40, [0] * (40 * 40))
    original = occupancy_path._grid_geometry
    calls = {"count": 0}

    def counted(value):
        calls["count"] += 1
        return original(value)

    monkeypatch.setattr(occupancy_path, "_grid_geometry", counted)
    segment_occupancy_status(
        grid, (0.15, 0.15), (3.25, 3.25), sample_step_m=0.05
    )
    grid_path_status(
        grid, (0.15, 0.15), (3.25, 3.25), max_expansions=5000
    )
    assert calls["count"] == 2


def test_footprint_out_of_bounds_dominates_unknown():
    data = [-1] * 9
    data[4] = 0
    grid = _grid(3, 3, data, resolution=0.5)

    # The center is known free but the configured footprint reaches outside
    # the map, so the hard status is out-of-bounds rather than unknown/free.
    assert (
        footprint_occupancy_status(grid, 0.75, 0.75, radius_m=1.1)
        == STATUS_OUT_OF_BOUNDS
    )


def test_grid_path_routes_around_a_wall_instead_of_rejecting_straight_segment():
    width, height = 10, 7
    data = [0] * (width * height)
    # A one-cell vertical wall blocks the direct ray but leaves a one-cell
    # corridor above and below it.
    for row in range(height):
        if row != height - 1:
            data[row * width + 4] = 100
    grid = _grid(width, height, data, resolution=1.0)

    direct = segment_occupancy_status(
        grid, (0.5, 3.5), (8.5, 3.5), sample_step_m=0.5
    )
    assert direct["status"] == STATUS_OCCUPIED
    result = grid_path_status(
        grid,
        (0.5, 3.5),
        (8.5, 3.5),
        max_expansions=500,
    )
    assert result["reachable"] is True
    assert result["status"] == STATUS_FREE
    assert result["path_cell_count"] > 0


def test_grid_path_reports_no_path_for_a_closed_wall():
    width, height = 8, 5
    data = [0] * (width * height)
    for row in range(height):
        data[row * width + 3] = 100
    grid = _grid(width, height, data, resolution=1.0)
    result = grid_path_status(
        grid,
        (0.5, 2.5),
        (6.5, 2.5),
        max_expansions=500,
    )
    assert result["reachable"] is False
    assert result["status"] == "no_path"


def test_grid_path_respects_rotated_map_origin():
    width, height = 5, 5
    grid = _grid(width, height, [0] * (width * height), resolution=1.0)
    angle = 0.5 * 3.141592653589793
    grid.info.origin.orientation.z = __import__("math").sin(angle / 2.0)
    grid.info.origin.orientation.w = __import__("math").cos(angle / 2.0)
    # Local (0.5, 0.5) -> world (-0.5, 0.5) under a +90 degree origin;
    # local (2.5, 0.5) -> world (-0.5, 2.5).
    result = grid_path_status(grid, (-0.5, 0.5), (-0.5, 2.5))
    assert result["reachable"] is True


def test_batch_collision_cache_preserves_results_and_rejects_stale_grid():
    import pytest

    grid = _grid(30, 20, [0] * 600, resolution=0.1)
    batch = OccupancyTraversal(grid, robot_radius_m=0.2)
    for goal in ((2.45, 1.05), (2.45, 1.25), (2.25, 1.05)):
        kwargs = dict(robot_radius_m=0.2, max_expansions=1000)
        reference = grid_path_status(grid, (0.55, 1.05), goal, **kwargs)
        shared = grid_path_status(grid, (0.55, 1.05), goal, traversal=batch, **kwargs)
        assert shared["status"] == reference["status"]
        assert shared["reachable"] == reference["reachable"]
        assert shared["path_cost_m"] == pytest.approx(reference["path_cost_m"])
    assert batch.cache_hits > len(batch.states)
    changed = _grid(30, 20, [100] * 600, resolution=0.1)
    with pytest.raises(ValueError):
        grid_path_status(changed, (0.55, 1.05), (2.45, 1.05),
                         robot_radius_m=0.2, traversal=batch)
    assert grid_path_status(changed, (0.55, 1.05), (2.45, 1.05),
                            robot_radius_m=0.2)["reachable"] is False


def test_known_only_route_cannot_cut_unknown_diagonal_corners():
    grid = _grid(2, 2, [0, -1, -1, 0], resolution=1.0)
    result = grid_path_status(grid, (0.5, 0.5), (1.5, 1.5), unknown_is_blocked=True)
    assert result["reachable"] is False
    permissive = grid_path_status(grid, (0.5, 0.5), (1.5, 1.5))
    assert permissive["reachable"] is True
    assert permissive["unknown"] is True


def test_free_route_is_not_marked_unknown_due_to_unselected_search_neighbours():
    data = [0] * 25
    data[1 * 5 + 2] = -1
    grid = _grid(5, 5, data, resolution=1.0)
    result = grid_path_status(grid, (1.5, 2.5), (3.5, 2.5))
    assert result["reachable"] is True
    assert result["unknown"] is False


def test_shared_search_resumes_past_first_goal_and_reuses_settled_targets():
    grid = _grid(20, 1, [0] * 20, resolution=1.0)
    batch = OccupancyTraversal(grid)
    first = grid_path_status(grid, (0.5, 0.5), (3.5, 0.5), traversal=batch)
    second = grid_path_status(grid, (0.5, 0.5), (15.5, 0.5), traversal=batch)
    nearer = grid_path_status(grid, (0.5, 0.5), (7.5, 0.5), traversal=batch)
    assert first["reachable"] and second["reachable"] and nearer["reachable"]
    assert first["path_cost_m"] == 3.0
    assert second["path_cost_m"] == 15.0
    assert nearer["path_cost_m"] == 7.0
    assert nearer["expanded_count"] == second["expanded_count"]


def test_batch_expansion_budget_is_shared_not_multiplied_by_anchor_count():
    grid = _grid(20, 20, [0] * 400, resolution=1.0)
    batch = OccupancyTraversal(grid)
    for goal in ((15.5, 15.5), (19.5, 19.5), (18.5, 10.5)):
        result = grid_path_status(grid, (0.5, 0.5), goal,
                                  traversal=batch, max_expansions=10)
        assert result["status"] == "search_limit"
        assert result["expanded_count"] == 10
    near = grid_path_status(grid, (0.5, 0.5), (1.5, 0.5),
                            traversal=batch, max_expansions=10)
    assert near["reachable"] is True


def test_shared_multi_goal_costs_match_independent_astar_on_varied_maps():
    import random
    import pytest

    rng = random.Random(24)
    for radius in (0.0, 0.1):
        for _ in range(5):
            data = [100 if rng.random() < 0.12 else 0 for _ in range(20 * 16)]
            grid = _grid(20, 16, data, resolution=0.1)
            batch = OccupancyTraversal(grid, robot_radius_m=radius)
            for _ in range(8):
                goal = (rng.randrange(1, 19) * 0.1 + 0.05,
                        rng.randrange(1, 15) * 0.1 + 0.05)
                standalone = grid_path_status(grid, (0.55, 0.75), goal,
                                              robot_radius_m=radius, max_expansions=1000)
                shared = grid_path_status(grid, (0.55, 0.75), goal,
                                          robot_radius_m=radius, max_expansions=1000,
                                          traversal=batch)
                assert shared["reachable"] == standalone["reachable"]
                if shared["reachable"]:
                    assert shared["path_cost_m"] == pytest.approx(standalone["path_cost_m"])
