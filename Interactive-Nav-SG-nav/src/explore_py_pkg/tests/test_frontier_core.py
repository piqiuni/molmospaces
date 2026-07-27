import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from explore_py_pkg.frontier_core import FrontierCluster, FrontierConfig, FrontierExplorerCore, GridSpec, OccupancyGridData
from explore_py_pkg.state import (
    CLUSTER_ACTIVE,
    CLUSTER_FAILED,
    CLUSTER_UNREACHABLE,
    SUBGOAL_REACHED,
    ExplorerState,
    ExplorerStateConfig,
)
from explore_py_pkg.value_maps import ValueMapFusion


class FakeInfo:
    pass


class FakeOrigin:
    pass


class FakePose:
    pass


class FakeGrid:
    pass


def make_grid(width, height, free_rect):
    data = [-1] * (width * height)
    x0, y0, x1, y1 = free_rect
    for y in range(y0, y1):
        for x in range(x0, x1):
            data[y * width + x] = 0
    return OccupancyGridData(GridSpec(width, height, 1.0, 0.0, 0.0, "map"), data)


def make_value_grid(width, height, hot_cell):
    msg = FakeGrid()
    msg.info = FakeInfo()
    msg.info.width = width
    msg.info.height = height
    msg.info.resolution = 1.0
    msg.info.origin = FakeOrigin()
    msg.info.origin.position = FakePose()
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.data = [0] * (width * height)
    x, y = hot_cell
    msg.data[y * width + x] = 100
    return msg


def test_extracts_frontier_clusters_from_occ_only():
    grid = make_grid(10, 10, (2, 2, 8, 8))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2))
    clusters = core.extract_frontier_clusters(grid, robot_xy=(5.0, 5.0))

    assert clusters
    assert all(cluster.subgoal_world is not None for cluster in clusters)
    assert all(grid.cell(*cluster.subgoal_cell) == 0 for cluster in clusters)


def test_unknown_component_area_counts_connected_unknown_cells_in_metric_area():
    width = height = 7
    data = [100] * (width * height)
    data[2 * width + 2] = 0
    for x, y in ((3, 2), (4, 2), (4, 3)):
        data[y * width + x] = -1
    grid = OccupancyGridData(
        GridSpec(width, height, 0.5, 0.0, 0.0, "map"), data
    )
    core = FrontierExplorerCore(
        FrontierConfig(unknown_component_radius_m=4.0)
    )

    area_m2 = core._unknown_component_area_m2(
        grid, [(2, 2)], centroid_cell=(2.0, 2.0)
    )

    assert math.isclose(area_m2, 0.75)


def test_expected_visible_area_caps_large_component_by_frontier_aperture():
    width = height = 30
    data = [-1] * (width * height)
    for y in range(10, 20):
        for x in range(2, 10):
            data[y * width + x] = 0
    grid = OccupancyGridData(
        GridSpec(width, height, 0.1, 0.0, 0.0, "map"), data
    )
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            sensor_range_m=5.0,
            unknown_component_radius_m=5.0,
            require_footprint_free=False,
            require_turning_clearance=False,
        )
    )

    cluster = core._build_cluster(
        grid,
        [(9, 14), (9, 15), (9, 16), (9, 17)],
        robot_xy=(0.45, 1.55),
    )

    assert cluster is not None
    assert math.isclose(cluster.frontier_length_m, 0.4)
    assert cluster.unknown_component_area_m2 > 2.0
    assert math.isclose(cluster.expected_visible_unknown_area_m2, 2.0)


def test_subgoal_is_not_robot_current_cell_when_min_distance_is_set():
    grid = make_grid(12, 12, (2, 2, 10, 10))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2, min_subgoal_distance_m=1.5))
    cluster = core.select_next_cluster(grid, robot_xy=(5.5, 5.5))

    assert cluster is not None
    dx = cluster.subgoal_world[0] - 5.5
    dy = cluster.subgoal_world[1] - 5.5
    assert (dx * dx + dy * dy) ** 0.5 >= 1.0


def test_subgoal_rejects_all_candidates_inside_hard_min_distance():
    width = height = 9
    grid = OccupancyGridData(
        GridSpec(width, height, 0.1, 0.0, 0.0, "map"),
        [0] * (width * height),
    )
    core = FrontierExplorerCore(
        FrontierConfig(
            subgoal_search_radius_cells=2,
            min_subgoal_distance_m=1.0,
            hard_min_subgoal_distance_m=0.5,
            use_voronoi_viewpoints=False,
            require_footprint_free=False,
            require_turning_clearance=False,
        )
    )

    subgoal = core._choose_subgoal_cell(grid, [(4, 4)], robot_xy=(0.45, 0.45))

    assert subgoal is None


def test_hard_min_subgoal_distance_uses_world_coordinates():
    grid = OccupancyGridData(
        GridSpec(12, 3, 0.1, 0.0, 0.0, "map"),
        [0] * (12 * 3),
    )
    robot_xy = (0.099, 0.15)
    core = FrontierExplorerCore(
        FrontierConfig(
            subgoal_search_radius_cells=6,
            min_subgoal_distance_m=0.5,
            hard_min_subgoal_distance_m=0.5,
            use_voronoi_viewpoints=False,
            require_footprint_free=False,
            require_turning_clearance=False,
        )
    )

    subgoal = core._choose_subgoal_cell(grid, [(0, 1)], robot_xy=robot_xy)

    assert subgoal is not None
    subgoal_world = grid.spec.grid_to_world(*subgoal)
    assert math.dist(subgoal_world, robot_xy) >= 0.5


def test_subgoal_prefers_middle_of_long_frontier_over_endpoint():
    width, height = 16, 8
    data = [-1] * (width * height)
    frontier_cells = [(x, 4) for x in range(2, 12)]
    for x, y in frontier_cells:
        data[y * width + x] = 0
    grid = OccupancyGridData(GridSpec(width, height, 1.0, 0.0, 0.0, "map"), data)
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            min_subgoal_distance_m=1.0,
            use_voronoi_viewpoints=False,
            require_footprint_free=False,
            require_turning_clearance=False,
        )
    )

    subgoal = core._choose_subgoal_cell(grid, frontier_cells, robot_xy=(2.5, 4.5))

    assert subgoal is not None
    assert 5 <= subgoal[0] <= 8


def test_subgoal_rejects_candidate_without_free_footprint():
    width, height = 16, 8
    data = [-1] * (width * height)
    frontier_cells = [(x, 4) for x in range(2, 12)]
    for x, y in frontier_cells:
        data[y * width + x] = 0
    grid = OccupancyGridData(GridSpec(width, height, 0.1, 0.0, 0.0, "map"), data)
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            subgoal_search_radius_cells=4,
            robot_radius_m=0.25,
            footprint_safety_margin_m=0.05,
            require_footprint_free=True,
        )
    )

    subgoal = core._choose_subgoal_cell(grid, frontier_cells, robot_xy=(0.25, 0.45))

    assert subgoal is None


def test_footprint_rejects_unknown_cells_around_known_free_viewpoint():
    width = height = 9
    data = [-1] * (width * height)
    data[4 * width + 4] = 0
    grid = OccupancyGridData(GridSpec(width, height, 0.1, 0.0, 0.0, "map"), data)
    core = FrontierExplorerCore(
        FrontierConfig(
            footprint_unknown_is_free=False,
        )
    )

    assert core._footprint_is_free(grid, (4, 4), radius_cells=1) is False


def test_subgoal_rejects_candidate_without_turning_clearance():
    width, height = 7, 5
    data = [-1] * (width * height)
    frontier_cells = [(x, 2) for x in range(1, 6)]
    for x, y in frontier_cells:
        data[y * width + x] = 0
    grid = OccupancyGridData(GridSpec(width, height, 1.0, 0.0, 0.0, "map"), data)
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            subgoal_search_radius_cells=0,
            use_voronoi_viewpoints=False,
            robot_radius_m=0.0,
            footprint_safety_margin_m=0.0,
            turning_safety_margin_m=1.1,
            require_footprint_free=True,
            require_turning_clearance=True,
            footprint_unknown_is_free=False,
        )
    )

    subgoal = core._choose_subgoal_cell(grid, frontier_cells, robot_xy=(3.5, 2.5))

    assert subgoal is None


def test_subgoal_yaw_faces_frontier_centroid():
    width, height = 12, 12
    data = [-1] * (width * height)
    for y in range(1, 10):
        for x in range(3, 8):
            data[y * width + x] = 0
        data[y * width + 2] = 100
        data[y * width + 8] = 100
    frontier_cells = [(x, 9) for x in range(3, 8)]
    grid = OccupancyGridData(GridSpec(width, height, 1.0, 0.0, 0.0, "map"), data)
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            subgoal_search_radius_cells=5,
            target_frontier_offset_m=2.0,
            min_viewpoint_frontier_distance_m=2.0,
            max_viewpoint_frontier_distance_m=4.0,
            min_obstacle_clearance_m=2.0,
            max_clearance_check_m=4.0,
            require_turning_clearance=False,
        )
    )

    cluster = core._build_cluster(grid, frontier_cells, robot_xy=(5.5, 4.5))

    assert cluster is not None
    expected = math.atan2(
        cluster.centroid_world[1] - cluster.subgoal_world[1],
        cluster.centroid_world[0] - cluster.subgoal_world[0],
    )
    assert abs(cluster.subgoal_yaw - expected) < 1e-9


def test_voronoi_viewpoint_stands_back_from_frontier_in_corridor():
    width, height = 12, 12
    data = [-1] * (width * height)
    for y in range(1, 10):
        for x in range(3, 8):
            data[y * width + x] = 0
        data[y * width + 2] = 100
        data[y * width + 8] = 100
    frontier_cells = [(x, 9) for x in range(3, 8)]
    grid = OccupancyGridData(GridSpec(width, height, 1.0, 0.0, 0.0, "map"), data)
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            subgoal_search_radius_cells=5,
            target_frontier_offset_m=2.0,
            min_viewpoint_frontier_distance_m=2.0,
            max_viewpoint_frontier_distance_m=4.0,
            min_obstacle_clearance_m=2.0,
            max_clearance_check_m=4.0,
        )
    )

    subgoal = core._choose_subgoal_cell(grid, frontier_cells, robot_xy=(5.5, 4.5))

    assert subgoal is not None
    assert subgoal[0] == 5
    assert subgoal[1] <= 7
    assert min(((subgoal[0] - fx) ** 2 + (subgoal[1] - fy) ** 2) ** 0.5 for fx, fy in frontier_cells) >= 2.0


def test_far_frontier_is_penalized_but_not_hard_filtered():
    width, height = 24, 8
    data = [-1] * (width * height)
    for y in range(2, 6):
        for x in range(2, 6):
            data[y * width + x] = 0
        for x in range(15, 21):
            data[y * width + x] = 0
    grid = OccupancyGridData(GridSpec(width, height, 1.0, 0.0, 0.0, "map"), data)
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            local_horizon_m=3.0,
            far_cluster_penalty=0.5,
            far_cluster_penalty_saturation_m=4.0,
            information_weight=1.0,
            distance_weight=0.55,
        )
    )

    clusters = core.extract_frontier_clusters(grid, robot_xy=(3.5, 3.5))
    far_clusters = [cluster for cluster in clusters if cluster.distance_to_robot > 8.0]

    assert far_clusters
    assert far_clusters[0].score_terms["far_cluster_penalty"] > 0.0
    assert far_clusters[0].score > -1.0


def test_initial_local_selection_prefers_near_backward_frontier():
    core = FrontierExplorerCore(FrontierConfig(initial_local_radius_m=3.0, initial_backward_weight=0.5))
    front = FrontierCluster(
        cluster_id="front",
        cells=[(0, 0)] * 100,
        centroid_cell=(0.0, 0.0),
        centroid_world=(3.0, 0.0),
        subgoal_cell=(3, 0),
        subgoal_world=(3.0, 0.0),
        subgoal_yaw=0.0,
        information_gain=100.0,
        distance_to_robot=3.0,
        score=10.0,
    )
    back = FrontierCluster(
        cluster_id="back",
        cells=[(0, 0)] * 10,
        centroid_cell=(0.0, 0.0),
        centroid_world=(-1.5, 0.0),
        subgoal_cell=(-1, 0),
        subgoal_world=(-1.5, 0.0),
        subgoal_yaw=0.0,
        information_gain=10.0,
        distance_to_robot=1.5,
        score=1.0,
    )

    chosen = core.select_initial_local_cluster([front, back], robot_xy=(0.0, 0.0), robot_yaw=0.0)

    assert chosen is back


def test_continuity_is_soft_cost_not_hard_gate():
    core = FrontierExplorerCore(
        FrontierConfig(
            continuity_cost_weight=0.2,
            continuity_cost_saturation_m=4.0,
            receding_distance_weight=0.0,
        )
    )
    state = ExplorerState()
    near = FrontierCluster(
        cluster_id="near",
        cells=[(0, 0)],
        centroid_cell=(0.0, 0.0),
        centroid_world=(1.0, 0.0),
        subgoal_cell=(1, 0),
        subgoal_world=(1.0, 0.0),
        subgoal_yaw=0.0,
        information_gain=1.0,
        distance_to_robot=1.0,
        score=0.1,
    )
    far = FrontierCluster(
        cluster_id="far",
        cells=[(0, 0)] * 100,
        centroid_cell=(0.0, 0.0),
        centroid_world=(8.0, 0.0),
        subgoal_cell=(8, 0),
        subgoal_world=(8.0, 0.0),
        subgoal_yaw=0.0,
        information_gain=100.0,
        distance_to_robot=8.0,
        score=10.0,
    )
    state.last_subgoal_world = (0.0, 0.0)
    core._score_cluster(near, grid=None, robot_xy=(0.0, 0.0), value_provider=None, state=state)
    core._score_cluster(far, grid=None, robot_xy=(0.0, 0.0), value_provider=None, state=state)

    ranked = core.rank_clusters([far, near], robot_xy=(0.0, 0.0), state=None)

    assert ranked[0] is far
    assert far.score_terms["continuity_cost"] > near.score_terms["continuity_cost"]


def test_covered_cluster_is_not_selected_again():
    grid = make_grid(10, 10, (2, 2, 8, 8))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2))
    state = ExplorerState()
    first = core.select_next_cluster(grid, robot_xy=(5.0, 5.0), state=state)
    assert first is not None

    state.start_goal(first, robot_xy=(5.0, 5.0))
    state.mark_active_reached()
    clusters = core.extract_frontier_clusters(grid, robot_xy=(5.0, 5.0), state=state)

    assert all(cluster.cluster_id != first.cluster_id for cluster in clusters)


def test_single_failed_cluster_stays_available_with_penalty():
    grid = make_grid(10, 10, (2, 2, 8, 8))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2))
    state = ExplorerState()
    first = core.select_next_cluster(grid, robot_xy=(5.0, 5.0), state=state)
    assert first is not None

    now = time.time()
    state.start_goal(first, robot_xy=(5.0, 5.0), now=now)
    state.mark_active_failed("move_base_aborted", now=now + 1.0)
    clusters = core.extract_frontier_clusters(grid, robot_xy=(5.0, 5.0), state=state)

    assert state.records[first.cluster_id].status == CLUSTER_ACTIVE
    assert state.failure_penalty(first) > 0.0
    assert any(cluster.cluster_id == first.cluster_id for cluster in clusters)


def test_repeated_failed_cluster_is_temporarily_unavailable():
    grid = make_grid(10, 10, (2, 2, 8, 8))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2))
    state = ExplorerState(ExplorerStateConfig(failed_cluster_max_failures=2))
    first = core.select_next_cluster(grid, robot_xy=(5.0, 5.0), state=state)
    assert first is not None

    now = time.time()
    state.start_goal(first, robot_xy=(5.0, 5.0), now=now)
    state.mark_active_failed("move_base_aborted", now=now + 1.0)
    state.start_goal(first, robot_xy=(5.0, 5.0), now=now + 2.0)
    state.mark_active_failed("move_base_aborted", now=now + 3.0)
    clusters = core.extract_frontier_clusters(grid, robot_xy=(5.0, 5.0), state=state)

    assert state.records[first.cluster_id].status == CLUSTER_FAILED
    assert all(cluster.cluster_id != first.cluster_id for cluster in clusters)


def test_failed_goal_blocks_nearby_subgoals_even_if_cluster_id_changes():
    state = ExplorerState(
        ExplorerStateConfig(
            failed_point_soft_blacklist_sec=120.0,
            failed_point_blacklist_sec=120.0,
            failed_point_blacklist_radius_m=1.25,
        )
    )
    cluster = type(
        "Cluster",
        (),
        {
            "cluster_id": "old",
            "centroid_world": (0.0, 0.0),
            "subgoal_world": (6.45, 13.15),
        },
    )()

    state.start_goal(cluster, robot_xy=(7.25, 13.94), now=10.0)
    state.mark_active_failed("goal_stalled", now=40.0)

    assert state.is_goal_point_blocked((6.45, 13.15), now=41.0)
    assert state.is_goal_point_blocked((6.25, 13.15), now=41.0)
    assert not state.is_goal_point_blocked((4.50, 13.15), now=41.0)


def test_frontier_gone_requires_consecutive_confirmations_and_min_age():
    state = ExplorerState(
        ExplorerStateConfig(
            frontier_gone_confirm_ticks=3,
            frontier_gone_min_goal_age_sec=5.0,
            reached_point_blacklist_sec=1.0,
            visit_viewpoint_once=True,
        )
    )
    cluster = type(
        "Cluster",
        (),
        {
            "cluster_id": "frontier",
            "centroid_world": (0.0, 0.0),
            "subgoal_world": (1.0, 1.0),
        },
    )()

    state.start_goal(cluster, robot_xy=(0.0, 0.0), now=10.0)

    assert not state.mark_active_covered_if_frontier_gone(False, now=11.0)
    assert state.active_goal is not None
    assert state.active_goal.frontier_gone_count == 1

    assert not state.mark_active_covered_if_frontier_gone(True, now=12.0)
    assert state.active_goal is not None
    assert state.active_goal.frontier_gone_count == 0

    assert not state.mark_active_covered_if_frontier_gone(False, now=13.0)
    assert not state.mark_active_covered_if_frontier_gone(False, now=14.0)
    assert not state.mark_active_covered_if_frontier_gone(False, now=14.5)
    assert state.active_goal is not None

    assert state.mark_active_covered_if_frontier_gone(False, now=15.1)
    assert state.active_goal is None
    assert state.visited_viewpoints == []
    assert not state.is_goal_point_blocked((1.0, 1.0), now=17.0)


def test_reaching_goal_is_not_delayed_by_minimum_lifetime():
    state = ExplorerState(
        ExplorerStateConfig(
            goal_reach_tolerance_m=0.35,
            min_goal_lifetime_sec=8.0,
        )
    )
    cluster = type(
        "Cluster",
        (),
        {
            "cluster_id": "near-goal",
            "centroid_world": (1.0, 0.0),
            "subgoal_world": (0.3, 0.0),
            "subgoal_yaw": 0.0,
        },
    )()
    state.start_goal(cluster, robot_xy=(0.0, 0.0), now=10.0)

    progress = state.update_goal_progress((0.0, 0.0), now=10.1)

    assert progress == SUBGOAL_REACHED


def test_reached_viewpoint_is_permanently_blocked_by_position():
    state = ExplorerState(
        ExplorerStateConfig(
            visit_viewpoint_once=True,
            visited_viewpoint_radius_m=0.5,
            reached_point_blacklist_sec=1.0,
        )
    )
    cluster = type(
        "Cluster",
        (),
        {
            "cluster_id": "visited",
            "centroid_world": (3.0, 1.0),
            "subgoal_world": (1.0, 1.0),
        },
    )()

    state.start_goal(cluster, robot_xy=(0.0, 1.0), now=10.0)
    state.mark_active_reached(now=20.0)

    assert state.is_goal_point_blocked((1.0, 1.0), now=100000.0)
    assert state.is_goal_point_blocked((1.4, 1.0), now=100000.0)
    assert state.is_goal_point_blocked((1.50000001, 1.0), now=100000.0)
    assert not state.is_goal_point_blocked((1.6, 1.0), now=100000.0)


def test_failed_viewpoint_is_not_marked_visited():
    state = ExplorerState(ExplorerStateConfig(visit_viewpoint_once=True))
    cluster = type(
        "Cluster",
        (),
        {
            "cluster_id": "failed",
            "centroid_world": (3.0, 1.0),
            "subgoal_world": (1.0, 1.0),
        },
    )()

    state.start_goal(cluster, robot_xy=(0.0, 1.0), now=10.0)
    state.mark_active_failed("move_base_aborted", now=20.0)

    assert state.visited_viewpoints == []


def test_reached_viewpoint_with_remaining_frontier_marks_it_unreachable():
    grid = make_grid(10, 10, (2, 2, 8, 8))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2))
    state = ExplorerState()
    cluster = core.select_next_cluster(grid, robot_xy=(5.0, 5.0), state=state)

    assert cluster is not None
    state.start_goal(cluster, robot_xy=(5.0, 5.0))
    state.mark_active_frontier_unreachable()

    record = state.records[cluster.cluster_id]
    assert record.status == CLUSTER_UNREACHABLE
    assert state.is_goal_point_blocked(cluster.subgoal_world)
    assert state.is_frontier_unreachable(cluster.centroid_world)
    assert not state.is_cluster_available(cluster)


def test_unreachable_frontier_blocks_nearby_reclustered_candidate():
    state = ExplorerState(ExplorerStateConfig(unreachable_frontier_radius_m=1.0))
    reached = type(
        "Cluster",
        (),
        {
            "cluster_id": "old-id",
            "centroid_world": (8.0, 3.0),
            "subgoal_world": (5.0, 3.0),
        },
    )()
    reclustered = type(
        "Cluster",
        (),
        {
            "cluster_id": "new-id",
            "centroid_world": (8.6, 3.2),
            "subgoal_world": (6.0, 3.0),
        },
    )()
    separate = type(
        "Cluster",
        (),
        {
            "cluster_id": "separate",
            "centroid_world": (10.0, 3.0),
            "subgoal_world": (8.0, 3.0),
        },
    )()

    state.start_goal(reached, robot_xy=(5.0, 3.0), now=10.0)
    state.mark_active_frontier_unreachable(now=20.0)

    assert not state.is_cluster_available(reclustered, now=21.0)
    assert state.is_cluster_available(separate, now=21.0)


def test_active_goal_keeps_frontier_reference_separate_from_viewpoint():
    state = ExplorerState()
    cluster = type(
        "Cluster",
        (),
        {
            "cluster_id": "frontier",
            "centroid_world": (8.0, 3.0),
            "subgoal_world": (5.0, 3.0),
            "subgoal_yaw": 0.0,
        },
    )()

    goal = state.start_goal(cluster, robot_xy=(4.0, 3.0), now=10.0)

    assert goal.point == (5.0, 3.0)
    assert goal.frontier_point == (8.0, 3.0)


def test_llm_value_grid_changes_candidate_ranking_without_generating_goal():
    grid = make_grid(14, 8, (3, 2, 11, 6))
    core = FrontierExplorerCore(
        FrontierConfig(
            min_cluster_cells=1,
            information_weight=0.0,
            distance_weight=0.0,
            llm_weight=1.0,
            semantic_weight=0.0,
        )
    )
    fusion = ValueMapFusion()
    fusion.set_llm_value_grid(make_value_grid(14, 8, hot_cell=(10, 4)))

    chosen = core.select_next_cluster(grid, robot_xy=(4.0, 4.0), value_provider=fusion)

    assert chosen is not None
    assert grid.cell(*chosen.subgoal_cell) == 0
    assert chosen.score_terms["llm"] > 0.0
