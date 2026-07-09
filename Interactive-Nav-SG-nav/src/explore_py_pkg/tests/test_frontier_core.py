import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from explore_py_pkg.frontier_core import FrontierConfig, FrontierExplorerCore, GridSpec, OccupancyGridData
from explore_py_pkg.state import CLUSTER_ACTIVE, CLUSTER_FAILED, ExplorerState, ExplorerStateConfig
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


def test_subgoal_is_not_robot_current_cell_when_min_distance_is_set():
    grid = make_grid(12, 12, (2, 2, 10, 10))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2, min_subgoal_distance_m=1.5))
    cluster = core.select_next_cluster(grid, robot_xy=(5.5, 5.5))

    assert cluster is not None
    dx = cluster.subgoal_world[0] - 5.5
    dy = cluster.subgoal_world[1] - 5.5
    assert (dx * dx + dy * dy) ** 0.5 >= 1.0


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
    state = ExplorerState(ExplorerStateConfig(frontier_gone_confirm_ticks=3, frontier_gone_min_goal_age_sec=5.0))
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


def test_reached_pose_only_blocks_point_but_keeps_cluster_active():
    grid = make_grid(10, 10, (2, 2, 8, 8))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2))
    state = ExplorerState()
    cluster = core.select_next_cluster(grid, robot_xy=(5.0, 5.0), state=state)

    assert cluster is not None
    state.start_goal(cluster, robot_xy=(5.0, 5.0))
    state.mark_active_reached_pose_only()

    record = state.records[cluster.cluster_id]
    assert record.status == CLUSTER_ACTIVE
    assert state.is_goal_point_blocked(cluster.subgoal_world)


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
