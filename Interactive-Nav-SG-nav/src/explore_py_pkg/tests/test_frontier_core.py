import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from explore_py_pkg.frontier_core import FrontierConfig, FrontierExplorerCore, GridSpec, OccupancyGridData
from explore_py_pkg.state import ExplorerState
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


def test_failed_cluster_is_temporarily_unavailable():
    grid = make_grid(10, 10, (2, 2, 8, 8))
    core = FrontierExplorerCore(FrontierConfig(min_cluster_cells=2))
    state = ExplorerState()
    first = core.select_next_cluster(grid, robot_xy=(5.0, 5.0), state=state)
    assert first is not None

    now = time.time()
    state.start_goal(first, robot_xy=(5.0, 5.0), now=now)
    state.mark_active_failed("move_base_aborted", now=now + 1.0)
    clusters = core.extract_frontier_clusters(grid, robot_xy=(5.0, 5.0), state=state)

    assert all(cluster.cluster_id != first.cluster_id for cluster in clusters)


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
