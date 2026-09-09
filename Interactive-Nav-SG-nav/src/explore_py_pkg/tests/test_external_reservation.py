import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from explore_py_pkg.external_reservation import resolve_external_frontier_reservation
from explore_py_pkg.frontier_core import FrontierCluster, GridSpec


def make_cluster(cluster_id: str = "current") -> FrontierCluster:
    return FrontierCluster(
        cluster_id=cluster_id,
        cells=[(2, 3)],
        centroid_cell=(2.0, 3.0),
        centroid_world=(2.5, 3.5),
        subgoal_cell=(2, 2),
        subgoal_world=(2.5, 2.5),
        subgoal_yaw=0.5,
        information_gain=1.0,
        distance_to_robot=2.0,
    )


def test_missing_frontier_reservation_retains_dispatched_goal() -> None:
    command = {
        "cluster_id": "gone_after_scan",
        "candidate_id": "frontier:gone_after_scan",
        "goal_xyyaw": [4.2, 1.6, -0.7],
        "frontier_point": [4.5, 1.2],
    }
    grid_spec = GridSpec(100, 100, 0.1, 0.0, 0.0, "map")

    cluster, retained = resolve_external_frontier_reservation(
        command,
        [make_cluster()],
        grid_spec=grid_spec,
        robot_xy=(1.2, 1.6),
    )

    assert retained
    assert cluster is not None
    assert cluster.cluster_id == "gone_after_scan"
    assert cluster.subgoal_world == (4.2, 1.6)
    assert cluster.centroid_world == (4.5, 1.2)
    assert cluster.subgoal_cell == (42, 16)
    assert math.isclose(cluster.distance_to_robot, 3.0)


def test_current_frontier_reservation_still_uses_live_cluster() -> None:
    current = make_cluster("still_present")

    cluster, retained = resolve_external_frontier_reservation(
        {"cluster_id": "still_present", "goal_xyyaw": [9.0, 9.0, 0.0]},
        [current],
        grid_spec=None,
        robot_xy=None,
    )

    assert not retained
    assert cluster is current
