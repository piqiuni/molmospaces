from explore_py_pkg.frontier_core import FrontierCluster
from explore_py_pkg.proposals import build_proposal_snapshot, cluster_to_proposal


def make_cluster() -> FrontierCluster:
    return FrontierCluster(
        cluster_id="cluster_1",
        cells=[(1, 2), (1, 3), (2, 3)],
        centroid_cell=(1.3, 2.6),
        centroid_world=(2.0, 3.0),
        subgoal_cell=(0, 2),
        subgoal_world=(1.0, 3.0),
        subgoal_yaw=0.5,
        information_gain=18.0,
        distance_to_robot=2.5,
        unknown_component_area_m2=12.5,
        score=1.2,
        score_terms={
            "information": 0.8,
            "distance": 0.3,
            "semantic": 0.9,
            "llm": 1.0,
        },
    )


def test_proposal_contains_geometry_but_not_semantic_scores() -> None:
    proposal = cluster_to_proposal(make_cluster(), "map")
    assert proposal["goal_xyyaw"] == [1.0, 3.0, 0.5]
    assert proposal["raw_features"]["frontier_cell_count"] == 3
    assert proposal["raw_features"]["unknown_component_area_m2"] == 12.5
    assert proposal["geometry"]["proposal_score_terms"] == {
        "information": 0.8,
        "distance": 0.3,
    }


def test_frontier_exhausted_requires_ready_scan_and_no_active_proposal() -> None:
    exhausted = build_proposal_snapshot(
        [],
        ready=True,
        frame_id="map",
        robot_xy=(0.0, 0.0),
        robot_yaw=0.25,
        initial_scan_complete=True,
        timestamp=1.0,
    )
    assert exhausted["frontier_exhausted"] is True
    assert exhausted["robot_yaw"] == 0.25

    active = build_proposal_snapshot(
        [],
        ready=True,
        frame_id="map",
        robot_xy=(0.0, 0.0),
        active_cluster_id="cluster_1",
        initial_scan_complete=True,
        timestamp=1.0,
    )
    assert active["frontier_exhausted"] is False


def test_initial_scan_hides_frontier_proposals_until_complete() -> None:
    snapshot = build_proposal_snapshot(
        [make_cluster()],
        ready=True,
        frame_id="map",
        robot_xy=(0.0, 0.0),
        initial_scan_complete=False,
        timestamp=1.0,
    )

    assert snapshot["initial_scan_complete"] is False
    assert snapshot["proposal_count"] == 0
    assert snapshot["proposals"] == []
    assert snapshot["frontier_exhausted"] is False
