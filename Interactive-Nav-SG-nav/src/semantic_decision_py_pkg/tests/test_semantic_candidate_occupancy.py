from __future__ import annotations

import threading
import sys
import time
import types
from types import SimpleNamespace

from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate


def _install_ros_import_stubs_if_needed() -> None:
    """Keep this pure candidate test runnable outside a ROS installation."""

    try:
        import rospy  # noqa: F401
        from nav_msgs.msg import OccupancyGrid  # noqa: F401
        from std_msgs.msg import String  # noqa: F401
        return
    except ImportError:
        pass
    rospy = types.ModuleType("rospy")
    rospy.init_node = lambda *_args, **_kwargs: None
    sys.modules["rospy"] = rospy
    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.OccupancyGrid = type("OccupancyGrid", (), {})
    nav_msgs_msg.Odometry = type("Odometry", (), {})
    nav_msgs.msg = nav_msgs_msg
    sys.modules["nav_msgs"] = nav_msgs
    sys.modules["nav_msgs.msg"] = nav_msgs_msg
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})
    std_msgs.msg = std_msgs_msg
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg


_install_ros_import_stubs_if_needed()
from semantic_candidate_node import SemanticCandidateNode


def _grid(width: int, height: int, data: list[int]):
    position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    origin = SimpleNamespace(position=position, orientation=orientation)
    info = SimpleNamespace(
        width=width,
        height=height,
        resolution=1.0,
        origin=origin,
    )
    stamp = SimpleNamespace(to_sec=lambda: 10.0)
    return SimpleNamespace(info=info, data=data, header=SimpleNamespace(stamp=stamp))


def _node(grid, *, source_stamp_delta: float = 0.0, max_pair_age_s: float = 1.0):
    node = object.__new__(SemanticCandidateNode)
    node._occupancy_state_lock = threading.RLock()
    node.robot_xy = (0.5, 0.5)
    node.occupancy_grid = grid
    now = time.monotonic()
    node.robot_xy_stamp_s = 10.0
    node.occupancy_grid_stamp_s = 10.0 + source_stamp_delta
    node.robot_xy_received_at = now - 0.01
    node.occupancy_grid_received_at = now - 0.01
    node.candidate_occupancy_max_pair_age_s = max_pair_age_s
    node.candidate_occupancy_preflight_enabled = True
    node.candidate_occupancy_reject_occupied = True
    node.candidate_occupancy_sample_step_m = 0.1
    node.candidate_occupancy_start_ignore_m = 0.0
    node.candidate_occupancy_endpoint_radius_m = 0.0
    node.candidate_occupancy_threshold = 50
    node.candidate_occupancy_use_astar = False
    node.filtered_occupancy_candidate_ids = []
    node.occupancy_preflight_timing = {}
    return node


def _candidate() -> BehaviorCandidate:
    return BehaviorCandidate(
        candidate_id="frontier:occupied",
        behavior_type="EXPLORE",
        source="test",
        target_id="frontier:occupied",
        target_name="occupied",
        goal_xyyaw=[1.5, 0.5, 0.0],
    )


def test_truncated_grid_is_treated_as_unknown_by_known_free_check():
    node = _node(_grid(2, 2, [0, 0, 0]))
    node.portal_goal_known_free_radius_m = 0.0
    node.portal_goal_occupied_threshold = 50

    assert node._goal_is_known_free([0.5, 0.5, 0.0]) is False


def test_mismatched_occ_odom_pair_does_not_hard_filter_candidate():
    # The goal lies in an occupied cell, but the source stamps are ten seconds
    # apart.  A stale/mixed pair must be retained for move_base to re-check.
    node = _node(_grid(2, 1, [0, 100]), source_stamp_delta=10.0)

    result = node._filter_candidate_goals_by_occupancy([_candidate()])

    assert len(result) == 1
    assert node.filtered_occupancy_candidate_ids == []
    assert node.occupancy_preflight_timing["pair_reason"] == "pair_age_exceeded"
    assert node.occupancy_preflight_timing["pair_accepted"] is False


def test_coherent_occ_odom_pair_preserves_hard_filter_behavior():
    node = _node(_grid(2, 1, [0, 100]))

    result = node._filter_candidate_goals_by_occupancy([_candidate()])

    assert result == []
    assert node.filtered_occupancy_candidate_ids == ["frontier:occupied"]
    assert node.occupancy_preflight_timing["pair_reason"] == "paired"


def test_free_centreline_through_narrow_door_is_not_a_safe_body_route():
    # Two rooms connected by a one-cell opening: endpoints and centre-line
    # are free, but a one-cell-radius body cannot pass through the doorway.
    width, height = 13, 9
    data = [0] * (width * height)
    for row in range(height):
        if row != 4:
            data[row * width + 6] = 100
    node = _node(_grid(width, height, data))
    node.robot_xy = (2.5, 4.5)
    node.candidate_occupancy_endpoint_radius_m = 1.0
    node.candidate_occupancy_use_astar = True
    node.candidate_occupancy_astar_max_expansions = 1000
    candidate = _candidate()
    candidate.goal_xyyaw = [10.5, 4.5, 0.0]
    assert node._filter_candidate_goals_by_occupancy([candidate]) == []
    assert node.occupancy_preflight_timing["astar_no_path_count"] == 1

    # Widen the opening to fit the same robot and publish a fresh grid receipt.
    for row in (3, 5):
        data[row * width + 6] = 0
    node.occupancy_grid = _grid(width, height, list(data))
    candidate = _candidate()
    candidate.goal_xyyaw = [10.5, 4.5, 0.0]
    assert len(node._filter_candidate_goals_by_occupancy([candidate])) == 1
    assert node.occupancy_preflight_timing["astar_reachable_count"] == 1


def test_local_path_around_wall_is_used_even_when_straight_ray_is_blocked():
    width, height = 13, 9
    data = [0] * (width * height)
    for row in range(6):
        data[row * width + 6] = 100
    node = _node(_grid(width, height, data))
    node.robot_xy = (2.5, 4.5)
    node.candidate_occupancy_use_astar = True
    node.candidate_occupancy_astar_max_expansions = 1000
    candidate = _candidate()
    candidate.goal_xyyaw = [10.5, 4.5, 0.0]
    assert len(node._filter_candidate_goals_by_occupancy([candidate])) == 1
    assert node.occupancy_preflight_timing["astar_reachable_count"] == 1


def test_callbacks_record_source_and_receipt_timestamps():
    node = object.__new__(SemanticCandidateNode)
    node._occupancy_state_lock = threading.RLock()
    node.robot_xy = None
    node.robot_xy_stamp_s = None
    node.robot_xy_received_at = None
    node.occupancy_grid = None
    node.occupancy_grid_stamp_s = None
    node.occupancy_grid_received_at = None
    stamp = SimpleNamespace(to_sec=lambda: 42.5)
    odom = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.0, y=2.0),
            )
        ),
    )
    grid = _grid(1, 1, [0])
    grid.header.stamp = stamp

    node._odom_callback(odom)
    node._occupancy_callback(grid)

    assert node.robot_xy == (1.0, 2.0)
    assert node.robot_xy_stamp_s == 42.5
    assert node.occupancy_grid_stamp_s == 42.5
    assert node.robot_xy_received_at is not None
    assert node.occupancy_grid_received_at is not None


def test_nonpositive_pair_age_disables_timestamp_guard_for_replay():
    node = _node(_grid(2, 1, [0, 100]), max_pair_age_s=0.0)
    node.robot_xy_stamp_s = None
    node.occupancy_grid_stamp_s = None
    node.robot_xy_received_at = None
    node.occupancy_grid_received_at = None

    result = node._filter_candidate_goals_by_occupancy([_candidate()])

    assert result == []
    assert node.occupancy_preflight_timing["pair_reason"] == "age_check_disabled"
