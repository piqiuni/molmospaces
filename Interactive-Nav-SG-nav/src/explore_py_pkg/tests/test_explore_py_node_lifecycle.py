import math
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rospy
from nav_msgs.msg import Odometry

import explore_py_node
from explore_py_node import ExplorePyNode


class _FakeState:
    def __init__(self):
        self.seen = []

    def update_seen_clusters(self, clusters):
        self.seen = list(clusters)


class _ResetDuringExtractionCore:
    def __init__(self, node, cluster):
        self.node = node
        self.cluster = cluster
        self.selected_robot_xy = None

    def extract_frontier_clusters(self, grid, robot_xy, value_provider, state):
        assert grid is self.node.latest_grid
        assert value_provider is self.node.value_fusion
        self.node._reset_generation += 1
        self.node.robot_xy = None
        return [self.cluster]

    def select_initial_local_cluster(self, clusters, robot_xy, robot_yaw):
        self.selected_robot_xy = robot_xy
        return clusters[0]


class _ClosedPublisher:
    def publish(self, _msg):
        raise rospy.ROSException("publish() to a closed topic")


class _RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _IdleState:
    active_goal = None


class _ActiveState:
    active_goal = object()


def _odom_with_yaw(yaw: float) -> Odometry:
    msg = Odometry()
    msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


def test_compute_next_subgoal_discards_result_from_reset_generation():
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node._reset_generation = 4
    node.latest_grid = object()
    node.robot_xy = (1.0, 2.0)
    node.robot_yaw = 0.3
    node.value_fusion = object()
    node.state = _FakeState()
    node.sent_goal_count = 0
    node.initial_local_goal_count = 3
    node.latest_clusters = ["previous"]
    node.last_selected_cluster = "previous"
    cluster = object()
    node.core = _ResetDuringExtractionCore(node, cluster)

    result = node.compute_next_subgoal(publish_selection=True)

    assert result is None
    assert node.core.selected_robot_xy == (1.0, 2.0)
    assert node.latest_clusters == ["previous"]
    assert node.last_selected_cluster == "previous"


def test_initial_spin_timer_ignores_closed_publisher(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_active = True
    node.initial_spin_done = False
    node.initial_spin_angular_speed = 0.4
    node.initial_spin_accumulated_yaw = 0.0
    node.initial_spin_angle_rad = 1.0
    node.initial_spin_start_time = 0.0
    node.initial_spin_timeout_sec = float("inf")
    node.state = _IdleState()
    node.cmd_vel_pub = _ClosedPublisher()
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)

    node._initial_spin_cmd_timer_callback(None)


def test_initial_spin_tick_starts_and_high_rate_odom_stops_at_yaw_threshold(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_active = False
    node.initial_spin_done = False
    node.initial_spin_start_time = 0.0
    node.initial_spin_done_time = 0.0
    node.initial_spin_last_yaw = None
    node.initial_spin_accumulated_yaw = 0.0
    node.initial_spin_angle_rad = 0.5
    node.initial_spin_timeout_sec = 20.0
    node.initial_spin_angular_speed = 0.4
    node.initial_spin_reason = "pending"
    node.latest_grid = object()
    node.robot_xy = (0.0, 0.0)
    node.robot_yaw = 3.0
    node.state = _IdleState()
    node.cmd_vel_pub = _RecordingPublisher()
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(explore_py_node.time, "time", lambda: 42.0)

    # Preserve the normal explorer tick as the scan arm point.  The command
    # timer only maintains an already-running scan.
    node._tick_initial_spin()
    assert node.initial_spin_active is True
    assert node.initial_spin_start_time == 42.0
    assert node.cmd_vel_pub.messages[-1].angular.z == 0.4

    # This crosses the -pi/pi boundary.  Completion occurs in odom_callback,
    # with no call to _tick_initial_spin.
    node.odom_callback(_odom_with_yaw(-2.75))

    assert node.initial_spin_done is True
    assert node.initial_spin_active is False
    assert node.initial_spin_reason == "completed"
    assert node.initial_spin_accumulated_yaw > node.initial_spin_angle_rad
    assert node.cmd_vel_pub.messages[-1].angular.z == 0.0


def test_initial_spin_timer_does_not_start_an_inactive_pending_spin(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_active = False
    node.initial_spin_done = False
    node.initial_spin_start_time = 0.0
    node.initial_spin_done_time = 0.0
    node.initial_spin_last_yaw = None
    node.initial_spin_accumulated_yaw = 0.0
    node.initial_spin_angle_rad = 1.0
    node.initial_spin_timeout_sec = 20.0
    node.initial_spin_angular_speed = 0.4
    node.initial_spin_reason = "pending"
    node.latest_grid = object()
    node.robot_xy = (0.0, 0.0)
    node.robot_yaw = 0.0
    node.state = _IdleState()
    node.cmd_vel_pub = _RecordingPublisher()
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(explore_py_node.time, "time", lambda: 42.0)

    node._initial_spin_cmd_timer_callback(None)

    assert node.initial_spin_active is False
    assert node.initial_spin_start_time == 0.0
    assert node.initial_spin_reason == "pending"
    assert node.cmd_vel_pub.messages == []


def test_initial_spin_stops_immediately_when_an_active_goal_arrives(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_active = True
    node.initial_spin_done = False
    node.initial_spin_start_time = 10.0
    node.initial_spin_done_time = 0.0
    node.initial_spin_accumulated_yaw = 0.2
    node.initial_spin_angle_rad = 1.0
    node.initial_spin_timeout_sec = 20.0
    node.initial_spin_reason = "spinning"
    node.state = _ActiveState()
    node.cmd_vel_pub = _RecordingPublisher()
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(explore_py_node.time, "time", lambda: 11.0)

    node._initial_spin_cmd_timer_callback(None)

    assert node.initial_spin_done is True
    assert node.initial_spin_active is False
    assert node.initial_spin_reason == "skipped_active_goal"
    assert node.cmd_vel_pub.messages[-1].angular.z == 0.0


def test_initial_spin_timeout_without_odom_stops_the_robot(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_active = True
    node.initial_spin_done = False
    node.initial_spin_start_time = 10.0
    node.initial_spin_done_time = 0.0
    node.initial_spin_accumulated_yaw = 0.0
    node.initial_spin_angle_rad = 1.0
    node.initial_spin_timeout_sec = 2.0
    node.initial_spin_reason = "spinning"
    node.state = _IdleState()
    node.cmd_vel_pub = _RecordingPublisher()
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)

    node._advance_initial_spin(12.1)

    assert node.initial_spin_done is True
    assert node.initial_spin_active is False
    assert node.initial_spin_reason == "timeout"
    assert node.cmd_vel_pub.messages[-1].angular.z == 0.0


def test_initial_spin_settle_gates_navigation_without_republishing_spin_command(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_enabled = True
    node.initial_spin_active = False
    node.initial_spin_done = True
    node.initial_spin_done_time = 10.0
    node.initial_spin_reason = "completed"
    node.initial_spin_settle_sec = 1.0
    node.state = _IdleState()
    node.cmd_vel_pub = _RecordingPublisher()
    now = [10.5]
    monkeypatch.setattr(explore_py_node.time, "time", lambda: now[0])
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)

    assert node._should_run_initial_spin() is True
    node._initial_spin_cmd_timer_callback(None)
    assert node.cmd_vel_pub.messages == []

    now[0] = 11.0
    assert node._should_run_initial_spin() is False


def test_initial_spin_settle_applies_after_timeout_and_not_when_disabled_or_active(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_enabled = True
    node.initial_spin_done = True
    node.initial_spin_done_time = 10.0
    node.initial_spin_reason = "timeout"
    node.initial_spin_settle_sec = 1.0
    node.state = _IdleState()
    monkeypatch.setattr(explore_py_node.time, "time", lambda: 10.5)

    assert node._should_run_initial_spin() is True

    node.initial_spin_enabled = False
    assert node._should_run_initial_spin() is False

    node.initial_spin_enabled = True
    node.state = _ActiveState()
    assert node._should_run_initial_spin() is False


def test_initial_spin_timer_does_not_command_before_map_and_odom_are_ready(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_active = False
    node.initial_spin_done = False
    node.initial_spin_reason = "pending"
    node.latest_grid = None
    node.robot_xy = None
    node.robot_yaw = None
    node.state = _IdleState()
    node.cmd_vel_pub = _RecordingPublisher()
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(explore_py_node.time, "time", lambda: 42.0)

    node._initial_spin_cmd_timer_callback(None)

    assert node.initial_spin_active is False
    assert node.initial_spin_done is False
    assert node.cmd_vel_pub.messages == []


def test_initial_spin_tick_uses_the_shared_controller(monkeypatch):
    node = ExplorePyNode.__new__(ExplorePyNode)
    node._lifecycle_lock = threading.RLock()
    node.initial_spin_active = False
    node.initial_spin_done = False
    node.initial_spin_start_time = 0.0
    node.initial_spin_done_time = 0.0
    node.initial_spin_last_yaw = None
    node.initial_spin_accumulated_yaw = 0.0
    node.initial_spin_angle_rad = 1.0
    node.initial_spin_timeout_sec = 20.0
    node.initial_spin_angular_speed = 0.4
    node.initial_spin_reason = "pending"
    node.latest_grid = object()
    node.robot_xy = (0.0, 0.0)
    node.robot_yaw = 0.0
    node.state = _IdleState()
    node.cmd_vel_pub = _RecordingPublisher()
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(explore_py_node.time, "time", lambda: 42.0)

    node._tick_initial_spin()

    assert node.initial_spin_active is True
    assert node.initial_spin_start_time == 42.0
    assert node.cmd_vel_pub.messages[-1].angular.z == 0.4


def test_reset_initial_spin_state_clears_yaw_for_the_next_scene():
    node = ExplorePyNode.__new__(ExplorePyNode)
    node.initial_spin_enabled = True
    node.initial_spin_done = True
    node.initial_spin_active = True
    node.initial_spin_start_time = 10.0
    node.initial_spin_done_time = 11.0
    node.initial_spin_last_yaw = 1.0
    node.initial_spin_accumulated_yaw = 6.4
    node.initial_spin_reason = "completed"

    node._reset_initial_spin_state()

    assert node.initial_spin_done is False
    assert node.initial_spin_active is False
    assert node.initial_spin_start_time == 0.0
    assert node.initial_spin_done_time == 0.0
    assert node.initial_spin_last_yaw is None
    assert node.initial_spin_accumulated_yaw == 0.0
    assert node.initial_spin_reason == "pending"


def test_tick_keeps_dispatched_goal_when_frontier_refresh_removes_source():
    class ActiveGoal:
        point = (1.0, 2.0)

    class State:
        def __init__(self):
            self.active_goal = ActiveGoal()
            self.frontier_gone_checks = 0

        def update_goal_progress(self, _robot_xy, robot_yaw=None):
            del robot_yaw
            return None

        def mark_active_covered_if_frontier_gone(self, *_args, **_kwargs):
            self.frontier_gone_checks += 1
            self.active_goal = None

    class Core:
        @staticmethod
        def is_free_world(_grid, _point):
            return True

    node = ExplorePyNode.__new__(ExplorePyNode)
    node.latest_grid = object()
    node.robot_xy = (0.0, 0.0)
    node.robot_yaw = 0.0
    node.external_behavior_control = False
    node.state = State()
    node.core = Core()
    node.goal_republish_interval_sec = 0.0
    node._should_run_initial_spin = lambda: False
    node._active_goal_has_frontier = lambda: False
    node._maybe_replan_rotation_oscillation = lambda: False
    node._fail_if_global_plan_not_current_goal = lambda: None
    node._fail_if_local_plan_missing = lambda: None
    node._publish_frontiers = lambda: None
    node._publish_status = lambda: None
    cancellations = []
    node._cancel_move_base_goal = cancellations.append

    node.tick(None)

    assert node.state.active_goal is not None
    assert node.state.frontier_gone_checks == 0
    assert cancellations == []
