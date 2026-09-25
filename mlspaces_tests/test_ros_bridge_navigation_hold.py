from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest

from molmo_spaces.policy.learned_policy import ros_bridge_policy as bridge


@pytest.fixture
def harness(monkeypatch):
    events = []
    clock = SimpleNamespace(now=10.0, sleeps=0)

    def sleep(seconds):
        clock.now += seconds
        clock.sleeps += 1

    monkeypatch.setattr(bridge, "time", SimpleNamespace(
        monotonic=lambda: clock.now, perf_counter=lambda: clock.now, sleep=sleep,
    ))
    policy = bridge.RosBridgePolicy.__new__(bridge.RosBridgePolicy)
    policy._timing_acc_ms = {}
    policy._step_idx = 0
    policy._rospy = SimpleNamespace(
        is_shutdown=lambda: False, loginfo=Mock(), logwarn=Mock(),
        loginfo_throttle=Mock(), logwarn_throttle=Mock(),
    )
    policy._next_common_stamp = lambda: SimpleNamespace(to_sec=lambda: 100.0 + policy._step_idx)
    policy._publish_odom_and_tf = lambda *_: events.append("odom") or True
    policy._realtime_gt_publisher = None
    policy.map_warmup_skip_frames = 0
    policy.step_ready_barrier_enabled = True
    policy._step_ready_bootstrap_complete = True
    policy.step_ready_timeout_s = 30.0
    policy.step_ready_bootstrap_timeout_s = 60.0
    policy.immediate_noop_after_publish = False
    policy._extract_image_from_observation = lambda _: np.zeros((2, 2, 3), dtype=np.uint8)
    policy._to_image_msg = lambda *_, **__: object()
    policy._obs_pub = SimpleNamespace(publish=lambda _: events.append("rgb"))
    policy._enqueue_step_frame = lambda *args: events.append(("frame", args[2]))
    policy._step_frame_queue = SimpleNamespace(qsize=lambda: 0)
    policy._step_frame_queue_peak = 0
    policy._extra_image_pub = None
    policy.publish_pointcloud = True
    policy.publish_camera_info = False
    policy.depth_camera_name = "head_camera"
    policy._depth_scan_pub = None
    policy._extract_depth_from_observation = lambda _: ("head_camera_depth", np.ones((2, 2)))
    policy._extract_intrinsics_from_observation = lambda *_: (1.0, 1.0, 1.0, 1.0)
    policy._normalize_intrinsics_to_image_shape = lambda intrinsics, **_: intrinsics
    policy._intrinsics_from_fov = lambda *_, **__: (1.0, 1.0, 1.0, 1.0)
    policy._to_depth_msg = lambda *_, **__: object()
    policy._depth_to_pointcloud_msg = lambda *_, **__: object()
    policy._depth_pub = SimpleNamespace(publish=lambda _: events.append("depth"))
    policy._pointcloud_pub = SimpleNamespace(publish=lambda _: events.append("pointcloud"))

    def ready(**kwargs):
        events.append(("ready", kwargs["reason"]))
        assert kwargs["require_current_stamp"] is True
        assert "rgb" in events and "depth" in events and "pointcloud" in events
        policy.last_step_ready_diagnostics = {"ready_satisfied": True, "timed_out": False}
        return True

    policy._wait_for_step_ready = Mock(side_effect=ready)
    policy._publish_fresh_command_gate = lambda _: events.append("gate")
    policy._publish_step_sync = lambda _: events.append("sync")
    policy._wait_for_step_capture_ack = lambda _: events.append("capture_ack") or True
    policy._record_timing = lambda value: setattr(policy, "last_timing_ms", dict(value))
    policy.action_timeout_s = 0.02
    policy.blocking_observation_republish_period_s = 0.0
    policy.cmd_vel_timeout_s = 1.0
    policy.require_fresh_cmd_vel = True
    policy.require_move_base_active_for_cmd_vel = False
    policy._lock = threading.Lock()
    policy._latest_action = None
    policy._latest_cmd_vel = None
    policy._latest_cmd_vel_mono_s = 0.0
    policy._move_base_active = False
    policy._cmd_vel_to_base_action = Mock(return_value={"base": np.array([1., 2., 3.])})
    policy.last_cmd_vel_lateral_rejected = False
    view = SimpleNamespace(
        move_group_ids=lambda: ["base", "left_arm", "right_arm"],
        get_noop_ctrl_dict=lambda names: {name: np.zeros(3) for name in names},
    )
    policy.task = SimpleNamespace(env=SimpleNamespace(current_robot=SimpleNamespace(robot_view=view)))
    return SimpleNamespace(policy=policy, clock=clock, events=events)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_hold_skips_only_command_wait_and_keeps_sensors_ready_capture(harness, bootstrap):
    policy, events = harness.policy, harness.events
    policy._step_ready_bootstrap_complete = not bootstrap
    policy.last_action_timed_out = True
    # A queued action must not take control during an interaction hold.
    policy._latest_action = {"base": np.ones(3), "done": True}
    action = policy.get_action({}, hold_navigation=True)

    assert action["done"] is False
    assert set(action) == {"base", "left_arm", "right_arm", "done"}
    assert harness.clock.sleeps == 0
    assert not policy.last_action_timed_out
    assert policy.last_action_source == "navigation_hold"
    policy._cmd_vel_to_base_action.assert_not_called()
    assert policy._step_idx == 1
    assert policy.last_timing_ms["action_wait_after_ready"] == 0.0
    assert events[:5] == ["odom", "rgb", ("frame", 0), "depth", "pointcloud"]
    assert events[5:] == (
        ([("ready", "bootstrap")] if bootstrap else [])
        + [("ready", "barrier"), "gate", "sync", "capture_ack"]
    )


def test_hold_keeps_waiting_until_ready(harness):
    policy = harness.policy
    # Exercise the actual barrier wait, with readiness arriving on a later poll.
    policy._wait_for_step_ready = bridge.RosBridgePolicy._wait_for_step_ready.__get__(policy)
    policy.step_ready_bootstrap_republish_period_s = 0.5
    policy._step_ready_for_current = lambda **_: harness.clock.sleeps >= 3
    policy._step_ready_payload = lambda: {}
    policy.get_action({}, hold_navigation=True)

    assert harness.clock.sleeps == 3
    assert policy.last_action_source == "navigation_hold"
    assert policy.last_timing_ms["step_ready_wait"] > 0
    assert policy.last_timing_ms["action_wait_after_ready"] == pytest.approx(0.0)


def test_hold_does_not_bypass_failed_ready(harness):
    def timeout(**_):
        harness.policy.last_step_ready_diagnostics = {"ready_satisfied": False, "timed_out": True}
        return False

    harness.policy._wait_for_step_ready.side_effect = timeout
    harness.policy.get_action({}, hold_navigation=True)
    assert harness.clock.sleeps > 0
    assert harness.policy.last_action_timed_out
    assert harness.policy.last_action_source == "timeout_noop"


def test_hold_without_enabled_ready_barrier(harness):
    harness.policy.step_ready_barrier_enabled = False
    harness.policy.get_action({}, hold_navigation=True)
    harness.policy._wait_for_step_ready.assert_not_called()
    assert harness.clock.sleeps == 0
    assert harness.policy.last_action_source == "navigation_hold"


def test_hold_does_not_mask_ros_shutdown(harness):
    harness.policy._rospy.is_shutdown = lambda: True
    assert harness.policy.get_action({}, hold_navigation=True) is None
    assert harness.policy.last_action_source == "ros_shutdown"
    assert harness.policy._step_idx == 0


def test_normal_navigation_still_waits_and_can_time_out(harness):
    harness.policy.get_action({})
    assert harness.clock.sleeps > 0
    assert harness.policy.last_action_timed_out
    assert harness.policy.last_action_source == "timeout_noop"


def test_navigation_resumes_with_fresh_command_after_hold(harness):
    policy = harness.policy
    policy.get_action({}, hold_navigation=True)

    def fresh_command(_):
        policy._latest_cmd_vel = object()
        policy._latest_cmd_vel_mono_s = harness.clock.now

    policy._publish_fresh_command_gate = fresh_command
    action = policy.get_action({})
    assert policy.last_action_source == "cmd_vel"
    assert not policy.last_action_timed_out
    np.testing.assert_array_equal(action["base"], [1., 2., 3.])
    assert policy._step_idx == 2


def test_rollout_holds_twenty_steps_without_skipping_task_or_torso(harness):
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "InteractiveNav"))
    from scripts.InteractiveNav.run_nav_ros_sim import NavRosRolloutRunner

    policy = harness.policy
    task = policy.task
    actions, before, after = [], [], []
    task.env.current_model = SimpleNamespace(opt=SimpleNamespace(enableflags=0))
    task.config = SimpleNamespace(policy_dt_ms=200.0)
    task.reset = lambda: ({}, {})
    task.get_observations = lambda: {}
    task.observation_cache = []
    task.is_done = lambda: len(actions) >= 21
    task.step = lambda action: actions.append(action) or ({}, 0., False, False, [{}])
    controller = SimpleNamespace(
        prepare=lambda _: None,
        before_step=lambda _task, step: before.append(step),
        after_step=lambda _task, step: after.append(step),
        should_pause_navigation=lambda: len(actions) < 20,
        consume_force_observation_request=lambda: False,
        view_torso_target=lambda: [.4],
        after_task_step=lambda: None,
        finalize=lambda _: None,
    )
    policy.force_interaction_controller = controller
    policy.prepare_episode_reset = lambda: None
    policy.reset = lambda: None
    policy.max_consecutive_action_timeouts = 1
    policy.step_log_every_n_steps = 0
    policy.sim_timing_log_every_n_steps = 0
    policy.publish_realtime_gt_now = Mock()

    def fresh_command(_):
        if len(actions) == 20:
            policy._latest_cmd_vel = object()
            policy._latest_cmd_vel_mono_s = harness.clock.now

    policy._publish_fresh_command_gate = fresh_command
    NavRosRolloutRunner.run_single_rollout(episode_seed=1, task=task, policy=policy)

    assert before == after == list(range(21))
    assert len(actions) == policy._step_idx == 21
    for action in actions[:20]:
        assert "base" not in action
        np.testing.assert_array_equal(action["torso"], [.4])
    np.testing.assert_array_equal(actions[20]["base"], [1., 2., 3.])
    assert harness.events.count("capture_ack") == 21
    assert [e for e in harness.events if isinstance(e, tuple) and e[0] == "frame"] == [
        ("frame", step) for step in range(21)
    ]
    assert policy._wait_for_step_ready.call_count == 21
    assert harness.clock.sleeps == 0
