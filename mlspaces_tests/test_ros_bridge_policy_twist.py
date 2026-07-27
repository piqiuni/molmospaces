import inspect
import math
import threading

import numpy as np

from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy


def test_odom_twist_is_disabled_by_default() -> None:
    parameter = inspect.signature(RosBridgePolicy.__init__).parameters["publish_odom_twist"]

    assert parameter.default is False


def test_estimate_planar_twist_uses_body_frame() -> None:
    previous = np.array([1.0, 2.0, math.pi / 2.0], dtype=np.float32)
    current = np.array([1.0, 2.2, math.pi / 2.0], dtype=np.float32)

    vx, vy, wz = RosBridgePolicy._estimate_planar_twist(previous, current, 0.2)

    assert math.isclose(vx, 1.0, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)
    assert math.isclose(wz, 0.0, abs_tol=1e-6)


def test_estimate_planar_twist_wraps_yaw_delta() -> None:
    previous = np.array([0.0, 0.0, math.pi - 0.05], dtype=np.float32)
    current = np.array([0.0, 0.0, -math.pi + 0.05], dtype=np.float32)

    vx, vy, wz = RosBridgePolicy._estimate_planar_twist(previous, current, 0.2)

    assert math.isclose(vx, 0.0, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)
    assert math.isclose(wz, 0.5, abs_tol=1e-5)


def test_extract_planar_twist_uses_instantaneous_base_qvel() -> None:
    base_group = type("BaseGroup", (), {"joint_vel": np.array([0.0, 1.0, 0.4])})()
    robot_view = type(
        "RobotView",
        (),
        {"get_move_group": lambda self, name: base_group},
    )()
    robot = type("Robot", (), {"robot_view": robot_view})()
    env = type("Env", (), {"current_robot": robot})()
    task = type("Task", (), {"env": env})()
    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy.task = task

    twist = policy._extract_planar_twist_from_task(math.pi / 2.0)

    assert twist is not None
    vx, vy, wz = twist
    assert math.isclose(vx, 1.0, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)
    assert math.isclose(wz, 0.4, abs_tol=1e-6)


def test_publish_realtime_gt_now_forces_current_snapshot() -> None:
    calls = []

    class Publisher:
        def publish(self, task, *, stamp, step_index, force):
            calls.append((task, stamp, step_index, force))
            return {"frame_index": step_index}

    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy.task = object()
    policy._step_idx = 7
    policy._realtime_gt_publisher = Publisher()
    policy._latest_gt_payload = None
    policy._next_common_stamp = lambda: "stamp"

    payload = policy.publish_realtime_gt_now(step_index=11)

    assert payload == {"frame_index": 11}
    assert policy._latest_gt_payload == payload
    assert calls == [(policy.task, "stamp", 11, True)]


def test_missing_navigation_arm_actions_hold_the_current_reset_pose() -> None:
    left = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    right = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    base = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    class RobotView:
        def move_group_ids(self):
            return ["base", "left_arm", "right_arm"]

        def get_noop_ctrl_dict(self, names):
            values = {"base": base, "left_arm": left, "right_arm": right}
            return {name: values[name] for name in names}

    action = {"done": False}
    RosBridgePolicy._fill_missing_navigation_holds(action, RobotView())

    np.testing.assert_array_equal(action["base"], base)
    np.testing.assert_array_equal(action["left_arm"], left)
    np.testing.assert_array_equal(action["right_arm"], right)
    assert action["left_arm"] is not left
    assert action["right_arm"] is not right


def test_tf_keepalive_republishes_only_cached_pose_transforms() -> None:
    calls = []
    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy.publish_odom = True
    policy.base_frame_id = "base"
    policy.pointcloud_frame_id = "lidar"
    policy._tf_cache_lock = threading.Lock()
    policy._latest_odom_tf_state = tuple(float(value) for value in range(10))
    policy._latest_base_to_lidar_tf = tuple(float(value) for value in range(7))
    policy._next_common_stamp = lambda: "fresh-stamp"
    policy._publish_odom_and_base_tf_from_state = lambda state, stamp: calls.append(("odom", state, stamp))
    policy._publish_base_to_lidar_tf_from_state = lambda state, stamp: calls.append(("lidar", state, stamp))

    policy._tf_keepalive_callback(None)

    assert calls == [
        ("odom", tuple(float(value) for value in range(10)), "fresh-stamp"),
        ("lidar", tuple(float(value) for value in range(7)), "fresh-stamp"),
    ]
