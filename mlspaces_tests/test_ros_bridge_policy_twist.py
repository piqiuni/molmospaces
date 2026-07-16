import inspect
import math

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
