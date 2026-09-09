from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from scripts.InteractiveNav.manual_interactive_nav_camera import (
    ManualExocentricCameraController,
)
from scripts.InteractiveNav.manual_interactive_nav_policy import (
    CameraControlCommand,
    ManualInteractiveNavPolicy,
)


def fake_env(yaw: float = 0.0):
    pose = np.eye(4, dtype=float)
    pose[:2, :2] = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    base = SimpleNamespace(pose=pose)
    robot_view = SimpleNamespace(base=base)
    robot = SimpleNamespace(robot_view=robot_view)
    return SimpleNamespace(current_robot=robot)


def test_forward_and_backward_follow_robot_yaw():
    policy = ManualInteractiveNavPolicy(env=fake_env(math.pi / 2), start_listener=False)
    policy.press_key("w")
    np.testing.assert_allclose(policy.get_action()["base"], [0.0, 0.035, 0.0], atol=1e-6)
    policy.release_key("w")
    policy.press_key("s")
    np.testing.assert_allclose(policy.get_action()["base"], [0.0, -0.035, 0.0], atol=1e-6)


def test_a_and_d_rotate_in_place_and_cancel():
    policy = ManualInteractiveNavPolicy(env=fake_env(), start_listener=False)
    policy.press_key("a")
    action = policy.get_action()["base"]
    np.testing.assert_allclose(action[:2], [0.0, 0.0])
    assert action[2] > 0
    policy.press_key("d")
    np.testing.assert_allclose(policy.get_action()["base"], [0.0, 0.0, 0.0])


def test_op_events_are_edge_triggered():
    policy = ManualInteractiveNavPolicy(env=fake_env(), start_listener=False)
    policy.press_key("o")
    policy.press_key("o")
    assert [event.name for event in policy.drain_events()] == ["open_nearest"]
    policy.release_key("o")
    policy.press_key("o")
    policy.press_key("p")
    assert [event.name for event in policy.drain_events()] == [
        "open_nearest",
        "close_nearest",
    ]


def test_release_stops_continuous_commands():
    policy = ManualInteractiveNavPolicy(env=fake_env(), start_listener=False)
    policy.press_key("w")
    assert np.linalg.norm(policy.get_action()["base"]) > 0
    policy.release_key("w")
    np.testing.assert_allclose(policy.get_action()["base"], [0.0, 0.0, 0.0])


def test_camera_key_mapping():
    policy = ManualInteractiveNavPolicy(env=fake_env(), start_listener=False)
    for key in ("i", "l", ";", "."):
        policy.press_key(key)
    command = policy.get_camera_command()
    assert command.forward > 0
    assert command.right > 0
    assert command.yaw > 0
    assert command.pitch > 0


def test_extra_window_keys_drive_robot_and_camera_without_listener():
    policy = ManualInteractiveNavPolicy(env=fake_env(), start_listener=False)
    action = policy.get_action(extra_pressed={"w", "a"})["base"]
    assert action[0] > 0
    assert action[2] > 0
    command = policy.get_camera_command(extra_pressed={"i", "l"})
    assert command.forward > 0
    assert command.right > 0


def test_camera_pose_is_orthonormal_and_screen_right_is_consistent():
    camera = ManualExocentricCameraController(
        camera_name="test",
        position=np.asarray([0.0, 0.0, 2.0]),
        forward=np.asarray([1.0, 0.0, -1.0]),
    )
    pose = camera.pose()
    assert np.isclose(np.linalg.norm(pose.forward), 1.0)
    assert np.isclose(np.linalg.norm(pose.up), 1.0)
    assert np.isclose(np.dot(pose.forward, pose.up), 0.0, atol=1e-6)
    before = camera.position.copy()
    camera.apply(CameraControlCommand(right=1.0))
    np.testing.assert_allclose(camera.position - before, [0.0, -1.0, 0.0], atol=1e-6)


def test_camera_pitch_clamps_and_reset_restores_initial_pose():
    camera = ManualExocentricCameraController.from_spherical(
        camera_name="test",
        target=np.asarray([0.0, 0.0, 0.0]),
        distance=5.0,
        azimuth_deg=0.0,
        elevation_deg=-50.0,
    )
    initial = camera.pose()
    camera.apply(CameraControlCommand(forward=1.0, yaw=0.5, pitch=10.0))
    assert camera.pitch <= camera.max_pitch
    reset = camera.reset()
    np.testing.assert_allclose(reset.position, initial.position)
    np.testing.assert_allclose(reset.forward, initial.forward)
    np.testing.assert_allclose(reset.up, initial.up)


def test_over_shoulder_camera_uses_robot_frame_offsets():
    robot_pose = np.eye(4, dtype=float)
    robot_pose[:2, :2] = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    robot_pose[:3, 3] = [4.0, 5.0, 0.0]
    camera = ManualExocentricCameraController.from_robot_pose(
        camera_name="test",
        robot_pose=robot_pose,
        position_offset_robot=np.asarray([-2.0, -1.0, 2.0]),
        lookat_offset_robot=np.asarray([1.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(camera.position, [5.0, 3.0, 2.0], atol=1e-6)
    expected_target = np.asarray([4.0, 6.0, 1.0])
    expected_forward = expected_target - camera.position
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(camera.pose().forward, expected_forward, atol=1e-6)


def test_over_shoulder_camera_follows_robot_and_keeps_relative_pose():
    robot_pose = np.eye(4, dtype=float)
    camera = ManualExocentricCameraController.from_robot_pose(
        camera_name="test",
        robot_pose=robot_pose,
        position_offset_robot=np.asarray([-2.0, -1.0, 2.0]),
        lookat_offset_robot=np.asarray([1.0, 0.0, 1.0]),
    )
    moved_robot_pose = np.eye(4, dtype=float)
    moved_robot_pose[:2, :2] = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    moved_robot_pose[:3, 3] = [4.0, 5.0, 0.0]
    camera.follow_robot_pose(moved_robot_pose)
    expected_position = moved_robot_pose[:3, 3] + moved_robot_pose[:3, :3] @ np.asarray(
        [-2.0, -1.0, 2.0]
    )
    np.testing.assert_allclose(camera.position, expected_position, atol=1e-6)
    camera.apply(CameraControlCommand(forward=0.2), robot_pose=moved_robot_pose)
    position_after_manual_move = camera.position.copy()
    translated_robot_pose = moved_robot_pose.copy()
    translated_robot_pose[:2, 3] += [1.0, 2.0]
    camera.follow_robot_pose(translated_robot_pose)
    np.testing.assert_allclose(
        camera.position - position_after_manual_move, [1.0, 2.0, 0.0], atol=1e-6
    )
