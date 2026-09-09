from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.InteractiveNav.manual_interactive_nav_policy import CameraControlCommand


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a degenerate camera vector")
    return np.asarray(vector, dtype=np.float32) / norm


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    yaw: float
    pitch: float


class ManualExocentricCameraController:
    def __init__(
        self,
        *,
        camera_name: str,
        position: np.ndarray,
        forward: np.ndarray,
        fov_deg: float = 55.0,
        min_pitch_deg: float = -85.0,
        max_pitch_deg: float = -5.0,
    ) -> None:
        self.camera_name = str(camera_name)
        self.fov_deg = float(fov_deg)
        self.min_pitch = math.radians(float(min_pitch_deg))
        self.max_pitch = math.radians(float(max_pitch_deg))
        forward = _normalize(np.asarray(forward, dtype=float))
        self._initial_position = np.asarray(position, dtype=np.float32).copy()
        self._initial_yaw = float(math.atan2(forward[1], forward[0]))
        self._initial_pitch = float(math.asin(np.clip(forward[2], -1.0, 1.0)))
        self.position = self._initial_position.copy()
        self.yaw = self._initial_yaw
        self.pitch = float(np.clip(self._initial_pitch, self.min_pitch, self.max_pitch))
        self._robot_anchor_pose: np.ndarray | None = None
        self._relative_position_robot: np.ndarray | None = None
        self._relative_yaw_robot: float | None = None
        self._initial_relative_position_robot: np.ndarray | None = None
        self._initial_relative_yaw_robot: float | None = None

    @classmethod
    def from_spherical(
        cls,
        *,
        camera_name: str,
        target: np.ndarray,
        distance: float,
        azimuth_deg: float,
        elevation_deg: float,
        fov_deg: float = 55.0,
    ) -> "ManualExocentricCameraController":
        target = np.asarray(target, dtype=float)
        azimuth = math.radians(float(azimuth_deg))
        elevation = math.radians(float(elevation_deg))
        offset = float(distance) * np.asarray(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                -math.sin(elevation),
            ],
            dtype=float,
        )
        position = target + offset
        return cls(
            camera_name=camera_name,
            position=position,
            forward=target - position,
            fov_deg=fov_deg,
        )

    @staticmethod
    def _yaw_from_robot_pose(robot_pose: np.ndarray) -> float:
        return float(math.atan2(robot_pose[1, 0], robot_pose[0, 0]))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    @property
    def robot_anchored(self) -> bool:
        return self._relative_position_robot is not None

    def attach_to_robot(self, robot_pose: np.ndarray) -> CameraPose:
        robot_pose = np.asarray(robot_pose, dtype=float)
        if robot_pose.shape != (4, 4):
            raise ValueError(f"robot_pose must have shape (4, 4), got {robot_pose.shape}")
        rotation = robot_pose[:3, :3]
        translation = robot_pose[:3, 3]
        robot_yaw = self._yaw_from_robot_pose(robot_pose)
        self._relative_position_robot = (
            rotation.T @ (np.asarray(self.position, dtype=float) - translation)
        ).astype(np.float32)
        self._relative_yaw_robot = self._wrap_to_pi(self.yaw - robot_yaw)
        self._initial_relative_position_robot = self._relative_position_robot.copy()
        self._initial_relative_yaw_robot = float(self._relative_yaw_robot)
        return self.follow_robot_pose(robot_pose)

    def follow_robot_pose(self, robot_pose: np.ndarray) -> CameraPose:
        if not self.robot_anchored:
            return self.pose()
        robot_pose = np.asarray(robot_pose, dtype=float)
        if robot_pose.shape != (4, 4):
            raise ValueError(f"robot_pose must have shape (4, 4), got {robot_pose.shape}")
        assert self._relative_position_robot is not None
        assert self._relative_yaw_robot is not None
        self._robot_anchor_pose = robot_pose.copy()
        self.position = (
            robot_pose[:3, 3] + robot_pose[:3, :3] @ self._relative_position_robot
        ).astype(np.float32)
        self.yaw = self._wrap_to_pi(
            self._yaw_from_robot_pose(robot_pose) + self._relative_yaw_robot
        )
        return self.pose()

    @classmethod
    def from_robot_pose(
        cls,
        *,
        camera_name: str,
        robot_pose: np.ndarray,
        position_offset_robot: np.ndarray,
        lookat_offset_robot: np.ndarray,
        fov_deg: float = 65.0,
    ) -> "ManualExocentricCameraController":
        """Create a free camera from robot-frame offsets.

        RBY1 uses +x forward and +y left, so a negative y position offset
        places the camera behind the robot's right shoulder.
        """
        robot_pose = np.asarray(robot_pose, dtype=float)
        if robot_pose.shape != (4, 4):
            raise ValueError(f"robot_pose must have shape (4, 4), got {robot_pose.shape}")
        rotation = robot_pose[:3, :3]
        translation = robot_pose[:3, 3]
        position = translation + rotation @ np.asarray(position_offset_robot, dtype=float)
        target = translation + rotation @ np.asarray(lookat_offset_robot, dtype=float)
        camera = cls(
            camera_name=camera_name,
            position=position,
            forward=target - position,
            fov_deg=fov_deg,
        )
        camera.attach_to_robot(robot_pose)
        return camera

    def pose(self) -> CameraPose:
        forward = np.asarray(
            [
                math.cos(self.pitch) * math.cos(self.yaw),
                math.cos(self.pitch) * math.sin(self.yaw),
                math.sin(self.pitch),
            ],
            dtype=np.float32,
        )
        forward = _normalize(forward)
        right = _normalize(np.cross(forward, np.asarray([0.0, 0.0, 1.0])))
        up = _normalize(np.cross(right, forward))
        return CameraPose(
            position=self.position.copy(),
            forward=forward,
            up=up,
            yaw=float(self.yaw),
            pitch=float(self.pitch),
        )

    def apply(
        self,
        command: CameraControlCommand,
        *,
        robot_pose: np.ndarray | None = None,
    ) -> CameraPose:
        if self.robot_anchored:
            if robot_pose is not None:
                self.follow_robot_pose(robot_pose)
            if self._robot_anchor_pose is None:
                raise RuntimeError("Robot-anchored camera is missing its anchor pose")
        current = self.pose()
        flat_forward = np.asarray([current.forward[0], current.forward[1], 0.0])
        if float(np.linalg.norm(flat_forward)) < 1e-8:
            flat_forward = np.asarray([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        flat_forward = _normalize(flat_forward)
        # Match Camera.get_pose(): image-right is cross(forward, world-up).
        right = _normalize(np.asarray([flat_forward[1], -flat_forward[0], 0.0]))
        world_displacement = (
            float(command.forward) * flat_forward + float(command.right) * right
        )
        if self.robot_anchored:
            assert self._relative_position_robot is not None
            assert self._relative_yaw_robot is not None
            assert self._robot_anchor_pose is not None
            self._relative_position_robot = (
                self._relative_position_robot
                + self._robot_anchor_pose[:3, :3].T @ world_displacement
            ).astype(np.float32)
            self._relative_yaw_robot = self._wrap_to_pi(
                self._relative_yaw_robot + command.yaw
            )
        else:
            self.position = (self.position + world_displacement).astype(np.float32)
            self.yaw = float(self.yaw + command.yaw)
        self.pitch = float(
            np.clip(self.pitch + command.pitch, self.min_pitch, self.max_pitch)
        )
        if self.robot_anchored:
            assert self._robot_anchor_pose is not None
            return self.follow_robot_pose(self._robot_anchor_pose)
        return self.pose()

    def reset(self) -> CameraPose:
        if self.robot_anchored:
            assert self._initial_relative_position_robot is not None
            assert self._initial_relative_yaw_robot is not None
            assert self._robot_anchor_pose is not None
            self._relative_position_robot = self._initial_relative_position_robot.copy()
            self._relative_yaw_robot = float(self._initial_relative_yaw_robot)
            self.pitch = float(np.clip(self._initial_pitch, self.min_pitch, self.max_pitch))
            return self.follow_robot_pose(self._robot_anchor_pose)
        self.position = self._initial_position.copy()
        self.yaw = self._initial_yaw
        self.pitch = float(np.clip(self._initial_pitch, self.min_pitch, self.max_pitch))
        return self.pose()

    def register(self, env: Any) -> CameraPose:
        pose = self.pose()
        env.camera_manager.add_camera(
            self.camera_name,
            pose.position,
            pose.forward,
            pose.up,
            fov=self.fov_deg,
        )
        return pose

    def update_registered_camera(self, env: Any) -> CameraPose:
        pose = self.pose()
        camera = env.camera_manager.registry[self.camera_name]
        camera.pos = pose.position.copy()
        camera.forward = pose.forward.copy()
        camera.up = pose.up.copy()
        camera.fov = self.fov_deg
        return pose
