"""rosbridge transport that keeps the native ROS navigation topic interface intact.

This module runs in the MolmoSpaces Conda environment and requires only
`roslibpy`; ROS itself runs in the separate RoboStack Conda environment.
"""

from __future__ import annotations

import base64
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

try:
    import roslibpy
except ImportError:  # Optional: only the simulator-side transport needs it.
    roslibpy = None


@dataclass(frozen=True)
class TwistCommand:
    linear_x: float
    linear_y: float
    angular_z: float
    stamp_sec: float


def _stamp(value: float | None = None) -> dict[str, int]:
    value = time.time() if value is None else float(value)
    secs = int(value)
    return {"secs": secs, "nsecs": int((value - secs) * 1_000_000_000)}


def _header(frame_id: str, stamp_sec: float | None = None) -> dict[str, Any]:
    return {"seq": 0, "stamp": _stamp(stamp_sec), "frame_id": frame_id}


class RoslibpyBridgeClient:
    """Publish native ROS1 messages through rosbridge without importing rospy."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        *,
        odom_topic: str = "/odom",
        tf_topic: str = "/tf",
        pointcloud_topic: str = "/registered_scan",
        cmd_vel_topic: str = "/cmd_vel_stamped",
        move_base_status_topic: str = "/move_base/status",
        reset_topic: str = "/nav_system/reset",
    ) -> None:
        if roslibpy is None:
            raise ImportError(
                "RoslibpyBridgeClient requires roslibpy. Install it in the "
                "MolmoSpaces environment with `pip install roslibpy`."
            )
        self.host = host
        self.port = int(port)
        self.ros = roslibpy.Ros(host=self.host, port=self.port)
        self._connected = False
        self._lock = threading.Lock()
        self._latest_cmd_vel: TwistCommand | None = None
        self._move_base_active = False
        self._callbacks: list[Callable[[TwistCommand], None]] = []
        self._odom_pub = roslibpy.Topic(self.ros, odom_topic, "nav_msgs/Odometry")
        self._tf_pub = roslibpy.Topic(self.ros, tf_topic, "tf2_msgs/TFMessage")
        self._pointcloud_pub = roslibpy.Topic(self.ros, pointcloud_topic, "sensor_msgs/PointCloud2")
        self._reset_pub = roslibpy.Topic(self.ros, reset_topic, "std_msgs/Empty")
        self._cmd_vel_sub = roslibpy.Topic(self.ros, cmd_vel_topic, "geometry_msgs/TwistStamped")
        self._move_base_status_sub = roslibpy.Topic(
            self.ros, move_base_status_topic, "actionlib_msgs/GoalStatusArray"
        )

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ros.is_connected

    @property
    def latest_cmd_vel(self) -> TwistCommand | None:
        with self._lock:
            return self._latest_cmd_vel

    @property
    def move_base_active(self) -> bool:
        with self._lock:
            return self._move_base_active

    def add_command_callback(self, callback: Callable[[TwistCommand], None]) -> None:
        self._callbacks.append(callback)

    def connect(self, timeout_s: float = 10.0) -> None:
        self.ros.run()
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while not self.ros.is_connected and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self.ros.is_connected:
            self.ros.terminate()
            raise TimeoutError(f"rosbridge unavailable at ws://{self.host}:{self.port}")
        self._cmd_vel_sub.subscribe(self._on_cmd_vel)
        self._move_base_status_sub.subscribe(self._on_move_base_status)
        self._connected = True

    def close(self) -> None:
        if self._connected:
            self._cmd_vel_sub.unsubscribe()
            self._move_base_status_sub.unsubscribe()
        self._connected = False
        self.ros.terminate()

    def publish_odom(
        self,
        *,
        position_xyz: Iterable[float],
        orientation_xyzw: Iterable[float],
        frame_id: str = "tf_frame_odom",
        child_frame_id: str = "tf_frame_base_link",
        stamp_sec: float | None = None,
    ) -> None:
        px, py, pz = map(float, position_xyz)
        qx, qy, qz, qw = map(float, orientation_xyzw)
        self._odom_pub.publish(roslibpy.Message({
            "header": _header(frame_id, stamp_sec), "child_frame_id": child_frame_id,
            "pose": {"pose": {"position": {"x": px, "y": py, "z": pz},
                               "orientation": {"x": qx, "y": qy, "z": qz, "w": qw}}},
            "twist": {"twist": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                                 "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}},
        }))

    def publish_transform(
        self,
        *,
        parent_frame: str,
        child_frame: str,
        translation_xyz: Iterable[float],
        rotation_xyzw: Iterable[float],
        stamp_sec: float | None = None,
    ) -> None:
        tx, ty, tz = map(float, translation_xyz)
        qx, qy, qz, qw = map(float, rotation_xyzw)
        self._tf_pub.publish(roslibpy.Message({"transforms": [{
            "header": _header(parent_frame, stamp_sec), "child_frame_id": child_frame,
            "transform": {"translation": {"x": tx, "y": ty, "z": tz},
                          "rotation": {"x": qx, "y": qy, "z": qz, "w": qw}},
        }]}))

    def publish_xyz32_pointcloud(
        self,
        points_xyz: Iterable[Iterable[float]],
        *,
        frame_id: str = "tf_frame_lidar",
        stamp_sec: float | None = None,
    ) -> None:
        packed = bytearray()
        count = 0
        for x, y, z in points_xyz:
            packed.extend(struct.pack("<fff", float(x), float(y), float(z)))
            count += 1
        self._pointcloud_pub.publish(roslibpy.Message({
            "header": _header(frame_id, stamp_sec), "height": 1, "width": count,
            "fields": [{"name": "x", "offset": 0, "datatype": 7, "count": 1},
                       {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                       {"name": "z", "offset": 8, "datatype": 7, "count": 1}],
            "is_bigendian": False, "point_step": 12, "row_step": 12 * count,
            "data": base64.b64encode(bytes(packed)).decode("ascii"), "is_dense": True,
        }))

    def publish_reset(self) -> None:
        self._reset_pub.publish(roslibpy.Message({}))

    def _on_cmd_vel(self, message: dict[str, Any]) -> None:
        twist = message.get("twist", {})
        linear, angular = twist.get("linear", {}), twist.get("angular", {})
        stamp = message.get("header", {}).get("stamp", {})
        command = TwistCommand(
            float(linear.get("x", 0.0)), float(linear.get("y", 0.0)),
            float(angular.get("z", 0.0)),
            float(stamp.get("secs", 0)) + float(stamp.get("nsecs", 0)) / 1e9,
        )
        with self._lock:
            self._latest_cmd_vel = command
        for callback in tuple(self._callbacks):
            callback(command)

    def _on_move_base_status(self, message: dict[str, Any]) -> None:
        active = any(int(status.get("status", -1)) == 1 for status in message.get("status_list", []))
        with self._lock:
            self._move_base_active = active
