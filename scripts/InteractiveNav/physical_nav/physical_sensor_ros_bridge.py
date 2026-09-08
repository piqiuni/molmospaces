#!/usr/bin/env python3
"""Direct Go2 RGB-D WebSocket to ROS bridge.

The sensor socket and ROS publishers are independent from the dashboard. The
dashboard may receive a latest-only mirror, but it is never on the sensor path.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
import io
import json
import logging
import math
import sys
import threading
import time
import urllib.request
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
import tf2_ros
from PIL import Image as PILImage

from physical_protocol import decode_wire_packet


def _patch_roslogging_findcaller_for_py311() -> None:
    """Avoid rosgraph's Python 3.11 ``findCaller`` recursion.

    Under the Conda 3.11 interpreter ``rospy.init_node`` otherwise busy-loops
    inside ``rosgraph.roslogging.RospyLogger.findCaller`` before registration,
    consuming a core and never publishing. Mirrors the gateway workaround so
    this bridge can run with the Conda interpreter (required for ``websockets``).
    """
    if sys.version_info < (3, 11):
        return
    try:
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_physical_sensor_safe", False):
        return

    def _safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 4:
            result = result[:3]
        return result

    _safe_find_caller._physical_sensor_safe = True
    roslogging.RospyLogger.findCaller = _safe_find_caller


def _decode(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("data", "")
    return np.asarray(PILImage.open(io.BytesIO(base64.b64decode(value))))


def _quat_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


class SensorRosBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.last_stamp = -float("inf")
        self.last_seq = -1
        self.telemetry: dict[str, Any] = {}
        self._telemetry_history: deque[tuple[float, dict[str, Any]]] = deque(maxlen=64)
        self._telemetry_lock = threading.Lock()
        self.static_frames: set[str] = set()
        self.rgb_pub = rospy.Publisher("/physical_nav/rgb/image_raw", Image, queue_size=1, tcp_nodelay=True)
        self.depth_pub = rospy.Publisher("/physical_nav/depth/image_raw", Image, queue_size=1, tcp_nodelay=True)
        self.info_pub = rospy.Publisher("/physical_nav/camera_info", CameraInfo, queue_size=1, latch=True)
        self.depth_info_pub = rospy.Publisher("/physical_nav/depth_camera_info", CameraInfo, queue_size=1, latch=True)
        self.odom_pub = rospy.Publisher("/physical_nav/odom", Odometry, queue_size=1, tcp_nodelay=True)
        self.cloud_pub = rospy.Publisher("/physical_nav/points", PointCloud2, queue_size=1, tcp_nodelay=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._packet_lock = threading.Lock()
        self._packet_event = threading.Event()
        self._latest_packet: dict[str, Any] | None = None
        threading.Thread(target=self._publish_loop, name="sensor-ros-publisher", daemon=True).start()
        self._mirror_enabled = bool(args.web_mirror_enabled and args.web_url)
        self._mirror_lock = threading.Lock()
        self._mirror_event = threading.Event()
        self._latest_mirror_packet: dict[str, Any] | None = None
        # The dashboard used to receive telemetry over the legacy 12334
        # WebSocket.  The direct transport delivers it separately, so mirror a
        # latest-only copy for the same dashboard/record telemetry state.
        self._latest_mirror_telemetry: dict[str, Any] | None = None
        if self._mirror_enabled:
            threading.Thread(target=self._mirror_loop, name="sensor-web-mirror", daemon=True).start()
        rospy.Timer(rospy.Duration(0.05), self.refresh_tf)

    @staticmethod
    def _image_msg(array: np.ndarray, encoding: str, stamp: rospy.Time, frame: str) -> Image:
        array = np.ascontiguousarray(array)
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, frame
        msg.height, msg.width = array.shape[:2]
        msg.encoding, msg.is_bigendian = encoding, 0
        msg.step, msg.data = int(array.strides[0]), array.tobytes(order="C")
        return msg

    def enqueue_sensor_packet(self, packet: dict[str, Any]) -> None:
        stamp = float(packet.get("stamp", 0.0) or 0.0)
        with self._packet_lock:
            pending = float(self._latest_packet.get("stamp", -float("inf")) if self._latest_packet else -float("inf"))
            if stamp <= max(self.last_stamp, pending):
                return
            self._latest_packet = packet
            self._packet_event.set()
        if self._mirror_enabled:
            with self._mirror_lock:
                self._latest_mirror_packet = packet
                self._mirror_event.set()

    def _publish_loop(self) -> None:
        while not rospy.is_shutdown():
            self._packet_event.wait(0.5)
            with self._packet_lock:
                packet = self._latest_packet
                self._latest_packet = None
                self._packet_event.clear()
            if packet is None:
                continue
            try:
                self.publish(packet)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "sensor ROS publish: %s", exc)

    def _mirror_loop(self) -> None:
        base_url = self.args.web_url.rstrip("/")
        while not rospy.is_shutdown():
            self._mirror_event.wait(0.5)
            with self._mirror_lock:
                packet = self._latest_mirror_packet
                self._latest_mirror_packet = None
                telemetry_snapshot = self._latest_mirror_telemetry
                self._latest_mirror_telemetry = None
                self._mirror_event.clear()
            if telemetry_snapshot is not None:
                try:
                    body = json.dumps(telemetry_snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    request = urllib.request.Request(base_url + "/api/telemetry", data=body, headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(request, timeout=0.5):
                        pass
                except Exception as exc:
                    rospy.logwarn_throttle(10.0, "optional dashboard telemetry mirror: %s", exc)
            if packet is None:
                continue
            try:
                body = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                request = urllib.request.Request(base_url + "/api/raw-frame", data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=0.5):
                    pass
            except Exception as exc:
                rospy.logwarn_throttle(10.0, "optional dashboard sensor mirror: %s", exc)

    def publish(self, packet: dict[str, Any]) -> None:
        stamp_value = float(packet.get("stamp", time.time()) or time.time())
        seq = int(packet.get("seq", -1))
        if stamp_value <= self.last_stamp or (stamp_value <= 0 and seq <= self.last_seq):
            return
        self.last_stamp, self.last_seq = stamp_value, seq
        stamp = rospy.Time.from_sec(stamp_value)
        rgb = _decode(packet["rgb"])
        depth = _decode(packet["depth"]).astype(np.uint16, copy=False)
        if rgb.ndim == 3 and rgb.shape[2] >= 3:
            rgb = np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1])
        rgb_frame = str(packet.get("camera_frame") or self.args.camera_frame)
        depth_frame = str(packet.get("depth_frame") or rgb_frame)
        self.rgb_pub.publish(self._image_msg(rgb, "bgr8", stamp, rgb_frame))
        self.depth_pub.publish(self._image_msg(depth, "16UC1", stamp, depth_frame))
        rgb_intrinsics = packet.get("rgb_intrinsics") or packet.get("intrinsics", {})
        depth_intrinsics = packet.get("depth_intrinsics") or rgb_intrinsics
        self.info_pub.publish(self.camera_info(rgb_intrinsics, stamp, rgb_frame))
        self.depth_info_pub.publish(self.camera_info(depth_intrinsics, stamp, depth_frame))
        self.publish_cloud(depth, depth_intrinsics, stamp, depth_frame, float(packet.get("depth_scale", 0.001)))
        if isinstance(packet.get("telemetry"), dict):
            self.update_telemetry(packet["telemetry"], stamp_value)
        matched_telemetry = self.telemetry_at(stamp_value)
        if matched_telemetry:
            # Date odometry with the camera capture stamp. This lets the
            # mapper request the historical pose matching this cloud instead
            # of pairing a delayed frame with the current robot pose.
            self.publish_pose(matched_telemetry, stamp, depth_frame)

    def update_telemetry(self, telemetry: dict[str, Any], stamp: float) -> None:
        snapshot = dict(telemetry)
        with self._telemetry_lock:
            self.telemetry = snapshot
            self._telemetry_history.append((float(stamp), snapshot))
        if self._mirror_enabled:
            with self._mirror_lock:
                self._latest_mirror_telemetry = snapshot
                self._mirror_event.set()

    def telemetry_at(self, stamp: float) -> dict[str, Any]:
        with self._telemetry_lock:
            if not self._telemetry_history:
                return dict(self.telemetry)
            _, snapshot = min(self._telemetry_history, key=lambda item: abs(item[0] - stamp))
            return dict(snapshot)

    @staticmethod
    def camera_info(intrinsics: dict[str, Any], stamp: rospy.Time, frame: str) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp, msg.header.frame_id = stamp, frame
        msg.width, msg.height = int(intrinsics.get("width", 0)), int(intrinsics.get("height", 0))
        msg.K = [float(intrinsics.get("fx", 0)), 0.0, float(intrinsics.get("cx", 0)), 0.0, float(intrinsics.get("fy", 0)), float(intrinsics.get("cy", 0)), 0.0, 0.0, 1.0]
        msg.D = [float(value) for value in (intrinsics.get("distortion") or [])[:5]]
        msg.distortion_model = str(intrinsics.get("distortion_model", "plumb_bob") or "plumb_bob")
        return msg

    def publish_cloud(self, depth: np.ndarray, intrinsics: dict[str, Any], stamp: rospy.Time, frame: str, scale: float) -> None:
        fx, fy, cx, cy = [float(intrinsics.get(key, 0)) for key in ("fx", "fy", "cx", "cy")]
        if fx <= 0.0 or fy <= 0.0:
            return
        stride = max(1, int(self.args.point_stride))
        sampled = np.asarray(depth[::stride, ::stride], dtype=np.float32) * scale
        valid = sampled > 0.0
        obstacle_depth = max(0.0, float(self.args.max_depth_m))
        no_return_depth = max(obstacle_depth + 1e-3, float(self.args.no_return_depth_m))
        projected_z = np.where(valid & (sampled > obstacle_depth), no_return_depth, sampled)
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            return
        z = projected_z[rows, cols]
        x = (cols.astype(np.float32) * stride - cx) * z / fx
        y = (rows.astype(np.float32) * stride - cy) * z / fy
        points = np.empty((rows.size, 3), dtype=np.float32)
        points[:, 0], points[:, 1], points[:, 2] = x, y, z
        header = Header(stamp=stamp, frame_id=frame)
        msg = PointCloud2(
            header=header, height=1, width=int(points.shape[0]),
            fields=[PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1), PointField("z", 8, PointField.FLOAT32, 1)],
            is_bigendian=False, point_step=12, row_step=int(points.shape[0] * 12),
            data=points.tobytes(order="C"), is_dense=False,
        )
        self.cloud_pub.publish(msg)

    def publish_pose(self, telemetry: dict[str, Any], stamp: rospy.Time, depth_frame: str | None = None) -> None:
        position = list(telemetry.get("position") or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0]
        velocity = list(telemetry.get("velocity") or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0]
        imu = telemetry.get("imu")
        raw_quaternion = imu.get("quaternion") or imu.get("orientation") if isinstance(imu, dict) else None
        if isinstance(raw_quaternion, (list, tuple)) and len(raw_quaternion) >= 4:
            x, y, z, w = float(raw_quaternion[1]), float(raw_quaternion[2]), float(raw_quaternion[3]), float(raw_quaternion[0])
        else:
            yaw = float(telemetry.get("yaw", 0.0) or 0.0)
            x, y, z, w = 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "tf_frame_odom"
        odom.child_frame_id = "tf_frame_base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position[:3])
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = x, y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = z, w
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = map(float, velocity[:2])
        self.odom_pub.publish(odom)
        transform = TransformStamped(header=odom.header, child_frame_id=odom.child_frame_id)
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, position[:3])
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)
        for frame in {str(self.args.camera_frame), str(depth_frame or self.args.camera_frame)}:
            if frame in self.static_frames:
                continue
            static = TransformStamped()
            static.header.stamp, static.header.frame_id, static.child_frame_id = stamp, self.args.camera_parent, frame
            static.transform.translation.x, static.transform.translation.y, static.transform.translation.z = float(self.args.camera_x), float(self.args.camera_y), float(self.args.camera_z)
            qx, qy, qz, qw = _quat_multiply(_quat_rpy(self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw), (0.5, -0.5, 0.5, -0.5))
            static.transform.rotation.x, static.transform.rotation.y, static.transform.rotation.z, static.transform.rotation.w = qx, qy, qz, qw
            self.static_broadcaster.sendTransform(static)
            self.static_frames.add(frame)

    def refresh_tf(self, _event: Any) -> None:
        if self.telemetry:
            self.publish_pose(self.telemetry, rospy.Time.now())


async def run(args: argparse.Namespace) -> None:
    bridge = SensorRosBridge(args)
    import websockets

    async def handler(websocket: Any, *_path: Any) -> None:
        async for raw in websocket:
            try:
                packet = decode_wire_packet(raw)
                if packet.get("type") == "sensor_frame":
                    bridge.enqueue_sensor_packet(packet)
                elif packet.get("type") == "telemetry" and isinstance(packet.get("telemetry"), dict):
                    bridge.update_telemetry(packet["telemetry"], float(packet.get("stamp", time.time()) or time.time()))
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "sensor packet: %s", exc)

    async with websockets.serve(handler, args.host, args.port, max_size=args.max_message_mb * 1024 * 1024, ping_interval=None):
        rospy.loginfo("direct sensor ROS WebSocket: ws://%s:%d", args.host, args.port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12335)
    parser.add_argument("--url", default="")
    parser.add_argument("--point-stride", type=int, default=6)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--no-return-depth-m", type=float, default=8.05)
    parser.add_argument("--camera-frame", default="d435i_color_optical_frame")
    parser.add_argument("--camera-parent", default="tf_frame_base_link")
    parser.add_argument("--camera-x", type=float, default=0.03)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=0.98)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--camera-pitch", type=float, default=0.1396263)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--max-message-mb", type=int, default=16)
    parser.add_argument("--web-url", default="http://127.0.0.1:8765")
    parser.add_argument("--web-mirror-enabled", action="store_true")
    # roslaunch appends ROS remapping and log arguments after the executable.
    # They are consumed by rospy, not this argparse interface.
    args, _unknown_ros_args = parser.parse_known_args()
    _patch_roslogging_findcaller_for_py311()
    rospy.init_node("physical_sensor_ros_bridge")
    for name in ("host", "port", "url", "point_stride", "max_depth_m", "no_return_depth_m", "camera_frame", "camera_parent", "camera_x", "camera_y", "camera_z", "camera_roll", "camera_pitch", "camera_yaw", "max_message_mb", "web_url", "web_mirror_enabled"):
        setattr(args, name, rospy.get_param("~" + name, getattr(args, name)))
    if args.url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(str(args.url))
            if parsed.port:
                args.port = parsed.port
        except ValueError:
            rospy.logwarn("invalid sensor WebSocket URL: %s", args.url)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
