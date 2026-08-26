#!/usr/bin/env python3
"""ROS1-side gateway for the read-only Go2 WebSocket service.

This node deliberately runs with the system ROS Python (normally
``/usr/bin/python3``). It polls encoded frames from the algorithm/web process
over localhost HTTP, publishes standard ROS messages, and sends detector/map
JSON back to the web process. The Go2 link itself remains WebSocket-only.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import threading
import time
import urllib.request
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, String
import tf2_ros
from PIL import Image as PILImage


def _decode(encoded: str) -> np.ndarray:
    image = PILImage.open(io.BytesIO(base64.b64decode(encoded)))
    return np.asarray(image)


class PhysicalRosGateway:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.args.occupancy_period = max(0.0, float(self.args.occupancy_period))
        self.last_seq = -1; self.last_raw: dict[str, Any] | None = None
        self.world_frame = str(getattr(self.args, "world_frame", "tf_frame_map"))
        self.rgb_pub = rospy.Publisher("/physical_nav/rgb/image_raw", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/physical_nav/depth/image_raw", Image, queue_size=1)
        self.info_pub = rospy.Publisher("/physical_nav/camera_info", CameraInfo, queue_size=1)
        self.cloud_pub = rospy.Publisher("/physical_nav/points", PointCloud2, queue_size=1)
        self.odom_pub = rospy.Publisher("/physical_nav/odom", Odometry, queue_size=1)
        self.detection_pub = rospy.Publisher("/physical_nav/detections", String, queue_size=1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(); self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0)); self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._static_sent = False; self._lock = threading.Lock(); self._telemetry: dict[str, Any] = {}; self._last_state_poll = 0.0; self._last_occupancy_post = 0.0
        # Detections originate in the non-ROS YOLOE worker and are republished
        # below for the existing mapper.  Do not subscribe to the same topic
        # here: that would feed our own message back into the HTTP state loop.
        for topic, name in (("/physical_nav/unified_graph", "graph"), ("/physical_nav/consistency", "consistency"),):
            rospy.Subscriber(topic, String, self._json_callback(name), queue_size=1)
        rospy.Subscriber("/physical_nav/occupancy", __import__("nav_msgs.msg", fromlist=["OccupancyGrid"]).OccupancyGrid, self._occupancy_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(args.rate, 1e-3)), self._poll)

    def _post_state(self, name: str, value: Any) -> None:
        payload = json.dumps({"name": name, "value": value}, ensure_ascii=False).encode()
        request = urllib.request.Request(self.args.web_url.rstrip("/") + "/api/ros-state", data=payload, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=.3).read()
        except Exception:
            pass

    def _json_callback(self, name: str):
        def callback(msg: String) -> None:
            try:
                value = json.loads(msg.data); self._post_state(name, value)
            except Exception as exc: rospy.logwarn_throttle(5.0, "physical ROS state %s: %s", name, exc)
        return callback

    def _occupancy_callback(self, msg: Any) -> None:
        # Keep the map visible in the browser without serializing ROS metadata.
        now = time.monotonic()
        if now - self._last_occupancy_post < self.args.occupancy_period:
            return
        self._last_occupancy_post = now
        self._post_state("occupancy", {"width": msg.info.width, "height": msg.info.height, "resolution": msg.info.resolution, "origin": {"x": msg.info.origin.position.x, "y": msg.info.origin.position.y}, "data": list(msg.data)})

    def _poll(self, _event: Any) -> None:
        try:
            with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/raw-frame", timeout=.5) as response:
                raw = json.loads(response.read().decode())
            if int(raw.get("seq", -1)) <= self.last_seq or not raw.get("rgb") or not raw.get("depth"): return
            self.last_seq = int(raw["seq"]); self.last_raw = raw; self._publish(raw)
            if time.monotonic() - self._last_state_poll >= self.args.state_period:
                self._publish_state(); self._last_state_poll = time.monotonic()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical raw-frame polling: %s", exc)

    def _publish_state(self) -> None:
        try:
            with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/state", timeout=.6) as response: state = json.loads(response.read().decode())
            transform_cache: dict[str, Any] = {}
            for item in state.get("detections", []):
                if not isinstance(item, dict):
                    continue
                source_frame = str(item.get("source_frame", "") or "")
                if source_frame not in transform_cache:
                    try:
                        transform_cache[source_frame] = self.tf_buffer.lookup_transform(self.world_frame, source_frame, rospy.Time(0), rospy.Duration(0.05)) if source_frame else None
                    except Exception:
                        transform_cache[source_frame] = None
            detections = [self._map_detection(item, transform_cache.get(str(item.get("source_frame", "") or ""))) for item in state.get("detections", []) if isinstance(item, dict)]
            self.detection_pub.publish(json.dumps({"seq": state.get("frame_seq", -1), "stamp": state.get("frame_stamp", 0), "detections": detections}, ensure_ascii=False, separators=(",", ":")))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical state polling: %s", exc)

    @staticmethod
    def _point_dict(point: tuple[float, float, float]) -> dict[str, float]:
        return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}

    @staticmethod
    def _point3(value: Any) -> tuple[float, float, float] | None:
        if isinstance(value, dict):
            try:
                return tuple(float(value.get(axis, 0.0)) for axis in ("x", "y", "z"))
            except (TypeError, ValueError):
                return None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return tuple(float(item) for item in value[:3])
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _rotate_point(quaternion: Any, point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z, w = (float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w))
        px, py, pz = point
        # q * p * q^-1, expanded to avoid tf2_geometry_msgs/cv_bridge deps.
        tx = 2.0 * (y * pz - z * py); ty = 2.0 * (z * px - x * pz); tz = 2.0 * (x * py - y * px)
        return (px + w * tx + y * tz - z * ty, py + w * ty + z * tx - x * tz, pz + w * tz + x * ty - y * tx)

    def _map_detection(self, detection: dict[str, Any], transform: Any = None) -> dict[str, Any]:
        mapped = dict(detection)
        source_frame = str(detection.get("source_frame", "") or "")
        source_point = detection.get("camera_box3d_center") or detection.get("camera_position")
        if not source_frame or not isinstance(source_point, (dict, list, tuple)):
            mapped.setdefault("map_transform_status", "telemetry_fallback")
            return mapped
        if isinstance(source_point, dict):
            try: point = (float(source_point.get("x", 0.0)), float(source_point.get("y", 0.0)), float(source_point.get("z", 0.0)))
            except (TypeError, ValueError):
                mapped["map_transform_status"] = "telemetry_fallback"; return mapped
        elif len(source_point) >= 3:
            try: point = (float(source_point[0]), float(source_point[1]), float(source_point[2]))
            except (TypeError, ValueError):
                mapped["map_transform_status"] = "telemetry_fallback"; return mapped
        else:
            mapped["map_transform_status"] = "telemetry_fallback"; return mapped
        try:
            if transform is None:
                raise RuntimeError("sensor-to-map transform unavailable")
            translated = self._rotate_point(transform.transform.rotation, point)
            translated = (translated[0] + float(transform.transform.translation.x), translated[1] + float(transform.transform.translation.y), translated[2] + float(transform.transform.translation.z))
        except Exception:
            mapped["map_transform_status"] = "telemetry_fallback"
            return mapped
        mapped["world_position"] = self._point_dict(translated)
        mapped["position"] = self._point_dict(translated)
        mapped["world_box3d_center"] = self._point_dict(translated)
        mapped["aabb_center"] = [translated[0], translated[1], translated[2]]
        mapped["box3d_center"] = [translated[0], translated[1], translated[2]]
        source_size = self._point3(detection.get("camera_box3d_size"))
        if source_size is not None:
            transformed_corners = []
            for sx in (-0.5, 0.5):
                for sy in (-0.5, 0.5):
                    for sz in (-0.5, 0.5):
                        corner = (point[0] + sx * abs(source_size[0]), point[1] + sy * abs(source_size[1]), point[2] + sz * abs(source_size[2]))
                        rotated = self._rotate_point(transform.transform.rotation, corner)
                        transformed_corners.append((rotated[0] + float(transform.transform.translation.x), rotated[1] + float(transform.transform.translation.y), rotated[2] + float(transform.transform.translation.z)))
            mins = tuple(min(corner[index] for corner in transformed_corners) for index in range(3))
            maxs = tuple(max(corner[index] for corner in transformed_corners) for index in range(3))
            map_center = tuple((mins[index] + maxs[index]) / 2.0 for index in range(3))
            map_size = tuple(max(maxs[index] - mins[index], 0.01) for index in range(3))
            mapped["world_box3d_center"] = self._point_dict(map_center)
            mapped["aabb_center"] = list(map_center)
            mapped["box3d_center"] = list(map_center)
            mapped["world_box3d_size"] = self._point_dict(map_size)
            mapped["aabb_size"] = list(map_size)
            mapped["box3d_size"] = list(map_size)
        mapped["map_frame"] = self.world_frame
        mapped["map_transform_status"] = "tf"
        mapped["map_transform_source_frame"] = source_frame
        return mapped

    def _publish(self, raw: dict[str, Any]) -> None:
        rgb = _decode(raw["rgb"]); depth = _decode(raw["depth"])
        if rgb.ndim == 3 and rgb.shape[2] >= 3: rgb = np.ascontiguousarray(rgb[:, :, ::-1])
        if depth.dtype != np.uint16: depth = depth.astype(np.uint16)
        stamp = rospy.Time.from_sec(float(raw.get("stamp", time.time()))); frame = str(raw.get("camera_frame", self.args.camera_frame)); rgb_msg = _image_msg(rgb, "bgr8", stamp, frame); depth_msg = _image_msg(depth, "16UC1", stamp, frame)
        info = CameraInfo(); info.header.stamp = stamp; info.header.frame_id = frame; info.width = int(raw.get("width", rgb.shape[1])); info.height = int(raw.get("height", rgb.shape[0])); intr = raw.get("intrinsics", {}); info.K = [float(intr.get("fx", 0)), 0, float(intr.get("cx", 0)), 0, float(intr.get("fy", 0)), float(intr.get("cy", 0)), 0, 0, 1]
        distortion = [float(value) for value in (intr.get("distortion") or [])[:5]]
        info.D = distortion
        info.distortion_model = str(intr.get("distortion_model", "plumb_bob") or "plumb_bob")
        self.rgb_pub.publish(rgb_msg); self.depth_pub.publish(depth_msg); self.info_pub.publish(info)
        self._publish_cloud(depth, intr, stamp, frame); self._publish_pose(raw.get("telemetry", {}), stamp)

    def _publish_cloud(self, depth: np.ndarray, intr: dict[str, Any], stamp: Any, frame: str) -> None:
        fx, fy, cx, cy = [_to_float(intr.get(k)) for k in ("fx", "fy", "cx", "cy")]; scale = _to_float(self.last_raw.get("depth_scale", .001) if self.last_raw else .001); points = []
        if fx <= 0 or fy <= 0: return
        for v in range(0, depth.shape[0], self.args.point_stride):
            for u in range(0, depth.shape[1], self.args.point_stride):
                z = float(depth[v, u]) * scale
                if 0 < z <= self.args.max_depth_m: points.append(((u-cx)*z/fx, (v-cy)*z/fy, z))
        header = Header(); header.stamp = stamp; header.frame_id = frame; self.cloud_pub.publish(pc2.create_cloud_xyz32(header, points))

    def _publish_pose(self, telemetry: dict[str, Any], stamp: Any) -> None:
        position = telemetry.get("position", [0, 0, 0]); velocity = telemetry.get("velocity", [0, 0, 0]); yaw = _to_float(telemetry.get("yaw", telemetry.get("imu", {}).get("rpy", [0, 0, 0])[2] if telemetry.get("imu") else 0))
        odom = Odometry(); odom.header.stamp = stamp; odom.header.frame_id = "tf_frame_odom"; odom.child_frame_id = "tf_frame_base_link"; odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = [_to_float(v) for v in position[:3]]; odom.twist.twist.linear.x, odom.twist.twist.linear.y = [_to_float(v) for v in velocity[:2]]; self.odom_pub.publish(odom)
        transform = TransformStamped(); transform.header.stamp = stamp; transform.header.frame_id = "tf_frame_odom"; transform.child_frame_id = "tf_frame_base_link"; transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = [_to_float(v) for v in position[:3]]; transform.transform.rotation.z = math.sin(yaw/2); transform.transform.rotation.w = math.cos(yaw/2); self.tf_broadcaster.sendTransform(transform)
        if not self._static_sent:
            static = TransformStamped(); static.header.stamp = stamp; static.header.frame_id = self.args.camera_parent; static.child_frame_id = self.args.camera_frame; static.transform.translation.x, static.transform.translation.y, static.transform.translation.z = self.args.camera_x, self.args.camera_y, self.args.camera_z; cr, sr = math.cos(self.args.camera_roll/2), math.sin(self.args.camera_roll/2); cp, sp = math.cos(self.args.camera_pitch/2), math.sin(self.args.camera_pitch/2); cy, sy = math.cos(self.args.camera_yaw/2), math.sin(self.args.camera_yaw/2); static.transform.rotation.w = cr*cp*cy + sr*sp*sy; static.transform.rotation.x = sr*cp*cy - cr*sp*sy; static.transform.rotation.y = cr*sp*cy + sr*cp*sy; static.transform.rotation.z = cr*cp*sy - sr*sp*cy; self.static_broadcaster.sendTransform(static); self._static_sent = True


def _to_float(value: Any) -> float:
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def _image_msg(array: np.ndarray, encoding: str, stamp: Any, frame: str) -> Image:
    array = np.ascontiguousarray(array); msg = Image(); msg.header.stamp = stamp; msg.header.frame_id = frame; msg.height = int(array.shape[0]); msg.width = int(array.shape[1]); msg.encoding = encoding; msg.is_bigendian = 0; msg.step = int(array.strides[0]); msg.data = array.tobytes(); return msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--web-url", default="http://127.0.0.1:8765"); p.add_argument("--rate", type=float, default=10.); p.add_argument("--state-period", type=float, default=.2); p.add_argument("--occupancy-period", type=float, default=.5); p.add_argument("--point-stride", type=int, default=4); p.add_argument("--max-depth-m", type=float, default=8.); p.add_argument("--world-frame", default="tf_frame_map"); p.add_argument("--camera-frame", default="d435i_color_optical_frame"); p.add_argument("--camera-parent", default="tf_frame_base_link"); p.add_argument("--camera-x", type=float, default=0.); p.add_argument("--camera-y", type=float, default=0.); p.add_argument("--camera-z", type=float, default=0.); p.add_argument("--camera-roll", type=float, default=0.); p.add_argument("--camera-pitch", type=float, default=0.); p.add_argument("--camera-yaw", type=float, default=0.); args, _unknown = p.parse_known_args()
    # ROS launch appends __name/__log remappings; ignore those in the local
    # CLI parser so the gateway can also be run directly.
    rospy.init_node("physical_ros_gateway", anonymous=False)
    for name in ("web_url", "rate", "state_period", "occupancy_period", "point_stride", "max_depth_m", "world_frame", "camera_frame", "camera_parent", "camera_x", "camera_y", "camera_z", "camera_roll", "camera_pitch", "camera_yaw"):
        setattr(args, name, rospy.get_param("~" + name, getattr(args, name)))
    PhysicalRosGateway(args); rospy.spin()


if __name__ == "__main__": main()
