#!/usr/bin/env python3
"""HTTP bridge between Habitat public observations and the full ROS graph stack."""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path as FilePath
import threading
import time
import traceback
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import rospy
from actionlib_msgs.msg import GoalID, GoalStatusArray
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Empty, String
import tf2_ros


def _decode_rgb(payload: dict[str, Any]) -> np.ndarray:
    value = payload.get("image_b64")
    if not isinstance(value, str) or not value:
        raise ValueError("image_b64 is required")
    raw = base64.b64decode(value, validate=True)
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("image_b64 is not decodable")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _decode_array(value: Any, *, dtype: np.dtype, name: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an encoded array object")
    try:
        shape = tuple(int(item) for item in value["shape"])
        raw = base64.b64decode(str(value["data_b64"]), validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    try:
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    except ValueError as exc:
        raise ValueError(f"{name} byte count does not match shape") from exc


def _image_message(
    array: np.ndarray,
    encoding: str,
    stamp: rospy.Time,
    frame_id: str,
    *,
    sequence: int = 0,
) -> Image:
    data = np.ascontiguousarray(array)
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.header.seq = max(0, int(sequence))
    message.height = int(data.shape[0])
    message.width = int(data.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(data.strides[0])
    message.data = data.tobytes(order="C")
    return message


class _State:
    def __init__(self, args: argparse.Namespace) -> None:
        rospy.init_node("habitat_v2_full_stack_bridge", anonymous=True, disable_signals=True)
        self.args = args
        dump_dir = os.environ.get("HABITAT_BRIDGE_DUMP_RGB_DIR", "")
        self._dump_rgb_dir = FilePath(dump_dir) if dump_dir else None
        self._direct_detector_url = os.environ.get("HABITAT_DIRECT_DETECTOR_URL", "").rstrip("/")
        if self._dump_rgb_dir is not None:
            self._dump_rgb_dir.mkdir(parents=True, exist_ok=True)
        self._request_lock = threading.Lock()
        self._condition = threading.Condition()
        self._latest_graph: dict[str, Any] = {}
        self._latest_candidates: dict[str, Any] = {}
        self._latest_detections: dict[str, Any] = {}
        self._latest_detection_stamp = 0.0
        self._candidate_sequence = -1
        self._last_target_context: tuple[str, tuple[str, ...], bool] | None = None
        self._candidate_context_waits = 0
        self._candidate_context_skips = 0
        self._requests = 0
        self._detector_ticks = 0
        self._detector_step_interval = max(1, int(args.detector_step_interval))
        self._failures = 0
        self._selection_requests = 0
        self._selection_failures = 0
        self._pending_step_sync: tuple[rospy.Time, dict[str, Any], dict[str, Any]] | None = None
        self._latest_cmd_vel = {"linear_x": 0.0, "angular_z": 0.0}
        self._latest_cmd_vel_monotonic = 0.0
        self._latest_measured_twist = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        self._last_measurement_pose_ros: tuple[float, float, float] | None = None
        self._latest_move_base_status = 0
        self._latest_global_plan: list[list[float]] = []
        self._latest_local_plan: list[list[float]] = []
        self._last_goal_key: tuple[str, float, float, float] | None = None
        self._latest_pose: tuple[np.ndarray, float] | None = None

        self.rgb_pub = rospy.Publisher(args.rgb_topic, Image, queue_size=1)
        self.depth_pub = rospy.Publisher(args.depth_topic, Image, queue_size=1)
        self.info_pub = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=1)
        self.mapping_cloud_pub = rospy.Publisher(args.mapping_scan_topic, PointCloud2, queue_size=1)
        self.local_cloud_pub = rospy.Publisher(args.registered_scan_topic, PointCloud2, queue_size=1)
        self.odom_pub = rospy.Publisher(args.odom_topic, Odometry, queue_size=1)
        self.map_pub = (
            None
            if args.original_ros_navigation
            else rospy.Publisher(args.occupancy_topic, OccupancyGrid, queue_size=1, latch=True)
        )
        self.raw_map_pub = (
            None
            if args.original_ros_navigation
            else rospy.Publisher(args.raw_occupancy_topic, OccupancyGrid, queue_size=1, latch=True)
        )
        self.global_plan_pub = (
            None
            if args.original_ros_navigation
            else rospy.Publisher(args.global_plan_topic, Path, queue_size=1, latch=True)
        )
        self.local_plan_pub = (
            None
            if args.original_ros_navigation
            else rospy.Publisher(args.local_plan_topic, Path, queue_size=1, latch=True)
        )
        self.diagnostic_panel_pub = rospy.Publisher(args.diagnostic_panel_topic, Image, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher(args.target_topic, String, queue_size=1, latch=True)
        self.selection_pub = rospy.Publisher(args.selected_behavior_topic, String, queue_size=8)
        self.simple_goal_pub = rospy.Publisher(args.simple_goal_topic, PoseStamped, queue_size=2)
        self.mapping_reset_pub = rospy.Publisher(args.mapping_reset_topic, Empty, queue_size=1)
        self.move_base_cancel_pub = rospy.Publisher(args.move_base_cancel_topic, GoalID, queue_size=1)
        self.step_sync_pub = rospy.Publisher(args.step_sync_topic, String, queue_size=8)
        # Direct detector fallback results must enter the same ROS semantic-map
        # stream as the timer-driven detector.  Without this relay the policy
        # could track a target while the original graph/recorder had no object
        # node to draw.
        self.detection_relay_pub = rospy.Publisher(args.detections_topic, String, queue_size=4)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(args.graph_topic, String, self._graph_callback, queue_size=1)
        rospy.Subscriber(args.candidates_topic, String, self._candidate_callback, queue_size=1)
        rospy.Subscriber(args.detections_topic, String, self._detection_callback, queue_size=1)
        rospy.Subscriber(args.cmd_vel_topic, Twist, self._cmd_vel_callback, queue_size=1)
        rospy.Subscriber(args.move_base_status_topic, GoalStatusArray, self._status_callback, queue_size=1)
        rospy.Subscriber(args.move_base_global_plan_topic, Path, self._global_plan_callback, queue_size=1)
        rospy.Subscriber(args.move_base_local_plan_topic, Path, self._local_plan_callback, queue_size=1)
        # Habitat advances synchronously and detector/M2 requests can take
        # seconds.  ROS costmaps still require a fresh transform at their own
        # controller rate, so republish the last measured pose without
        # extrapolating motion between simulator steps.
        self._pose_timer = rospy.Timer(rospy.Duration(0.05), self._pose_heartbeat)

    @staticmethod
    def _parse(message: String) -> dict[str, Any]:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _graph_callback(self, message: String) -> None:
        with self._condition:
            self._latest_graph = self._parse(message)
            self._condition.notify_all()

    def _candidate_callback(self, message: String) -> None:
        payload = self._parse(message)
        try:
            sequence = int(payload.get("sequence", -1))
        except (TypeError, ValueError):
            sequence = -1
        with self._condition:
            self._latest_candidates = payload
            self._candidate_sequence = max(self._candidate_sequence, sequence)
            self._condition.notify_all()

    def _detection_callback(self, message: String) -> None:
        payload = self._parse(message)
        stamp = float(payload.get("stamp_sec", 0.0) or 0.0) + 1e-9 * float(
            payload.get("stamp_nsec", 0.0) or 0.0
        )
        with self._condition:
            self._latest_detections = payload
            self._latest_detection_stamp = stamp
            self._condition.notify_all()

    def _cmd_vel_callback(self, message: Twist) -> None:
        with self._condition:
            self._latest_cmd_vel = {
                "linear_x": float(message.linear.x),
                "angular_z": float(message.angular.z),
            }
            self._latest_cmd_vel_monotonic = time.monotonic()
            self._condition.notify_all()

    def _status_callback(self, message: GoalStatusArray) -> None:
        if not message.status_list:
            return
        with self._condition:
            self._latest_move_base_status = int(message.status_list[-1].status)
            self._condition.notify_all()

    @staticmethod
    def _path_rows(message: Path) -> list[list[float]]:
        rows: list[list[float]] = []
        for pose in message.poses:
            q = pose.pose.orientation
            yaw = math.atan2(
                2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
                1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
            )
            rows.append([float(pose.pose.position.x), float(pose.pose.position.y), yaw])
        return rows

    def _global_plan_callback(self, message: Path) -> None:
        with self._condition:
            self._latest_global_plan = self._path_rows(message)
            self._condition.notify_all()

    def _local_plan_callback(self, message: Path) -> None:
        with self._condition:
            self._latest_local_plan = self._path_rows(message)
            self._condition.notify_all()

    @staticmethod
    def _quaternion(yaw: float) -> tuple[float, float, float, float]:
        return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))

    def _pose_heartbeat(self, _event: rospy.TimerEvent) -> None:
        pose = self._latest_pose
        if pose is None:
            return
        self._publish_pose(rospy.Time.now(), pose[0], pose[1], remember=False)

    def _publish_pose(
        self,
        stamp: rospy.Time,
        gps_xy: np.ndarray,
        compass: float,
        *,
        remember: bool = True,
    ) -> None:
        # Habitat GPS uses [x, y] with forward=[cos(h), -sin(h)].  Reflect its
        # second axis so ROS receives the standard forward=[cos(yaw), sin(yaw)].
        ros_x = float(gps_xy[0])
        ros_y = -float(gps_xy[1])
        yaw = float(compass)
        if remember:
            self._latest_pose = (np.asarray(gps_xy, dtype=np.float32).copy(), yaw)
            previous = self._last_measurement_pose_ros
            if previous is None:
                measured = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
            else:
                dt = max(float(self.args.sim_dt_s), 1e-6)
                delta_yaw = math.atan2(
                    math.sin(yaw - previous[2]),
                    math.cos(yaw - previous[2]),
                )
                world_vx = (ros_x - previous[0]) / dt
                world_vy = (ros_y - previous[1]) / dt
                # nav_msgs/Odometry twist is expressed in child_frame_id.  Use
                # the midpoint heading for an arc executed during one Habitat
                # control step, rather than reporting map-frame displacement.
                midpoint_yaw = previous[2] + 0.5 * delta_yaw
                measured = {
                    "linear_x": math.cos(midpoint_yaw) * world_vx
                    + math.sin(midpoint_yaw) * world_vy,
                    "linear_y": -math.sin(midpoint_yaw) * world_vx
                    + math.cos(midpoint_yaw) * world_vy,
                    "angular_z": delta_yaw / dt,
                }
                for key, value in measured.items():
                    if abs(value) < 1e-5:
                        measured[key] = 0.0
            self._latest_measured_twist = measured
            self._last_measurement_pose_ros = (ros_x, ros_y, yaw)
        qx, qy, qz, qw = self._quaternion(yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.args.odom_frame
        odom.child_frame_id = self.args.base_frame
        odom.pose.pose.position.x = ros_x
        odom.pose.pose.position.y = ros_y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self._latest_measured_twist["linear_x"])
        odom.twist.twist.linear.y = float(self._latest_measured_twist["linear_y"])
        odom.twist.twist.angular.z = float(self._latest_measured_twist["angular_z"])
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.args.odom_frame
        transform.child_frame_id = self.args.base_frame
        transform.transform.translation.x = ros_x
        transform.transform.translation.y = ros_y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_pub.sendTransform(transform)

        lidar = TransformStamped()
        lidar.header.stamp = stamp
        lidar.header.frame_id = self.args.base_frame
        lidar.child_frame_id = self.args.lidar_frame
        lidar.transform.translation.z = float(self.args.camera_height_m)
        lidar.transform.rotation.w = 1.0
        self.tf_pub.sendTransform(lidar)

        optical = TransformStamped()
        optical.header.stamp = stamp
        optical.header.frame_id = self.args.base_frame
        optical.child_frame_id = self.args.camera_frame
        optical.transform.translation.z = float(self.args.camera_height_m)
        # REP-103 optical frame: z forward, x right, y down.
        optical.transform.rotation.x = -0.5
        optical.transform.rotation.y = 0.5
        optical.transform.rotation.z = -0.5
        optical.transform.rotation.w = 0.5
        self.tf_pub.sendTransform(optical)

    def _publish_pointcloud(
        self,
        stamp: rospy.Time,
        depth: np.ndarray,
        camera_k: list[float],
    ) -> None:
        stride = max(1, int(self.args.pointcloud_stride))
        z = np.asarray(depth[::stride, ::stride], dtype=np.float32)
        rows = np.arange(0, depth.shape[0], stride, dtype=np.float32)[:, None]
        cols = np.arange(0, depth.shape[1], stride, dtype=np.float32)[None, :]
        fx, fy = max(float(camera_k[0]), 1e-6), max(float(camera_k[4]), 1e-6)
        cx, cy = float(camera_k[2]), float(camera_k[5])
        # Convert camera optical coordinates to ROS base axes at the lidar origin.
        x = z
        y = -(cols - cx) * z / fx
        zz = -(rows - cy) * z / fy
        valid = np.isfinite(z) & (z > 0.05) & (z <= float(self.args.pointcloud_max_depth_m))
        xyz = np.stack([x, y, zz], axis=-1).astype(np.float32)
        xyz[~valid] = np.nan
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.args.lidar_frame
        cloud.height = int(xyz.shape[0])
        cloud.width = int(xyz.shape[1])
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = int(cloud.point_step * cloud.width)
        cloud.is_dense = False
        cloud.data = np.ascontiguousarray(xyz).tobytes(order="C")
        self.mapping_cloud_pub.publish(cloud)
        self.local_cloud_pub.publish(cloud)

    def _publish_map(self, stamp: rospy.Time, payload: dict[str, Any], *, raw: bool = False) -> None:
        key = "raw_occupancy" if raw else "occupancy"
        grid = _decode_array(payload.get(key), dtype=np.int8, name=key)
        if grid.ndim != 2:
            raise ValueError("occupancy must be a 2-D int8 array")
        resolution = float(payload.get("map_resolution_m", 0.0))
        origin = payload.get("map_origin_xy")
        if resolution <= 0.0 or not isinstance(origin, list) or len(origin) != 2:
            raise ValueError("map resolution/origin are invalid")
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = self.args.map_frame
        message.info.resolution = resolution
        message.info.width = int(grid.shape[1])
        message.info.height = int(grid.shape[0])
        message.info.origin.position.x = float(origin[0])
        message.info.origin.position.y = float(origin[1])
        message.info.origin.orientation.w = 1.0
        message.data = grid.reshape(-1).astype(np.int8).tolist()
        publisher = self.raw_map_pub if raw else self.map_pub
        if publisher is not None:
            publisher.publish(message)

    def _publish_path(self, stamp: rospy.Time, rows: Any, publisher: rospy.Publisher) -> None:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.args.map_frame
        if not isinstance(rows, list):
            rows = []
        for raw in rows:
            if not isinstance(raw, list) or len(raw) < 2:
                continue
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(raw[0])
            pose.pose.position.y = float(raw[1])
            yaw = float(raw[2]) if len(raw) > 2 else 0.0
            qx, qy, qz, qw = self._quaternion(yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)
        publisher.publish(path)

    def _publish_sensor_frame(self, stamp: rospy.Time, payload: dict[str, Any]) -> None:
        rgb = _decode_rgb(payload)
        depth = _decode_array(payload.get("depth"), dtype=np.float32, name="depth")
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth dimensions do not match RGB")
        camera = payload.get("camera_info")
        if not isinstance(camera, dict) or len(camera.get("K") or []) != 9:
            raise ValueError("camera_info with nine K values is required")
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.args.camera_frame
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        sequence = max(0, int(context.get("step", 0) or 0))
        if self._dump_rgb_dir is not None and sequence > 0:
            cv2.imwrite(
                str(self._dump_rgb_dir / f"rgb_step_{sequence:06d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
        info.header.seq = sequence
        info.width = int(rgb.shape[1])
        info.height = int(rgb.shape[0])
        info.K = [float(value) for value in camera["K"]]
        self.info_pub.publish(info)
        self.depth_pub.publish(
            _image_message(depth.astype(np.float32), "32FC1", stamp, self.args.camera_frame, sequence=sequence)
        )
        self.rgb_pub.publish(
            _image_message(rgb.astype(np.uint8), "rgb8", stamp, self.args.camera_frame, sequence=sequence)
        )
        self._publish_pointcloud(stamp, depth, info.K)

    def _publish_step_sync(
        self,
        stamp: rospy.Time,
        payload: dict[str, Any],
        detections: dict[str, Any],
    ) -> None:
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        sequence = max(0, int(context.get("step", 0) or 0))
        self.step_sync_pub.publish(
            String(
                data=json.dumps(
                    {
                        "step_index": sequence,
                        "stamp_sec": float(stamp.to_sec()),
                        # Carry the exact detector snapshot in the same message
                        # that freezes the recorder frame.  Cross-topic callback
                        # ordering is otherwise nondeterministic and produced
                        # videos whose RGB frame preceded its detection boxes.
                        "external_detections": detections,
                    },
                    separators=(",", ":"),
                )
            )
        )

    def _publish_target_context(self, target: str, labels: list[str], *, visible: bool) -> None:
        self.target_pub.publish(
            String(
                data=json.dumps(
                    {
                        "enabled": True,
                        "target_name": target,
                        "object_labels": labels,
                        "visible": bool(visible),
                    },
                    separators=(",", ":"),
                )
            )
        )

    def _lift_direct_detections(
        self,
        detections: dict[str, Any],
        payload: dict[str, Any],
        gps: np.ndarray,
        compass: float,
        stamp: rospy.Time,
    ) -> dict[str, Any]:
        """Attach public RGB-D world geometry to direct 2-D fallback boxes."""

        depth = _decode_array(payload.get("depth"), dtype=np.float32, name="depth")
        if depth.ndim == 3:
            depth = depth[..., 0]
        camera = payload.get("camera_info") if isinstance(payload.get("camera_info"), dict) else {}
        intrinsics = [float(value) for value in camera.get("K", [])]
        if len(intrinsics) != 9:
            return detections
        fx, fy = max(intrinsics[0], 1e-6), max(intrinsics[4], 1e-6)
        cx, cy = intrinsics[2], intrinsics[5]
        height, width = depth.shape
        cos_h, sin_h = math.cos(compass), math.sin(compass)
        lifted: list[dict[str, Any]] = []
        for source in detections.get("detections") or []:
            if not isinstance(source, dict):
                continue
            row = dict(source)
            bbox = list(row.get("bbox") or row.get("bbox_xyxy") or [])
            if len(bbox) == 4:
                values = [float(value) for value in bbox]
                if max(abs(value) for value in values) <= 1.5:
                    x0, y0, x1, y1 = (
                        values[0] * width,
                        values[1] * height,
                        values[2] * width,
                        values[3] * height,
                    )
                else:
                    x0, y0, x1, y1 = values
                ix0 = max(0, min(width - 1, int(math.floor(x0))))
                iy0 = max(0, min(height - 1, int(math.floor(y0))))
                ix1 = max(ix0 + 1, min(width, int(math.ceil(x1))))
                iy1 = max(iy0 + 1, min(height, int(math.ceil(y1))))
                crop = depth[iy0:iy1, ix0:ix1]
                valid = crop[np.isfinite(crop) & (crop > 0.05)]
                if valid.size:
                    z = float(np.median(valid))
                    u = 0.5 * (x0 + x1)
                    v = 0.5 * (y0 + y1)
                    lateral = (u - cx) * z / fx
                    vertical = (v - cy) * z / fy
                    world_x = float(gps[0] + cos_h * z + sin_h * lateral)
                    # ROS map coordinates reflect Habitat GPS axis 1.  Positive
                    # image lateral points to camera-right, hence -cos(yaw) in
                    # the ROS map y component.
                    world_y = float(-gps[1] + sin_h * z - cos_h * lateral)
                    world_z = float(self.args.camera_height_m - vertical)
                    width_m = max(0.10, abs(x1 - x0) * z / fx)
                    height_m = max(0.10, abs(y1 - y0) * z / fy)
                    center = {"x": world_x, "y": world_y, "z": world_z}
                    size = {"x": width_m, "y": 0.30, "z": height_m}
                    row.update(
                        {
                            "world_position": center,
                            "world_box3d_center": center,
                            "world_box3d_size": size,
                            "position": center,
                            "box3d_center": center,
                            "box3d_size": size,
                        }
                    )
            lifted.append(row)
        return {
            **detections,
            "stamp_sec": int(stamp.secs),
            "stamp_nsec": int(stamp.nsecs),
            "detections": lifted,
            "bridge_rgbd_lift": True,
        }

    def _publish_diagnostic_panel(
        self,
        stamp: rospy.Time,
        payload: dict[str, Any],
        gps_xy: np.ndarray,
        graph: dict[str, Any],
    ) -> None:
        """Publish a public top-down OCC/route/goal-object diagnostic image."""

        grid = _decode_array(payload.get("occupancy"), dtype=np.int8, name="occupancy")
        canvas = np.full((*grid.shape, 3), 55, dtype=np.uint8)
        canvas[grid == 0] = (225, 225, 225)
        canvas[grid >= 50] = (20, 20, 20)
        resolution = float(payload["map_resolution_m"])
        origin = payload["map_origin_xy"]

        def pixel(x: float, y: float) -> tuple[int, int]:
            return (
                int(round((x - float(origin[0])) / resolution)),
                int(round((y - float(origin[1])) / resolution)),
            )

        route = payload.get("global_plan_xyyaw") or []
        route_pts = np.asarray([pixel(float(row[0]), float(row[1])) for row in route if isinstance(row, list) and len(row) >= 2], dtype=np.int32)
        if len(route_pts) >= 2:
            cv2.polylines(canvas, [route_pts.reshape((-1, 1, 2))], False, (255, 120, 20), 2)
        labels = {str(value).casefold().replace("_", " ") for value in payload.get("object_labels") or []}
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_label = str(node.get("semantic_class") or node.get("label") or node.get("name") or "").casefold().replace("_", " ")
            center = node.get("centroid") or node.get("aabb_center")
            if node_label not in labels or not isinstance(center, list) or len(center) < 2:
                continue
            cv2.drawMarker(canvas, pixel(float(center[0]), float(center[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        robot = pixel(float(gps_xy[0]), -float(gps_xy[1]))
        cv2.circle(canvas, robot, 5, (255, 80, 0), -1)
        known = np.argwhere(grid >= 0)
        if known.size:
            y0, x0 = np.maximum(known.min(axis=0) - 20, 0)
            y1, x1 = np.minimum(known.max(axis=0) + 21, np.asarray(grid.shape))
            canvas = canvas[int(y0):int(y1), int(x0):int(x1)]
        canvas = cv2.resize(canvas, (720, 720), interpolation=cv2.INTER_NEAREST)
        self.diagnostic_panel_pub.publish(_image_message(canvas, "bgr8", stamp, self.args.map_frame))

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._request_lock:
            self._requests += 1
            try:
                detector_tick = ((self._requests - 1) % self._detector_step_interval) == 0
                if detector_tick:
                    self._detector_ticks += 1
                gps = np.asarray(payload.get("gps"), dtype=np.float32).reshape(2)
                compass = float(payload.get("compass"))
                target = str(payload.get("object_category") or "")
                if not target:
                    raise ValueError("object_category is required")
                stamp = rospy.Time.now()
                self._publish_pose(stamp, gps, compass)
                if not self.args.original_ros_navigation:
                    self._publish_map(stamp, payload)
                    self._publish_map(stamp, payload, raw=True)
                    self._publish_path(stamp, payload.get("global_plan_xyyaw"), self.global_plan_pub)
                    self._publish_path(stamp, payload.get("local_plan_xyyaw"), self.local_plan_pub)
                labels = list(payload.get("object_labels") or [target])
                self._publish_sensor_frame(stamp, payload)
                deadline = time.monotonic() + float(self.args.update_wait_s)
                expected_stamp = float(stamp.to_sec())
                if detector_tick:
                    with self._condition:
                        while (
                            self._latest_detection_stamp + 1e-6 < expected_stamp
                            and time.monotonic() < deadline
                        ):
                            self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
                        detections = (
                            dict(self._latest_detections)
                            if self._latest_detection_stamp + 1e-6 >= expected_stamp
                            else {
                                "stamp_sec": int(stamp.secs),
                                "stamp_nsec": int(stamp.nsecs),
                                "detections": [],
                                "stale_result_discarded": True,
                            }
                        )
                else:
                    with self._condition:
                        detections = dict(self._latest_detections)
                    detections["stale_result_reused"] = True
                    detections["source_step_interval"] = self._detector_step_interval
                # Fail-safe for the timer-driven ROS detector: if no stamped
                # topic result arrived, query the same dedicated YOLO worker
                # with this exact RGB frame.  ROS semantic mapping remains
                # active; this only restores public detector evidence for the
                # Habitat/M2 bridge and keeps the frame-step association exact.
                direct_fallback = False
                if detector_tick and self._direct_detector_url and not (detections.get("detections") or []):
                    try:
                        request = Request(
                            self._direct_detector_url + "/detect",
                            data=json.dumps(
                                {"image_b64": payload["image_b64"]},
                                separators=(",", ":"),
                            ).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urlopen(request, timeout=2.0) as handle:
                            direct = json.loads(handle.read().decode("utf-8"))
                        if isinstance(direct, dict) and isinstance(direct.get("detections"), list):
                            detections = {**detections, "detections": direct["detections"], "direct_fallback": True}
                            direct_fallback = True
                    except (OSError, URLError, ValueError):
                        pass
                if direct_fallback:
                    # The timer-driven ROS detector missed this exact frame, so
                    # publish the same direct 2-D result, lifted only with public
                    # RGB-D/GPS/Compass, into the original semantic-mapping topic.
                    # This keeps the policy, graph, and recorder on one evidence
                    # stream without exposing Habitat goal geometry.
                    detections = self._lift_direct_detections(
                        detections,
                        payload,
                        gps,
                        compass,
                        stamp,
                    )
                    self.detection_relay_pub.publish(
                        String(data=json.dumps(detections, separators=(",", ":")))
                    )
                # The Habitat task vocabulary and YOLOE/Module-1 vocabulary
                # differ for several ObjectNav classes (couch/sofa,
                # television/tv, plant/potted plant).  Use the shared alias
                # expansion here so a valid detector result actually reaches
                # the target-context and M3 gates.
                normalized_labels = {
                    str(value).replace("_", " ").casefold()
                    for value in labels
                }
                try:
                    from habitat_v2_adapter.policy import goal_label_aliases

                    normalized_labels.update(
                        str(value).replace("_", " ").casefold()
                        for value in goal_label_aliases(target)
                    )
                except Exception:
                    pass
                visible = any(
                    isinstance(row, dict)
                    and normalized_labels.intersection(
                        {
                            str(row.get("semantic_class") or "").replace("_", " ").casefold(),
                            str(row.get("semantic_class_raw") or "").replace("_", " ").casefold(),
                        }
                    )
                    for row in (detections.get("detections") or [])
                )
                # Candidate generation is asynchronous and timer-driven.  A
                # stable target/visibility state cannot produce new semantic
                # information, so never wait for another 1 Hz publication in
                # that case.  Only a real context transition is published and
                # synchronized once; graph/frontier updates otherwise remain
                # opportunistic snapshots.
                context_key = (
                    target.strip().casefold(),
                    tuple(sorted(str(value).strip().casefold() for value in labels)),
                    bool(visible),
                )
                context_changed = context_key != self._last_target_context
                if context_changed:
                    with self._condition:
                        before_target_context = self._candidate_sequence
                    self._publish_target_context(target, labels, visible=visible)
                    self._last_target_context = context_key
                    self._candidate_context_waits += 1
                    target_deadline = time.monotonic() + min(2.0, float(self.args.update_wait_s))
                    with self._condition:
                        while (
                            self._candidate_sequence <= before_target_context
                            and time.monotonic() < target_deadline
                        ):
                            self._condition.wait(
                                timeout=max(0.0, target_deadline - time.monotonic())
                            )
                else:
                    self._candidate_context_skips += 1
                with self._condition:
                    graph = dict(self._latest_graph)
                    candidates = dict(self._latest_candidates)
                # The recorder freezes its six-panel frame on step_sync.  Emit
                # that marker only after ROS has consumed this RGB frame and
                # returned the corresponding detector/graph snapshot, preventing
                # the common one-frame-late detection overlay.
                if self.args.defer_step_sync_to_diagnostic:
                    self._pending_step_sync = (stamp, dict(payload), dict(detections))
                else:
                    self._publish_step_sync(stamp, payload, detections)
                    self._publish_diagnostic_panel(stamp, payload, gps, graph)
                return {
                    "ready": True,
                    "graph": graph,
                    "candidate_payload": candidates,
                    "detections": detections,
                    "candidate_sequence": self._candidate_sequence,
                    "candidate_context_changed": context_changed,
                    "cmd_vel": dict(self._latest_cmd_vel),
                    "cmd_vel_age_s": (
                        time.monotonic() - self._latest_cmd_vel_monotonic
                        if self._latest_cmd_vel_monotonic > 0.0
                        else None
                    ),
                    "odom_twist": dict(self._latest_measured_twist),
                    "detector_tick": detector_tick,
                    "detector_step_interval": self._detector_step_interval,
                    "move_base_status": self._latest_move_base_status,
                    "global_plan_xyyaw": list(self._latest_global_plan),
                    "local_plan_xyyaw": list(self._latest_local_plan),
                }
            except Exception:
                self._failures += 1
                raise

    def publish_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Publish an external M2 choice for recorder/observer parity only."""

        self._selection_requests += 1
        try:
            behavior_type = str(payload.get("behavior_type") or "").upper()
            goal = list(payload.get("goal_xyyaw") or [])
            if behavior_type not in {"EXPLORE", "NAVIGATE"}:
                raise ValueError("selection behavior must be EXPLORE or NAVIGATE")
            if len(goal) < 2:
                raise ValueError("selection goal_xyyaw requires x and y")
            if payload.get("interaction_command") is not None:
                raise ValueError("interaction_command is forbidden in Habitat navigation-only mode")
            mirrored = dict(payload)
            mirrored["behavior_type"] = behavior_type
            # M2 selects in Habitat's public GPS frame, whose second axis is the
            # reflection of ROS map y.  Odom and Path are already reflected in
            # this bridge; reflect the recorder-only selected goal as well so
            # the original recorder's endpoint consistency check does not hide
            # an otherwise valid global/local plan.
            mirrored["goal_xyyaw"] = [
                float(goal[0]),
                -float(goal[1]),
                float(goal[2]) if len(goal) > 2 else 0.0,
            ]
            mirrored["module3_enabled"] = False
            mirrored["execution_owner"] = (
                "move_base" if self.args.original_ros_navigation else "habitat_adapter"
            )
            mirrored["telemetry_only"] = not self.args.original_ros_navigation
            self.selection_pub.publish(
                String(data=json.dumps(mirrored, separators=(",", ":")))
            )
            if self.args.original_ros_navigation:
                goal_yaw = float(goal[2]) if len(goal) > 2 else 0.0
                key = (str(payload.get("candidate_id") or ""), float(goal[0]), -float(goal[1]), goal_yaw)
                if key != self._last_goal_key:
                    message = PoseStamped()
                    message.header.stamp = rospy.Time.now()
                    message.header.frame_id = self.args.map_frame
                    message.pose.position.x = key[1]
                    message.pose.position.y = key[2]
                    qx, qy, qz, qw = self._quaternion(goal_yaw)
                    message.pose.orientation.x = qx
                    message.pose.orientation.y = qy
                    message.pose.orientation.z = qz
                    message.pose.orientation.w = qw
                    self.simple_goal_pub.publish(message)
                    self._last_goal_key = key
            return {"ready": True, "published": True}
        except Exception:
            self._selection_failures += 1
            raise

    def publish_diagnostic(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Publish evaluator-only topdown diagnostics, then release step_sync."""

        image = _decode_rgb(payload)
        with self._request_lock:
            pending = self._pending_step_sync
            self._pending_step_sync = None
            stamp = pending[0] if pending is not None else rospy.Time.now()
            self.diagnostic_panel_pub.publish(
                _image_message(image, "rgb8", stamp, self.args.map_frame)
            )
            if pending is not None:
                self._publish_step_sync(pending[0], pending[1], pending[2])
        return {"ready": True, "published": True}

    def health(self) -> dict[str, Any]:
        return {
            "ready": not rospy.is_shutdown(),
            "mode": "full_navigation_stack",
            "module3_enabled": False,
            "requests": self._requests,
            "failures": self._failures,
            "selection_requests": self._selection_requests,
            "selection_failures": self._selection_failures,
            "candidate_sequence": self._candidate_sequence,
            "candidate_context_waits": self._candidate_context_waits,
            "candidate_context_skips": self._candidate_context_skips,
            "graph_nodes": len(self._latest_graph.get("nodes") or []),
            "graph_edges": len(self._latest_graph.get("edges") or []),
            "original_ros_navigation": bool(self.args.original_ros_navigation),
            "move_base_status": self._latest_move_base_status,
            "odom_twist": dict(self._latest_measured_twist),
            "detector_ticks": self._detector_ticks,
            "detector_step_interval": self._detector_step_interval,
        }

    def reset(self, _payload: dict[str, Any]) -> dict[str, Any]:
        with self._request_lock:
            self.move_base_cancel_pub.publish(GoalID(stamp=rospy.Time.now(), id=""))
            self.mapping_reset_pub.publish(Empty())
            self._last_goal_key = None
            self._last_target_context = None
            self._latest_global_plan = []
            self._latest_local_plan = []
            self._latest_move_base_status = 0
            self._latest_cmd_vel = {"linear_x": 0.0, "angular_z": 0.0}
            self._latest_cmd_vel_monotonic = 0.0
            self._latest_measured_twist = {
                "linear_x": 0.0,
                "linear_y": 0.0,
                "angular_z": 0.0,
            }
            self._last_measurement_pose_ros = None
            self._latest_pose = None
        return {"ready": True, "reset": True}


def _handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HabitatFullROSBridge/1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(HTTPStatus.OK, state.health())

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/step", "/selection", "/diagnostic", "/reset"}:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 48 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                if self.path == "/step":
                    result = state.step(payload)
                elif self.path == "/diagnostic":
                    result = state.publish_diagnostic(payload)
                elif self.path == "/reset":
                    result = state.reset(payload)
                else:
                    result = state.publish_selection(payload)
                self._send(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12230)
    parser.add_argument("--update-wait-s", type=float, default=1.5)
    parser.add_argument("--camera-height-m", type=float, default=1.31)
    parser.add_argument(
        "--sim-dt-s",
        type=float,
        default=0.1,
        help="Habitat control-step duration used to derive measured odometry twist",
    )
    parser.add_argument(
        "--detector-step-interval",
        type=int,
        default=2,
        help="Run Module-1 detection every N Habitat steps (2 at dt=0.1s gives 5 Hz)",
    )
    parser.add_argument("--map-frame", default="tf_frame_map")
    parser.add_argument("--odom-frame", default="tf_frame_odom")
    parser.add_argument("--base-frame", default="tf_frame_base_link")
    parser.add_argument("--lidar-frame", default="tf_frame_lidar")
    parser.add_argument("--camera-frame", default="tf_frame_camera")
    parser.add_argument("--original-ros-navigation", action="store_true")
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--pointcloud-max-depth-m", type=float, default=5.0)
    parser.add_argument("--rgb-topic", default="/habitat_v2/full/rgb")
    parser.add_argument("--depth-topic", default="/habitat_v2/full/depth")
    parser.add_argument("--camera-info-topic", default="/habitat_v2/full/camera_info")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--mapping-scan-topic", default="/molmo_spaces/organized_depth_scan")
    parser.add_argument("--registered-scan-topic", default="/registered_scan")
    parser.add_argument("--occupancy-topic", default="/struct_mapping/occ_map")
    parser.add_argument("--raw-occupancy-topic", default="/struct_mapping/raw_occ_map")
    parser.add_argument("--global-plan-topic", default="/habitat_v2/policy/global_plan")
    parser.add_argument("--local-plan-topic", default="/habitat_v2/policy/local_plan")
    parser.add_argument("--diagnostic-panel-topic", default="/habitat_v2/diagnostics/topdown_goal_occ")
    parser.add_argument("--defer-step-sync-to-diagnostic", action="store_true")
    parser.add_argument("--target-topic", default="/semantic_decision/target")
    parser.add_argument("--selected-behavior-topic", default="/semantic_decision/selected_behavior")
    parser.add_argument("--step-sync-topic", default="/habitat_v2/full/step_sync")
    parser.add_argument("--graph-topic", default="/semantic_mapping/unified_graph")
    parser.add_argument("--candidates-topic", default="/semantic_decision/candidates")
    parser.add_argument("--detections-topic", default="/semantic_mapping/object_detections")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--simple-goal-topic", default="/move_base_simple/goal")
    parser.add_argument("--move-base-cancel-topic", default="/move_base/cancel")
    parser.add_argument("--move-base-status-topic", default="/move_base/status")
    parser.add_argument("--move-base-global-plan-topic", default="/move_base/OrientedGlobalPlanner/plan")
    parser.add_argument("--move-base-local-plan-topic", default="/move_base/DWAPlannerROS/local_plan")
    parser.add_argument("--mapping-reset-topic", default="/nav_system/reset")
    args = parser.parse_args()
    state = _State(args)
    server = ThreadingHTTPServer((args.host, args.port), _handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
