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
import logging
import math
import struct
import sys
import threading
import time
import urllib.request
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import Point, TransformStamped, PointStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, String
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from PIL import Image as PILImage


def _decode(encoded: str) -> np.ndarray:
    image = PILImage.open(io.BytesIO(base64.b64decode(encoded)))
    return np.asarray(image)


def _patch_roslogging_findcaller_for_py311() -> None:
    """Avoid rosgraph's Python 3.11 ``findCaller`` recursion."""
    if sys.version_info < (3, 11):
        return
    try:
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_physical_gateway_safe", False):
        return

    def _safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 3:
            return result[0], result[1], result[2], None
        return result

    _safe_find_caller._physical_gateway_safe = True
    roslogging.RospyLogger.findCaller = _safe_find_caller


class PhysicalRosGateway:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.args.occupancy_period = max(0.0, float(self.args.occupancy_period))
        self.last_seq = -1
        self.last_stamp = float("-inf")
        self.last_raw: dict[str, Any] | None = None
        self.world_frame = str(getattr(self.args, "world_frame", "tf_frame_map"))
        self.rgb_pub = rospy.Publisher("/physical_nav/rgb/image_raw", Image, queue_size=1)
        self.detection_overlay_pub = rospy.Publisher(
            "/physical_nav/detections_overlay", Image, queue_size=1, latch=True
        )
        self.depth_pub = rospy.Publisher("/physical_nav/depth/image_raw", Image, queue_size=1)
        self.info_pub = rospy.Publisher("/physical_nav/camera_info", CameraInfo, queue_size=1)
        self.depth_info_pub = rospy.Publisher(
            "/physical_nav/depth_camera_info", CameraInfo, queue_size=1
        )
        # Latest-only transport with TCP_NODELAY keeps a fresh cloud from
        # waiting behind a large serialized PointCloud2 packet.
        self.cloud_pub = rospy.Publisher(
            "/physical_nav/points", PointCloud2, queue_size=1, tcp_nodelay=True
        )
        self.segmented_cloud_pub = rospy.Publisher(
            "/physical_nav/segmented_cloud", PointCloud2, queue_size=1
        )
        self.segmented_cloud_world_pub = rospy.Publisher(
            "/physical_nav/segmented_cloud_world", PointCloud2, queue_size=1
        )
        self.boxes_pub = rospy.Publisher("/physical_nav/boxes_3d", MarkerArray, queue_size=1)
        self.boxes_world_pub = rospy.Publisher(
            "/physical_nav/boxes_3d_world", MarkerArray, queue_size=1, latch=True
        )
        self.odom_pub = rospy.Publisher(
            "/physical_nav/odom", Odometry, queue_size=1, tcp_nodelay=True
        )
        self.detection_pub = rospy.Publisher("/physical_nav/detections", String, queue_size=1)
        # M1 needs public semantic names and a stable identity, while the raw
        # mapper should continue receiving untouched detector instances.  A
        # separate topic prevents M1-only identity hints from changing 3-D
        # tracking/merge behaviour.
        self.attribute_detection_pub = rospy.Publisher(
            "/physical_nav/attribute_detections", String, queue_size=1
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(); self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0)); self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        visualization = rospy.get_param("~object_detection", {})
        if not isinstance(visualization, dict):
            visualization = {}
        self._box_hold_s = max(
            0.0, float(visualization.get("debug_box_hold_s", args.box_hold_s))
        )
        self._box_match_distance_m = max(
            0.05,
            float(visualization.get("debug_box_match_distance_m", args.box_match_distance_m)),
        )
        self._box_smoothing_alpha = min(
            1.0,
            max(
                0.01,
                float(visualization.get("debug_box_smoothing_alpha", args.box_smoothing_alpha)),
            ),
        )
        self._box_min_confirmations = max(
            1,
            int(visualization.get("debug_box_min_confirmations", args.box_min_confirmations)),
        )
        self._world_box_tracks: dict[int, dict[str, Any]] = {}
        self._next_world_box_track_id = 1
        self._last_detection_receipt: tuple[Any, Any] | None = None
        self._static_sent = False; self._static_frames: set[str] = set(); self._lock = threading.Lock(); self._telemetry: dict[str, Any] = {}; self._last_grid_post: dict[str, float] = {}; self._last_occupancy_post = 0.0; self._mapped_post_lock = threading.Lock(); self._mapped_post_busy = False
        self._debug_cloud_lock = threading.Lock()
        self._pending_debug_cloud: tuple[list[dict[str, Any]], dict[str, Any], Any] | None = None
        self._projection_cache: dict[str, Any] = {}
        self._state_poll_lock = threading.Lock()
        self._pose_timer = rospy.Timer(rospy.Duration(0.05), self._refresh_pose_tf)
        # Detections originate in the non-ROS YOLOE worker and are republished
        # below for the existing mapper.  Do not subscribe to the same topic
        # here: that would feed our own message back into the HTTP state loop.
        for topic, name in (("/physical_nav/unified_graph", "graph"), ("/physical_nav/consistency", "consistency"),):
            rospy.Subscriber(topic, String, self._json_callback(name), queue_size=1)
        # Shadow navigation/decision outputs are observed for rendering and
        # evaluation only. They are never forwarded to a robot controller.
        for topic, name in (
            ("/explore_py/status", "explore_status"),
            ("/semantic_decision/candidates", "candidates"),
            ("/semantic_decision/selected_behavior", "selection"),
            ("/semantic_decision/execution_state", "execution_state"),
            ("/semantic_decision/behavior_feedback", "behavior_feedback"),
            ("/semantic_decision/decision_trace", "decision_trace"),
            ("/physical_nav/interaction_result", "interaction_result"),
            ("/physical_nav/mllm_events", "mllm_events"),
        ):
            rospy.Subscriber(topic, String, self._json_callback(name), queue_size=2)
        rospy.Subscriber("/explore_py/current_subgoal", PointStamped, self._subgoal_callback, queue_size=2)
        for topic, name in (
            (self.args.global_plan_topic, "global_plan"),
            (self.args.local_global_plan_topic, "local_global_plan"),
            (self.args.local_plan_topic, "local_plan"),
        ):
            if topic:
                rospy.Subscriber(topic, Path, self._plan_callback(name), queue_size=1)
        for topic, name in (
            (self.args.occupancy_grid_topic, "occupancy"),
            (self.args.room_grid_topic, "room_grid"),
            (self.args.global_costmap_topic, "global_costmap"),
            (self.args.local_costmap_topic, "local_costmap"),
        ):
            if topic:
                rospy.Subscriber(topic, OccupancyGrid, self._grid_callback(name), queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(args.rate, 1e-3)), self._poll)
        # Detection/graph projection may take longer than one camera period.
        # Keep it off the RGB-D timer so 10 Hz image and point-cloud delivery
        # isn't serialized behind presentation/debug work.
        self._state_timer = rospy.Timer(
            rospy.Duration(max(args.state_period, 0.05)),
            self._poll_state,
        )
        # Visualization is latest-frame-only and runs independently of the
        # state/detection processing path at the same configured frequency.
        self._debug_cloud_timer = rospy.Timer(
            rospy.Duration(max(args.state_period, 0.05)),
            self._publish_pending_debug_cloud,
        )

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

    def _subgoal_callback(self, msg: PointStamped) -> None:
        self._post_state("current_subgoal", {"point": [float(msg.point.x), float(msg.point.y), float(msg.point.z)], "frame_id": str(msg.header.frame_id or "")})

    @staticmethod
    def _plan_payload(msg: Path) -> dict[str, Any]:
        poses = []
        for stamped in msg.poses:
            p = stamped.pose.position
            q = stamped.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            poses.append([float(p.x), float(p.y), float(yaw)])
        return {"frame_id": str(msg.header.frame_id or ""), "poses": poses}

    def _plan_callback(self, name: str):
        def callback(msg: Path) -> None:
            self._post_state(name, self._plan_payload(msg))
        return callback

    @staticmethod
    def _grid_payload(msg: OccupancyGrid) -> dict[str, Any]:
        origin = msg.info.origin
        return {
            "width": int(msg.info.width),
            "height": int(msg.info.height),
            "resolution": float(msg.info.resolution),
            "frame_id": str(getattr(msg.header, "frame_id", "") or ""),
            "origin": {
                "x": float(origin.position.x),
                "y": float(origin.position.y),
                "z": float(origin.position.z),
                "qx": float(origin.orientation.x),
                "qy": float(origin.orientation.y),
                "qz": float(origin.orientation.z),
                "qw": float(origin.orientation.w),
            },
            "data": list(msg.data),
        }

    def _grid_callback(self, name: str):
        def callback(msg: OccupancyGrid) -> None:
            # Keep map-stage snapshots visible in the browser and available to
            # the canonical offline renderer without serializing ROS objects.
            now = time.monotonic()
            if now - self._last_grid_post.get(name, 0.0) < self.args.occupancy_period:
                return
            self._last_grid_post[name] = now
            self._post_state(name, self._grid_payload(msg))
        return callback

    def _occupancy_callback(self, msg: Any) -> None:
        """Compatibility callback retained for small unit-test stubs."""
        now = time.monotonic()
        if now - self._last_occupancy_post < self.args.occupancy_period:
            return
        self._last_occupancy_post = now
        self._post_state("occupancy", self._grid_payload(msg))

    def _claim_frame(self, raw: dict[str, Any]) -> bool:
        """Accept each capture once, including across bridge reconnects.

        The dog bridge sequence is process-local and resets when that process
        restarts. Capture time is stable across such restarts and also lets us
        reject delayed packets from an older WebSocket session.
        """
        if not raw.get("rgb") or not raw.get("depth"):
            return False
        seq = int(raw.get("seq", -1))
        try:
            stamp = float(raw.get("stamp", 0.0))
        except (TypeError, ValueError):
            stamp = 0.0
        has_stamp = math.isfinite(stamp) and stamp > 0.0
        if has_stamp:
            if stamp <= self.last_stamp:
                return False
        elif seq <= self.last_seq:
            return False
        if self.last_seq >= 0 and seq < self.last_seq:
            rospy.loginfo(
                "physical sensor sequence reset: %d -> %d; accepting newer capture stamp %.6f",
                self.last_seq,
                seq,
                stamp,
            )
        self.last_seq = seq
        if has_stamp:
            self.last_stamp = stamp
        return True

    def _poll(self, _event: Any) -> None:
        try:
            with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/raw-frame", timeout=.5) as response:
                raw = json.loads(response.read().decode())
            if not self._claim_frame(raw):
                return
            self.last_raw = raw
            self._publish(raw)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical raw-frame polling: %s", exc)

    def _poll_state(self, _event: Any) -> None:
        if not self._state_poll_lock.acquire(blocking=False):
            return
        try:
            self._publish_state()
        finally:
            self._state_poll_lock.release()

    def _refresh_pose_tf(self, _event: Any) -> None:
        """Refresh dynamic TF independently of the camera/WebSocket cadence."""
        telemetry = self._telemetry
        if telemetry:
            self._publish_pose(telemetry, rospy.Time.now())

    def _publish_state(self) -> None:
        try:
            # ROS/RViz uses a dedicated projection containing bounded compact
            # point clouds. Browser dashboards keep polling the lighter
            # /api/state-summary and do not pay this transport cost.
            with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/ros-state", timeout=.6) as response: state = json.loads(response.read().decode())
            detection_meta = state.get("detection_meta") or {}
            receipt = (
                detection_meta.get("seq", -1),
                detection_meta.get("stamp", 0.0),
            ) if isinstance(detection_meta, dict) else (-1, 0.0)
            detection_stamp = rospy.Time.from_sec(
                float(receipt[1] or state.get("frame_stamp", time.time()) or time.time())
            )
            if (rospy.Time.now() - detection_stamp).to_sec() > 0.08:
                detection_stamp = rospy.Time.now()
            if receipt == self._last_detection_receipt:
                # Do not count the same HTTP state snapshot as repeated YOLO
                # confirmations. Keep publishing held tracks so their timeout
                # remains observable even if the detector stops completely.
                self._publish_held_world_boxes(detection_stamp)
                return
            self._last_detection_receipt = receipt
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
            # Use the same history-smoothed portal geometry for semantic
            # mapping that is used by boxes_3d_world. Otherwise the point
            # cloud/marker follows the stable OBB while Graph receives the
            # raw PCA axis and can be rotated by roughly 90 degrees.
            stable_preview = self._stable_world_boxes(detections)
            stable_portals = [
                item for item in stable_preview
                if str(item.get("semantic_class") or "").casefold() in {"door", "portal", "gate"}
            ]
            for detection in detections:
                if str(detection.get("semantic_class") or "").casefold() not in {"door", "portal", "gate"} or not stable_portals:
                    continue
                center = self._point3(detection.get("world_box3d_center"))
                if center is None:
                    continue
                stable = min(
                    stable_portals,
                    key=lambda item: math.sqrt(sum(
                        (center[index] - self._point3(item.get("world_box3d_center"))[index]) ** 2
                        for index in range(3)
                    )) if self._point3(item.get("world_box3d_center")) is not None else float("inf"),
                )
                stable_center = self._point3(stable.get("world_box3d_center"))
                stable_size = self._point3(stable.get("world_box3d_size"))
                if stable_center is None or stable_size is None:
                    continue
                detection["world_box3d_center"] = list(stable_center)
                detection["world_box3d_size"] = list(stable_size)
                detection["world_box3d_marker_size"] = list(stable_size)
                detection["world_box3d_orientation"] = list(stable.get("world_box3d_orientation") or [0.0, 0.0, 0.0, 1.0])
                detection["world_box3d_yaw"] = stable.get("world_box3d_yaw")
                detection["yaw"] = stable.get("world_box3d_yaw")
                detection["aabb_center"] = list(stable_center)
                detection["aabb_size"] = list(stable_size)
                detection["box3d_center"] = list(stable_center)
                detection["box3d_size"] = list(stable_size)
            mapping_detections = [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "camera_segment_points",
                        "world_segment_points",
                        "camera_segment_points_f32",
                        "world_segment_points_f32",
                        "segment_point_count",
                    }
                }
                for item in detections
            ]
            detection_envelope = {
                "seq": receipt[0],
                "stamp": receipt[1],
                "detections": mapping_detections,
            }
            self.detection_pub.publish(json.dumps(detection_envelope, ensure_ascii=False, separators=(",", ":")))
            attribute_envelope = dict(detection_envelope)
            attribute_envelope["detections"] = [
                self._attribute_detection(item) for item in mapping_detections
            ]
            self.attribute_detection_pub.publish(
                json.dumps(attribute_envelope, ensure_ascii=False, separators=(",", ":"))
            )
            with self._debug_cloud_lock:
                self._pending_debug_cloud = (detections, transform_cache, detection_stamp)
            self._publish_detection_overlay(state)
            # Keep a compact, map-aligned evidence view for the LAN page while
            # preserving the raw YOLOE masks in ``detections``.
            compact = [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "mask",
                        "camera_segment_points",
                        "world_segment_points",
                        "camera_segment_points_f32",
                        "world_segment_points_f32",
                    }
                }
                for item in detections
            ]
            self._post_mapped_state({"seq": receipt[0], "stamp": receipt[1], "map_frame": self.world_frame, "detections": compact})
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical state polling: %s", exc)

    @staticmethod
    def _attribute_detection(item: dict[str, Any]) -> dict[str, Any]:
        """Add the minimum stable/public fields required by physical M1."""
        result = dict(item)
        label = str(
            item.get("semantic_class")
            or item.get("raw_class")
            or item.get("class")
            or "object"
        ).strip().casefold()
        result.setdefault("semantic_name", label)
        result.setdefault("category", label)
        result.setdefault("name", label)
        result.setdefault("bbox_2d", item.get("bbox"))
        if result.get("visible_fraction") is None:
            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                area = max(1.0, abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1])))
                result["visible_fraction"] = min(
                    1.0, float(item.get("mask_area", 0) or 0) / area
                )
        if result.get("distance_m") is None:
            result["distance_m"] = item.get("depth_median_m")

        # Quantized map-space identity is stable across nearby frames but is
        # deliberately scoped to this M1-only topic.  It avoids one Qwen call
        # per image without forcing raw detector IDs into ObjectMapStore.
        center = item.get("world_position") or item.get("position") or {}
        try:
            if isinstance(center, dict):
                xyz = [float(center.get(axis, 0.0)) for axis in ("x", "y", "z")]
            else:
                xyz = [float(value) for value in center[:3]]
            cells = [int(round(value / 0.5)) for value in xyz]
            result.setdefault(
                "instance_id",
                "physical_{}_{}_{}_{}".format(label.replace(" ", "_"), *cells),
            )
        except (TypeError, ValueError, IndexError):
            result.setdefault(
                "instance_id",
                f"physical_{label.replace(' ', '_')}_{int(item.get('capture_seq', 0) or 0)}",
            )
        return result

    def _post_mapped_state(self, value: dict[str, Any]) -> None:
        # A slow local HTTP client must not stall the 10 Hz image/point-cloud
        # timer. Keep at most one in-flight compact evidence post; the next
        # polling cycle will carry a fresher frame if this one is delayed.
        with self._mapped_post_lock:
            if self._mapped_post_busy:
                return
            self._mapped_post_busy = True

        def worker() -> None:
            try:
                self._post_state("mapped_detections", value)
            finally:
                with self._mapped_post_lock:
                    self._mapped_post_busy = False

        threading.Thread(target=worker, name="physical-mapped-state", daemon=True).start()

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
    def _debug_color(label: str) -> tuple[int, int, int]:
        palette = (
            (245, 90, 70), (70, 200, 245), (80, 220, 120),
            (235, 190, 60), (180, 90, 235), (65, 150, 245),
            (235, 90, 170), (90, 220, 215),
        )
        token = str(label or "object")
        index = sum((offset + 1) * ord(char) for offset, char in enumerate(token))
        return palette[index % len(palette)]

    @staticmethod
    def _rgb_float(color: tuple[int, int, int]) -> float:
        red, green, blue = color
        packed = (int(red) << 16) | (int(green) << 8) | int(blue)
        return struct.unpack("f", struct.pack("I", packed))[0]

    @classmethod
    def _colored_cloud(cls, stamp: Any, frame: str, rows: list[tuple[float, float, float, float]]) -> PointCloud2:
        header = Header(stamp=stamp, frame_id=frame)
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, rows)

    @classmethod
    def _box_marker_array(
        cls,
        detections: list[dict[str, Any]],
        stamp: Any,
        frame: str,
        *,
        center_key: str,
        size_key: str,
        orientation_key: str | None = None,
        marker_size_key: str | None = None,
        persistent: bool = False,
    ) -> MarkerArray:
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        markers = [clear]
        edges = (
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        )
        for index, detection in enumerate(detections):
            center = cls._point3(detection.get(center_key))
            size = cls._point3(detection.get(marker_size_key or size_key))
            if center is None or size is None:
                continue
            size = tuple(max(abs(float(value)), 0.01) for value in size)
            color = cls._debug_color(str(detection.get("semantic_class") or "object"))
            corners = [
                Point(
                    x=center[0] + sx * size[0] * 0.5,
                    y=center[1] + sy * size[1] * 0.5,
                    z=center[2] + sz * size[2] * 0.5,
                )
                for sx, sy, sz in (
                    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                )
            ]
            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = frame
            box.ns = "physical_yoloe_boxes"
            track_id = int(detection.get("visualization_track_id", index))
            box.id = track_id * 2
            box.type = Marker.LINE_LIST
            box.action = Marker.ADD
            orientation = detection.get(orientation_key) if orientation_key else None
            if isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
                box.pose.orientation.x = float(orientation[0])
                box.pose.orientation.y = float(orientation[1])
                box.pose.orientation.z = float(orientation[2])
                box.pose.orientation.w = float(orientation[3])
            else:
                box.pose.orientation.w = 1.0
            box.pose.position.x, box.pose.position.y, box.pose.position.z = center
            box.scale.x = 0.025
            box.color.r, box.color.g, box.color.b = [channel / 255.0 for channel in color]
            box.color.a = 0.95
            box.lifetime = rospy.Duration(0.0 if persistent else 0.75)
            for first, second in edges:
                # Points are local to the marker pose so RViz applies the
                # measured camera/base orientation instead of forcing every
                # box to be world-axis aligned.
                local_first = Point(
                    x=(corners[first].x - center[0]),
                    y=(corners[first].y - center[1]),
                    z=(corners[first].z - center[2]),
                )
                local_second = Point(
                    x=(corners[second].x - center[0]),
                    y=(corners[second].y - center[1]),
                    z=(corners[second].z - center[2]),
                )
                box.points.extend((local_first, local_second))
            markers.append(box)

            label = Marker()
            label.header = box.header
            label.ns = "physical_yoloe_labels"
            label.id = track_id * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = center[0]
            label.pose.position.y = center[1]
            label.pose.position.z = center[2] + size[2] * 0.5 + 0.08
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.lifetime = box.lifetime
            label.text = "%s %.2f" % (
                str(detection.get("semantic_class") or "object"),
                float(detection.get("confidence", 0.0) or 0.0),
            )
            markers.append(label)
        return MarkerArray(markers=markers)

    @staticmethod
    def _bbox_iou(first: Any, second: Any) -> float:
        if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
            return 0.0
        if len(first) < 4 or len(second) < 4:
            return 0.0
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2]) - float(first[0])) * max(
            0.0, float(first[3]) - float(first[1])
        )
        second_area = max(0.0, float(second[2]) - float(second[0])) * max(
            0.0, float(second[3]) - float(second[1])
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 1e-9 else 0.0

    @staticmethod
    def _aabb_iou(
        first_center: tuple[float, float, float],
        first_size: tuple[float, float, float],
        second_center: tuple[float, float, float],
        second_size: tuple[float, float, float],
    ) -> float:
        first_min = [first_center[i] - abs(first_size[i]) * 0.5 for i in range(3)]
        first_max = [first_center[i] + abs(first_size[i]) * 0.5 for i in range(3)]
        second_min = [second_center[i] - abs(second_size[i]) * 0.5 for i in range(3)]
        second_max = [second_center[i] + abs(second_size[i]) * 0.5 for i in range(3)]
        overlap = [
            max(0.0, min(first_max[i], second_max[i]) - max(first_min[i], second_min[i]))
            for i in range(3)
        ]
        intersection = overlap[0] * overlap[1] * overlap[2]
        first_volume = abs(first_size[0] * first_size[1] * first_size[2])
        second_volume = abs(second_size[0] * second_size[1] * second_size[2])
        union = first_volume + second_volume - intersection
        return intersection / union if union > 1e-9 else 0.0

    def _stable_world_boxes(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Associate and smooth world-frame boxes for display only.

        The semantic mapper continues to consume raw detections.  This cache
        affects only ``boxes_3d_world`` so a missed YOLO frame or a short label
        flip does not make RViz/Foxglove markers jump or disappear.
        """
        now = time.monotonic()
        unmatched = set(self._world_box_tracks)
        alpha = self._box_smoothing_alpha
        for detection in sorted(
            detections,
            key=lambda item: float(item.get("confidence", 0.0) or 0.0),
            reverse=True,
        ):
            center = self._point3(detection.get("world_box3d_center"))
            size = self._point3(detection.get("world_box3d_size"))
            if center is None or size is None:
                continue
            if not all(math.isfinite(value) for value in center + size):
                continue
            size = tuple(max(0.01, abs(value)) for value in size)
            best_id = None
            best_score = float("inf")
            for track_id in unmatched:
                track = self._world_box_tracks[track_id]
                old_center = track["center"]
                old_size = track["size"]
                distance = math.sqrt(sum((center[i] - old_center[i]) ** 2 for i in range(3)))
                overlap_3d = self._aabb_iou(center, size, old_center, old_size)
                overlap_2d = self._bbox_iou(detection.get("bbox"), track.get("bbox"))
                if distance > self._box_match_distance_m and overlap_3d < 0.05:
                    continue
                score = distance - 0.35 * overlap_3d - 0.15 * overlap_2d
                if score < best_score:
                    best_id, best_score = track_id, score
            if best_id is None:
                best_id = self._next_world_box_track_id
                self._next_world_box_track_id += 1
                self._world_box_tracks[best_id] = {
                    "center": center,
                    "size": size,
                    "bbox": detection.get("bbox"),
                    "hits": 0,
                    "last_seen": now,
                    "class_scores": {},
                    "detection": dict(detection),
                }
            else:
                unmatched.discard(best_id)
            track = self._world_box_tracks[best_id]
            if track["hits"] > 0:
                track["center"] = tuple(
                    (1.0 - alpha) * track["center"][i] + alpha * center[i]
                    for i in range(3)
                )
                track["size"] = tuple(
                    (1.0 - alpha) * track["size"][i] + alpha * size[i]
                    for i in range(3)
                )
            track["hits"] += 1
            track["last_seen"] = now
            track["bbox"] = detection.get("bbox")
            label = str(detection.get("semantic_class") or "object")
            scores = track["class_scores"]
            scores[label] = float(scores.get(label, 0.0)) + float(
                detection.get("confidence", 0.0) or 0.0
            )
            track["detection"] = dict(detection)
            if label.casefold() in {"door", "portal", "gate"}:
                orientation = detection.get("world_box3d_orientation")
                raw_yaw = None
                if isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
                    raw_yaw = self._quaternion_yaw([float(value) for value in orientation[:4]])
                elif detection.get("world_box3d_yaw") is not None:
                    try:
                        raw_yaw = float(detection.get("world_box3d_yaw"))
                    except (TypeError, ValueError):
                        raw_yaw = None
                if raw_yaw is not None and math.isfinite(raw_yaw):
                    previous_yaw = track.get("yaw")
                    if previous_yaw is None:
                        track["yaw"] = raw_yaw
                    else:
                        candidates = [raw_yaw + index * (math.pi * 0.5) for index in range(-2, 3)]
                        aligned_yaw = min(candidates, key=lambda value: abs(0.5 * math.atan2(
                            math.sin(2.0 * (value - float(previous_yaw))),
                            math.cos(2.0 * (value - float(previous_yaw))),
                        )))
                        delta = 0.5 * math.atan2(
                            math.sin(2.0 * (aligned_yaw - float(previous_yaw))),
                            math.cos(2.0 * (aligned_yaw - float(previous_yaw))),
                        )
                        track["yaw"] = float(previous_yaw) if abs(delta) > math.pi * 0.25 else float(previous_yaw) + 0.20 * delta

        expired = [
            track_id
            for track_id, track in self._world_box_tracks.items()
            if now - float(track["last_seen"]) > self._box_hold_s
        ]
        for track_id in expired:
            del self._world_box_tracks[track_id]

        stable = []
        for track_id, track in sorted(self._world_box_tracks.items()):
            if int(track["hits"]) < self._box_min_confirmations:
                continue
            item = dict(track["detection"])
            item["visualization_track_id"] = track_id
            item["world_box3d_center"] = list(track["center"])
            item["world_box3d_size"] = list(track["size"])
            item["world_box3d_marker_size"] = list(track["size"])
            if track["class_scores"]:
                item["semantic_class"] = max(
                    track["class_scores"], key=track["class_scores"].get
                )
            if str(item.get("semantic_class") or "").casefold() in {"door", "portal", "gate"} and track.get("yaw") is not None:
                item["world_box3d_orientation"] = [
                    0.0,
                    0.0,
                    math.sin(float(track["yaw"]) * 0.5),
                    math.cos(float(track["yaw"]) * 0.5),
                ]
                item["world_box3d_yaw"] = float(track["yaw"])
            elif item.get("world_box3d_yaw") is not None:
                # Non-portal graph boxes must not retain the detector's
                # pre-TF yaw when the marker has already been fitted in the
                # map frame.
                item["yaw"] = float(item["world_box3d_yaw"])
            stable.append(item)
        return stable

    def _publish_detection_overlay(self, state: dict[str, Any]) -> None:
        meta = state.get("detection_meta") or {}
        encoded = meta.get("overlay_jpeg") if isinstance(meta, dict) else None
        try:
            if encoded:
                rgb = _decode(str(encoded))
            else:
                # The compact state summary intentionally excludes the large
                # base64 overlay so detection tracking cannot miss its 0.6 s
                # deadline. Fetch the existing binary debug image separately.
                with urllib.request.urlopen(
                    self.args.web_url.rstrip("/") + "/camera-overlay.jpg",
                    timeout=0.4,
                ) as response:
                    payload = response.read()
                rgb = np.asarray(
                    PILImage.open(io.BytesIO(payload)).convert("RGB")
                )
            if rgb.ndim == 3 and rgb.shape[2] >= 3:
                # PIL returns RGB for both transport paths; ROS publishes bgr8.
                rgb = np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1])
            stamp = rospy.Time.from_sec(
                float(meta.get("stamp", state.get("frame_stamp", time.time())) or time.time())
            )
            frame = str(meta.get("camera_frame") or self.args.camera_frame)
            self.detection_overlay_pub.publish(_image_msg(rgb, "bgr8", stamp, frame))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical detection overlay: %s", exc)

    def _publish_detection_debug(
        self,
        detections: list[dict[str, Any]],
        transform_cache: dict[str, Any],
        stamp: Any,
    ) -> None:
        started = time.perf_counter()
        camera_rows: list[tuple[float, float, float, float]] = []
        world_rows: list[tuple[float, float, float, float]] = []
        camera_frame = self.args.camera_frame
        for detection in detections:
            source_frame = str(detection.get("source_frame") or self.args.camera_frame)
            camera_frame = source_frame or camera_frame
            transform = transform_cache.get(source_frame)
            rgb = self._rgb_float(self._debug_color(str(detection.get("semantic_class") or "object")))
            for value in self._segment_points(
                detection, "camera_segment_points_f32", "camera_segment_points"
            ):
                point = self._point3(value)
                if point is None or not all(math.isfinite(axis) for axis in point):
                    continue
                camera_rows.append((point[0], point[1], point[2], rgb))
                if transform is not None:
                    rotated = self._rotate_point(transform.transform.rotation, point)
                    world_rows.append((
                        rotated[0] + float(transform.transform.translation.x),
                        rotated[1] + float(transform.transform.translation.y),
                        rotated[2] + float(transform.transform.translation.z),
                        rgb,
                    ))
            if transform is None:
                for value in self._segment_points(
                    detection, "world_segment_points_f32", "world_segment_points"
                ):
                    point = self._point3(value)
                    if point is None or not all(math.isfinite(axis) for axis in point):
                        continue
                    world_rows.append((point[0], point[1], point[2], rgb))
        self.segmented_cloud_pub.publish(self._colored_cloud(stamp, camera_frame, camera_rows))
        self.segmented_cloud_world_pub.publish(self._colored_cloud(stamp, self.world_frame, world_rows))
        self.boxes_pub.publish(self._box_marker_array(
            detections,
            stamp,
            camera_frame,
            center_key="camera_box3d_center",
            size_key="camera_box3d_size",
        ))
        stable_world = self._stable_world_boxes(detections)
        self.boxes_world_pub.publish(self._box_marker_array(
            stable_world,
            stamp,
            self.world_frame,
            center_key="world_box3d_center",
            size_key="world_box3d_size",
            orientation_key="world_box3d_orientation",
            marker_size_key="world_box3d_marker_size",
            persistent=True,
        ))
        rospy.loginfo_throttle(
            10.0,
            "segmented cloud timing: %.1f ms, camera_points=%d world_points=%d detections=%d",
            (time.perf_counter() - started) * 1000.0,
            len(camera_rows), len(world_rows), len(detections),
        )

    def _publish_pending_debug_cloud(self, _event: Any) -> None:
        """Publish only the newest visualization payload."""
        with self._debug_cloud_lock:
            pending = self._pending_debug_cloud
            self._pending_debug_cloud = None
        if pending is not None:
            self._publish_detection_debug(*pending)

    def _publish_held_world_boxes(self, stamp: Any) -> None:
        stable_world = self._stable_world_boxes([])
        self.boxes_world_pub.publish(
            self._box_marker_array(
                stable_world,
                stamp,
                self.world_frame,
                center_key="world_box3d_center",
                size_key="world_box3d_size",
                orientation_key="world_box3d_orientation",
                marker_size_key="world_box3d_marker_size",
                persistent=True,
            )
        )

    @staticmethod
    def _segment_points(
        detection: dict[str, Any], binary_key: str, legacy_key: str
    ) -> list[tuple[float, float, float]]:
        encoded = detection.get(binary_key)
        if encoded:
            try:
                values = np.frombuffer(base64.b64decode(str(encoded)), dtype="<f4")
                if values.size % 3 == 0:
                    return [tuple(float(axis) for axis in row) for row in values.reshape(-1, 3)]
            except Exception:
                pass
        rows = []
        for value in detection.get(legacy_key) or []:
            point = PhysicalRosGateway._point3(value)
            if point is not None:
                rows.append(point)
        return rows

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
        semantic_label = str(
            detection.get("semantic_class") or detection.get("raw_class") or ""
        ).strip().casefold()
        # Planar interaction targets need their complete visible extent. For
        # ordinary objects, the outer 2% of RGB-D mask points is frequently a
        # wall/floor leak and makes boxes metres larger than the object.
        bounds_low, bounds_high = (
            (0.02, 0.98)
            if semantic_label in {"door", "portal"}
            else (0.10, 0.90)
        )
        source_point = detection.get("camera_box3d_center") or detection.get("camera_position")
        # The world segmented cloud below is generated from these exact
        # camera-frame points and the TF transform.  Prefer their transformed
        # geometry for the box too, so a box can never be in the worker's
        # legacy telemetry frame while the cloud is in tf_frame_map.
        camera_segments = self._segment_points(
            detection, "camera_segment_points_f32", "camera_segment_points"
        )
        transformed_segments: list[tuple[float, float, float]] = []
        if transform is not None and camera_segments:
            for value in camera_segments:
                point = self._point3(value)
                if point is None or not all(math.isfinite(axis) for axis in point):
                    continue
                rotated = self._rotate_point(transform.transform.rotation, point)
                transformed_segments.append((
                    rotated[0] + float(transform.transform.translation.x),
                    rotated[1] + float(transform.transform.translation.y),
                    rotated[2] + float(transform.transform.translation.z),
                ))
        if transformed_segments:
            # Use the same finite transformed samples that are sent to
            # segmented_cloud_world. Quantiles reject an occasional depth
            # outlier while keeping the box centered on the visible cloud.
            values = np.asarray(transformed_segments, dtype=np.float32)
            mins = np.quantile(values, bounds_low, axis=0)
            maxs = np.quantile(values, bounds_high, axis=0)
            center = (mins + maxs) * 0.5
            size = np.maximum(maxs - mins, 0.01)
            obb_center, obb_size, obb_orientation = self._fit_world_obb(values)
            mapped["world_box3d_center"] = [float(axis) for axis in obb_center]
            mapped["world_box3d_size"] = [float(axis) for axis in obb_size]
            mapped["world_box3d_marker_size"] = [float(axis) for axis in obb_size]
            # ``size`` remains the world AABB for occupancy/semantic
            # clearance, while the marker gets a horizontal OBB fitted from
            # the exact TF-transformed segmented points.  Setting the marker
            # quaternion to identity here made the point cloud and its box
            # disagree whenever the door was rotated in the map.
            mapped["world_box3d_orientation"] = obb_orientation
            mapped["world_box3d_yaw"] = self._quaternion_yaw(obb_orientation)
            # The semantic/top-down path consumes ``yaw``; keep it aligned
            # with the orientation used by the world 3-D marker.
            mapped["yaw"] = mapped["world_box3d_yaw"]
            mapped["world_position"] = self._point_dict(tuple(float(axis) for axis in center))
            mapped["position"] = self._point_dict(tuple(float(axis) for axis in center))
            mapped["aabb_center"] = [float(axis) for axis in center]
            mapped["aabb_size"] = [float(axis) for axis in size]
            mapped["box3d_center"] = [float(axis) for axis in center]
            mapped["box3d_size"] = [float(axis) for axis in size]
            mapped["map_frame"] = self.world_frame
            mapped["map_transform_status"] = "tf_segment_points"
            mapped["map_transform_source_frame"] = source_frame
            return mapped
        # If TF is temporarily unavailable, segmented_cloud_world falls back
        # to the worker-provided world samples. Derive the fallback box from
        # those same samples instead of mixing them with a different worker
        # OBB center/size.
        world_segments = self._segment_points(
            detection, "world_segment_points_f32", "world_segment_points"
        )
        if world_segments:
            values = np.asarray(
                [point for point in (self._point3(value) for value in world_segments)
                 if point is not None and all(math.isfinite(axis) for axis in point)],
                dtype=np.float32,
            )
            if values.size:
                mins = np.quantile(values, bounds_low, axis=0)
                maxs = np.quantile(values, bounds_high, axis=0)
                center = (mins + maxs) * 0.5
                size = np.maximum(maxs - mins, 0.01)
                obb_center, obb_size, obb_orientation = self._fit_world_obb(values)
                mapped["world_box3d_center"] = [float(axis) for axis in obb_center]
                mapped["world_box3d_size"] = [float(axis) for axis in obb_size]
                mapped["world_box3d_marker_size"] = [float(axis) for axis in obb_size]
                mapped["world_box3d_orientation"] = obb_orientation
                mapped["world_box3d_yaw"] = self._quaternion_yaw(obb_orientation)
                mapped["yaw"] = mapped["world_box3d_yaw"]
                mapped["world_position"] = self._point_dict(tuple(float(axis) for axis in center))
                mapped["position"] = self._point_dict(tuple(float(axis) for axis in center))
                mapped["aabb_center"] = [float(axis) for axis in center]
                mapped["aabb_size"] = [float(axis) for axis in size]
                mapped["box3d_center"] = [float(axis) for axis in center]
                mapped["box3d_size"] = [float(axis) for axis in size]
                mapped["map_transform_status"] = "world_segment_points_fallback"
                return mapped
        if not source_frame or not isinstance(source_point, (dict, list, tuple)):
            mapped.setdefault("map_transform_status", "telemetry_fallback")
            self._copy_world_fallback(mapped)
            return mapped
        if isinstance(source_point, dict):
            try: point = (float(source_point.get("x", 0.0)), float(source_point.get("y", 0.0)), float(source_point.get("z", 0.0)))
            except (TypeError, ValueError):
                mapped["map_transform_status"] = "telemetry_fallback"; self._copy_world_fallback(mapped); return mapped
        elif len(source_point) >= 3:
            try: point = (float(source_point[0]), float(source_point[1]), float(source_point[2]))
            except (TypeError, ValueError):
                mapped["map_transform_status"] = "telemetry_fallback"; self._copy_world_fallback(mapped); return mapped
        else:
            mapped["map_transform_status"] = "telemetry_fallback"; self._copy_world_fallback(mapped); return mapped
        try:
            if transform is None:
                raise RuntimeError("sensor-to-map transform unavailable")
            translated = self._rotate_point(transform.transform.rotation, point)
            translated = (translated[0] + float(transform.transform.translation.x), translated[1] + float(transform.transform.translation.y), translated[2] + float(transform.transform.translation.z))
        except Exception:
            mapped["map_transform_status"] = "telemetry_fallback"
            self._copy_world_fallback(mapped)
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
            # The marker must use the same transform-derived geometry as the
            # mapped box.  Do not overwrite it with the worker's legacy
            # telemetry-frame marker size: that value can be expressed in a
            # different frame and makes a correct point cloud appear to have
            # a displaced/inconsistent 3-D box.
            mapped["world_box3d_marker_size"] = list(map_size)
            mapped["world_box3d_orientation"] = list(
                mapped.get("world_box3d_orientation") or [0.0, 0.0, 0.0, 1.0]
            )
            mapped["world_box3d_yaw"] = self._quaternion_yaw(
                mapped["world_box3d_orientation"]
            )
            mapped["yaw"] = mapped["world_box3d_yaw"]
            mapped["aabb_size"] = list(map_size)
            mapped["box3d_size"] = list(map_size)
        mapped["map_frame"] = self.world_frame
        mapped["map_transform_status"] = "tf"
        mapped["map_transform_source_frame"] = source_frame
        return mapped

    @staticmethod
    def _quaternion_yaw(quaternion: list[float]) -> float:
        if len(quaternion) < 4:
            return 0.0
        z = float(quaternion[2])
        w = float(quaternion[3])
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

    @staticmethod
    def _fit_world_obb(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Fit a yaw-only world OBB to the same points used by the cloud."""
        points = np.asarray(values, dtype=np.float64)
        center = np.mean(points, axis=0)
        if points.shape[0] < 3:
            return center, np.maximum(np.ptp(points, axis=0), 0.01), [0.0, 0.0, 0.0, 1.0]
        centered = points - center
        covariance = centered[:, :2].T @ centered[:, :2] / max(1, points.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        horizontal = eigenvectors[:, int(np.argmax(eigenvalues))]
        yaw = math.atan2(float(horizontal[1]), float(horizontal[0]))
        if math.cos(yaw) < 0.0 or (abs(math.cos(yaw)) < 1e-6 and math.sin(yaw) < 0.0):
            yaw += math.pi
        c, s = math.cos(yaw), math.sin(yaw)
        axes = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # ``axes`` is the local-to-world rotation.  Since points are stored
        # as row vectors, multiplying by ``axes`` projects world rows back to
        # the local OBB frame.  Keeping this convention makes the fitted
        # dimensions and quaternion describe the same box.
        projected = centered @ axes
        mins = np.quantile(projected, 0.02, axis=0)
        maxs = np.quantile(projected, 0.98, axis=0)
        local_center = (mins + maxs) * 0.5
        obb_center = center + axes @ local_center
        obb_size = np.maximum(maxs - mins, 0.01)
        orientation = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        return obb_center, obb_size, orientation

    def _copy_world_fallback(self, mapped: dict[str, Any]) -> None:
        """Expose worker-computed telemetry world geometry when TF is late."""
        center = self._point3(
            mapped.get("world_box3d_center")
            or mapped.get("box3d_center")
            or mapped.get("aabb_center")
            or mapped.get("world_position")
        )
        size = self._point3(
            mapped.get("world_box3d_size")
            or mapped.get("box3d_size")
            or mapped.get("aabb_size")
        )
        if center is not None:
            mapped["world_box3d_center"] = list(center)
        if size is not None:
            mapped["world_box3d_size"] = list(size)
            # Keep the marker size in the fallback world frame.  Preserve the
            # worker OBB quaternion when TF is temporarily unavailable; do
            # not erase the angle with an identity quaternion.
            mapped["world_box3d_marker_size"] = list(size)

    def _publish(self, raw: dict[str, Any]) -> None:
        if isinstance(raw.get("telemetry"), dict):
            self._telemetry = dict(raw["telemetry"])
        rgb = _decode(raw["rgb"]); depth = _decode(raw["depth"])
        if rgb.ndim == 3 and rgb.shape[2] >= 3: rgb = np.ascontiguousarray(rgb[:, :, ::-1])
        if depth.dtype != np.uint16: depth = depth.astype(np.uint16)
        stamp = rospy.Time.from_sec(float(raw.get("stamp", time.time())))
        rgb_frame = str(raw.get("camera_frame", self.args.camera_frame))
        depth_frame = str(raw.get("depth_frame", rgb_frame))
        rgb_msg = _image_msg(rgb, "bgr8", stamp, rgb_frame)
        depth_msg = _image_msg(depth, "16UC1", stamp, depth_frame)
        # RViz transforms visualization messages at their header time. A
        # queued WebSocket receipt can otherwise make point clouds appear to
        # trail the live TF even when TF itself is current.
        if (rospy.Time.now() - stamp).to_sec() > 0.08:
            stamp = rospy.Time.now()
            rgb_msg = _image_msg(rgb, "bgr8", stamp, rgb_frame)
            depth_msg = _image_msg(depth, "16UC1", stamp, depth_frame)
        intr = raw.get("rgb_intrinsics") or raw.get("intrinsics", {})
        depth_intr = raw.get("depth_intrinsics") or intr
        info = CameraInfo(); info.header.stamp = stamp; info.header.frame_id = rgb_frame; info.width = int(intr.get("width", rgb.shape[1])); info.height = int(intr.get("height", rgb.shape[0])); info.K = [float(intr.get("fx", 0)), 0, float(intr.get("cx", 0)), 0, float(intr.get("fy", 0)), float(intr.get("cy", 0)), 0, 0, 1]
        distortion = [float(value) for value in (intr.get("distortion") or [])[:5]]
        info.D = distortion
        info.distortion_model = str(intr.get("distortion_model", "plumb_bob") or "plumb_bob")
        depth_info = CameraInfo()
        depth_info.header.stamp = stamp
        depth_info.header.frame_id = depth_frame
        depth_info.width = int(depth_intr.get("width", depth.shape[1]))
        depth_info.height = int(depth_intr.get("height", depth.shape[0]))
        depth_info.K = [
            float(depth_intr.get("fx", 0)), 0.0,
            float(depth_intr.get("cx", 0)), 0.0,
            float(depth_intr.get("fy", 0)),
            float(depth_intr.get("cy", 0)), 0.0, 0.0, 1.0,
        ]
        depth_info.D = [float(value) for value in (depth_intr.get("distortion") or [])[:5]]
        depth_info.distortion_model = str(
            depth_intr.get("distortion_model", "plumb_bob") or "plumb_bob"
        )
        self.rgb_pub.publish(rgb_msg); self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info); self.depth_info_pub.publish(depth_info)
        self._publish_cloud(depth, raw.get("depth_intrinsics") or intr, stamp, depth_frame); self._publish_pose(raw.get("telemetry", {}), stamp, depth_frame)

    def _publish_cloud(self, depth: np.ndarray, intr: dict[str, Any], stamp: Any, frame: str) -> None:
        started = time.perf_counter()
        fx, fy, cx, cy = [_to_float(intr.get(k)) for k in ("fx", "fy", "cx", "cy")]; scale = _to_float(self.last_raw.get("depth_scale", .001) if self.last_raw else .001)
        if fx <= 0 or fy <= 0: return
        obstacle_depth = max(0.0, float(self.args.max_depth_m))
        no_return_depth = max(
            obstacle_depth + 1e-3,
            float(self.args.no_return_depth_m),
        )
        stride = max(1, int(self.args.point_stride))
        sampled = np.asarray(depth[::stride, ::stride], dtype=np.float32) * scale
        valid = sampled > 0.0
        obstacle_mask = valid & (sampled <= obstacle_depth)
        no_return_mask = valid & ~obstacle_mask
        # Preserve the bearing of a finite return beyond the mapping horizon,
        # but clamp its endpoint just outside the obstacle range. The mapper's
        # clear-only layer and costmap_2d can ray-trace it without drawing a
        # GMapping occupied endpoint.
        projected_z = np.where(no_return_mask, no_return_depth, sampled)
        cache_key = (depth.shape[0], depth.shape[1], stride, fx, fy, cx, cy)
        cached = self._projection_cache.get("global")
        if cached is None or cached[0] != cache_key:
            u = np.arange(0, depth.shape[1], stride, dtype=np.float32)
            v = np.arange(0, depth.shape[0], stride, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)
            cached = (cache_key, (uu - cx) / fx, (vv - cy) / fy)
            self._projection_cache["global"] = cached
        _, x_factor, y_factor = cached
        xyz = np.column_stack((
            (x_factor * projected_z)[valid],
            (y_factor * projected_z)[valid],
            projected_z[valid],
        )).astype(np.float32, copy=False)
        obstacle_points = int(np.count_nonzero(obstacle_mask))
        no_return_points = int(np.count_nonzero(no_return_mask))
        header = Header(); header.stamp = stamp; header.frame_id = frame
        msg = PointCloud2(
            header=header, height=1, width=int(xyz.shape[0]),
            fields=[PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1), PointField("z", 8, PointField.FLOAT32, 1)],
            is_bigendian=False, point_step=12, row_step=int(xyz.shape[0] * 12),
            data=xyz.tobytes(order="C"), is_dense=False,
        )
        self.cloud_pub.publish(msg)
        rospy.loginfo_throttle(
            5.0,
            "mapping cloud: obstacle_points=%d no_return_points=%d obstacle_depth=%.2f no_return_depth=%.2f",
            obstacle_points,
            no_return_points,
            obstacle_depth,
            no_return_depth,
        )
        rospy.loginfo_throttle(
            10.0,
            "global cloud timing: %.1f ms, points=%d stride=%d",
            (time.perf_counter() - started) * 1000.0,
            int(xyz.shape[0]), stride,
        )

    def _publish_pose(self, telemetry: dict[str, Any], stamp: Any, depth_frame: str | None = None) -> None:
        # The Go2 sensor websocket can queue a capture for a few hundred ms.
        # Publishing that old capture timestamp as TF makes RViz extrapolate a
        # visibly lagging robot/camera pose.  Keep synchronized timestamps when
        # fresh, but re-date delayed TF/odom at receipt time.
        try:
            now = rospy.Time.now()
            if (now - stamp).to_sec() > 0.08:
                stamp = now
        except Exception:
            pass
        position = telemetry.get("position", [0, 0, 0]); velocity = telemetry.get("velocity", [0, 0, 0]); yaw = _to_float(telemetry.get("yaw", telemetry.get("imu", {}).get("rpy", [0, 0, 0])[2] if telemetry.get("imu") else 0))
        quaternion = _telemetry_quaternion(telemetry)
        odom = Odometry(); odom.header.stamp = stamp; odom.header.frame_id = "tf_frame_odom"; odom.child_frame_id = "tf_frame_base_link"; odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = [_to_float(v) for v in position[:3]]; odom.twist.twist.linear.x, odom.twist.twist.linear.y = [_to_float(v) for v in velocity[:2]]
        if quaternion is None:
            quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = quaternion
        self.odom_pub.publish(odom)
        transform = TransformStamped(); transform.header.stamp = stamp; transform.header.frame_id = "tf_frame_odom"; transform.child_frame_id = "tf_frame_base_link"; transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = [_to_float(v) for v in position[:3]]; transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = quaternion; self.tf_broadcaster.sendTransform(transform)
        static_frames = [self.args.camera_frame]
        if depth_frame and depth_frame != self.args.camera_frame:
            static_frames.append(depth_frame)
        for static_frame in static_frames:
            if static_frame in self._static_frames:
                continue
            static = TransformStamped(); static.header.stamp = stamp; static.header.frame_id = self.args.camera_parent; static.child_frame_id = self.args.camera_frame; static.transform.translation.x, static.transform.translation.y, static.transform.translation.z = self.args.camera_x, self.args.camera_y, self.args.camera_z
            # REP-103 optical frame -> Go2 base: optical x=right, y=down,
            # z=forward maps to base x=forward, y=left, z=up.  This is a
            # coordinate convention, not an additional physical mounting tilt.
            mount = _quat_from_rpy(self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw)
            optical = (0.5, -0.5, 0.5, -0.5)  # base <- d435i_color_optical
            qx, qy, qz, qw = _quat_multiply(mount, optical)
            static.transform.rotation.x, static.transform.rotation.y = qx, qy
            static.transform.rotation.z, static.transform.rotation.w = qz, qw
            static.child_frame_id = static_frame
            self.static_broadcaster.sendTransform(static); self._static_frames.add(static_frame)
        self._static_sent = bool(self._static_frames)


def _to_float(value: Any) -> float:
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def _telemetry_quaternion(telemetry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Read live orientation as (x, y, z, w), preferring camera pose/IMU."""
    # This quaternion drives odom -> base_link.  A D435i motion quaternion is
    # in the camera motion-module frame and must never replace the Go2 body
    # attitude here.
    candidates = [telemetry.get("camera_pose"), telemetry.get("d435i_pose"), telemetry.get("pose"), telemetry.get("imu")]
    for source in candidates:
        if not isinstance(source, dict):
            continue
        raw = source.get("quaternion") or source.get("orientation")
        if isinstance(raw, dict):
            try:
                values = [_to_float(raw[key]) for key in ("x", "y", "z", "w")]
            except KeyError:
                continue
        elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
            values = [_to_float(value) for value in raw[:4]]
            if source is telemetry.get("imu"):
                values = [values[1], values[2], values[3], values[0]]
        else:
            continue
        norm = math.sqrt(sum(value * value for value in values))
        if norm > 1e-6 and math.isfinite(norm):
            return tuple(value / norm for value in values)
    return None


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


def _quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)


def _image_msg(array: np.ndarray, encoding: str, stamp: Any, frame: str) -> Image:
    array = np.ascontiguousarray(array); msg = Image(); msg.header.stamp = stamp; msg.header.frame_id = frame; msg.height = int(array.shape[0]); msg.width = int(array.shape[1]); msg.encoding = encoding; msg.is_bigendian = 0; msg.step = int(array.strides[0]); msg.data = array.tobytes(); return msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--web-url", default="http://127.0.0.1:8765")
    p.add_argument("--rate", type=float, default=10.)
    p.add_argument("--state-period", type=float, default=.2)
    # Forward every fresh map publication to the web/semantic consumers. The
    # mapper itself already controls map generation; .5 s here imposed an
    # avoidable 2 Hz ceiling and made rotations appear to smear walls.
    p.add_argument("--occupancy-period", type=float, default=.1)
    p.add_argument("--point-stride", type=int, default=6)
    p.add_argument("--max-depth-m", type=float, default=8.)
    p.add_argument("--no-return-depth-m", type=float, default=8.05)
    p.add_argument("--world-frame", default="tf_frame_map")
    p.add_argument("--camera-frame", default="d435i_color_optical_frame")
    p.add_argument("--camera-parent", default="tf_frame_base_link")
    p.add_argument("--camera-x", type=float, default=.03)
    p.add_argument("--camera-y", type=float, default=0.)
    p.add_argument("--camera-z", type=float, default=.62)
    p.add_argument("--camera-roll", type=float, default=0.)
    p.add_argument("--camera-pitch", type=float, default=0.)
    p.add_argument("--camera-yaw", type=float, default=0.)
    p.add_argument("--box-hold-s", type=float, default=3.0)
    p.add_argument("--box-match-distance-m", type=float, default=.60)
    p.add_argument("--box-smoothing-alpha", type=float, default=1.0)
    p.add_argument("--box-min-confirmations", type=int, default=2)
    p.add_argument("--occupancy-grid-topic", default="/physical_nav/occupancy")
    p.add_argument("--room-grid-topic", default="/physical_nav/room_segment_grid")
    p.add_argument("--global-costmap-topic", default="/move_base/global_costmap/costmap")
    p.add_argument("--local-costmap-topic", default="/move_base/local_costmap/costmap")
    p.add_argument("--global-plan-topic", default="/move_base/GlobalPlanner/plan")
    p.add_argument("--local-global-plan-topic", default="/move_base/DWAPlannerROS/global_plan")
    p.add_argument("--local-plan-topic", default="/move_base/DWAPlannerROS/local_plan")
    args, _unknown = p.parse_known_args()
    # ROS launch appends __name/__log remappings; ignore those in the local
    # CLI parser so the gateway can also be run directly.
    _patch_roslogging_findcaller_for_py311()
    rospy.init_node("physical_ros_gateway", anonymous=False)
    for name in ("web_url", "rate", "state_period", "occupancy_period", "point_stride", "max_depth_m", "no_return_depth_m", "world_frame", "camera_frame", "camera_parent", "camera_x", "camera_y", "camera_z", "camera_roll", "camera_pitch", "camera_yaw", "box_hold_s", "box_match_distance_m", "box_smoothing_alpha", "box_min_confirmations", "occupancy_grid_topic", "room_grid_topic", "global_costmap_topic", "local_costmap_topic", "global_plan_topic", "local_global_plan_topic", "local_plan_topic"):
        setattr(args, name, rospy.get_param("~" + name, getattr(args, name)))
    PhysicalRosGateway(args); rospy.spin()


if __name__ == "__main__": main()
