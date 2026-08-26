#!/usr/bin/env python3
"""HTTP bridge between Habitat public observations and the full ROS graph stack."""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
import traceback
from typing import Any

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
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
        self._request_lock = threading.Lock()
        self._condition = threading.Condition()
        self._latest_graph: dict[str, Any] = {}
        self._latest_candidates: dict[str, Any] = {}
        self._latest_detections: dict[str, Any] = {}
        self._latest_detection_stamp = 0.0
        self._candidate_sequence = -1
        self._requests = 0
        self._failures = 0
        self._selection_requests = 0
        self._selection_failures = 0
        self._pending_step_sync: tuple[rospy.Time, dict[str, Any]] | None = None

        self.rgb_pub = rospy.Publisher(args.rgb_topic, Image, queue_size=1)
        self.depth_pub = rospy.Publisher(args.depth_topic, Image, queue_size=1)
        self.info_pub = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=1)
        self.odom_pub = rospy.Publisher(args.odom_topic, Odometry, queue_size=1)
        self.map_pub = rospy.Publisher(args.occupancy_topic, OccupancyGrid, queue_size=1, latch=True)
        self.raw_map_pub = rospy.Publisher(args.raw_occupancy_topic, OccupancyGrid, queue_size=1, latch=True)
        self.global_plan_pub = rospy.Publisher(args.global_plan_topic, Path, queue_size=1, latch=True)
        self.local_plan_pub = rospy.Publisher(args.local_plan_topic, Path, queue_size=1, latch=True)
        self.diagnostic_panel_pub = rospy.Publisher(args.diagnostic_panel_topic, Image, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher(args.target_topic, String, queue_size=1, latch=True)
        self.selection_pub = rospy.Publisher(args.selected_behavior_topic, String, queue_size=8)
        self.step_sync_pub = rospy.Publisher(args.step_sync_topic, String, queue_size=8)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(args.graph_topic, String, self._graph_callback, queue_size=1)
        rospy.Subscriber(args.candidates_topic, String, self._candidate_callback, queue_size=1)
        rospy.Subscriber(args.detections_topic, String, self._detection_callback, queue_size=1)

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

    @staticmethod
    def _quaternion(yaw: float) -> tuple[float, float, float, float]:
        return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))

    def _publish_pose(self, stamp: rospy.Time, gps_xy: np.ndarray, compass: float) -> None:
        # Habitat GPS uses [x, y] with forward=[cos(h), -sin(h)].  Reflect its
        # second axis so ROS receives the standard forward=[cos(yaw), sin(yaw)].
        ros_x = float(gps_xy[0])
        ros_y = -float(gps_xy[1])
        qx, qy, qz, qw = self._quaternion(float(compass))
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.args.map_frame
        odom.child_frame_id = self.args.base_frame
        odom.pose.pose.position.x = ros_x
        odom.pose.pose.position.y = ros_y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.args.map_frame
        transform.child_frame_id = self.args.camera_frame
        transform.transform.translation.x = ros_x
        transform.transform.translation.y = ros_y
        transform.transform.translation.z = float(self.args.camera_height_m)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_pub.sendTransform(transform)

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
        (self.raw_map_pub if raw else self.map_pub).publish(message)

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

    def _publish_step_sync(self, stamp: rospy.Time, payload: dict[str, Any]) -> None:
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        sequence = max(0, int(context.get("step", 0) or 0))
        self.step_sync_pub.publish(
            String(
                data=json.dumps(
                    {"step_index": sequence, "stamp_sec": float(stamp.to_sec())},
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
                gps = np.asarray(payload.get("gps"), dtype=np.float32).reshape(2)
                compass = float(payload.get("compass"))
                target = str(payload.get("object_category") or "")
                if not target:
                    raise ValueError("object_category is required")
                stamp = rospy.Time.now()
                before = self._candidate_sequence
                self._publish_pose(stamp, gps, compass)
                self._publish_map(stamp, payload)
                self._publish_map(stamp, payload, raw=True)
                self._publish_path(stamp, payload.get("global_plan_xyyaw"), self.global_plan_pub)
                self._publish_path(stamp, payload.get("local_plan_xyyaw"), self.local_plan_pub)
                labels = list(payload.get("object_labels") or [target])
                # Seed the graph with the public task target.  Visibility is
                # updated after this frame's detector result is available.
                self._publish_target_context(target, labels, visible=False)
                self._publish_sensor_frame(stamp, payload)
                deadline = time.monotonic() + float(self.args.update_wait_s)
                expected_stamp = float(stamp.to_sec())
                with self._condition:
                    while (
                        (
                            self._candidate_sequence <= before
                            or self._latest_detection_stamp + 1e-6 < expected_stamp
                        )
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
                normalized_labels = {str(value).replace("_", " ").casefold() for value in labels}
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
                # Candidate generation is asynchronous.  Wait for one revision
                # after publishing this frame's visibility so the response cannot
                # contain the previous target-context state.
                with self._condition:
                    before_target_context = self._candidate_sequence
                self._publish_target_context(target, labels, visible=visible)
                target_deadline = time.monotonic() + min(2.0, float(self.args.update_wait_s))
                with self._condition:
                    while (
                        self._candidate_sequence <= before_target_context
                        and time.monotonic() < target_deadline
                    ):
                        self._condition.wait(
                            timeout=max(0.0, target_deadline - time.monotonic())
                        )
                    graph = dict(self._latest_graph)
                    candidates = dict(self._latest_candidates)
                # The recorder freezes its six-panel frame on step_sync.  Emit
                # that marker only after ROS has consumed this RGB frame and
                # returned the corresponding detector/graph snapshot, preventing
                # the common one-frame-late detection overlay.
                if self.args.defer_step_sync_to_diagnostic:
                    self._pending_step_sync = (stamp, dict(payload))
                else:
                    self._publish_step_sync(stamp, payload)
                    self._publish_diagnostic_panel(stamp, payload, gps, graph)
                return {
                    "ready": True,
                    "graph": graph,
                    "candidate_payload": candidates,
                    "detections": detections,
                    "candidate_sequence": self._candidate_sequence,
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
            mirrored["goal_xyyaw"] = [
                float(goal[0]),
                float(goal[1]),
                float(goal[2]) if len(goal) > 2 else 0.0,
            ]
            mirrored["module3_enabled"] = False
            mirrored["execution_owner"] = "habitat_adapter"
            mirrored["telemetry_only"] = True
            self.selection_pub.publish(
                String(data=json.dumps(mirrored, separators=(",", ":")))
            )
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
                self._publish_step_sync(pending[0], pending[1])
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
            "graph_nodes": len(self._latest_graph.get("nodes") or []),
            "graph_edges": len(self._latest_graph.get("edges") or []),
        }


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
            if self.path not in {"/step", "/selection", "/diagnostic"}:
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
    parser.add_argument("--map-frame", default="habitat_v2_map")
    parser.add_argument("--base-frame", default="habitat_v2_base")
    parser.add_argument("--camera-frame", default="habitat_v2_camera")
    parser.add_argument("--rgb-topic", default="/habitat_v2/full/rgb")
    parser.add_argument("--depth-topic", default="/habitat_v2/full/depth")
    parser.add_argument("--camera-info-topic", default="/habitat_v2/full/camera_info")
    parser.add_argument("--odom-topic", default="/odom")
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
