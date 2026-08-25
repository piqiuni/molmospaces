#!/usr/bin/env python3
"""Strict HTTP-to-ROS relay for the original detector-only Module-1 node.

The Habitat Challenge process remains ROS-free.  It sends the original
``ExternalHttpProvider`` request over loopback HTTP; this process publishes
only RGB, depth, and camera intrinsics to the original InteractiveNav
``object_detection_node`` and returns only 2-D detector boxes.  It never
receives ObjectGoal, GPS, Compass, actions, interaction commands, simulator
metrics, or ground-truth scene data.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


_FORBIDDEN_REQUEST_FIELDS = {
    "action",
    "behavior_type",
    "collision",
    "goal_position",
    "goal_positions",
    "interaction_command",
    "metric",
    "pathfinder",
    "semantic",
    "semantic_observation",
    "view_points",
    "viewpoint",
    "world_position",
}
_ALLOWED_REQUEST_FIELDS = {"image_b64", "stamp_sec", "stamp_nsec", "depth", "camera_info"}
_ALLOWED_DETECTION_FIELDS = {"semantic_class", "semantic_class_raw", "confidence", "bbox", "source_model"}


@dataclass(frozen=True)
class _RequestFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    width: int
    height: int
    K: tuple[float, ...]


def _decode_frame(payload: dict[str, Any]) -> _RequestFrame:
    forbidden = sorted(_FORBIDDEN_REQUEST_FIELDS & set(payload))
    if forbidden:
        raise ValueError(f"forbidden bridge request field(s): {', '.join(forbidden)}")
    unknown = sorted(set(payload) - _ALLOWED_REQUEST_FIELDS)
    if unknown:
        raise ValueError(f"unsupported bridge request field(s): {', '.join(unknown)}")
    image_b64 = payload.get("image_b64")
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("image_b64 is required")
    try:
        encoded = np.frombuffer(base64.b64decode(image_b64, validate=True), dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_b64 is not valid base64") from exc
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("image_b64 is not a decodable JPEG/PNG image")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]

    camera_info = payload.get("camera_info")
    if not isinstance(camera_info, dict):
        raise ValueError("camera_info is required")
    try:
        supplied_width = int(camera_info["width"])
        supplied_height = int(camera_info["height"])
        K = tuple(float(value) for value in camera_info["K"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera_info must contain width, height, and nine K values") from exc
    if supplied_width != width or supplied_height != height or len(K) != 9:
        raise ValueError("camera_info dimensions/intrinsics do not match RGB image")

    depth_payload = payload.get("depth")
    if depth_payload is None:
        depth_m = np.full((height, width), np.nan, dtype=np.float32)
    else:
        if not isinstance(depth_payload, dict):
            raise ValueError("depth must use the ExternalHttpProvider object schema")
        try:
            dtype = np.dtype(str(depth_payload["dtype"]))
            shape = tuple(int(value) for value in depth_payload["shape"])
            raw = base64.b64decode(str(depth_payload["data_b64"]), validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("depth payload is invalid") from exc
        if dtype != np.dtype(np.float32) or shape not in {(height, width), (height, width, 1)}:
            raise ValueError("depth must be float32 with the RGB image dimensions")
        try:
            depth_m = np.frombuffer(raw, dtype=dtype).reshape(shape)
        except ValueError as exc:
            raise ValueError("depth byte count does not match its declared shape") from exc
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        depth_m = np.ascontiguousarray(depth_m.astype(np.float32, copy=False))
    return _RequestFrame(rgb=rgb, depth_m=depth_m, width=width, height=height, K=K)


def _image_message(array: np.ndarray, encoding: str, stamp: rospy.Time, frame_id: str) -> Image:
    contiguous = np.ascontiguousarray(array)
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = int(contiguous.shape[0])
    message.width = int(contiguous.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(contiguous.strides[0])
    message.data = contiguous.tobytes(order="C")
    return message


class _RelayState:
    def __init__(self, args: argparse.Namespace) -> None:
        self._frame_id = str(args.frame_id)
        self._timeout_s = float(args.timeout_s)
        self._rgb_pub = rospy.Publisher(args.rgb_topic, Image, queue_size=1)
        self._depth_pub = rospy.Publisher(args.depth_topic, Image, queue_size=1)
        self._info_pub = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=1)
        self._condition = threading.Condition()
        self._request_lock = threading.Lock()
        self._responses: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._next_stamp_ns = max(time.time_ns(), 1)
        self._detections_sub = rospy.Subscriber(
            args.object_detections_topic,
            String,
            self._detections_callback,
            queue_size=10,
        )

    def health(self) -> dict[str, Any]:
        connected = min(
            self._rgb_pub.get_num_connections(),
            self._depth_pub.get_num_connections(),
            self._info_pub.get_num_connections(),
        )
        return {
            "ready": bool(not rospy.is_shutdown() and connected > 0),
            "mode": "detector_only",
            "transport": "ros_http_bridge",
            "module2_enabled": False,
            "module3_enabled": False,
            "upstream": "original_object_detection_node",
            "source_model": "yoloe-26x-seg-pf.pt",
            "ros_master_uri": os.environ.get("ROS_MASTER_URI", ""),
            "subscriber_connections": int(connected),
        }

    def detect(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = _decode_frame(payload)
        # The upstream node intentionally operates on its latest frame in a timer.
        # Serialize requests and use a unique ROS timestamp, so an old timer result
        # can never be returned to a later Habitat observation or another episode.
        with self._request_lock:
            stamp = self._new_stamp()
            key = (int(stamp.secs), int(stamp.nsecs))
            with self._condition:
                self._responses.pop(key, None)
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = self._frame_id
            info.width = frame.width
            info.height = frame.height
            info.K = list(frame.K)
            self._info_pub.publish(info)
            self._depth_pub.publish(_image_message(frame.depth_m, "32FC1", stamp, self._frame_id))
            self._rgb_pub.publish(_image_message(frame.rgb, "rgb8", stamp, self._frame_id))
            deadline = time.monotonic() + self._timeout_s
            with self._condition:
                while key not in self._responses:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise RuntimeError("timed out waiting for original Module-1 detector response")
                    self._condition.wait(timeout=remaining)
                detections = self._responses.pop(key)
        return {"detections": detections}

    def _new_stamp(self) -> rospy.Time:
        self._next_stamp_ns = max(self._next_stamp_ns + 1, time.time_ns())
        secs, nsecs = divmod(self._next_stamp_ns, 1_000_000_000)
        return rospy.Time(secs=int(secs), nsecs=int(nsecs))

    def _detections_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            secs = int(payload["stamp_sec"])
            nsecs = int(payload["stamp_nsec"])
            rows = payload.get("detections", [])
            if not isinstance(rows, list):
                return
            normalized = [row for item in rows if (row := self._normalize_detection(item)) is not None]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self._condition:
            self._responses[(secs, nsecs)] = normalized
            self._condition.notify_all()

    @staticmethod
    def _normalize_detection(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        label = str(item.get("semantic_class") or "").strip()
        raw_label = str(item.get("semantic_class_raw") or label).strip()
        bbox = item.get("bbox")
        if not label or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
            return None
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        row = {
            "semantic_class": label.casefold().replace(" ", "_"),
            "semantic_class_raw": raw_label.casefold().replace(" ", "_"),
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "source_model": str(item.get("source_model") or "yoloe-26x-seg-pf.pt"),
        }
        if set(row) - _ALLOWED_DETECTION_FIELDS:
            raise AssertionError("relay detection schema violation")
        return row


def _handler(state: _RelayState):
    class RelayHandler(BaseHTTPRequestHandler):
        server_version = "HabitatModule1RosRelay/1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send_json(HTTPStatus.OK, state.health())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 24 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                self._send_json(HTTPStatus.OK, state.detect(payload))
            except (RuntimeError, ValueError) as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc), "detections": []})

    return RelayHandler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12188)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--frame-id", default="habitat_v2_camera")
    parser.add_argument("--rgb-topic", default="/habitat_v2/module1/rgb")
    parser.add_argument("--depth-topic", default="/habitat_v2/module1/depth")
    parser.add_argument("--camera-info-topic", default="/habitat_v2/module1/camera_info")
    parser.add_argument("--object-detections-topic", default="/habitat_v2/module1/object_detections")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout_s <= 0.0:
        parser.error("port must be in [1, 65535] and timeout must be positive")
    return args


def main() -> int:
    args = _parse_args()
    rospy.init_node("habitat_v2_module1_ros_relay", anonymous=True, disable_signals=True)
    state = _RelayState(args)
    server = ThreadingHTTPServer((args.host, args.port), _handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        rospy.signal_shutdown("HTTP relay stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
