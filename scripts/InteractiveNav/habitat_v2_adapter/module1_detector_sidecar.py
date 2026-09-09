#!/usr/bin/env python3
"""HTTP transport sidecar for the original Module-1 detector-only seam.

The process receives the exact ``ExternalHttpProvider`` request emitted by the
original InteractiveNav detector backend and normalizes an existing local
YOLOv7 service response into its detector schema.  It does not import Habitat,
does not receive GPS/Compass/ObjectGoal, and has no interaction/action API.
"""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


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


class _SidecarState:
    def __init__(self, upstream_yolo: str, timeout_s: float) -> None:
        self.upstream_yolo = upstream_yolo.rstrip("/") + "/yolov7"
        self.timeout_s = float(timeout_s)

    def detect(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            base64.b64decode(image_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image_b64 is not valid base64") from exc
        camera_info = payload.get("camera_info")
        if not isinstance(camera_info, dict):
            raise ValueError("camera_info is required for the Module-1 detector bridge")
        try:
            width = int(camera_info["width"])
            height = int(camera_info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("camera_info must contain positive width and height") from exc
        if width < 1 or height < 1:
            raise ValueError("camera_info width and height must be positive")
        if "depth" in payload and not isinstance(payload["depth"], dict):
            raise ValueError("depth must use the original ExternalHttpProvider object schema")

        upstream_payload = json.dumps({"image": image_b64}).encode("utf-8")
        request = Request(
            self.upstream_yolo,
            data=upstream_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                upstream = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError) as exc:
            raise RuntimeError(f"upstream YOLOv7 request failed: {exc}") from exc
        if not isinstance(upstream, dict):
            raise RuntimeError("upstream YOLOv7 response is not a JSON object")
        if upstream.get("error"):
            raise RuntimeError(f"upstream YOLOv7 error: {upstream['error']}")
        boxes = upstream.get("boxes")
        logits = upstream.get("logits")
        phrases = upstream.get("phrases")
        if not isinstance(boxes, list) or not isinstance(logits, list) or not isinstance(phrases, list):
            raise RuntimeError("upstream YOLOv7 response lacks boxes/logits/phrases lists")
        detections = []
        for box, confidence, label in zip(boxes, logits, phrases):
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
                score = float(confidence)
            except (TypeError, ValueError):
                continue
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.01:
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            x1, x2 = sorted((max(0.0, min(float(width - 1), x1)), max(0.0, min(float(width), x2))))
            y1, y2 = sorted((max(0.0, min(float(height - 1), y1)), max(0.0, min(float(height), y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            semantic_class = str(label or "").strip().casefold().replace(" ", "_")
            if not semantic_class:
                continue
            row = {
                "semantic_class": semantic_class,
                "semantic_class_raw": semantic_class,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
                "source_model": "vlfm_yolov7_e6e",
            }
            if set(row) - _ALLOWED_DETECTION_FIELDS:
                raise AssertionError("internal sidecar detection schema violation")
            detections.append(row)
        return {"detections": detections}


def _handler(state: _SidecarState):
    class Module1DetectorHandler(BaseHTTPRequestHandler):
        server_version = "HabitatModule1DetectorSidecar/1"

        def log_message(self, _format: str, *_args: Any) -> None:
            # The evaluator already records per-call evidence.  Avoid noisy
            # default access logs mixed into user terminal output.
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/health":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ready": True,
                    "mode": "detector_only",
                    "transport": "http_sidecar",
                    "module2_enabled": False,
                    "module3_enabled": False,
                    "upstream": "yolov7",
                },
            )

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/detect":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                self._send_json(HTTPStatus.OK, state.detect(payload))
            except (RuntimeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc), "detections": []})

    return Module1DetectorHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12186)
    parser.add_argument("--upstream-yolo", default="http://127.0.0.1:12184")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout_s <= 0.0:
        parser.error("port must be in [1, 65535] and timeout must be positive")
    server = ThreadingHTTPServer((args.host, args.port), _handler(_SidecarState(args.upstream_yolo, args.timeout_s)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
