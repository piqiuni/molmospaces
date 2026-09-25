#!/usr/bin/env python3
"""One stateless YOLOE HTTP inference replica for Habitat experiments."""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np

from semantic_mapping_py_pkg.detector_backends import (
    YoloeLocalProvider,
    _load_mapping_config,
    _map_open_vocabulary_class,
    _normalize_label,
)


class _UltralyticsPromptProvider:
    """Expose text-prompted and closed-set Ultralytics models via the M1 schema."""

    def __init__(self, args: argparse.Namespace) -> None:
        from ultralytics import YOLO, YOLOE, YOLOWorld

        self.model_path = str(Path(args.model_path).resolve())
        self.mode = str(args.model_mode)
        self.device = str(args.device)
        self.imgsz = int(args.imgsz)
        self.confidence = float(args.confidence_threshold)
        self.iou = float(args.iou_threshold)
        self.max_detections = int(args.max_detections)
        self.keep_unknown = bool(args.keep_unknown_open_set)
        self.mapping = _load_mapping_config(str(args.class_mapping))
        self.prompts = [item.strip() for item in str(args.prompt_list).split(",") if item.strip()]
        if self.mode == "yoloe_text":
            self.model = YOLOE(self.model_path)
            self.model.set_classes(self.prompts)
        elif self.mode == "yolo_world_text":
            self.model = YOLOWorld(self.model_path)
            self.model.set_classes(self.prompts)
        elif self.mode == "yolo_closed":
            self.model = YOLO(self.model_path)
        else:
            raise ValueError(f"unsupported model mode: {self.mode}")

    def detect_2d(self, rgb_image, _depth_image, _camera_info, _stamp):
        results = self.model.predict(
            source=rgb_image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            # Closed-set COCO classes must use class-aware NMS.  Class-agnostic
            # suppression can remove a TV box when it overlaps a chair/plant,
            # even though the offline YOLO segmentation model detects both.
            agnostic_nms=(self.mode != "yolo_closed"),
            max_det=self.max_detections,
            verbose=False,
            save=False,
        )
        if not results or results[0].boxes is None:
            return []
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        rows = []
        polygons = None
        if getattr(result, "masks", None) is not None:
            # The original ROS M1 box3d backend expects polygon vertices in
            # image pixels (not the normalized xyn representation).
            polygons = result.masks.xy
        for index, (bbox, confidence, class_id) in enumerate(zip(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy().astype(int),
        )):
            raw_name = str(names.get(int(class_id), str(class_id)))
            semantic_class = _map_open_vocabulary_class(raw_name, self.mapping)
            if semantic_class == "unknown_open_set" and not self.keep_unknown:
                continue
            row = {
                    "semantic_class_raw": _normalize_label(raw_name),
                    "semantic_class": semantic_class,
                    "confidence": float(confidence),
                    "bbox": [float(value) for value in bbox.tolist()],
                    "source_model": Path(self.model_path).name,
                }
            # Segmentation checkpoints expose normalized polygon masks.  Keep
            # them in the M1 JSON schema so the original box3d backend can
            # perform its depth/instance lifting; detection-only checkpoints
            # simply omit this field.
            if polygons is not None and index < len(polygons):
                polygon = np.asarray(polygons[index], dtype=np.float32)
                if polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3:
                    row["mask"] = polygon.tolist()
                    row["mask_area"] = int(round(abs(float(cv2.contourArea(polygon)))))
            rows.append(row)
        return rows


class _WorkerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_path = str(Path(args.model_path).resolve())
        self.model_mode = str(args.model_mode)
        self.prompt_list = str(args.prompt_list)
        self.replica_id = int(args.replica_id)
        if self.model_mode == "yoloe_pf":
            self._provider = YoloeLocalProvider(
                model_path=self.model_path,
                confidence_threshold=float(args.confidence_threshold),
                iou_threshold=float(args.iou_threshold),
                imgsz=int(args.imgsz),
                device=str(args.device),
                max_detections=int(args.max_detections),
                keep_unknown_open_set=bool(args.keep_unknown_open_set),
                class_mapping=str(args.class_mapping),
            )
        else:
            self._provider = _UltralyticsPromptProvider(args)
        # The gateway leases a replica to one request at a time.  Keep this
        # lock as a fail-closed guard for direct callers as well.
        self._inference_lock = threading.Lock()
        self.requests = 0
        self.failures = 0

    @staticmethod
    def _decode_image(payload: dict[str, Any]) -> np.ndarray:
        value = payload.get("image_b64")
        if not isinstance(value, str) or not value:
            raise ValueError("image_b64 is required")
        try:
            encoded = base64.b64decode(value, validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("image_b64 is invalid") from exc
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("image_b64 is not a decodable image")
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def detect(self, payload: dict[str, Any]) -> dict[str, Any]:
        rgb = self._decode_image(payload)
        with self._inference_lock:
            self.requests += 1
            try:
                detections = self._provider.detect_2d(rgb, None, None, None)
            except Exception:
                self.failures += 1
                raise
        return {
            "detections": detections,
            "replica_id": self.replica_id,
            "source_model": Path(self.model_path).name,
        }

    def health(self) -> dict[str, Any]:
        return {
            "ready": True,
            "replica_id": self.replica_id,
            "model_path": self.model_path,
            "model_mode": self.model_mode,
            "prompt_list": self.prompt_list if "text" in self.model_mode else "",
            "requests": self.requests,
            "failures": self.failures,
        }


def _handler(state: _WorkerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HabitatYOLOEReplica/1"

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
            if self.path != "/detect":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 24 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                self._send(HTTPStatus.OK, state.detect(payload))
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc), "detections": []})
            except Exception as exc:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"YOLOE inference failed: {exc}", "detections": []},
                )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--replica-id", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-mode",
        choices=("yoloe_pf", "yoloe_text", "yolo_world_text", "yolo_closed"),
        default="yoloe_pf",
    )
    parser.add_argument(
        "--prompt-list",
        default="chair,bed,potted plant,toilet,tv,couch",
        help="Comma-separated ObjectNav classes for text-prompted models.",
    )
    parser.add_argument("--class-mapping", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--keep-unknown-open-set", action="store_true")
    args = parser.parse_args()
    if not Path(args.model_path).is_file():
        parser.error(f"model does not exist: {args.model_path}")
    state = _WorkerState(args)
    # Force lazy model construction before advertising readiness.
    if hasattr(state._provider, "_get_model"):
        state._provider._get_model()
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
