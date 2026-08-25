#!/usr/bin/env python3
"""Loopback-only GroundingDINO worker for the Habitat-v2 adapter.

Run this file in the existing VLFM Python environment.  It deliberately owns
the cross-environment complexity (model loading, JPEG decoding, and HTTP) so
the Challenge-2023 evaluator only needs a small JSON interface:

``POST /detect`` with ``image_jpeg_base64`` and a one-class ``caption``;
``{detections: [{label, confidence, bbox_xyxy_normalized}]}`` in response.

The worker consumes only RGB frames supplied by the adapter.  It has no
Habitat import and no access to task goals beyond the public category prompt.
"""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from threading import Lock
import time
from typing import Any

import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    return parser.parse_args()


def _decode_rgb(value: object) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError("image_jpeg_base64 must be a base64 string")
    try:
        encoded = base64.b64decode(value, validate=True)
        return np.asarray(Image.open(BytesIO(encoded)).convert("RGB")).copy()
    except Exception as exc:  # pragma: no cover - guarded by HTTP error path
        raise ValueError(f"invalid JPEG payload: {exc}") from exc


class _DetectorApplication:
    """Own the model and serialize inference behind one small interface."""

    def __init__(self, box_threshold: float, text_threshold: float) -> None:
        from vlfm.vlm.grounding_dino import GroundingDINO

        self._model = GroundingDINO(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        self._lock = Lock()

    def health(self) -> dict[str, object]:
        return {"ready": True, "detector": "grounding_dino"}

    def detect(self, payload: dict[str, object]) -> dict[str, object]:
        caption = payload.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("caption must be one non-empty public ObjectGoal phrase")
        image = _decode_rgb(payload.get("image_jpeg_base64"))
        started = time.monotonic()
        # GroundingDINO mutates no request state, but one GPU inference at a
        # time makes latency deterministic and prevents concurrent OOMs.
        with self._lock:
            detections = self._model.predict(image, caption=caption).to_json()
        rows = []
        for box, confidence, phrase in zip(
            detections.get("boxes", []),
            detections.get("logits", []),
            detections.get("phrases", []),
        ):
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                normalized = [float(np.clip(float(value), 0.0, 1.0)) for value in box]
                score = float(confidence)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "label": str(phrase),
                    "confidence": score,
                    "bbox_xyxy_normalized": normalized,
                }
            )
        rows.sort(key=lambda item: (-float(item["confidence"]), item["bbox_xyxy_normalized"]))
        return {
            "detections": rows,
            "latency_s": time.monotonic() - started,
        }


def _handler(application: _DetectorApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HabitatGroundingDINO/1"

        def _write(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write(HTTPStatus.OK, application.health())
            else:
                self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 16 MiB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request payload must be a JSON object")
                self._write(HTTPStatus.OK, application.detect(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - protects the evaluator process
                self._write(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"detector failed: {exc}"})

        def log_message(self, _format: str, *_args: object) -> None:
            # The parent launch script owns an explicit file log under /home/ldl.
            return

    return Handler


def main() -> int:
    args = _parse_args()
    application = _DetectorApplication(args.box_threshold, args.text_threshold)
    server = ThreadingHTTPServer((args.host, args.port), _handler(application))
    print(f"GroundingDINO ready on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
