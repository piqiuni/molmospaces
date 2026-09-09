#!/usr/bin/env python3
"""Loopback-only worker for the already-present VLFM PointNav checkpoint.

This process is intentionally independent of Habitat Challenge 2023.  It owns
the incompatible Torch/VLFM runtime and exposes just one local control seam:

``POST /act`` with a public normalized depth image and public ``rho, theta``
for an M2-selected route waypoint; ``{action_index: 0..3}`` in response.

The released checkpoint was trained with discrete PointNav actions in the order
``STOP, FORWARD, LEFT, RIGHT``.  Its STOP prediction is *not* an ObjectNav
success signal and is converted to a harmless local fallback by ``policy.py``.
This worker has no Habitat import, no semantic input, and no task-goal geometry.
"""

from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock
import time
from typing import Any
import zlib

import numpy as np


_ACTION_NAMES = ("stop", "forward", "left", "right")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12183)
    parser.add_argument(
        "--weights",
        default="/home/ldl/molmospaces-exp-compare/external_methods/vlfm/data/pointnav_weights.pth",
        help="existing released VLFM PointNav checkpoint; no download is performed",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _decode_depth(payload: dict[str, object]) -> np.ndarray:
    encoded = payload.get("depth_float16_zlib_base64")
    shape = payload.get("depth_shape")
    if not isinstance(encoded, str):
        raise ValueError("depth_float16_zlib_base64 must be a base64 string")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("depth_shape must be [height, width]")
    try:
        height, width = (int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth_shape values must be integers") from exc
    if not (1 <= height <= 2048 and 1 <= width <= 2048):
        raise ValueError("depth_shape is outside the supported public-sensor range")
    try:
        decoded = zlib.decompress(base64.b64decode(encoded, validate=True))
    except Exception as exc:  # pragma: no cover - exercised through HTTP errors
        raise ValueError(f"invalid compressed depth payload: {exc}") from exc
    expected_bytes = height * width * np.dtype(np.float16).itemsize
    if len(decoded) != expected_bytes:
        raise ValueError("depth payload byte count does not match depth_shape")
    depth = np.frombuffer(decoded, dtype=np.float16).astype(np.float32).reshape(height, width)
    return np.clip(np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


class _PointNavApplication:
    """Load one recurrent discrete PointNav model and serialize its inference."""

    def __init__(self, weights: str, device: str) -> None:
        import torch
        import torch.nn as nn

        from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import PointNavResNetNet
        from vlfm.policy.utils.pointnav_policy import _load_checkpoint_with_config_type_shims

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is unavailable in the VLFM runtime")
        checkpoint = _load_checkpoint_with_config_type_shims(weights)
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise RuntimeError("PointNav checkpoint has no state dictionary")

        self._net = PointNavResNetNet(discrete_actions=True, no_fwd_dict=True)
        network_state = {
            key.removeprefix("net.").replace("prev_action_embedding.", "prev_action_embedding_discrete."): value
            for key, value in state.items()
            if str(key).startswith("net.")
        }
        incompatible = self._net.load_state_dict(network_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "PointNav network/checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self._head = nn.Linear(512, 4)
        try:
            self._head.load_state_dict(
                {
                    "weight": state["action_distribution.linear.weight"],
                    "bias": state["action_distribution.linear.bias"],
                },
                strict=True,
            )
        except KeyError as exc:
            raise RuntimeError("PointNav checkpoint has no categorical four-action head") from exc
        self._net.to(self._device).eval()
        self._head.to(self._device).eval()
        self._hidden = None
        self._previous_action = None
        self._first_step = True
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        self._hidden = self._torch.zeros(
            1,
            self._net.num_recurrent_layers,
            512,
            device=self._device,
            dtype=self._torch.float32,
        )
        self._previous_action = self._torch.zeros(1, 1, device=self._device, dtype=self._torch.long)
        self._first_step = True

    def health(self) -> dict[str, object]:
        return {
            "ready": True,
            "controller": "vlfm_pointnav_resnet_discrete",
            "actions": list(_ACTION_NAMES),
            "device": str(self._device),
        }

    def act(self, payload: dict[str, object]) -> dict[str, object]:
        depth = _decode_depth(payload)
        try:
            rho = float(payload["rho"])
            theta = float(payload["theta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("rho and theta must be finite numbers") from exc
        if not (np.isfinite(rho) and np.isfinite(theta) and 0.0 <= rho <= 100.0 and abs(theta) <= np.pi + 1e-3):
            raise ValueError("rho/theta outside the public local-goal range")
        started = time.monotonic()
        with self._lock, self._torch.inference_mode():
            if bool(payload.get("reset", False)):
                self.reset()
            # The released discrete controller's visual-FC expects a 4x4 spatial
            # tensor.  Center-cropping the 4:3 v2 depth image to square before a
            # 224x224 resize preserves the central forward view rather than
            # stretching its geometry; only public depth values are transformed.
            image = self._torch.from_numpy(depth).to(self._device).unsqueeze(0).unsqueeze(0)
            height, width = image.shape[-2:]
            if height > width:
                offset = (height - width) // 2
                image = image[:, :, offset : offset + width, :]
            elif width > height:
                offset = (width - height) // 2
                image = image[:, :, :, offset : offset + height]
            image = self._torch.nn.functional.interpolate(
                image, size=(224, 224), mode="bilinear", align_corners=False
            )
            observations = {
                "depth": image.permute(0, 2, 3, 1),
                "pointgoal_with_gps_compass": self._torch.tensor([[rho, theta]], device=self._device),
            }
            masks = self._torch.tensor([[not self._first_step]], device=self._device, dtype=self._torch.bool)
            features, self._hidden = self._net(observations, self._hidden, self._previous_action, masks)
            action_index = int(self._head(features).argmax(dim=-1).item())
            self._previous_action = self._torch.tensor([[action_index]], device=self._device, dtype=self._torch.long)
            self._first_step = False
            if self._device.type == "cuda":
                self._torch.cuda.synchronize(self._device)
        return {
            "action_index": action_index,
            "action_name": _ACTION_NAMES[action_index],
            "latency_s": time.monotonic() - started,
        }


def _handler(application: _PointNavApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HabitatPointNav/1"

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
            if self.path != "/act":
                self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 3 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 3 MiB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request payload must be a JSON object")
                self._write(HTTPStatus.OK, application.act(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - protects evaluator process
                self._write(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"PointNav worker failed: {exc}"})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> int:
    args = _parse_args()
    application = _PointNavApplication(args.weights, args.device)
    server = ThreadingHTTPServer((args.host, args.port), _handler(application))
    print(f"PointNav worker ready on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
