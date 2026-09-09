"""Loopback client for the existing VLFM PointNav local-controller worker.

The Habitat Challenge process deliberately has no VLFM/Torch dependency.  It
ships only public normalized depth plus a public relative waypoint derived from
the M2-selected map route.  The worker returns one of the released PointNav
checkpoint's four *local* actions.  The adapter remains solely responsible for
the official continuous-action conversion and never lets a worker ``STOP``
become an ObjectNav ``velocity_stop`` action.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
import zlib

import numpy as np


@dataclass(frozen=True)
class PointNavResult:
    """Validated local action from the isolated PointNav runtime."""

    action_index: int | None
    error: str = ""
    latency_s: float = 0.0


class PointNavClient:
    """Transport public depth observations without importing the worker stack."""

    def __init__(self, endpoint: str, timeout_s: float, metrics_path: str | None = None) -> None:
        self._endpoint = endpoint.rstrip("/") + "/act"
        self._timeout_s = timeout_s
        self._metrics_path = Path(metrics_path) if metrics_path else None

    @staticmethod
    def _encode_depth(depth: np.ndarray) -> tuple[str, list[int]]:
        """Serialize a finite public normalized depth image compactly and losslessly enough.

        The Challenge sensor uses a float32 image in roughly ``[0, 1]``.  Sending
        float16 after zlib keeps the loopback request small while preserving much
        more range resolution than the controller needs.  No semantic data, RGB,
        or task internals are included.
        """

        image = np.asarray(depth, dtype=np.float32)
        if image.ndim == 3:
            image = image[..., 0]
        if image.ndim != 2 or not image.size:
            raise ValueError("PointNav worker requires a non-empty HxW depth image")
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0).astype(np.float16, copy=False)
        compressed = zlib.compress(image.tobytes(order="C"), level=1)
        return base64.b64encode(compressed).decode("ascii"), [int(image.shape[0]), int(image.shape[1])]

    def act(
        self,
        depth: np.ndarray,
        rho: float,
        theta: float,
        *,
        reset: bool,
        context: dict[str, Any] | None = None,
    ) -> PointNavResult:
        """Request one recurrent PointNav decision from public local inputs."""

        started = time.monotonic()
        error = ""
        action_index: int | None = None
        try:
            encoded_depth, shape = self._encode_depth(depth)
            payload = json.dumps(
                {
                    "depth_float16_zlib_base64": encoded_depth,
                    "depth_shape": shape,
                    "rho": float(rho),
                    "theta": float(theta),
                    "reset": bool(reset),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            request = Request(
                self._endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self._timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("PointNav response is not a JSON object")
            if body.get("error"):
                raise RuntimeError(str(body["error"]))
            candidate = body.get("action_index")
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate not in {0, 1, 2, 3}:
                raise ValueError("PointNav response has an invalid action_index")
            action_index = candidate
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            error = str(exc)
        result = PointNavResult(action_index=action_index, error=error, latency_s=time.monotonic() - started)
        self._record(result, rho, theta, bool(reset), context or {})
        return result

    def _record(
        self,
        result: PointNavResult,
        rho: float,
        theta: float,
        reset: bool,
        context: dict[str, Any],
    ) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "role": "pointnav_local_control",
            "latency_s": result.latency_s,
            "rho_m": float(rho),
            "theta_rad": float(theta),
            "reset": reset,
            "action_index": result.action_index,
            "error": result.error,
            **context,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
