"""Detector-only bridge to the original InteractiveNav Module-1 backend.

Habitat has no ROS runtime on this host.  The bridge therefore uses a local
HTTP sidecar for transport, while invoking the original Module-1
``ExternalHttpProvider`` and ``ExternalHttpDetector`` classes for the actual
detector request/normalization boundary.  It is deliberately limited to public
RGB-D and camera intrinsics and returns only 2-D detector evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


@dataclass(frozen=True)
class FullRosStackResult:
    graph: dict[str, Any]
    candidate_payload: dict[str, Any]
    detections: dict[str, Any]
    candidate_sequence: int
    cmd_vel: dict[str, float] | None = None
    cmd_vel_age_s: float | None = None
    move_base_status: int = 0
    global_plan_xyyaw: tuple[tuple[float, float, float], ...] = ()
    local_plan_xyyaw: tuple[tuple[float, float, float], ...] = ()
    error: str = ""
    latency_s: float = 0.0


class FullRosStackClient:
    """Publish public Habitat state to the full navigation-only ROS stack."""

    def __init__(self, endpoint: str, timeout_s: float, metrics_path: str | None = None) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._metrics_path = Path(metrics_path) if metrics_path else None

    def health(self) -> dict[str, Any]:
        request = Request(self._endpoint + "/health", method="GET")
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError) as exc:
            raise RuntimeError(f"full ROS stack health check failed: {exc}") from exc
        if not isinstance(payload, dict) or not payload.get("ready"):
            raise RuntimeError("full ROS stack is not ready")
        if payload.get("mode") != "full_navigation_stack" or payload.get("module3_enabled") is not False:
            raise RuntimeError("full ROS stack identity/module3 boundary is invalid")
        return payload

    def publish_selection(self, payload: dict[str, Any]) -> str:
        """Mirror Habitat's already-made M2 choice to ROS observers.

        This endpoint is telemetry-only: the ROS behavior executor is absent and
        a failure must never change the Habitat control decision.
        """

        try:
            request = Request(
                self._endpoint + "/selection",
                data=json.dumps(dict(payload), separators=(",", ":")).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=min(self._timeout_s, 2.0)) as handle:
                response = json.loads(handle.read().decode("utf-8"))
            if not isinstance(response, dict) or response.get("error"):
                raise RuntimeError(str(response.get("error") or "invalid selection response"))
            return ""
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            return str(exc)

    def reset_navigation(self, episode: dict[str, Any] | None = None) -> str:
        """Reset only ROS mapping/navigation state between Habitat episodes."""

        try:
            request = Request(
                self._endpoint + "/reset",
                data=json.dumps(dict(episode or {}), separators=(",", ":")).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=min(self._timeout_s, 3.0)) as handle:
                response = json.loads(handle.read().decode("utf-8"))
            if not isinstance(response, dict) or response.get("error"):
                raise RuntimeError(str(response.get("error") or "invalid reset response"))
            return ""
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            return str(exc)

    def publish_recorder_diagnostic(self, image: np.ndarray, *, step: int) -> str:
        """Publish an evaluator-only panel and release the matching ROS frame.

        The diagnostic contains posthoc Habitat geometry and is sent to the
        recorder seam only.  The bridge never returns it to the policy or ROS
        semantic/candidate modules.
        """

        try:
            request = Request(
                self._endpoint + "/diagnostic",
                data=json.dumps(
                    {"image_b64": self._jpeg(image), "step": int(step)},
                    separators=(",", ":"),
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=min(self._timeout_s, 3.0)) as handle:
                response = json.loads(handle.read().decode("utf-8"))
            if not isinstance(response, dict) or response.get("error"):
                raise RuntimeError(str(response.get("error") or "invalid diagnostic response"))
            return ""
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            return str(exc)

    @staticmethod
    def _encoded_array(array: np.ndarray, dtype: np.dtype) -> dict[str, Any]:
        value = np.ascontiguousarray(np.asarray(array, dtype=dtype))
        return {
            "shape": list(value.shape),
            "data_b64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }

    @staticmethod
    def _jpeg(image: np.ndarray) -> str:
        import cv2

        rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[..., :3])
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("could not encode Habitat RGB frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def step(
        self,
        *,
        image: np.ndarray,
        depth_m: np.ndarray,
        gps_xy: np.ndarray,
        compass: float,
        object_category: str,
        occupancy_ros: np.ndarray,
        raw_occupancy_ros: np.ndarray | None = None,
        global_plan_xyyaw_ros: list[list[float]] | None = None,
        local_plan_xyyaw_ros: list[list[float]] | None = None,
        map_resolution_m: float,
        map_origin_xy: tuple[float, float],
        hfov_degrees: float,
        context: dict[str, Any] | None = None,
    ) -> FullRosStackResult:
        started = time.monotonic()
        height, width = np.asarray(image).shape[:2]
        focal = (width / 2.0) / np.tan(np.deg2rad(float(hfov_degrees)) / 2.0)
        payload = {
            "image_b64": self._jpeg(image),
            "depth": self._encoded_array(depth_m, np.float32),
            "gps": [float(gps_xy[0]), float(gps_xy[1])],
            "compass": float(compass),
            "object_category": str(object_category),
            # The original candidate generator matches semantic labels.  Pass
            # public category aliases so Habitat's couch/television/potted
            # plant names align with YOLOE's sofa/tv/plant graph labels.
            "object_labels": sorted(goal_label_aliases(object_category)),
            "occupancy": self._encoded_array(occupancy_ros, np.int8),
            "raw_occupancy": self._encoded_array(
                occupancy_ros if raw_occupancy_ros is None else raw_occupancy_ros,
                np.int8,
            ),
            "global_plan_xyyaw": list(global_plan_xyyaw_ros or []),
            "local_plan_xyyaw": list(local_plan_xyyaw_ros or []),
            "map_resolution_m": float(map_resolution_m),
            "map_origin_xy": [float(map_origin_xy[0]), float(map_origin_xy[1])],
            "camera_info": {
                "width": int(width),
                "height": int(height),
                "K": [float(focal), 0.0, width / 2.0, 0.0, float(focal), height / 2.0, 0.0, 0.0, 1.0],
            },
            "context": dict(context or {}),
        }
        error = ""
        response: dict[str, Any] = {}
        try:
            request = Request(
                self._endpoint + "/step",
                data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self._timeout_s) as handle:
                response = json.loads(handle.read().decode("utf-8"))
            if not isinstance(response, dict):
                raise RuntimeError("full ROS stack returned a non-object response")
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            if isinstance(exc, HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except OSError:
                    detail = ""
                error = f"{exc}: {detail}" if detail else str(exc)
            else:
                error = str(exc)
            response = {}
        result = FullRosStackResult(
            graph=dict(response.get("graph") or {}),
            candidate_payload=dict(response.get("candidate_payload") or {}),
            detections=dict(response.get("detections") or {}),
            candidate_sequence=int(response.get("candidate_sequence", -1) or -1),
            cmd_vel=(dict(response["cmd_vel"]) if isinstance(response.get("cmd_vel"), dict) else None),
            cmd_vel_age_s=(
                float(response["cmd_vel_age_s"])
                if response.get("cmd_vel_age_s") is not None
                else None
            ),
            move_base_status=int(response.get("move_base_status", 0) or 0),
            global_plan_xyyaw=tuple(
                tuple(float(value) for value in row[:3])
                for row in (response.get("global_plan_xyyaw") or [])
                if isinstance(row, list) and len(row) >= 3
            ),
            local_plan_xyyaw=tuple(
                tuple(float(value) for value in row[:3])
                for row in (response.get("local_plan_xyyaw") or [])
                if isinstance(row, list) and len(row) >= 3
            ),
            error=error,
            latency_s=time.monotonic() - started,
        )
        if self._metrics_path is not None:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "role": "full_ros_navigation_stack",
                "latency_s": result.latency_s,
                "error": result.error,
                "candidate_sequence": result.candidate_sequence,
                "candidate_count": len(result.candidate_payload.get("candidates") or []),
                "graph_nodes": len(result.graph.get("nodes") or []),
                "graph_edges": len(result.graph.get("edges") or []),
                **(context or {}),
            }
            with self._metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return result


_FORBIDDEN_RESPONSE_FIELDS = {
    "interaction_command",
    "interaction",
    "behavior_type",
    "action",
    "goal_position",
    "goal_positions",
    "view_points",
    "viewpoint",
    "world_position",
    "world_box3d_center",
    "world_box3d_size",
    "semantic_observation",
    "metric",
    "pathfinder",
}
_ALLOWED_RESPONSE_FIELDS = {
    "semantic_class",
    "semantic_class_raw",
    "confidence",
    "bbox",
    "projection_method",
    "source_frame",
    "source_model",
    "mask",
    "mask_area",
}


@dataclass(frozen=True)
class Module1Detection:
    """One normalized public detector box from the original Module-1 seam."""

    label: str
    raw_label: str
    confidence: float
    bbox_xyxy_normalized: tuple[float, float, float, float]
    source_model: str


@dataclass(frozen=True)
class Module1DetectionResult:
    detections: tuple[Module1Detection, ...]
    error: str = ""
    latency_s: float = 0.0


@dataclass(frozen=True)
class _Stamp:
    secs: int
    nsecs: int


@dataclass(frozen=True)
class _CameraInfo:
    width: int
    height: int
    K: tuple[float, ...]


class Module1DetectorSidecarClient:
    """Use original Module-1 provider classes over a strict local sidecar.

    The sidecar's HTTP protocol exactly matches the original
    ``ExternalHttpProvider`` contract.  The adapter owns depth-to-meters and all
    GPS/Compass projection; no task internals, TF, semantic sensor, or action
    channel crosses this boundary.
    """

    def __init__(
        self,
        endpoint: str,
        timeout_s: float,
        include_depth: bool,
        original_module1_scripts: str,
        metrics_path: str | None = None,
    ) -> None:
        self._base_endpoint = endpoint.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._include_depth = bool(include_depth)
        self._original_module1_scripts = Path(original_module1_scripts).expanduser().resolve()
        self._metrics_path = Path(metrics_path) if metrics_path else None
        self._detector: Any | None = None

    def health(self) -> dict[str, Any]:
        """Read sidecar identity before an evaluation claims Module-1 integration."""

        request = Request(self._base_endpoint + "/health", method="GET")
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, ValueError) as exc:
            raise RuntimeError(f"Module-1 sidecar health check failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Module-1 sidecar health response is not a JSON object")
        if not payload.get("ready") or payload.get("mode") != "detector_only":
            raise RuntimeError("Module-1 sidecar is not ready in detector_only mode")
        if payload.get("module3_enabled") is not False:
            raise RuntimeError("Module-1 sidecar must report module3_enabled=false")
        return payload

    def detect(
        self,
        image: np.ndarray,
        depth_m: np.ndarray | None,
        hfov_degrees: float,
        *,
        stamp_index: int,
        context: dict[str, Any] | None = None,
    ) -> Module1DetectionResult:
        started = time.monotonic()
        error = ""
        detections: tuple[Module1Detection, ...] = ()
        try:
            detector = self._load_original_detector()
            rgb = np.asarray(image)
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise ValueError("Module-1 bridge requires an RGB image")
            height, width = rgb.shape[:2]
            camera_info = self._camera_info(width, height, hfov_degrees)
            stamp = _Stamp(secs=int(stamp_index), nsecs=0)
            rows = detector.detect(
                np.ascontiguousarray(rgb[..., :3].astype(np.uint8)),
                np.ascontiguousarray(depth_m.astype(np.float32)) if self._include_depth and depth_m is not None else None,
                camera_info,
                stamp,
                "habitat_v2_camera",
            )
            detections = self._normalize(rows, width, height)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            error = str(exc)
        result = Module1DetectionResult(
            detections=detections,
            error=error,
            latency_s=time.monotonic() - started,
        )
        self._record(result, context or {})
        return result

    def _load_original_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        if not self._original_module1_scripts.is_dir():
            raise RuntimeError(f"original Module-1 scripts directory is missing: {self._original_module1_scripts}")
        scripts = str(self._original_module1_scripts)
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        try:
            from semantic_mapping_py_pkg.detector_backends import ExternalHttpDetector, ExternalHttpProvider
        except ImportError as exc:
            raise RuntimeError("could not import original Module-1 detector backend") from exc
        provider = ExternalHttpProvider(
            self._base_endpoint + "/detect",
            timeout=self._timeout_s,
            include_depth=self._include_depth,
        )
        self._detector = ExternalHttpDetector(provider)
        return self._detector

    @staticmethod
    def _camera_info(width: int, height: int, hfov_degrees: float) -> _CameraInfo:
        focal = (width / 2.0) / np.tan(np.deg2rad(float(hfov_degrees)) / 2.0)
        return _CameraInfo(
            width=int(width),
            height=int(height),
            K=(float(focal), 0.0, width / 2.0, 0.0, float(focal), height / 2.0, 0.0, 0.0, 1.0),
        )

    @staticmethod
    def _normalize(rows: Any, width: int, height: int) -> tuple[Module1Detection, ...]:
        if not isinstance(rows, list):
            raise ValueError("original Module-1 detector returned a non-list result")
        normalized: list[Module1Detection] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            forbidden = sorted(_FORBIDDEN_RESPONSE_FIELDS & set(row))
            if forbidden:
                raise RuntimeError(f"Module-1 sidecar returned forbidden field(s): {', '.join(forbidden)}")
            unknown = sorted(set(row) - _ALLOWED_RESPONSE_FIELDS)
            if unknown:
                raise RuntimeError(f"Module-1 sidecar returned unsupported field(s): {', '.join(unknown)}")
            bbox = row.get("bbox")
            label = str(row.get("semantic_class") or "").strip()
            raw_label = str(row.get("semantic_class_raw") or label).strip()
            if not label or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in bbox)
                confidence = float(row.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((float(np.clip(x1 / max(1, width), 0.0, 1.0)), float(np.clip(x2 / max(1, width), 0.0, 1.0))))
            y1, y2 = sorted((float(np.clip(y1 / max(1, height), 0.0, 1.0)), float(np.clip(y2 / max(1, height), 0.0, 1.0))))
            if x2 <= x1 or y2 <= y1:
                continue
            normalized.append(
                Module1Detection(
                    label=label.casefold().replace(" ", "_"),
                    raw_label=raw_label.casefold().replace(" ", "_"),
                    confidence=confidence,
                    bbox_xyxy_normalized=(x1, y1, x2, y2),
                    source_model=str(row.get("source_model") or "module1_detector"),
                )
            )
        return tuple(sorted(normalized, key=lambda item: (-item.confidence, item.bbox_xyxy_normalized, item.label)))

    def _record(self, result: Module1DetectionResult, context: dict[str, Any]) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "role": "module1_detector_sidecar",
            "latency_s": result.latency_s,
            "detections": len(result.detections),
            # Keep enough public 2-D evidence to audit the concrete detector
            # checkpoint/labels used by a run.  Do not persist masks, 3-D
            # positions, simulator state, or any task metric here.
            "detections_2d": [
                {
                    "label": item.label,
                    "raw_label": item.raw_label,
                    "confidence": item.confidence,
                    "bbox_xyxy_normalized": list(item.bbox_xyxy_normalized),
                    "source_model": item.source_model,
                }
                for item in result.detections
            ],
            "error": result.error,
            **context,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def goal_label_aliases(goal: str) -> set[str]:
    """Normalize public v2 labels and detector labels without dataset internals."""

    normalized = str(goal or "").casefold().strip().replace(" ", "_")
    aliases = {
        "chair": {"chair"},
        "bed": {"bed"},
        "toilet": {"toilet"},
        "plant": {"plant", "potted_plant"},
        "potted_plant": {"plant", "potted_plant"},
        "sofa": {"sofa", "couch"},
        "couch": {"sofa", "couch"},
        "tv": {"tv", "television", "tv_monitor", "monitor"},
        "television": {"tv", "television", "tv_monitor", "monitor"},
        "tv_monitor": {"tv", "television", "tv_monitor", "monitor"},
    }
    return aliases.get(normalized, {normalized})
