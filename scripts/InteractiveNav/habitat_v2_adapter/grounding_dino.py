"""Small external-adapter client for the loopback GroundingDINO worker."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GroundingDinoDetection:
    """One normalized public-RGB detection returned by the external worker."""

    label: str
    confidence: float
    bbox_xyxy_normalized: tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundingDinoResult:
    detections: tuple[GroundingDinoDetection, ...]
    error: str = ""
    latency_s: float = 0.0


class GroundingDinoClient:
    """Hide image transport, response validation, and evidence logging here."""

    def __init__(self, endpoint: str, timeout_s: float, metrics_path: str | None = None) -> None:
        self._endpoint = endpoint.rstrip("/") + "/detect"
        self._timeout_s = timeout_s
        self._metrics_path = Path(metrics_path) if metrics_path else None

    @staticmethod
    def _jpeg_base64(image: np.ndarray) -> str:
        rgb = np.asarray(image)[..., :3].astype(np.uint8)
        encoded = BytesIO()
        Image.fromarray(rgb, mode="RGB").save(encoded, format="JPEG", quality=88)
        return base64.b64encode(encoded.getvalue()).decode("ascii")

    def detect(self, image: np.ndarray, caption: str, context: dict[str, Any] | None = None) -> GroundingDinoResult:
        started = time.monotonic()
        error = ""
        detections: tuple[GroundingDinoDetection, ...] = ()
        try:
            payload = json.dumps(
                {
                    "image_jpeg_base64": self._jpeg_base64(image),
                    "caption": caption,
                }
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
                raise ValueError("detector response is not a JSON object")
            if body.get("error"):
                raise RuntimeError(str(body["error"]))
            rows = body.get("detections", [])
            if not isinstance(rows, list):
                raise ValueError("detector response detections is not a list")
            normalized = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                box = row.get("bbox_xyxy_normalized")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(np.clip(float(value), 0.0, 1.0)) for value in box)
                    normalized.append(
                        GroundingDinoDetection(
                            label=str(row.get("label") or ""),
                            confidence=float(row.get("confidence", 0.0)),
                            bbox_xyxy_normalized=(x1, y1, x2, y2),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            detections = tuple(sorted(normalized, key=lambda item: (-item.confidence, item.bbox_xyxy_normalized)))
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            error = str(exc)
        result = GroundingDinoResult(detections=detections, error=error, latency_s=time.monotonic() - started)
        self._record(caption, result, context or {})
        return result

    def _record(self, caption: str, result: GroundingDinoResult, context: dict[str, Any]) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "role": "grounding_dino_goal_localization",
            "caption": caption,
            "latency_s": result.latency_s,
            "detections": len(result.detections),
            "error": result.error,
            **context,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True)
class YoloV7Detection:
    """One public-RGB COCO box returned by the isolated VLFM YOLOv7 service."""

    label: str
    confidence: float
    bbox_xyxy_normalized: tuple[float, float, float, float]


@dataclass(frozen=True)
class YoloV7Result:
    detections: tuple[YoloV7Detection, ...]
    error: str = ""
    latency_s: float = 0.0


class YoloV7Client:
    """Adapt the existing VLFM YOLOv7 loopback API to public Habitat RGB only.

    The worker is intentionally optional.  It receives one RGB JPEG and returns
    only COCO boxes for the already-public ObjectGoal category; it has no Habitat
    import, semantic sensor, task geometry, or action API.
    """

    def __init__(self, endpoint: str, timeout_s: float, metrics_path: str | None = None) -> None:
        self._endpoint = endpoint.rstrip("/") + "/yolov7"
        self._timeout_s = timeout_s
        self._metrics_path = Path(metrics_path) if metrics_path else None

    @staticmethod
    def _jpeg_base64(image: np.ndarray) -> str:
        return GroundingDinoClient._jpeg_base64(image)

    def detect(
        self,
        image: np.ndarray,
        target_label: str,
        context: dict[str, Any] | None = None,
    ) -> YoloV7Result:
        started = time.monotonic()
        error = ""
        detections: tuple[YoloV7Detection, ...] = ()
        try:
            payload = json.dumps({"image": self._jpeg_base64(image)}).encode("utf-8")
            request = Request(
                self._endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self._timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("YOLOv7 response is not a JSON object")
            if body.get("error"):
                raise RuntimeError(str(body["error"]))
            boxes = body.get("boxes")
            logits = body.get("logits")
            phrases = body.get("phrases")
            if not isinstance(boxes, list) or not isinstance(logits, list) or not isinstance(phrases, list):
                raise ValueError("YOLOv7 response has invalid boxes/logits/phrases fields")
            height, width = np.asarray(image).shape[:2]
            parsed = []
            for box, confidence, label in zip(boxes, logits, phrases):
                if not isinstance(box, list) or len(box) != 4 or str(label).casefold() != target_label.casefold():
                    continue
                try:
                    values = [float(value) for value in box]
                    score = float(confidence)
                except (TypeError, ValueError):
                    continue
                if max(abs(value) for value in values) > 1.01:
                    values = [
                        values[0] / max(1, width),
                        values[1] / max(1, height),
                        values[2] / max(1, width),
                        values[3] / max(1, height),
                    ]
                x1, y1, x2, y2 = (float(np.clip(value, 0.0, 1.0)) for value in values)
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                parsed.append(YoloV7Detection(str(label), score, (x1, y1, x2, y2)))
            detections = tuple(sorted(parsed, key=lambda item: (-item.confidence, item.bbox_xyxy_normalized)))
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            error = str(exc)
        result = YoloV7Result(detections=detections, error=error, latency_s=time.monotonic() - started)
        self._record(target_label, result, context or {})
        return result

    def _record(self, target_label: str, result: YoloV7Result, context: dict[str, Any]) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "role": "yolov7_goal_localization",
            "target_label": target_label,
            "latency_s": result.latency_s,
            "detections": len(result.detections),
            "error": result.error,
            **context,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True)
class MobileSamResult:
    """A public-RGB segmentation mask returned by the isolated MobileSAM worker."""

    mask: np.ndarray | None
    error: str = ""
    latency_s: float = 0.0


class MobileSamClient:
    """Use MobileSAM only to refine public RGB-D detector measurements.

    This client intentionally has no Habitat import and returns no semantics or
    actions.  It receives the same RGB image and a detector-provided pixel box,
    then returns a same-size binary mask for public depth sampling.
    """

    def __init__(self, endpoint: str, timeout_s: float, metrics_path: str | None = None) -> None:
        self._endpoint = endpoint.rstrip("/") + "/mobile_sam"
        self._timeout_s = timeout_s
        self._metrics_path = Path(metrics_path) if metrics_path else None

    def segment_bbox(
        self,
        image: np.ndarray,
        bbox_xyxy_pixels: tuple[int, int, int, int],
        context: dict[str, Any] | None = None,
    ) -> MobileSamResult:
        started = time.monotonic()
        error = ""
        mask: np.ndarray | None = None
        try:
            height, width = np.asarray(image).shape[:2]
            payload = json.dumps(
                {
                    "image": GroundingDinoClient._jpeg_base64(image),
                    "bbox": [int(value) for value in bbox_xyxy_pixels],
                }
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
                raise ValueError("MobileSAM response is not a JSON object")
            if body.get("error"):
                raise RuntimeError(str(body["error"]))
            encoded = body.get("cropped_mask")
            if not isinstance(encoded, str):
                raise ValueError("MobileSAM response has no cropped_mask string")
            values = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            if values.size != height * width:
                raise ValueError("MobileSAM mask shape does not match the RGB image")
            mask = values.reshape((height, width)).astype(bool)
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            error = str(exc)
        result = MobileSamResult(mask=mask, error=error, latency_s=time.monotonic() - started)
        self._record(bbox_xyxy_pixels, result, context or {})
        return result

    def _record(
        self,
        bbox_xyxy_pixels: tuple[int, int, int, int],
        result: MobileSamResult,
        context: dict[str, Any],
    ) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "role": "mobile_sam_goal_segmentation",
            "bbox_xyxy_pixels": list(bbox_xyxy_pixels),
            "latency_s": result.latency_s,
            "mask_pixels": int(result.mask.sum()) if result.mask is not None else 0,
            "error": result.error,
            **context,
        }
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
