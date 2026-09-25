#!/usr/bin/env python3
"""YOLOE-26l PF Seg worker for the physical platform.

This worker runs in the algorithm Python environment and consumes the
latest decoded D435i RGB-D pair directly from ROS.  An encoded HTTP/WebSocket
fallback is retained for replay compatibility only.  It runs YOLOE, lifts
masks to camera/base/map-frame geometry, and publishes the authoritative
detection report on ROS; the optional web mirror is presentation-only.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import math
from pathlib import Path
import sys
import threading
import time
import urllib.request
from typing import Any

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

# Keep the detector's embedded fallback geometry on the exact same camera-IMU
# convention as the direct ROS bridge.  The direct bridge is the authority for
# ``tf_frame_base_link -> d435i_*_optical_frame``; importing its small pure
# quaternion helpers avoids re-implementing the sensor-axis conjugation here.
# (The module has no node-start side effects at import time.)
from physical_sensor_ros_bridge import (  # noqa: E402
    _camera_imu_parent_correction_quaternion,
    _coerce_bool as _sensor_coerce_bool,
    _quat_multiply as _sensor_quat_multiply,
    _quat_rpy as _sensor_quat_rpy,
    _telemetry_body_position,
    _telemetry_body_quaternion,
    _nearest_capture_telemetry,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEMANTIC_MAPPING_SCRIPTS = (
    _REPO_ROOT
    / "Interactive-Nav-SG-nav"
    / "src"
    / "semantic_mapping_py_pkg"
    / "scripts"
)
if str(_SEMANTIC_MAPPING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_MAPPING_SCRIPTS))

from semantic_mapping_py_pkg.detection_filter import DetectionFilter, load_detection_filter_config


DEFAULT_DETECTOR_CONFIG = Path(__file__).resolve().parent / "config" / "physical_nav.yaml"


def _decode(value: str | dict[str, Any]) -> np.ndarray:
    # Legacy HTTP/replay frames wrap the base64 payload in an
    # ``{encoding,data}`` object; direct sensor frames use the string.
    if isinstance(value, dict):
        value = value.get("data", "")
    encoded = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError("cannot decode gateway frame")
    return image


def _mask_grid_shape(mask_source: Any) -> tuple[int, int] | None:
    """Read a segmentation tensor's spatial shape without copying its data."""

    try:
        shape = tuple(int(value) for value in getattr(mask_source, "shape"))
    except (AttributeError, TypeError, ValueError):
        try:
            shape = tuple(int(value) for value in np.asarray(mask_source).shape)
        except (TypeError, ValueError):
            return None
    if len(shape) < 2:
        return None
    return shape[-2], shape[-1]


def _materialize_selected_masks(
    mask_source: Any, indices: list[int]
) -> dict[int, np.ndarray]:
    """Copy only masks selected for RGB-D lifting from a model result.

    Ultralytics keeps ``masks.data`` on the GPU.  Converting the complete
    ``N x H x W`` tensor to a CPU array before confidence/class filtering was
    the largest avoidable transfer in crowded frames.  Indexing first keeps
    the common path bounded by the geometry budget; the full-copy fallback is
    retained for older tensor wrappers that do not support list indexing.
    """

    if mask_source is None or not indices:
        return {}
    unique_indices = list(dict.fromkeys(int(index) for index in indices if int(index) >= 0))
    if not unique_indices:
        return {}
    selected: Any
    try:
        if hasattr(mask_source, "detach"):
            selected = mask_source[unique_indices]
            selected = selected.detach().cpu().numpy()
        else:
            selected = np.asarray(mask_source)[unique_indices]
    except (TypeError, IndexError, AttributeError, ValueError):
        # Some torch/NumPy compatibility wrappers reject a Python list as an
        # index.  This path is less efficient but preserves old model support.
        try:
            if hasattr(mask_source, "detach"):
                selected = mask_source.detach().cpu().numpy()[unique_indices]
            else:
                selected = np.asarray(mask_source)[unique_indices]
        except (TypeError, IndexError, AttributeError, ValueError):
            return {}
    selected = np.asarray(selected)
    if selected.ndim == 4 and selected.shape[1] == 1:
        selected = selected[:, 0]
    if selected.ndim != 3:
        return {}
    return {
        source_index: selected[position]
        for position, source_index in enumerate(unique_indices)
        if position < selected.shape[0]
    }


def _project_depth_grid(
    depth_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    rgb_intr: dict[str, Any],
    depth_intr: dict[str, Any],
    extr: dict[str, Any],
    *,
    depth_values: np.ndarray | None = None,
    depth_scale: float = 1.0,
    projection_roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project every native-depth pixel into the detector mask grid.

    This is intentionally written as scalar affine combinations rather than
    constructing an ``H x W x 3`` point tensor and multiplying it by a 3x3
    matrix.  At 848x480 the latter allocates several large temporaries for
    every RGB-D receipt; the fused form produces identical integer pixels with
    substantially lower CPU/cache traffic.
    """
    height, width = depth_shape
    full_shape = (int(height), int(width))
    cropped = projection_roi is not None
    if projection_roi is None:
        roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, int(width), int(height)
    else:
        try:
            roi_x0, roi_y0, roi_x1, roi_y1 = (
                int(value) for value in projection_roi
            )
        except (TypeError, ValueError):
            roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, int(width), int(height)
            cropped = False
        roi_x0 = max(0, min(int(width), roi_x0))
        roi_x1 = max(roi_x0, min(int(width), roi_x1))
        roi_y0 = max(0, min(int(height), roi_y0))
        roi_y1 = max(roi_y0, min(int(height), roi_y1))
        if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
            roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, int(width), int(height)
            cropped = False
    roi_shape = (roi_y1 - roi_y0, roi_x1 - roi_x0)
    if depth_values is None:
        z = np.ones(roi_shape, dtype=np.float32)
    else:
        values = np.asarray(depth_values)
        if values.shape != full_shape:
            raise ValueError(f"depth projection shape mismatch: {values.shape} != {full_shape}")
        # Slice before converting/scaling.  With align(depth), the calibrated
        # colour/depth baseline makes this map depth-dependent; restricting
        # the conversion to the union of detector ROIs avoids touching the
        # complete 848x480 depth canvas for every receipt.
        z = np.asarray(values[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
        if depth_scale != 1.0:
            z = z * float(depth_scale)
    # A one-dimensional coordinate grid broadcasts against the depth plane
    # without materialising a second full mesh-grid copy.
    u_native = np.arange(roi_x0, roi_x1, dtype=np.float32)[None, :]
    v_native = np.arange(roi_y0, roi_y1, dtype=np.float32)[:, None]
    depth_fx = max(float(depth_intr.get("fx", 1)), 1e-6)
    depth_fy = max(float(depth_intr.get("fy", 1)), 1e-6)
    x = (u_native - float(depth_intr.get("cx", 0))) * z / depth_fx
    y = (v_native - float(depth_intr.get("cy", 0))) * z / depth_fy
    rotation_value = extr.get("rotation")
    if rotation_value is None:
        rotation_value = np.eye(3).reshape(-1)
    translation_value = extr.get("translation")
    if translation_value is None:
        translation_value = [0, 0, 0]
    rotation = np.asarray(rotation_value, dtype=np.float32).reshape(3, 3)
    translation = np.asarray(translation_value, dtype=np.float32).reshape(-1)
    tx = float(translation[0]) if translation.size >= 1 else 0.0
    ty = float(translation[1]) if translation.size >= 2 else 0.0
    tz = float(translation[2]) if translation.size >= 3 else 0.0
    color_x = (
        rotation[0, 0] * x
        + rotation[0, 1] * y
        + rotation[0, 2] * z
        + tx
    )
    color_y = (
        rotation[1, 0] * x
        + rotation[1, 1] * y
        + rotation[1, 2] * z
        + ty
    )
    color_z = (
        rotation[2, 0] * x
        + rotation[2, 1] * y
        + rotation[2, 2] * z
        + tz
    )
    denominator = np.maximum(color_z, 1e-6)
    rgb_width = max(float(rgb_intr.get("width", mask_shape[1])), 1.0)
    rgb_height = max(float(rgb_intr.get("height", mask_shape[0])), 1.0)
    mask_scale_x = mask_shape[1] / rgb_width
    mask_scale_y = mask_shape[0] / rgb_height
    projected_u = (
        color_x * float(rgb_intr.get("fx", 1)) / denominator
        + float(rgb_intr.get("cx", 0))
    ) * mask_scale_x
    projected_v = (
        color_y * float(rgb_intr.get("fy", 1)) / denominator
        + float(rgb_intr.get("cy", 0))
    ) * mask_scale_y
    u = np.rint(projected_u).astype(np.int32)
    v = np.rint(projected_v).astype(np.int32)
    inside = (
        (color_z > 0)
        & (u >= 0)
        & (u < mask_shape[1])
        & (v >= 0)
        & (v < mask_shape[0])
    )
    if not cropped:
        return u, v, inside
    # Keep the historical full-shaped return contract so callers and replay
    # fixtures need no special handling.  Pixels outside the conservative
    # union ROI are marked ``inside=False`` and can never be selected by the
    # per-box mask gate in ``_rgb_mask_to_depth``.
    full_u = np.zeros(full_shape, dtype=np.int32)
    full_v = np.zeros(full_shape, dtype=np.int32)
    full_inside = np.zeros(full_shape, dtype=bool)
    full_u[roi_y0:roi_y1, roi_x0:roi_x1] = u
    full_v[roi_y0:roi_y1, roi_x0:roi_x1] = v
    full_inside[roi_y0:roi_y1, roi_x0:roi_x1] = inside
    return full_u, full_v, full_inside


def _depth_projection_roi(
    boxes: np.ndarray,
    depth_shape: tuple[int, int],
    rgb_intr: dict[str, Any],
    *,
    margin: int = 96,
) -> tuple[int, int, int, int] | None:
    """Return a conservative depth ROI covering all RGB detection boxes.

    The exact per-instance gate in :func:`_rgb_mask_to_depth` uses the same
    scale and margin.  Computing a union here therefore changes no selected
    pixels; it only prevents the depth-dependent projection from evaluating
    pixels that no detector mask can consume.
    """
    values = np.asarray(boxes)
    if values.ndim != 2 or values.shape[1] < 4 or values.shape[0] == 0:
        return None
    height, width = int(depth_shape[0]), int(depth_shape[1])
    rgb_width = max(float(rgb_intr.get("width", values.shape[1])), 1.0)
    rgb_height = max(float(rgb_intr.get("height", values.shape[0])), 1.0)
    valid = values[:, :4].astype(np.float64, copy=False)
    # If one detector box is malformed, retain the old full-grid behavior
    # rather than silently dropping the pixels that its legacy per-box ROI
    # would have visited.
    if not np.isfinite(valid).all():
        return None
    left = float(np.min(valid[:, 0]))
    top = float(np.min(valid[:, 1]))
    right = float(np.max(valid[:, 2]))
    bottom = float(np.max(valid[:, 3]))
    x0 = max(0, int(math.floor(left * width / rgb_width)) - int(margin))
    x1 = min(width, int(math.ceil((right + 1.0) * width / rgb_width)) + int(margin))
    y0 = max(0, int(math.floor(top * height / rgb_height)) - int(margin))
    y1 = min(height, int(math.ceil((bottom + 1.0) * height / rgb_height)) + int(margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _rgb_mask_to_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    rgb_intr: dict[str, Any],
    depth_intr: dict[str, Any],
    extr: dict[str, Any],
    projection_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    depth_scale: float = 1.0,
    *,
    aligned_stream: bool = False,
) -> np.ndarray:
    """Project an RGB YOLO mask onto the native/aligned depth pixel grid."""
    if mask.shape == depth.shape and aligned_stream:
        # An explicitly aligned stream already places depth on the detector's
        # pixel grid.  Do not infer alignment from equal dimensions: an
        # ``align_to=none`` D435i profile can expose equal-sized RGB/depth
        # images with different optical centres and a non-zero baseline.
        # Projecting that stream is required to avoid a silent pixel mismatch.
        return mask.astype(bool)
    h, w = depth.shape[:2]
    if projection_maps is None:
        u, v, inside = _project_depth_grid(
            (h, w),
            tuple(mask.shape[:2]),
            rgb_intr,
            depth_intr,
            extr,
            depth_values=depth,
            depth_scale=depth_scale,
        )
    else:
        u, v, inside = projection_maps
    result = np.zeros((h, w), dtype=bool)
    # A detector mask is bounded by its box.  Restrict the expensive boolean
    # indexing to a conservative depth ROI instead of scanning the complete
    # 848x480 grid for every instance.  The exact projected-pixel test below
    # remains authoritative, so the margin only affects speed, not geometry.
    if bbox is None:
        y0, y1_depth, x0, x1_depth = 0, h, 0, w
        mask_x1 = mask.shape[1]
        mask_y1 = mask.shape[0]
    else:
        rgb_width = max(float(rgb_intr.get("width", mask.shape[1])), 1.0)
        rgb_height = max(float(rgb_intr.get("height", mask.shape[0])), 1.0)
        x1_rgb, y1_rgb, x2_rgb, y2_rgb = (
            float(value) for value in bbox
        )
        mask_scale_x = mask.shape[1] / rgb_width
        mask_scale_y = mask.shape[0] / rgb_height
        mask_x1 = int(x1_rgb * mask_scale_x)
        mask_y1 = int(y1_rgb * mask_scale_y)
        mask_x2 = int(x2_rgb * mask_scale_x) + 1
        mask_y2 = int(y2_rgb * mask_scale_y) + 1
        # RGB/depth optical centres and FOVs are not guaranteed to match.
        # A 32 px depth margin was fast but could clip a valid projected
        # surface on the D435i edge.  Keep the ROI optimization while using a
        # conservative margin that covers the calibrated baseline at 848x480;
        # the exact projected-pixel test below still rejects out-of-box depth.
        margin = 96
        x0 = max(0, int(math.floor(x1_rgb * w / rgb_width)) - margin)
        x1_depth = min(w, int(math.ceil((x2_rgb + 1.0) * w / rgb_width)) + margin)
        y0 = max(0, int(math.floor(y1_rgb * h / rgb_height)) - margin)
        y1_depth = min(h, int(math.ceil((y2_rgb + 1.0) * h / rgb_height)) + margin)
    if x1_depth <= x0 or y1_depth <= y0:
        return result
    roi_u = u[y0:y1_depth, x0:x1_depth]
    roi_v = v[y0:y1_depth, x0:x1_depth]
    roi_inside = inside[y0:y1_depth, x0:x1_depth] & (depth[y0:y1_depth, x0:x1_depth] > 0)
    if bbox is not None:
        roi_inside &= (
            (roi_u >= mask_x1)
            & (roi_u < mask_x2)
            & (roi_v >= mask_y1)
            & (roi_v < mask_y2)
        )
    roi_result = result[y0:y1_depth, x0:x1_depth]
    if np.any(roi_inside):
        # ``inside`` produced by _project_depth_grid already guarantees the
        # bounds.  Clip only selected pixels for compatibility with replay
        # callers that provide hand-built projection maps; clipping the full
        # ROI allocated two large arrays per detection.
        selected_u = np.clip(roi_u[roi_inside], 0, mask.shape[1] - 1)
        selected_v = np.clip(roi_v[roi_inside], 0, mask.shape[0] - 1)
        roi_result[roi_inside] = mask[selected_v, selected_u] > 0
    return result


def _post(url: str, value: Any) -> None:
    data = json.dumps({"name": "detections", "value": value}, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/ros-state", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=.6):
        pass


_WEB_DETECTION_KEYS = frozenset({
    # The browser overlay and status cards only need compact 2-D/box fields.
    # Masks and packed segment points remain on the authoritative ROS report;
    # sending them through the optional HTTP mirror made JSON serialization and
    # RuntimeState copies contend with the 10-Hz detector lane.
    "semantic_class", "semantic_class_raw", "raw_class", "possible_interaction_class",
    "m1_verification_required", "confidence", "bbox", "bbox_2d", "mask_area",
    "depth_median_m", "depth_valid_points", "camera_position", "camera_box3d_center",
    "camera_box3d_size", "camera_obb_center", "camera_obb_size",
    "camera_obb_orientation", "position", "world_position", "aabb_center", "aabb_size",
    "box3d_center", "box3d_size", "world_box3d_center", "world_box3d_size",
    "world_box3d_marker_size", "world_box3d_orientation", "world_box3d_yaw", "yaw",
    "source_frame", "projection_method", "map_transform_status", "capture_seq",
    "stamp", "geometry_skipped",
})


def _compact_web_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build the presentation-only report without masks/point-cloud payloads."""
    if not isinstance(report, dict):
        return {}
    compact = {
        key: value
        for key, value in report.items()
        if key not in {"overlay_jpeg", "detections"}
    }
    compact["detections"] = [
        {
            key: item[key]
            for key in _WEB_DETECTION_KEYS
            if key in item
        }
        for item in (report.get("detections") or [])
        if isinstance(item, dict)
    ]
    compact["web_compact"] = True
    return compact


def _transport_report_metadata(raw: Any) -> dict[str, Any]:
    """Expose the capture transport generation to optional web consumers."""

    if not isinstance(raw, dict):
        return {}
    timing = raw.get("capture_timing")
    timing = timing if isinstance(timing, dict) else {}
    session = raw.get("_transport_session", timing.get("transport_session"))
    if session in (None, ""):
        return {}
    connection = raw.get(
        "_transport_connection", timing.get("transport_connection", 0)
    )
    try:
        connection = int(connection or 0)
    except (TypeError, ValueError, OverflowError):
        return {}
    return {
        "transport_session": str(session),
        "transport_connection": connection,
    }


def _capture_observation_metadata(raw: dict, image_shape) -> dict:
    """M1 evidence identity/pose from this capture, never latest odometry."""
    metadata = {
        "capture_step": int(raw["seq"]),
        "stamp_sec": float(raw["stamp"]),
        "image_size": [int(image_shape[1]), int(image_shape[0])],
    }
    telemetry = raw.get("telemetry")
    if _capture_pose_error(telemetry) is None:
        position = _telemetry_body_position(telemetry)
        x, y, z, w = _telemetry_quaternion(telemetry)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        metadata["observation_pose_xyyaw"] = [float(position[0]), float(position[1]), yaw]
    return metadata


def _encode_detection_overlay(
    rgb: np.ndarray,
    detections: list[dict[str, Any]],
    quality: int = 92,
) -> str:
    """Encode the exact YOLO input receipt with readable 2-D boxes."""
    overlay = np.ascontiguousarray(rgb.copy())
    for detection in detections:
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        label = str(detection.get("semantic_class") or "object")
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        token = sum((index + 1) * ord(char) for index, char in enumerate(label))
        palette = (
            (70, 90, 245), (245, 200, 70), (120, 220, 80), (60, 190, 235),
            (235, 90, 180), (245, 150, 65), (215, 220, 90), (170, 90, 235),
        )
        color = palette[token % len(palette)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        text = f"{label} {confidence:.2f}"
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
        )
        text_top = max(0, y1 - height - baseline - 6)
        cv2.rectangle(
            overlay,
            (x1, text_top),
            (min(overlay.shape[1] - 1, x1 + width + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            overlay,
            text,
            (x1 + 4, max(height + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(
        ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, min(100, max(50, int(quality)))]
    )
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _encode_point_rows(points: np.ndarray) -> str:
    """Pack Nx3 float32 rows without expanding every coordinate into JSON."""
    rows = np.ascontiguousarray(points, dtype="<f4").reshape(-1, 3)
    return base64.b64encode(rows.tobytes()).decode("ascii")


def _compact_mask_coordinates(
    mask: np.ndarray, max_points: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded row/column samples without two full ``np.where`` arrays.

    The overlay contract still uses JSON row/column lists, but only a small
    representative sample is needed for that debug view.  ``flatnonzero``
    keeps one index vector and converts it after bounded striding, reducing
    temporary memory and Python allocator pressure on crowded 10 Hz frames.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or max_points <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    indices = np.flatnonzero(binary)
    if indices.size > int(max_points):
        step = max(1, int(math.ceil(indices.size / float(max_points))))
        indices = indices[::step][: int(max_points)]
    width = int(binary.shape[1])
    return (
        (indices // width).astype(np.int32, copy=False),
        (indices % width).astype(np.int32, copy=False),
    )


def _mask_uint8_view(mask: np.ndarray) -> np.ndarray:
    """Return a connected-components/contour-compatible uint8 mask.

    A boolean RGB-D mask already stores one byte per pixel with values 0/1.
    Viewing a contiguous boolean array as uint8 is therefore exact and avoids
    copying the complete 848x480 canvas for every selected detection.  Other
    dtypes and non-contiguous views retain the defensive conversion path.
    """
    array = np.asarray(mask)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_ and array.flags.c_contiguous:
        return array.view(np.uint8)
    return np.asarray(array > 0, dtype=np.uint8)


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.asarray([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]], dtype=np.float32)


def _telemetry_quaternion(telemetry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Use exactly the normalized body orientation published in sensor TF."""
    return _telemetry_body_quaternion(telemetry)


def _capture_pose_error(telemetry: Any) -> str | None:
    """A missing body pose is not the world origin. Gate live 3-D admission."""
    if not isinstance(telemetry, dict) or not telemetry:
        return "missing_capture_pose"
    position = telemetry.get("position")
    if not isinstance(position, (list, tuple, np.ndarray)):
        return "missing_capture_position"
    if _telemetry_body_position(telemetry) is None:
        return "invalid_capture_position"
    return None if _telemetry_body_quaternion(telemetry) is not None else "missing_or_invalid_capture_orientation"


def _quaternion_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=np.float32,
    )


_OPTICAL_TO_BASE = np.asarray(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)

# Optional 3-D support clustering is disabled in the physical profile, but
# replay/configuration callers may enable it.  Cache the optional SciPy import
# and the fixed 26-neighbourhood so an enabled multi-instance frame does not
# repeat import/module lookup and kernel allocation for every object.
_NDIMAGE = None
_NDIMAGE_IMPORT_ATTEMPTED = False
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _world_transform(
    telemetry: dict[str, Any],
    translation: np.ndarray,
    rpy: tuple[float, float, float],
    *,
    optical_frame: bool = False,
    depth_to_color_extrinsics: dict[str, Any] | None = None,
    camera_imu_enabled: bool = True,
    camera_imu_use_yaw: bool = False,
    camera_imu_max_roll_rad: float = 0.7,
    camera_imu_max_pitch_rad: float = 0.7,
    camera_imu_max_yaw_rad: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one camera-to-world affine transform for a complete RGB-D frame.

    All detections in one receipt share the same mount and body pose. Building
    this matrix once avoids repeating quaternion parsing, trigonometry and
    small matrix allocations for every instance in the YOLO result.
    """
    rotation = np.eye(3, dtype=np.float32)
    offset = np.zeros(3, dtype=np.float32)
    if optical_frame:
        if isinstance(depth_to_color_extrinsics, dict):
            try:
                rotation = np.asarray(
                    depth_to_color_extrinsics.get(
                        "rotation", np.eye(3).reshape(-1)
                    ),
                    dtype=np.float32,
                ).reshape(3, 3)
                raw_offset = np.asarray(
                    depth_to_color_extrinsics.get("translation", [0.0, 0.0, 0.0]),
                    dtype=np.float32,
                ).reshape(-1)
                if raw_offset.size >= 3:
                    offset = raw_offset[:3]
                else:
                    rotation = np.eye(3, dtype=np.float32)
            except (TypeError, ValueError, IndexError):
                rotation = np.eye(3, dtype=np.float32)
                offset = np.zeros(3, dtype=np.float32)
        rotation = _OPTICAL_TO_BASE @ rotation
    mount_rotation = _rotation(*rpy)
    rotation = mount_rotation @ rotation
    if optical_frame:
        offset = _OPTICAL_TO_BASE @ offset
    offset = mount_rotation @ offset + np.asarray(translation, dtype=np.float32).reshape(3)

    # ``physical_sensor_ros_bridge`` applies the calibrated D435i gravity
    # correction to the *complete* parent<-camera transform.  Applying only
    # the orientation would leave the elevated camera at its nominal x/y
    # position while the aluminium rod is leaning; that produces a horizontal
    # point-cloud/3-D-box displacement proportional to the 0.98 m lever arm.
    # Reuse the bridge's axis-conjugation helper so a correction expressed in
    # D435 optical axes is converted to base axes in exactly the same way.
    if optical_frame:
        mount_quaternion = _sensor_quat_multiply(
            _sensor_quat_rpy(*rpy),
            (0.5, -0.5, 0.5, -0.5),
        )
        correction_quaternion, correction_valid = (
            _camera_imu_parent_correction_quaternion(
                telemetry,
                mount_quaternion,
                enabled=_sensor_coerce_bool(camera_imu_enabled, True),
                use_yaw=_sensor_coerce_bool(camera_imu_use_yaw, False),
                max_roll_rad=float(camera_imu_max_roll_rad),
                max_pitch_rad=float(camera_imu_max_pitch_rad),
                max_yaw_rad=float(camera_imu_max_yaw_rad),
            )
        )
        if correction_valid:
            correction_rotation = _quaternion_matrix(correction_quaternion)
            # This is T' = T_correction * T_nominal, matching
            # ``_apply_camera_imu_correction`` in the ROS bridge.  Translation
            # must rotate too; otherwise the camera's physical sway is only
            # half corrected and embedded fallback world points disagree with
            # TF-transformed segmented clouds.
            rotation = correction_rotation @ rotation
            offset = correction_rotation @ offset

    quaternion = _telemetry_quaternion(telemetry)
    if quaternion is not None:
        body_rotation = _quaternion_matrix(quaternion)
    else:
        # Do not eagerly evaluate an optional malformed IMU rpy when an
        # explicit body yaw is available.
        if "yaw" in telemetry:
            yaw = float(telemetry["yaw"])
        else:
            imu = telemetry.get("imu")
            imu_rpy = imu.get("rpy") if isinstance(imu, dict) else None
            yaw = float(imu_rpy[2]) if isinstance(imu_rpy, (list, tuple)) and len(imu_rpy) >= 3 else 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        body_rotation = np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    rotation = body_rotation @ rotation
    offset = body_rotation @ offset
    position = np.asarray(telemetry.get("position", [0.0, 0.0, 0.0])[:3], dtype=np.float32)
    if position.size < 3:
        position = np.pad(position, (0, 3 - position.size))
    offset = offset + position[:3]
    return rotation.astype(np.float32, copy=False), offset.astype(np.float32, copy=False)


def _world_points(
    points: np.ndarray,
    telemetry: dict[str, Any],
    translation: np.ndarray,
    rpy: tuple[float, float, float],
    *,
    optical_frame: bool = False,
    depth_to_color_extrinsics: dict[str, Any] | None = None,
    world_transform: tuple[np.ndarray, np.ndarray] | None = None,
    camera_imu_enabled: bool = True,
    camera_imu_use_yaw: bool = False,
    camera_imu_max_roll_rad: float = 0.7,
    camera_imu_max_pitch_rad: float = 0.7,
    camera_imu_max_yaw_rad: float = 0.35,
) -> np.ndarray:
    """Lift camera points into the odom/map frame.

    D435i points use REP-103 optical axes (x right, y down, z forward), while
    Go2 base/odom uses x forward, y left, z up.  ``optical_frame`` is opt-in to
    preserve the historical helper contract used by simulator-side tests.
    """
    camera_points = np.asarray(points, dtype=np.float32)
    if world_transform is None:
        # Keep the standalone helper on the same path as ``infer``.  In
        # particular this applies the camera-IMU correction and rotates the
        # elevated mount translation; older callers that omit the precomputed
        # matrix must not silently regress to an uncorrected rod transform.
        world_transform = _world_transform(
            telemetry,
            np.asarray(translation, dtype=np.float32),
            rpy,
            optical_frame=optical_frame,
            depth_to_color_extrinsics=depth_to_color_extrinsics,
            camera_imu_enabled=camera_imu_enabled,
            camera_imu_use_yaw=camera_imu_use_yaw,
            camera_imu_max_roll_rad=camera_imu_max_roll_rad,
            camera_imu_max_pitch_rad=camera_imu_max_pitch_rad,
            camera_imu_max_yaw_rad=camera_imu_max_yaw_rad,
        )
    rotation, offset = world_transform
    rotation_matrix = np.asarray(rotation, dtype=np.float32)
    translation_vector = np.asarray(offset, dtype=np.float32)
    # Use one output buffer for the affine transform.  The expression
    # ``points @ R.T + t`` allocates a second full N×3 temporary for the
    # broadcasted translation; in crowded frames that copy is repeated once
    # per selected detection.  ``out`` plus in-place translation is bitwise
    # equivalent for the float32 inputs used by the detector.
    world = np.empty_like(camera_points, dtype=np.float32)
    np.matmul(camera_points, rotation_matrix.T, out=world)
    world += translation_vector
    return world


def _label(raw: str) -> str:
    value = str(raw or "").strip().lower().replace(" ", "_")
    aliases = {"refrigerator": "fridge", "couch": "sofa", "dining_table": "table", "dinning_table": "table", "doorframe": "doorframe"}
    if value in aliases: return aliases[value]
    for keyword in ("refrigerator", "fridge", "door", "cabinet", "drawer", "chair", "table", "sofa", "couch", "bed", "bottle", "cup", "bowl", "sink", "toilet", "shelf", "box", "microwave"):
        if keyword in value: return aliases.get(keyword, keyword)
    return value


def _load_object_detection_config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = payload.get("object_detection", payload)
    if not isinstance(config, dict):
        raise ValueError("object_detection config must be a mapping")
    return config


def _supported_euclidean_cluster_mask(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Keep every spatially supported cluster without shrinking its extent.

    Occupied voxels provide a bounded-cost connectivity approximation.  The
    previous implementation found a cluster and then retained only a fixed
    radius around its centroid, which truncated every object larger than
    ``3 * eps``.  Here a component is accepted as a whole, and multiple
    disconnected object surfaces are intentionally retained.
    """
    if points.shape[0] <= min_points:
        return np.ones(points.shape[0], dtype=bool)
    voxel_size = max(float(eps), 1e-3)
    voxel_keys, inverse, counts = np.unique(
        np.floor(points / voxel_size).astype(np.int32),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    # The previous implementation walked every occupied voxel through a
    # Python dictionary/BFS.  A dense 3-D occupancy grid is much cheaper for
    # the usual RGB-D object ROI (and ``ndimage.label`` executes the connected
    # component pass in C).  Keep the original sparse fallback below for a
    # very large/sparse volume or hosts where SciPy is not installed.
    global _NDIMAGE, _NDIMAGE_IMPORT_ATTEMPTED
    if not _NDIMAGE_IMPORT_ATTEMPTED:
        _NDIMAGE_IMPORT_ATTEMPTED = True
        try:
            from scipy import ndimage as _scipy_ndimage  # type: ignore
        except Exception:
            _scipy_ndimage = None
        _NDIMAGE = _scipy_ndimage
    ndimage = _NDIMAGE
    if ndimage is not None:
        grid_min = voxel_keys.min(axis=0)
        shifted = voxel_keys - grid_min
        grid_shape = tuple(
            int(value) for value in (voxel_keys.max(axis=0) - grid_min + 1)
        )
        # Avoid a pathological allocation when a few outliers span many
        # metres.  Two million bytes for occupancy plus the int32 labels is a
        # bounded per-instance working set and is reclaimed before the next
        # detector object is lifted.
        if grid_shape and int(np.prod(grid_shape, dtype=np.int64)) <= 2_000_000:
            occupancy = np.zeros(grid_shape, dtype=np.uint8)
            occupancy[tuple(shifted.T)] = 1
            labels, component_count = ndimage.label(
                occupancy,
                structure=_CONNECTIVITY_26,
            )
            voxel_labels = labels[tuple(shifted.T)]
            component_sizes = np.bincount(
                voxel_labels,
                weights=counts,
                minlength=int(component_count) + 1,
            )
            keep = component_sizes >= min_points
            point_keep = keep[voxel_labels][inverse]
            if int(np.count_nonzero(point_keep)) >= min_points:
                return point_keep
            return np.ones(points.shape[0], dtype=bool)

    # Keep the sparse fallback allocation-free with respect to Python lists;
    # ``key.tolist()`` was visible in CPU profiles on hosts without SciPy.
    lookup = {
        (int(key[0]), int(key[1]), int(key[2])): index
        for index, key in enumerate(voxel_keys)
    }
    visited = np.zeros(voxel_keys.shape[0], dtype=bool)
    accepted_voxels: list[int] = []
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for start in range(voxel_keys.shape[0]):
        if visited[start]:
            continue
        queue, component = [start], []
        visited[start] = True
        while queue:
            index = queue.pop()
            component.append(index)
            key = voxel_keys[index]
            for dx, dy, dz in neighbor_offsets:
                neighbor = lookup.get((int(key[0] + dx), int(key[1] + dy), int(key[2] + dz)))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if int(np.sum(counts[np.asarray(component, dtype=np.int32)])) >= min_points:
            accepted_voxels.extend(component)
    if not accepted_voxels:
        return np.ones(points.shape[0], dtype=bool)
    keep = np.isin(inverse, np.asarray(accepted_voxels, dtype=np.int32))
    return keep if int(np.count_nonzero(keep)) >= min_points else np.ones(points.shape[0], dtype=bool)


def _prepare_instance_points(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    bbox: tuple[int, int, int, int],
    config: dict[str, Any],
    semantic_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Clean one PF-Seg mask and return its camera points and depths."""
    # Keep the compact uint8 representation for OpenCV throughout this
    # stage.  Detector masks are normally already uint8 after RGB->depth
    # projection; converting those to ``binary > 0`` again copied the full
    # 848x480 canvas for every instance.  OpenCV treats every non-zero uint8
    # value as foreground, so a contiguous uint8 mask can be used directly.
    binary = np.asarray(mask)
    if binary.ndim != 2:
        return None
    if binary.dtype != np.uint8:
        binary = _mask_uint8_view(binary)
    elif not binary.flags.c_contiguous:
        binary = np.ascontiguousarray(binary)
    min_component = max(1, int(config.get("mask_component_min_area", 48)))
    # Connected-components is linear in the image area.  YOLO masks are
    # sparse and usually occupy a small box; crop to the actual support first
    # and restore coordinates below.  This removes a full-resolution scan for
    # every detection without changing component boundaries.
    # ``np.where`` on a 848x480 mask allocates two Python-visible index
    # arrays and was repeatedly showing up in post-processing profiles.  The
    # OpenCV C++ scan returns the same support bounding box without exposing
    # every coordinate until component extraction below.
    crop_x, crop_y, crop_width, crop_height = cv2.boundingRect(binary)
    if crop_width <= 0 or crop_height <= 0:
        return None
    crop_x0, crop_y0 = int(crop_x), int(crop_y)
    crop_x1, crop_y1 = crop_x0 + int(crop_width), crop_y0 + int(crop_height)
    cropped_binary = binary[crop_y0:crop_y1, crop_x0:crop_x1]
    # Most detector masks contain far fewer than 65k connected components.
    # A uint16 label image halves the temporary working set and is noticeably
    # faster on the native 848x480 path.  Extremely fragmented/noisy masks can
    # exceed the uint16 label range; retain the int32 fallback for those rather
    # than changing component admission semantics.
    try:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            cropped_binary, 8, cv2.CV_16U
        )
    except cv2.error:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            cropped_binary, 8, cv2.CV_32S
        )
    if count <= 1:
        return None
    _ = bbox  # retained in the public helper contract for detector call sites
    candidates = [index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_component]
    if not candidates:
        return None
    min_valid = max(4, int(config.get("min_valid_points", 12)))
    low_q = min(max(float(config.get("depth_band_lower_quantile", 0.02)), 0.0), 1.0)
    high_q = min(max(float(config.get("depth_band_upper_quantile", 0.98)), low_q), 1.0)
    depth_scale = float(config.get("depth_scale", 1.0))
    max_depth = float(config.get("max_depth_m", 8.0))
    kept_rows: list[np.ndarray] = []
    kept_cols: list[np.ndarray] = []
    kept_depths: list[np.ndarray] = []
    planar_labels = {
        str(item).strip().casefold()
        for item in config.get("planar_semantic_labels", ("door", "portal"))
        if str(item).strip()
    }
    # Filter every sufficiently large 2-D component independently. A global
    # depth band can erase a valid disconnected surface that lies farther
    # away than the dominant component of the same instance mask.
    for component in candidates:
        # Restrict the label scan to the component bounding box.  Scanning the
        # full 384x640 mask once per component dominated CPU post-processing
        # when YOLO returned many instances.
        x0 = int(stats[component, cv2.CC_STAT_LEFT]) + crop_x0; y0 = int(stats[component, cv2.CC_STAT_TOP]) + crop_y0
        width = int(stats[component, cv2.CC_STAT_WIDTH]); height = int(stats[component, cv2.CC_STAT_HEIGHT])
        label_y0, label_x0 = y0 - crop_y0, x0 - crop_x0
        # ``np.where(labels == component)`` materializes two int64 arrays and
        # was a measurable part of the 10-Hz budget for large masks.  OpenCV
        # already owns the connected-component image; finding its support in
        # C returns compact int32 coordinates and avoids the temporary Python
        # index arrays.  The resulting coordinates are identical to the
        # previous NumPy path.
        component_region = np.ascontiguousarray(
            labels[label_y0:label_y0 + height, label_x0:label_x0 + width]
            == component,
            dtype=np.uint8,
        )
        component_pixels = cv2.findNonZero(component_region)
        if component_pixels is None:
            continue
        local_cols = component_pixels[:, 0, 0].astype(np.int32, copy=False)
        local_rows = component_pixels[:, 0, 1].astype(np.int32, copy=False)
        rows, cols = local_rows + y0, local_cols + x0
        depths = depth[rows, cols] * depth_scale
        valid = np.isfinite(depths) & (depths > 0.0) & (depths <= max_depth)
        rows, cols, depths = rows[valid], cols[valid], depths[valid]
        if depths.size < min_valid:
            continue
        if depths.size >= max(min_valid * 2, 32):
            low_value, high_value = _fast_quantile_pair(
                depths, low_q, high_q, axis=0
            )
            low, high = float(low_value), float(high_value)
            in_band = (depths >= low) & (depths <= high)
            if int(np.count_nonzero(in_band)) >= min_valid:
                rows, cols, depths = rows[in_band], cols[in_band], depths[in_band]
        # A door/portal is predominantly a single depth plane.  RGB-D
        # segmentation often leaks through its edges or holes onto the wall
        # behind it; those points create an artificially wide world box after
        # projection.  Use a robust MAD band only for configured planar
        # classes, retaining the full 2-D extent of the actual plane.
        if semantic_label and str(semantic_label).strip().casefold() in planar_labels and depths.size >= max(min_valid * 2, 32):
            median = float(np.median(depths))
            mad = float(np.median(np.abs(depths - median)))
            tolerance = max(float(config.get("planar_depth_min_tolerance_m", 0.10)), 4.0 * mad)
            planar_band = np.abs(depths - median) <= tolerance
            if int(np.count_nonzero(planar_band)) >= min_valid:
                rows, cols, depths = rows[planar_band], cols[planar_band], depths[planar_band]
        kept_rows.append(rows)
        kept_cols.append(cols)
        kept_depths.append(depths)
    if not kept_depths:
        return None
    dense_rows = np.concatenate(kept_rows)
    dense_cols = np.concatenate(kept_cols)
    dense_depths = np.concatenate(kept_depths)
    filtered_mask = np.zeros(binary.shape, dtype=bool)
    filtered_mask[dense_rows, dense_cols] = True
    stride = max(1, int(config.get("point_stride", 4)))
    rows = dense_rows[::stride]
    cols = dense_cols[::stride]
    depths = dense_depths[::stride]
    fx, fy, cx, cy = intrinsics
    # Allocate the final float32 point matrix once.  ``np.stack(...).astype``
    # first builds a temporary float64 N×3 array because the pixel-coordinate
    # arithmetic promotes against the Python intrinsics, then copies it back
    # to float32.  The per-column writes retain the exact casted values while
    # removing that full-size intermediate allocation from every instance.
    points = np.empty((rows.size, 3), dtype=np.float32)
    points[:, 0] = (cols - cx) * depths / fx
    points[:, 1] = (rows - cy) * depths / fy
    points[:, 2] = depths
    if bool(config.get("enable_euclidean_cluster", True)):
        cluster_keep = _supported_euclidean_cluster_mask(
            points,
            float(config.get("cluster_eps", 0.18)),
            max(4, int(config.get("cluster_min_points", 12))),
        )
        points = points[cluster_keep]
        depths = depths[cluster_keep]
    if points.shape[0] < min_valid:
        return None
    # ``filtered_mask``, ``points`` and robust bounds now derive from one
    # inlier definition. Point stride changes density only, not semantics.
    return filtered_mask, points, depths


def _robust_bounds(points: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    low = min(max(float(config.get("bbox_quantile_lower", 0.10)), 0.0), 1.0)
    high = min(max(float(config.get("bbox_quantile_upper", 0.90)), low), 1.0)
    lower, upper = _fast_quantile_pair(points, low, high, axis=0)
    return lower.astype(np.float32, copy=False), upper.astype(np.float32, copy=False)


def _fast_quantile(values: np.ndarray, quantile: float, *, axis: int = 0) -> np.ndarray:
    """Compute one linear quantile with a cache-friendly quicksort.

    ``np.quantile`` is general-purpose and performs several reductions and
    temporary allocations for every RGB-D instance.  Our geometry arrays are
    finite, dense ``N x 3`` samples and always use one axis, so sorting once
    and interpolating the two neighbouring order statistics is equivalent to
    NumPy's default linear method while being considerably cheaper on the
    10-Hz path.  Keeping this helper local also avoids changing the wire or
    geometry contract for replay callers.
    """
    array = np.asarray(values)
    if array.ndim == 0:
        return array.astype(np.float32, copy=False)
    length = int(array.shape[axis])
    if length <= 0:
        return np.full(array.shape[:axis] + array.shape[axis + 1 :], np.nan)
    ordered = np.sort(array, axis=axis, kind="quicksort")
    return _interpolate_ordered_quantile(ordered, quantile, axis=axis)


def _interpolate_ordered_quantile(
    ordered: np.ndarray, quantile: float, *, axis: int = 0
) -> np.ndarray:
    """Interpolate one quantile from an already sorted array."""
    length = int(ordered.shape[axis])
    if length <= 0:
        return np.full(ordered.shape[:axis] + ordered.shape[axis + 1 :], np.nan)
    position = min(max(float(quantile), 0.0), 1.0) * float(length - 1)
    lower_index = int(math.floor(position))
    upper_index = min(lower_index + 1, length - 1)
    fraction = position - float(lower_index)
    # Geometry callers always reduce the first axis (the sample axis).  A
    # direct index avoids ``np.take``'s generic dispatcher and temporary
    # index-array handling; this helper is called twice for every robust box
    # and can otherwise show up in crowded-frame profiles.  Keep the generic
    # path for replay callers that request another axis.
    lower = (
        ordered[lower_index]
        if axis == 0
        else np.take(ordered, lower_index, axis=axis)
    )
    if upper_index == lower_index or fraction <= 0.0:
        return lower
    upper = (
        ordered[upper_index]
        if axis == 0
        else np.take(ordered, upper_index, axis=axis)
    )
    return lower + (upper - lower) * fraction


def _fast_quantile_pair(
    values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two quantiles while sorting the geometry samples only once."""
    ordered = np.sort(np.asarray(values), axis=axis, kind="quicksort")
    return (
        _interpolate_ordered_quantile(ordered, lower_quantile, axis=axis),
        _interpolate_ordered_quantile(ordered, upper_quantile, axis=axis),
    )


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    """Convert a right-handed 3x3 rotation matrix to an xyzw quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = 2.0 * math.sqrt(max(1e-12, 1.0 + float(diagonal[index]) - float(diagonal[next_index]) - float(diagonal[last_index])))
        values = [0.0, 0.0, 0.0, 0.0]
        values[index] = 0.25 * scale
        values[3] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        values[next_index] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        values[last_index] = (matrix[last_index, index] + matrix[index, last_index]) / scale
        x, y, z, w = values
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return quaternion.astype(float).tolist()


def _oriented_bounds(
    points: np.ndarray,
    config: dict[str, Any],
    *,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit an upright instance OBB; only horizontal yaw comes from points."""
    # ``infer`` already needs the arithmetic world-point mean for its
    # low-object rejection gate.  Accepting that value avoids a second full
    # N×3 reduction immediately before the OBB fit while preserving the
    # original two-argument helper contract for replay/tests.
    if center is None:
        center = np.mean(points, axis=0).astype(np.float32)
    else:
        center = np.asarray(center, dtype=np.float32).reshape(3)
    if points.shape[0] < 3 or float(np.max(np.ptp(points, axis=0))) < 1e-4:
        mins, maxs = _robust_bounds(points, config)
        return (mins + maxs) / 2.0, np.maximum(maxs - mins, 0.01), [0.0, 0.0, 0.0, 1.0]
    centered = points.astype(np.float64) - center.astype(np.float64)
    covariance_xy = centered[:, :2].T @ centered[:, :2] / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xy)
    horizontal = eigenvectors[:, int(np.argmax(eigenvalues))]
    yaw = math.atan2(float(horizontal[1]), float(horizontal[0]))
    # Canonicalize the 180-degree eigenvector ambiguity for stable labels.
    if math.cos(yaw) < 0.0 or (abs(math.cos(yaw)) < 1e-6 and math.sin(yaw) < 0.0):
        yaw += math.pi
    c, s = math.cos(yaw), math.sin(yaw)
    axes = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # ``axes`` is the local-to-world rotation.  With row-vector points,
    # multiplying by ``axes`` projects world points into the local OBB frame.
    projected = centered @ axes
    low = min(max(float(config.get("bbox_quantile_lower", 0.02)), 0.0), 1.0)
    high = min(max(float(config.get("bbox_quantile_upper", 0.98)), low), 1.0)
    mins, maxs = _fast_quantile_pair(projected, low, high, axis=0)
    local_center = (mins + maxs) / 2.0
    world_center = center + axes @ local_center
    size = np.maximum(maxs - mins, 0.01).astype(np.float32)
    return world_center.astype(np.float32), size, [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _is_ground_like_box(center: np.ndarray, size: np.ndarray, config: dict[str, Any]) -> bool:
    return bool(
        max(float(size[0]), float(size[1])) >= float(config.get("ground_reject_min_xy_span", 1.20))
        and float(size[2]) <= float(config.get("ground_reject_max_height", 0.40))
        and float(center[2]) <= float(config.get("ground_reject_max_center_z", 0.25))
    )


def _is_excluded_scene_label(value: Any, config: dict[str, Any]) -> bool:
    """Reject compound open-vocabulary place/structure labels from YAML."""
    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    tokens = {token for token in normalized.split("_") if token}
    filter_config = config.get("detection_filter", config)
    excluded = {
        str(item or "").strip().casefold().replace("-", "_").replace(" ", "_")
        for item in filter_config.get("excluded_label_tokens", [])
        if str(item or "").strip()
    }
    return any(item in tokens if "_" not in item else item in normalized for item in excluded)


def _is_implausibly_large_object(label: Any, size: np.ndarray, config: dict[str, Any]) -> bool:
    """Keep close-up interaction targets while gating giant ordinary tracks."""
    semantic = str(label or "").casefold()
    if any(marker in semantic for marker in ("door", "portal", "fridge", "refrigerator", "cabinet", "drawer")):
        return False
    span = max(abs(float(value)) for value in size)
    volume = abs(float(size[0]) * float(size[1]) * float(size[2]))
    return bool(
        span > float(config.get("max_noninteraction_box_span_m", 2.50))
        or volume > float(config.get("max_noninteraction_box_volume_m3", 4.00))
    )


def _passes_class_plausibility(
    label: Any,
    bbox: tuple[int, int, int, int] | list[int],
    world_size: np.ndarray,
    config: dict[str, Any],
) -> bool:
    """Apply conservative class-specific gates after RGB-D lifting.

    Open-vocabulary fridge predictions commonly attach to small cabinet
    patches or an entire wall.  These bounds reject those two failure modes
    while retaining partial, normally sized refrigerators.
    """

    semantic = str(label or "").strip().casefold()
    rules = (config.get("class_plausibility") or {}).get(semantic)
    if not isinstance(rules, dict):
        return True
    if len(bbox) >= 4:
        width_px = max(0.0, float(bbox[2]) - float(bbox[0]))
        height_px = max(0.0, float(bbox[3]) - float(bbox[1]))
        if width_px * height_px < float(rules.get("min_bbox_area_px", 0.0)):
            return False
        if min(width_px, height_px) < float(rules.get("min_bbox_side_px", 0.0)):
            return False
    size = np.abs(np.asarray(world_size, dtype=np.float32).reshape(-1)[:3])
    if size.size < 3:
        return False
    horizontal_span = max(float(size[0]), float(size[1]))
    height_m = float(size[2])
    volume_m3 = float(np.prod(size))
    return bool(
        height_m >= float(rules.get("min_height_m", 0.0))
        and height_m <= float(rules.get("max_height_m", math.inf))
        and horizontal_span <= float(rules.get("max_horizontal_span_m", math.inf))
        and volume_m3 <= float(rules.get("max_volume_m3", math.inf))
    )


def _build_rgb_mask_polygons(
    mask: np.ndarray | None,
    source_polygons: Any,
    rgb_shape: tuple[int, ...],
) -> list[list[list[int]]]:
    """Materialize one detector mask's RGB overlay polygons.

    This remains outside the threaded 3-D worker: Ultralytics' lazy ``masks.xy``
    property is not guaranteed to be thread-safe.  The helper is pure after
    ``source_polygons`` has been selected and keeps the historical polygon
    clipping/approximation contract byte-for-byte.
    """
    if mask is None or source_polygons is None:
        return []
    rgb_mask_polygons: list[list[list[int]]] = []
    polygon_array = np.asarray(source_polygons, dtype=object)
    if polygon_array.ndim == 2 and polygon_array.shape[-1] == 2:
        source_polygons = [source_polygons]
    for source_polygon in source_polygons:
        polygon = np.asarray(source_polygon, dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] < 3:
            continue
        polygon[:, 0] = np.clip(polygon[:, 0], 0, rgb_shape[1] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, rgb_shape[0] - 1)
        rgb_mask_polygons.append(np.rint(polygon).astype(int).tolist())
    return rgb_mask_polygons


def _geometry_lift_candidate(
    mask: np.ndarray | None,
    depth: np.ndarray,
    rgb_intr: dict[str, Any],
    depth_intr: dict[str, Any],
    projection_extrinsics: dict[str, Any],
    projection_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    bbox: tuple[int, int, int, int],
    depth_scale: float,
    aligned_stream: bool,
    intrinsics: tuple[float, float, float, float],
    geometry_config: dict[str, Any],
    semantic_label: str,
    world_transform: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Lift one selected mask into RGB-D/world geometry.

    The function is deliberately stateless so a persistent executor can run
    multiple candidates concurrently.  It returns all NumPy intermediates to
    the caller in candidate order; no ROS/model state is touched here.
    """
    projection_started = time.perf_counter()
    if mask is not None and mask.shape != depth.shape:
        mask = _rgb_mask_to_depth(
            mask.astype(np.uint8),
            depth,
            rgb_intr,
            depth_intr,
            projection_extrinsics,
            projection_maps,
            bbox,
            depth_scale,
            aligned_stream=aligned_stream,
        )
    elif mask is not None:
        mask = mask > 0.5
    else:
        mask = np.zeros(depth.shape, dtype=bool)
    projection_ms = (time.perf_counter() - projection_started) * 1000.0
    fx, fy, _cx, _cy = intrinsics
    if fx <= 0 or fy <= 0:
        return {
            "ok": False,
            "reason": "invalid_depth_intrinsics",
            "mask_projection_ms": projection_ms,
            "depth_geometry_ms": 0.0,
        }
    geometry_started = time.perf_counter()
    try:
        prepared = _prepare_instance_points(
            mask,
            depth,
            intrinsics,
            bbox,
            geometry_config,
            semantic_label,
        )
    except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
        return {
            "ok": False,
            "reason": f"geometry_error:{type(exc).__name__}",
            "mask_projection_ms": projection_ms,
            "depth_geometry_ms": (time.perf_counter() - geometry_started) * 1000.0,
        }
    if prepared is None:
        return {
            "ok": False,
            "reason": "insufficient_depth_points",
            "mask_projection_ms": projection_ms,
            "depth_geometry_ms": (time.perf_counter() - geometry_started) * 1000.0,
        }
    filtered_mask, points, values = prepared
    camera_mins, camera_maxs = _robust_bounds(points, geometry_config)
    camera_center = (camera_mins + camera_maxs) / 2
    camera_size = np.maximum(camera_maxs - camera_mins, 0.01)
    world = _world_points(
        points,
        {},
        np.zeros(3, dtype=np.float32),
        (0.0, 0.0, 0.0),
        optical_frame=False,
        world_transform=world_transform,
    )
    mins, maxs = _robust_bounds(world, geometry_config)
    center = (mins + maxs) / 2
    size = np.maximum(maxs - mins, 0.01)
    world_aabb_min_z = float(np.min(world[:, 2]))
    world_aabb_max_z = float(np.max(world[:, 2]))
    world_mean = np.mean(world, axis=0).astype(np.float32)
    obb_center, obb_size, obb_orientation = _oriented_bounds(
        world, geometry_config, center=world_mean
    )
    # Preserve the OBB fitted from the complete instance cloud in a
    # sensor-frame representation.  The local ROS gateway can then compose
    # it with the exact capture-time TF instead of fitting a second box from
    # the bounded (at most a few hundred point) visualization sample.
    world_rotation, world_offset = world_transform
    camera_obb_center = np.asarray(world_rotation, dtype=np.float64).T @ (
        np.asarray(obb_center, dtype=np.float64)
        - np.asarray(world_offset, dtype=np.float64)
    )
    camera_obb_axes = (
        np.asarray(world_rotation, dtype=np.float64).T
        @ _quaternion_matrix(tuple(obb_orientation)).astype(np.float64)
    )
    camera_obb_orientation = _rotation_matrix_to_quaternion(camera_obb_axes)
    obb_yaw = math.atan2(
        2.0 * float(obb_orientation[3]) * float(obb_orientation[2]),
        1.0 - 2.0 * float(obb_orientation[2]) * float(obb_orientation[2]),
    )
    geometry_ms = (time.perf_counter() - geometry_started) * 1000.0
    if bool(geometry_config.get("reject_low_mean_height", True)) and float(world_mean[2]) <= float(geometry_config.get("min_mean_height_m", 0.08)):
        return {
            "ok": False,
            "reason": "low_mean_height",
            "mask_projection_ms": projection_ms,
            "depth_geometry_ms": geometry_ms,
        }
    if _is_ground_like_box(center, size, geometry_config):
        return {"ok": False, "reason": "ground_like_box", "mask_projection_ms": projection_ms, "depth_geometry_ms": geometry_ms}
    if _is_implausibly_large_object(semantic_label, size, geometry_config):
        return {"ok": False, "reason": "implausibly_large_box", "mask_projection_ms": projection_ms, "depth_geometry_ms": geometry_ms}
    if not _passes_class_plausibility(semantic_label, bbox, size, geometry_config):
        return {"ok": False, "reason": "class_plausibility", "mask_projection_ms": projection_ms, "depth_geometry_ms": geometry_ms}
    return {
        "ok": True,
        "mask": filtered_mask,
        "points": points,
        "values": values,
        "world": world,
        "camera_center": camera_center,
        "camera_size": camera_size,
        "camera_obb_center": camera_obb_center.astype(np.float32),
        "camera_obb_size": obb_size,
        "camera_obb_orientation": camera_obb_orientation,
        "center": center,
        "size": size,
        "world_aabb_min_z": world_aabb_min_z,
        "world_aabb_max_z": world_aabb_max_z,
        "obb_center": obb_center,
        "obb_size": obb_size,
        "obb_orientation": obb_orientation,
        "obb_yaw": obb_yaw,
        "mask_projection_ms": projection_ms,
        "depth_geometry_ms": geometry_ms,
    }


def _context_world_transform(raw):
    """Compose the sensor's exact base<-depth transform with the body pose."""
    contract = raw.get("depth_to_base")
    if contract is None:
        authoritative = (raw.get("capture_pose_match") or {}).get("source") == "sensor_capture_context"
        return None, "missing_capture_transform" if authoritative else None
    try:
        if contract["parent_frame"] != "tf_frame_base_link":
            return None, "unsupported_capture_transform_parent"
        if contract["child_frame"] != raw.get("depth_frame"):
            return None, "mismatched_capture_transform_frame"
        translation = np.asarray(contract["translation"], dtype=np.float64)
        quaternion = np.asarray(contract["quaternion_xyzw"], dtype=np.float64)
        if translation.shape != (3,) or quaternion.shape != (4,):
            return None, "invalid_capture_transform"
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
            return None, "invalid_capture_transform"
        if np.any(np.abs(translation) > np.finfo(np.float32).max):
            return None, "invalid_capture_transform"
        norm = math.hypot(*quaternion)
        if not math.isfinite(norm) or norm < 1e-6:
            return None, "invalid_capture_transform"
        body_rotation, body_translation = _world_transform(
            raw["telemetry"], np.zeros(3, dtype=np.float32), (0., 0., 0.),
        )
        return (
            body_rotation @ _quaternion_matrix(quaternion / norm),
            body_rotation @ translation.astype(np.float32) + body_translation,
        ), None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid_capture_transform"


class _ExactRgbdPairBuffer:
    """Bound callback skew, but expose only the latest complete capture.

    Used under the worker's frame lock. Live bridge stamps are in a monotonic
    ROS clock domain. rospy rewrites Header.seq independently per topic, so
    capture stamps, not topic sequence counters, identify a matching pair.
    Arrays are already owned by callbacks and are not copied here.
    """

    def __init__(self, capacity=4, require_context=False):
        self.capacity = max(1, int(capacity))
        self.require_context = bool(require_context)
        self.pending = {"rgb": {}, "depth": {}}
        if self.require_context:
            self.pending["context"] = {}
        self.latest = None

    def push(self, kind, image, stamp, seq):
        if seq <= 0 and not self.require_context:
            self.pending = {"rgb": {}, "depth": {}}
            self.latest = None
            return  # Legacy seq=0 uses the existing timestamp-tolerance path.
        if not math.isfinite(stamp) or stamp <= 0.:
            return
        if self.latest is not None and stamp <= self.latest[1]:
            return
        key = stamp
        current = self.pending[kind]
        current[key] = (seq, image)
        if all(key in entries for entries in self.pending.values()):
            rgb_seq, rgb = self.pending["rgb"].pop(key)
            depth_seq, depth = self.pending["depth"].pop(key)
            self.latest = (max(rgb_seq, depth_seq), stamp, rgb, depth, rgb_seq, depth_seq)
            if self.require_context:
                self.latest += (self.pending["context"].pop(key)[1],)
            # Once a newer complete capture exists, older unmatched halves
            # cannot become the next inference input.
            for entries in self.pending.values():
                for old_key in list(entries):
                    if old_key <= stamp:
                        del entries[old_key]
        while len(current) > self.capacity:
            del current[next(iter(current))]


def _warmup_detector(model: Any, args: argparse.Namespace) -> None:
    """Validate the actual inference path before accepting any sensor frames."""
    import torch

    started = time.perf_counter()
    frame = np.zeros((int(args.imgsz), int(args.imgsz), 3), dtype=np.uint8)
    print(f"YOLOE startup: warming up device={args.device} (2 synthetic frames)", flush=True)
    try:
        for _ in range(2):
            model.predict(
                source=frame, device=args.device, imgsz=args.imgsz,
                conf=args.conf, iou=args.iou, max_det=args.max_det,
                verbose=False, save=False,
            )
        actual_device = model.predictor.device
        if str(args.device) != "cpu" and actual_device.type != "cuda":
            raise RuntimeError(f"requested {args.device}, but predictor is on {actual_device}")
        if actual_device.type == "cuda":
            torch.cuda.synchronize(actual_device)
            detail = (
                f"name={torch.cuda.get_device_name(actual_device)} "
                f"allocated_mib={torch.cuda.memory_allocated(actual_device) / 1024 ** 2:.1f}"
            )
        else:
            detail = "explicit CPU mode"
    except Exception as exc:
        raise RuntimeError(
            f"YOLOE startup warmup FAILED for {args.device}; detector is not ready: {exc}"
        ) from exc
    print(
        f"YOLOE startup: warmup OK requested={args.device} actual={actual_device} "
        f"{detail} elapsed_ms={(time.perf_counter() - started) * 1000:.1f}; "
        "waiting for live ROS RGB-D",
        flush=True,
    )


class YoloeWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        # Register ROS subscriptions before loading the large GPU model.  This
        # also keeps the worker usable from the mlspaces environment, where
        # cv_bridge is not installed.
        print("YOLOE startup: initializing ROS", flush=True)
        # ROS Noetic's Python 3.8 roslogging walks the Python frame stack in
        # ``findCaller``.  Under the Python 3.11 mlspaces interpreter this
        # compatibility path can spin indefinitely before the node registers.
        # The caller metadata is not used by this worker, so use a constant
        # lightweight implementation for ROS logging.
        try:
            import rosgraph.roslogging
            rosgraph.roslogging.RospyLogger.findCaller = lambda self, *a, **k: ("<physical_yoloe>", 0, "<worker>", None)
        except Exception:
            pass
        rospy.init_node("physical_yoloe_bridge", anonymous=True)
        # The direct sensor bridge owns these private parameters.  Mirror them
        # here when available so an operator can disable/limit TF correction in
        # one launch file without leaving YOLO's embedded fallback on a
        # different transform.  The defaults intentionally match the sensor
        # bridge for standalone replay/tests where no ROS parameter exists.
        sensor_param_ns = "/physical_sensor_ros_bridge"
        self._telemetry_max_delta_sec = float(getattr(args, "telemetry_max_delta_sec", 0.15))
        self._last_telemetry_match = {}
        try:
            args.camera_imu_enabled = _sensor_coerce_bool(
                rospy.get_param(
                    sensor_param_ns + "/camera_imu_enabled",
                    getattr(args, "camera_imu_enabled", True),
                ),
                True,
            )
            args.camera_imu_use_yaw = _sensor_coerce_bool(
                rospy.get_param(
                    sensor_param_ns + "/camera_imu_use_yaw",
                    getattr(args, "camera_imu_use_yaw", False),
                ),
                False,
            )
            args.camera_imu_max_roll_rad = float(
                rospy.get_param(
                    sensor_param_ns + "/camera_imu_max_roll_rad",
                    getattr(args, "camera_imu_max_roll_rad", 0.7),
                )
            )
            args.camera_imu_max_pitch_rad = float(
                rospy.get_param(
                    sensor_param_ns + "/camera_imu_max_pitch_rad",
                    getattr(args, "camera_imu_max_pitch_rad", 0.7),
                )
            )
            args.camera_imu_max_yaw_rad = float(
                rospy.get_param(
                    sensor_param_ns + "/camera_imu_max_yaw_rad",
                    getattr(args, "camera_imu_max_yaw_rad", 0.35),
                )
            )
        except Exception as exc:
            # A ROS master may expose no sensor node during offline replay;
            # retaining the matching defaults is safer than disabling a valid
            # correction based on a transient parameter lookup failure.
            rospy.logdebug("camera-IMU TF parameter lookup skipped: %s", exc)
        print("YOLOE startup: importing ultralytics", flush=True)
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("YOLOE worker requires ultralytics in the algorithm Python environment") from exc
        print(f"YOLOE startup: loading model {args.model_path}", flush=True)
        self.args = args; self.model = YOLOE(args.model_path); print("YOLOE startup: model loaded", flush=True); self.last_seq = -1; self.last_stamp = float("-inf"); self.rotation = _rotation(args.camera_roll, args.camera_pitch, args.camera_yaw); self.translation = np.asarray([args.camera_x, args.camera_y, args.camera_z], dtype=np.float32); self._projection_cache_key = None; self._projection_cache = None; self._profile_count = 0; self._profile_totals: dict[str, float] = {}; self._geometry_cursor = 0
        self.detector_config = _load_object_detection_config(args.detector_config)
        self._geometry_workers = max(
            1, min(8, int(self.detector_config.get("geometry_workers", 4)))
        )
        self._geometry_executor = ThreadPoolExecutor(
            max_workers=self._geometry_workers,
            thread_name_prefix="yolo-geometry",
        )
        # Futures that exceeded a frame's geometry deadline keep running only
        # until the underlying NumPy/OpenCV call returns.  Track them across
        # receipts so a later frame never queues more work behind an already
        # occupied worker; this bounds both latency and retained RGB-D frames.
        self._geometry_inflight: set[Future[Any]] = set()
        self._geometry_abandoned_total = 0
        rospy.on_shutdown(self._shutdown_geometry_executor)
        # Keep the physical YAML authoritative.  Previously the CLI defaults
        # silently won, so changing confidence_threshold did not affect the
        # running worker unless start_physical_nav.sh also passed --conf.
        class_thresholds = self.detector_config.get("class_confidence_thresholds") or {}
        self.args.conf = max(0.001, min(
            [float(self.detector_config.get("confidence_threshold", args.conf))]
            + [float(value) for value in class_thresholds.values()]
        ))
        self.args.iou = max(
            0.0,
            min(1.0, float(self.detector_config.get("iou_threshold", args.iou))),
        )
        self.args.max_det = max(
            1, int(self.detector_config.get("max_detections", args.max_det))
        )
        # The YAML is the physical profile's source of truth.  The launcher
        # intentionally does not duplicate every geometry flag, so do not let
        # argparse defaults silently change stride/depth processing.
        self.args.imgsz = max(32, int(self.detector_config.get("imgsz", args.imgsz)))
        self.args.point_stride = max(1, int(self.detector_config.get("point_stride", args.point_stride)))
        self.args.min_valid_points = max(4, int(self.detector_config.get("min_valid_points", args.min_valid_points)))
        self.args.max_depth_m = max(0.1, float(self.detector_config.get("max_depth_m", args.max_depth_m)))
        filter_config = load_detection_filter_config(args.detector_config)
        self.detection_filter = DetectionFilter(filter_config)
        self.class_confidence_thresholds = {
            str(label).strip().casefold().replace(" ", "_"): max(0.001, min(1.0, float(value)))
            for label, value in class_thresholds.items()
        }
        # Failure must escape startup, not enter the per-frame fallback loop.
        # Synthetic warmup outputs are never published to the semantic graph.
        _warmup_detector(self.model, self.args)
        self.report_pub = rospy.Publisher("/physical_nav/yolo_report", String, queue_size=1)
        # A report contains the overlay JPEG, sparse masks and compact point
        # samples. JSON encoding that payload on the inference thread can cost
        # several milliseconds and, more importantly, lets ROS serialization
        # backpressure delay the next GPU receipt. Keep one newest report and
        # publish it from a daemon worker; capture stamps make dropped
        # intermediate reports unambiguous to the gateway/tracker.
        self._report_lock = threading.Lock()
        self._report_event = threading.Event()
        self._pending_report: dict[str, Any] | None = None
        threading.Thread(
            target=self._report_publish_loop,
            name="yolo-ros-report",
            daemon=True,
        ).start()
        self._rgb = None
        self._depth = None
        self._rgb_stamp = 0.0
        self._depth_stamp = 0.0
        # ROS overwrites Header.seq with independent publisher counters. Keep
        # them for diagnostics; the normalized capture stamp pairs RGB-D.
        self._rgb_seq = 0
        self._depth_seq = 0
        # RGB and depth are published as two ROS messages for one sensor
        # capture.  Their callbacks can interleave (RGB N with depth N-1), so
        # keep a small diagnostic counter and never lift a mixed pair.  The
        # detector must wait for the matching capture stamp instead of silently
        # applying the previous depth image to the current RGB mask.
        self._pair_mismatch_count = 0
        self._pair_mismatch_last: tuple[Any, ...] | None = None
        self._rgbd_pairs = _ExactRgbdPairBuffer(require_context=bool(args.capture_context_topic))
        self._camera_info = None
        self._depth_camera_info = None
        self._depth_scale = 0.001
        self._depth_to_color_extrinsics: dict[str, Any] = {}
        # ROS callbacks and the inference loop run on different threads.  The
        # worker must consume one coherent, latest-only RGB-D pair instead of
        # repeatedly JPEG/PNG encoding the same callback buffers while waiting
        # for the next message.
        self._frame_lock = threading.Lock()
        self._last_frame_key: tuple[Any, ...] | None = None
        self._telemetry_lock = threading.Lock()
        self._telemetry_history: list[tuple[float, dict[str, Any]]] = []
        # A reconnect can restart the Go2 source clock and sequence.  Keep
        # telemetry history scoped to the same transport generation as the
        # RGB-D stream; otherwise nearest-neighbour lookup can pair a fresh
        # image with an old session's pose at the same wall timestamp.
        self._telemetry_transport_session: str | None = None
        self._telemetry_transport_connection = -1
        self._telemetry_retired_sessions: set[str] = set()
        self._web_report_lock = threading.Lock()
        self._web_report_event = threading.Event()
        self._pending_web_report: dict[str, Any] | None = None
        if self.args.web_url:
            threading.Thread(target=self._web_mirror_loop, name="yolo-web-mirror", daemon=True).start()
        rospy.Subscriber("/physical_nav/rgb/image_raw", Image, self._rgb_callback, queue_size=1)
        rospy.Subscriber("/physical_nav/depth/image_raw", Image, self._depth_callback, queue_size=1)
        if args.capture_context_topic:
            rospy.Subscriber(args.capture_context_topic, String,
                             self._capture_context_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("/physical_nav/camera_info", CameraInfo, self._camera_info_callback, queue_size=1)
        rospy.Subscriber(
            "/physical_nav/depth_camera_info",
            CameraInfo,
            self._depth_camera_info_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/physical_nav/telemetry",
            String,
            self._telemetry_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/physical_nav/depth_calibration",
            String,
            self._calibration_callback,
            queue_size=1,
        )
        print(
            f"YOLOE loaded: {args.model_path} device={args.device} "
            f"conf={self.args.conf:.3f} iou={self.args.iou:.3f} max_det={self.args.max_det} "
            f"geometry_workers={self._geometry_workers} "
            f"capture_context={args.capture_context_topic or 'legacy_telemetry_matching'} "
            f"detection_filter={self.detection_filter.enabled} config={args.detector_config}",
            flush=True,
        )

    def _shutdown_geometry_executor(self) -> None:
        """Stop persistent geometry workers during ROS shutdown."""
        executor = getattr(self, "_geometry_executor", None)
        if executor is None:
            return
        for future in tuple(getattr(self, "_geometry_inflight", ())):
            future.cancel()
        getattr(self, "_geometry_inflight", set()).clear()
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python 3.8 compatibility
            executor.shutdown(wait=False)
        self._geometry_executor = None

    def _rgb_callback(self, msg):
        try:
            channels = 3 if msg.encoding.lower() in ("bgr8", "rgb8") else 1
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            row_width = int(msg.step)
            rows = raw.reshape(int(msg.height), row_width)
            image = rows[:, : int(msg.width) * channels].reshape(int(msg.height), int(msg.width), channels)
            image = image[:, :, ::-1].copy() if msg.encoding.lower() == "rgb8" else image.copy()
            with self._frame_lock:
                self._rgb = image
                self._rgb_stamp = msg.header.stamp.to_sec()
                self._rgb_seq = int(getattr(msg.header, "seq", 0) or 0)
                if not hasattr(self, "_rgbd_pairs"):
                    self._rgbd_pairs = _ExactRgbdPairBuffer()
                self._rgbd_pairs.push("rgb", image, self._rgb_stamp, self._rgb_seq)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "YOLO RGB conversion failed: %s", exc)

    def _depth_callback(self, msg):
        try:
            dtype = np.float32 if msg.encoding.lower() == "32fc1" else np.uint16
            itemsize = np.dtype(dtype).itemsize
            raw = np.frombuffer(msg.data, dtype=dtype)
            rows = raw.reshape(int(msg.height), int(msg.step) // itemsize)
            depth = rows[:, : int(msg.width)].copy()
            with self._frame_lock:
                self._depth = depth
                self._depth_stamp = msg.header.stamp.to_sec()
                self._depth_seq = int(getattr(msg.header, "seq", 0) or 0)
                if not hasattr(self, "_rgbd_pairs"):
                    self._rgbd_pairs = _ExactRgbdPairBuffer()
                self._rgbd_pairs.push("depth", depth, self._depth_stamp, self._depth_seq)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "YOLO depth conversion failed: %s", exc)

    def _capture_context_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict) or payload.get("version") != 1:
                return
            stamp = float(payload["stamp"])
            if not math.isfinite(stamp) or stamp <= 0.:
                return
            for name in ("rgb_intrinsics", "depth_intrinsics"):
                intrinsics = payload[name]
                if not isinstance(intrinsics, dict):
                    return
                for key in ("width", "height", "fx", "fy"):
                    value = float(intrinsics[key])
                    if not math.isfinite(value) or value <= 0.:
                        return
                if not all(math.isfinite(float(intrinsics[key])) for key in ("cx", "cy")):
                    return
            if not isinstance(payload["telemetry"], dict):
                return
            if not isinstance(payload["capture_pose_match"], dict):
                return
            if not isinstance(payload["depth_to_color_extrinsics"], dict):
                return
            if not all(isinstance(payload[name], str) and payload[name]
                       for name in ("camera_frame", "depth_frame")):
                return
            scale = float(payload["depth_scale"])
            if not math.isfinite(scale) or scale <= 0.:
                return
            payload["seq"] = int(payload["seq"])
            with self._frame_lock:
                if self._rgbd_pairs.require_context:
                    self._rgbd_pairs.push("context", payload, stamp, payload["seq"])
        except (TypeError, ValueError, KeyError, OverflowError):
            rospy.logwarn_throttle(5.0, "YOLO capture context conversion failed")

    def _camera_info_callback(self, msg):
        with self._frame_lock:
            self._camera_info = msg

    def _depth_camera_info_callback(self, msg):
        with self._frame_lock:
            self._depth_camera_info = msg

    def _calibration_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return
            depth_scale = float(payload.get("depth_scale", 0.001) or 0.001)
            extrinsics = payload.get("depth_to_color_extrinsics") or {}
            if not math.isfinite(depth_scale) or depth_scale <= 0.0:
                return
            # The latched sensor contract carries the actual tolerance. This
            # also works when YOLO starts before roslaunch loads parameters.
            pose_delta = payload.get("telemetry_max_delta_sec")
            if pose_delta is not None:
                try:
                    pose_delta = float(pose_delta)
                    if not math.isfinite(pose_delta) or pose_delta < 0.:
                        pose_delta = None
                except (TypeError, ValueError, OverflowError):
                    pose_delta = None
                if pose_delta is not None:
                    with self._telemetry_lock:
                        self._telemetry_max_delta_sec = pose_delta
            with self._frame_lock:
                self._depth_scale = depth_scale
                self._depth_to_color_extrinsics = dict(extrinsics)
        except (TypeError, ValueError, json.JSONDecodeError):
            rospy.logwarn_throttle(5.0, "YOLO depth calibration conversion failed")

    def _telemetry_callback(self, msg: String) -> None:
        """Keep a bounded pose history for capture-time 3-D projection."""
        try:
            payload = json.loads(msg.data)
            telemetry = payload.get("telemetry", payload)
            if not isinstance(telemetry, dict):
                return
            stamp = float(payload.get("stamp", telemetry.get("received_at", time.time())))
            if not math.isfinite(stamp) or stamp <= 0.0:
                return
            snapshot = dict(telemetry)
            # The direct sensor bridge attaches private source-generation
            # markers to the nested telemetry object.  Older/replay callers
            # may put them beside that object, so accept both forms.  A
            # tagged stream rejects untagged packets after activation: a
            # delayed legacy socket must not reintroduce a stale pose.
            session_value = snapshot.get("_transport_session", payload.get("_transport_session"))
            session = str(session_value) if session_value not in (None, "") else None
            try:
                connection = int(
                    snapshot.get(
                        "_transport_connection",
                        payload.get("_transport_connection", 0),
                    )
                    or 0
                )
            except (TypeError, ValueError):
                connection = 0
            with self._telemetry_lock:
                current_session = getattr(self, "_telemetry_transport_session", None)
                current_connection = int(
                    getattr(self, "_telemetry_transport_connection", -1)
                )
                retired = getattr(self, "_telemetry_retired_sessions", None)
                if retired is None:
                    retired = set()
                    self._telemetry_retired_sessions = retired
                if session is None:
                    if current_session is not None:
                        return
                elif session in retired:
                    return
                elif current_session is None:
                    self._telemetry_transport_session = session
                    self._telemetry_transport_connection = connection
                    # Discard any legacy/unmarked samples collected before
                    # the first generation marker became available.  They
                    # belong to an unknown source clock and must not be used
                    # to pose a newly tagged RGB-D frame.
                    self._telemetry_history.clear()
                elif session != current_session:
                    retired.add(current_session)
                    self._telemetry_transport_session = session
                    self._telemetry_transport_connection = connection
                    self._telemetry_history.clear()
                elif connection < current_connection:
                    return
                elif connection > current_connection:
                    # Same process/session, but a newer WebSocket connection:
                    # its source clock may also restart, so discard old poses.
                    self._telemetry_transport_connection = connection
                    self._telemetry_history.clear()
                self._telemetry_history.append((stamp, snapshot))
                # Telemetry is normally 5 Hz.  Keep only a few seconds so a
                # reconnect cannot make nearest-neighbour lookup expensive or
                # pair a frame with a stale pose from a previous session.
                if len(self._telemetry_history) > 64:
                    del self._telemetry_history[:-64]
        except (TypeError, ValueError, json.JSONDecodeError):
            rospy.logwarn_throttle(5.0, "YOLO telemetry conversion failed")

    def _telemetry_at(self, stamp: float) -> dict[str, Any]:
        with self._telemetry_lock:
            limit = float(getattr(self, "_telemetry_max_delta_sec", 0.15))
            snapshot, delta = _nearest_capture_telemetry(
                self._telemetry_history, stamp, limit,
            )
            self._last_telemetry_match = {
                "delta_sec": delta, "max_delta_sec": limit,
                "accepted": bool(snapshot),
            }
        return snapshot

    def _queue_web_report(self, report: dict[str, Any]) -> None:
        """Mirror only the newest report without blocking the detector."""
        if not self.args.web_url:
            return
        with self._web_report_lock:
            self._pending_web_report = report
            self._web_report_event.set()

    def _queue_report(self, report: dict[str, Any]) -> None:
        """Queue the newest ROS report without serializing on inference."""
        with self._report_lock:
            self._pending_report = report
            self._report_event.set()

    def _report_publish_loop(self) -> None:
        while not rospy.is_shutdown():
            self._report_event.wait(1.0)
            with self._report_lock:
                report = self._pending_report
                self._pending_report = None
                self._report_event.clear()
            if report is None:
                continue
            try:
                self.report_pub.publish(
                    String(
                        data=json.dumps(
                            report, ensure_ascii=False, separators=(",", ":")
                        )
                    )
                )
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "YOLO ROS report publish failed: %s", exc)

    def _web_mirror_loop(self) -> None:
        while not rospy.is_shutdown():
            self._web_report_event.wait(1.0)
            with self._web_report_lock:
                report = self._pending_web_report
                self._pending_web_report = None
                self._web_report_event.clear()
            if report is None:
                continue
            try:
                _post(self.args.web_url, _compact_web_report(report))
            except Exception:
                # ROS is authoritative; a dashboard outage must not affect
                # the 10 Hz detector loop.
                pass

    def _note_pair_mismatch(self, mismatch):
        # Polling one unmatched capture repeatedly is one diagnostic event.
        if getattr(self, "_pair_mismatch_last", None) != mismatch:
            self._pair_mismatch_count = int(getattr(self, "_pair_mismatch_count", 0)) + 1
            self._pair_mismatch_last = mismatch

    def _latest_raw_frame(self):
        with self._frame_lock:
            rgb, depth = self._rgb, self._depth
            rgb_stamp, depth_stamp = self._rgb_stamp, self._depth_stamp
            rgb_seq, depth_seq = self._rgb_seq, self._depth_seq
            pairs = getattr(self, "_rgbd_pairs", None)
            context = None
            require_context = pairs is not None and pairs.require_context
            if pairs is not None and (require_context or (rgb_seq > 0 and depth_seq > 0)):
                if pairs.latest is None:
                    self._note_pair_mismatch((rgb_seq, depth_seq, rgb_stamp, depth_stamp))
                    return None
                _, stamp, rgb, depth, rgb_seq, depth_seq = pairs.latest[:6]
                if require_context:
                    context = pairs.latest[6]
                rgb_stamp = depth_stamp = stamp
            info = self._camera_info
            depth_info = self._depth_camera_info
            depth_scale = float(self._depth_scale)
            depth_to_color_extrinsics = dict(self._depth_to_color_extrinsics)
            # Republish/reconnect can change the topic counters for one
            # capture. It is still the same input, not another observation.
            frame_key = (rgb_stamp, depth_stamp)
            if self._last_frame_key == frame_key:
                return None
            # Exact stamps are shared across the two sensor publications;
            # publisher counters need not match (e.g. RGB-only subscribers
            # existed before YOLO connected). Never relax the live timestamp
            # check just because topic sequence counters happen to match.
            if rgb_seq > 0 and depth_seq > 0 and rgb_stamp != depth_stamp:
                mismatch = (rgb_stamp, depth_stamp)
                # The inference loop polls faster than the camera. Count a
                # dropped pair once, rather than turning one callback race
                # into hundreds of misleading mismatch events.
                self._note_pair_mismatch(mismatch)
                return None
            if rgb is None or depth is None or abs(rgb_stamp - depth_stamp) > 0.12:
                return None
        if context is not None:
            for name, array in (("rgb_intrinsics", rgb), ("depth_intrinsics", depth)):
                if (context[name]["height"], context[name]["width"]) != array.shape[:2]:
                    rospy.logwarn_throttle(5.0, "YOLO capture context image dimensions disagree")
                    return None
        elif info is None or depth_info is None:
            return None
        # Callback buffers are copied on receipt, so retaining these references
        # is safe and avoids another full-resolution copy.  Mark the pair only
        # after camera info is available; a latched-info race must be retried.
        with self._frame_lock:
            self._last_frame_key = frame_key
            # A complete pair closes the previous mismatch episode. If the
            # same callback race happens after a later capture, count it as a
            # new diagnostic event.
            self._pair_mismatch_last = None
        stamp = max(rgb_stamp, depth_stamp)
        if context is not None:
            return {
                **context,
                "stamp": stamp,
                "rgb_array": rgb, "depth_array": depth,
                "rgb_seq": int(rgb_seq), "depth_seq": int(depth_seq),
            }
        def intrinsics(camera_info):
            return {
                "width": int(camera_info.width), "height": int(camera_info.height),
                "fx": float(camera_info.K[0]), "fy": float(camera_info.K[4]),
                "cx": float(camera_info.K[2]), "cy": float(camera_info.K[5]),
                "distortion_model": str(camera_info.distortion_model or ""),
                "distortion": [float(value) for value in camera_info.D],
            }
        telemetry = self._telemetry_at(stamp)
        return {
            "seq": int(max(rgb_seq, depth_seq)) if rgb_seq > 0 and depth_seq > 0 else int(round(stamp * 1000.0)),
            "stamp": stamp,
            "rgb_array": rgb,
            "depth_array": depth,
            "rgb_intrinsics": intrinsics(info),
            "depth_intrinsics": intrinsics(depth_info),
            "depth_scale": depth_scale,
            "depth_to_color_extrinsics": depth_to_color_extrinsics,
            "camera_frame": str(info.header.frame_id or "d435i_color_optical_frame"),
            "depth_frame": str(depth_info.header.frame_id or "d435i_depth_optical_frame"),
            "telemetry": telemetry,
            "capture_pose_match": dict(getattr(self, "_last_telemetry_match", {})),
            "rgb_seq": int(rgb_seq),
            "depth_seq": int(depth_seq),
        }

    def _claim_frame(self, raw: dict[str, Any]) -> bool:
        """Accept each capture once, including after the dog bridge restarts.

        The bridge sequence is process-local and returns to zero on restart.
        Capture timestamps remain monotonic across that restart, so they are
        the primary receipt identity; sequence is retained as a fallback for
        sources that do not provide a usable timestamp.
        """
        if not (raw.get("rgb_array") is not None or raw.get("rgb")) or not (raw.get("depth_array") is not None or raw.get("depth")):
            return False
        seq = int(raw.get("seq", -1))
        try:
            stamp = float(raw.get("stamp", 0.0))
        except (TypeError, ValueError):
            stamp = 0.0
        has_stamp = math.isfinite(stamp) and stamp > 0.0
        if has_stamp and seq > 0:
            # Prefer the bridge-local sequence.  A lower stamp is expected
            # after a Go2 restart/clock step and must not suppress the frame.
            if seq == self.last_seq and stamp <= self.last_stamp:
                return False
            if seq < self.last_seq:
                # A legacy/restarted ROS publisher may reset Header.seq.  If
                # its stamp also rolls back, accept the first frame as a new
                # generation; subsequent duplicates are still rejected by the
                # equality check above.  This is intentionally only enabled
                # for stamped frames, preserving un-stamped bag semantics.
                if stamp > self.last_stamp + 1e-6:
                    self.last_seq = -1
                    self.last_stamp = float("-inf")
                elif (
                    stamp > self.last_stamp - 1.0
                    or seq > max(32, self.last_seq // 2)
                ):
                    return False
                else:
                    self.last_seq = -1
                    self.last_stamp = float("-inf")
        elif has_stamp:
            if stamp <= self.last_stamp:
                return False
        elif seq <= self.last_seq:
            return False
        if self.last_seq >= 0 and seq < self.last_seq:
            print(
                f"YOLOE source sequence reset: {self.last_seq} -> {seq}; "
                f"continuing from capture stamp {stamp:.6f}",
                flush=True,
            )
        self.last_seq = seq
        if has_stamp:
            self.last_stamp = stamp
        return True

    def _projection_maps(
        self,
        depth_shape: tuple[int, int],
        mask_shape: tuple[int, int],
        rgb_intr: dict[str, Any],
        depth_intr: dict[str, Any],
        extr: dict[str, Any],
        depth_values: np.ndarray | None = None,
        depth_scale: float = 1.0,
        projection_roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build RGB->depth pixel geometry.

        A rigid translation between the D435i colour/depth sensors makes the
        projection depth-dependent.  Geometry with a non-zero baseline is
        therefore computed from the current depth image and is not put in the
        cross-frame cache.  Pure rotation (or an already aligned stream) is
        depth-independent and can still use the cached z=1 map.
        """
        sx = mask_shape[1] / max(float(rgb_intr.get("width", mask_shape[1])), 1.0)
        sy = mask_shape[0] / max(float(rgb_intr.get("height", mask_shape[0])), 1.0)
        rotation_value = extr.get("rotation")
        if rotation_value is None:
            rotation_value = np.eye(3).reshape(-1)
        translation_value = extr.get("translation")
        if translation_value is None:
            translation_value = (0, 0, 0)
        roi_key = None if projection_roi is None else tuple(int(value) for value in projection_roi)
        key = (depth_shape, mask_shape, roi_key, tuple(round(float(rgb_intr.get(k, 0)) * (sx if k in ("fx", "cx") else sy), 6) for k in ("fx", "fy", "cx", "cy")), tuple(round(float(depth_intr.get(k, 0)), 6) for k in ("fx", "fy", "cx", "cy")), tuple(round(float(v), 6) for v in rotation_value), tuple(round(float(v), 6) for v in translation_value))
        if (
            depth_values is None
            and key == self._projection_cache_key
            and self._projection_cache is not None
        ):
            return self._projection_cache
        result = _project_depth_grid(
            depth_shape,
            mask_shape,
            rgb_intr,
            depth_intr,
            extr,
            depth_values=depth_values,
            depth_scale=depth_scale,
            projection_roi=projection_roi,
        )
        if depth_values is None:
            self._projection_cache_key, self._projection_cache = key, result
        return result

    def infer(self, raw: dict[str, Any]) -> dict[str, Any]:
        infer_started = time.perf_counter()
        transport_metadata = _transport_report_metadata(raw)
        decode_started = time.perf_counter()
        # Live ROS frames are already decoded by the sensor bridge.  Keep the
        # encoded fallback for replay/HTTP fixtures, but never round-trip a
        # current frame through JPEG/PNG merely to call YOLOE.
        rgb = raw.get("rgb_array")
        depth = raw.get("depth_array")
        if rgb is None:
            rgb = _decode(raw["rgb"])
        if depth is None:
            depth = _decode(raw["depth"])
        depth = np.asarray(depth).astype(np.float32, copy=False)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        rgb_intr = raw.get("rgb_intrinsics") or raw.get("intrinsics", {})
        transport_metadata.update(_capture_observation_metadata(raw, rgb.shape))
        depth_intr = raw.get("depth_intrinsics") or raw.get("intrinsics", {})
        depth_frame = str(raw.get("depth_frame") or raw.get("camera_frame", "d435i_depth_optical_frame"))
        fx, fy, cx, cy = [float(depth_intr.get(k, 0)) for k in ("fx", "fy", "cx", "cy")]
        predict_started = time.perf_counter()
        result = self.model.predict(source=rgb, device=self.args.device, imgsz=self.args.imgsz, conf=self.args.conf, iou=self.args.iou, max_det=self.args.max_det, verbose=False, save=False)[0]
        predict_ms = (time.perf_counter() - predict_started) * 1000.0
        boxes = getattr(result, "boxes", None); masks = getattr(result, "masks", None); detections = []; debug_point_sets = []
        if boxes is None:
            return {
                "seq": raw["seq"],
                "stamp": raw["stamp"],
                **transport_metadata,
                "camera_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")),
                "model": self.args.model_path,
                "inference_ms": (time.perf_counter() - infer_started) * 1000.0,
                "overlay_jpeg": _encode_detection_overlay(rgb, []),
                "detections": [],
            }
        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy); confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf); classes = boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls, dtype=int); names = getattr(result, "names", {})
        # Keep the model tensor on its original device until confidence/class
        # filtering and the geometry cap have selected the few masks that can
        # actually be lifted.  The previous eager ``masks.data.cpu()`` copied
        # every instance on crowded frames, even though most were serialized
        # as 2-D-only records.
        mask_source = getattr(masks, "data", None) if masks is not None else None
        mask_shape = _mask_grid_shape(mask_source)
        model_mask_polygons = None
        model_mask_polygons_loaded = False
        depth_to_color = raw.get("depth_to_color_extrinsics") or {}
        aligned_stream = str(raw.get("depth_frame") or "") == str(
            raw.get("camera_frame") or ""
        )
        projection_extrinsics = {} if aligned_stream else depth_to_color
        capture_pose_error = _capture_pose_error(raw.get("telemetry"))
        tf_status = raw.get("capture_tf_status", "unchecked")
        if tf_status not in {"unchecked", "published", "duplicate"}:
            capture_pose_error = str(tf_status)
        world_transform = None
        if capture_pose_error is None:
            world_transform, capture_pose_error = _context_world_transform(raw)
        if capture_pose_error is None and world_transform is None:
            world_transform = _world_transform(
                raw.get("telemetry", {}),
                self.translation,
                (self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw),
                optical_frame=True,
                depth_to_color_extrinsics=(
                    depth_to_color if depth_frame != str(raw.get("camera_frame") or "") else None
                ),
                camera_imu_enabled=getattr(self.args, "camera_imu_enabled", True),
                camera_imu_use_yaw=getattr(self.args, "camera_imu_use_yaw", False),
                camera_imu_max_roll_rad=getattr(self.args, "camera_imu_max_roll_rad", 0.7),
                camera_imu_max_pitch_rad=getattr(self.args, "camera_imu_max_pitch_rad", 0.7),
                camera_imu_max_yaw_rad=getattr(self.args, "camera_imu_max_yaw_rad", 0.35),
            )
        projection_maps = None
        if capture_pose_error is None and mask_shape is not None and (
            mask_shape != depth.shape or not aligned_stream
        ):
            translation = np.asarray(
                depth_to_color.get("translation") or (0.0, 0.0, 0.0),
                dtype=np.float32,
            ).reshape(-1)
            has_baseline = translation.size >= 3 and bool(
                np.linalg.norm(translation[:3]) > 1e-6
            )
            depth_scale = float(raw.get("depth_scale", 0.001) or 0.001)
            # A depth-aligned stream already lives in the colour pixel
            # geometry.  Otherwise use the calibrated extrinsics; when the
            # baseline is non-zero the map must be evaluated with this frame's
            # actual depth values rather than the cached z=1 approximation.
            projection_roi = (
                _depth_projection_roi(xyxy, depth.shape, rgb_intr)
                if has_baseline and not aligned_stream
                else None
            )
            projection_maps = self._projection_maps(
                depth.shape,
                mask_shape,
                rgb_intr,
                depth_intr,
                projection_extrinsics,
                depth_values=(
                    depth
                    if has_baseline and not aligned_stream
                    else None
                ),
                depth_scale=depth_scale,
                projection_roi=projection_roi,
            )
        post_started = time.perf_counter()
        post_parts = {
            "mask_transfer": 0.0,
            "mask_projection": 0.0,
            "depth_geometry": 0.0,
            "contours": 0.0,
            "serialization": 0.0,
            "polygon_materialization": 0.0,
        }
        # Filter and rank before RGB-D lifting.  YOLOE can return many
        # low-value scene labels; limiting the expensive geometry stage keeps
        # the detector real-time while prioritising interaction targets.
        candidates: list[tuple[int, int, float, dict[str, Any], str]] = []
        interaction_labels = {"door", "portal", "fridge", "refrigerator", "locker"}
        for index, _box in enumerate(xyxy):
            raw_name = names.get(int(classes[index]), str(int(classes[index]))) if isinstance(names, dict) else str(int(classes[index]))
            if _is_excluded_scene_label(raw_name, self.detector_config):
                continue
            filtered_label = self.detection_filter.apply_one(
                {"semantic_class_raw": str(raw_name), "semantic_class": _label(raw_name)}
            )
            if filtered_label is None:
                continue
            threshold = self.class_confidence_thresholds.get(
                str(filtered_label["semantic_class"]).casefold(),
                float(self.detector_config.get("confidence_threshold", self.args.conf)),
            )
            if float(confs[index]) < threshold:
                continue
            priority = 0 if str(filtered_label["semantic_class"]).casefold() in interaction_labels else 1
            candidates.append((priority, index, float(confs[index]), filtered_label, str(raw_name)))
        max_geometry_instances = max(1, int(self.detector_config.get("max_geometry_instances", 24)))
        candidates.sort(key=lambda item: (item[0], -item[2]))
        # Keep all interaction candidates at the front of every receipt, but
        # rotate ordinary scene candidates across receipts.  A hard per-frame
        # cap is necessary for 10 Hz RGB-D lifting; without this cursor an
        # object that ranks below the first twelve boxes would never acquire a
        # 3-D observation and could never enter the persistent semantic map.
        interaction_candidates = [item for item in candidates if item[0] == 0]
        ordinary_candidates = [item for item in candidates if item[0] != 0]
        geometry_candidates = interaction_candidates[:max_geometry_instances]
        remaining_slots = max(0, max_geometry_instances - len(geometry_candidates))
        selected_ordinary: list[tuple[int, int, float, dict[str, Any], str]] = []
        if remaining_slots and ordinary_candidates:
            cursor = int(getattr(self, "_geometry_cursor", 0)) % len(ordinary_candidates)
            selected_count = min(remaining_slots, len(ordinary_candidates))
            selected_ordinary = [
                ordinary_candidates[(cursor + offset) % len(ordinary_candidates)]
                for offset in range(selected_count)
            ]
            self._geometry_cursor = (cursor + selected_count) % len(ordinary_candidates)
            geometry_candidates.extend(selected_ordinary)
        elif not ordinary_candidates:
            self._geometry_cursor = 0
        selected_indices = {int(item[1]) for item in geometry_candidates}
        skipped_candidates = [
            item for item in candidates if int(item[1]) not in selected_indices
        ]
        # Selected candidates can still fail the depth/box admission gates.
        # Keep those as 2-D-only records so a bad depth patch does not make
        # the detector receipt appear to lose an object; the gateway filters
        # ``geometry_skipped`` before semantic-map admission.
        geometry_failed: list[
            tuple[tuple[int, int, float, dict[str, Any], str], str]
        ] = []
        if capture_pose_error is not None:
            # Keep detector recall/overlay while excluding fabricated world
            # boxes from tracking. Do not transfer/project masks without pose.
            geometry_candidates = []
            skipped_candidates = list(candidates)
        geometry_candidates_before_budget = list(geometry_candidates)
        geometry_budget_ms = max(
            0.0, float(self.detector_config.get("geometry_budget_ms", 75.0))
        )
        geometry_budget_sec = geometry_budget_ms / 1000.0
        geometry_skip_reason = capture_pose_error or "max_geometry_instances"
        geometry_loop_started = time.perf_counter()
        geometry_deadline = (
            geometry_loop_started + geometry_budget_sec
            if geometry_budget_sec > 0.0
            else None
        )
        geometry_skipped_records = [
            (candidate, geometry_skip_reason) for candidate in skipped_candidates
        ]
        # These options are frame-constant.  Reusing one read-only mapping
        # avoids copying the complete detector YAML once per candidate; the
        # semantic label is already passed separately to
        # ``_prepare_instance_points`` for planar-depth filtering.
        geometry_config = dict(self.detector_config)
        geometry_config.update(
            {
                "point_stride": self.args.point_stride,
                "max_depth_m": self.args.max_depth_m,
                "min_valid_points": self.args.min_valid_points,
                "depth_scale": float(raw.get("depth_scale", 0.001) or 0.001),
            }
        )
        mask_transfer_started = time.perf_counter()
        selected_masks = _materialize_selected_masks(
            mask_source,
            [int(item[1]) for item in geometry_candidates_before_budget],
        )
        post_parts["mask_transfer"] = (
            time.perf_counter() - mask_transfer_started
        ) * 1000.0
        # Submit only the stateless 3-D lifting stage.  Polygon materialization
        # remains on this thread because Ultralytics' lazy ``masks.xy`` cache
        # is not guaranteed to be thread-safe.  The deadline applies to both
        # submission and result collection: unfinished work is abandoned at
        # the deadline and never awaited by this or a later inference cycle.
        geometry_executor = getattr(self, "_geometry_executor", None)
        geometry_workers = max(1, int(getattr(self, "_geometry_workers", 1)))
        inflight_geometry = getattr(self, "_geometry_inflight", None)
        if not isinstance(inflight_geometry, set):
            inflight_geometry = set()
            self._geometry_inflight = inflight_geometry
        reaped_geometry = 0
        reaped_geometry_errors = 0
        for future in tuple(inflight_geometry):
            if not future.done():
                continue
            inflight_geometry.discard(future)
            reaped_geometry += 1
            try:
                # Consume worker exceptions even though an expired frame's
                # result is intentionally never admitted to the current map.
                future.result()
            except Exception:
                reaped_geometry_errors += 1
        available_geometry_workers = max(
            0, geometry_workers - len(inflight_geometry)
        )
        pending_geometry: dict[Future[Any], tuple[int, tuple[Any, ...]]] = {}
        completed_geometry: list[tuple[int, tuple[Any, ...], dict[str, Any]]] = []
        submitted_count = 0
        timed_out_count = 0
        worker_error_count = 0
        deadline_wait_started = time.perf_counter()

        def _submit_geometry(candidate_index: int, candidate: tuple[Any, ...]) -> bool:
            nonlocal submitted_count
            _priority, index, _confidence, filtered_label, raw_name = candidate
            box = xyxy[index]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            mask = selected_masks.get(int(index))
            polygon_started = time.perf_counter()
            rgb_mask_polygons: list[list[list[int]]] = []
            if mask is not None:
                source_polygons = None
                try:
                    if not nonlocal_model_loaded[0]:
                        materialize_started = time.perf_counter()
                        # ``masks.xy`` may be lazy; materialize it once on the
                        # inference thread before worker threads read entries.
                        nonlocal_model_polygons[0] = getattr(masks, "xy", None)
                        nonlocal_model_loaded[0] = True
                        post_parts["polygon_materialization"] += (
                            time.perf_counter() - materialize_started
                        ) * 1000.0
                    if nonlocal_model_polygons[0] is not None and index < len(nonlocal_model_polygons[0]):
                        source_polygons = nonlocal_model_polygons[0][index]
                except (TypeError, IndexError):
                    source_polygons = None
                if source_polygons is not None:
                    rgb_mask_polygons = _build_rgb_mask_polygons(
                        mask, source_polygons, rgb.shape
                    )
                else:
                    display_mask = cv2.resize(
                        (mask > 0.5).astype(np.uint8),
                        (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    display_contours, _ = cv2.findContours(
                        display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for display_contour in sorted(
                        display_contours, key=cv2.contourArea, reverse=True
                    ):
                        if display_contour.shape[0] >= 3:
                            approx = cv2.approxPolyDP(
                                display_contour,
                                max(1.0, 0.0025 * cv2.arcLength(display_contour, True)),
                                True,
                            )
                            if approx.shape[0] >= 3:
                                rgb_mask_polygons.append(
                                    approx[:, 0, :].astype(int).tolist()
                                )
            polygon_ms = (time.perf_counter() - polygon_started) * 1000.0
            task_args = (
                mask,
                depth,
                rgb_intr,
                depth_intr,
                projection_extrinsics,
                projection_maps,
                (x1, y1, x2, y2),
                float(raw.get("depth_scale", 0.001) or 0.001),
                aligned_stream,
                (fx, fy, cx, cy),
                geometry_config,
                str(filtered_label["semantic_class"]),
                world_transform,
            )
            metadata = (candidate, (x1, y1, x2, y2), rgb_mask_polygons, polygon_ms)
            if geometry_executor is None:
                result = _geometry_lift_candidate(*task_args)
                completed_geometry.append((candidate_index, metadata, result))
            else:
                # Polygon extraction itself is synchronous.  If it consumed
                # the remaining budget, do not enqueue a stale 3-D task.
                if geometry_deadline is not None and time.perf_counter() >= geometry_deadline:
                    post_parts["mask_projection"] += polygon_ms
                    return False
                future = geometry_executor.submit(
                    _geometry_lift_candidate, *task_args
                )
                pending_geometry[future] = (candidate_index, metadata)
            submitted_count += 1
            return True

        def _collect_finished(futures: set[Future[Any]]) -> None:
            nonlocal worker_error_count
            for future in futures:
                candidate_index, metadata = pending_geometry.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    worker_error_count += 1
                    result = {
                        "ok": False,
                        "reason": f"geometry_worker_error:{type(exc).__name__}",
                        "mask_projection_ms": 0.0,
                        "depth_geometry_ms": 0.0,
                    }
                completed_geometry.append((candidate_index, metadata, result))

        # Mutable cells let the nested submit function avoid racing the lazy
        # polygon property while preserving the existing timing field.
        nonlocal_model_polygons = [model_mask_polygons]
        nonlocal_model_loaded = [model_mask_polygons_loaded]
        next_candidate_index = 0
        if geometry_executor is None and geometry_deadline is not None:
            # A synchronous fallback cannot enforce a wall-clock deadline.
            # Production always owns an executor; fail closed if it has been
            # shut down instead of blocking the detector thread indefinitely.
            geometry_skipped_records.extend(
                (candidate, "geometry_executor_unavailable")
                for candidate in geometry_candidates_before_budget
            )
        elif geometry_executor is None:
            # ``geometry_budget_ms: 0`` deliberately means unlimited and is
            # retained for deterministic offline/replay tests.
            for candidate_index, candidate in enumerate(
                geometry_candidates_before_budget
            ):
                _submit_geometry(candidate_index, candidate)
            next_candidate_index = len(geometry_candidates_before_budget)
        elif available_geometry_workers <= 0:
            geometry_skipped_records.extend(
                (candidate, "geometry_workers_busy")
                for candidate in geometry_candidates_before_budget
            )
        else:
            while (
                next_candidate_index < len(geometry_candidates_before_budget)
                or pending_geometry
            ):
                while (
                    next_candidate_index < len(geometry_candidates_before_budget)
                    and len(pending_geometry) < available_geometry_workers
                ):
                    if (
                        geometry_deadline is not None
                        and time.perf_counter() >= geometry_deadline
                    ):
                        break
                    candidate = geometry_candidates_before_budget[next_candidate_index]
                    if not _submit_geometry(next_candidate_index, candidate):
                        break
                    next_candidate_index += 1
                if not pending_geometry:
                    break
                timeout = None
                if geometry_deadline is not None:
                    timeout = max(0.0, geometry_deadline - time.perf_counter())
                    if timeout <= 0.0:
                        break
                done, _not_done = wait(
                    tuple(pending_geometry),
                    timeout=timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    break
                _collect_finished(done)

            # Race-safe nonblocking collection at the boundary.  Futures that
            # really remain active retain at most one slot each; queued tasks
            # are cancelled and no stale result can leak into another frame.
            done_at_deadline = {
                future for future in pending_geometry if future.done()
            }
            if done_at_deadline:
                _collect_finished(done_at_deadline)
            for future, (_candidate_index, metadata) in tuple(pending_geometry.items()):
                candidate = metadata[0]
                post_parts["mask_projection"] += float(metadata[3])
                pending_geometry.pop(future)
                timed_out_count += 1
                geometry_skipped_records.append(
                    (candidate, "geometry_budget_ms")
                )
                if not future.cancel():
                    inflight_geometry.add(future)
                    self._geometry_abandoned_total = int(
                        getattr(self, "_geometry_abandoned_total", 0)
                    ) + 1
            if next_candidate_index < len(geometry_candidates_before_budget):
                remaining = geometry_candidates_before_budget[next_candidate_index:]
                timed_out_count += len(remaining)
                geometry_skipped_records.extend(
                    (candidate, "geometry_budget_ms") for candidate in remaining
                )

        post_parts["geometry_budget_ms"] = geometry_budget_ms
        post_parts["geometry_wait_ms"] = (
            time.perf_counter() - deadline_wait_started
        ) * 1000.0
        post_parts["geometry_submitted"] = float(submitted_count)
        post_parts["geometry_completed"] = float(len(completed_geometry))
        post_parts["geometry_timed_out"] = float(timed_out_count)
        post_parts["geometry_inflight"] = float(len(inflight_geometry))
        post_parts["geometry_abandoned_total"] = float(
            getattr(self, "_geometry_abandoned_total", 0)
        )
        post_parts["geometry_reaped"] = float(reaped_geometry)
        post_parts["geometry_worker_errors"] = float(
            worker_error_count + reaped_geometry_errors
        )
        completed_geometry.sort(key=lambda item: item[0])
        geometry_candidates = [
            geometry_candidates_before_budget[candidate_index]
            for candidate_index, _metadata, _result in completed_geometry
        ]
        for _candidate_index, metadata, result in completed_geometry:
            candidate, (x1, y1, x2, y2), rgb_mask_polygons, polygon_ms = metadata
            _priority, index, _confidence, filtered_label, raw_name = candidate
            post_parts["mask_projection"] += polygon_ms + float(result.get("mask_projection_ms", 0.0))
            post_parts["depth_geometry"] += float(result.get("depth_geometry_ms", 0.0))
            if not result.get("ok"):
                geometry_failed.append((candidate, str(result.get("reason", "geometry_failed"))))
                continue
            mask = result["mask"]
            points = result["points"]
            values = result["values"]
            world = result["world"]
            contour_started = time.perf_counter()
            contours, _hierarchy = cv2.findContours(
                _mask_uint8_view(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            mask_polygons: list[list[list[int]]] = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                epsilon = max(1.0, 0.0025 * cv2.arcLength(contour, True))
                approximated = cv2.approxPolyDP(contour, epsilon, True)
                if approximated.shape[0] >= 3:
                    mask_polygons.append(approximated[:, 0, :].astype(int).tolist())
            post_parts["contours"] += (time.perf_counter() - contour_started) * 1000.0
            sparse_rows, sparse_cols = _compact_mask_coordinates(mask, 3000)
            debug_point_sets.append((points, world))
            detections.append({"semantic_class": filtered_label["semantic_class"], "semantic_class_raw": filtered_label["semantic_class_raw"], "raw_class": str(raw_name), "possible_interaction_class": "fridge" if str(filtered_label["semantic_class"]).casefold() == "locker" else None, "m1_verification_required": str(filtered_label["semantic_class"]).casefold() == "locker", "confidence": float(confs[index]), "bbox": [x1, y1, x2, y2], "mask": {"rows": sparse_rows.astype(int).tolist(), "cols": sparse_cols.astype(int).tolist()}, "mask_polygon": mask_polygons[0] if mask_polygons else [], "mask_polygons": mask_polygons, "rgb_mask_polygons": rgb_mask_polygons, "mask_area": int(np.count_nonzero(mask)), "depth_median_m": float(np.median(values)), "depth_valid_points": int(values.size), "camera_position": {"x": float(result["camera_center"][0]), "y": float(result["camera_center"][1]), "z": float(result["camera_center"][2])}, "camera_box3d_center": result["camera_center"].astype(float).tolist(), "camera_box3d_size": result["camera_size"].astype(float).tolist(), "camera_obb_center": result["camera_obb_center"].astype(float).tolist(), "camera_obb_size": result["camera_obb_size"].astype(float).tolist(), "camera_obb_orientation": result["camera_obb_orientation"], "position": {"x": float(result["center"][0]), "y": float(result["center"][1]), "z": float(result["center"][2])}, "world_position": {"x": float(result["center"][0]), "y": float(result["center"][1]), "z": float(result["center"][2])}, "aabb_center": result["center"].astype(float).tolist(), "aabb_size": result["size"].astype(float).tolist(), "world_aabb_min_z": result["world_aabb_min_z"], "world_aabb_max_z": result["world_aabb_max_z"], "box3d_center": result["obb_center"].astype(float).tolist(), "box3d_size": result["obb_size"].astype(float).tolist(), "world_box3d_center": result["obb_center"].astype(float).tolist(), "world_box3d_size": result["obb_size"].astype(float).tolist(), "world_box3d_marker_size": result["obb_size"].astype(float).tolist(), "world_box3d_orientation": result["obb_orientation"], "yaw": float(result["obb_yaw"]), "world_box3d_yaw": float(result["obb_yaw"]), "source_frame": depth_frame, "source_model": self.args.model_path, "projection_method": "physical_yoloe_rgbd_mask_obb", "map_transform_status": "telemetry_fallback", "capture_seq": int(raw["seq"]), "stamp": float(raw["stamp"])})
        debug_max_points = max(32, int(self.detector_config.get("debug_cloud_max_points_per_instance", 400)))
        serialization_started = time.perf_counter()
        for detection, (point_set, world_point_set) in zip(detections, debug_point_sets):
            stride = max(1, int(math.ceil(float(point_set.shape[0]) / debug_max_points)))
            # These points come from the exact RGB-D receipt used by YOLOE;
            # RViz therefore never pairs a segmentation mask with newer depth.
            camera_debug = point_set[::stride][:debug_max_points]
            world_debug = world_point_set[::stride][:debug_max_points]
            detection["segment_point_count"] = int(camera_debug.shape[0])
            detection["camera_segment_points_f32"] = _encode_point_rows(camera_debug)
            detection["world_segment_points_f32"] = _encode_point_rows(world_debug)
        post_parts["serialization"] = (time.perf_counter() - serialization_started) * 1000.0
        # Preserve accepted 2-D detections even when their 3-D lift was
        # intentionally capped.  Downstream graph/point-cloud consumers can
        # require a world box, while the overlay and future frame-level
        # consumers still retain the detector's full recall.
        skipped_records = geometry_skipped_records + geometry_failed
        for (
            (_priority, index, _confidence, filtered_label, raw_name),
            skip_reason,
        ) in skipped_records:
            box = xyxy[index]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            detections.append(
                {
                    "semantic_class": filtered_label["semantic_class"],
                    "semantic_class_raw": filtered_label["semantic_class_raw"],
                    "raw_class": str(raw_name),
                    "confidence": float(confs[index]),
                    "bbox": [x1, y1, x2, y2],
                    "geometry_skipped": True,
                    "geometry_skip_reason": skip_reason,
                    "capture_seq": int(raw["seq"]),
                    "stamp": float(raw["stamp"]),
                    "source_frame": depth_frame,
                    "source_model": self.args.model_path,
                }
            )
        post_parts["geometry_candidates"] = float(len(geometry_candidates))
        post_parts["geometry_skipped"] = float(len(skipped_records))
        post_ms = (time.perf_counter() - post_started) * 1000.0
        overlay_started = time.perf_counter()
        overlay_jpeg = _encode_detection_overlay(
            rgb,
            detections,
            int(self.detector_config.get("debug_overlay_jpeg_quality", 92)),
        )
        overlay_ms = (time.perf_counter() - overlay_started) * 1000.0
        total_ms = (time.perf_counter() - infer_started) * 1000.0
        timings_ms = {"decode": decode_ms, "model_predict": predict_ms, "postprocess": post_ms, "overlay_encode": overlay_ms, "total": total_ms, "post_parts": post_parts}
        return {
            "seq": raw["seq"],
            "stamp": raw["stamp"],
            **transport_metadata,
            "camera_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")),
            "depth_frame": depth_frame,
            "model": self.args.model_path,
            "inference_ms": total_ms,
            "timings_ms": timings_ms,
            "capture_pose_status": capture_pose_error or "valid",
            "capture_pose_match": dict(raw.get("capture_pose_match") or {}),
            "capture_tf_status": raw.get("capture_tf_status", "unchecked"),
            "capture_tf_conflict_count": raw.get("capture_tf_conflict_count", 0),
            "camera_imu_match": raw.get("camera_imu_match", {"status": "unchecked"}),
            "overlay_jpeg": overlay_jpeg,
            "detections": detections,
        }

    def run(self) -> None:
        profile_started = time.monotonic()
        profile_count = 0
        profile_totals: dict[str, float] = {}
        last_input_warning = time.monotonic()
        while True:
            cycle_started = time.monotonic()
            raw = None
            try:
                raw = self._latest_raw_frame()
                if raw is None:
                    if time.monotonic() - last_input_warning >= 10.0:
                        print(
                            "YOLOE waiting: no usable new ROS RGB-D pair; "
                            "check /physical_nav/rgb/image_raw, /physical_nav/depth/image_raw "
                            "and capture_context (model warmup already passed)",
                            flush=True,
                        )
                        last_input_warning = time.monotonic()
                    time.sleep(0.01)
                    continue
                last_input_warning = time.monotonic()
                if not self._claim_frame(raw): time.sleep(.02); continue
                report = self.infer(raw)
                cycle_ms = (time.monotonic() - cycle_started) * 1000.0
                report["cycle_ms"] = cycle_ms
                # ROS is the authoritative detector output.  The HTTP post is
                # retained only as an optional dashboard mirror for backwards
                # compatibility; a missing web server must never stop YOLO.
                self._queue_report(report)
                self._queue_web_report(report)
                profile_count += 1
                for name, value in (report.get("timings_ms") or {}).items():
                    if isinstance(value, (int, float)):
                        profile_totals[name] = profile_totals.get(name, 0.0) + float(value)
                if time.monotonic() - profile_started >= 10.0:
                    averages = {name: round(value / max(profile_count, 1), 1) for name, value in profile_totals.items()}
                    print(f"YOLOE profile n={profile_count} avg_ms={averages} last_cycle_ms={cycle_ms:.1f} last_parts={(report.get('timings_ms') or {}).get('post_parts', {})}", flush=True)
                    profile_started, profile_count, profile_totals = time.monotonic(), 0, {}
                # ``rate`` is a cycle target, not an additional post-inference
                # delay. The former fixed sleep halved a 100 ms pipeline from
                # about 9 Hz to about 4.5 Hz.
                elapsed = time.monotonic() - cycle_started
                time.sleep(max(0., 1. / self.args.rate - elapsed))
            except Exception as exc:
                # A malformed mask/depth sample must not make the detector
                # topic go silent.  Publish an explicit empty receipt for the
                # captured frame so downstream latest-only consumers advance
                # their timestamp and do not retain an apparently current
                # stale box forever.  Geometry errors are isolated from the
                # next GPU inference cycle; the error field is diagnostic and
                # never admitted to semantic mapping as a detection.
                print(f"YOLOE worker warning: {exc}", flush=True)
                if isinstance(raw, dict):
                    try:
                        fallback_rgb = raw.get("rgb_array")
                        fallback_overlay = (
                            _encode_detection_overlay(fallback_rgb, [])
                            if isinstance(fallback_rgb, np.ndarray)
                            else ""
                        )
                        fallback = {
                            "seq": raw.get("seq", -1),
                            "stamp": raw.get("stamp", 0.0),
                            **_transport_report_metadata(raw),
                            "camera_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")),
                            "depth_frame": str(raw.get("depth_frame", "")),
                            "model": self.args.model_path,
                            "inference_ms": (time.monotonic() - cycle_started) * 1000.0,
                            "timings_ms": {"total": (time.monotonic() - cycle_started) * 1000.0},
                            "overlay_jpeg": fallback_overlay,
                            "detections": [],
                            "inference_error": str(exc)[:240],
                        }
                        self._queue_report(fallback)
                        self._queue_web_report(fallback)
                    except Exception as fallback_exc:
                        print(f"YOLOE fallback report warning: {fallback_exc}", flush=True)
                time.sleep(min(max(float(self.args.retry_s), 0.0), 1.0 / max(float(self.args.rate), 1e-3)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--web-url", default="http://127.0.0.1:8765")
    p.add_argument(
        "--model-path",
        default="/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt",
    )
    p.add_argument("--detector-config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=.35)
    p.add_argument("--iou", type=float, default=.7)
    p.add_argument("--max-det", type=int, default=50)
    p.add_argument("--rate", type=float, default=10.)
    p.add_argument("--retry-s", type=float, default=1.)
    p.add_argument(
        "--point-stride",
        type=int,
        default=8,
        help="RGB-D geometry sampling stride; 8 keeps boxes stable while reducing CPU post-processing",
    )
    p.add_argument("--min-valid-points", type=int, default=12)
    p.add_argument("--max-depth-m", type=float, default=8.)
    p.add_argument("--telemetry-max-delta-sec", type=float, default=.15,
                   help="Replay/startup pose tolerance; live sensor calibration is authoritative")
    p.add_argument("--capture-context-topic", default="/physical_nav/capture_context",
                   help="Per-capture pose/calibration authority; empty string enables legacy replay matching")
    p.add_argument("--camera-x", type=float, default=0.)
    p.add_argument("--camera-y", type=float, default=0.)
    p.add_argument("--camera-z", type=float, default=0.)
    p.add_argument("--camera-roll", type=float, default=0.)
    p.add_argument("--camera-pitch", type=float, default=0.)
    p.add_argument("--camera-yaw", type=float, default=0.)
    p.add_argument("--camera-imu-enabled", action="store_true", default=True)
    p.add_argument(
        "--disable-camera-imu", action="store_false", dest="camera_imu_enabled"
    )
    p.add_argument("--camera-imu-use-yaw", action="store_true", default=False)
    p.add_argument("--camera-imu-max-roll-rad", type=float, default=0.7)
    p.add_argument("--camera-imu-max-pitch-rad", type=float, default=0.7)
    p.add_argument("--camera-imu-max-yaw-rad", type=float, default=0.35)
    args = p.parse_args()
    YoloeWorker(args).run()


if __name__ == "__main__": main()
