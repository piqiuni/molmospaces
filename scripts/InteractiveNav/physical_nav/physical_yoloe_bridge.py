#!/usr/bin/env python3
"""YOLOE-26l PF Seg worker for the physical platform.

This worker intentionally has no ROS or Unitree dependency. It polls the
encoded D435i frame from the local WebSocket/web gateway, runs YOLOE in the
algorithm Python environment, lifts masks to camera/base/map-frame geometry,
and posts JSON detections back to the local ROS gateway.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import sys
import time
import urllib.request
from typing import Any

import cv2
import numpy as np


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


def _decode(value: str) -> np.ndarray:
    encoded = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError("cannot decode gateway frame")
    return image


def _rgb_mask_to_depth(mask: np.ndarray, depth: np.ndarray, rgb_intr: dict[str, Any], depth_intr: dict[str, Any], extr: dict[str, Any]) -> np.ndarray:
    """Project an RGB YOLO mask onto the native/aligned depth pixel grid."""
    if mask.shape == depth.shape:
        return mask.astype(bool)
    h, w = depth.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    z = depth.astype(np.float32)
    valid = z > 0
    x = (xx - float(depth_intr.get("cx", 0))) * z / max(float(depth_intr.get("fx", 1)), 1e-6)
    y = (yy - float(depth_intr.get("cy", 0))) * z / max(float(depth_intr.get("fy", 1)), 1e-6)
    points = np.stack((x, y, z), axis=-1)
    rotation = np.asarray(extr.get("rotation", np.eye(3).reshape(-1)), dtype=np.float32).reshape(3, 3)
    translation = np.asarray(extr.get("translation", [0, 0, 0]), dtype=np.float32).reshape(1, 1, 3)
    color_points = points @ rotation.T + translation
    cz = color_points[..., 2]
    u = np.rint(color_points[..., 0] * float(rgb_intr.get("fx", 1)) / np.maximum(cz, 1e-6) + float(rgb_intr.get("cx", 0))).astype(np.int32)
    v = np.rint(color_points[..., 1] * float(rgb_intr.get("fy", 1)) / np.maximum(cz, 1e-6) + float(rgb_intr.get("cy", 0))).astype(np.int32)
    inside = valid & (cz > 0) & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    result = np.zeros((h, w), dtype=bool)
    result[inside] = mask[v[inside], u[inside]] > 0
    return result


def _post(url: str, value: Any) -> None:
    data = json.dumps({"name": "detections", "value": value}, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/ros-state", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=.6):
        pass


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


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.asarray([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]], dtype=np.float32)


def _telemetry_quaternion(telemetry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return the live body/camera quaternion as (x, y, z, w).

    Unitree publishes ``imu.quaternion`` as (w, x, y, z).  A future D435i
    tracking source may publish an explicit ``camera_pose.quaternion`` in
    either common mapping/list form; prefer that when present.
    """
    candidates = [telemetry.get("camera_pose"), telemetry.get("d435i_pose"), telemetry.get("pose"), telemetry.get("camera_imu")]
    candidates.append(telemetry.get("imu"))
    for source in candidates:
        if not isinstance(source, dict):
            continue
        raw = source.get("quaternion") or source.get("orientation")
        if isinstance(raw, dict):
            try:
                values = [float(raw[k]) for k in ("x", "y", "z", "w")]
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
            try:
                values = [float(v) for v in raw[:4]]
            except (TypeError, ValueError):
                continue
            if source is telemetry.get("imu"):
                values = [values[1], values[2], values[3], values[0]]
        else:
            continue
        norm = float(np.linalg.norm(values))
        if norm > 1e-6 and np.isfinite(norm):
            return tuple(float(v / norm) for v in values)
    return None


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


def _world_points(
    points: np.ndarray,
    telemetry: dict[str, Any],
    translation: np.ndarray,
    rpy: tuple[float, float, float],
    *,
    optical_frame: bool = False,
) -> np.ndarray:
    """Lift camera points into the odom/map frame.

    D435i points use REP-103 optical axes (x right, y down, z forward), while
    Go2 base/odom uses x forward, y left, z up.  ``optical_frame`` is opt-in to
    preserve the historical helper contract used by simulator-side tests.
    """
    camera = points @ _OPTICAL_TO_BASE.T if optical_frame else points
    base = camera @ _rotation(*rpy).T + translation
    position = np.asarray(telemetry.get("position", [0, 0, 0])[:3], dtype=np.float32)
    quaternion = _telemetry_quaternion(telemetry)
    if quaternion is not None:
        # Dynamic roll/pitch/yaw from the live IMU correct the camera rod tilt;
        # the fixed extrinsic is only the rigid base-to-camera offset.
        base = base @ _quaternion_matrix(quaternion).T
    else:
        yaw = float(telemetry.get("yaw", telemetry.get("imu", {}).get("rpy", [0, 0, 0])[2] if telemetry.get("imu") else 0.0))
        c, s = math.cos(yaw), math.sin(yaw)
        base[:, :2] = base[:, :2] @ np.asarray([[c, -s], [s, c]], dtype=np.float32).T
    base += position
    return base


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
    lookup = {tuple(key.tolist()): index for index, key in enumerate(voxel_keys)}
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
    binary = np.asarray(mask, dtype=np.uint8) > 0
    min_component = max(1, int(config.get("mask_component_min_area", 48)))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
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
    # Filter every sufficiently large 2-D component independently. A global
    # depth band can erase a valid disconnected surface that lies farther
    # away than the dominant component of the same instance mask.
    for component in candidates:
        rows, cols = np.where(labels == component)
        depths = depth[rows, cols] * depth_scale
        valid = np.isfinite(depths) & (depths > 0.0) & (depths <= max_depth)
        rows, cols, depths = rows[valid], cols[valid], depths[valid]
        if depths.size < min_valid:
            continue
        if depths.size >= max(min_valid * 2, 32):
            low, high = float(np.quantile(depths, low_q)), float(np.quantile(depths, high_q))
            in_band = (depths >= low) & (depths <= high)
            if int(np.count_nonzero(in_band)) >= min_valid:
                rows, cols, depths = rows[in_band], cols[in_band], depths[in_band]
        # A door/portal is predominantly a single depth plane.  RGB-D
        # segmentation often leaks through its edges or holes onto the wall
        # behind it; those points create an artificially wide world box after
        # projection.  Use a robust MAD band only for configured planar
        # classes, retaining the full 2-D extent of the actual plane.
        planar_labels = {
            str(item).strip().casefold()
            for item in config.get("planar_semantic_labels", ("door", "portal"))
            if str(item).strip()
        }
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
    points = np.stack([(cols - cx) * depths / fx, (rows - cy) * depths / fy, depths], axis=1).astype(np.float32)
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
    return np.quantile(points, low, axis=0).astype(np.float32), np.quantile(points, high, axis=0).astype(np.float32)


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


def _oriented_bounds(points: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit an upright instance OBB; only horizontal yaw comes from points."""
    center = np.mean(points, axis=0).astype(np.float32)
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
    projected = centered @ axes
    low = min(max(float(config.get("bbox_quantile_lower", 0.02)), 0.0), 1.0)
    high = min(max(float(config.get("bbox_quantile_upper", 0.98)), low), 1.0)
    mins = np.quantile(projected, low, axis=0)
    maxs = np.quantile(projected, high, axis=0)
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


class YoloeWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("YOLOE worker requires ultralytics in the algorithm Python environment") from exc
        self.args = args; self.model = YOLOE(args.model_path); self.last_seq = -1; self.last_stamp = float("-inf"); self.rotation = _rotation(args.camera_roll, args.camera_pitch, args.camera_yaw); self.translation = np.asarray([args.camera_x, args.camera_y, args.camera_z], dtype=np.float32)
        self.detector_config = _load_object_detection_config(args.detector_config)
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
        filter_config = load_detection_filter_config(args.detector_config)
        self.detection_filter = DetectionFilter(filter_config)
        self.class_confidence_thresholds = {
            str(label).strip().casefold().replace(" ", "_"): max(0.001, min(1.0, float(value)))
            for label, value in class_thresholds.items()
        }
        print(
            f"YOLOE loaded: {args.model_path} device={args.device} "
            f"conf={self.args.conf:.3f} iou={self.args.iou:.3f} max_det={self.args.max_det} "
            f"detection_filter={self.detection_filter.enabled} config={args.detector_config}",
            flush=True,
        )

    def _claim_frame(self, raw: dict[str, Any]) -> bool:
        """Accept each capture once, including after the dog bridge restarts.

        The bridge sequence is process-local and returns to zero on restart.
        Capture timestamps remain monotonic across that restart, so they are
        the primary receipt identity; sequence is retained as a fallback for
        sources that do not provide a usable timestamp.
        """
        if not raw.get("rgb") or not raw.get("depth"):
            return False
        seq = int(raw.get("seq", -1))
        try:
            stamp = float(raw.get("stamp", 0.0))
        except (TypeError, ValueError):
            stamp = 0.0
        has_stamp = math.isfinite(stamp) and stamp > 0.0
        if has_stamp:
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

    def infer(self, raw: dict[str, Any]) -> dict[str, Any]:
        infer_started = time.perf_counter()
        rgb = _decode(raw["rgb"]); depth = _decode(raw["depth"]).astype(np.float32)
        rgb_intr = raw.get("rgb_intrinsics") or raw.get("intrinsics", {})
        depth_intr = raw.get("depth_intrinsics") or raw.get("intrinsics", {})
        fx, fy, cx, cy = [float(depth_intr.get(k, 0)) for k in ("fx", "fy", "cx", "cy")]
        result = self.model.predict(source=rgb, device=self.args.device, imgsz=self.args.imgsz, conf=self.args.conf, iou=self.args.iou, max_det=self.args.max_det, verbose=False, save=False)[0]
        boxes = getattr(result, "boxes", None); masks = getattr(result, "masks", None); detections = []; debug_point_sets = []
        if boxes is None:
            return {
                "seq": raw["seq"],
                "stamp": raw["stamp"],
                "camera_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")),
                "model": self.args.model_path,
                "inference_ms": (time.perf_counter() - infer_started) * 1000.0,
                "overlay_jpeg": _encode_detection_overlay(rgb, []),
                "detections": [],
            }
        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy); confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf); classes = boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls, dtype=int); names = getattr(result, "names", {})
        mask_data = None
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_data = masks.data.detach().cpu().numpy() if hasattr(masks.data, "detach") else np.asarray(masks.data)
        for index, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in box]; raw_name = names.get(int(classes[index]), str(int(classes[index]))) if isinstance(names, dict) else str(int(classes[index])); mask = mask_data[index] if mask_data is not None and index < len(mask_data) else None
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
            if mask is not None and mask.shape != depth.shape:
                mask = _rgb_mask_to_depth(mask.astype(np.uint8), depth, rgb_intr, depth_intr, raw.get("depth_to_color_extrinsics") or {})
            else: mask = mask > .5 if mask is not None else np.zeros(depth.shape, dtype=bool)
            if fx <= 0 or fy <= 0:
                continue
            geometry_config = dict(self.detector_config)
            geometry_config["semantic_label"] = filtered_label["semantic_class"]
            geometry_config.update({"point_stride": self.args.point_stride, "max_depth_m": self.args.max_depth_m, "min_valid_points": self.args.min_valid_points, "depth_scale": float(raw.get("depth_scale", .001))})
            prepared = _prepare_instance_points(mask, depth.astype(np.float32), (fx, fy, cx, cy), (x1, y1, x2, y2), geometry_config, filtered_label["semantic_class"])
            if prepared is None:
                continue
            mask, points, values = prepared
            camera_mins, camera_maxs = _robust_bounds(points, geometry_config)
            camera_center = (camera_mins + camera_maxs) / 2
            camera_size = np.maximum(camera_maxs - camera_mins, .01)
            world = _world_points(points, raw.get("telemetry", {}), self.translation, (self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw), optical_frame=True); mins, maxs = _robust_bounds(world, geometry_config); center = (mins + maxs) / 2; size = np.maximum(maxs - mins, .01)
            obb_center, obb_size, obb_orientation = _oriented_bounds(world, geometry_config)
            if bool(geometry_config.get("reject_low_mean_height", True)) and float(np.mean(world[:, 2])) <= float(geometry_config.get("min_mean_height_m", 0.08)):
                continue
            if _is_ground_like_box(center, size, geometry_config):
                continue
            if _is_implausibly_large_object(filtered_label["semantic_class"], size, geometry_config):
                continue
            if not _passes_class_plausibility(
                filtered_label["semantic_class"],
                (x1, y1, x2, y2),
                size,
                geometry_config,
            ):
                continue
            # Preserve every supported component as a compact ordered
            # boundary.  The exact same filtered mask already drives the 3-D
            # points and box, so the web overlay must not silently collapse it
            # back to the single largest component.
            contours, _hierarchy = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            mask_polygons: list[list[list[int]]] = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                epsilon = max(1.0, 0.0025 * cv2.arcLength(contour, True))
                approximated = cv2.approxPolyDP(contour, epsilon, True)
                if approximated.shape[0] >= 3:
                    mask_polygons.append(approximated[:, 0, :].astype(int).tolist())
            # Keep the legacy field for older consumers while the plural
            # field is authoritative for physical-platform visualization.
            mask_polygon = mask_polygons[0] if mask_polygons else []
            sparse_rows, sparse_cols = np.where(mask)
            if sparse_rows.size > 3000: sparse_rows, sparse_cols = sparse_rows[::max(1, sparse_rows.size // 3000)], sparse_cols[::max(1, sparse_cols.size // 3000)]
            debug_point_sets.append((points, world))
            detections.append({"semantic_class": filtered_label["semantic_class"], "semantic_class_raw": filtered_label["semantic_class_raw"], "raw_class": str(raw_name), "confidence": float(confs[index]), "bbox": [x1, y1, x2, y2], "mask": {"rows": sparse_rows.astype(int).tolist(), "cols": sparse_cols.astype(int).tolist()}, "mask_polygon": mask_polygon, "mask_polygons": mask_polygons, "mask_area": int(np.count_nonzero(mask)), "depth_median_m": float(np.median(values)), "depth_valid_points": int(values.size), "camera_position": {"x": float(camera_center[0]), "y": float(camera_center[1]), "z": float(camera_center[2])}, "camera_box3d_center": camera_center.astype(float).tolist(), "camera_box3d_size": camera_size.astype(float).tolist(), "position": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, "world_position": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, "aabb_center": center.astype(float).tolist(), "aabb_size": size.astype(float).tolist(), "box3d_center": obb_center.astype(float).tolist(), "box3d_size": obb_size.astype(float).tolist(), "world_box3d_center": obb_center.astype(float).tolist(), "world_box3d_size": obb_size.astype(float).tolist(), "world_box3d_marker_size": obb_size.astype(float).tolist(), "world_box3d_orientation": obb_orientation, "source_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")), "source_model": self.args.model_path, "projection_method": "physical_yoloe_rgbd_mask_obb", "map_transform_status": "telemetry_fallback", "capture_seq": int(raw["seq"]), "stamp": float(raw["stamp"])})
        debug_max_points = max(32, int(self.detector_config.get("debug_cloud_max_points_per_instance", 400)))
        for detection, (point_set, world_point_set) in zip(detections, debug_point_sets):
            stride = max(1, int(math.ceil(float(point_set.shape[0]) / debug_max_points)))
            # These points come from the exact RGB-D receipt used by YOLOE;
            # RViz therefore never pairs a segmentation mask with newer depth.
            camera_debug = point_set[::stride][:debug_max_points]
            world_debug = world_point_set[::stride][:debug_max_points]
            detection["segment_point_count"] = int(camera_debug.shape[0])
            detection["camera_segment_points_f32"] = _encode_point_rows(camera_debug)
            detection["world_segment_points_f32"] = _encode_point_rows(world_debug)
        return {
            "seq": raw["seq"],
            "stamp": raw["stamp"],
            "camera_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")),
            "model": self.args.model_path,
            "inference_ms": (time.perf_counter() - infer_started) * 1000.0,
            "overlay_jpeg": _encode_detection_overlay(
                rgb,
                detections,
                int(self.detector_config.get("debug_overlay_jpeg_quality", 92)),
            ),
            "detections": detections,
        }

    def run(self) -> None:
        while True:
            cycle_started = time.monotonic()
            try:
                with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/raw-frame", timeout=.8) as response: raw = json.loads(response.read().decode())
                if not self._claim_frame(raw): time.sleep(.02); continue
                report = self.infer(raw)
                _post(self.args.web_url, report)
                # ``rate`` is a cycle target, not an additional post-inference
                # delay. The former fixed sleep halved a 100 ms pipeline from
                # about 9 Hz to about 4.5 Hz.
                elapsed = time.monotonic() - cycle_started
                time.sleep(max(0., 1. / self.args.rate - elapsed))
            except Exception as exc:
                print(f"YOLOE worker warning: {exc}", flush=True); time.sleep(self.args.retry_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--web-url", default="http://127.0.0.1:8765"); p.add_argument("--model-path", default="/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt"); p.add_argument("--detector-config", type=Path, default=DEFAULT_DETECTOR_CONFIG); p.add_argument("--device", default="cuda:0"); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--conf", type=float, default=.35); p.add_argument("--iou", type=float, default=.7); p.add_argument("--max-det", type=int, default=50); p.add_argument("--rate", type=float, default=10.); p.add_argument("--retry-s", type=float, default=1.); p.add_argument("--point-stride", type=int, default=4); p.add_argument("--min-valid-points", type=int, default=12); p.add_argument("--max-depth-m", type=float, default=8.); p.add_argument("--camera-x", type=float, default=0.); p.add_argument("--camera-y", type=float, default=0.); p.add_argument("--camera-z", type=float, default=0.); p.add_argument("--camera-roll", type=float, default=0.); p.add_argument("--camera-pitch", type=float, default=0.); p.add_argument("--camera-yaw", type=float, default=0.); args = p.parse_args(); YoloeWorker(args).run()


if __name__ == "__main__": main()
