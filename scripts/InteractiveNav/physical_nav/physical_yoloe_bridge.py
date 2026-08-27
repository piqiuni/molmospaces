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


def _post(url: str, value: Any) -> None:
    data = json.dumps({"name": "detections", "value": value}, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/ros-state", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=.6):
        pass


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


def _largest_euclidean_cluster(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Keep the dominant compact depth surface without an sklearn dependency."""
    if points.shape[0] <= min_points:
        return points
    step = max(1, int(math.ceil(points.shape[0] / 800.0)))
    sampled = points[::step]
    visited = np.zeros(sampled.shape[0], dtype=bool)
    best: list[int] = []
    for start in range(sampled.shape[0]):
        if visited[start]:
            continue
        queue, cluster = [start], []
        visited[start] = True
        while queue:
            index = queue.pop()
            cluster.append(index)
            delta = sampled - sampled[index]
            neighbors = np.where(np.sum(delta * delta, axis=1) <= eps * eps)[0]
            for neighbor in neighbors.tolist():
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(cluster) > len(best):
            best = cluster
    if len(best) < min_points:
        return points
    cluster_points = sampled[np.asarray(best, dtype=np.int32)]
    center = np.mean(cluster_points, axis=0)
    keep = np.linalg.norm(points - center[None, :], axis=1) <= max(eps * 1.5, 1e-3)
    result = points[keep]
    return result if result.shape[0] >= min_points else cluster_points


def _prepare_instance_points(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Clean one PF-Seg mask and return its camera points and depths."""
    binary = np.asarray(mask, dtype=np.uint8) > 0
    min_component = max(1, int(config.get("mask_component_min_area", 48)))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return None
    x1, y1, x2, y2 = bbox
    center_x = max(0, min(binary.shape[1] - 1, int(round((x1 + x2) * 0.5))))
    center_y = max(0, min(binary.shape[0] - 1, int(round((y1 + y2) * 0.5))))
    center_component = int(labels[center_y, center_x])
    candidates = [index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_component]
    if not candidates:
        return None
    selected = center_component if center_component in candidates else max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    cleaned = labels == selected
    rows, cols = np.where(cleaned)
    stride = max(1, int(config.get("point_stride", 4)))
    rows, cols = rows[::stride], cols[::stride]
    depths = depth[rows, cols] * float(config.get("depth_scale", 1.0))
    valid = np.isfinite(depths) & (depths > 0.0) & (depths <= float(config.get("max_depth_m", 8.0)))
    rows, cols, depths = rows[valid], cols[valid], depths[valid]
    min_valid = max(4, int(config.get("min_valid_points", 12)))
    if depths.size < min_valid:
        return None
    bottom_ratio = min(max(float(config.get("drop_bottom_ratio", 0.12)), 0.0), 0.9)
    if bottom_ratio > 0.0:
        threshold = float(np.quantile(rows, 1.0 - bottom_ratio))
        keep = rows < threshold
        if int(np.count_nonzero(keep)) >= min_valid:
            rows, cols, depths = rows[keep], cols[keep], depths[keep]
    low_q = min(max(float(config.get("depth_band_lower_quantile", 0.05)), 0.0), 1.0)
    high_q = min(max(float(config.get("depth_band_upper_quantile", 0.70)), low_q), 1.0)
    low, high = float(np.quantile(depths, low_q)), float(np.quantile(depths, high_q))
    keep = (depths >= low) & (depths <= high)
    if int(np.count_nonzero(keep)) >= min_valid:
        rows, cols, depths = rows[keep], cols[keep], depths[keep]
    fx, fy, cx, cy = intrinsics
    points = np.stack([(cols - cx) * depths / fx, (rows - cy) * depths / fy, depths], axis=1).astype(np.float32)
    if bool(config.get("enable_euclidean_cluster", True)):
        points = _largest_euclidean_cluster(
            points,
            float(config.get("cluster_eps", 0.18)),
            max(4, int(config.get("cluster_min_points", 12))),
        )
    if points.shape[0] < min_valid:
        return None
    return cleaned, points, depths


def _robust_bounds(points: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    low = min(max(float(config.get("bbox_quantile_lower", 0.10)), 0.0), 1.0)
    high = min(max(float(config.get("bbox_quantile_upper", 0.90)), low), 1.0)
    return np.quantile(points, low, axis=0).astype(np.float32), np.quantile(points, high, axis=0).astype(np.float32)


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


class YoloeWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("YOLOE worker requires ultralytics in the algorithm Python environment") from exc
        self.args = args; self.model = YOLOE(args.model_path); self.last_seq = -1; self.rotation = _rotation(args.camera_roll, args.camera_pitch, args.camera_yaw); self.translation = np.asarray([args.camera_x, args.camera_y, args.camera_z], dtype=np.float32)
        self.detector_config = _load_object_detection_config(args.detector_config)
        filter_config = load_detection_filter_config(args.detector_config)
        self.detection_filter = DetectionFilter(filter_config)
        print(
            f"YOLOE loaded: {args.model_path} device={args.device} "
            f"detection_filter={self.detection_filter.enabled} config={args.detector_config}",
            flush=True,
        )

    def infer(self, raw: dict[str, Any]) -> dict[str, Any]:
        infer_started = time.perf_counter()
        rgb = _decode(raw["rgb"]); depth = _decode(raw["depth"]).astype(np.float32); intr = raw.get("intrinsics", {}); fx, fy, cx, cy = [float(intr.get(k, 0)) for k in ("fx", "fy", "cx", "cy")]
        result = self.model.predict(source=rgb, device=self.args.device, imgsz=self.args.imgsz, conf=self.args.conf, iou=self.args.iou, max_det=self.args.max_det, verbose=False, save=False)[0]
        boxes = getattr(result, "boxes", None); masks = getattr(result, "masks", None); detections = []
        if boxes is None: return {"seq": raw["seq"], "stamp": raw["stamp"], "model": self.args.model_path, "inference_ms": (time.perf_counter() - infer_started) * 1000.0, "detections": []}
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
            if mask is not None and mask.shape != depth.shape:
                import cv2
                mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else: mask = mask > .5 if mask is not None else np.zeros(depth.shape, dtype=bool)
            if fx <= 0 or fy <= 0:
                continue
            geometry_config = dict(self.detector_config)
            geometry_config.update({"point_stride": self.args.point_stride, "max_depth_m": self.args.max_depth_m, "min_valid_points": self.args.min_valid_points, "depth_scale": float(raw.get("depth_scale", .001))})
            prepared = _prepare_instance_points(mask, depth.astype(np.float32), (fx, fy, cx, cy), (x1, y1, x2, y2), geometry_config)
            if prepared is None:
                continue
            mask, points, values = prepared
            camera_mins, camera_maxs = _robust_bounds(points, geometry_config)
            camera_center = (camera_mins + camera_maxs) / 2
            camera_size = np.maximum(camera_maxs - camera_mins, .01)
            world = _world_points(points, raw.get("telemetry", {}), self.translation, (self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw), optical_frame=True); mins, maxs = _robust_bounds(world, geometry_config); center = (mins + maxs) / 2; size = np.maximum(maxs - mins, .01)
            if bool(geometry_config.get("reject_low_mean_height", True)) and float(np.mean(world[:, 2])) <= float(geometry_config.get("min_mean_height_m", 0.08)):
                continue
            if _is_ground_like_box(center, size, geometry_config):
                continue
            if _is_implausibly_large_object(filtered_label["semantic_class"], size, geometry_config):
                continue
            sparse_rows, sparse_cols = np.where(mask)
            if sparse_rows.size > 3000: sparse_rows, sparse_cols = sparse_rows[::max(1, sparse_rows.size // 3000)], sparse_cols[::max(1, sparse_cols.size // 3000)]
            detections.append({"semantic_class": filtered_label["semantic_class"], "semantic_class_raw": filtered_label["semantic_class_raw"], "raw_class": str(raw_name), "confidence": float(confs[index]), "bbox": [x1, y1, x2, y2], "mask": {"rows": sparse_rows.astype(int).tolist(), "cols": sparse_cols.astype(int).tolist()}, "mask_area": int(np.count_nonzero(mask)), "depth_median_m": float(np.median(values)), "depth_valid_points": int(values.size), "camera_position": {"x": float(camera_center[0]), "y": float(camera_center[1]), "z": float(camera_center[2])}, "camera_box3d_center": camera_center.astype(float).tolist(), "camera_box3d_size": camera_size.astype(float).tolist(), "position": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, "world_position": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, "aabb_center": center.astype(float).tolist(), "aabb_size": size.astype(float).tolist(), "box3d_center": center.astype(float).tolist(), "box3d_size": size.astype(float).tolist(), "source_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")), "source_model": self.args.model_path, "projection_method": "physical_yoloe_rgbd_mask", "map_transform_status": "telemetry_fallback", "capture_seq": int(raw["seq"]), "stamp": float(raw["stamp"])})
        return {"seq": raw["seq"], "stamp": raw["stamp"], "model": self.args.model_path, "inference_ms": (time.perf_counter() - infer_started) * 1000.0, "detections": detections}

    def run(self) -> None:
        while True:
            try:
                with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/raw-frame", timeout=.8) as response: raw = json.loads(response.read().decode())
                seq = int(raw.get("seq", -1))
                if seq <= self.last_seq or not raw.get("rgb") or not raw.get("depth"): time.sleep(.02); continue
                self.last_seq = seq; report = self.infer(raw); _post(self.args.web_url, report); time.sleep(max(0., 1. / self.args.rate))
            except Exception as exc:
                print(f"YOLOE worker warning: {exc}", flush=True); time.sleep(self.args.retry_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--web-url", default="http://127.0.0.1:8765"); p.add_argument("--model-path", default="/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt"); p.add_argument("--detector-config", type=Path, default=DEFAULT_DETECTOR_CONFIG); p.add_argument("--device", default="cuda:0"); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--conf", type=float, default=.35); p.add_argument("--iou", type=float, default=.7); p.add_argument("--max-det", type=int, default=50); p.add_argument("--rate", type=float, default=10.); p.add_argument("--retry-s", type=float, default=1.); p.add_argument("--point-stride", type=int, default=4); p.add_argument("--min-valid-points", type=int, default=12); p.add_argument("--max-depth-m", type=float, default=8.); p.add_argument("--camera-x", type=float, default=0.); p.add_argument("--camera-y", type=float, default=0.); p.add_argument("--camera-z", type=float, default=0.); p.add_argument("--camera-roll", type=float, default=0.); p.add_argument("--camera-pitch", type=float, default=0.); p.add_argument("--camera-yaw", type=float, default=0.); args = p.parse_args(); YoloeWorker(args).run()


if __name__ == "__main__": main()
