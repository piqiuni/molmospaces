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
import time
import urllib.request
from typing import Any

import cv2
import numpy as np


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


def _world_points(points: np.ndarray, telemetry: dict[str, Any], translation: np.ndarray, rpy: tuple[float, float, float]) -> np.ndarray:
    base = points @ _rotation(*rpy).T + translation
    position = np.asarray(telemetry.get("position", [0, 0, 0])[:3], dtype=np.float32)
    yaw = float(telemetry.get("yaw", telemetry.get("imu", {}).get("rpy", [0, 0, 0])[2] if telemetry.get("imu") else 0.0))
    c, s = math.cos(yaw), math.sin(yaw); rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    base[:, :2] = base[:, :2] @ rot.T + position[:2]
    base[:, 2] += position[2]
    return base


def _label(raw: str) -> str:
    value = str(raw or "").strip().lower().replace(" ", "_")
    aliases = {"refrigerator": "fridge", "couch": "sofa", "dining_table": "table", "dinning_table": "table", "doorframe": "doorframe"}
    if value in aliases: return aliases[value]
    for keyword in ("refrigerator", "fridge", "door", "cabinet", "drawer", "chair", "table", "sofa", "couch", "bed", "bottle", "cup", "bowl", "sink", "toilet", "shelf", "box", "microwave"):
        if keyword in value: return aliases.get(keyword, keyword)
    return value


class YoloeWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("YOLOE worker requires ultralytics in the algorithm Python environment") from exc
        self.args = args; self.model = YOLOE(args.model_path); self.last_seq = -1; self.rotation = _rotation(args.camera_roll, args.camera_pitch, args.camera_yaw); self.translation = np.asarray([args.camera_x, args.camera_y, args.camera_z], dtype=np.float32)
        print(f"YOLOE loaded: {args.model_path} device={args.device}", flush=True)

    def infer(self, raw: dict[str, Any]) -> dict[str, Any]:
        rgb = _decode(raw["rgb"]); depth = _decode(raw["depth"]).astype(np.float32); intr = raw.get("intrinsics", {}); fx, fy, cx, cy = [float(intr.get(k, 0)) for k in ("fx", "fy", "cx", "cy")]
        result = self.model.predict(source=rgb, device=self.args.device, imgsz=self.args.imgsz, conf=self.args.conf, iou=self.args.iou, max_det=self.args.max_det, verbose=False, save=False)[0]
        boxes = getattr(result, "boxes", None); masks = getattr(result, "masks", None); detections = []
        if boxes is None: return {"seq": raw["seq"], "stamp": raw["stamp"], "model": self.args.model_path, "detections": []}
        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy); confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf); classes = boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls, dtype=int); names = getattr(result, "names", {})
        mask_data = None
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_data = masks.data.detach().cpu().numpy() if hasattr(masks.data, "detach") else np.asarray(masks.data)
        for index, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in box]; raw_name = names.get(int(classes[index]), str(int(classes[index]))) if isinstance(names, dict) else str(int(classes[index])); mask = mask_data[index] if mask_data is not None and index < len(mask_data) else None
            if mask is not None and mask.shape != depth.shape:
                import cv2
                mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else: mask = mask > .5 if mask is not None else np.zeros(depth.shape, dtype=bool)
            rows, cols = np.where(mask); stride = max(1, int(self.args.point_stride)); rows, cols = rows[::stride], cols[::stride]; values = depth[rows, cols] * float(raw.get("depth_scale", .001)); valid = np.isfinite(values) & (values > 0) & (values <= self.args.max_depth_m); rows, cols, values = rows[valid], cols[valid], values[valid]
            if values.size < self.args.min_valid_points or fx <= 0 or fy <= 0: continue
            points = np.stack([(cols-cx)*values/fx, (rows-cy)*values/fy, values], axis=1).astype(np.float32)
            camera_center = np.percentile(points, 50, axis=0)
            world = _world_points(points, raw.get("telemetry", {}), self.translation, (self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw)); mins, maxs = np.percentile(world, 10, axis=0), np.percentile(world, 90, axis=0); center = (mins + maxs) / 2; size = np.maximum(maxs - mins, .01)
            sparse_rows, sparse_cols = np.where(mask)
            if sparse_rows.size > 3000: sparse_rows, sparse_cols = sparse_rows[::max(1, sparse_rows.size // 3000)], sparse_cols[::max(1, sparse_cols.size // 3000)]
            detections.append({"semantic_class": _label(raw_name), "raw_class": str(raw_name), "confidence": float(confs[index]), "bbox": [x1, y1, x2, y2], "mask": {"rows": sparse_rows.astype(int).tolist(), "cols": sparse_cols.astype(int).tolist()}, "mask_area": int(np.count_nonzero(mask)), "depth_median_m": float(np.median(values)), "depth_valid_points": int(values.size), "camera_position": {"x": float(camera_center[0]), "y": float(camera_center[1]), "z": float(camera_center[2])}, "position": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, "world_position": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, "aabb_center": center.astype(float).tolist(), "aabb_size": size.astype(float).tolist(), "box3d_center": center.astype(float).tolist(), "box3d_size": size.astype(float).tolist(), "source_frame": str(raw.get("camera_frame", "d435i_color_optical_frame")), "source_model": self.args.model_path, "projection_method": "physical_yoloe_rgbd_mask", "capture_seq": int(raw["seq"]), "stamp": float(raw["stamp"])})
        return {"seq": raw["seq"], "stamp": raw["stamp"], "model": self.args.model_path, "detections": detections}

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
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--web-url", default="http://127.0.0.1:8765"); p.add_argument("--model-path", default="/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt"); p.add_argument("--device", default="cuda:0"); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--conf", type=float, default=.35); p.add_argument("--iou", type=float, default=.7); p.add_argument("--max-det", type=int, default=50); p.add_argument("--rate", type=float, default=10.); p.add_argument("--retry-s", type=float, default=1.); p.add_argument("--point-stride", type=int, default=4); p.add_argument("--min-valid-points", type=int, default=12); p.add_argument("--max-depth-m", type=float, default=8.); p.add_argument("--camera-x", type=float, default=0.); p.add_argument("--camera-y", type=float, default=0.); p.add_argument("--camera-z", type=float, default=0.); p.add_argument("--camera-roll", type=float, default=0.); p.add_argument("--camera-pitch", type=float, default=0.); p.add_argument("--camera-yaw", type=float, default=0.); args = p.parse_args(); YoloeWorker(args).run()


if __name__ == "__main__": main()
