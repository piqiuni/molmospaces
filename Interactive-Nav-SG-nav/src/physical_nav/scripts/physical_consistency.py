#!/usr/bin/env python3
"""Pure-Python geometry checks used by the installed ROS node."""
from __future__ import annotations
import math
from typing import Any, Iterable, Mapping

def _num(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v); return x if math.isfinite(x) else default
    except (TypeError, ValueError): return default

def _point3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        return tuple(_num(value.get(axis)) for axis in ("x", "y", "z"))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(_num(item) for item in value[:3])
    return None

def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a); bx1, by1, bx2, by2 = map(float, b)
    inter = max(0., min(ax2, bx2)-max(ax1, bx1)) * max(0., min(ay2, by2)-max(ay1, by1))
    aa = max(0., ax2-ax1)*max(0., ay2-ay1); ab = max(0., bx2-bx1)*max(0., by2-by1)
    return inter / max(1e-9, aa + ab - inter)

def evaluate_detection(det: Mapping[str, Any], *, projected_bbox: Iterable[float] | None = None,
                       observed_depth_m: Iterable[float] | None = None, map_node: Mapping[str, Any] | None = None,
                       previous: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    t = {"bbox_iou": .45, "bbox_center_px": 35., "depth_abs_m": .25, "map_distance_m": .5, "temporal_jump_m": .6}; t.update(thresholds or {})
    metrics: dict[str, float] = {}; reasons: list[str] = []; bbox = det.get("bbox") or det.get("bbox_2d")
    if projected_bbox is not None and bbox is not None and len(bbox) == 4:
        metrics["bbox_iou"] = _iou(bbox, projected_bbox); ac = ((float(bbox[0])+float(bbox[2]))/2, (float(bbox[1])+float(bbox[3]))/2); bc = ((float(projected_bbox[0])+float(projected_bbox[2]))/2, (float(projected_bbox[1])+float(projected_bbox[3]))/2); metrics["bbox_center_px"] = math.hypot(ac[0]-bc[0], ac[1]-bc[1])
        if metrics["bbox_iou"] < float(t["bbox_iou"]): reasons.append("reprojection_bbox_iou_low")
        if metrics["bbox_center_px"] > float(t["bbox_center_px"]): reasons.append("reprojection_center_offset")
    if observed_depth_m is not None:
        depths = sorted(_num(v) for v in observed_depth_m if _num(v) > 0); pos = det.get("position") or det.get("world_position") or {}
        if depths and isinstance(pos, Mapping) and pos.get("z") is not None:
            metrics["depth_median_abs_m"] = abs(_num(pos.get("z")) - depths[len(depths)//2])
            if metrics["depth_median_abs_m"] > float(t["depth_abs_m"]): reasons.append("depth_projection_mismatch")
        if depths: metrics["depth_valid_ratio"] = min(1., len(depths)/max(1., _num(det.get("mask_area"), len(depths))))
    sensor_depth = _num(det.get("depth_median_m"), -1.)
    camera_point = _point3(det.get("camera_position"))
    if sensor_depth > 0.:
        metrics["sensor_depth_m"] = sensor_depth
        metrics["depth_valid_points"] = _num(det.get("depth_valid_points"))
        if camera_point is not None:
            metrics["rgbd_depth_lift_abs_m"] = abs(camera_point[2] - sensor_depth)
            if metrics["rgbd_depth_lift_abs_m"] > float(t["depth_abs_m"]): reasons.append("rgbd_depth_lift_mismatch")
    if map_node:
        p = det.get("world_position") or det.get("position") or {}; q = map_node.get("world_position") or map_node.get("position") or map_node.get("centroid") or map_node.get("aabb_center") or {}
        p3, q3 = _point3(p), _point3(q)
        if p3 is not None and q3 is not None:
            metrics["map_distance_m"] = math.sqrt(sum((a - b) ** 2 for a, b in zip(p3, q3)))
            metrics["map_xy_distance_m"] = math.hypot(p3[0] - q3[0], p3[1] - q3[1])
            metrics["map_z_abs_m"] = abs(p3[2] - q3[2])
            if metrics["map_distance_m"] > float(t["map_distance_m"]): reasons.append("map_node_offset")
            if metrics["map_z_abs_m"] > float(t["depth_abs_m"]): reasons.append("map_depth_offset")
    if previous:
        p = det.get("world_position") or det.get("position") or {}; q = previous.get("world_position") or previous.get("position") or {}
        p3, q3 = _point3(p), _point3(q)
        if p3 is not None and q3 is not None:
            metrics["temporal_jump_m"] = math.hypot(p3[0] - q3[0], p3[1] - q3[1])
            if metrics["temporal_jump_m"] > float(t["temporal_jump_m"]): reasons.append("temporal_position_jump")
    confidence = _num(det.get("confidence")); status = "fail" if len(reasons) >= 2 else ("warn" if reasons or confidence < .35 else "pass")
    return {"object_id": det.get("instance_id", det.get("id", "")), "status": status, "metrics": metrics, "reasons": reasons, "confidence": confidence}

def evaluate_frame(detections: list[Mapping[str, Any]], *, graph: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    nodes = (graph or {}).get("nodes", []); reports = []
    for det in detections:
        label = det.get("semantic_class", det.get("class", "")); candidates = [n for n in nodes if n.get("label") == label or n.get("semantic_class") == label]
        det_point = _point3(det.get("world_position") or det.get("position"))
        if det_point is not None and candidates:
            node = min(candidates, key=lambda item: math.dist(det_point, _point3(item.get("world_position") or item.get("position") or item.get("centroid") or item.get("aabb_center")) or (float("inf"),) * 3))
        else:
            node = candidates[0] if candidates else None
        report = evaluate_detection(det, map_node=node, thresholds=thresholds)
        if node is None:
            report["status"] = "warn" if report["status"] == "pass" else report["status"]
            report["reasons"].append("missing_map_node")
        reports.append(report)
    counts = {s: sum(r["status"] == s for r in reports) for s in ("pass", "warn", "fail")}; return {"status": "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass"), "counts": counts, "detections": reports}
