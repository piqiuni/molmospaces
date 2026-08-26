#!/usr/bin/env python3
"""Pure-Python perception-to-map consistency metrics for the physical UI."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]; bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1); area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def evaluate_detection(detection: Mapping[str, Any], *, projected_bbox: Iterable[float] | None = None,
                       observed_depth_m: Iterable[float] | None = None, map_node: Mapping[str, Any] | None = None,
                       previous: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    t = {"bbox_iou": .45, "bbox_center_px": 35., "depth_abs_m": .25, "map_distance_m": .5, "temporal_jump_m": .6}
    t.update({str(k): _num(v) for k, v in (thresholds or {}).items()}); metrics: dict[str, float] = {}; reasons: list[str] = []
    bbox = detection.get("bbox") or detection.get("bbox_2d")
    if projected_bbox is not None and bbox is not None and len(bbox) == 4:
        metrics["bbox_iou"] = bbox_iou(bbox, projected_bbox)
        ax, ay = (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2
        bx, by = (float(projected_bbox[0]) + float(projected_bbox[2])) / 2, (float(projected_bbox[1]) + float(projected_bbox[3])) / 2
        metrics["bbox_center_px"] = math.hypot(ax - bx, ay - by)
        if metrics["bbox_iou"] < t["bbox_iou"]: reasons.append("reprojection_bbox_iou_low")
        if metrics["bbox_center_px"] > t["bbox_center_px"]: reasons.append("reprojection_center_offset")
    if observed_depth_m is not None:
        depths = sorted(float(v) for v in observed_depth_m if _num(v) > 0)
        if depths:
            median = depths[len(depths) // 2]; position = detection.get("position") or detection.get("world_position") or {}
            z = position.get("z") if isinstance(position, Mapping) else None
            if z is not None:
                metrics["depth_median_abs_m"] = abs(_num(z) - median)
                if metrics["depth_median_abs_m"] > t["depth_abs_m"]: reasons.append("depth_projection_mismatch")
            metrics["depth_valid_ratio"] = min(1.0, float(len(depths)) / max(1.0, _num(detection.get("mask_area"), len(depths))))
    if map_node:
        p = detection.get("world_position") or detection.get("position") or {}; q = map_node.get("world_position") or map_node.get("position") or map_node.get("centroid") or map_node.get("aabb_center") or {}
        if isinstance(p, Mapping) and isinstance(q, Mapping):
            metrics["map_distance_m"] = math.sqrt(sum((_num(p.get(k)) - _num(q.get(k))) ** 2 for k in ("x", "y", "z")))
            if metrics["map_distance_m"] > t["map_distance_m"]: reasons.append("map_node_offset")
    if previous:
        p = detection.get("world_position") or detection.get("position") or {}; q = previous.get("world_position") or previous.get("position") or {}
        if isinstance(p, Mapping) and isinstance(q, Mapping):
            metrics["temporal_jump_m"] = math.hypot(_num(p.get("x")) - _num(q.get("x")), _num(p.get("y")) - _num(q.get("y")))
            if metrics["temporal_jump_m"] > t["temporal_jump_m"]: reasons.append("temporal_position_jump")
    confidence = _num(detection.get("confidence")); status = "fail" if len(reasons) >= 2 else ("warn" if reasons or confidence < .35 else "pass")
    return {"object_id": detection.get("instance_id", detection.get("id", "")), "status": status, "metrics": metrics, "reasons": reasons, "confidence": confidence}


def evaluate_frame(detections: list[Mapping[str, Any]], *, graph: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    nodes = (graph or {}).get("nodes", []) if isinstance(graph, Mapping) else []; reports = []
    for det in detections:
        label = det.get("semantic_class", det.get("class", "")); candidate = next((node for node in nodes if node.get("label") == label or node.get("semantic_class") == label), None)
        reports.append(evaluate_detection(det, map_node=candidate, thresholds=thresholds))
    counts = {status: sum(report["status"] == status for report in reports) for status in ("pass", "warn", "fail")}
    return {"status": "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass"), "counts": counts, "detections": reports}
