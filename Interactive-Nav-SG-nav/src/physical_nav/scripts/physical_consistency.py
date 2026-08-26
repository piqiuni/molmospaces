#!/usr/bin/env python3
"""Pure-Python perception-to-map consistency checks for the ROS node."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _point3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        return tuple(_num(value.get(axis)) for axis in ("x", "y", "z"))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(_num(item) for item in value[:3])
    return None


def _vector3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    point = _point3(value)
    return point if point is not None else default


def _size3(value: Any, default: tuple[float, float, float] = (0.2, 0.2, 0.2)) -> tuple[float, float, float]:
    point = _point3(value)
    if point is None:
        return default
    return tuple(max(abs(component), 0.02) for component in point)


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _world_to_camera(point: tuple[float, float, float], telemetry: Mapping[str, Any], camera_translation: Sequence[float], camera_rpy: Sequence[float]) -> tuple[float, float, float]:
    position = _vector3(telemetry.get("position"), (0.0, 0.0, 0.0))
    imu = telemetry.get("imu")
    fallback_yaw = imu.get("rpy", [0.0, 0.0, 0.0])[2] if isinstance(imu, Mapping) else 0.0
    yaw = _num(telemetry.get("yaw"), _num(fallback_yaw))
    dx, dy = point[0] - position[0], point[1] - position[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    base = (cy * dx + sy * dy, -sy * dx + cy * dy, point[2] - position[2])
    translation = _vector3(camera_translation, (0.0, 0.0, 0.0))
    shifted = (base[0] - translation[0], base[1] - translation[1], base[2] - translation[2])
    rpy = list(camera_rpy or (0.0, 0.0, 0.0))
    rpy.extend([0.0] * (3 - len(rpy)))
    rotation = _rotation_matrix(_num(rpy[0]), _num(rpy[1]), _num(rpy[2]))
    return tuple(sum(rotation[row][axis] * shifted[row] for row in range(3)) for axis in range(3))


def project_map_node(map_node: Mapping[str, Any], *, intrinsics: Mapping[str, Any], telemetry: Mapping[str, Any], camera_translation: Sequence[float] = (0.0, 0.0, 0.0), camera_rpy: Sequence[float] = (0.0, 0.0, 0.0), image_size: Sequence[int] | None = None) -> dict[str, Any] | None:
    """Project a global graph node's 3-D box into the current RGB image."""
    center = _point3(map_node.get("world_box3d_center") or map_node.get("aabb_center") or map_node.get("box3d_center") or map_node.get("world_position") or map_node.get("position") or map_node.get("centroid"))
    if center is None:
        return None
    size = _size3(map_node.get("world_box3d_size") or map_node.get("aabb_size") or map_node.get("box3d_size") or map_node.get("size"))
    corners = [_world_to_camera((center[0] + sx * size[0], center[1] + sy * size[1], center[2] + sz * size[2]), telemetry, camera_translation, camera_rpy) for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]
    visible = [point for point in corners if point[2] > 1e-3]
    if not visible:
        return None
    fx, fy, cx, cy = (_num(intrinsics.get(key)) for key in ("fx", "fy", "cx", "cy"))
    if fx <= 0.0 or fy <= 0.0:
        return None
    projected = [(fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy) for point in visible]
    bbox = [min(point[0] for point in projected), min(point[1] for point in projected), max(point[0] for point in projected), max(point[1] for point in projected)]
    if image_size is not None and len(image_size) >= 2:
        width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
        bbox = [max(0.0, min(float(width - 1), bbox[0])), max(0.0, min(float(height - 1), bbox[1])), max(0.0, min(float(width - 1), bbox[2])), max(0.0, min(float(height - 1), bbox[3]))]
    depth_values = sorted(point[2] for point in visible)
    middle = len(depth_values) // 2
    depth_m = 0.5 * (depth_values[(len(depth_values) - 1) // 2] + depth_values[middle])
    return {"bbox": bbox, "depth_m": float(depth_m), "camera_center": _world_to_camera(center, telemetry, camera_translation, camera_rpy), "visible_corners": len(visible)}


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def evaluate_detection(detection: Mapping[str, Any], *, projected_bbox: Iterable[float] | None = None, observed_depth_m: Iterable[float] | None = None, map_node: Mapping[str, Any] | None = None, previous: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None, projected_depth_m: float | None = None) -> dict[str, Any]:
    thresholds_all = {"bbox_iou": 0.45, "bbox_center_px": 35.0, "depth_abs_m": 0.25, "map_distance_m": 0.5, "temporal_jump_m": 0.6}
    thresholds_all.update({str(key): _num(value) for key, value in (thresholds or {}).items()})
    metrics: dict[str, float] = {}
    reasons: list[str] = []
    bbox = detection.get("bbox") or detection.get("bbox_2d")
    if projected_bbox is not None and bbox is not None and len(bbox) == 4:
        metrics["bbox_iou"] = bbox_iou(bbox, projected_bbox)
        actual_center = ((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2)
        expected_center = ((float(projected_bbox[0]) + float(projected_bbox[2])) / 2, (float(projected_bbox[1]) + float(projected_bbox[3])) / 2)
        metrics["bbox_center_px"] = math.hypot(actual_center[0] - expected_center[0], actual_center[1] - expected_center[1])
        if metrics["bbox_iou"] < thresholds_all["bbox_iou"]: reasons.append("reprojection_bbox_iou_low")
        if metrics["bbox_center_px"] > thresholds_all["bbox_center_px"]: reasons.append("reprojection_center_offset")
    if projected_depth_m is not None and _num(detection.get("depth_median_m"), -1.0) > 0.0:
        metrics["map_projected_depth_abs_m"] = abs(_num(detection.get("depth_median_m")) - _num(projected_depth_m))
        if metrics["map_projected_depth_abs_m"] > thresholds_all["depth_abs_m"]: reasons.append("map_reprojection_depth_offset")
    if observed_depth_m is not None:
        depths = sorted(_num(value) for value in observed_depth_m if _num(value) > 0)
        position = detection.get("position") or detection.get("world_position") or {}
        if depths and isinstance(position, Mapping) and position.get("z") is not None:
            metrics["depth_median_abs_m"] = abs(_num(position.get("z")) - depths[len(depths) // 2])
            if metrics["depth_median_abs_m"] > thresholds_all["depth_abs_m"]: reasons.append("depth_projection_mismatch")
        if depths: metrics["depth_valid_ratio"] = min(1.0, len(depths) / max(1.0, _num(detection.get("mask_area"), len(depths))))
    sensor_depth = _num(detection.get("depth_median_m"), -1.0)
    camera_point = _point3(detection.get("camera_position"))
    if sensor_depth > 0.0:
        metrics["sensor_depth_m"] = sensor_depth
        metrics["depth_valid_points"] = _num(detection.get("depth_valid_points"))
        if camera_point is not None:
            metrics["rgbd_depth_lift_abs_m"] = abs(camera_point[2] - sensor_depth)
            if metrics["rgbd_depth_lift_abs_m"] > thresholds_all["depth_abs_m"]: reasons.append("rgbd_depth_lift_mismatch")
    if map_node:
        actual = detection.get("world_position") or detection.get("position") or {}
        expected = map_node.get("world_position") or map_node.get("position") or map_node.get("centroid") or map_node.get("aabb_center") or {}
        actual3, expected3 = _point3(actual), _point3(expected)
        if actual3 is not None and expected3 is not None:
            metrics["map_distance_m"] = math.dist(actual3, expected3)
            metrics["map_xy_distance_m"] = math.hypot(actual3[0] - expected3[0], actual3[1] - expected3[1])
            metrics["map_z_abs_m"] = abs(actual3[2] - expected3[2])
            if metrics["map_distance_m"] > thresholds_all["map_distance_m"]: reasons.append("map_node_offset")
            if metrics["map_z_abs_m"] > thresholds_all["depth_abs_m"]: reasons.append("map_depth_offset")
    if previous:
        actual, old = detection.get("world_position") or detection.get("position") or {}, previous.get("world_position") or previous.get("position") or {}
        actual3, old3 = _point3(actual), _point3(old)
        if actual3 is not None and old3 is not None:
            metrics["temporal_jump_m"] = math.hypot(actual3[0] - old3[0], actual3[1] - old3[1])
            if metrics["temporal_jump_m"] > thresholds_all["temporal_jump_m"]: reasons.append("temporal_position_jump")
    confidence = _num(detection.get("confidence"))
    status = "fail" if len(reasons) >= 2 else ("warn" if reasons or confidence < 0.35 else "pass")
    return {"object_id": detection.get("instance_id", detection.get("id", "")), "status": status, "metrics": metrics, "reasons": reasons, "confidence": confidence}


def evaluate_frame(detections: list[Mapping[str, Any]], *, graph: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None, projection_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    nodes = (graph or {}).get("nodes", []) if isinstance(graph, Mapping) else []
    reports = []
    for detection in detections:
        label = detection.get("semantic_class", detection.get("class", ""))
        candidates = [node for node in nodes if node.get("label") == label or node.get("semantic_class") == label or node.get("semantic_name") == label]
        det_point = _point3(detection.get("world_position") or detection.get("position"))
        candidate = min(candidates, key=lambda node: math.dist(det_point, _point3(node.get("world_position") or node.get("position") or node.get("centroid") or node.get("aabb_center")) or (float("inf"),) * 3)) if det_point is not None and candidates else (candidates[0] if candidates else None)
        projected = None
        if candidate is not None and projection_context:
            projected = project_map_node(candidate, intrinsics=projection_context.get("intrinsics", {}), telemetry=projection_context.get("telemetry", {}), camera_translation=projection_context.get("camera_translation", (0.0, 0.0, 0.0)), camera_rpy=projection_context.get("camera_rpy", (0.0, 0.0, 0.0)), image_size=projection_context.get("image_size"))
        report = evaluate_detection(detection, projected_bbox=projected.get("bbox") if projected else None, projected_depth_m=projected.get("depth_m") if projected else None, map_node=candidate, thresholds=thresholds)
        if projected is not None: report["projection"] = projected
        if candidate is None:
            report["status"] = "warn" if report["status"] == "pass" else report["status"]
            report["reasons"].append("missing_map_node")
        reports.append(report)
    counts = {status: sum(report["status"] == status for report in reports) for status in ("pass", "warn", "fail")}
    return {"status": "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass"), "counts": counts, "detections": reports}
