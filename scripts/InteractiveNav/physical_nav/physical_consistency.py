#!/usr/bin/env python3
"""Pure-Python perception-to-map consistency metrics for the physical UI."""

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
    if point is None:
        return default
    return point


def _size3(value: Any, default: tuple[float, float, float] = (0.2, 0.2, 0.2)) -> tuple[float, float, float]:
    point = _point3(value)
    if point is None:
        return default
    # A mapper can briefly publish a position-only node.  A small finite box
    # lets us still test its projected center without inventing a zero-area
    # image rectangle.
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


def _quaternion_matrix(quaternion: Sequence[float]) -> tuple[tuple[float, float, float], ...] | None:
    if len(quaternion) < 4:
        return None
    x, y, z, w = [_num(value) for value in quaternion[:4]]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-6:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _world_to_camera(
    point: tuple[float, float, float],
    telemetry: Mapping[str, Any],
    camera_translation: Sequence[float],
    camera_rpy: Sequence[float],
    *,
    optical_frame: bool = False,
) -> tuple[float, float, float]:
    """Transform a map-frame point into the D435i optical frame.

    The YOLOE worker uses ``p_base = R_camera_to_base p_camera + t`` and a
    planar Go2 pose.  This is the exact inverse, kept dependency-free so the
    ROS Noetic consistency node can run with system Python.
    """
    position = _vector3(telemetry.get("position"), (0.0, 0.0, 0.0))
    yaw = _num(telemetry.get("yaw"), _num((telemetry.get("imu") or {}).get("rpy", [0.0, 0.0, 0.0])[2] if isinstance(telemetry.get("imu"), Mapping) else 0.0))
    dx, dy, dz = point[0] - position[0], point[1] - position[1], point[2] - position[2]
    body_rotation = _quaternion_matrix(telemetry.get("quaternion") or [])
    if body_rotation is not None:
        # Full live orientation (including pitch/roll) from Go2/D435i pose.
        relative = (dx, dy, dz)
        base = tuple(
            sum(body_rotation[column][row] * relative[row] for row in range(3))
            for column in range(3)
        )
    else:
        cy, sy = math.cos(yaw), math.sin(yaw)
        # world -> base, inverse of the worker's planar base -> world transform.
        base = (cy * dx + sy * dy, -sy * dx + cy * dy, dz)
    translation = _vector3(camera_translation, (0.0, 0.0, 0.0))
    shifted = (base[0] - translation[0], base[1] - translation[1], base[2] - translation[2])
    rpy = list(camera_rpy or (0.0, 0.0, 0.0))
    while len(rpy) < 3:
        rpy.append(0.0)
    rotation = _rotation_matrix(_num(rpy[0]), _num(rpy[1]), _num(rpy[2]))
    # R.T @ shifted gives the nominal camera frame.  Convert base axes to the
    # D435i REP-103 optical axes when requested (x right, y down, z forward).
    mounted = tuple(sum(rotation[row][axis] * shifted[row] for row in range(3)) for axis in range(3))
    if optical_frame:
        # inverse of optical->base: optical = [ -base_y, -base_z, base_x ]
        return (-mounted[1], -mounted[2], mounted[0])
    return mounted


def _project_camera_point(point: tuple[float, float, float], intrinsics: Mapping[str, Any]) -> tuple[float, float] | None:
    if point[2] <= 1e-3:
        return None
    fx, fy = _num(intrinsics.get("fx")), _num(intrinsics.get("fy"))
    cx, cy = _num(intrinsics.get("cx")), _num(intrinsics.get("cy"))
    if fx <= 0.0 or fy <= 0.0:
        return None
    x, y = point[0] / point[2], point[1] / point[2]
    distortion = list(intrinsics.get("distortion") or [])
    distortion_model = str(intrinsics.get("distortion_model", "")).lower()
    # RealSense may advertise inverse-brown-conrady.  Applying the forward
    # Brown-Conrady polynomial to that stream would make reprojection worse,
    # so retain the calibrated pinhole fallback until an inverse solver is
    # supplied.
    if len(distortion) >= 4 and "inverse" not in distortion_model:
        k1, k2, p1, p2 = [_num(value) for value in distortion[:4]]
        k3 = _num(distortion[4]) if len(distortion) >= 5 else 0.0
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
        x, y = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x), y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return fx * x + cx, fy * y + cy


def _transform_world_to_camera(point: tuple[float, float, float], transform: Mapping[str, Any]) -> tuple[float, float, float]:
    translation = _vector3(transform.get("translation"), (0.0, 0.0, 0.0))
    quaternion = list(transform.get("quaternion") or [])
    if len(quaternion) < 4:
        return point
    x, y, z, w = [_num(value) for value in quaternion[:4]]
    px, py, pz = point
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    return (
        px + w * tx + y * tz - z * ty + translation[0],
        py + w * ty + z * tx - x * tz + translation[1],
        pz + w * tz + x * ty - y * tx + translation[2],
    )


def project_map_node(
    map_node: Mapping[str, Any],
    *,
    intrinsics: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    camera_translation: Sequence[float] = (0.0, 0.0, 0.0),
    camera_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    image_size: Sequence[int] | None = None,
    world_to_camera: Mapping[str, Any] | None = None,
    camera_optical: bool = False,
) -> dict[str, Any] | None:
    """Project a global graph node's 3-D box into the current RGB image.

    Returns ``None`` when the node has no usable position or is entirely behind
    the camera.  The returned depth is camera-Z (the same quantity used by
    D435i/YOLOE), not world-Z.
    """
    center = _point3(
        map_node.get("world_box3d_center")
        or map_node.get("aabb_center")
        or map_node.get("box3d_center")
        or map_node.get("world_position")
        or map_node.get("position")
        or map_node.get("centroid")
    )
    if center is None:
        return None
    size = _size3(
        map_node.get("world_box3d_size")
        or map_node.get("aabb_size")
        or map_node.get("box3d_size")
        or map_node.get("size")
    )
    def transform_corner(corner: tuple[float, float, float]) -> tuple[float, float, float]:
        if world_to_camera:
            return _transform_world_to_camera(corner, world_to_camera)
        return _world_to_camera(corner, telemetry, camera_translation, camera_rpy, optical_frame=camera_optical)

    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                corners.append(transform_corner((center[0] + sx * size[0], center[1] + sy * size[1], center[2] + sz * size[2])))
    visible = [point for point in corners if point[2] > 1e-3]
    if not visible:
        return None
    if _num(intrinsics.get("fx")) <= 0.0 or _num(intrinsics.get("fy")) <= 0.0:
        return None
    projected = [pixel for point in visible if (pixel := _project_camera_point(point, intrinsics)) is not None]
    if not projected:
        return None
    bbox = [min(point[0] for point in projected), min(point[1] for point in projected), max(point[0] for point in projected), max(point[1] for point in projected)]
    if image_size is not None and len(image_size) >= 2:
        width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
        bbox = [max(0.0, min(float(width - 1), bbox[0])), max(0.0, min(float(height - 1), bbox[1])), max(0.0, min(float(width - 1), bbox[2])), max(0.0, min(float(height - 1), bbox[3]))]
    depth_values = sorted(point[2] for point in visible)
    middle = len(depth_values) // 2
    depth_m = 0.5 * (depth_values[(len(depth_values) - 1) // 2] + depth_values[middle])
    return {
        "bbox": bbox,
        "depth_m": float(depth_m),
        "camera_center": _transform_world_to_camera(center, world_to_camera) if world_to_camera else _world_to_camera(center, telemetry, camera_translation, camera_rpy, optical_frame=camera_optical),
        "visible_corners": len(visible),
    }


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]; bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1); area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def evaluate_detection(detection: Mapping[str, Any], *, projected_bbox: Iterable[float] | None = None,
                       observed_depth_m: Iterable[float] | None = None, map_node: Mapping[str, Any] | None = None,
                       previous: Mapping[str, Any] | None = None, thresholds: Mapping[str, float] | None = None,
                       projected_depth_m: float | None = None) -> dict[str, Any]:
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
    if projected_depth_m is not None and _num(detection.get("depth_median_m"), -1.0) > 0.0:
        metrics["map_projected_depth_abs_m"] = abs(_num(detection.get("depth_median_m")) - _num(projected_depth_m))
        if metrics["map_projected_depth_abs_m"] > t["depth_abs_m"]:
            reasons.append("map_reprojection_depth_offset")
    if observed_depth_m is not None:
        depths = sorted(float(v) for v in observed_depth_m if _num(v) > 0)
        if depths:
            median = depths[len(depths) // 2]; position = detection.get("position") or detection.get("world_position") or {}
            z = position.get("z") if isinstance(position, Mapping) else None
            if z is not None:
                metrics["depth_median_abs_m"] = abs(_num(z) - median)
                if metrics["depth_median_abs_m"] > t["depth_abs_m"]: reasons.append("depth_projection_mismatch")
            metrics["depth_valid_ratio"] = min(1.0, float(len(depths)) / max(1.0, _num(detection.get("mask_area"), len(depths))))
    sensor_depth = _num(detection.get("depth_median_m"), -1.0)
    camera_point = _point3(detection.get("camera_position"))
    if sensor_depth > 0.0:
        metrics["sensor_depth_m"] = sensor_depth
        metrics["depth_valid_points"] = _num(detection.get("depth_valid_points"))
        if camera_point is not None:
            metrics["rgbd_depth_lift_abs_m"] = abs(camera_point[2] - sensor_depth)
            if metrics["rgbd_depth_lift_abs_m"] > t["depth_abs_m"]:
                reasons.append("rgbd_depth_lift_mismatch")
    if map_node:
        p = detection.get("world_position") or detection.get("position") or {}; q = map_node.get("world_position") or map_node.get("position") or map_node.get("centroid") or map_node.get("aabb_center") or {}
        p3, q3 = _point3(p), _point3(q)
        if p3 is not None and q3 is not None:
            metrics["map_distance_m"] = math.sqrt(sum((a - b) ** 2 for a, b in zip(p3, q3)))
            metrics["map_xy_distance_m"] = math.hypot(p3[0] - q3[0], p3[1] - q3[1])
            metrics["map_z_abs_m"] = abs(p3[2] - q3[2])
            if metrics["map_distance_m"] > t["map_distance_m"]: reasons.append("map_node_offset")
            if metrics["map_z_abs_m"] > t["depth_abs_m"]: reasons.append("map_depth_offset")
    if previous:
        p = detection.get("world_position") or detection.get("position") or {}; q = previous.get("world_position") or previous.get("position") or {}
        p3, q3 = _point3(p), _point3(q)
        if p3 is not None and q3 is not None:
            metrics["temporal_jump_m"] = math.hypot(p3[0] - q3[0], p3[1] - q3[1])
            if metrics["temporal_jump_m"] > t["temporal_jump_m"]: reasons.append("temporal_position_jump")
    confidence = _num(detection.get("confidence")); status = "fail" if len(reasons) >= 2 else ("warn" if reasons or confidence < .35 else "pass")
    return {"object_id": detection.get("instance_id", detection.get("id", "")), "status": status, "metrics": metrics, "reasons": reasons, "confidence": confidence}


def evaluate_frame(
    detections: list[Mapping[str, Any]],
    *,
    graph: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
    projection_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    nodes = (graph or {}).get("nodes", []) if isinstance(graph, Mapping) else []; reports = []
    for det in detections:
        label = det.get("semantic_class", det.get("class", "")); candidates = [node for node in nodes if node.get("label") == label or node.get("semantic_class") == label]
        det_point = _point3(det.get("world_position") or det.get("position"))
        if det_point is not None and candidates:
            candidate = min(candidates, key=lambda node: math.dist(det_point, _point3(node.get("world_position") or node.get("position") or node.get("centroid") or node.get("aabb_center")) or (float("inf"),) * 3))
        else:
            candidate = candidates[0] if candidates else None
        projected = None
        if candidate is not None and projection_context:
            projected = project_map_node(
                candidate,
                intrinsics=projection_context.get("intrinsics", {}),
                telemetry=projection_context.get("telemetry", {}),
                camera_translation=projection_context.get("camera_translation", (0.0, 0.0, 0.0)),
                camera_rpy=projection_context.get("camera_rpy", (0.0, 0.0, 0.0)),
                image_size=projection_context.get("image_size"),
                world_to_camera=projection_context.get("world_to_camera"),
                camera_optical=bool(projection_context.get("camera_optical", False)),
            )
        report = evaluate_detection(
            det,
            projected_bbox=projected.get("bbox") if projected else None,
            projected_depth_m=projected.get("depth_m") if projected else None,
            map_node=candidate,
            thresholds=thresholds,
        )
        if projected is not None:
            report["projection"] = projected
        if candidate is None:
            report["status"] = "warn" if report["status"] == "pass" else report["status"]
            report["reasons"].append("missing_map_node")
        reports.append(report)
    counts = {status: sum(report["status"] == status for report in reports) for status in ("pass", "warn", "fail")}
    return {"status": "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass"), "counts": counts, "detections": reports}
