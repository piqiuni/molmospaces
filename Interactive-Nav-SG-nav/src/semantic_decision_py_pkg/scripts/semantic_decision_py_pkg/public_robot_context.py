"""Coordinate conversion for model context only; execution poses are untouched."""
from __future__ import annotations

import math
from typing import Any


def graph_robot_pose_context(
    source_xyz: Any, source_frame: str, graph_frame: str, *,
    source_stamp_sec: float | None = None,
    transform: tuple[Any, Any] | None = None,
    transform_error: str = "",
) -> dict[str, Any]:
    source_frame = str(source_frame or "").lstrip("/")
    graph_frame = str(graph_frame or "").lstrip("/")
    source = {
        "source_xy": list(source_xyz or [])[:2],
        "source_frame_id": source_frame or None,
        "source_pose_stamp_sec": source_stamp_sec,
        "graph_frame_id": graph_frame or None,
    }
    result = {
        "robot_graph_xy": None,
        "robot_graph_frame_id": graph_frame or None,
        "robot_graph_pose_source": source,
    }
    try:
        xyz = [float(value) for value in list(source_xyz or [])[:3]]
        if len(xyz) < 2 or not all(math.isfinite(value) for value in xyz):
            raise ValueError("invalid_pose")
        if len(xyz) == 2:
            xyz.append(0.0)
    except (TypeError, ValueError):
        source.update(kind="unavailable", reason="robot_pose_unavailable")
        return result
    if not source_frame or not graph_frame:
        source.update(kind="unavailable", reason="coordinate_frame_unknown")
        return result
    if source_frame == graph_frame:
        result["robot_graph_xy"] = xyz[:2]
        source["kind"] = "same_frame"
        return result
    if transform is None:
        source.update(kind="unavailable", reason=transform_error or "transform_unavailable")
        return result
    try:
        translation, rotation = transform
        tx, ty, _ = map(float, translation)
        qx, qy, qz, qw = map(float, rotation)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("invalid_quaternion")
        qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
        x, y, z = xyz
        graph_xy = [
            tx + (1 - 2 * (qy * qy + qz * qz)) * x + 2 * (qx * qy - qz * qw) * y + 2 * (qx * qz + qy * qw) * z,
            ty + 2 * (qx * qy + qz * qw) * x + (1 - 2 * (qx * qx + qz * qz)) * y + 2 * (qy * qz - qx * qw) * z,
        ]
        if not all(math.isfinite(value) for value in graph_xy):
            raise ValueError("nonfinite_transform")
    except (TypeError, ValueError):
        source.update(kind="unavailable", reason="invalid_transform")
        return result
    result["robot_graph_xy"] = graph_xy
    source["kind"] = "tf_transform"
    return result
