"""Pure approach geometry shared by deployments; callers own navigation state."""

from __future__ import annotations

import math
from typing import Any


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def portal_approach_pose(
    robot_xy: tuple[float, float],
    target_xy: tuple[float, float],
    node: dict[str, Any],
    standoff_m: float,
    side_multiplier: float = 1.0,
    tangent_offset_m: float = 0.0,
    *,
    previous_side: float | None = None,
    side_hysteresis_m: float = 0.0,
) -> tuple[list[float], float | None]:
    selected_side = previous_side
    attributes = node.get("attributes") or {}
    reference_center = list(
        attributes.get("interaction_reference_aabb_center")
        or node.get("aabb_center")
        or target_xy
    )
    if len(reference_center) >= 2:
        target_xy = float(reference_center[0]), float(reference_center[1])
    size = list(
        attributes.get("interaction_reference_aabb_size")
        or node.get("aabb_size")
        or []
    )
    boundary_distance = 0.0
    size_x = max(0.0, float(size[0])) if len(size) >= 1 else 0.0
    size_y = max(0.0, float(size[1])) if len(size) >= 2 else 0.0
    major = max(size_x, size_y)
    minor = min(size_x, size_y)
    elongated = major > 1e-6 and major / max(minor, 1e-6) >= 1.35
    reference_yaw = attributes.get("interaction_reference_yaw")
    if reference_yaw is None:
        reference_yaw = attributes.get("yaw")
    try:
        reference_yaw = (
            None if reference_yaw is None else float(reference_yaw)
        )
    except (TypeError, ValueError):
        reference_yaw = None
    if reference_yaw is not None:
        # Physical OBB yaw is the door's long/tangent axis. Its
        # perpendicular is the actual door normal even when the door is
        # rotated relative to the map X/Y axes.
        normal_x = -math.sin(reference_yaw)
        normal_y = math.cos(reference_yaw)
        signed_distance = (
            (robot_xy[0] - target_xy[0]) * normal_x
            + (robot_xy[1] - target_xy[1]) * normal_y
        )
        side = 1.0 if signed_distance >= 0.0 else -1.0
        hysteresis = max(0.0, float(side_hysteresis_m))
        if previous_side is not None:
            # Change sides only after the robot is clearly beyond the
            # door plane. Near-plane noise must retain the old stance.
            if signed_distance * previous_side >= -hysteresis:
                side = previous_side
        selected_side = side
        normal_x *= side * float(side_multiplier)
        normal_y *= side * float(side_multiplier)
        boundary_distance = 0.5 * minor
    elif elongated:
        # The short AABB axis is the doorway normal.  Use the robot side
        # for the primary pose and allow the caller to request the other
        # side with side_multiplier=-1.
        if size_x <= size_y:
            normal_x, normal_y = 1.0, 0.0
        else:
            normal_x, normal_y = 0.0, 1.0
        side = 1.0 if (
            (robot_xy[0] - target_xy[0]) * normal_x
            + (robot_xy[1] - target_xy[1]) * normal_y
        ) >= 0.0 else -1.0
        normal_x *= side * float(side_multiplier)
        normal_y *= side * float(side_multiplier)
        boundary_distance = 0.5 * minor
    else:
        # A radial robot-to-door bearing is not a door normal.  In the
        # minimal-GT stream the portal AABB can legitimately be [0, 0, 0];
        # using that bearing creates diagonal/side-offset interaction
        # points (and makes the chosen face depend on where the robot
        # happened to be).  Prefer an explicitly persisted normal when one
        # exists.  Otherwise quantize the bearing only to select the
        # source side of a cardinal fallback, never to define an arbitrary
        # diagonal interaction direction.
        explicit_axis = (
            attributes.get("interaction_approach_axis_xy")
            or attributes.get("portal_normal_xy")
            or attributes.get("door_normal_xy")
        )
        axis_norm = 0.0
        if isinstance(explicit_axis, (list, tuple)) and len(explicit_axis) >= 2:
            try:
                candidate_x = float(explicit_axis[0])
                candidate_y = float(explicit_axis[1])
                axis_norm = math.hypot(candidate_x, candidate_y)
            except (TypeError, ValueError):
                axis_norm = 0.0
        if axis_norm > 1e-6:
            normal_x = candidate_x / axis_norm
            normal_y = candidate_y / axis_norm
        else:
            dx = float(robot_xy[0]) - float(target_xy[0])
            dy = float(robot_xy[1]) - float(target_xy[1])
            # Use the dominant world axis for side selection.  This keeps
            # the pose on a straight cardinal ray instead of the old
            # robot-bearing diagonal when geometry is unavailable.
            if abs(dx) >= abs(dy):
                normal_x, normal_y = (1.0 if dx >= 0.0 else -1.0), 0.0
            else:
                normal_x, normal_y = 0.0, (1.0 if dy >= 0.0 else -1.0)
        if size_x > 1e-6 and size_y > 1e-6:
            ray_denominator = abs(normal_x) / (0.5 * size_x) + abs(normal_y) / (0.5 * size_y)
            if ray_denominator > 1e-6:
                boundary_distance = 1.0 / ray_denominator
        normal_x *= float(side_multiplier)
        normal_y *= float(side_multiplier)
    offset = max(0.0, standoff_m) + boundary_distance
    tangent_x, tangent_y = -normal_y, normal_x
    x = target_xy[0] + normal_x * offset + tangent_x * float(tangent_offset_m)
    y = target_xy[1] + normal_y * offset + tangent_y * float(tangent_offset_m)
    # Every candidate faces the interaction reference, including the
    # opposite-side and tangential fallbacks.
    yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
    return [x, y, yaw], selected_side
