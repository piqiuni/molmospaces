"""Public-geometry contracts for clearance-aware portal approach poses."""
from __future__ import annotations

import math


def is_clearance_aware_portal(candidate: dict) -> bool:
    return (
        str(candidate.get("behavior_type", "")).upper() == "INTERACT"
        and bool((candidate.get("metadata") or {}).get("portal_clearance_aware_approach"))
    )


def bounded_portal_tolerances(goal, center, axis, xy_tolerance, yaw_tolerance,
                              position_angle_tolerance, yaw_angle_tolerance, *, shrink_arrival_disc=True):
    """Fit the entire arrival disc and yaw interval inside one immutable face."""
    try:
        x, y, yaw = map(float, goal[:3])
        cx, cy = map(float, center[:2])
        ax, ay = map(float, axis[:2])
        xy, yt, pa, ya = map(float, (
            xy_tolerance, yaw_tolerance, position_angle_tolerance, yaw_angle_tolerance))
        if not all(math.isfinite(v) for v in (x, y, yaw, cx, cy, ax, ay, xy, yt, pa, ya)):
            return None
        norm = math.hypot(ax, ay)
        radius = math.hypot(x-cx, y-cy)
        if norm < 1e-6 or radius < 1e-6 or min(xy, yt, pa, ya) <= 0.0:
            return None
        face_angle = math.atan2(ay, ax)
        position_offset = abs(math.atan2(math.sin(math.atan2(y-cy, x-cx)-face_angle),
                                        math.cos(math.atan2(y-cy, x-cx)-face_angle)))
        heading_offset = abs(math.atan2(math.sin(yaw-face_angle-math.pi),
                                       math.cos(yaw-face_angle-math.pi)))
        position_budget = min(math.pi/2, pa-position_offset-0.005)
        yaw_budget = ya-heading_offset-0.005
        if min(position_budget, yaw_budget) <= 0.0:
            return None
        if shrink_arrival_disc:
            xy = min(xy, radius*math.sin(position_budget))
        yt = min(yt, yaw_budget)
        if xy < 0.05 or yt < 0.05:
            return None
        return {"distance_tolerance_m": xy, "yaw_tolerance_rad": yt,
                "planned_face_position_offset_rad": position_offset,
                "planned_face_yaw_offset_rad": heading_offset}
    except (TypeError, ValueError, IndexError):
        return None


def portal_approach_profile(candidate: dict, goal):
    command = candidate.get("interaction_command") or {}
    metadata = candidate.get("metadata") or {}
    base = metadata.get("portal_approach_base_tolerances") or [
        command.get("navigation_goal_position_tolerance_m", 0.15),
        command.get("navigation_goal_yaw_tolerance_rad", 0.15),
    ]
    return bounded_portal_tolerances(
        goal, command.get("interaction_target_center_xy", []),
        command.get("interaction_approach_axis_xy", []), *base[:2],
        command.get("interaction_front_position_tolerance_rad", 0.15),
        command.get("interaction_front_yaw_tolerance_rad", 0.15),
        shrink_arrival_disc=metadata.get("portal_shrink_arrival_disc", True),
    )
