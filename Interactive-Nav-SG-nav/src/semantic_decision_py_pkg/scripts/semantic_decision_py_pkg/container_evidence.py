"""Public M1 evidence contracts, independent of ROS, TF and request ownership."""

from __future__ import annotations

import math


def finite_public_bbox(value: object) -> list[float] | None:
    """Normalize one detector box carried by a targeted M1 update."""

    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right - left < 1.0 or bottom - top < 1.0:
        return None
    return [left, top, right, bottom]


def public_step_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        step = int(value)
    except (TypeError, ValueError):
        return None
    return step if step >= 0 else None


def container_visual_truncation_reason(update: dict) -> str:
    """Reject only missing lateral contact evidence for a container.

    A low camera can clip the bottom of a tall drawer/fridge box while the
    front plane and all model-provided action regions remain visible.  That
    is different from clipping the left/right boundary, which prevents a
    reliable frontality judgement.  Older messages without the public edge
    list remain fail-closed.
    """

    if not bool(update.get("visual_evidence_truncated", False)):
        return ""
    raw_edges = update.get("visual_evidence_truncated_edges")
    if not isinstance(raw_edges, (list, tuple, set)):
        return "m1_visual_evidence_truncated"
    edges = {str(edge).strip().casefold() for edge in raw_edges}
    if not edges:
        return "m1_visual_evidence_truncated"
    if edges.intersection({"left", "right", "unknown"}):
        return "m1_visual_evidence_laterally_truncated"
    return ""


def container_m1_pre_action_ready(
    update: dict,
    request: dict,
    *,
    require_direct_front: bool = True,
) -> str:
    """Validate the public fresh-M1 contract before opening a container.

    The state machine makes the final transition.  This helper only turns
    malformed/stale updates into an ordinary bounded re-observation rather
    than allowing a ring pose or stale detector frame to reach the bridge.
    """

    status = str(update.get("attribute_status") or "").strip().casefold()
    if status != "ready":
        return f"m1_attribute_status_{status or 'missing'}"
    if update.get("is_currently_visible") is not True:
        return "m1_target_not_currently_visible"
    truncation_reason = container_visual_truncation_reason(update)
    if truncation_reason:
        return truncation_reason
    if finite_public_bbox(update.get("observed_bbox_2d")) is None:
        return "m1_public_bbox_unavailable"
    capture_step = public_step_or_none(
        update.get("observation_capture_step")
        or update.get("attribute_capture_step")
    )
    if capture_step is None:
        return "m1_capture_step_unavailable"
    minimum_capture_step = public_step_or_none(
        request.get("minimum_capture_step")
    )
    if (
        minimum_capture_step is not None
        and capture_step <= minimum_capture_step
    ):
        return "m1_capture_not_fresh"
    view_state = str(update.get("view_state") or "unknown").strip().casefold()
    allowed_views = (
        {"front"}
        if require_direct_front
        else {"front", "oblique"}
    )
    if view_state not in allowed_views:
        return f"m1_view_state_{view_state}"
    if update.get("front_surface_visible") is not True:
        return "m1_front_surface_not_visible"
    if update.get("approach_ready") is not True:
        return "m1_approach_not_ready"
    if bool(update.get("needs_reobserve", False)):
        return "m1_needs_reobserve"
    return "ready"


def container_m1_front_axis_from_capture_pose(
    candidate: dict,
    capture_pose_xyyaw: list | tuple,
    *,
    selected_face_axis_xy: object = None,
    selected_face_index: object = None,
    selected_face_id: object = "",
) -> dict | None:
    """Freeze a world-space face only after M1 confirms the current image.

    M1 intentionally supplies a categorical visual claim, not a map-space
    normal.  The calibrated capture pose supplies the metric half: the
    target-to-camera ray is meaningful only because the accepted M1 image
    established that it sees the target's usable front.  Never substitute a
    graph ``interaction_approach_axis_xy`` here: it may be oracle geometry.
    """

    metadata = candidate.get("metadata") or {}
    if not bool(metadata.get("container_m1_front_axis_from_capture", False)):
        return {}
    if bool(metadata.get("container_m1_face_selection_enabled", False)):
        selected_axis = list(selected_face_axis_xy or [])
        if len(selected_axis) >= 2:
            try:
                axis_x = float(selected_axis[0])
                axis_y = float(selected_axis[1])
            except (TypeError, ValueError):
                return None
            norm = math.hypot(axis_x, axis_y)
            if not math.isfinite(norm) or norm <= 1e-6:
                return None
            axis_x /= norm
            axis_y /= norm
            try:
                staging_index = int(selected_face_index)
            except (TypeError, ValueError):
                staging_index = -1
            return {
                "m1_front_axis_xy": [axis_x, axis_y],
                "m1_front_yaw": math.atan2(-axis_y, -axis_x),
                "m1_front_axis_source": "m1_selected_montage_view_aabb_cardinal_face",
                "m1_front_staging_index": staging_index,
                "m1_front_face_id": str(selected_face_id or ""),
            }
        try:
            staging_index = int(
                metadata.get(
                    "container_two_stage_staging_goal_option_index",
                    metadata.get("interaction_approach_goal_option_index", 0),
                )
            )
        except (TypeError, ValueError):
            return None
        axes = list(metadata.get("container_face_axis_xy_by_staging_index") or [])
        axis_values = (
            list(axes[staging_index] or [])
            if 0 <= staging_index < len(axes)
            else []
        )
        if len(axis_values) < 2:
            return None
        try:
            axis_x = float(axis_values[0])
            axis_y = float(axis_values[1])
        except (TypeError, ValueError):
            return None
        norm = math.hypot(axis_x, axis_y)
        if not math.isfinite(norm) or norm <= 1e-6:
            return None
        axis_x /= norm
        axis_y /= norm
        return {
            "m1_front_axis_xy": [axis_x, axis_y],
            "m1_front_yaw": math.atan2(-axis_y, -axis_x),
            "m1_front_axis_source": "m1_confirmed_obb_face_normal",
            "m1_front_staging_index": staging_index,
        }
    anchor = list(metadata.get("container_geometry_anchor_xy") or [])
    if len(anchor) < 2 or len(capture_pose_xyyaw) < 2:
        return None
    try:
        axis_x = float(capture_pose_xyyaw[0]) - float(anchor[0])
        axis_y = float(capture_pose_xyyaw[1]) - float(anchor[1])
    except (TypeError, ValueError):
        return None
    norm = math.hypot(axis_x, axis_y)
    if not math.isfinite(norm) or norm <= 1e-6:
        return None
    axis_x /= norm
    axis_y /= norm
    return {
        "m1_front_axis_xy": [axis_x, axis_y],
        "m1_front_yaw": math.atan2(-axis_y, -axis_x),
        "m1_front_axis_source": "m1_confirmed_capture_pose",
    }
