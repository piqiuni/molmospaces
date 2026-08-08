from __future__ import annotations

from typing import Any


VISUAL_INTERACTION_PLANNING_INSTRUCTION = """Inspect one composite RGB input for the selected interaction target after navigation.
The background is the complete head-camera image with the selected target outlined. A labelled inset is a padded crop of that same outlined target. Use the full-image background to judge the robot/object viewing relationship and the inset only to localize visible actionable regions.
Return one JSON object only, with this exact shape:
{"target_type":"door|drawer_container|other_container|unknown","action":"open|scan","operation_method":"hinged_push|hinged_pull|hinged_unknown|double_hinged|slide_left|slide_right|pull|unknown","view_state":"front|oblique|side_or_back|occluded|unknown","approach_ready":false,"reposition_required":false,"open_regions":[{"center":[0.0,0.0],"confidence":0.0}],"confidence":0.0,"reason":"short reason"}

Rules:
- Coordinates are normalized to the padded target-crop inset: x=0 is left, y=0 is top, and both values must be in [0,1].
- For a door, describe the visible door mechanism in operation_method. Use hinged_unknown when push versus pull cannot be inferred from the images. Use double_hinged for a visible two-leaf hinged door. open_regions may contain visible handle or actionable panel centers.
- For a drawer_container, action must be scan, operation_method must be pull, and open_regions must contain the center of every independently openable visible drawer front or handle, ordered from top to bottom. Do not invent occluded drawers.
- For another container, only set approach_ready=true when the object is visibly front-facing enough to safely identify a door/lid/pull surface. If the outlined object is side-on, rear-facing, occluded, outside the outline, or its front cannot be judged, set approach_ready=false, reposition_required=true, open_regions=[], operation_method=unknown, and view_state=side_or_back, occluded, or unknown. Never guess a handle or opening direction from a side view.
- For a usable front view of another container with a single visible door or lid, action must be open and open_regions should contain its visible handle or actionable panel center.
- Use unknown rather than guessing when the evidence is insufficient.
- Do not output joint names, object metadata, trajectories, forces, markdown, or extra keys."""


def visual_interaction_planning_context(
    *,
    object_id: str,
    object_name: str,
    expected_target_type: str,
    requested_action: str = "open",
    target_bbox_xyxy: list[int] | None = None,
    full_image_size: list[int] | None = None,
    approach_pose_label: str = "",
    alternate_view_count: int = 0,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "object_id": str(object_id),
        "object_name": str(object_name),
        "expected_target_type": str(expected_target_type or "unknown"),
        "requested_action": str(requested_action or "open"),
        "coordinate_frame": "normalized_target_crop",
        "image_inputs": ["full_headcam_with_target_outline_and_padded_target_inset"],
    }
    if target_bbox_xyxy:
        context["target_bbox_xyxy_in_full_image"] = [
            int(value) for value in target_bbox_xyxy[:4]
        ]
    if full_image_size:
        context["full_image_size_px"] = [int(value) for value in full_image_size[:2]]
    if approach_pose_label:
        context["approach_pose_label"] = str(approach_pose_label)
    if alternate_view_count > 0:
        context["alternate_view_count"] = max(0, int(alternate_view_count))
    return context
