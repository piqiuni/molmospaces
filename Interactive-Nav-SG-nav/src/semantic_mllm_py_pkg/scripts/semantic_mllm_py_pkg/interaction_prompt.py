from __future__ import annotations

from typing import Any


VISUAL_INTERACTION_PLANNING_INSTRUCTION = """Inspect the cropped RGB image of the selected interaction target after the robot has arrived in front of it.
Return one JSON object only, with this exact shape:
{"target_type":"door|drawer_container|other_container|unknown","action":"open|scan","operation_method":"hinged_push|hinged_pull|hinged_unknown|double_hinged|slide_left|slide_right|pull|unknown","open_regions":[{"center":[0.0,0.0],"confidence":0.0}],"confidence":0.0,"reason":"short reason"}

Rules:
- Coordinates are normalized to the cropped image: x=0 is left, y=0 is top, and both values must be in [0,1].
- For a door, describe the visible door mechanism in operation_method. Use hinged_unknown when push versus pull cannot be inferred from one image. Use double_hinged for a visible two-leaf hinged door. open_regions may contain visible handle or actionable panel centers.
- For a drawer_container, action must be scan, operation_method must be pull, and open_regions must contain the center of every independently openable visible drawer front or handle, ordered from top to bottom. Do not invent occluded drawers.
- For another container with a single visible door or lid, action must be open and open_regions should contain its visible handle or actionable panel center.
- Use unknown rather than guessing when the evidence is insufficient.
- Do not output joint names, object metadata, trajectories, forces, markdown, or extra keys."""


def visual_interaction_planning_context(
    *,
    object_id: str,
    object_name: str,
    expected_target_type: str,
    requested_action: str = "open",
) -> dict[str, Any]:
    return {
        "object_id": str(object_id),
        "object_name": str(object_name),
        "expected_target_type": str(expected_target_type or "unknown"),
        "requested_action": str(requested_action or "open"),
        "coordinate_frame": "normalized_target_crop",
    }
