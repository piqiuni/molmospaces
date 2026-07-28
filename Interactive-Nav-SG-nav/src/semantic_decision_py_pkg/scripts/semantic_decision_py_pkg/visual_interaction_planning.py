from __future__ import annotations

import math


_DRAWER_NAME_TOKENS = (
    "drawer",
    "dresser",
    "chestofdrawers",
    "chest_of_drawers",
    "chest of drawers",
)


def infer_visual_interaction_target_type(candidate: dict, node: dict) -> str:
    metadata = candidate.get("metadata") or {}
    attributes = node.get("attributes") or {}
    tokens = " ".join(
        str(value or "")
        for value in (
            candidate.get("interaction_type"),
            metadata.get("node_type"),
            node.get("type"),
            node.get("name"),
            node.get("label"),
            attributes.get("category"),
            attributes.get("semantic_name"),
            candidate.get("target_name"),
        )
    ).casefold()
    if "portal" in tokens or "door" in tokens or "gate" in tokens:
        return "door"
    if any(token in tokens for token in _DRAWER_NAME_TOKENS):
        return "drawer_container"
    if any(token in tokens for token in ("container", "cabinet", "fridge", "refrigerator")):
        return "other_container"
    return "unknown"


def candidate_with_visual_operation_plan(candidate: dict, plan: dict) -> dict:
    planned = dict(candidate)
    interaction = dict(planned.get("interaction_command") or {})
    interaction["visual_operation_plan"] = dict(plan)
    interaction["operation_method"] = str(plan.get("operation_method") or "unknown")
    interaction["open_regions"] = list(plan.get("open_regions") or [])
    for simulator_only_key in (
        "interaction_group_id",
        "interaction_groups",
        "joint_names",
        "close_other_joint_names",
        "close_other_joints",
        "part_ids",
        "open_fraction_threshold",
    ):
        interaction.pop(simulator_only_key, None)
    # This is deliberately attached only by ``candidate_with_direct_drawer_scan``.
    # A normal MLLM plan with empty regions must not gain evaluator-side access to
    # an ungrounded full-container scan merely because an old candidate carried a
    # box field.
    interaction.pop("drawer_container_bbox_2d", None)
    if str(plan.get("target_type") or "") == "drawer_container":
        interaction.update(
            {
                "action": "scan",
                "interaction_mode": "drawer_scan",
                "sequence_type": "drawer_scan",
            }
        )
    else:
        interaction["action"] = str(
            plan.get("action") or interaction.get("action") or "open"
        )
    planned["interaction_command"] = interaction
    return planned


def current_visible_bbox_2d(node: dict) -> list[float] | None:
    """Return a normalized current public image box, if one is available."""

    if not bool(node.get("is_currently_visible")):
        return None
    attributes = node.get("attributes") or {}
    raw_box = (
        attributes.get("projected_bbox_2d")
        or node.get("projected_bbox_2d")
        or attributes.get("bbox_2d")
        or node.get("bbox_2d")
    )
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw_box[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right - left < 1.0 or bottom - top < 1.0:
        return None
    return [left, top, right, bottom]


def candidate_with_direct_drawer_scan(
    candidate: dict,
    node: dict,
    *,
    capture_step: object,
) -> dict | None:
    """Build the V3 sealed drawer-scan request from a current public box.

    The navigation side does not infer joints or hidden drawer locations.  It
    only transmits the currently visible container box and its public capture
    step.  The evaluator is then responsible for matching that pair against
    its recent public frames and for executing the frozen private drawer skill.
    """

    box = current_visible_bbox_2d(node)
    if isinstance(capture_step, bool):
        return None
    try:
        public_capture_step = int(capture_step)
    except (TypeError, ValueError):
        return None
    if box is None or public_capture_step < 0:
        return None
    planned = candidate_with_visual_operation_plan(
        candidate,
        {
            "target_type": "drawer_container",
            "action": "scan",
            "operation_method": "pull",
            "open_regions": [],
        },
    )
    interaction = dict(planned.get("interaction_command") or {})
    interaction["drawer_container_bbox_2d"] = box
    interaction["drawer_container_capture_step"] = public_capture_step
    planned["interaction_command"] = interaction
    return planned


def action_for_opaque_open_contract(action: object, *, enabled: bool) -> str:
    """Project an action onto an evaluator's opaque-object ``open`` API.

    V3 keeps part and joint resolution private, so it accepts only
    ``open(opaque_object_id)``.  Normal ROS simulation retains its richer action
    vocabulary unless this explicit adapter is enabled.
    """

    normalized = str(action or "open").casefold()
    return "open" if enabled else normalized
