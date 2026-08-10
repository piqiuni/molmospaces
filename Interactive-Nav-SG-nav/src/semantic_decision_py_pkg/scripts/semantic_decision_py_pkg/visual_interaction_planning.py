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
        # ``drawer_scan`` is the evaluator/V3 macro: each visible drawer is
        # opened for an observation and then closed again.  Normal semantic
        # exploration needs a different postcondition: the selected drawer(s)
        # stay open so the graph can expose newly visible contents.  Keep the
        # distinction explicit in the public command instead of inferring it
        # from the action after the command reaches the bridge.
        sequence_type = str(
            plan.get("drawer_sequence_type") or plan.get("sequence_type") or "drawer_scan"
        ).strip().casefold()
        if sequence_type == "drawer_open":
            interaction.update(
                {
                    "action": "open",
                    "interaction_mode": "drawer_open",
                    "sequence_type": "drawer_open",
                }
            )
        else:
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


def _public_capture_step(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        capture_step = int(value)
    except (TypeError, ValueError):
        return None
    return capture_step if capture_step >= 0 else None


def current_visible_bbox_capture_step(node: dict) -> int | None:
    """Return the public frame that supplied a currently visible node box.

    A graph-level ``capture_step`` is insufficient by itself: an object can
    remain in the graph while its 2-D box is from an older observation.  The
    direct drawer-scan contract must bind the box to the exact public frame
    that observed the container.
    """

    if not bool(node.get("is_currently_visible")):
        return None
    attributes = node.get("attributes") or {}
    for key in (
        "last_observation_frame_index",
        "observation_capture_step",
        "frame_index",
        "capture_step",
    ):
        capture_step = _public_capture_step(attributes.get(key))
        if capture_step is not None:
            return capture_step
    return _public_capture_step(node.get("capture_step"))


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


def _normalized_visible_drawer_regions(value: object) -> list[dict]:
    """Normalize public M1 crop-relative regions without inventing drawers."""

    if not isinstance(value, list):
        return []
    regions: list[dict] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        center = raw.get("center")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        try:
            x, y = float(center[0]), float(center[1])
            confidence = float(raw.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(confidence)):
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        normalized = {
            "center": [round(x, 4), round(y, 4)],
            "confidence": max(0.0, min(1.0, confidence)),
        }
        if any(
            (normalized["center"][0] - item["center"][0]) ** 2
            + (normalized["center"][1] - item["center"][1]) ** 2
            < 0.0016
            for item in regions
        ):
            continue
        regions.append(normalized)
        if len(regions) >= 8:
            break
    return sorted(regions, key=lambda item: (item["center"][1], item["center"][0]))


def candidate_with_visual_drawer_scan(
    candidate: dict,
    *,
    drawer_bbox_2d: object,
    capture_step: object,
    action_regions: object,
    visual_plan: dict | None = None,
) -> dict | None:
    """Build a sealed drawer scan only from one fresh, front M1 observation.

    ``drawer_bbox_2d`` and ``capture_step`` are public detector evidence paired
    with the M1 image.  ``action_regions`` are crop-relative points the model
    actually saw.  Returning ``None`` on any missing part is intentional: no
    caller may turn a side/occluded/empty visual observation into a bridge call.
    """

    if not isinstance(drawer_bbox_2d, (list, tuple)) or len(drawer_bbox_2d) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in drawer_bbox_2d[:4])
        public_capture_step = int(capture_step)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if public_capture_step < 0 or right - left < 1.0 or bottom - top < 1.0:
        return None
    regions = _normalized_visible_drawer_regions(action_regions)
    if not regions:
        return None
    plan = {
        "target_type": "drawer_container",
        "action": "scan",
        "operation_method": "pull",
        "drawer_sequence_type": "drawer_scan",
        "open_regions": regions,
        "coordinate_frame": "normalized_target_crop",
        **dict(visual_plan or {}),
    }
    # The action contract cannot be overridden by a partial M1 patch.
    plan.update(
        {
            "target_type": "drawer_container",
            "action": "scan",
            "operation_method": "pull",
            "drawer_sequence_type": "drawer_scan",
            "open_regions": regions,
            "coordinate_frame": "normalized_target_crop",
        }
    )
    planned = candidate_with_visual_operation_plan(candidate, plan)
    interaction = dict(planned.get("interaction_command") or {})
    interaction["drawer_container_bbox_2d"] = [left, top, right, bottom]
    interaction["drawer_container_capture_step"] = public_capture_step
    interaction["visual_operation_plan"] = plan
    planned["interaction_command"] = interaction
    return planned


def candidate_with_visual_drawer_open(
    candidate: dict,
    *,
    drawer_bbox_2d: object,
    capture_step: object,
    action_regions: object,
    visual_plan: dict | None = None,
) -> dict | None:
    """Build a normal-exploration drawer-open command from fresh M1 evidence.

    This is intentionally separate from :func:`candidate_with_visual_drawer_scan`.
    The latter is a sealed benchmark macro whose observable contract is
    open/observe/close.  The normal interactive-exploration contract instead
    leaves the grounded drawer fronts open after the action, allowing a later
    RGB/graph update to reveal container contents.
    """

    if not isinstance(drawer_bbox_2d, (list, tuple)) or len(drawer_bbox_2d) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in drawer_bbox_2d[:4])
        public_capture_step = int(capture_step)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if public_capture_step < 0 or right - left < 1.0 or bottom - top < 1.0:
        return None
    regions = _normalized_visible_drawer_regions(action_regions)
    if not regions:
        return None
    plan = {
        "target_type": "drawer_container",
        "action": "open",
        "operation_method": "pull",
        "drawer_sequence_type": "drawer_open",
        "open_regions": regions,
        "coordinate_frame": "normalized_target_crop",
        **dict(visual_plan or {}),
    }
    # The public contract cannot be overridden by a partial M1 response.
    plan.update(
        {
            "target_type": "drawer_container",
            "action": "open",
            "operation_method": "pull",
            "drawer_sequence_type": "drawer_open",
            "open_regions": regions,
            "coordinate_frame": "normalized_target_crop",
        }
    )
    planned = candidate_with_visual_operation_plan(candidate, plan)
    interaction = dict(planned.get("interaction_command") or {})
    interaction["drawer_container_bbox_2d"] = [left, top, right, bottom]
    interaction["drawer_container_capture_step"] = public_capture_step
    interaction["visual_operation_plan"] = plan
    planned["interaction_command"] = interaction
    return planned


def fresh_direct_drawer_scan_candidate(
    candidate: dict,
    node: dict,
    *,
    graph_capture_step: object,
    graph_revision: object,
    minimum_graph_capture_step: object,
    minimum_graph_revision: object,
    rgb_image_sequence: object,
    minimum_rgb_image_sequence: object,
    rgb_capture_step: object,
    minimum_rgb_capture_step: object,
) -> tuple[dict | None, str]:
    """Build a scan request only from a post-arrival public RGB/GT frame.

    The executor snapshots the graph and RGB stream when the approach pose is
    reached.  This helper rejects every pre-arrival graph box, a box belonging
    to another capture step, and a graph update that did not have a fresh RGB
    observation.  Its returned candidate is therefore safe to send directly
    to the evaluator's ``bbox + capture_step`` drawer-scan route.
    """

    current_graph_capture_step = _public_capture_step(graph_capture_step)
    baseline_graph_capture_step = _public_capture_step(minimum_graph_capture_step)
    current_graph_revision = _public_capture_step(graph_revision)
    baseline_graph_revision = _public_capture_step(minimum_graph_revision)
    if current_graph_capture_step is None:
        return None, "missing_graph_capture_step"
    if baseline_graph_capture_step is not None:
        if current_graph_capture_step <= baseline_graph_capture_step:
            return None, "graph_capture_not_fresh"
    elif (
        current_graph_revision is None
        or baseline_graph_revision is None
        or current_graph_revision <= baseline_graph_revision
    ):
        # When the arrival snapshot lacked a capture step, a later graph
        # revision is the minimum evidence that this is not the old box.
        return None, "graph_revision_not_fresh"

    node_capture_step = current_visible_bbox_capture_step(node)
    if node_capture_step != current_graph_capture_step:
        return None, "target_not_observed_in_current_capture"

    try:
        current_rgb_image_sequence = int(rgb_image_sequence)
        baseline_rgb_image_sequence = int(minimum_rgb_image_sequence)
    except (TypeError, ValueError):
        return None, "missing_rgb_image_sequence"
    if current_rgb_image_sequence <= baseline_rgb_image_sequence:
        return None, "rgb_image_not_fresh"

    current_rgb_capture_step = _public_capture_step(rgb_capture_step)
    baseline_rgb_capture_step = _public_capture_step(minimum_rgb_capture_step)
    if baseline_rgb_capture_step is not None:
        if (
            current_rgb_capture_step is None
            or current_rgb_capture_step <= baseline_rgb_capture_step
        ):
            return None, "rgb_capture_not_fresh"
    if (
        current_rgb_capture_step is not None
        and current_rgb_capture_step < current_graph_capture_step
    ):
        return None, "rgb_precedes_gt_capture"

    planned = candidate_with_direct_drawer_scan(
        candidate,
        node,
        capture_step=current_graph_capture_step,
    )
    if planned is None:
        return None, "target_bbox_unavailable"
    return planned, "ready"


def action_for_opaque_open_contract(action: object, *, enabled: bool) -> str:
    """Project an action onto an evaluator's opaque-object ``open`` API.

    V3 keeps part and joint resolution private, so it accepts only
    ``open(opaque_object_id)``.  Normal ROS simulation retains its richer action
    vocabulary unless this explicit adapter is enabled.
    """

    normalized = str(action or "open").casefold()
    return "open" if enabled else normalized
