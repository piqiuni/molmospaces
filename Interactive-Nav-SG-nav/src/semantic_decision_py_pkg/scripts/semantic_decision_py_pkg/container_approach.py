"""Pure container viewpoint/action geometry; the executor owns retries and state."""

from __future__ import annotations

import math
import re
from typing import Any

from .approach_geometry import normalize_angle


def _goal_xyyaw_option(value: Any) -> tuple[float, float, float] | None:
    values = list(value or [])
    if len(values) < 2:
        return None
    try:
        return (
            float(values[0]),
            float(values[1]),
            float(values[2]) if len(values) > 2 else 0.0,
        )
    except (TypeError, ValueError):
        return None


def container_two_stage_staging_goal_options(
    candidate: dict[str, Any] | None,
) -> list[tuple[float, float, float]]:
    """Return the immutable outer M1 staging sequence for a container."""

    metadata = (candidate or {}).get("metadata") or {}
    if not bool(metadata.get("container_two_stage_approach", False)):
        return []
    options: list[tuple[float, float, float]] = []
    for raw in list(metadata.get("container_staging_goal_xyyaw_candidates") or []):
        option = _goal_xyyaw_option(raw)
        if option is not None:
            options.append(option)
    return options


def container_two_stage_m1_viewpoint_order(
    candidate: dict[str, Any] | None,
) -> list[int]:
    """Return valid anchor indices in the candidate's M1 evidence order.

    New geometry publishes a face-diverse order separately from ordinary
    navigation fallback order.  Legacy candidates remain deterministic by
    falling back to their immutable staging sequence.
    """

    staging_goals = container_two_stage_staging_goal_options(candidate)
    metadata = (candidate or {}).get("metadata") or {}
    raw_order = list(metadata.get("container_m1_viewpoint_order") or [])
    order: list[int] = []
    for raw_index in raw_order:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(staging_goals) and index not in order:
            order.append(index)
    order.extend(index for index in range(len(staging_goals)) if index not in order)
    return order


def container_two_stage_m1_preflight_batch_indices(
    candidate: dict[str, Any] | None,
    requested_index: int,
) -> list[int]:
    """Return all still-usable outer anchors for one make-plan batch.

    The requested canonical index remains first, but a negative make-plan result
    does not require another state-machine dispatch.  Remaining anchors follow
    the immutable, face-diverse M1 order.  Already sampled or already known
    unavailable indices are excluded without changing the canonical index used
    by capture/action mappings.
    """

    staging_goals = container_two_stage_staging_goal_options(candidate)
    if not staging_goals:
        return []
    metadata = (candidate or {}).get("metadata") or {}
    excluded: set[int] = _container_two_stage_redundant_view_indices(candidate)
    for key in (
        "container_m1_unavailable_staging_indices",
        "interaction_observation_viewpoint_staging_indices",
        "container_m1_rejected_face_staging_indices",
    ):
        for raw_index in list(metadata.get(key) or []):
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(staging_goals):
                excluded.add(index)
    try:
        requested = int(requested_index)
    except (TypeError, ValueError):
        requested = -1
    batch: list[int] = []
    if 0 <= requested < len(staging_goals) and requested not in excluded:
        batch.append(requested)
    for index in container_two_stage_m1_viewpoint_order(candidate):
        if index not in excluded and index not in batch:
            batch.append(index)
    return batch


_AABB_FAN_LABEL_RE = re.compile(
    r"aabb_fan_[^_]+_[^_]+_angle_(?P<angle>[+-]?[0-9.]+)_clearance_(?P<clearance>[0-9.]+)"
)


def container_two_stage_m1_anchor_priority(
    candidate: dict[str, Any] | None,
    staging_index: int,
) -> tuple[float, float, float, int]:
    """Rank reachable container anchors without changing canonical indices.

    Drawer AABB-fan anchors prefer the smallest surface clearance and then the
    smallest absolute angular offset.  The canonical index remains the final
    tie-breaker so capture/action mappings stay immutable.  Legacy container
    layouts retain their published order.
    """

    metadata = (candidate or {}).get("metadata") or {}
    labels = list(metadata.get("container_staging_pose_labels") or [])
    try:
        index = int(staging_index)
    except (TypeError, ValueError):
        index = 0
    label = str(labels[index]) if 0 <= index < len(labels) else ""
    robot_distances = list(
        metadata.get("container_anchor_robot_distance_m_by_staging_index") or []
    )
    try:
        robot_distance_m = (
            float(robot_distances[index])
            if 0 <= index < len(robot_distances)
            else float(index)
        )
    except (TypeError, ValueError):
        robot_distance_m = float(index)
    match = _AABB_FAN_LABEL_RE.fullmatch(label)
    if match is None:
        if label.startswith("aabb_face_"):
            return 0.0, 0.0, robot_distance_m, index
        return float(index), 0.0, robot_distance_m, index
    try:
        angle_deg = float(match.group("angle"))
        clearance_m = float(match.group("clearance"))
    except (TypeError, ValueError):
        return float(index), 0.0, robot_distance_m, index
    return clearance_m, abs(angle_deg), robot_distance_m, index


def container_two_stage_next_m1_viewpoint_index(
    candidate: dict[str, Any] | None,
    *,
    excluded_indices: set[int] | list[int] | tuple[int, ...] = (),
) -> int | None:
    """Return the next usable outer anchor in the explicit M1 view order.

    ``excluded_indices`` combines views that already produced a targeted M1
    response with anchors that navigation could not reach.  Keeping this small
    policy helper independent from the state machine lets executor-side
    navigation failures advance the same evidence plan instead of falling back
    to the legacy linear ring order.
    """

    excluded: set[int] = _container_two_stage_redundant_view_indices(candidate)
    for raw_index in excluded_indices:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index >= 0:
            excluded.add(index)
    for index in container_two_stage_m1_viewpoint_order(candidate):
        if index not in excluded:
            return index
    return None


def _container_two_stage_redundant_view_indices(candidate: dict | None) -> set[int]:
    """Exclude aliases of sampled camera poses, not just sampled anchor IDs."""

    metadata = (candidate or {}).get("metadata") or {}
    sampled = [
        pose for raw in metadata.get("container_m1_sampled_capture_poses_xyyaw", [])
        if (pose := _goal_xyyaw_option(raw)) is not None
    ]
    viewed = metadata.get("interaction_observation_viewpoint_staging_indices") or []
    for index in viewed:
        pose = container_two_stage_capture_goal_for_staging(candidate, index)
        if pose is not None:
            sampled.append(pose)
    excluded = set()
    for index in container_two_stage_m1_viewpoint_order(candidate):
        pose = container_two_stage_capture_goal_for_staging(candidate, index)
        if pose is not None and any(
            math.hypot(pose[0] - old[0], pose[1] - old[1]) < 0.25
            and abs(normalize_angle(pose[2] - old[2])) < 0.25
            for old in sampled
        ):
            excluded.add(index)
    return excluded


def container_two_stage_face_indices(
    candidate: dict[str, Any] | None,
    staging_index: int,
) -> list[int]:
    """Return all immutable anchors on the selected AABB cardinal face."""

    metadata = (candidate or {}).get("metadata") or {}
    labels = [str(value) for value in metadata.get("container_staging_pose_labels") or []]
    try:
        selected = labels[int(staging_index)]
    except (IndexError, TypeError, ValueError):
        return []
    match = re.match(r"aabb_(?:fan|face)_(pos_x|pos_y|neg_x|neg_y)(?:_|$)", selected)
    if match is None:
        return []
    face = match.group(1)
    pattern = re.compile(rf"aabb_(?:fan|face)_{re.escape(face)}(?:_|$)")
    return [index for index, label in enumerate(labels) if pattern.match(label)]


def container_two_stage_capture_goal_for_staging(
    candidate: dict[str, Any] | None,
    staging_goal_option_index: int,
) -> tuple[float, float, float] | None:
    """Return the M1 capture goal paired with one navigation anchor.

    New candidates can keep navigation-only outer anchors separate from the
    closer visual capture stance through
    ``container_m1_capture_goal_xyyaw_by_staging_index``.  Old recordings do
    not have that mapping, in which case their staging goal remains the capture
    goal for backward compatibility.
    """

    metadata = (candidate or {}).get("metadata") or {}
    if not bool(metadata.get("container_two_stage_approach", False)):
        return None
    try:
        index = int(staging_goal_option_index)
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None
    raw_capture_goals = metadata.get(
        "container_m1_capture_goal_xyyaw_by_staging_index"
    )
    if isinstance(raw_capture_goals, (list, tuple)) and raw_capture_goals:
        if index >= len(raw_capture_goals):
            return None
        return _goal_xyyaw_option(raw_capture_goals[index])
    staging_goals = container_two_stage_staging_goal_options(candidate)
    return staging_goals[index] if index < len(staging_goals) else None


def container_two_stage_action_goal_for_staging(
    candidate: dict[str, Any] | None,
    staging_goal_option_index: int,
) -> tuple[float, float, float] | None:
    """Look up the type-safe physical goal paired with one outer staging pose."""

    metadata = (candidate or {}).get("metadata") or {}
    if not bool(metadata.get("container_two_stage_approach", False)):
        return None
    try:
        index = int(staging_goal_option_index)
    except (TypeError, ValueError):
        return None
    action_goals = list(
        metadata.get("container_action_goal_xyyaw_by_staging_index") or []
    )
    if index < 0 or index >= len(action_goals):
        return None
    return _goal_xyyaw_option(action_goals[index])


def container_two_stage_action_goal_options_for_staging(
    candidate: dict[str, Any] | None,
    staging_goal_option_index: int,
) -> list[tuple[float, float, float]]:
    """Return bounded same-face physical goals for one accepted outer M1 view.

    New candidates carry a primary physical stance plus left/right tangent
    alternatives.  Older recordings and externally produced candidates contain
    only the primary mapping, so retain that as a fail-closed one-option
    fallback instead of silently treating a missing new field as an error.
    """

    metadata = (candidate or {}).get("metadata") or {}
    if not bool(metadata.get("container_two_stage_approach", False)):
        return []
    try:
        index = int(staging_goal_option_index)
    except (TypeError, ValueError):
        return []
    if index < 0:
        return []
    raw_by_staging = list(
        metadata.get("container_action_goal_xyyaw_options_by_staging_index") or []
    )
    raw_options = (
        list(raw_by_staging[index])
        if index < len(raw_by_staging) and isinstance(raw_by_staging[index], list)
        else []
    )
    options: list[tuple[float, float, float]] = []
    for raw in raw_options:
        option = _goal_xyyaw_option(raw)
        if option is None or any(
            math.hypot(option[0] - prior[0], option[1] - prior[1]) <= 1e-6
            and abs(normalize_angle(option[2] - prior[2])) <= 1e-6
            for prior in options
        ):
            continue
        options.append(option)
    primary = container_two_stage_action_goal_for_staging(candidate, index)
    if primary is None:
        return options
    if not options:
        return [primary]
    # The geometry producer promises primary first.  Be defensive when an
    # older or hand-authored candidate does not: physical traces and legacy
    # fields must still retain the primary at index zero.
    if not (
        math.hypot(options[0][0] - primary[0], options[0][1] - primary[1]) <= 1e-6
        and abs(normalize_angle(options[0][2] - primary[2])) <= 1e-6
    ):
        options = [primary] + [
            option
            for option in options
            if math.hypot(option[0] - primary[0], option[1] - primary[1]) > 1e-6
            or abs(normalize_angle(option[2] - primary[2])) > 1e-6
        ]
    return options


def container_two_stage_action_goal_options_from_m1_front_evidence(
    candidate: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
) -> tuple[list[tuple[float, float, float]], list[str], dict[str, Any]] | None:
    """Project safe physical poses from one accepted M1-confirmed front ray.

    M1's interface deliberately has no world-space normal: one RGB image
    cannot author map geometry.  Instead it confirms that the calibrated
    object-to-camera ray at its *actual capture pose* is a usable front.  This
    helper hides the surface-clearance projection behind that small contract so
    the active decision need not wait for a later graph regeneration to use the
    newly established face.

    ``None`` preserves legacy candidates that never published this contract.
    A new candidate that explicitly opts in but lacks valid frozen evidence is
    handled fail-closed by its caller rather than falling back to a graph/oracle
    front axis.
    """

    metadata = (candidate or {}).get("metadata") or {}
    if not bool(metadata.get("container_m1_front_axis_from_capture", False)):
        return None
    if not isinstance(evidence, dict):
        return None
    axis_values = list(evidence.get("m1_front_axis_xy") or [])
    anchor_values = list(metadata.get("container_geometry_anchor_xy") or [])
    size_values = list(metadata.get("container_geometry_aabb_size_xy") or [])
    if len(axis_values) < 2 or len(anchor_values) < 2 or len(size_values) < 2:
        return None
    try:
        axis_x = float(axis_values[0])
        axis_y = float(axis_values[1])
        anchor_x = float(anchor_values[0])
        anchor_y = float(anchor_values[1])
        half_x = 0.5 * abs(float(size_values[0]))
        half_y = 0.5 * abs(float(size_values[1]))
        physical_standoff_m = float(
            metadata.get("container_physical_action_standoff_m")
        )
        lateral_offset_m = max(
            0.0, float(metadata.get("container_action_lateral_offset_m", 0.0))
        )
    except (TypeError, ValueError):
        return None
    norm = math.hypot(axis_x, axis_y)
    if (
        norm <= 1e-6
        or half_x <= 1e-6
        or half_y <= 1e-6
        or not math.isfinite(physical_standoff_m)
        or physical_standoff_m < 0.0
    ):
        return None
    axis_x /= norm
    axis_y /= norm
    ray_denominator = abs(axis_x) / half_x + abs(axis_y) / half_y
    if ray_denominator <= 1e-6:
        return None
    boundary_distance = 1.0 / ray_denominator
    offset = boundary_distance + physical_standoff_m
    primary_x = anchor_x + axis_x * offset
    primary_y = anchor_y + axis_y * offset
    primary_yaw = math.atan2(anchor_y - primary_y, anchor_x - primary_x)
    options: list[tuple[float, float, float]] = [
        (primary_x, primary_y, primary_yaw)
    ]
    labels = ["m1_confirmed_front_physical_action"]
    if lateral_offset_m > 1e-6:
        tangent_x, tangent_y = -axis_y, axis_x
        for direction, label in (
            (1.0, "m1_confirmed_front_physical_action_tangent_left"),
            (-1.0, "m1_confirmed_front_physical_action_tangent_right"),
        ):
            x = primary_x + direction * lateral_offset_m * tangent_x
            y = primary_y + direction * lateral_offset_m * tangent_y
            options.append((x, y, math.atan2(anchor_y - y, anchor_x - x)))
            labels.append(label)
    front = {
        "m1_front_axis_xy": [axis_x, axis_y],
        "m1_front_yaw": primary_yaw,
        "m1_front_axis_source": str(
            evidence.get("m1_front_axis_source")
            or "m1_confirmed_capture_pose"
        ),
        "m1_front_capture_pose_xyyaw": list(
            evidence.get("capture_pose_xyyaw") or []
        ),
        "m1_front_capture_step": evidence.get("capture_step"),
    }
    return options, labels, front


def container_two_stage_action_pose_labels_for_staging(
    candidate: dict[str, Any] | None,
    staging_goal_option_index: int,
    option_count: int,
) -> list[str]:
    """Return index-aligned labels without making trace schema mandatory."""

    metadata = (candidate or {}).get("metadata") or {}
    try:
        staging_index = int(staging_goal_option_index)
    except (TypeError, ValueError):
        staging_index = 0
    raw_by_staging = list(
        metadata.get("container_action_pose_option_labels_by_staging_index") or []
    )
    raw_labels = (
        list(raw_by_staging[staging_index])
        if staging_index >= 0
        and staging_index < len(raw_by_staging)
        and isinstance(raw_by_staging[staging_index], list)
        else []
    )
    legacy_labels = list(
        metadata.get("container_action_pose_labels_by_staging_index") or []
    )
    primary_label = (
        str(legacy_labels[staging_index])
        if staging_index >= 0 and staging_index < len(legacy_labels)
        else f"staging_{staging_index}_physical_action"
    )
    labels: list[str] = []
    for option_index in range(max(0, int(option_count))):
        raw = raw_labels[option_index] if option_index < len(raw_labels) else ""
        labels.append(
            str(raw)
            if str(raw).strip()
            else primary_label
            if option_index == 0
            else f"{primary_label}_tangent_{option_index}"
        )
    return labels


def is_container_two_stage_physical_action(candidate: dict[str, Any] | None) -> bool:
    """Whether the active candidate is between M1 acceptance and the bridge."""

    metadata = (candidate or {}).get("metadata") or {}
    return bool(
        metadata.get("container_two_stage_approach", False)
        and str(metadata.get("container_two_stage_phase") or "").casefold()
        == "physical_action"
    )


def is_container_two_stage_m1_capture(candidate: dict[str, Any] | None) -> bool:
    """Whether a container is travelling to its direct M1 capture pose."""

    metadata = (candidate or {}).get("metadata") or {}
    return bool(
        metadata.get("container_two_stage_approach", False)
        and str(metadata.get("container_two_stage_phase") or "").casefold()
        == "m1_capture"
    )
