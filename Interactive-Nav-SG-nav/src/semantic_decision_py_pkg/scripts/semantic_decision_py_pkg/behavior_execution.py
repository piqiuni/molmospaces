from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Any

from .behavior_candidates import (
    BEHAVIOR_EXPLORE,
    BEHAVIOR_INTERACT,
    BEHAVIOR_NAVIGATE,
    BEHAVIOR_SCAN,
)


STATE_IDLE = "IDLE"
STATE_PREPARING_EXPLORE = "PREPARING_EXPLORE"
STATE_NAVIGATING = "NAVIGATING"
STATE_FINALIZING_EXPLORE = "FINALIZING_EXPLORE"
STATE_APPROACH_INTERACTION = "APPROACH_INTERACTION"
STATE_WAITING_FOR_DRAWER_SCAN = "WAIT_FOR_DRAWER_SCAN"
STATE_WAITING_FOR_INTERACTION_OBSERVATION = "WAITING_FOR_INTERACTION_OBSERVATION"
STATE_SCANNING = "SCANNING"
STATE_INTERACTING = "INTERACTING"
STATE_VERIFYING = "VERIFYING"
STATE_SUCCEEDED = "SUCCEEDED"
STATE_FAILED = "FAILED"


def is_static_portal_interaction_feedback(
    candidate: dict[str, Any] | None,
    detail: dict[str, Any] | None,
) -> bool:
    """Return whether an interaction result identifies a non-articulated portal.

    The force/evaluator bridge can establish that a portal is a fixed opening
    in the same callback that receives the interaction command.  Such a result
    is terminal feedback, not a request to enter graph verification.  Check all
    public result spellings because opaque backends may omit
    ``interaction_capability`` while still publishing ``static_open`` or the
    explicit executor source.
    """

    candidate = candidate or {}
    detail = detail or {}
    metadata = candidate.get("metadata") or {}
    command = candidate.get("interaction_command") or {}
    node_type = str(
        metadata.get("node_type")
        or command.get("node_type")
        or candidate.get("node_type")
        or ""
    ).strip().casefold()
    if node_type != "portal":
        return False
    capability = str(
        detail.get("interaction_capability")
        or detail.get("capability")
        or ""
    ).strip().casefold()
    state = str(
        detail.get("state")
        or detail.get("post_state")
        or ""
    ).strip().casefold()
    source = str(detail.get("source") or "").strip().casefold()
    return bool(
        capability == "static"
        or state in {"static", "static_open", "static_closed"}
        or source == "executor_static_portal"
    )


def is_interaction_pose_precondition_failure(detail: dict[str, Any] | None) -> bool:
    """Return whether no physical interaction was attempted because approach failed.

    This is deliberately separate from an object/action failure.  The bridge
    rejects such a command before applying force, so the planner must retry or
    switch approach poses rather than learn that the object itself is blocked.
    """

    detail = detail or {}
    failure_reason = str(
        detail.get("failure_reason") or detail.get("reason") or ""
    ).strip().casefold()
    verification_source = str(detail.get("verification_source") or "").strip().casefold()
    return bool(
        failure_reason
        in {
            "interaction_pose_invalid",
            "interaction_pose_poll_exhausted",
            "interaction_approach_options_exhausted",
            "interaction_wrong_face",
            # A refrigerator sweep is checked privately by the simulator before
            # force is applied.  Treat it like an approach precondition: retry
            # another safe public ring pose rather than mark the appliance
            # permanently non-interactable.
            "unsafe_open_sweep",
        }
        or verification_source == "executor_pose_precondition"
    )


def interaction_observation_disposition(
    detail: dict[str, Any] | None,
    *,
    drawer_pre_action: bool = False,
    container_pre_action: bool = False,
) -> str:
    """Classify a fresh M1 portal observation without issuing an action.

    The values are deliberately command-neutral.  The ROS executor owns image
    acquisition and M1 invocation; this pure function only maps its public
    result to the next state-machine branch.
    """

    detail = detail or {}
    attributes = detail.get("attributes")
    merged = dict(attributes) if isinstance(attributes, dict) else {}
    merged.update(detail)
    attribute_status = str(
        merged.get("attribute_status") or merged.get("mllm_status") or "ready"
    ).strip().casefold()
    visible = merged.get(
        "is_currently_visible",
        merged.get("currently_visible", merged.get("visible")),
    )
    if attribute_status != "ready" or visible is not True:
        return "retry"
    if drawer_pre_action:
        view_state = str(merged.get("view_state") or "unknown").strip().casefold()
        front_visible = merged.get("front_surface_visible") is True
        approach_ready = merged.get("approach_ready") is True
        regions_ready = merged.get("drawer_action_regions_ready") is True
        if not regions_ready:
            # The executor validates crop-relative centers and the paired
            # detector box before setting this flag.  The state machine still
            # requires the M1 frontality contract here so an empty or side-view
            # observation can only request another view, never a bridge call.
            return "retry"
        if (
            view_state not in {"front", "oblique"}
            or not front_visible
            or not approach_ready
            or bool(merged.get("needs_reobserve", False))
        ):
            return "retry"
        return "execute"

    if container_pre_action:
        # A container's remembered map/AABB is useful only to reach a
        # re-observation pose.  Opening is authorized by the *fresh* M1 view
        # at that pose, not by the ring geometry or an older crop.
        state = str(
            merged.get("state")
            or merged.get("interaction_state")
            or merged.get("coarse_state")
            or "unknown"
        ).strip().casefold()
        if state in {"open", "opened", "static_open", "static"}:
            return "finish_without_action"
        if state in {"blocked", "unavailable", "static_closed", "locked"}:
            # M1 is visual evidence, not an authoritative capability oracle.
            # A single "locked"/"unavailable" judgement is therefore only an
            # inconclusive view and has to consume the bounded evidence policy
            # below before the candidate can be deferred.
            return "retry"
        if state not in {"closed", "ajar"}:
            return "retry"
        view_state = str(merged.get("view_state") or "unknown").strip().casefold()
        if (
            view_state not in {"front", "oblique"}
            or merged.get("front_surface_visible") is not True
            or merged.get("approach_ready") is not True
            or bool(merged.get("needs_reobserve", False))
        ):
            return "retry"
        return "execute"

    state = str(
        merged.get("state")
        or merged.get("interaction_state")
        or merged.get("coarse_state")
        or merged.get("post_state")
        or "unknown"
    ).strip().casefold()
    if state in {"closed", "ajar"}:
        return "execute"
    if state in {"open", "opened", "static_open", "static"}:
        return "finish_without_action"
    if state in {"blocked", "unavailable", "static_closed", "locked"}:
        # See the container branch above: M1 must not one-shot terminalize a
        # portal or container merely from a semantic label in one image.
        return "retry"
    return "retry"


def interaction_pose_validation(
    expected_pose_xyyaw: list[Any] | tuple[Any, ...] | None,
    actual_pose_xyyaw: list[Any] | tuple[Any, ...] | None,
    *,
    distance_tolerance_m: float,
    yaw_tolerance_rad: float,
) -> dict[str, Any]:
    """Evaluate an approach pose using the same public contract as the bridge."""

    expected = list(expected_pose_xyyaw or [])
    actual = list(actual_pose_xyyaw or [])
    if len(expected) < 3 or len(actual) < 3:
        return {
            "checked": False,
            "valid": False,
            "reason": "interaction_pose_unavailable",
            "expected_pose_xyyaw": expected,
            "actual_pose_xyyaw": actual,
        }
    expected = [float(value) for value in expected[:3]]
    actual = [float(value) for value in actual[:3]]
    position_error_m = math.hypot(actual[0] - expected[0], actual[1] - expected[1])
    yaw_error_rad = abs(normalize_angle(actual[2] - expected[2]))
    distance_tolerance_m = max(0.05, float(distance_tolerance_m))
    yaw_tolerance_rad = max(0.05, float(yaw_tolerance_rad))
    return {
        "checked": True,
        "valid": bool(
            position_error_m <= distance_tolerance_m
            and yaw_error_rad <= yaw_tolerance_rad
        ),
        "expected_pose_xyyaw": expected,
        "actual_pose_xyyaw": actual,
        "position_error_m": position_error_m,
        "yaw_error_rad": yaw_error_rad,
        "distance_tolerance_m": distance_tolerance_m,
        "yaw_tolerance_rad": yaw_tolerance_rad,
    }


def candidate_with_effective_interaction_approach(
    candidate: dict[str, Any],
    approach_pose_xyyaw: list[Any] | tuple[Any, ...],
    *,
    goal_option_index: int,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind the selected navigable approach pose to an interaction command.

    ``goal_xyyaw`` remains the decision-time primary goal so candidate identity
    and its stable fallback ordering do not change.  The chosen executable
    pose is recorded separately and becomes the bridge precondition.
    """

    result = dict(candidate or {})
    approach = list(approach_pose_xyyaw or [])
    if len(approach) < 2:
        return result
    if len(approach) < 3:
        approach.append(0.0)
    approach = [float(value) for value in approach[:3]]
    interaction = dict(result.get("interaction_command") or {})
    metadata = dict(result.get("metadata") or {})
    planned_command_pose = list(interaction.get("interaction_approach_pose_xyyaw") or [])
    if planned_command_pose:
        interaction.setdefault(
            "planned_interaction_approach_pose_xyyaw", planned_command_pose
        )
    interaction["interaction_approach_pose_xyyaw"] = list(approach)
    metadata.setdefault("planned_goal_xyyaw", list(result.get("goal_xyyaw") or []))
    metadata["effective_interaction_approach_pose_xyyaw"] = list(approach)
    metadata["interaction_approach_goal_option_index"] = max(
        0, int(goal_option_index)
    )
    if attempts is not None:
        metadata["interaction_approach_attempts"] = [dict(item) for item in attempts]
    result["interaction_command"] = interaction
    result["metadata"] = metadata
    return result


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def committed_turn_sign(
    angular_error: float,
    pi_tie_tolerance_rad: float = 0.20,
    pi_tie_turn_sign: int = -1,
) -> int:
    error = normalize_angle(angular_error)
    if abs(abs(error) - math.pi) <= max(0.0, float(pi_tie_tolerance_rad)):
        return 1 if int(pi_tie_turn_sign) >= 0 else -1
    return 1 if error >= 0.0 else -1


def prerotation_control_step_budget(
    initial_error_rad: float,
    exit_tolerance_rad: float,
    speed_rad_s: float,
    control_dt_s: float,
    max_control_steps: int,
) -> int:
    """Bound a V3 pre-rotation by simulator control steps, not ROS publishes.

    A V3 RGB header sequence identifies the evaluator action which will
    consume the next fresh command.  Budget only the angular distance still
    outside the forward sector, so a worst-case pi-to-30-degree pre-turn needs
    eleven 200 ms control steps at the 1.25 rad/s navigation cap.
    """

    remaining_rad = max(
        0.0, abs(float(initial_error_rad)) - max(0.0, float(exit_tolerance_rad))
    )
    angular_step_rad = abs(float(speed_rad_s)) * max(0.0, float(control_dt_s))
    if remaining_rad <= 0.0 or angular_step_rad <= 0.0:
        return 0
    required_motion_steps = int(math.ceil(remaining_rad / angular_step_rad))
    return min(
        max(1, int(max_control_steps)),
        required_motion_steps,
    )


def prerotation_rgb_step_gate(
    *,
    last_sent_rgb_step_seq: int | None,
    current_rgb_step_seq: int | None,
    nonzero_commands_sent: int,
    max_control_steps: int,
) -> str:
    """Decide whether the V3 pre-rotation may send one new nonzero command.

    Repeated image republishes retain their header sequence and must never
    authorize another action.  A sequence reset is fail-closed because it may
    belong to a new evaluator episode.  The caller records a ``send`` by
    assigning ``current_rgb_step_seq`` to ``last_sent_rgb_step_seq``.
    """

    if current_rgb_step_seq is None:
        return "wait"
    if last_sent_rgb_step_seq is not None:
        if int(current_rgb_step_seq) < int(last_sent_rgb_step_seq):
            return "stop"
        if int(current_rgb_step_seq) == int(last_sent_rgb_step_seq):
            return "wait"
    if int(nonzero_commands_sent) >= max(0, int(max_control_steps)):
        return "stop"
    return "send"


def path_lookahead_point(
    start_xy: tuple[float, float],
    path_xy: list[tuple[float, float]],
    lookahead_m: float,
) -> tuple[float, float] | None:
    if not path_xy:
        return None
    previous = (float(start_xy[0]), float(start_xy[1]))
    traveled = 0.0
    target_distance = max(0.0, float(lookahead_m))
    for point in path_xy:
        current = (float(point[0]), float(point[1]))
        segment = math.hypot(current[0] - previous[0], current[1] - previous[1])
        traveled += segment
        if traveled >= target_distance and math.hypot(
            current[0] - start_xy[0], current[1] - start_xy[1]
        ) > 1e-3:
            return current
        previous = current
    endpoint = path_xy[-1]
    if math.hypot(endpoint[0] - start_xy[0], endpoint[1] - start_xy[1]) <= 1e-3:
        return None
    return float(endpoint[0]), float(endpoint[1])


def navigation_prerotation_heading_target(
    path_lookahead: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Return a pre-turn heading only when a reachable path supplied one.

    The final goal is deliberately not an input: a global path can initially
    leave in a direction very different from its endpoint bearing.
    """

    if path_lookahead is None:
        return None
    return float(path_lookahead[0]), float(path_lookahead[1])


def navigation_goal_options(candidate: dict[str, Any]) -> list[tuple[float, float, float]]:
    raw_options = [candidate.get("goal_xyyaw")]
    raw_options.extend(
        list((candidate.get("metadata") or {}).get("goal_xyyaw_candidates") or [])
    )
    options: list[tuple[float, float, float]] = []
    for raw_option in raw_options:
        values = list(raw_option or [])
        if len(values) < 2:
            continue
        option = (
            float(values[0]),
            float(values[1]),
            float(values[2]) if len(values) > 2 else 0.0,
        )
        if any(
            math.hypot(option[0] - previous[0], option[1] - previous[1]) <= 1e-6
            and abs(normalize_angle(option[2] - previous[2])) <= 1e-6
            for previous in options
        ):
            continue
        options.append(option)
    return options


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
    excluded: set[int] = set()
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

    excluded: set[int] = set()
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


def navigation_should_prerotate(behavior_type: str) -> bool:
    """Return whether a path-backed rear-goal pre-turn is permitted.

    Frontier execution used to skip this stage in an attempt to start moving
    sooner.  That hands a roughly-behind frontier straight to DWA, which can
    alternate its turn direction without translating.  The executor now uses
    the first segment of a verified global plan and a bounded control-step
    budget for *all* navigation behaviours, including EXPLORE.  This is still
    deliberately independent of final-yaw alignment: a frontier does not need
    to finish at its requested viewing yaw.
    """

    return str(behavior_type or "").upper() in {
        BEHAVIOR_EXPLORE,
        BEHAVIOR_INTERACT,
        BEHAVIOR_NAVIGATE,
    }


def navigation_requires_final_yaw(
    behavior_type: str,
    final_align_enabled: bool,
    primary_goal_values: list[Any],
) -> bool:
    """Frontier viewpoints need position reachability, not a strict terminal yaw."""

    return bool(
        final_align_enabled
        and str(behavior_type or "").upper() != BEHAVIOR_EXPLORE
        and len(primary_goal_values) > 2
    )


def is_post_interaction_traversal_navigation(
    candidate: dict[str, Any] | None,
) -> bool:
    """Scope stale-costmap retries to the sealed portal continuation only."""

    candidate = candidate or {}
    metadata = candidate.get("metadata") or {}
    return bool(
        str(candidate.get("behavior_type") or "").upper() == BEHAVIOR_NAVIGATE
        and metadata.get("post_interaction_traversal")
    )


@dataclass(frozen=True)
class PostInteractionCostmapBaseline:
    """Map publications observed before a successful portal continuation.

    The post-open gate must follow the actual planner input, not merely a
    convenient costmap notification: raw SLAM occupancy -> semantic planning
    occupancy -> global costmap.  Receipt counters are local to the executor
    rather than ROS header sequences, which can reset with move_base.
    """

    portal_id: str
    source_event_id: str
    receipt_count: int
    header_seq: int | None = None
    update_receipt_count: int = 0
    update_header_seq: int | None = None
    raw_occupancy_receipt_count: int = 0
    raw_occupancy_header_seq: int | None = None
    raw_occupancy_header_stamp_sec: float | None = None
    planning_occupancy_receipt_count: int = 0
    planning_occupancy_header_seq: int | None = None
    planning_occupancy_header_stamp_sec: float | None = None
    interaction_result_stamp_sec: float | None = None


@dataclass(frozen=True)
class PostInteractionRawMapBarrier:
    """One raw map receipt admitted after a portal-open result.

    ``planning_occupancy_receipt_count`` is captured in the raw-map callback,
    so a planning map already received before this raw map can never satisfy
    the next stage merely because the executor wakes late.
    """

    receipt_count: int
    header_seq: int | None
    header_stamp_sec: float | None
    planning_occupancy_receipt_count: int
    planning_occupancy_header_seq: int | None = None
    planning_occupancy_header_stamp_sec: float | None = None


@dataclass(frozen=True)
class PostInteractionPlanningMapBarrier:
    """One planning-map receipt after a qualifying raw map.

    The costmap counters are sampled at this receipt.  A later global full or
    incremental update is therefore causally downstream of the planner input
    accepted by this barrier.
    """

    raw_map: PostInteractionRawMapBarrier
    raw_fresh_source: str
    receipt_count: int
    header_seq: int | None
    header_stamp_sec: float | None
    planning_fresh_source: str
    costmap_receipt_count: int
    costmap_header_seq: int | None
    costmap_update_receipt_count: int
    costmap_update_header_seq: int | None


def _positive_finite_stamp(value: object) -> float | None:
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(stamp) or stamp <= 0.0:
        return None
    return stamp


def post_interaction_raw_occupancy_fresh_source(
    baseline: PostInteractionCostmapBaseline | None,
    current_receipt_count: int,
    current_header_stamp_sec: float | None,
) -> str:
    """Return how a raw OCC receipt proves it followed the open result.

    When both stamps are available, require the source map's stamp to be
    strictly newer than the evaluator result.  A zero/missing stamp falls back
    to the executor's post-result receipt boundary and is labelled explicitly
    for benchmark diagnostics.
    """

    if baseline is None or int(current_receipt_count) <= int(
        baseline.raw_occupancy_receipt_count
    ):
        return ""
    result_stamp = _positive_finite_stamp(baseline.interaction_result_stamp_sec)
    raw_stamp = _positive_finite_stamp(current_header_stamp_sec)
    if result_stamp is not None and raw_stamp is not None:
        return "header_stamp" if raw_stamp > result_stamp else ""
    if result_stamp is not None:
        return "receipt_after_result_no_raw_stamp"
    return "receipt_after_result"


def post_interaction_planning_occupancy_fresh_source(
    raw_barrier: PostInteractionRawMapBarrier | None,
    current_receipt_count: int,
    current_header_stamp_sec: float | None,
) -> str:
    """Return how a planning OCC receipt proves it follows the raw map.

    Semantic mapping preserves the raw occupancy stamp on
    ``planning_occ_map``.  If both stamps are present, require that source
    relationship.  Otherwise the raw callback's receipt snapshot still
    guarantees a strictly later local receipt.
    """

    if raw_barrier is None or int(current_receipt_count) <= int(
        raw_barrier.planning_occupancy_receipt_count
    ):
        return ""
    raw_stamp = _positive_finite_stamp(raw_barrier.header_stamp_sec)
    planning_stamp = _positive_finite_stamp(current_header_stamp_sec)
    if raw_stamp is not None and planning_stamp is not None:
        # Matching is normal because semantic_mapping copies the raw header;
        # a later stamp is also valid if a downstream map rebuild occurs.
        return "source_header_stamp" if planning_stamp >= raw_stamp else ""
    if raw_stamp is not None:
        return "receipt_after_raw_no_planning_stamp"
    return "receipt_after_raw"


def post_interaction_costmap_is_fresh(
    baseline: PostInteractionCostmapBaseline | None,
    current_receipt_count: int,
    current_update_receipt_count: int = 0,
) -> bool:
    """Return whether either global-map stream advanced after the open result."""

    return bool(
        post_interaction_costmap_fresh_source(
            baseline,
            current_receipt_count,
            current_update_receipt_count,
        )
    )


def post_interaction_costmap_fresh_source(
    baseline: PostInteractionCostmapBaseline | None,
    current_receipt_count: int,
    current_update_receipt_count: int = 0,
) -> str:
    """Identify the stream that made a post-open planner-map gate fresh.

    Prefer the costmap delta topic whenever both streams are new: it is the
    low-latency actual update path and avoids tying correctness to expensive
    full-grid publications.
    """

    if baseline is None:
        return ""
    return post_interaction_costmap_receipts_fresh_source(
        baseline.receipt_count,
        baseline.update_receipt_count,
        current_receipt_count,
        current_update_receipt_count,
    )


def post_interaction_costmap_receipts_fresh_source(
    baseline_receipt_count: int,
    baseline_update_receipt_count: int,
    current_receipt_count: int,
    current_update_receipt_count: int = 0,
) -> str:
    """Return a later costmap source from explicit receipt-counter bounds."""

    if int(current_update_receipt_count) > int(baseline_update_receipt_count):
        return "costmap_update"
    if int(current_receipt_count) > int(baseline_receipt_count):
        return "full"
    return ""


def post_interaction_costmap_baseline_keys(
    source_event_id: object,
    portal_id: object,
) -> tuple[str, ...]:
    """Return exact-event then portal fallback keys for a traversal barrier."""

    event_id = str(source_event_id or "").strip()
    normalized_portal_id = str(portal_id or "").strip()
    keys: list[str] = []
    if event_id:
        keys.append(f"event:{event_id}")
    if normalized_portal_id:
        keys.append(f"portal:{normalized_portal_id}")
    return tuple(keys)


def post_open_path_retryable_preflight_reason(reason: str) -> bool:
    """Return whether a post-open ``make_plan`` miss can heal with map updates.

    The target point can briefly be outside the newly rebuilt global costmap,
    which presents as either no path or a path whose endpoint remains short of
    the crossing waypoint.  During ROS/costmap startup the same wait can also
    briefly lack a transform or the make-plan service.  None of those outcomes
    invalidates the already successful portal interaction, so wait for a
    concrete path rather than fail-open into ``move_base``.
    """

    return str(reason or "").casefold() in {
        "empty_plan",
        "endpoint_mismatch",
        "pose_unavailable",
        "service_unavailable",
    }


def post_open_path_is_confirmed(plan_reachable: bool, reason: str) -> bool:
    """Require a real make-plan result for the one-shot door crossing.

    Normal navigation preserves its configurable fail-open behavior when the
    service is temporarily unavailable.  A post-open crossing is different:
    it is explicitly held until a path has been observed, so a fail-open
    ``True`` paired with ``pose_unavailable`` or ``service_unavailable`` is
    never sufficient.
    """

    return bool(plan_reachable) and str(reason or "").casefold() == "reachable"


def bounded_empty_plan_retry_delay(
    now_s: float,
    deadline_s: float,
    interval_s: float,
) -> float | None:
    """Return the next retry delay without sleeping beyond the retry window."""

    remaining_s = float(deadline_s) - float(now_s)
    if remaining_s <= 0.0:
        return None
    return min(max(0.0, float(interval_s)), remaining_s)


def requires_graph_verification(module3: str, selection: dict[str, Any] | None) -> bool:
    if str(module3 or "").casefold() == "rule_verified":
        return True
    selection = selection or {}
    metadata = selection.get("metadata") or {}
    return bool(
        str(selection.get("behavior_type") or "").upper() == BEHAVIOR_NAVIGATE
        and metadata.get("target_goal")
        and metadata.get("verify_target_visibility", True)
    )


def target_ready_for_graph_verification(selection: dict[str, Any] | None) -> bool:
    selection = selection or {}
    metadata = selection.get("metadata") or {}
    return bool(
        str(selection.get("behavior_type") or "").upper() == BEHAVIOR_NAVIGATE
        and metadata.get("target_goal")
        and metadata.get("verify_target_visibility", True)
        and metadata.get("target_visible_now")
        and metadata.get("target_reliably_observed")
        and not bool(metadata.get("target_navigation_required", True))
    )


def is_stuck_recovery_failure(detail: dict[str, Any]) -> bool:
    reason = str(detail.get("reason") or "").lower()
    status = str(detail.get("status") or "").lower()
    # Global preflight failures and a DWA failure to produce a local plan are
    # both useful evidence of *no progress* when they recur across different
    # subgoals.  Treating only the executor's own watchdog timeout as stuck
    # made the recovery path unreachable in practice: a sequence of empty
    # plans reset the counter before it could back the robot away from a wall.
    return bool(
        reason
        in {
            "navigation_stagnation",
            "make_plan_unreachable",
            "local_plan_unavailable",
            "local_planner_unavailable",
        }
        or "oscillat" in status
        or "failed to get a plan" in status
        or "failed to produce path" in status
    )


def next_interaction_approach_option_index(
    *,
    behavior_type: str,
    failure_detail: dict[str, Any],
    selected_option_index: int,
    attempted_navigation_count: int,
    max_navigation_attempts: int,
    goal_option_count: int,
) -> int | None:
    """Return one conservative INTERACT approach fallback, if available.

    A semantic interaction remains committed while its approach pose changes.
    A verified executor stagnation, an exhausted simulator-step pose poll, or
    a visual frontality gate is allowed to advance to another approach pose.
    A bounded number of actual navigation attempts prevents a large candidate
    list from consuming the whole episode.
    """

    if str(behavior_type or "").upper() != BEHAVIOR_INTERACT:
        return None
    if str(failure_detail.get("reason") or "").lower() not in {
        "navigation_stagnation",
        "interaction_pose_poll_exhausted",
        "interaction_pose_invalid",
        "unsafe_open_sweep",
        "visual_reposition_required",
        "navigation_timeout",
        "navigation_step_sync_stall",
        "navigation_terminal_failure",
        "final_yaw_alignment_failed",
    }:
        return None
    if int(attempted_navigation_count) >= max(1, int(max_navigation_attempts)):
        return None
    next_index = int(selected_option_index) + 1
    if next_index < 0 or next_index >= max(0, int(goal_option_count)):
        return None
    return next_index


def safe_grid_motion_distance(
    data: list[int] | tuple[int, ...],
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
    start_xyyaw: tuple[float, float, float],
    direction_sign: float,
    requested_distance_m: float,
    robot_radius_m: float,
    safety_margin_m: float,
    occupied_threshold: int = 50,
    unknown_is_blocked: bool = True,
) -> float:
    width = int(width)
    height = int(height)
    resolution = float(resolution)
    requested = max(0.0, float(requested_distance_m))
    if width <= 0 or height <= 0 or resolution <= 0.0 or requested <= 0.0:
        return 0.0
    if len(data) < width * height:
        return 0.0
    clearance = max(0.0, float(robot_radius_m) + float(safety_margin_m))
    footprint_cells = int(math.ceil(clearance / resolution))
    sample_step = max(0.02, 0.5 * resolution)
    direction = 1.0 if float(direction_sign) >= 0.0 else -1.0
    start_x, start_y, yaw = (float(value) for value in start_xyyaw)
    origin_x, origin_y = (float(value) for value in origin_xy)

    def footprint_is_clear(center_x: float, center_y: float) -> bool:
        center_col = int(math.floor((center_x - origin_x) / resolution))
        center_row = int(math.floor((center_y - origin_y) / resolution))
        for row in range(center_row - footprint_cells, center_row + footprint_cells + 1):
            for col in range(center_col - footprint_cells, center_col + footprint_cells + 1):
                cell_x = origin_x + (col + 0.5) * resolution
                cell_y = origin_y + (row + 0.5) * resolution
                if math.hypot(cell_x - center_x, cell_y - center_y) > clearance:
                    continue
                if col < 0 or row < 0 or col >= width or row >= height:
                    return False
                value = int(data[row * width + col])
                if value >= int(occupied_threshold) or (unknown_is_blocked and value < 0):
                    return False
        return True

    safe_distance = 0.0
    distance = min(sample_step, requested)
    while distance <= requested + 1e-9:
        center_x = start_x + direction * distance * math.cos(yaw)
        center_y = start_y + direction * distance * math.sin(yaw)
        if not footprint_is_clear(center_x, center_y):
            break
        safe_distance = min(distance, requested)
        if safe_distance >= requested:
            break
        distance = min(requested, distance + sample_step)
    return safe_distance


@dataclass
class NavigationProgressWatchdog:
    timeout_s: float = 12.0
    min_displacement_m: float = 0.10
    min_yaw_change_rad: float = 0.15
    min_goal_distance_reduction_m: float = 0.02
    # A deliberate rear-goal rotation is handled before move_base with its own
    # finite action budget.  During ordinary navigation, yaw-only movement is
    # not progress: otherwise a DWA left/right oscillation can reset this
    # watchdog forever while the robot stays in place.  Keep the legacy
    # default for callers that explicitly use a rotation-only phase.
    allow_yaw_progress: bool = True
    # When supplied, the evaluator's public step clock is authoritative.  A
    # wall-clock timeout is unreliable while VLM calls or several simulators
    # share a host, and used to abort valid in-place turns prematurely.
    timeout_task_steps: int | None = None
    reference_xy: tuple[float, float] | None = None
    reference_yaw: float | None = None
    reference_goal_distance_m: float | None = None
    last_progress_at: float | None = None
    reference_step_index: int | None = None

    def reset(
        self,
        pose: tuple[float, ...] | None,
        now: float,
        goal_distance_m: float | None = None,
        task_step_index: int | None = None,
    ) -> None:
        self.reference_xy = None if pose is None else (float(pose[0]), float(pose[1]))
        self.reference_yaw = (
            float(pose[2]) if pose is not None and len(pose) >= 3 else None
        )
        self.reference_goal_distance_m = (
            float(goal_distance_m)
            if goal_distance_m is not None and math.isfinite(float(goal_distance_m))
            else None
        )
        self.last_progress_at = float(now) if pose is not None else None
        self.reference_step_index = (
            int(task_step_index) if task_step_index is not None else None
        )

    def observe(
        self,
        pose: tuple[float, ...] | None,
        now: float,
        *,
        goal_distance_m: float | None = None,
        local_plan_fresh: bool = False,
        task_step_index: int | None = None,
    ) -> bool:
        if self.timeout_s <= 0.0 or pose is None:
            return False
        if self.reference_xy is None or self.last_progress_at is None:
            self.reset(pose, now, goal_distance_m, task_step_index)
            return False
        displacement = math.hypot(
            float(pose[0]) - float(self.reference_xy[0]),
            float(pose[1]) - float(self.reference_xy[1]),
        )
        yaw_change = 0.0
        if len(pose) >= 3 and self.reference_yaw is not None:
            yaw_change = abs(normalize_angle(float(pose[2]) - self.reference_yaw))
        if displacement >= self.min_displacement_m or (
            self.allow_yaw_progress and yaw_change >= self.min_yaw_change_rad
        ):
            self.reset(pose, now, goal_distance_m, task_step_index)
            return False
        if (
            local_plan_fresh
            and goal_distance_m is not None
            and math.isfinite(float(goal_distance_m))
        ):
            current_goal_distance_m = float(goal_distance_m)
            if self.reference_goal_distance_m is None:
                self.reference_goal_distance_m = current_goal_distance_m
            elif (
                self.reference_goal_distance_m - current_goal_distance_m
                >= max(0.0, float(self.min_goal_distance_reduction_m))
            ):
                # A fresh local trajectory plus a material reduction in
                # distance-to-goal is real progress even when DWA deliberately
                # drives below the coarse pose-displacement threshold.
                self.reset(pose, now, current_goal_distance_m, task_step_index)
                return False
        if (
            self.timeout_task_steps is not None
            and task_step_index is not None
            and self.reference_step_index is not None
        ):
            return (
                int(task_step_index) - int(self.reference_step_index)
                >= max(1, int(self.timeout_task_steps))
            )
        return float(now) - self.last_progress_at >= self.timeout_s

    def goal_distance_reduction_m(self, goal_distance_m: float | None) -> float:
        if (
            goal_distance_m is None
            or self.reference_goal_distance_m is None
            or not math.isfinite(float(goal_distance_m))
        ):
            return 0.0
        return float(self.reference_goal_distance_m) - float(goal_distance_m)


@dataclass
class SemanticNavigationProgressSupervisor:
    """Keep no-progress memory across private move_base worker replacements.

    A corridor waypoint or actionlib retry is an implementation detail, not a
    new semantic subgoal.  This supervisor deliberately lives above those
    workers: the current anchor gets one finite evaluator-step budget, while a
    second mission-level budget survives anchor/candidate changes until the
    robot makes material translational progress.
    """

    subgoal_timeout_task_steps: int = 60
    mission_timeout_task_steps: int = 180
    min_displacement_m: float = 0.10
    min_goal_distance_reduction_m: float = 0.02
    min_yaw_error_reduction_rad: float = 0.02
    subgoal_key: str = ""
    subgoal_reference_xy: tuple[float, float] | None = None
    subgoal_reference_goal_distance_m: float | None = None
    subgoal_reference_yaw_error_rad: float | None = None
    subgoal_reference_step_index: int | None = None
    mission_reference_xy: tuple[float, float] | None = None
    mission_reference_step_index: int | None = None

    @staticmethod
    def _finite(value: float | None) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    def reset(self) -> None:
        self.subgoal_key = ""
        self.subgoal_reference_xy = None
        self.subgoal_reference_goal_distance_m = None
        self.subgoal_reference_yaw_error_rad = None
        self.subgoal_reference_step_index = None
        self.mission_reference_xy = None
        self.mission_reference_step_index = None

    def note_success(
        self, pose: tuple[float, ...] | None, task_step_index: int | None
    ) -> None:
        if pose is None or task_step_index is None:
            self.reset()
            return
        xy = (float(pose[0]), float(pose[1]))
        step = int(task_step_index)
        self.subgoal_key = ""
        self.subgoal_reference_xy = None
        self.subgoal_reference_goal_distance_m = None
        self.subgoal_reference_yaw_error_rad = None
        self.subgoal_reference_step_index = None
        self.mission_reference_xy = xy
        self.mission_reference_step_index = step

    def observe(
        self,
        *,
        subgoal_key: str,
        pose: tuple[float, ...] | None,
        task_step_index: int | None,
        goal_distance_m: float | None = None,
        yaw_error_rad: float | None = None,
        allow_yaw_progress: bool = False,
    ) -> dict:
        if pose is None or task_step_index is None or not str(subgoal_key):
            return {"subgoal_stalled": False, "mission_stalled": False}
        xy = (float(pose[0]), float(pose[1]))
        step = int(task_step_index)
        goal_distance = self._finite(goal_distance_m)
        yaw_error = self._finite(yaw_error_rad)

        if self.mission_reference_xy is None or self.mission_reference_step_index is None:
            self.mission_reference_xy = xy
            self.mission_reference_step_index = step
        mission_displacement = math.hypot(
            xy[0] - self.mission_reference_xy[0],
            xy[1] - self.mission_reference_xy[1],
        )
        if mission_displacement >= max(0.0, float(self.min_displacement_m)):
            self.mission_reference_xy = xy
            self.mission_reference_step_index = step

        if str(subgoal_key) != self.subgoal_key:
            self.subgoal_key = str(subgoal_key)
            self.subgoal_reference_xy = xy
            self.subgoal_reference_goal_distance_m = goal_distance
            self.subgoal_reference_yaw_error_rad = yaw_error
            self.subgoal_reference_step_index = step
        else:
            displacement = math.hypot(
                xy[0] - float(self.subgoal_reference_xy[0]),
                xy[1] - float(self.subgoal_reference_xy[1]),
            ) if self.subgoal_reference_xy is not None else 0.0
            distance_progress = bool(
                goal_distance is not None
                and self.subgoal_reference_goal_distance_m is not None
                and self.subgoal_reference_goal_distance_m - goal_distance
                >= max(0.0, float(self.min_goal_distance_reduction_m))
            )
            yaw_progress = bool(
                allow_yaw_progress
                and yaw_error is not None
                and self.subgoal_reference_yaw_error_rad is not None
                and self.subgoal_reference_yaw_error_rad - yaw_error
                >= max(0.0, float(self.min_yaw_error_reduction_rad))
            )
            if (
                displacement >= max(0.0, float(self.min_displacement_m))
                or distance_progress
                or yaw_progress
            ):
                self.subgoal_reference_xy = xy
                self.subgoal_reference_goal_distance_m = goal_distance
                self.subgoal_reference_yaw_error_rad = yaw_error
                self.subgoal_reference_step_index = step

        subgoal_elapsed = max(
            0,
            step
            - int(
                self.subgoal_reference_step_index
                if self.subgoal_reference_step_index is not None
                else step
            ),
        )
        mission_elapsed = max(
            0,
            step
            - int(
                self.mission_reference_step_index
                if self.mission_reference_step_index is not None
                else step
            ),
        )
        return {
            "subgoal_stalled": subgoal_elapsed
            >= max(1, int(self.subgoal_timeout_task_steps)),
            "mission_stalled": mission_elapsed
            >= max(1, int(self.mission_timeout_task_steps)),
            "subgoal_key": self.subgoal_key,
            "subgoal_elapsed_task_steps": subgoal_elapsed,
            "mission_elapsed_task_steps": mission_elapsed,
            "subgoal_timeout_task_steps": max(
                1, int(self.subgoal_timeout_task_steps)
            ),
            "mission_timeout_task_steps": max(
                1, int(self.mission_timeout_task_steps)
            ),
        }


@dataclass
class ExecutionConfig:
    navigation_timeout_s: float = 180.0
    interaction_navigation_timeout_s: float = 180.0
    interaction_timeout_s: float = 30.0
    drawer_scan_wait_timeout_s: float = 8.0
    interaction_observation_timeout_s: float = 8.0
    # A negative M1 result is treated as an inconclusive image.  By default
    # collect one fresh confirmation at the held pose before changing view.
    interaction_observation_same_pose_samples_per_view: int = 2
    # Zero retains each candidate's legacy ``interaction_observation_max_attempts``
    # as its viewpoint budget.  A positive value is an explicit cap on distinct
    # capture viewpoints, separate from the total request budget.
    interaction_observation_max_viewpoints: int = 0
    # Zero derives a finite total budget from viewpoints x samples-per-view.
    interaction_observation_max_total_requests: int = 0
    # A second planned M1 capture pose is a distinct visual experiment only
    # when navigation actually reaches it.  This is deliberately tighter than
    # the ordinary capture/physical-ready envelope, which remains useful for
    # the first capture and the physical bridge contract.
    container_m1_distinct_view_arrival_tolerance_m: float = 0.05
    container_m1_distinct_view_arrival_yaw_tolerance_rad: float = 0.08
    verification_timeout_s: float = 30.0
    explore_prepare_timeout_s: float = 10.0
    explore_finalize_timeout_s: float = 10.0
    scan_timeout_s: float = 15.0


class BehaviorExecutionStateMachine:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()
        self.reset()

    def reset(self) -> None:
        self.state = STATE_IDLE
        self.candidate: dict[str, Any] | None = None
        self.started_at = 0.0
        self.state_started_at = 0.0
        self.error = ""

    def start(self, candidate: dict[str, Any], now: float | None = None) -> list[dict[str, Any]]:
        if self.state != STATE_IDLE:
            raise RuntimeError(f"Executor is busy in state {self.state}")
        now = time.monotonic() if now is None else float(now)
        self.candidate = dict(candidate)
        self.started_at = now
        behavior_type = str(candidate.get("behavior_type") or "")
        if behavior_type == BEHAVIOR_EXPLORE:
            return self._transition(
                STATE_PREPARING_EXPLORE,
                now,
                {"kind": "reserve_frontier", "candidate": self.candidate},
            )
        if behavior_type == BEHAVIOR_NAVIGATE:
            if target_ready_for_graph_verification(candidate):
                return self._transition(STATE_VERIFYING, now)
            return self._transition(
                STATE_NAVIGATING,
                now,
                {"kind": "navigate", "candidate": self.candidate},
            )
        if behavior_type == BEHAVIOR_INTERACT:
            requires_approach = bool((candidate.get("metadata") or {}).get("requires_approach", True))
            if requires_approach:
                return self._transition(
                    STATE_APPROACH_INTERACTION,
                    now,
                    {"kind": "navigate", "candidate": self.candidate},
                )
            if self._interaction_requires_observation():
                return self._request_interaction_observation({}, now)
            return self._transition(
                STATE_INTERACTING,
                now,
                {"kind": "interact", "candidate": self.candidate},
            )
        if behavior_type == BEHAVIOR_SCAN:
            return self._transition(
                STATE_SCANNING,
                now,
                {"kind": "scan", "candidate": self.candidate},
            )
        raise ValueError(f"Unsupported behavior type: {behavior_type}")

    def on_scan_result(
        self,
        success: bool,
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.state != STATE_SCANNING or self._behavior_type() != BEHAVIOR_SCAN:
            return []
        return self._finish(bool(success), detail or {}, now)

    def on_explore_ready(
        self, detail: dict[str, Any] | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        if self.state != STATE_PREPARING_EXPLORE or self._behavior_type() != BEHAVIOR_EXPLORE:
            return []
        detail = detail or {}
        goal_values = list(detail.get("goal_xyyaw") or detail.get("goal") or [])
        if len(goal_values) < 2 or self.candidate is None:
            return self._finish(False, {"reason": "explore_ready_missing_goal", **detail}, now)
        yaw = float(goal_values[2]) if len(goal_values) > 2 else 0.0
        self.candidate["goal_xyyaw"] = [
            float(goal_values[0]),
            float(goal_values[1]),
            yaw,
        ]
        metadata = dict(self.candidate.get("metadata") or {})
        if detail.get("frame_id"):
            metadata["frame_id"] = str(detail["frame_id"])
        self.candidate["metadata"] = metadata
        now = time.monotonic() if now is None else float(now)
        return self._transition(
            STATE_NAVIGATING,
            now,
            {"kind": "navigate", "candidate": self.candidate},
        )

    def on_explore_result(
        self, success: bool, detail: dict[str, Any] | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        if self._behavior_type() != BEHAVIOR_EXPLORE:
            return []
        if self.state not in {STATE_PREPARING_EXPLORE, STATE_FINALIZING_EXPLORE}:
            return []
        return self._finish(success, detail or {}, now)

    def on_navigation_result(
        self,
        success: bool,
        detail: dict[str, Any] | None = None,
        now: float | None = None,
        *,
        wait_for_drawer_scan: bool = False,
    ) -> list[dict[str, Any]]:
        now = time.monotonic() if now is None else float(now)
        if self.state == STATE_NAVIGATING and self._behavior_type() == BEHAVIOR_EXPLORE:
            return self._transition(
                STATE_FINALIZING_EXPLORE,
                now,
                {
                    "kind": "finalize_frontier",
                    "candidate": self.candidate,
                    "success": bool(success),
                    "detail": detail or {},
                },
            )
        if self.state == STATE_NAVIGATING and self._behavior_type() == BEHAVIOR_NAVIGATE:
            if success and bool(
                (self.candidate.get("metadata") or {}).get(
                    "verify_target_visibility", False
                )
            ):
                return self._transition(STATE_VERIFYING, now)
            return self._finish(success, detail or {}, now)
        if self.state != STATE_APPROACH_INTERACTION:
            return []
        if not success:
            return self._finish(False, detail or {}, now)
        metadata = (self.candidate or {}).get("metadata") or {}
        if self._container_two_stage_staging_active(metadata):
            # A recovery anchor is intentionally farther than the visual
            # capture point.  Reaching that anchor never authorizes M1: move
            # to the paired direct capture pose first.  Legacy candidates have
            # no distinct mapping, so the helper returns no transition and the
            # old same-pose M1 behaviour remains intact.
            capture_navigation = self._begin_container_two_stage_m1_capture(
                detail or {}, now
            )
            if capture_navigation:
                return capture_navigation
        if self._interaction_requires_observation():
            return self._request_interaction_observation(detail or {}, now)
        if wait_for_drawer_scan:
            return self._transition(
                STATE_WAITING_FOR_DRAWER_SCAN,
                now,
                {"kind": "wait_for_drawer_scan", "candidate": self.candidate},
            )
        return self._transition(
            STATE_INTERACTING,
            now,
            {"kind": "interact", "candidate": self.candidate},
        )

    def on_interaction_observation_result(
        self,
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Consume one fresh M1 re-observation before a physical interaction.

        This method is intentionally ROS-free.  The executor responds to the
        ``request_interaction_observation`` command, invokes M1 on a fresh
        current frame, and calls this method with its public attributes.
        """

        if (
            self.state != STATE_WAITING_FOR_INTERACTION_OBSERVATION
            or self.candidate is None
        ):
            return []
        now = time.monotonic() if now is None else float(now)
        observation = self._normalized_interaction_observation(detail)
        metadata = dict(self.candidate.get("metadata") or {})
        metadata["last_interaction_observation"] = dict(observation)
        metadata["interaction_observation_viewpoint_pending"] = False
        self.candidate["metadata"] = metadata

        drawer_pre_action = bool(metadata.get("drawer_pre_action_observation"))
        container_pre_action = bool(
            metadata.get("container_pre_action_observation")
        )
        disposition = interaction_observation_disposition(
            observation,
            drawer_pre_action=drawer_pre_action,
            container_pre_action=container_pre_action,
        )
        required_source = str(
            metadata.get("interaction_observation_source") or ""
        ).strip().casefold()
        observation_source = str(
            observation.get("attribute_source")
            or observation.get("state_source")
            or ""
        ).strip().casefold()
        if required_source and required_source not in observation_source:
            observation.setdefault("reason", "interaction_observation_wrong_source")
            disposition = "retry"
        minimum_capture_step = metadata.get("interaction_observation_min_capture_step")
        capture_step = self._interaction_observation_capture_step(observation)
        if minimum_capture_step is not None and (
            capture_step is None or capture_step < int(minimum_capture_step)
        ):
            observation.setdefault("reason", "interaction_observation_not_fresh")
            disposition = "retry"
        if (
            container_pre_action
            and str(
                observation.get("container_visual_precondition_reason") or "ready"
            ).casefold()
            != "ready"
        ):
            # The executor has already rejected a stale/truncated/non-front
            # targeted M1 update.  In particular, do not let an untrusted
            # `open` label bypass the front-surface contract.
            observation.setdefault(
                "reason",
                str(observation.get("container_visual_precondition_reason")),
            )
            disposition = "retry"

        if disposition == "execute":
            if (
                self._container_two_stage_staging_active(metadata)
                or self._container_two_stage_m1_capture_active(metadata)
            ):
                action_navigation = self._begin_container_two_stage_action_approach(
                    observation,
                    now,
                )
                if action_navigation:
                    return action_navigation
                # A ready-looking M1 payload is not sufficient by itself for a
                # two-stage container.  The executor must have bound it to the
                # currently selected outer pose before a closer physical goal is
                # allowed.  Treat a missing/mismatched binding as another outer
                # observation failure, never as an inner action authorization.
                observation.setdefault(
                    "reason", "container_m1_evidence_not_bound_to_staging_pose"
                )
                disposition = "retry"
            if disposition == "execute":
                metadata["observation_required"] = False
                metadata["reobserve"] = False
                metadata["interaction_observation_resolved"] = True
                self.candidate["metadata"] = metadata
                return self._transition(
                    STATE_INTERACTING,
                    now,
                    {
                        "kind": "interact",
                        "candidate": self.candidate,
                        "observation": observation,
                    },
                )
        if disposition == "finish_without_action":
            return self._finish(
                True,
                {
                    **observation,
                    "action_executed": False,
                    "observation_outcome": "finish_without_action",
                    "reason": "interaction_not_required_after_observation",
                },
                now,
            )
        if disposition == "terminal":
            return self._finish(
                False,
                {
                    **observation,
                    "action_executed": False,
                    "observation_outcome": "terminal",
                    "reason": str(
                        observation.get("reason")
                        or "interaction_unavailable_after_observation"
                    ),
                },
                now,
            )

        attempts = int(metadata.get("interaction_observation_attempts", 0) or 0)
        samples_at_viewpoint = int(
            metadata.get("interaction_observation_samples_at_viewpoint", 0) or 0
        )
        viewpoints = int(
            metadata.get("interaction_observation_viewpoint_count", 0) or 0
        )
        max_samples_per_view = self._interaction_observation_same_pose_samples_per_view()
        max_viewpoints = self._interaction_observation_max_viewpoints()
        max_total_requests = self._interaction_observation_max_total_requests()
        if drawer_pre_action or container_pre_action:
            # A single M1 negative is only an inconclusive image, not evidence
            # that the container is impossible to interact with.  First obtain
            # one independently fresh frame while holding the exact capture
            # pose.  Only then spend a distinct navigation anchor/viewpoint.
            if (
                attempts < max_total_requests
                and samples_at_viewpoint < max_samples_per_view
            ):
                return self._request_interaction_observation(observation, now)
            if attempts < max_total_requests and viewpoints < max_viewpoints:
                return self._advance_interaction_reobservation_approach(
                    observation,
                    now,
                    drawer_pre_action=drawer_pre_action,
                )
            precondition_kind = "drawer" if drawer_pre_action else "container"
            return self._defer_container_m1_evidence(
                observation,
                now,
                drawer_pre_action=drawer_pre_action,
                reason=f"{precondition_kind}_m1_evidence_inconclusive",
            )
        if attempts < self._interaction_observation_max_attempts():
            return self._request_interaction_observation(observation, now)
        return self._finish(
            False,
            {
                **observation,
                "action_executed": False,
                "observation_outcome": "terminal_unresolved",
                "observation_attempts": attempts,
                "reason": "interaction_observation_unresolved",
            },
            now,
        )

    def _interaction_requires_observation(self) -> bool:
        metadata = (self.candidate or {}).get("metadata") or {}
        return bool(
            self._behavior_type() == BEHAVIOR_INTERACT
            and metadata.get("observation_required", False)
        )

    @staticmethod
    def _container_two_stage_staging_active(metadata: dict[str, Any]) -> bool:
        return bool(
            metadata.get("container_two_stage_approach", False)
            and str(metadata.get("container_two_stage_phase") or "staging").casefold()
            == "staging"
        )

    @staticmethod
    def _container_two_stage_m1_capture_active(metadata: dict[str, Any]) -> bool:
        return bool(
            metadata.get("container_two_stage_approach", False)
            and str(metadata.get("container_two_stage_phase") or "").casefold()
            == "m1_capture"
        )

    def _container_m1_capture_requires_distinct_view_arrival(
        self,
        metadata: dict[str, Any],
        capture_goal: tuple[float, float, float],
    ) -> tuple[bool, tuple[float, float, float] | None]:
        """Return whether a mapped capture must reach a genuinely new view.

        The outer-anchor-to-capture contract can deliberately accept a broad
        first arrival.  Once M1 has sampled one mapped capture pose, however,
        a later capture target that is materially different must not reuse that
        broad envelope: doing so would count the same camera pose twice.  The
        previous value is the pose that owned an actual M1 request, not merely
        a navigation anchor that happened to be visited.
        """

        previous_goal = _goal_xyyaw_option(
            metadata.get("container_m1_last_sampled_capture_goal_xyyaw")
        )
        if previous_goal is None:
            return False, None
        try:
            distance_tolerance_m = float(
                self.config.container_m1_distinct_view_arrival_tolerance_m
            )
            yaw_tolerance_rad = float(
                self.config.container_m1_distinct_view_arrival_yaw_tolerance_rad
            )
        except (TypeError, ValueError):
            return False, previous_goal
        if distance_tolerance_m <= 0.0 or yaw_tolerance_rad <= 0.0:
            return False, previous_goal
        is_distinct = bool(
            math.hypot(
                capture_goal[0] - previous_goal[0],
                capture_goal[1] - previous_goal[1],
            )
            > distance_tolerance_m
            or abs(normalize_angle(capture_goal[2] - previous_goal[2]))
            > yaw_tolerance_rad
        )
        return is_distinct, previous_goal

    @staticmethod
    def _container_two_stage_evidence_matches_staging(
        evidence: dict[str, Any],
        staging_goal: tuple[float, float, float],
    ) -> bool:
        """Check only the public pose token captured with the targeted M1 view."""

        observed = _goal_xyyaw_option(evidence.get("staging_pose_xyyaw"))
        if observed is None:
            return False
        return bool(
            math.hypot(observed[0] - staging_goal[0], observed[1] - staging_goal[1])
            <= 1e-6
            and abs(normalize_angle(observed[2] - staging_goal[2])) <= 1e-6
        )

    def _begin_container_two_stage_m1_capture(
        self,
        arrival_detail: dict[str, Any],
        now: float,
    ) -> list[dict[str, Any]]:
        """Move from a navigation-only anchor to its direct M1 capture pose.

        The anchor exists solely to obtain a costmap-reachable route around an
        inflated obstacle shoulder.  It is not a camera-distance adjustment and
        must never be used as M1 evidence.  Old candidates do not carry a
        separate capture mapping, in which case this method deliberately
        returns no command and preserves their legacy staging observation.
        """

        if self.candidate is None:
            return []
        candidate = dict(self.candidate)
        metadata = dict(candidate.get("metadata") or {})
        if not self._container_two_stage_staging_active(metadata):
            return []
        if metadata.get("container_two_stage_mapping_ready") is False:
            return []
        try:
            staging_index = max(
                0, int(metadata.get("interaction_approach_goal_option_index", 0))
            )
        except (TypeError, ValueError):
            return []
        staging_goals = container_two_stage_staging_goal_options(candidate)
        capture_goal = container_two_stage_capture_goal_for_staging(
            candidate, staging_index
        )
        if staging_index >= len(staging_goals) or capture_goal is None:
            return []
        anchor_goal = staging_goals[staging_index]
        if (
            math.hypot(capture_goal[0] - anchor_goal[0], capture_goal[1] - anchor_goal[1])
            <= 1e-6
            and abs(normalize_angle(capture_goal[2] - anchor_goal[2])) <= 1e-6
        ):
            return []
        interaction = dict(candidate.get("interaction_command") or {})
        try:
            capture_ready_distance_m = float(
                interaction.get(
                    "container_m1_capture_ready_distance_m",
                    interaction.get("container_physical_action_ready_distance_m"),
                )
            )
        except (TypeError, ValueError):
            return []
        if capture_ready_distance_m <= 0.0:
            return []
        interaction["interaction_approach_pose_xyyaw"] = list(capture_goal)
        interaction["interaction_ready_distance_m"] = capture_ready_distance_m
        interaction["navigation_goal_position_tolerance_m"] = capture_ready_distance_m
        interaction["navigation_goal_yaw_tolerance_rad"] = float(
            interaction.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
        )
        distinct_view_arrival_required, prior_sampled_capture_goal = (
            self._container_m1_capture_requires_distinct_view_arrival(
                metadata, capture_goal
            )
        )
        metadata.update(
            {
                "container_two_stage_phase": "m1_capture",
                "container_two_stage_staging_goal_option_index": staging_index,
                "container_two_stage_navigation_anchor_pose_xyyaw": list(anchor_goal),
                "container_two_stage_capture_pose_xyyaw": list(capture_goal),
                "effective_interaction_approach_pose_xyyaw": list(capture_goal),
                # The immutable anchor index remains the mapping key.  The
                # private one-goal capture navigation always dispatches option
                # zero, so its retries cannot be mistaken for another anchor.
                "interaction_approach_goal_option_index": staging_index,
                "goal_xyyaw_candidates": [],
                "interaction_observation_samples_at_viewpoint": 0,
                "interaction_observation_viewpoint_pending": False,
                # This applies only after M1 has actually sampled a different
                # mapped capture pose.  It is intentionally absent from legacy
                # candidates which have no direct-capture mapping.
                "container_m1_capture_requires_distinct_view_arrival": (
                    distinct_view_arrival_required
                ),
                "container_m1_capture_prior_sampled_goal_xyyaw": (
                    list(prior_sampled_capture_goal)
                    if prior_sampled_capture_goal is not None
                    else []
                ),
            }
        )
        candidate["goal_xyyaw"] = list(capture_goal)
        candidate["interaction_command"] = interaction
        candidate["metadata"] = metadata
        self.candidate = candidate
        return self._transition(
            STATE_APPROACH_INTERACTION,
            now,
            {
                "kind": "navigate",
                "candidate": self.candidate,
                "start_goal_option_index": 0,
                "interaction_approach_attempts": [
                    dict(item)
                    for item in metadata.get("interaction_approach_attempts") or []
                    if isinstance(item, dict)
                ],
                "reason": "container_navigation_anchor_to_m1_capture",
                "navigation_anchor_arrival": dict(arrival_detail),
            },
        )

    def _begin_container_two_stage_action_approach(
        self,
        observation: dict[str, Any],
        now: float,
    ) -> list[dict[str, Any]]:
        """Replace an accepted outer M1 barrier with its paired inner goal.

        No visual request is emitted by this transition.  The fresh M1 result
        remains attached to the outer staging pose and the next navigation
        completion is the only path that can issue the physical bridge command.
        """

        if self.candidate is None:
            return []
        candidate = dict(self.candidate)
        metadata = dict(candidate.get("metadata") or {})
        if not (
            self._container_two_stage_staging_active(metadata)
            or self._container_two_stage_m1_capture_active(metadata)
        ):
            return []
        try:
            staging_index = max(
                0,
                int(
                    metadata.get(
                        "container_two_stage_staging_goal_option_index",
                        metadata.get("interaction_approach_goal_option_index", 0),
                    )
                ),
            )
        except (TypeError, ValueError):
            return []
        staging_goals = container_two_stage_staging_goal_options(candidate)
        if staging_index >= len(staging_goals):
            return []
        capture_goal = container_two_stage_capture_goal_for_staging(
            candidate, staging_index
        )
        if capture_goal is None:
            return []
        # Candidate generation marks an incomplete outer-to-inner mapping
        # explicitly.  Even if an older trace still happens to retain a stale
        # option list, never use it to authorize a close physical move.
        if metadata.get("container_two_stage_mapping_ready") is False:
            return []
        evidence = metadata.get("accepted_container_m1_evidence")
        if (
            not isinstance(evidence, dict)
            or not self._container_two_stage_evidence_matches_staging(
                evidence, capture_goal
            )
        ):
            return []
        shared_anchor_pose = bool(metadata.get("container_anchor_shared_pose", False))
        front_evidence_action = (
            None
            if shared_anchor_pose
            else container_two_stage_action_goal_options_from_m1_front_evidence(
                candidate, evidence
            )
        )
        front_axis_required = bool(
            metadata.get("container_m1_front_axis_from_capture", False)
        )
        if shared_anchor_pose:
            action_goals = [staging_goals[staging_index]]
            action_labels = ["shared_anchor_physical_action"]
            front_evidence = {
                key: evidence[key]
                for key in (
                    "m1_front_axis_xy",
                    "m1_front_yaw",
                    "m1_front_axis_source",
                    "capture_pose_xyyaw",
                    "capture_step",
                )
                if key in evidence
            }
            action_geometry_source = "shared_selected_anchor"
        elif front_evidence_action is not None:
            action_goals, action_labels, front_evidence = front_evidence_action
            action_geometry_source = "m1_confirmed_capture_front_axis"
        elif front_axis_required and metadata.get("container_geometry_anchor_xy"):
            # The candidate explicitly requested this contract, but the
            # executor did not bind an M1-confirmed capture ray.  Do not quietly
            # fall back to a possibly stale staging face or a graph/oracle axis.
            return []
        else:
            action_goals = container_two_stage_action_goal_options_for_staging(
                candidate, staging_index
            )
            action_labels = container_two_stage_action_pose_labels_for_staging(
                candidate,
                staging_index,
                len(action_goals),
            )
            front_evidence = {}
            action_geometry_source = "legacy_staging_face"
        if not action_goals:
            return []
        action_goal = action_goals[0]
        action_label = action_labels[0]
        interaction = dict(candidate.get("interaction_command") or {})
        interaction["interaction_approach_pose_xyyaw"] = list(action_goal)
        if front_evidence.get("m1_front_axis_xy"):
            interaction["interaction_approach_axis_xy"] = list(
                front_evidence["m1_front_axis_xy"]
            )
            # Carry the face contract all the way to the simulator bridge.  A
            # categorical M1 ``front`` flag is not an authorization by itself:
            # the bridge must independently check that the arrived base yaw
            # faces the frozen, M1-confirmed world-space face normal.
            interaction["interaction_front_axis_source"] = str(
                front_evidence.get("m1_front_axis_source") or "m1_capture"
            )
            interaction["interaction_front_axis_validation_required"] = True
            interaction["interaction_target_center_xy"] = list(
                metadata.get("container_geometry_anchor_xy") or []
            )
        try:
            physical_ready_distance_m = float(
                interaction.get("container_physical_action_ready_distance_m")
            )
        except (TypeError, ValueError):
            return []
        if physical_ready_distance_m <= 0.0:
            return []
        interaction["interaction_ready_distance_m"] = physical_ready_distance_m
        interaction["navigation_goal_position_tolerance_m"] = physical_ready_distance_m
        interaction["navigation_goal_yaw_tolerance_rad"] = float(
            interaction.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
        )
        metadata.update(
            {
                "container_two_stage_phase": "physical_action",
                "container_two_stage_staging_goal_option_index": staging_index,
                "container_two_stage_staging_pose_xyyaw": list(
                    staging_goals[staging_index]
                ),
                "container_two_stage_capture_pose_xyyaw": list(capture_goal),
                # Keep the original primary fields for old readers/traces, but
                # expose the bounded same-face physical sequence separately.
                # It is consumed before moving to another outer M1 stance.
                "container_two_stage_action_goal_xyyaw": list(action_goal),
                "container_two_stage_action_pose_label": action_label,
                "container_two_stage_action_geometry_source": action_geometry_source,
                **front_evidence,
                "container_two_stage_action_goal_xyyaw_options": [
                    list(goal) for goal in action_goals
                ],
                "container_two_stage_action_pose_option_labels": list(action_labels),
                "container_two_stage_action_goal_option_index": 0,
                "container_m1_evidence_staging_goal_option_index": staging_index,
                "container_m1_evidence_staging_pose_xyyaw": list(
                    staging_goals[staging_index]
                ),
                "container_m1_evidence_capture_pose_xyyaw": list(capture_goal),
                # The inner phase must never request another M1 image.  It
                # retains the drawer's visual action regions on the interaction
                # command but clears only the observation-state flags.
                "observation_required": False,
                "reobserve": False,
                "container_pre_action_observation": False,
                "drawer_pre_action_observation": False,
                "m1_observation_staging_required": False,
                "interaction_observation_resolved": True,
                "effective_interaction_approach_pose_xyyaw": list(action_goal),
                "interaction_approach_goal_option_index": 0,
                "goal_xyyaw_candidates": [
                    list(goal) for goal in action_goals[1:]
                ],
                "interaction_approach_pose_labels": list(action_labels),
            }
        )
        candidate["goal_xyyaw"] = list(action_goal)
        candidate["interaction_command"] = interaction
        candidate["metadata"] = metadata
        self.candidate = candidate
        if shared_anchor_pose:
            # The selected anchor has already passed navigation arrival and the
            # accepted M1 evidence is bound to this exact pose.  There is no
            # second physical navigation target in the shared-anchor contract;
            # proceed directly to the bridge, which performs its normal final
            # pose/state validation before applying force.
            return self._transition(
                STATE_INTERACTING,
                now,
                {
                    "kind": "interact",
                    "candidate": self.candidate,
                    "observation": observation,
                    "reason": "container_m1_ready_at_shared_action_anchor",
                },
            )
        return self._transition(
            STATE_APPROACH_INTERACTION,
            now,
            {
                "kind": "navigate",
                "candidate": self.candidate,
                "start_goal_option_index": 0,
                "interaction_approach_attempts": [
                    dict(item)
                    for item in metadata.get("interaction_approach_attempts") or []
                    if isinstance(item, dict)
                ],
                "reason": "container_m1_ready_navigate_physical_action_pose",
                "observation": observation,
            },
        )

    def retry_container_two_stage_staging(
        self,
        *,
        next_staging_goal_option_index: int,
        interaction_approach_attempts: list[dict[str, Any]],
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return from an inner failure to the next outer staging viewpoint.

        A closer action pose cannot be re-observed: it has already consumed the
        accepted outer M1 evidence.  Re-enter the original outer sequence with
        the original flags restored so the next usable stance earns a new image.
        """

        if self.candidate is None or not bool(
            ((self.candidate.get("metadata") or {}).get("container_two_stage_approach", False))
        ):
            return []
        phase = str(
            ((self.candidate.get("metadata") or {}).get("container_two_stage_phase") or "")
        ).casefold()
        if phase not in {"physical_action", "m1_capture", "staging"}:
            return []
        if self.state not in {
            STATE_APPROACH_INTERACTION,
            STATE_INTERACTING,
            STATE_WAITING_FOR_INTERACTION_OBSERVATION,
        }:
            return []
        candidate = dict(self.candidate)
        metadata = dict(candidate.get("metadata") or {})
        staging_goals = container_two_stage_staging_goal_options(candidate)
        try:
            next_index = int(next_staging_goal_option_index)
        except (TypeError, ValueError):
            return []
        if next_index < 0 or next_index >= len(staging_goals):
            return []
        staging_labels = list(metadata.get("container_staging_pose_labels") or [])
        interaction = dict(candidate.get("interaction_command") or {})
        interaction["interaction_approach_pose_xyyaw"] = list(staging_goals[next_index])
        try:
            staging_ready_distance_m = float(
                interaction.get("container_staging_ready_distance_m")
            )
        except (TypeError, ValueError):
            return []
        if staging_ready_distance_m <= 0.0:
            return []
        interaction["interaction_ready_distance_m"] = staging_ready_distance_m
        interaction["navigation_goal_position_tolerance_m"] = staging_ready_distance_m
        interaction["navigation_goal_yaw_tolerance_rad"] = float(
            interaction.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
        )
        # The next outer stance must request an image newer than the M1 frame
        # that authorized the failed inner approach.  Keep that accepted
        # capture as the monotonic observation baseline before clearing the
        # one-use evidence token below; otherwise a delayed response captured
        # before the inner attempt could satisfy the next outer M1 request.
        accepted_capture_step = self._interaction_observation_capture_step(
            metadata.get("accepted_container_m1_evidence")
            if isinstance(metadata.get("accepted_container_m1_evidence"), dict)
            else {}
        )
        previous_capture_step = self._interaction_observation_capture_step(
            {
                "capture_step": metadata.get(
                    "interaction_observation_after_capture_step"
                )
            }
        )
        if accepted_capture_step is not None:
            metadata["interaction_observation_after_capture_step"] = max(
                int(accepted_capture_step),
                int(previous_capture_step)
                if previous_capture_step is not None
                else int(accepted_capture_step),
            )
        metadata.update(
            {
                "container_two_stage_phase": "staging",
                "container_two_stage_last_inner_failure": dict(detail or {}),
                "observation_required": bool(
                    metadata.get(
                        "container_two_stage_staging_observation_required", True
                    )
                ),
                "reobserve": True,
                "container_pre_action_observation": bool(
                    metadata.get(
                        "container_two_stage_staging_container_pre_action_observation",
                        False,
                    )
                ),
                "drawer_pre_action_observation": bool(
                    metadata.get(
                        "container_two_stage_staging_drawer_pre_action_observation",
                        False,
                    )
                ),
                "m1_observation_staging_required": True,
                "interaction_observation_resolved": False,
                "goal_xyyaw_candidates": [list(goal) for goal in staging_goals],
                "interaction_approach_pose_labels": staging_labels,
                "effective_interaction_approach_pose_xyyaw": list(
                    staging_goals[next_index]
                ),
                "interaction_approach_goal_option_index": next_index,
                "interaction_observation_samples_at_viewpoint": 0,
                "interaction_observation_viewpoint_pending": True,
            }
        )
        for key in (
            "accepted_container_m1_evidence",
            "accepted_container_m1_capture_pose_xyyaw",
            "accepted_container_m1_capture_step",
            "container_m1_evidence_staging_goal_option_index",
            "container_m1_evidence_staging_pose_xyyaw",
            "container_m1_evidence_capture_pose_xyyaw",
            "container_two_stage_capture_pose_xyyaw",
            "container_two_stage_navigation_anchor_pose_xyyaw",
            "container_m1_capture_requires_distinct_view_arrival",
            "container_m1_capture_prior_sampled_goal_xyyaw",
        ):
            metadata.pop(key, None)
        candidate["interaction_command"] = interaction
        candidate["metadata"] = metadata
        # Preserve the canonical primary outer goal.  The executor receives the
        # original index below, so its preflight/debug trace remains aligned
        # with the immutable outer-to-inner mapping.
        candidate["goal_xyyaw"] = list(staging_goals[0])
        self.candidate = candidate
        now = time.monotonic() if now is None else float(now)
        return self._transition(
            STATE_APPROACH_INTERACTION,
            now,
            {
                "kind": "navigate",
                "candidate": self.candidate,
                "start_goal_option_index": next_index,
                "interaction_approach_attempts": [
                    dict(item) for item in interaction_approach_attempts
                ],
                "reason": (
                "container_m1_capture_failed_next_outer_staging"
                if phase == "m1_capture"
                else (
                    "container_staging_navigation_failed_next_m1_viewpoint"
                    if phase == "staging"
                    else "container_inner_action_failed_next_outer_staging"
                )
                ),
            },
        )

    def _interaction_observation_max_attempts(self) -> int:
        metadata = (self.candidate or {}).get("metadata") or {}
        try:
            return max(1, int(metadata.get("interaction_observation_max_attempts", 2)))
        except (TypeError, ValueError):
            return 2

    def _interaction_observation_same_pose_samples_per_view(self) -> int:
        metadata = (self.candidate or {}).get("metadata") or {}
        configured = metadata.get(
            "interaction_observation_same_pose_samples_per_view",
            getattr(self.config, "interaction_observation_same_pose_samples_per_view", 2),
        )
        try:
            return max(1, int(configured))
        except (TypeError, ValueError):
            return 2

    def _interaction_observation_max_viewpoints(self) -> int:
        metadata = (self.candidate or {}).get("metadata") or {}
        configured = metadata.get(
            "interaction_observation_max_viewpoints",
            getattr(self.config, "interaction_observation_max_viewpoints", 0),
        )
        try:
            configured_value = int(configured or 0)
        except (TypeError, ValueError):
            configured_value = 0
        if configured_value > 0:
            return configured_value
        return self._interaction_observation_max_attempts()

    def _interaction_observation_max_total_requests(self) -> int:
        metadata = (self.candidate or {}).get("metadata") or {}
        configured = metadata.get(
            "interaction_observation_max_total_requests",
            getattr(self.config, "interaction_observation_max_total_requests", 0),
        )
        try:
            configured_value = int(configured or 0)
        except (TypeError, ValueError):
            configured_value = 0
        if configured_value > 0:
            return configured_value
        return min(
            self._interaction_observation_max_attempts(),
            self._interaction_observation_max_viewpoints()
            * self._interaction_observation_same_pose_samples_per_view(),
        )

    def _defer_container_m1_evidence(
        self,
        observation: dict[str, Any],
        now: float,
        *,
        drawer_pre_action: bool,
        reason: str,
    ) -> list[dict[str, Any]]:
        """Finish this attempt without declaring the container impossible.

        M1 is a visual authorizer.  When its bounded evidence plan is exhausted
        it can defer the candidate for a later exploration/cooldown cycle, but
        it cannot write a permanent candidate exclusion from image evidence.
        """

        metadata = dict((self.candidate or {}).get("metadata") or {})
        precondition_kind = "drawer" if drawer_pre_action else "container"
        return self._finish(
            False,
            {
                **observation,
                "action_executed": False,
                "observation_outcome": "m1_evidence_inconclusive",
                "m1_evidence_inconclusive": True,
                "retryable": True,
                "terminal_candidate_exclusion": False,
                "failure_stage": "interaction_visual_precondition",
                "reason": reason,
                "observation_attempts": int(
                    metadata.get("interaction_observation_attempts", 0) or 0
                ),
                "observation_samples_per_view": (
                    self._interaction_observation_same_pose_samples_per_view()
                ),
                "observation_viewpoints": int(
                    metadata.get("interaction_observation_viewpoint_count", 0) or 0
                ),
                "observation_max_viewpoints": (
                    self._interaction_observation_max_viewpoints()
                ),
                "observation_max_total_requests": (
                    self._interaction_observation_max_total_requests()
                ),
                "precondition_kind": precondition_kind,
            },
            now,
        )

    def defer_container_m1_viewpoint_navigation(
        self,
        detail: dict[str, Any] | None = None,
        *,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Defer a container after reachable M1 views cannot be completed.

        Navigation failures before a new camera capture do not prove that the
        object is impossible to interact with.  This is especially important
        immediately after the startup scan, when every anchor can temporarily
        return ``empty_plan`` against an incomplete map.  Preserve the same
        retryable, non-exclusion outcome used after inconclusive visual evidence
        instead of turning that one map generation into a permanent object fact.
        """

        if self.candidate is None:
            return []
        metadata = dict(self.candidate.get("metadata") or {})
        try:
            observation_attempts = int(
                metadata.get("interaction_observation_attempts", 0) or 0
            )
        except (TypeError, ValueError):
            observation_attempts = 0
        if not (
            (
                self._container_two_stage_staging_active(metadata)
                or self._container_two_stage_m1_capture_active(metadata)
            )
            and bool(metadata.get("m1_observation_staging_required", False))
        ):
            return []
        drawer_pre_action = bool(metadata.get("drawer_pre_action_observation"))
        kind = "drawer" if drawer_pre_action else "container"
        staging_goal_count = len(container_two_stage_staging_goal_options(self.candidate))
        unavailable_indices: set[int] = set()
        for raw_index in list(
            (detail or {}).get("container_m1_unavailable_staging_indices")
            or metadata.get("container_m1_unavailable_staging_indices")
            or []
        ):
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < staging_goal_count:
                unavailable_indices.add(index)
        all_anchors_unreachable = bool(
            staging_goal_count > 0
            and len(unavailable_indices) >= staging_goal_count
        )
        return self._defer_container_m1_evidence(
            {
                **dict(detail or {}),
                "m1_viewpoint_navigation_inconclusive": True,
                "m1_capture_not_reached": observation_attempts <= 0,
                "all_container_anchors_unreachable": all_anchors_unreachable,
                "container_anchor_count": staging_goal_count,
                "container_unreachable_anchor_count": len(unavailable_indices),
            },
            time.monotonic() if now is None else float(now),
            drawer_pre_action=drawer_pre_action,
            reason=f"{kind}_m1_evidence_inconclusive_viewpoint_navigation",
        )

    def _advance_interaction_reobservation_approach(
        self,
        observation: dict[str, Any],
        now: float,
        *,
        drawer_pre_action: bool,
    ) -> list[dict[str, Any]]:
        """Move to one bounded new capture viewpoint after same-pose sampling.

        The caller has already collected the configured fresh samples at the
        held M1 capture pose.  For the three-phase container contract this
        method returns to the next *navigation anchor* and lets normal arrival
        logic move inward to its paired direct capture pose; it never treats a
        navigation-only anchor image as M1 evidence.
        """

        if self.candidate is None:
            return []
        metadata = dict(self.candidate.get("metadata") or {})
        is_direct_capture = self._container_two_stage_m1_capture_active(metadata)
        goal_options = (
            container_two_stage_staging_goal_options(self.candidate)
            if is_direct_capture
            else navigation_goal_options(self.candidate)
        )
        try:
            selected_index = max(
                0,
                int(
                    metadata.get(
                        "container_two_stage_staging_goal_option_index",
                        metadata.get("interaction_approach_goal_option_index", 0),
                    )
                    if is_direct_capture
                    else metadata.get("interaction_approach_goal_option_index", 0)
                ),
            )
        except (TypeError, ValueError):
            selected_index = 0
        if is_direct_capture:
            viewed_indices: set[int] = set()
            for raw_index in list(
                metadata.get("interaction_observation_viewpoint_staging_indices")
                or []
            ):
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(goal_options):
                    viewed_indices.add(index)
            # The current direct capture has just returned a result even if an
            # older trace did not record its first request bookkeeping.
            viewed_indices.add(selected_index)
            view_state = str(observation.get("view_state") or "").strip().casefold()
            if view_state in {"side", "side_or_back", "back", "rear"}:
                rejected_face_indices = container_two_stage_face_indices(
                    self.candidate, selected_index
                )
                viewed_indices.update(rejected_face_indices)
                previously_rejected: set[int] = set()
                for raw_index in metadata.get(
                    "container_m1_rejected_face_staging_indices", []
                ):
                    try:
                        index = int(raw_index)
                    except (TypeError, ValueError):
                        continue
                    if index >= 0:
                        previously_rejected.add(index)
                metadata["container_m1_rejected_face_staging_indices"] = sorted(
                    previously_rejected | set(rejected_face_indices)
                )
            unavailable_indices: set[int] = set()
            for key in (
                "container_m1_unavailable_staging_indices",
                "container_m1_rejected_face_staging_indices",
            ):
                for raw_index in list(metadata.get(key) or []):
                    try:
                        index = int(raw_index)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= index < len(goal_options):
                        unavailable_indices.add(index)
            next_index = container_two_stage_next_m1_viewpoint_index(
                self.candidate,
                excluded_indices=viewed_indices | unavailable_indices,
            )
        else:
            next_index = selected_index + 1
        precondition_kind = "drawer" if drawer_pre_action else "container"
        if next_index is None or next_index >= len(goal_options):
            return self._defer_container_m1_evidence(
                observation,
                now,
                drawer_pre_action=drawer_pre_action,
                reason=f"{precondition_kind}_m1_evidence_inconclusive_no_alternate_viewpoint",
            )
        capture_step = self._interaction_observation_capture_step(observation)
        previous_baseline = self._interaction_observation_capture_step(
            {"capture_step": metadata.get("interaction_observation_after_capture_step")}
        )
        if capture_step is not None and (
            previous_baseline is None or capture_step > previous_baseline
        ):
            metadata["interaction_observation_after_capture_step"] = capture_step
        metadata["interaction_visual_reobserve_count"] = int(
            metadata.get("interaction_visual_reobserve_count", 0) or 0
        ) + 1
        metadata["interaction_visual_reobserve_from_goal_option_index"] = selected_index
        metadata["interaction_visual_reobserve_next_goal_option_index"] = next_index
        metadata["last_interaction_visual_precondition"] = dict(observation)
        if drawer_pre_action:
            # Preserve these existing diagnostics for drawer-specific reports.
            metadata["drawer_visual_reobserve_count"] = int(
                metadata.get("drawer_visual_reobserve_count", 0) or 0
            ) + 1
            metadata["drawer_visual_reobserve_from_goal_option_index"] = selected_index
            metadata["drawer_visual_reobserve_next_goal_option_index"] = next_index
            metadata["last_drawer_visual_precondition"] = dict(observation)
        self.candidate["metadata"] = metadata
        if is_direct_capture:
            # Restore the next immutable anchor and reset the per-view sample
            # counter.  ``retry_container_two_stage_staging`` also clears the
            # prior request's evidence/baseline so a delayed response cannot
            # authorize this new view.
            return self.retry_container_two_stage_staging(
                next_staging_goal_option_index=next_index,
                interaction_approach_attempts=[
                    dict(item)
                    for item in metadata.get("interaction_approach_attempts") or []
                    if isinstance(item, dict)
                ],
                detail={
                    **observation,
                    "reason": f"{precondition_kind}_m1_next_capture_viewpoint",
                    "interaction_visual_reobserve_from_goal_option_index": selected_index,
                    "interaction_visual_reobserve_next_goal_option_index": next_index,
                },
                now=now,
            )
        return self._transition(
            STATE_APPROACH_INTERACTION,
            now,
            {
                "kind": "navigate",
                "candidate": self.candidate,
                "start_goal_option_index": next_index,
                "interaction_approach_attempts": [
                    dict(item)
                    for item in metadata.get("interaction_approach_attempts") or []
                    if isinstance(item, dict)
                ],
                "reason": f"{precondition_kind}_visual_reobserve_next_approach",
            },
        )

    @staticmethod
    def _interaction_observation_capture_step(
        detail: dict[str, Any] | None,
    ) -> int | None:
        detail = detail or {}
        for key in (
            "observation_capture_step",
            "attribute_capture_step",
            "capture_step",
            "rgb_step_seq",
            "current_rgb_step_seq",
            "step",
        ):
            value = detail.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _normalized_interaction_observation(
        detail: dict[str, Any] | None,
    ) -> dict[str, Any]:
        detail = dict(detail or {})
        attributes = detail.get("attributes")
        normalized = dict(attributes) if isinstance(attributes, dict) else {}
        normalized.update(detail)
        return normalized

    def _request_interaction_observation(
        self,
        detail: dict[str, Any],
        now: float,
    ) -> list[dict[str, Any]]:
        if self.candidate is None:
            return []
        metadata = dict(self.candidate.get("metadata") or {})
        previous_capture_step = self._interaction_observation_capture_step(detail)
        baseline_capture_step = self._interaction_observation_capture_step(
            {"capture_step": metadata.get("interaction_observation_after_capture_step")}
        )
        if previous_capture_step is not None and (
            baseline_capture_step is None
            or previous_capture_step > baseline_capture_step
        ):
            baseline_capture_step = previous_capture_step
            metadata["interaction_observation_after_capture_step"] = baseline_capture_step
        viewpoints = int(metadata.get("interaction_observation_viewpoint_count", 0) or 0)
        if self._container_two_stage_m1_capture_active(metadata):
            try:
                staging_index = int(
                    metadata.get("container_two_stage_staging_goal_option_index", 0)
                )
            except (TypeError, ValueError):
                staging_index = 0
            viewed_indices: list[int] = []
            for raw_index in list(
                metadata.get("interaction_observation_viewpoint_staging_indices")
                or []
            ):
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if index >= 0 and index not in viewed_indices:
                    viewed_indices.append(index)
            if staging_index not in viewed_indices:
                viewed_indices.append(staging_index)
            viewpoints = len(viewed_indices)
            metadata["interaction_observation_viewpoint_staging_indices"] = viewed_indices
            metadata["interaction_observation_viewpoint_count"] = viewpoints
            # Record only a capture pose which is about to issue M1.  Reaching
            # an outer anchor or failing to reach a later direct capture must
            # not become a sampled viewpoint or consume the evidence budget.
            capture_goal = _goal_xyyaw_option(
                metadata.get("container_two_stage_capture_pose_xyyaw")
            )
            if capture_goal is None:
                capture_goal = container_two_stage_capture_goal_for_staging(
                    self.candidate, staging_index
                )
            if capture_goal is not None:
                metadata["container_m1_last_sampled_capture_goal_xyyaw"] = list(
                    capture_goal
                )
                metadata["container_m1_last_sampled_capture_staging_index"] = (
                    staging_index
                )
        elif viewpoints <= 0:
            # Legacy candidates do not transition through a distinct capture
            # pose.  Their first request still owns one explicit viewpoint.
            viewpoints = 1
            metadata["interaction_observation_viewpoint_count"] = viewpoints
            metadata["interaction_observation_samples_at_viewpoint"] = 0
        samples_at_viewpoint = (
            int(metadata.get("interaction_observation_samples_at_viewpoint", 0) or 0)
            + 1
        )
        metadata["interaction_observation_samples_at_viewpoint"] = samples_at_viewpoint
        metadata["interaction_observation_viewpoint_pending"] = True
        attempts = int(metadata.get("interaction_observation_attempts", 0) or 0) + 1
        metadata["interaction_observation_attempts"] = attempts
        minimum_capture_step = (
            int(baseline_capture_step) + 1
            if baseline_capture_step is not None
            else None
        )
        metadata["interaction_observation_min_capture_step"] = minimum_capture_step
        self.candidate["metadata"] = metadata
        interaction = self.candidate.get("interaction_command") or {}
        return self._transition(
            STATE_WAITING_FOR_INTERACTION_OBSERVATION,
            now,
            {
                "kind": "request_interaction_observation",
                "candidate": self.candidate,
                "node_id": interaction.get("node_id") or self.candidate.get("target_id"),
                "object_id": interaction.get("object_id") or self.candidate.get("target_name"),
                "attempt": attempts,
                "max_attempts": self._interaction_observation_max_attempts(),
                "same_pose_sample": samples_at_viewpoint,
                "same_pose_samples_per_view": (
                    self._interaction_observation_same_pose_samples_per_view()
                ),
                "viewpoint": viewpoints,
                "max_viewpoints": self._interaction_observation_max_viewpoints(),
                "max_total_requests": self._interaction_observation_max_total_requests(),
                "min_capture_step": minimum_capture_step,
                "require_current_visibility": True,
                "require_attribute_status": "ready",
                "required_attribute_source": metadata.get(
                    "interaction_observation_source", "mllm_attribute_inference"
                ),
                "reason": metadata.get(
                    "observation_reason", "mllm_portal_state_unknown"
                ),
            },
        )

    def retry_interaction_approach(
        self,
        *,
        start_goal_option_index: int,
        interaction_approach_attempts: list[dict[str, Any]],
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return from an unexecuted interaction to a bounded next approach.

        The evaluator may reject an interaction command after its authoritative
        simulator-pose check.  That is not an object failure: keep the same
        candidate and transition back to approach navigation without emitting a
        terminal feedback event to the decision layer.
        """

        if self.state != STATE_INTERACTING or self.candidate is None:
            return []
        metadata = dict(self.candidate.get("metadata") or {})
        metadata["interaction_approach_attempts"] = [
            dict(item) for item in interaction_approach_attempts
        ]
        if detail:
            metadata["last_interaction_pose_validation"] = dict(detail)
        self.candidate["metadata"] = metadata
        now = time.monotonic() if now is None else float(now)
        return self._transition(
            STATE_APPROACH_INTERACTION,
            now,
            {
                "kind": "navigate",
                "candidate": self.candidate,
                "start_goal_option_index": max(0, int(start_goal_option_index)),
                "interaction_approach_attempts": [
                    dict(item) for item in interaction_approach_attempts
                ],
            },
        )

    def on_drawer_scan_ready(
        self,
        candidate: dict[str, Any],
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Start the sealed drawer scan only after a fresh public frame.

        The executor supplies a candidate whose public 2-D box and capture
        step were both observed after navigation reached the approach pose.
        Keeping this as an explicit state prevents a stale pre-approach box
        from flowing into the evaluator-side direct drawer scan route.
        """

        if self.state != STATE_WAITING_FOR_DRAWER_SCAN:
            return []
        self.candidate = dict(candidate)
        now = time.monotonic() if now is None else float(now)
        return self._transition(
            STATE_INTERACTING,
            now,
            {
                "kind": "publish_drawer_scan",
                "candidate": self.candidate,
                "detail": detail or {},
            },
        )

    def on_drawer_scan_wait_failed(
        self,
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.state != STATE_WAITING_FOR_DRAWER_SCAN:
            return []
        return self._finish(
            False,
            detail or {"reason": "drawer_scan_fresh_frame_timeout"},
            now,
        )

    def on_interaction_result(
        self,
        success: bool,
        detail: dict[str, Any] | None = None,
        now: float | None = None,
        *,
        backend_success: bool | None = None,
    ) -> list[dict[str, Any]]:
        now = time.monotonic() if now is None else float(now)
        if self.state != STATE_INTERACTING:
            return []
        detail = dict(detail or {})
        backend_success = bool(success) if backend_success is None else bool(backend_success)
        if str(detail.get("failure_stage") or "") in {
            "interaction_visual_precondition",
            "interaction_execution",
        }:
            # These outcomes have already consumed their bounded observation or
            # execution contract.  They are terminal semantic failures, not
            # physical states for M3 to reinterpret or retry.
            return self._finish(False, detail, now)
        if is_interaction_pose_precondition_failure(detail):
            # No force/action reached the object.  The executor may instead
            # call retry_interaction_approach while still in INTERACTING; if it
            # reports this terminally, do not send it through post-action M3.
            return self._finish(False, detail, now)
        if is_static_portal_interaction_feedback(self.candidate, detail):
            # A public fixed-open aperture is already a terminal no-action
            # observation. A failed/non-open bridge response, however, still
            # gets M3's *post-action-only* audit below; it must not be silently
            # collapsed into static_open by the executor feedback spelling.
            static_state = str(
                detail.get("state") or detail.get("post_state") or ""
            ).strip().casefold()
            direct_success = bool(
                success
                or str(detail.get("status") or "").upper() == "SUCCEEDED"
                or static_state == "static_open"
            )
            if direct_success and static_state in {"static", "static_open"}:
                detail.setdefault("verification_mode", "direct_static_portal_feedback")
                detail.setdefault("verification_required", False)
                detail.setdefault("backend_success", direct_success)
                return self._finish(True, detail, now)
        metadata = dict(self.candidate.get("metadata") or {})
        metadata["interaction_backend_success"] = backend_success
        metadata["interaction_backend_result"] = dict(detail)
        self.candidate["metadata"] = metadata
        # Every physical result (including a failed action) gets one explicit
        # post-action verification request.  A failed backend result is kept
        # as an immutable fact; M3 may explain it or request a bounded retry,
        # but it cannot turn it into a successful terminal outcome.
        return self._transition(
            STATE_VERIFYING,
            now,
            {
                "kind": "verify_interaction",
                "candidate": self.candidate,
                "verification_phase": "post_interaction",
                "backend_success": backend_success,
                "interaction_result": detail,
            },
        )

    def on_graph_state(
        self, state: str, detail: dict[str, Any] | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        if self.state != STATE_VERIFYING or self.candidate is None:
            return []
        graph_detail = dict(detail or {})
        graph_detail.setdefault("state", state)
        if is_static_portal_interaction_feedback(self.candidate, graph_detail):
            # This is a fail-safe for a graph callback racing the direct
            # executor result.  It is still immediate and never depends on
            # ``verification_timeout_s``.
            graph_detail.setdefault("verification_mode", "direct_static_portal_feedback")
            graph_detail.setdefault("verification_required", False)
            return self._finish(
                str(state).strip().casefold() == "static_open",
                graph_detail,
                now,
            )
        expected = str((self.candidate.get("interaction_command") or {}).get("expected_state") or "open")
        if str(state) != expected:
            return []
        return self._finish_verified_interaction(True, graph_detail, now)

    def on_verification_result(
        self,
        success: bool,
        detail: dict[str, Any] | None = None,
        retry: bool = False,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.state != STATE_VERIFYING or self.candidate is None:
            return []
        now = time.monotonic() if now is None else float(now)
        if success:
            return self._finish_verified_interaction(
                True, detail or {"verified": True}, now
            )
        if retry:
            if not self._interaction_backend_succeeded():
                return self._finish(
                    False,
                    {
                        **(detail or {}),
                        "backend_success": False,
                        "reason": "interaction_backend_failed",
                    },
                    now,
                )
            return self._transition(
                STATE_INTERACTING,
                now,
                {"kind": "interact", "candidate": self.candidate, "retry": True},
            )
        return self._finish(False, detail or {"reason": "verification_failed"}, now)

    def on_backend_result(
        self,
        success: bool,
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Commit the sealed executor result without waiting for M3.

        The simulator/evaluator is authoritative about whether an interaction
        command was executed.  M3 is a post-action visual audit and therefore
        must not be able to replay a command (or hold the state machine in
        ``VERIFYING``) after a successful backend result.  Callers may launch
        that audit separately and attach its bounded re-observation request to
        telemetry.
        """

        if self.state != STATE_VERIFYING or self.candidate is None:
            return []
        now = time.monotonic() if now is None else float(now)
        detail = dict(detail or {})
        detail.setdefault("backend_success", bool(success))
        detail.setdefault("verification_required", False)
        detail.setdefault("verification_mode", "backend_postcondition")
        detail.setdefault("visual_audit_pending", True)
        return self._finish(bool(success), detail, now)

    def _interaction_backend_succeeded(self) -> bool:
        metadata = (self.candidate or {}).get("metadata") or {}
        return bool(metadata.get("interaction_backend_success", True))

    def _finish_verified_interaction(
        self,
        verification_success: bool,
        detail: dict[str, Any],
        now: float | None,
    ) -> list[dict[str, Any]]:
        """Merge M3 with the immutable backend result.

        In particular, ``verification_success=True`` cannot resurrect a
        physical action for which the backend reported failure.
        """

        if not verification_success:
            return self._finish(False, detail, now)
        if not self._interaction_backend_succeeded():
            return self._finish(
                False,
                {
                    **detail,
                    "backend_success": False,
                    "verification_success": True,
                    "reason": "interaction_backend_failed",
                },
                now,
            )
        return self._finish(
            True,
            {**detail, "backend_success": True, "verification_success": True},
            now,
        )

    def on_target_visibility(
        self,
        visible: bool,
        detail: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        if (
            self.state != STATE_VERIFYING
            or self.candidate is None
            or self._behavior_type() != BEHAVIOR_NAVIGATE
            or not bool((self.candidate.get("metadata") or {}).get("target_goal"))
        ):
            return []
        if not visible:
            return []
        return self._finish(True, detail or {"target_visible": True}, now)

    def timeout_reason(self, now: float | None = None) -> str:
        if self.state in {STATE_IDLE, STATE_SUCCEEDED, STATE_FAILED}:
            return ""
        now = time.monotonic() if now is None else float(now)
        elapsed = now - self.state_started_at
        if self.state == STATE_PREPARING_EXPLORE:
            return (
                "explore_prepare_timeout"
                if elapsed > self.config.explore_prepare_timeout_s
                else ""
            )
        if self.state == STATE_SCANNING:
            return (
                "scan_timeout"
                if self.config.scan_timeout_s > 0.0
                and elapsed > self.config.scan_timeout_s
                else ""
            )
        if self.state == STATE_NAVIGATING:
            return "navigation_timeout" if elapsed > self.config.navigation_timeout_s else ""
        if self.state == STATE_APPROACH_INTERACTION:
            return (
                "interaction_navigation_timeout"
                if elapsed > self.config.interaction_navigation_timeout_s
                else ""
            )
        if self.state == STATE_WAITING_FOR_DRAWER_SCAN:
            return (
                "drawer_scan_fresh_frame_timeout"
                if elapsed > self.config.drawer_scan_wait_timeout_s
                else ""
            )
        if self.state == STATE_WAITING_FOR_INTERACTION_OBSERVATION:
            return (
                "interaction_observation_timeout"
                if elapsed > self.config.interaction_observation_timeout_s
                else ""
            )
        if self.state == STATE_FINALIZING_EXPLORE:
            return (
                "explore_finalize_timeout"
                if elapsed > self.config.explore_finalize_timeout_s
                else ""
            )
        if self.state == STATE_INTERACTING:
            return "interaction_timeout" if elapsed > self.config.interaction_timeout_s else ""
        if self.state == STATE_VERIFYING:
            return "verification_timeout" if elapsed > self.config.verification_timeout_s else ""
        return ""

    def fail_timeout(self, reason: str, now: float | None = None) -> list[dict[str, Any]]:
        if self._behavior_type() == BEHAVIOR_EXPLORE and self.state in {
            STATE_PREPARING_EXPLORE,
            STATE_NAVIGATING,
        }:
            now = time.monotonic() if now is None else float(now)
            return self._transition(
                STATE_FINALIZING_EXPLORE,
                now,
                {
                    "kind": "finalize_frontier",
                    "candidate": self.candidate,
                    "success": False,
                    "detail": {"reason": reason},
                },
            )
        metadata = (self.candidate or {}).get("metadata") or {}
        if (
            self.state == STATE_WAITING_FOR_INTERACTION_OBSERVATION
            and str(reason or "") == "interaction_observation_timeout"
            and (
                bool(metadata.get("drawer_pre_action_observation"))
                or bool(metadata.get("container_pre_action_observation"))
            )
        ):
            # A missing M1 reply is visual uncertainty just like an oblique
            # reply.  Spend the bounded same-pose/viewpoint evidence plan; do
            # not let one endpoint timeout permanently exclude the container.
            return self.on_interaction_observation_result(
                {
                    "reason": "interaction_observation_timeout",
                    "attribute_status": "timeout",
                    "is_currently_visible": False,
                },
                now,
            )
        return self._finish(False, {"reason": reason}, now)

    def summary(self) -> dict[str, Any]:
        candidate = self.candidate or {}
        metadata = candidate.get("metadata") or {}
        return {
            "state": self.state,
            "candidate_id": candidate.get("candidate_id", ""),
            "behavior_type": self._behavior_type(),
            "error": self.error,
            "effective_goal_xyyaw": list(
                metadata.get("effective_interaction_approach_pose_xyyaw")
                or candidate.get("goal_xyyaw")
                or []
            ),
            "interaction_approach_goal_option_index": metadata.get(
                "interaction_approach_goal_option_index"
            ),
            "interaction_observation_attempts": metadata.get(
                "interaction_observation_attempts", 0
            ),
            "interaction_backend_success": metadata.get(
                "interaction_backend_success"
            ),
        }

    def _behavior_type(self) -> str:
        return "" if self.candidate is None else str(self.candidate.get("behavior_type") or "")

    def _transition(
        self, state: str, now: float, command: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.state = state
        self.state_started_at = now
        return [] if command is None else [command]

    def _finish(
        self, success: bool, detail: dict[str, Any], now: float | None
    ) -> list[dict[str, Any]]:
        now = time.monotonic() if now is None else float(now)
        self.state = STATE_SUCCEEDED if success else STATE_FAILED
        self.state_started_at = now
        self.error = "" if success else str(detail.get("reason") or detail.get("status") or "execution_failed")
        return [
            {
                "kind": "terminal",
                "success": bool(success),
                "detail": detail,
                "candidate": self.candidate,
            }
        ]
