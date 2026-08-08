from __future__ import annotations

from dataclasses import dataclass
import math
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
        }
        or verification_source == "executor_pose_precondition"
    )


def interaction_observation_disposition(
    detail: dict[str, Any] | None,
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
        return "terminal"
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
        "visual_reposition_required",
        "navigation_timeout",
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
    reference_xy: tuple[float, float] | None = None
    reference_yaw: float | None = None
    reference_goal_distance_m: float | None = None
    last_progress_at: float | None = None

    def reset(
        self,
        pose: tuple[float, ...] | None,
        now: float,
        goal_distance_m: float | None = None,
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

    def observe(
        self,
        pose: tuple[float, ...] | None,
        now: float,
        *,
        goal_distance_m: float | None = None,
        local_plan_fresh: bool = False,
    ) -> bool:
        if self.timeout_s <= 0.0 or pose is None:
            return False
        if self.reference_xy is None or self.last_progress_at is None:
            self.reset(pose, now, goal_distance_m)
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
            self.reset(pose, now, goal_distance_m)
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
                self.reset(pose, now, current_goal_distance_m)
                return False
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
class ExecutionConfig:
    navigation_timeout_s: float = 180.0
    interaction_navigation_timeout_s: float = 180.0
    interaction_timeout_s: float = 30.0
    drawer_scan_wait_timeout_s: float = 8.0
    interaction_observation_timeout_s: float = 8.0
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
        self.candidate["metadata"] = metadata

        disposition = interaction_observation_disposition(observation)
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
        max_attempts = self._interaction_observation_max_attempts()
        if attempts < max_attempts:
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

    def _interaction_observation_max_attempts(self) -> int:
        metadata = (self.candidate or {}).get("metadata") or {}
        try:
            return max(1, int(metadata.get("interaction_observation_max_attempts", 2)))
        except (TypeError, ValueError):
            return 2

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
        baseline_capture_step = metadata.get("interaction_observation_after_capture_step")
        if baseline_capture_step is None and previous_capture_step is not None:
            baseline_capture_step = previous_capture_step
            metadata["interaction_observation_after_capture_step"] = baseline_capture_step
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
