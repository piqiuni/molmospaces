from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from .behavior_candidates import BEHAVIOR_EXPLORE, BEHAVIOR_INTERACT, BEHAVIOR_NAVIGATE


STATE_IDLE = "IDLE"
STATE_PREPARING_EXPLORE = "PREPARING_EXPLORE"
STATE_NAVIGATING = "NAVIGATING"
STATE_FINALIZING_EXPLORE = "FINALIZING_EXPLORE"
STATE_APPROACH_INTERACTION = "APPROACH_INTERACTION"
STATE_WAITING_FOR_DRAWER_SCAN = "WAIT_FOR_DRAWER_SCAN"
STATE_INTERACTING = "INTERACTING"
STATE_VERIFYING = "VERIFYING"
STATE_SUCCEEDED = "SUCCEEDED"
STATE_FAILED = "FAILED"


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
    """Keep exploration goals responsive while preserving precise skill approaches."""

    return str(behavior_type or "").upper() != BEHAVIOR_EXPLORE


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
    Only a verified executor stagnation is allowed to advance to another
    approach pose: terminal move_base failures keep their existing handling,
    and a bounded number of actual navigation attempts prevents a large
    candidate list from consuming the whole episode.
    """

    if str(behavior_type or "").upper() != BEHAVIOR_INTERACT:
        return None
    if str(failure_detail.get("reason") or "").lower() != "navigation_stagnation":
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
        if (
            displacement >= self.min_displacement_m
            or yaw_change >= self.min_yaw_change_rad
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
    verification_timeout_s: float = 30.0
    explore_prepare_timeout_s: float = 10.0
    explore_finalize_timeout_s: float = 10.0


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
            return self._transition(
                STATE_INTERACTING,
                now,
                {"kind": "interact", "candidate": self.candidate},
            )
        raise ValueError(f"Unsupported behavior type: {behavior_type}")

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
        self, success: bool, detail: dict[str, Any] | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        now = time.monotonic() if now is None else float(now)
        if self.state != STATE_INTERACTING:
            return []
        if not success:
            return self._finish(False, detail or {}, now)
        return self._transition(STATE_VERIFYING, now)

    def on_graph_state(
        self, state: str, detail: dict[str, Any] | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        if self.state != STATE_VERIFYING or self.candidate is None:
            return []
        expected = str((self.candidate.get("interaction_command") or {}).get("expected_state") or "open")
        if str(state) != expected:
            return []
        return self._finish(True, detail or {"state": state}, now)

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
            return self._finish(True, detail or {"verified": True}, now)
        if retry:
            return self._transition(
                STATE_INTERACTING,
                now,
                {"kind": "interact", "candidate": self.candidate, "retry": True},
            )
        return self._finish(False, detail or {"reason": "verification_failed"}, now)

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
        return {
            "state": self.state,
            "candidate_id": "" if self.candidate is None else self.candidate.get("candidate_id", ""),
            "behavior_type": self._behavior_type(),
            "error": self.error,
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
