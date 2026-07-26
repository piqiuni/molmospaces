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


def is_stuck_recovery_failure(detail: dict[str, Any]) -> bool:
    reason = str(detail.get("reason") or "").lower()
    status = str(detail.get("status") or "").lower()
    return reason == "navigation_stagnation" or "oscillat" in status


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
    reference_xy: tuple[float, float] | None = None
    last_progress_at: float | None = None

    def reset(self, pose_xy: tuple[float, float] | None, now: float) -> None:
        self.reference_xy = pose_xy
        self.last_progress_at = float(now) if pose_xy is not None else None

    def observe(self, pose_xy: tuple[float, float] | None, now: float) -> bool:
        if self.timeout_s <= 0.0 or pose_xy is None:
            return False
        if self.reference_xy is None or self.last_progress_at is None:
            self.reset(pose_xy, now)
            return False
        displacement = math.hypot(
            float(pose_xy[0]) - float(self.reference_xy[0]),
            float(pose_xy[1]) - float(self.reference_xy[1]),
        )
        if displacement >= self.min_displacement_m:
            self.reset(pose_xy, now)
            return False
        return float(now) - self.last_progress_at >= self.timeout_s


@dataclass
class ExecutionConfig:
    navigation_timeout_s: float = 180.0
    interaction_navigation_timeout_s: float = 45.0
    interaction_timeout_s: float = 30.0
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
        self, success: bool, detail: dict[str, Any] | None = None, now: float | None = None
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
        return self._transition(
            STATE_INTERACTING,
            now,
            {"kind": "interact", "candidate": self.candidate},
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
