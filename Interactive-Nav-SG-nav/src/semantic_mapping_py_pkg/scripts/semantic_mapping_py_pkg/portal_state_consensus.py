from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any


def _normalize_state(value: object) -> str:
    state = str(value or "unknown").strip().casefold()
    if state in {"opened", "static_open"}:
        return "open"
    if state in {"shut", "static_closed"}:
        return "closed"
    return state if state in {"open", "closed", "ajar", "blocked"} else "unknown"


def _pose3(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        pose = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    return pose if all(math.isfinite(item) for item in pose) else None


def _angle_distance(left: float, right: float) -> float:
    return abs((float(left) - float(right) + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class _PortalEvidenceState:
    stable_state: str = ""
    proposal_state: str = ""
    proposal_views: list[tuple[int, tuple[float, float, float]]] = field(default_factory=list)
    cooldown_until_step: int = -1


class PortalStateConsensus:
    """Confirm portal state only from separated, consecutive visual views.

    The interface deliberately exposes only request admission and completed
    observations.  Scheduling, evidence diversity, state-change hysteresis,
    and evaluator-step cooldown remain local to this module.
    """

    def __init__(
        self,
        *,
        confirmation_count: int = 3,
        cooldown_steps: int = 300,
        min_position_gap_m: float = 0.25,
        min_yaw_gap_rad: float = 0.25,
    ) -> None:
        self.confirmation_count = max(1, int(confirmation_count))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.min_position_gap_m = max(0.0, float(min_position_gap_m))
        self.min_yaw_gap_rad = max(0.0, float(min_yaw_gap_rad))
        self._states: dict[str, _PortalEvidenceState] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._states.clear()

    def clear(self, object_id: str) -> None:
        with self._lock:
            self._states.pop(str(object_id or ""), None)

    def record_authoritative(
        self,
        object_id: str,
        portal_state: object,
        *,
        capture_step: int | None,
    ) -> bool:
        """Latch a physical interaction result and start the normal cooldown."""

        key = str(object_id or "")
        stable_state = _normalize_state(portal_state)
        if not key or stable_state == "unknown" or capture_step is None:
            return False
        step = int(capture_step)
        with self._lock:
            state = self._states.setdefault(key, _PortalEvidenceState())
            state.stable_state = stable_state
            state.proposal_state = ""
            state.proposal_views.clear()
            state.cooldown_until_step = step + self.cooldown_steps
        return True

    def can_request(
        self,
        object_id: str,
        *,
        capture_step: int | None,
        observation_pose_xyyaw: object,
    ) -> tuple[bool, str]:
        key = str(object_id or "")
        pose = _pose3(observation_pose_xyyaw)
        if not key or capture_step is None or pose is None:
            return False, "missing_distinct_view_pose"
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return True, "initial_evidence"
            if state.stable_state and int(capture_step) < state.cooldown_until_step:
                return False, "stable_state_cooldown"
            if any(self._poses_overlap(pose, prior_pose) for _step, prior_pose in state.proposal_views):
                return False, "duplicate_view_pose"
            return True, "confirmation_needed"

    def cached_stable_result(
        self,
        object_id: str,
        *,
        capture_step: int | None,
    ) -> dict[str, Any] | None:
        """Return the authoritative cached state while its cooldown is active.

        A targeted executor refresh can race with the third ordinary M1 vote:
        the decision was created while the portal was unknown, but by the time
        the fresh frame arrives consensus is already stable.  That is satisfied
        evidence, not an observation failure, so expose it without another
        model request.
        """

        key = str(object_id or "")
        if not key or capture_step is None:
            return None
        step = int(capture_step)
        with self._lock:
            state = self._states.get(key)
            if (
                state is None
                or not state.stable_state
                or step >= state.cooldown_until_step
            ):
                return None
            return self._result_from_state(
                state,
                accepted=True,
                reason="cached_stable_state_cooldown",
                observed_state=state.stable_state,
            )

    def observe(
        self,
        object_id: str,
        portal_state: object,
        *,
        capture_step: int | None,
        observation_pose_xyyaw: object,
    ) -> dict[str, Any]:
        key = str(object_id or "")
        observed_state = _normalize_state(portal_state)
        pose = _pose3(observation_pose_xyyaw)
        if not key or capture_step is None or pose is None:
            return self._result(
                accepted=False,
                reason="missing_distinct_view_pose",
                observed_state=observed_state,
            )
        step = int(capture_step)
        with self._lock:
            state = self._states.setdefault(key, _PortalEvidenceState())
            if state.stable_state and step < state.cooldown_until_step:
                return self._result_from_state(
                    state,
                    accepted=False,
                    reason="stable_state_cooldown",
                    observed_state=observed_state,
                )
            if observed_state == "unknown":
                state.proposal_state = ""
                state.proposal_views.clear()
                return self._result_from_state(
                    state,
                    accepted=False,
                    reason="unknown_state_not_confirmable",
                    observed_state=observed_state,
                )
            if state.stable_state and observed_state == state.stable_state:
                state.proposal_state = ""
                state.proposal_views.clear()
                state.cooldown_until_step = step + self.cooldown_steps
                return self._result_from_state(
                    state,
                    accepted=True,
                    reason="stable_state_reconfirmed",
                    observed_state=observed_state,
                )
            if observed_state != state.proposal_state:
                state.proposal_state = observed_state
                state.proposal_views = []
            if any(self._poses_overlap(pose, prior_pose) for _step, prior_pose in state.proposal_views):
                return self._result_from_state(
                    state,
                    accepted=False,
                    reason="duplicate_view_pose",
                    observed_state=observed_state,
                )
            state.proposal_views.append((step, pose))
            if len(state.proposal_views) < self.confirmation_count:
                return self._result_from_state(
                    state,
                    accepted=False,
                    reason="confirmation_pending",
                    observed_state=observed_state,
                )
            had_stable_state = bool(state.stable_state)
            state.stable_state = observed_state
            state.proposal_state = ""
            state.proposal_views.clear()
            state.cooldown_until_step = step + self.cooldown_steps
            return self._result_from_state(
                state,
                accepted=True,
                reason=(
                    "state_change_confirmed"
                    if had_stable_state
                    else "initial_state_confirmed"
                ),
                observed_state=observed_state,
            )

    def _poses_overlap(
        self,
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> bool:
        position_gap = math.hypot(left[0] - right[0], left[1] - right[1])
        yaw_gap = _angle_distance(left[2], right[2])
        return bool(
            position_gap < self.min_position_gap_m
            and yaw_gap < self.min_yaw_gap_rad
        )

    def _result_from_state(
        self,
        state: _PortalEvidenceState,
        *,
        accepted: bool,
        reason: str,
        observed_state: str,
    ) -> dict[str, Any]:
        return self._result(
            accepted=accepted,
            reason=reason,
            observed_state=observed_state,
            stable_state=state.stable_state,
            proposal_state=state.proposal_state,
            confirmation_count=len(state.proposal_views),
            cooldown_until_step=state.cooldown_until_step,
        )

    def _result(
        self,
        *,
        accepted: bool,
        reason: str,
        observed_state: str,
        stable_state: str = "",
        proposal_state: str = "",
        confirmation_count: int = 0,
        cooldown_until_step: int = -1,
    ) -> dict[str, Any]:
        return {
            "accepted": bool(accepted),
            "reason": str(reason),
            "observed_state": str(observed_state),
            "stable_state": str(stable_state),
            "proposal_state": str(proposal_state),
            "confirmation_count": int(confirmation_count),
            "confirmation_required": int(self.confirmation_count),
            "cooldown_until_step": int(cooldown_until_step),
        }
