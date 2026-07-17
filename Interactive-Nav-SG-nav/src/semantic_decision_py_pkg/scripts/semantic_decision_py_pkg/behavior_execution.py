from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .behavior_candidates import BEHAVIOR_EXPLORE, BEHAVIOR_INTERACT, BEHAVIOR_NAVIGATE


STATE_IDLE = "IDLE"
STATE_NAVIGATING = "NAVIGATING"
STATE_APPROACH_INTERACTION = "APPROACH_INTERACTION"
STATE_INTERACTING = "INTERACTING"
STATE_VERIFYING = "VERIFYING"
STATE_SUCCEEDED = "SUCCEEDED"
STATE_FAILED = "FAILED"


@dataclass
class ExecutionConfig:
    navigation_timeout_s: float = 180.0
    interaction_timeout_s: float = 30.0
    verification_timeout_s: float = 30.0


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
                STATE_NAVIGATING,
                now,
                {"kind": "explore_frontier", "candidate": self.candidate},
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

    def on_explore_result(
        self, success: bool, detail: dict[str, Any] | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        if self.state != STATE_NAVIGATING or self._behavior_type() != BEHAVIOR_EXPLORE:
            return []
        return self._finish(success, detail or {}, now)

    def on_navigation_result(
        self, success: bool, detail: dict[str, Any] | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        now = time.monotonic() if now is None else float(now)
        if self.state == STATE_NAVIGATING and self._behavior_type() == BEHAVIOR_NAVIGATE:
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

    def timeout_reason(self, now: float | None = None) -> str:
        if self.state in {STATE_IDLE, STATE_SUCCEEDED, STATE_FAILED}:
            return ""
        now = time.monotonic() if now is None else float(now)
        elapsed = now - self.state_started_at
        if self.state in {STATE_NAVIGATING, STATE_APPROACH_INTERACTION}:
            return "navigation_timeout" if elapsed > self.config.navigation_timeout_s else ""
        if self.state == STATE_INTERACTING:
            return "interaction_timeout" if elapsed > self.config.interaction_timeout_s else ""
        if self.state == STATE_VERIFYING:
            return "verification_timeout" if elapsed > self.config.verification_timeout_s else ""
        return ""

    def fail_timeout(self, reason: str, now: float | None = None) -> list[dict[str, Any]]:
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
