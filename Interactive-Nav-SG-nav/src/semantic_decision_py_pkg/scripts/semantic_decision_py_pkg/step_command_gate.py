"""Step-indexed command admission for simulator-synchronous behaviors.

The ROS bridge publishes RGB and a fresh-command window for evaluator step
``N`` independently.  ROS callback scheduling may deliver them in either
order, so an executor must retain both events until it can emit *one* command
for their shared step.  The following ``step_sync`` event acknowledges that
the action window has closed before a later command is admitted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class StepCommandAck:
    """Result of one command window after the bridge closed that action."""

    step_index: int
    action_source: str
    exact_step_sync: bool

    @property
    def command_applied(self) -> bool:
        return self.action_source == "cmd_vel"


class StepCommandGate:
    """Pair RGB/gate events by evaluator step and admit one command at a time.

    This object deliberately has no ROS dependency so callback reordering,
    duplicate delivery, and acknowledgement behavior are unit-testable.  A
    caller records all three streams and repeatedly calls :meth:`consume_step`;
    it receives an eligible step only when RGB and fresh-gate for that *same*
    step are pending and the previously emitted command has been acknowledged.
    """

    def __init__(
        self,
        *,
        max_pending_steps: int = 32,
        max_pair_age_s: float = 1.0,
    ) -> None:
        self.max_pending_steps = max(1, int(max_pending_steps))
        self.max_pair_age_s = max(0.0, float(max_pair_age_s))
        self.reset()

    def reset(self) -> None:
        self._rgb_at: dict[int, float] = {}
        self._gate_at: dict[int, float] = {}
        self._last_sent_step: int | None = None
        self._awaiting_ack_step: int | None = None
        self._last_sync_step: int | None = None
        self._acks: deque[StepCommandAck] = deque()
        self._dropped_stale_pairs = 0
        self._dropped_overflow_events = 0
        self._sequence_resets = 0
        self._inexact_sync_acks = 0

    @staticmethod
    def _now(now: float | None) -> float:
        return time.monotonic() if now is None else float(now)

    def record_rgb(self, step_index: int, *, now: float | None = None) -> None:
        self._record(self._rgb_at, step_index, self._now(now))

    def record_fresh_gate(self, step_index: int, *, now: float | None = None) -> None:
        self._record(self._gate_at, step_index, self._now(now))

    def record_step_sync(
        self,
        step_index: int,
        *,
        action_source: str = "",
    ) -> None:
        step = int(step_index)
        if self._last_sync_step is not None and step < self._last_sync_step:
            # A new evaluator episode must call reset().  Do not let an old
            # acknowledgement authorize a command in the new sequence.
            self._sequence_resets += 1
            return
        self._last_sync_step = step
        waiting = self._awaiting_ack_step
        if waiting is None or step < waiting:
            return
        exact = step == waiting
        if not exact:
            self._inexact_sync_acks += 1
        self._acks.append(
            StepCommandAck(
                step_index=waiting,
                action_source=str(action_source or ""),
                exact_step_sync=exact,
            )
        )
        self._awaiting_ack_step = None

    def consume_step(self, *, now: float | None = None) -> int | None:
        """Atomically consume the next RGB/fresh-gate pair, if one is valid."""

        if self._awaiting_ack_step is not None:
            return None
        current_time = self._now(now)
        self._discard_stale(current_time)
        common_steps = sorted(set(self._rgb_at).intersection(self._gate_at))
        if not common_steps:
            return None
        for step in common_steps:
            if self._last_sent_step is not None and step <= self._last_sent_step:
                self._rgb_at.pop(step, None)
                self._gate_at.pop(step, None)
                self._dropped_stale_pairs += 1
                continue
            self._rgb_at.pop(step, None)
            self._gate_at.pop(step, None)
            self._last_sent_step = step
            self._awaiting_ack_step = step
            return step
        return None

    def take_acks(self) -> list[StepCommandAck]:
        result = list(self._acks)
        self._acks.clear()
        return result

    def diagnostics(self) -> dict[str, object]:
        return {
            "pending_rgb_steps": sorted(self._rgb_at),
            "pending_fresh_gate_steps": sorted(self._gate_at),
            "last_sent_step": self._last_sent_step,
            "awaiting_ack_step": self._awaiting_ack_step,
            "last_step_sync": self._last_sync_step,
            "dropped_stale_pairs": self._dropped_stale_pairs,
            "dropped_overflow_events": self._dropped_overflow_events,
            "sequence_resets": self._sequence_resets,
            "inexact_sync_acks": self._inexact_sync_acks,
        }

    def _record(self, values: dict[int, float], step_index: int, now: float) -> None:
        step = int(step_index)
        if self._last_sent_step is not None and step < self._last_sent_step:
            self._sequence_resets += 1
            return
        values[step] = now
        self._trim(values)

    def _trim(self, values: dict[int, float]) -> None:
        while len(values) > self.max_pending_steps:
            oldest = min(values)
            values.pop(oldest, None)
            self._dropped_overflow_events += 1

    def _discard_stale(self, now: float) -> None:
        if self.max_pair_age_s <= 0.0:
            return
        cutoff = now - self.max_pair_age_s
        for step in sorted(set(self._rgb_at).intersection(self._gate_at)):
            if max(self._rgb_at[step], self._gate_at[step]) >= cutoff:
                continue
            self._rgb_at.pop(step, None)
            self._gate_at.pop(step, None)
            self._dropped_stale_pairs += 1
