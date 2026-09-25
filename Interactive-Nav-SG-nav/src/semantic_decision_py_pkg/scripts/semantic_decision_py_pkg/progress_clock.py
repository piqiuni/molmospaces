"""Deployment-neutral progress clocks; capture identity is never overwritten."""

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class ProgressClock:
    mode: str = "capture_step"
    period_s: float = 0.2

    def __post_init__(self):
        if self.mode not in {"capture_step", "monotonic"}:
            raise ValueError(f"Unknown progress clock: {self.mode}")
        if not math.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("Progress clock period must be finite and positive")

    def context(self, capture_step, *, now=None):
        tick = capture_step
        if self.mode == "monotonic":
            tick = int((time.monotonic() if now is None else now) / self.period_s)
        return {
            "observation_step": tick,
            "observation_step_clock": self.mode,
            "progress_clock_period_s": self.period_s if self.mode == "monotonic" else None,
            "source_capture_step": capture_step,
        }
