"""Transport-free, bounded left/right observation after a portal traversal.

An executor owns each sweep and supplies time, pose and cancellation on every
tick. The sweep only returns intents; it cannot publish motion or start threads.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class ExitObservationConfig:
    angle_rad: float = math.pi / 2.0
    speed_rad_s: float = 0.30
    tolerance_rad: float = 0.12
    settle_s: float = 0.5
    view_timeout_s: float = 6.0
    return_home: bool = True
    poll_interval_s: float = 0.05

    def __post_init__(self):
        positive = (self.angle_rad, self.speed_rad_s, self.tolerance_rad,
                    self.view_timeout_s, self.poll_interval_s)
        if not all(math.isfinite(v) and v > 0.0 for v in positive):
            raise ValueError("Observation angles, speeds and time limits must be finite and positive")
        if not math.isfinite(self.settle_s) or self.settle_s < 0.0:
            raise ValueError("Observation settle time must be finite and nonnegative")


@dataclass(frozen=True)
class ObservationStep:
    angular_z: float
    wait_s: float
    phase: str
    result: dict | None = None


class ExitObservationSweep:
    def __init__(self, home_yaw: float, config: ExitObservationConfig, *, now: float):
        if not math.isfinite(home_yaw) or not math.isfinite(now):
            raise ValueError("Observation needs a finite initial heading and clock")
        self.config = config
        self.targets = (_angle(home_yaw - config.angle_rad),
                        _angle(home_yaw + config.angle_rad))
        if config.return_home:
            self.targets += (_angle(home_yaw),)
        self.started_at = now
        self.deadline = now + config.view_timeout_s
        self.index = 0
        self.phase = "rotating"
        self.settle_until = now
        self.views = []
        self._result = None

    def _finish(self, status, now, **detail):
        self.phase = status
        self._result = {"enabled": True, "status": status, **detail}
        if status != "canceled":
            self._result["elapsed_s"] = max(0.0, now - self.started_at)
        return self._terminal_step()

    def _terminal_step(self):
        return ObservationStep(0.0, 0.0, self.phase,
                               {**self._result, "views": [dict(v) for v in self.views]})

    def advance(self, *, now: float, yaw: float | None, current=True, shutdown=False):
        if self._result is not None:
            return self._terminal_step()
        if not current:
            return self._finish("canceled", now)
        if shutdown:
            return self._finish("canceled", now, reason="shutdown")
        if self.phase == "settling":
            remaining = self.settle_until - now
            if remaining > 0.0:
                return ObservationStep(0.0, min(self.config.poll_interval_s, remaining), self.phase)
            self.index += 1
            if self.index == len(self.targets):
                return self._finish("completed", now)
            self.phase = "rotating"
            self.deadline = now + self.config.view_timeout_s
        if now >= self.deadline:
            return self._finish("timeout", now, failed_view_index=self.index)
        # A missing TF must not leave the previous nonzero rotation latched.
        if yaw is None or not math.isfinite(yaw):
            return ObservationStep(0.0, self.config.poll_interval_s, self.phase)
        target = self.targets[self.index]
        error = _angle(target - yaw)
        if abs(error) <= self.config.tolerance_rad:
            self.views.append({"index": self.index, "target_yaw": target, "yaw": yaw})
            self.phase = "settling"
            self.settle_until = now + self.config.settle_s
            return ObservationStep(0.0, min(self.config.poll_interval_s, self.config.settle_s), self.phase)
        speed = min(abs(self.config.speed_rad_s), max(0.05, abs(error) * 1.5))
        return ObservationStep(speed if error > 0.0 else -speed,
                               self.config.poll_interval_s, self.phase)
