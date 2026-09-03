"""Physical-platform velocity scaling and slew limiting.

This module is ROS-independent so the safety policy can be tested without a
running robot.  It is intentionally the last command filter before the
physical WebSocket bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Tuple


@dataclass(frozen=True)
class VelocitySafetyConfig:
    scale: float = 0.30
    # Match the ROS node's runtime fallback.  The previous 0.15 default was
    # lower than min_linear_mps and made the standalone safety limiter invalid.
    max_linear_mps: float = 0.50
    min_linear_mps: float = 0.30
    # Ignore small DWA residuals near a position goal.  Without a linear
    # deadband, a 0.01 m/s numerical correction is promoted to the configured
    # minimum physical speed and prevents the base from settling before its
    # terminal heading rotation.
    linear_deadband_mps: float = 0.05
    max_angular_rps: float = 1.20
    # Keep enough authority to turn the Go2 while still allowing fine final
    # alignment. 0.70 rad/s quantized every small DWA correction into a large
    # turn and caused repeated yaw overshoot near an interaction goal.
    min_angular_rps: float = 0.20
    angular_deadband_rps: float = 0.05
    max_linear_accel_mps2: float = 0.35
    max_angular_accel_rps2: float = 2.00
    stale_after_s: float = 0.50


class VelocitySafetyLimiter:
    """Scale, clamp and slew-limit ``(vx, vy, wz)`` commands.

    ``stop()`` is an immediate zero and intentionally bypasses the slew
    limiter.  This makes an operator stop, source timeout, or WebSocket
    disconnect fail safe.
    """

    def __init__(self, config: VelocitySafetyConfig | None = None) -> None:
        self.config = config or VelocitySafetyConfig()
        if not 0.0 < self.config.scale <= 1.0:
            raise ValueError("velocity scale must be in (0, 1]")
        if self.config.max_linear_mps <= 0 or self.config.max_angular_rps <= 0:
            raise ValueError("velocity limits must be positive")
        if self.config.min_linear_mps < 0 or self.config.min_linear_mps > self.config.max_linear_mps:
            raise ValueError("minimum linear speed must satisfy 0 <= min <= max")
        if self.config.linear_deadband_mps < 0 or self.config.linear_deadband_mps >= self.config.max_linear_mps:
            raise ValueError("linear deadband must satisfy 0 <= deadband < max")
        if self.config.min_angular_rps < 0 or self.config.min_angular_rps > self.config.max_angular_rps:
            raise ValueError("minimum angular speed must satisfy 0 <= min <= max")
        if self.config.angular_deadband_rps < 0 or self.config.angular_deadband_rps >= self.config.max_angular_rps:
            raise ValueError("angular deadband must satisfy 0 <= deadband < max")
        self._last = (0.0, 0.0, 0.0)
        self._last_at: float | None = None

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, float(value)))

    @staticmethod
    def _slew(value: float, previous: float, delta: float) -> float:
        return max(previous - delta, min(previous + delta, float(value)))

    def reset(self) -> Tuple[float, float, float]:
        self._last = (0.0, 0.0, 0.0)
        self._last_at = None
        return self._last

    def stop(self) -> Tuple[float, float, float]:
        return self.reset()

    def limit(
        self,
        vx: float,
        vy: float,
        wz: float,
        *,
        now: float | None = None,
    ) -> Tuple[float, float, float]:
        now = time.monotonic() if now is None else float(now)
        values = tuple(float(v) for v in (vx, vy, wz))
        if not all(math.isfinite(v) for v in values):
            return self.stop()
        scale = self.config.scale
        linear = self._clamp(values[0] * scale, self.config.max_linear_mps)
        if abs(linear) <= self.config.linear_deadband_mps:
            linear = 0.0
        elif abs(linear) < self.config.min_linear_mps:
            linear = math.copysign(self.config.min_linear_mps, linear)
        angular = self._clamp(values[2] * scale, self.config.max_angular_rps)
        # DWA/teleop often leaves a tiny yaw residue during straight motion.
        # Do not turn that residue into the configured minimum turn speed.
        if abs(angular) <= self.config.angular_deadband_rps:
            angular = 0.0
        elif abs(angular) < self.config.min_angular_rps:
            angular = math.copysign(self.config.min_angular_rps, angular)
        requested = (
            linear,
            0.0,  # Go2 physical base is currently configured for no lateral motion.
            angular,
        )
        if self._last_at is None:
            result = requested
        else:
            dt = now - self._last_at
            if dt <= 0.0 or dt > self.config.stale_after_s:
                result = requested
            else:
                linear_delta = self.config.max_linear_accel_mps2 * dt
                angular_delta = self.config.max_angular_accel_rps2 * dt
                result = (
                    self._slew(requested[0], self._last[0], linear_delta),
                    0.0,
                    self._slew(requested[2], self._last[2], angular_delta),
                )
        self._last = result
        self._last_at = now
        return result
