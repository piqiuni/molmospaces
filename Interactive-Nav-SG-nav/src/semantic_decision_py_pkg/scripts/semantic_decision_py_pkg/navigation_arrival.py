"""Goal-local arrival state, shared by active and terminal pose checks."""

from dataclasses import dataclass
import math


def path_deviation_m(pose, points) -> float:
    """Distance to a polyline, not just to its sampled vertices."""
    if pose is None or len(points) < 2:
        return 0.0
    best = float("inf")
    for left, right in zip(points, points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length_sq = dx * dx + dy * dy
        t = min(1.0, max(0.0, (
            (pose[0] - left[0]) * dx + (pose[1] - left[1]) * dy
        ) / length_sq)) if length_sq > 0.0 else 0.0
        best = min(best, math.hypot(
            pose[0] - left[0] - t * dx, pose[1] - left[1] - t * dy
        ))
    return best


@dataclass
class PositionToleranceLatch:
    goal: tuple[float, float, float]
    distance_tolerance_m: float
    latched: bool = False
    arrival_pose: tuple[float, float, float] | None = None
    arrival_step: int | None = None

    def observe(self, pose, step: int | None = None) -> bool:
        if self.latched:
            return True
        if pose is None or len(pose) < 3:
            return False
        values = tuple(float(value) for value in pose[:3])
        if not all(math.isfinite(value) for value in values):
            return False
        if math.hypot(values[0] - self.goal[0], values[1] - self.goal[1]) <= self.distance_tolerance_m:
            self.latched = True
            self.arrival_pose = values
            self.arrival_step = step
        return self.latched

    def apply(self, validation: dict) -> dict:
        result = dict(validation)
        if not self.latched or list(result.get("expected_pose_xyyaw") or []) != list(self.goal):
            return result
        result.update(
            position_tolerance_latched=True,
            position_latch_pose_xyyaw=list(self.arrival_pose),
            position_latch_step_index=self.arrival_step,
        )
        result["valid"] = bool(
            result.get("checked")
            and math.isfinite(float(result.get("yaw_error_rad", math.inf)))
            and float(result["yaw_error_rad"]) <= float(result.get("yaw_tolerance_rad", 0.0))
        )
        if result["valid"]:
            result["reason"] = "position_latched_yaw_ready"
        return result
