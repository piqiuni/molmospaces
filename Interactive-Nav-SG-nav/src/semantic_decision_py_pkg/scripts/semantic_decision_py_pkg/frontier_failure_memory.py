from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class FrontierFailureMemory:
    radius_m: float = 0.75
    cooldown_steps: int = 120
    failure_limit: int = 2
    entries: list[dict] = field(default_factory=list)

    def clear(self) -> None:
        self.entries.clear()

    def _matching(self, point: list, frame_id: str) -> list[dict]:
        if len(point) < 2 or not all(math.isfinite(float(value)) for value in point[:2]):
            return []
        return [entry for entry in self.entries
                if entry["frame_id"] == frame_id.lstrip("/")
                and math.dist(entry["point"], point[:2]) <= self.radius_m]

    def record_failure(self, point: list, frame_id: str, step: int) -> None:
        if len(point) < 2 or not all(math.isfinite(float(value)) for value in point[:2]):
            return
        matches = self._matching(point, frame_id)
        if matches:
            entry = min(matches, key=lambda item: math.dist(item["point"], point[:2]))
            entry["failures"] += 1
            entry["last_step"] = max(entry["last_step"], step)
        else:
            self.entries.append({"point": list(point[:2]), "frame_id": frame_id.lstrip("/"),
                                 "failures": 1, "last_step": step})

    def rejection(self, point: list, frame_id: str, step: int) -> str:
        matches = self._matching(point, frame_id)
        if any(entry["failures"] >= self.failure_limit for entry in matches):
            return "frontier_region_failed_requires_topology_change"
        if any(step < entry["last_step"] + self.cooldown_steps for entry in matches):
            return "frontier_region_failure_cooldown"
        return ""
