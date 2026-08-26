from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PortalJointState:
    closed_references: dict[str, float] = field(default_factory=dict)


class PortalStateTracker:
    """Infer portal open/closed state from per-episode joint readback.

    The first valid observation of each non-handle joint is treated as the
    episode's closed reference. Interactive door tasks are expected to expose
    the door before executing the first interaction and to initialize that door
    in its closed state.
    """

    def __init__(self, closed_threshold=0.10, open_threshold=0.67):
        self.closed_threshold = max(0.0, float(closed_threshold))
        self.open_threshold = min(1.0, max(self.closed_threshold, float(open_threshold)))
        self._states: dict[str, PortalJointState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(self, node_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        joint_infos = self._usable_joint_infos(observation)
        if not joint_infos:
            return {
                "state": "unknown",
                "open_fraction": None,
                "joint_open_fractions": {},
                "joint_closed_references": {},
                "state_source": "joint_readback_missing",
            }

        runtime = self._states.setdefault(str(node_id), PortalJointState())
        fractions: dict[str, float] = {}
        valid_joint_count = 0
        for index, info in enumerate(joint_infos):
            joint_name = str(info.get("joint_name") or f"joint_{index:02d}")
            joint_range = list(info.get("joint_range") or [])
            joint_value = info.get("joint_value")
            if len(joint_range) < 2 or joint_value is None:
                continue
            joint_min = float(joint_range[0])
            joint_max = float(joint_range[1])
            current = float(joint_value)
            reference = runtime.closed_references.setdefault(joint_name, current)
            travel = max(abs(joint_min - reference), abs(joint_max - reference))
            if travel <= 1e-6:
                continue
            fraction = min(1.0, max(0.0, abs(current - reference) / travel))
            fractions[joint_name] = fraction
            valid_joint_count += 1

        if valid_joint_count == 0:
            return {
                "state": "unknown",
                "open_fraction": None,
                "joint_open_fractions": {},
                "joint_closed_references": dict(runtime.closed_references),
                "state_source": "joint_readback_invalid",
            }

        values = list(fractions.values())
        if max(values) <= self.closed_threshold:
            state = "closed"
        elif min(values) >= self.open_threshold:
            state = "open"
        else:
            state = "ajar"
        return {
            "state": state,
            # A root portal is only as open as its least-open required leaf.
            "open_fraction": min(values),
            "joint_open_fractions": fractions,
            "joint_closed_references": dict(runtime.closed_references),
            "state_source": "joint_readback_from_episode_closed_reference",
        }

    @staticmethod
    def _usable_joint_infos(observation: dict[str, Any]) -> list[dict[str, Any]]:
        infos = list(observation.get("joint_infos") or [])
        if not infos and observation.get("joint_value") is not None:
            infos = [
                {
                    "joint_name": observation.get("primary_joint_name") or "primary_joint",
                    "joint_type": observation.get("joint_type"),
                    "joint_range": observation.get("joint_range") or [0.0, 0.0],
                    "joint_value": observation.get("joint_value"),
                }
            ]
        result = []
        for info in infos:
            joint_name = str(info.get("joint_name") or "")
            if "handle" in joint_name.lower():
                continue
            joint_type = str(info.get("joint_type") or "").lower()
            if joint_type and "hinge" not in joint_type and "slide" not in joint_type:
                continue
            result.append(dict(info))
        return result
