"""Built-in policies and adapters for the standalone V3 evaluator."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .types import InteractiveNavPolicy, PolicyAction, PolicyObservation


class NoOpPolicy:
    """Deterministic replay baseline that never moves or interacts."""

    name = "noop"
    uses_oracle_gt = False

    def reset(self, episode_public: dict[str, Any]) -> None:
        del episode_public

    def act(self, observation: PolicyObservation) -> PolicyAction:
        del observation
        return PolicyAction(kind="stop", metadata={"reason": "noop"})

    def close(self) -> None:
        return None


class ScriptedOraclePolicy:
    """GT-only upper-bound baseline.

    It is intentionally separated from learned/online policies.  The runner
    accepts ``oracle_interaction_id`` only from adapters declaring
    ``uses_oracle_gt=True`` and writes that flag to every result.
    """

    name = "scripted_oracle"
    uses_oracle_gt = True

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []
        self._cursor = 0
        self._active_navigation: dict[str, Any] | None = None

    def reset(self, episode_public: dict[str, Any]) -> None:
        self._steps = copy.deepcopy(episode_public.get("_oracle_steps", []))
        self._cursor = 0
        self._active_navigation = None

    def act(self, observation: PolicyObservation) -> PolicyAction:
        if self._cursor >= len(self._steps):
            return PolicyAction(kind="stop", metadata={"reason": "oracle_complete"})
        step = self._steps[self._cursor]
        kind = step.get("type")
        if kind == "open_joint":
            self._cursor += 1
            return PolicyAction(
                kind="interact",
                object_name=str(step["object_name"]),
                joint_index=int(step["joint_index"]),
                metadata={"oracle_interaction_id": step["interaction_id"]},
            )
        if kind == "navigate":
            # A V3 oracle waypoint is a goal condition, not one simulator
            # action.  Keep emitting the same pose target until the runner's
            # live pose readback satisfies the recorded tolerance.
            self._active_navigation = step
            return PolicyAction(
                kind="base",
                metadata={
                    "oracle_goal_point": list(step["goal_point"]),
                    "oracle_goal_yaw": float(step.get("goal_yaw", 0.0)),
                    "oracle_position_tolerance_m": float(step.get("position_tolerance_m", 0.25)),
                    "oracle_yaw_tolerance_rad": float(step.get("yaw_tolerance_rad", 0.35)),
                    "oracle_navigation_cursor": self._cursor,
                },
            )
        # Observation is evaluator-owned and does not require an environment action.
        self._cursor += 1
        return PolicyAction(kind="base", base_action={"base": np.zeros(3)}, metadata={"oracle_observe": True})

    def notify_action_result(self, action: PolicyAction, *, base_pose: np.ndarray | None = None) -> None:
        """Advance a navigation plan entry only after live pose convergence.

        This optional callback is evaluator-specific; ordinary policies never
        need it.  The policy intentionally receives only its own commanded
        goal/result here, not V3 supervision.
        """

        if action.kind != "base" or self._active_navigation is None or base_pose is None:
            return
        step = self._active_navigation
        target = np.asarray(step["goal_point"], dtype=float)
        pose = np.asarray(base_pose, dtype=float)
        position_error = float(np.linalg.norm(pose[:2] - target[:2]))
        yaw_error = float(np.arctan2(
            np.sin(float(step.get("goal_yaw", 0.0)) - float(pose[2])),
            np.cos(float(step.get("goal_yaw", 0.0)) - float(pose[2])),
        ))
        if (
            position_error <= float(step.get("position_tolerance_m", 0.25))
            and abs(yaw_error) <= float(step.get("yaw_tolerance_rad", 0.35))
        ):
            self._cursor += 1
            self._active_navigation = None

    def close(self) -> None:
        return None


class MolmoSpacesPolicyAdapter:
    """Adapt a regular ``BasePolicy`` that emits MolmoSpaces action dicts."""

    uses_oracle_gt = False

    def __init__(self, policy: Any, *, name: str | None = None) -> None:
        self.policy = policy
        self.name = name or type(policy).__name__

    def reset(self, episode_public: dict[str, Any]) -> None:
        del episode_public
        self.policy.reset()

    def act(self, observation: PolicyObservation) -> PolicyAction:
        action = self.policy.get_action(observation.observation)
        if action is None:
            return PolicyAction(kind="stop", metadata={"reason": "wrapped_policy_returned_none"})
        if not isinstance(action, dict):
            raise TypeError(f"Wrapped policy returned {type(action).__name__}, expected dict")
        if bool(action.get("done", False)):
            return PolicyAction(kind="stop", metadata={"wrapped_action": action})
        # The independent evaluator extends the normal MolmoSpaces base-action
        # transport with a small, explicit interaction request.  RosBridgePolicy
        # preserves unknown JSON keys, so a ROS policy may publish either:
        # {"action": {"kind": "interact", "object_name": ..., "joint_index": 0}}
        # or the action object itself.  No V3 interaction id is accepted from
        # non-oracle policies.
        if action.get("kind") == "interact":
            object_name = action.get("object_name")
            joint_index = action.get("joint_index")
            if not isinstance(object_name, str) or joint_index is None:
                raise ValueError(
                    "interactive policy action requires string object_name and joint_index"
                )
            return PolicyAction(
                kind="interact",
                object_name=object_name,
                joint_index=int(joint_index),
                operation=str(action.get("operation", "open")),
                metadata={"wrapped_action": action},
            )
        return PolicyAction(kind="base", base_action=action)

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


def build_builtin_policy(name: str) -> InteractiveNavPolicy:
    if name == "noop":
        return NoOpPolicy()
    if name == "scripted_oracle":
        return ScriptedOraclePolicy()
    raise ValueError(f"Unknown built-in policy {name!r}")
