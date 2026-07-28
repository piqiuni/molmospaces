from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from scripts.InteractiveNav import container_scene_probe as probe


@dataclass(frozen=True)
class InteractionExecutionRequest:
    domain: str
    object_name: str
    joint_name: str
    joint_index: int
    target_fraction: float = 1.0
    max_steps: int = 1500
    tolerance: float = 1e-3
    interaction_id: str | None = None


@dataclass
class InteractionExecutionResult:
    executor: str
    success: bool
    request: InteractionExecutionRequest
    metadata: dict[str, Any] = field(default_factory=dict)


class InteractionExecutor(Protocol):
    name: str

    def execute(self, env, request: InteractionExecutionRequest) -> InteractionExecutionResult: ...


class ForceInteractionExecutor:
    name = "force"

    def execute(self, env, request: InteractionExecutionRequest) -> InteractionExecutionResult:
        if request.domain not in {"channel", "container"}:
            raise ValueError(f"Force executor does not support domain={request.domain}")
        drive = probe.drive_joint_to_value_with_force(
            env,
            request.joint_name,
            _target_value(env, request.joint_name, request.target_fraction),
            max_steps=request.max_steps,
            tolerance=request.tolerance,
        )
        return InteractionExecutionResult(
            executor=self.name,
            success=bool(drive["reached"]),
            request=request,
            metadata={"drive": drive},
        )


def _target_value(env, joint_name: str, fraction: float) -> float:
    import mujoco

    joint_id = mujoco.mj_name2id(
        env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    lo, hi = [float(value) for value in env.current_model.jnt_range[joint_id]]
    closed, opened = probe.joint_closed_open_values([lo, hi])
    return closed + float(fraction) * (opened - closed)


EXECUTOR_REGISTRY: dict[str, type[InteractionExecutor]] = {
    "force": ForceInteractionExecutor,
}


def build_interaction_executor(name: str) -> InteractionExecutor:
    try:
        return EXECUTOR_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown interaction executor {name!r}; available={sorted(EXECUTOR_REGISTRY)}"
        ) from exc
