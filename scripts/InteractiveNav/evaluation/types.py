"""Small, serialisable types shared by the standalone V3 evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol


ActionKind = Literal["base", "interact", "stop"]


@dataclass
class PolicyObservation:
    """Information supplied to a non-oracle interactive-navigation policy.

    ``interactive_nav`` GT, target instance names, controlling joint names and
    oracle waypoints deliberately do not appear here.  The raw simulator
    observation remains available to adapters that need camera/state tensors.
    """

    observation: Any
    instruction: str
    step_index: int
    elapsed_seconds: float
    previous_action: dict[str, Any] | None


@dataclass
class PolicyAction:
    """A policy action accepted by the V3 evaluator.

    ``base`` carries a normal MolmoSpaces action dictionary, usually with a
    ``base`` entry.  ``interact`` identifies an observed object/joint rather
    than a V3 interaction id, so an evaluated policy need not know benchmark
    annotations.  The evaluator resolves and records the request.
    """

    kind: ActionKind
    base_action: dict[str, Any] | None = None
    object_name: str | None = None
    joint_index: int | None = None
    operation: Literal["open"] = "open"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InteractiveNavPolicy(Protocol):
    """Policy protocol for the standalone evaluator."""

    name: str
    uses_oracle_gt: bool

    def reset(self, episode_public: dict[str, Any]) -> None: ...

    def act(self, observation: PolicyObservation) -> PolicyAction: ...

    def close(self) -> None: ...


@dataclass
class InteractionRecord:
    requested_object_name: str | None
    requested_joint_index: int | None
    resolved_interaction_id: str | None
    success: bool
    joint_fraction_before: float | None
    joint_fraction_after: float | None
    executor: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    episode_index: int
    case_id: str
    house_index: int
    domains: list[str]
    interaction_requirement: str
    policy_name: str
    uses_oracle_gt: bool
    success: bool
    nav_success: bool
    required_interaction_success: bool
    sequence_success: bool
    non_interaction_success: bool | None
    terminal_reason: str
    step_count: int
    navigation_step_count: int
    interaction_action_count: int
    wrong_interaction_count: int
    navigation_path_length_m: float
    elapsed_seconds: float
    target_distance_m: float | None
    target_visibility_fraction: float | None
    interaction_records: list[dict[str, Any]]
    trace_path: str | None = None
    video_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
