from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import time
from typing import Any


_STATE_BY_ACTION = {
    "open": "open",
    "close": "closed",
}


def uses_direct_atomic_outcome_beliefs(
    module3: object, configured: object
) -> bool:
    """Return whether the evaluator-only outcome closure is enabled."""

    return bool(configured) and str(module3 or "").casefold() == "direct_atomic"


def expected_interaction_state(candidate: dict[str, Any]) -> str:
    """Return the state this interaction command claims to establish.

    This deliberately uses only the candidate's public object identifier and
    its own requested action.  It never reads a simulator articulation or
    object-state field.
    """

    interaction = candidate.get("interaction_command") or {}
    expected = str(interaction.get("expected_state") or "").strip().casefold()
    if expected in {"open", "closed"}:
        return expected
    return _STATE_BY_ACTION.get(
        str(interaction.get("action") or "").strip().casefold(), ""
    )


def interaction_target_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    """Return all opaque IDs by which a candidate may name its object."""

    interaction = candidate.get("interaction_command") or {}
    values = (
        candidate.get("target_id"),
        interaction.get("node_id"),
        interaction.get("object_id"),
    )
    unique: list[str] = []
    for value in values:
        identifier = str(value or "")
        if identifier and identifier not in unique:
            unique.append(identifier)
    return tuple(unique)


@dataclass(frozen=True)
class InteractionOutcomeBelief:
    """A local state belief inferred from a successful requested command.

    ``source`` is intentionally explicit so a consumer cannot mistake this for
    visual verification or a private MuJoCo state readback.
    """

    target_id: str
    state: str
    action: str
    source: str
    decision_id: str
    candidate_id: str
    timestamp: float


class InteractionOutcomeBeliefStore:
    """Episode-local command-outcome beliefs for evaluator direct-atomic mode."""

    def __init__(self) -> None:
        self._beliefs: dict[str, InteractionOutcomeBelief] = {}

    def clear(self) -> None:
        self._beliefs.clear()

    def record_success(
        self, candidate: dict[str, Any], now: float | None = None
    ) -> InteractionOutcomeBelief | None:
        """Record an expected state only after a successful requested action."""

        state = expected_interaction_state(candidate)
        target_ids = interaction_target_ids(candidate)
        if not state or not target_ids:
            return None
        interaction = candidate.get("interaction_command") or {}
        belief = InteractionOutcomeBelief(
            target_id=target_ids[0],
            state=state,
            action=str(interaction.get("action") or "").strip().casefold(),
            source="command_outcome_belief",
            decision_id=str(candidate.get("decision_id") or ""),
            candidate_id=str(candidate.get("candidate_id") or ""),
            timestamp=float(time.time() if now is None else now),
        )
        for target_id in target_ids:
            self._beliefs[target_id] = belief
        return belief

    def belief_for_candidate(
        self, candidate: dict[str, Any]
    ) -> InteractionOutcomeBelief | None:
        for target_id in interaction_target_ids(candidate):
            belief = self._beliefs.get(target_id)
            if belief is not None:
                return belief
        return None

    def candidate_is_satisfied(self, candidate: dict[str, Any]) -> bool:
        """Whether an interaction is already satisfied by a local belief."""

        if str(candidate.get("behavior_type") or "").upper() != "INTERACT":
            return False
        desired_state = expected_interaction_state(candidate)
        belief = self.belief_for_candidate(candidate)
        return bool(
            desired_state
            and belief is not None
            and str(belief.state) == desired_state
        )

    def as_list(self) -> list[dict[str, Any]]:
        unique: dict[tuple[str, str, str], InteractionOutcomeBelief] = {}
        for belief in self._beliefs.values():
            unique[(belief.target_id, belief.state, belief.candidate_id)] = belief
        return [
            asdict(belief)
            for belief in sorted(unique.values(), key=lambda item: item.target_id)
        ]

    def overlay_compact_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Attach local state beliefs to a compact graph without mutating it."""

        overlay = copy.deepcopy(graph)
        nodes: list[dict[str, Any]] = []
        for raw_node in list(overlay.get("nodes") or []):
            node = dict(raw_node)
            belief = self._beliefs.get(str(node.get("id") or ""))
            if belief is not None:
                node["interaction_state"] = belief.state
                node["interaction_state_source"] = belief.source
                node["interaction_state_belief"] = True
                if belief.state == "open":
                    node["requires_interaction"] = False
                    if str(node.get("type") or "").casefold() == "portal":
                        node["traversable"] = True
            nodes.append(node)
        overlay["nodes"] = nodes
        overlay["interaction_outcome_beliefs"] = self.as_list()
        return overlay
