"""Pure tests for V3 ROS policy terminal and liveness semantics."""

from __future__ import annotations

import pytest

from scripts.InteractiveNav.evaluation.benchmark_types import PolicyAction
from scripts.InteractiveNav.evaluation.ros_policy_termination import (
    COMMAND_STARVATION_REASON,
    NO_FRESH_ACTION_REASON,
    OBSERVATION_TURN_LIMIT_REASON,
    RosPolicyTerminationConfig,
    RosPolicyTerminationGuard,
    counts_toward_applied_action_budget,
)


def _no_fresh_action() -> PolicyAction:
    return PolicyAction(
        kind="observe",
        metadata={"reason": NO_FRESH_ACTION_REASON},
    )


def test_short_or_intermittent_no_fresh_wait_does_not_end_episode() -> None:
    guard = RosPolicyTerminationGuard(
        RosPolicyTerminationConfig(command_starvation_timeout_s=60.0)
    )

    assert guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=10.0, now_mono_s=11.0
    ) is None
    assert guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=11.0, now_mono_s=35.0
    ) is None
    assert guard.observe_action(
        PolicyAction(kind="base", base_action={"base": [0.0, 0.0, 0.0]}),
        now_mono_s=35.1,
    ) is None
    assert guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=40.0, now_mono_s=55.0
    ) is None

    snapshot = guard.snapshot(now_mono_s=55.0)
    assert snapshot["triggered"] is False
    assert snapshot["total_no_fresh_action_count"] == 3
    assert snapshot["consecutive_no_fresh_action_count"] == 1
    assert snapshot["max_consecutive_no_fresh_action_count"] == 2
    assert snapshot["progress_reset_count"] == 1


def test_sustained_no_fresh_wait_uses_wall_time_not_message_count() -> None:
    guard = RosPolicyTerminationGuard(
        RosPolicyTerminationConfig(command_starvation_timeout_s=60.0)
    )

    assert guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=100.0, now_mono_s=101.0
    ) is None
    # Only the second timeout is needed when the first call itself was followed
    # by a long evaluator delay: liveness is a wall-clock contract, not a count.
    terminal = guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=159.0, now_mono_s=160.0
    )

    assert terminal is not None
    assert terminal.reason == COMMAND_STARVATION_REASON
    assert terminal.source == "command_starvation"
    assert terminal.detail["consecutive_no_fresh_action_count"] == 2
    assert terminal.detail["no_fresh_wall_seconds"] == pytest.approx(60.0)


def test_evaluator_consumed_interaction_resets_command_starvation() -> None:
    guard = RosPolicyTerminationGuard(
        RosPolicyTerminationConfig(command_starvation_timeout_s=60.0)
    )
    assert guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=0.0, now_mono_s=50.0
    ) is None

    guard.note_progress(now_mono_s=50.0)

    assert guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=55.0, now_mono_s=65.0
    ) is None
    assert guard.snapshot(now_mono_s=65.0)["current_no_fresh_wall_seconds"] == 10.0


def test_same_turn_interaction_retracts_provisional_starvation_terminal() -> None:
    guard = RosPolicyTerminationGuard(
        RosPolicyTerminationConfig(command_starvation_timeout_s=10.0)
    )
    terminal = guard.observe_action(
        _no_fresh_action(),
        wait_started_mono_s=0.0,
        now_mono_s=10.0,
    )
    assert terminal is not None
    assert terminal.reason == COMMAND_STARVATION_REASON

    # The runner consumes a queued interaction before committing the pending
    # terminal.  It is a real command on this same observation turn.
    guard.note_progress(now_mono_s=10.0)

    assert guard.terminal is None
    assert guard.observe_observation_turn_count(
        1,
        applied_step_budget=100,
    ) is None
    assert guard.snapshot(now_mono_s=10.0)["triggered"] is False


def test_progress_never_retracts_goal_status_terminal() -> None:
    guard = RosPolicyTerminationGuard()
    terminal = guard.observe_goal_status(
        {"status": "EXPLORATION_EXHAUSTED", "detail": {"reason": "empty"}}
    )
    assert terminal is not None

    guard.note_progress(now_mono_s=10.0)

    assert guard.terminal is terminal


def test_applied_action_budget_excludes_only_waits_and_stop() -> None:
    assert counts_toward_applied_action_budget(_no_fresh_action()) is False
    assert counts_toward_applied_action_budget(PolicyAction(kind="stop")) is False
    assert counts_toward_applied_action_budget(PolicyAction(kind="observe")) is True
    assert counts_toward_applied_action_budget(PolicyAction(kind="view")) is True
    assert counts_toward_applied_action_budget(PolicyAction(kind="interact")) is True
    assert counts_toward_applied_action_budget(
        PolicyAction(kind="base", base_action={"base": [0.0, 0.0, 0.0]})
    ) is True


def test_observation_turn_limit_is_independent_from_applied_budget() -> None:
    guard = RosPolicyTerminationGuard(
        RosPolicyTerminationConfig(observation_turn_multiplier=4.0)
    )

    assert guard.observation_turn_limit(100) == 400
    assert guard.observe_observation_turn_count(
        399, applied_step_budget=100
    ) is None
    terminal = guard.observe_observation_turn_count(
        400, applied_step_budget=100
    )

    assert terminal is not None
    assert terminal.reason == OBSERVATION_TURN_LIMIT_REASON
    assert terminal.detail["applied_step_budget"] == 100


def test_fixed_round_observation_cap_is_hard_after_final_feedback_drain() -> None:
    """The V3 launch profile bounds non-applied turns without re-counting actions."""

    guard = RosPolicyTerminationGuard(
        RosPolicyTerminationConfig(observation_turn_multiplier=1.5)
    )

    assert guard.observation_turn_limit(2000) == 3000
    assert guard.observe_observation_turn_count(
        2999, applied_step_budget=2000
    ) is None
    terminal = guard.observe_observation_turn_count(
        3000, applied_step_budget=2000
    )

    assert terminal is not None
    assert terminal.reason == OBSERVATION_TURN_LIMIT_REASON
    assert terminal.source == "observation_turn_limit"
    assert terminal.detail == {
        "observation_turn_count": 3000,
        "observation_turn_limit": 3000,
        "applied_step_budget": 2000,
    }
    # The runner drains terminal feedback before calling this hard cap.  Once it
    # has fired, a later progress callback must not silently re-open the loop.
    guard.note_progress(now_mono_s=1.0)
    assert guard.terminal is terminal


@pytest.mark.parametrize(
    ("status", "reported_reason", "reason"),
    [
        ("SUCCEEDED", "target_goal_succeeded", "policy_goal_succeeded"),
        ("EXPLORATION_EXHAUSTED", "frontiers_empty", "policy_exploration_exhausted"),
        ("EXPLORATION_STALLED", "no_plan", "policy_exploration_stalled"),
        ("STARTUP_SCAN_FAILED", "scan_failed", "policy_startup_scan_failed"),
    ],
)
def test_public_terminal_goal_status_ends_rollout_without_scoring_it(
    status: str, reported_reason: str, reason: str
) -> None:
    guard = RosPolicyTerminationGuard()

    terminal = guard.observe_goal_status(
        {
            "status": status,
            "mission_mode": "semantic_interaction_object_goal",
            "decision_id": "decision_000004",
            "detail": {"reason": reported_reason},
        }
    )

    assert terminal is not None
    assert terminal.reason == reason
    assert terminal.source == "goal_status"
    # Deliberately no success field: evaluator GT remains authoritative.
    assert "success" not in terminal.detail


def test_generic_succeeded_goal_status_is_not_object_goal_completion() -> None:
    guard = RosPolicyTerminationGuard()

    assert guard.observe_goal_status(
        {"status": "SUCCEEDED", "detail": {"reason": "interaction_succeeded"}}
    ) is None
    assert guard.observe_goal_status(
        {
            "status": "SUCCEEDED",
            "mission_mode": "semantic_interaction_exploration",
            "detail": {"reason": "target_goal_succeeded"},
        }
    ) is None
    assert guard.observe_goal_status({"status": "SUCCEEDED"}) is None


@pytest.mark.parametrize(
    "status",
    ["ACTIVE", "DISABLED", "TARGET_REACHED", "TARGET_CONTAINER_INTERACTED", ""],
)
def test_nonterminal_goal_status_does_not_end_rollout(status: str) -> None:
    guard = RosPolicyTerminationGuard()

    assert guard.observe_goal_status({"status": status}) is None
    assert guard.snapshot(now_mono_s=0.0)["triggered"] is False


def test_zero_command_starvation_timeout_disables_only_liveness_guard() -> None:
    guard = RosPolicyTerminationGuard(
        RosPolicyTerminationConfig(command_starvation_timeout_s=0.0)
    )
    assert guard.observe_action(
        _no_fresh_action(), wait_started_mono_s=0.0, now_mono_s=10_000.0
    ) is None
    assert guard.snapshot(now_mono_s=10_000.0)["enabled"] is False

    terminal = guard.observe_goal_status({"status": "EXPLORATION_EXHAUSTED"})
    assert terminal is not None
    assert terminal.reason == "policy_exploration_exhausted"


@pytest.mark.parametrize("timeout", [-1.0, float("inf"), float("nan")])
def test_invalid_command_starvation_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        RosPolicyTerminationGuard(
            RosPolicyTerminationConfig(command_starvation_timeout_s=timeout)
        )


@pytest.mark.parametrize("multiplier", [0.0, 0.5, float("inf"), float("nan")])
def test_invalid_observation_turn_multiplier_is_rejected(multiplier: float) -> None:
    with pytest.raises(ValueError, match="finite and at least 1"):
        RosPolicyTerminationGuard(
            RosPolicyTerminationConfig(observation_turn_multiplier=multiplier)
        )
