from scripts.InteractiveNav.ros_completion_monitor import (
    CompletionMonitorConfig,
    CompletionState,
)


def test_frontier_completion_requires_confirmations_and_no_active_proposal() -> None:
    state = CompletionState(
        CompletionMonitorConfig(mode="frontier", frontier_confirmations=2)
    )
    active = {
        "ready": True,
        "frontier_exhausted": False,
        "active_proposal_id": "frontier_1",
        "proposal_count": 0,
    }
    empty = {
        "ready": True,
        "frontier_exhausted": True,
        "active_proposal_id": "",
        "proposal_count": 0,
    }

    assert state.update_frontier(active) is False
    assert state.update_frontier(empty) is False
    assert state.update_frontier(empty) is True
    assert state.reason == "frontier_exhausted"


def test_exploration_completion_ignores_target_success_and_holds_after_frontiers_exhausted() -> None:
    state = CompletionState(
        CompletionMonitorConfig(mode="semantic", post_completion_hold_steps=3)
    )

    assert state.update_semantic(
        {"status": "SUCCEEDED", "detail": {"reason": "target_goal_succeeded"}}
    ) is False
    assert state.target_goal_succeeded is True
    assert state.update_semantic(
        {
            "status": "EXPLORATION_EXHAUSTED",
            "detail": {
                "reason": "navigation_and_interaction_frontiers_exhausted"
            },
        }
    ) is True
    assert state.should_stop(10) is False
    assert state.is_holding(12) is True
    assert state.should_stop(13) is True


def test_object_goal_completion_stops_on_verified_target_success() -> None:
    state = CompletionState(CompletionMonitorConfig(mode="semantic"))

    assert state.update_semantic(
        {
            "status": "SUCCEEDED",
            "mission_mode": "semantic_interaction_object_goal",
            "detail": {"reason": "target_goal_succeeded"},
        }
    ) is True
    assert state.reason == "target_goal_succeeded"
