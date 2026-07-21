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


def test_semantic_completion_holds_requested_number_of_steps() -> None:
    state = CompletionState(
        CompletionMonitorConfig(mode="semantic", post_completion_hold_steps=3)
    )

    assert state.update_semantic(
        {"status": "SUCCEEDED", "detail": {"reason": "target_visible"}}
    ) is True
    assert state.should_stop(10) is False
    assert state.is_holding(12) is True
    assert state.should_stop(13) is True
