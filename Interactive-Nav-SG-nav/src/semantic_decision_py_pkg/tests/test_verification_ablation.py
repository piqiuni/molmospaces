from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    STATE_VERIFYING,
    STATE_SUCCEEDED,
)


def test_verification_can_finish_or_request_retry() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(
        {
            "candidate_id": "interaction:door_1:open",
            "behavior_type": "INTERACT",
            "interaction_command": {"expected_state": "open"},
        }
    )
    machine.on_navigation_result(True)
    machine.on_interaction_result(True)
    assert machine.state == STATE_VERIFYING
    commands = machine.on_verification_result(False, {"reason": "not_open"}, retry=True)
    assert machine.state == "INTERACTING"
    assert commands[0]["kind"] == "interact"
    machine.on_interaction_result(True)
    machine.on_verification_result(True, {"verified": True})
    assert machine.state == STATE_SUCCEEDED
