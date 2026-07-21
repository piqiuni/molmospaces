from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    ExecutionConfig,
    STATE_FINALIZING_EXPLORE,
    STATE_INTERACTING,
    STATE_NAVIGATING,
    STATE_PREPARING_EXPLORE,
    STATE_SUCCEEDED,
    STATE_VERIFYING,
)


def interaction_candidate(requires_approach=True):
    return {
        "candidate_id": "door_open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 2.0, 0.0],
        "interaction_command": {"expected_state": "open"},
        "metadata": {"requires_approach": requires_approach},
    }


def target_candidate():
    return {
        "candidate_id": "target:fridge_1",
        "behavior_type": "NAVIGATE",
        "target_id": "fridge_1",
        "goal_xyyaw": [4.0, 2.0, 0.0],
        "metadata": {
            "target_goal": True,
            "verify_target_visibility": True,
            "target_min_visible_pixels": 16,
        },
    }


def test_interaction_execution_orders_approach_action_and_verification() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(interaction_candidate(), now=0.0)
    assert commands[0]["kind"] == "navigate"
    commands = machine.on_navigation_result(True, now=1.0)
    assert machine.state == STATE_INTERACTING
    assert commands[0]["kind"] == "interact"
    assert machine.on_interaction_result(True, now=2.0) == []
    assert machine.state == STATE_VERIFYING
    assert machine.on_graph_state("closed", now=3.0) == []
    terminal = machine.on_graph_state("open", now=4.0)
    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True


def test_explore_behavior_reserves_navigates_and_finalizes() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "goal_xyyaw": [9.0, 9.0, 0.0],
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    assert machine.state == STATE_PREPARING_EXPLORE
    assert commands[0]["kind"] == "reserve_frontier"

    commands = machine.on_explore_ready(
        {"goal_xyyaw": [1.0, 2.0, 0.75], "frame_id": "map"}, now=1.0
    )
    assert machine.state == STATE_NAVIGATING
    assert commands[0]["kind"] == "navigate"
    assert commands[0]["candidate"]["goal_xyyaw"] == [1.0, 2.0, 0.75]
    assert commands[0]["candidate"]["metadata"]["frame_id"] == "map"

    commands = machine.on_navigation_result(True, {"status": "SUCCEEDED"}, now=2.0)
    assert machine.state == STATE_FINALIZING_EXPLORE
    assert commands[0]["kind"] == "finalize_frontier"
    assert commands[0]["success"] is True

    terminal = machine.on_explore_result(True, {"event": "frontier_gone"}, now=3.0)
    assert terminal[0]["success"] is True


def test_explore_reservation_failure_finishes_without_navigation() -> None:
    machine = BehaviorExecutionStateMachine()
    machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    terminal = machine.on_explore_result(
        False, {"reason": "frontier_candidate_not_available"}, now=1.0
    )
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is False


def test_explore_navigation_timeout_requests_frontier_finalization() -> None:
    machine = BehaviorExecutionStateMachine(
        ExecutionConfig(navigation_timeout_s=2.0)
    )
    machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    machine.on_explore_ready({"goal_xyyaw": [1.0, 2.0, 0.0]}, now=1.0)
    assert machine.timeout_reason(now=3.1) == "navigation_timeout"
    commands = machine.fail_timeout("navigation_timeout", now=3.1)
    assert machine.state == STATE_FINALIZING_EXPLORE
    assert commands[0]["kind"] == "finalize_frontier"
    assert commands[0]["success"] is False


def test_target_navigation_waits_for_visibility_verification() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(target_candidate(), now=0.0)
    assert commands[0]["kind"] == "navigate"

    assert machine.on_navigation_result(True, now=1.0) == []
    assert machine.state == STATE_VERIFYING
    assert machine.on_target_visibility(False, now=2.0) == []
    terminal = machine.on_target_visibility(
        True,
        detail={"visible_pixels": 24, "min_visible_pixels": 16},
        now=3.0,
    )

    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["kind"] == "terminal"
    assert terminal[0]["success"] is True


def test_navigation_without_visibility_requirement_finishes_immediately() -> None:
    candidate = target_candidate()
    candidate["metadata"]["verify_target_visibility"] = False
    machine = BehaviorExecutionStateMachine()
    machine.start(candidate, now=0.0)

    terminal = machine.on_navigation_result(True, now=1.0)

    assert machine.state == STATE_SUCCEEDED
    assert terminal[0]["success"] is True
