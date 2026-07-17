from semantic_decision_py_pkg.behavior_execution import (
    BehaviorExecutionStateMachine,
    STATE_INTERACTING,
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


def test_explore_behavior_delegates_to_explorer_skill() -> None:
    machine = BehaviorExecutionStateMachine()
    commands = machine.start(
        {
            "candidate_id": "frontier_1",
            "behavior_type": "EXPLORE",
            "metadata": {"cluster_id": "cluster_1"},
        },
        now=0.0,
    )
    assert commands[0]["kind"] == "explore_frontier"
    terminal = machine.on_explore_result(True, now=1.0)
    assert terminal[0]["success"] is True
