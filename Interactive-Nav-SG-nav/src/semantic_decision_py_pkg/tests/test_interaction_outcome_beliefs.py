from semantic_decision_py_pkg.interaction_outcome_beliefs import (
    InteractionOutcomeBeliefStore,
    expected_interaction_state,
    interaction_target_ids,
    uses_direct_atomic_outcome_beliefs,
)


def open_candidate() -> dict:
    return {
        "candidate_id": "interaction:obj_0007:open",
        "behavior_type": "INTERACT",
        "target_id": "node_obj_0007",
        "decision_id": "decision_000007",
        "interaction_command": {
            "node_id": "node_obj_0007",
            "object_id": "obj_0007",
            "action": "open",
            "expected_state": "open",
        },
    }


def test_successful_open_creates_id_only_outcome_belief() -> None:
    candidate = open_candidate()
    store = InteractionOutcomeBeliefStore()

    belief = store.record_success(candidate, now=12.5)

    assert belief is not None
    assert belief.target_id == "node_obj_0007"
    assert belief.state == "open"
    assert belief.source == "command_outcome_belief"
    assert belief.timestamp == 12.5
    assert interaction_target_ids(candidate) == ("node_obj_0007", "obj_0007")
    assert store.candidate_is_satisfied(candidate)


def test_outcome_belief_suppresses_only_the_same_requested_state() -> None:
    store = InteractionOutcomeBeliefStore()
    store.record_success(open_candidate(), now=1.0)
    close_candidate = open_candidate()
    close_candidate["candidate_id"] = "interaction:obj_0007:close"
    close_candidate["interaction_command"] = {
        **close_candidate["interaction_command"],
        "action": "close",
        "expected_state": "closed",
    }
    navigation_candidate = {**open_candidate(), "behavior_type": "NAVIGATE"}

    assert not store.candidate_is_satisfied(close_candidate)
    assert not store.candidate_is_satisfied(navigation_candidate)


def test_compact_graph_overlay_marks_belief_source_without_mutating_input() -> None:
    graph = {
        "nodes": [
            {
                "id": "node_obj_0007",
                "type": "portal",
                "interaction_state": "closed",
                "requires_interaction": True,
                "traversable": False,
            },
            {"id": "other", "type": "container", "interaction_state": "closed"},
        ]
    }
    store = InteractionOutcomeBeliefStore()
    store.record_success(open_candidate(), now=1.0)

    overlay = store.overlay_compact_graph(graph)

    assert graph["nodes"][0]["interaction_state"] == "closed"
    assert overlay["nodes"][0]["interaction_state"] == "open"
    assert overlay["nodes"][0]["interaction_state_source"] == "command_outcome_belief"
    assert overlay["nodes"][0]["requires_interaction"] is False
    assert overlay["nodes"][0]["traversable"] is True
    assert overlay["nodes"][1]["interaction_state"] == "closed"
    assert overlay["interaction_outcome_beliefs"] == [
        {
            "target_id": "node_obj_0007",
            "state": "open",
            "action": "open",
            "source": "command_outcome_belief",
            "decision_id": "decision_000007",
            "candidate_id": "interaction:obj_0007:open",
            "timestamp": 1.0,
        }
    ]


def test_unknown_action_does_not_create_a_state_belief() -> None:
    candidate = open_candidate()
    candidate["interaction_command"] = {"action": "scan", "object_id": "obj_0007"}
    store = InteractionOutcomeBeliefStore()

    assert expected_interaction_state(candidate) == ""
    assert store.record_success(candidate, now=1.0) is None
    assert store.as_list() == []


def test_outcome_beliefs_are_explicitly_limited_to_evaluator_direct_atomic_mode() -> None:
    assert uses_direct_atomic_outcome_beliefs("direct_atomic", True)
    assert not uses_direct_atomic_outcome_beliefs("rule_verified", True)
    assert not uses_direct_atomic_outcome_beliefs("direct_atomic", False)
