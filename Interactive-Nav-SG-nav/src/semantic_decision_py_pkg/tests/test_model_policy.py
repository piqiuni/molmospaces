from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.model_policy import (
    ModelCircuitBreaker,
    ModelPolicyClient,
    ModelPolicyConfig,
    compact_graph,
)


def make_candidate(candidate_id: str, target_relevance: float, distance_m: float) -> BehaviorCandidate:
    return BehaviorCandidate(
        candidate_id=candidate_id,
        behavior_type="NAVIGATE" if target_relevance else "EXPLORE",
        source="test",
        target_id=candidate_id,
        target_name=candidate_id,
        features={
            "target_relevance": target_relevance,
            "visibility_gain": 1.0,
            "distance_m": distance_m,
        },
    )


def test_mock_model_selects_target_relevant_candidate() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="mock"))
    selected = client.select(
        [make_candidate("frontier", 0.0, 1.0), make_candidate("target", 1.0, 4.0)],
        target_context={"enabled": True, "target_name": "fridge"},
        graph={},
    )
    assert selected.candidate_id == "target"


def test_compact_graph_keeps_interaction_state() -> None:
    graph = compact_graph(
        {
            "scene_id": "house_7",
            "graph_revision": 9,
            "nodes": [
                {
                    "id": "portal_1",
                    "type": "portal",
                    "label": "door",
                    "centroid": [1.0, 2.0, 0.0],
                    "attributes": {"connected_room_ids": [1, 2]},
                    "interaction": {
                        "state": "closed",
                        "requires_interaction": True,
                        "traversable": False,
                    },
                }
            ],
            "edges": [],
        }
    )
    assert graph["nodes"][0]["interaction_state"] == "closed"
    assert graph["nodes"][0]["connected_room_ids"] == [1, 2]


def test_circuit_breaker_opens_only_after_consecutive_timeouts() -> None:
    breaker = ModelCircuitBreaker(consecutive_timeout_limit=2, cooldown_s=30.0)

    assert breaker.record_failure("timed out", now=10.0) is False
    assert breaker.allow_request(now=10.0)
    assert breaker.record_failure("timed out", now=11.0) is True
    assert not breaker.allow_request(now=40.0)
    assert breaker.allow_request(now=41.0)


def test_circuit_breaker_resets_timeout_streak_after_non_timeout_failure() -> None:
    breaker = ModelCircuitBreaker(consecutive_timeout_limit=2, cooldown_s=30.0)

    breaker.record_failure("timed out", now=10.0)
    assert breaker.record_failure("HTTP 503", now=11.0) is False
    assert breaker.consecutive_timeouts == 0
