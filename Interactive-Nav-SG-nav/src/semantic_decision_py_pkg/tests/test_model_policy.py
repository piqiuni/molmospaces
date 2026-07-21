from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.model_policy import ModelPolicyClient, ModelPolicyConfig, compact_graph


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
