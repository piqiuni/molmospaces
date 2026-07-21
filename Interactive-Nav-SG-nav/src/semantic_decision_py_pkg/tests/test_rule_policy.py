from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.rule_policy import RulePolicy


def candidate(candidate_id, behavior_type, distance, node_type="", gain=1.0):
    return BehaviorCandidate(
        candidate_id=candidate_id,
        behavior_type=behavior_type,
        source="test",
        target_id=candidate_id,
        target_name=candidate_id,
        features={
            "exploration_gain": gain,
            "visibility_gain": gain,
            "semantic_gain": gain,
            "distance_m": distance,
            "interaction_cost": 1.0 if behavior_type == "INTERACT" else 0.0,
            "state_age_ratio": 0.0,
            "confidence": 1.0,
            "priority": 1.0,
        },
        metadata={"node_type": node_type},
    )


def test_nearby_portal_is_preferred_to_far_frontier() -> None:
    policy = RulePolicy()
    selected = policy.select(
        [
            candidate("frontier", "EXPLORE", distance=5.0),
            candidate("door", "INTERACT", distance=1.0, node_type="portal"),
        ]
    )
    assert selected.candidate_id == "door"
    assert selected.score_terms["type_bonus"] > 0.0


def test_policy_is_deterministic_and_rewards_continuity() -> None:
    policy = RulePolicy()
    first = candidate("a", "EXPLORE", distance=1.0)
    second = candidate("b", "EXPLORE", distance=1.0)
    assert policy.select([second, first]).candidate_id == "a"
    assert policy.select([second, first], current_candidate_id="b").candidate_id == "b"


def test_target_candidate_beats_equal_distance_exploration() -> None:
    policy = RulePolicy()
    target = candidate("target", "NAVIGATE", distance=2.0, gain=0.0)
    target.features["target_relevance"] = 1.0
    target.metadata["target_goal"] = True
    frontier = candidate("frontier", "EXPLORE", distance=2.0, gain=1.0)
    assert policy.select([frontier, target]).candidate_id == "target"


def test_nearby_interaction_bonus_applies_only_without_target() -> None:
    policy = RulePolicy()
    nearby = candidate("nearby", "INTERACT", distance=1.0, node_type="container")
    nearby.metadata["object_distance_m"] = 0.25
    far_frontier = candidate("frontier", "EXPLORE", distance=1.0, gain=1.0)
    assert policy.select([far_frontier, nearby]).candidate_id == "nearby"
    nearby.metadata["target_enabled"] = True
    assert policy.select([far_frontier, nearby]).candidate_id == "frontier"
