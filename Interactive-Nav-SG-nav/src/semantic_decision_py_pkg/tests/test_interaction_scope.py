import copy

import pytest

from semantic_decision_py_pkg.behavior_candidates import CandidateGenerator, CandidateGeneratorConfig
from semantic_decision_py_pkg.interaction_scope import InteractionGoalScope


def scope(*names, enabled=True):
    return InteractionGoalScope.from_config({"enabled": enabled, "allowed_semantic_types": list(names)})


@pytest.mark.parametrize("name,allowed", [("door", True), ("refrigerator", True), ("locker", False), ("water_dispenser", False)])
def test_shared_config_admission(name, allowed):
    assert scope("door", "fridge").allows_candidate({
        "behavior_type": "INTERACT", "metadata": {"semantic_name": name},
    }) is allowed


def test_config_can_allow_other_classes_or_disable_all_without_affecting_navigation():
    candidate = {"behavior_type": "INTERACT", "target_name": "locker"}
    assert scope("locker").allows_candidate(candidate)
    assert not scope("locker", enabled=False).allows_candidate(candidate)
    assert not scope().allows_candidate(candidate)
    assert InteractionGoalScope.from_config(None).allows_candidate(candidate)
    assert scope().allows_candidate({"behavior_type": "EXPLORE"})


@pytest.mark.parametrize("config", [[], {"allowed_semantic_types": "door"}, {"allowed_semantic_types": [None]}, {"enabled": "false", "allowed_semantic_types": []}])
def test_invalid_config_does_not_silently_allow_all(config):
    with pytest.raises(ValueError):
        InteractionGoalScope.from_config(config)


def test_locker_stays_in_graph_but_requires_committed_fridge_identity_for_goal():
    policy = scope("door", "fridge")
    generator = CandidateGenerator(CandidateGeneratorConfig(
        interaction_goal_scope=policy, container_pre_action_mllm=False,
    ))
    node = {
        "id": "locker1", "type": "object", "label": "locker", "name": "locker",
        "centroid": [2., 0., .9], "aabb_center": [2., 0., .9],
        "aabb_size": [.8, .6, 1.8], "room_id": 1, "state_age_sec": 0.,
        "attributes": {"source_semantic_name": "locker", "source_category": "locker",
                       "consecutive_observations": 2},
        "interaction": {"is_interactable": False, "state": "unknown", "state_confidence": .9},
    }
    graph = {"nodes": [node], "edges": []}
    before = copy.deepcopy(graph)
    assert generator.generate({}, graph, (0., 0.), {}) == []
    assert graph == before
    node["attributes"].update(m1_refrigerator_pending_confirmation=True,
                              m1_pending_observed_object_name="refrigerator")
    assert generator.generate({}, graph, (0., 0.), {}) == []
    node.update(type="container", label="refrigerator", name="refrigerator")
    node["attributes"].update(m1_name_override=True, m1_refrigerator_confirmed=True,
                              m1_refrigerator_pending_confirmation=False,
                              m1_observed_object_name="refrigerator", semantic_name="refrigerator")
    node["interaction"].update(is_interactable=True, requires_interaction=True,
                               interaction_mode="open_close", capability="open", state="closed")
    candidates = generator.generate({}, graph, (0., 0.), {})
    assert candidates and all(policy.allows_candidate(c) for c in candidates)
    assert candidates[0].metadata["interaction_semantic_type"] == "fridge"
    node["state_age_sec"] = 1000.
    node["is_currently_visible"] = False
    assert generator.generate({}, graph, (0., 0.), {}) == []
    node["attributes"]["persistent_tracking_node"] = True
    assert generator.generate({}, graph, (0., 0.), {})
    # A historical pending refrigerator answer must not undo a later correction.
    node["attributes"]["m1_observed_object_name"] = "water_dispenser"
    assert generator.generate({}, graph, (0., 0.), {}) == []


@pytest.mark.parametrize("allowed,expected", [(["microwave"], True), (["door", "fridge"], False), ([], False)])
def test_generator_whitelist_accepts_configured_non_fridge_class(allowed, expected):
    generator = CandidateGenerator(CandidateGeneratorConfig(
        interaction_goal_scope=scope(*allowed), container_pre_action_mllm=False,
    ))
    node = {
        "id": "microwave1", "type": "container", "label": "microwave", "name": "microwave",
        "aabb_center": [2., 0., 1.], "aabb_size": [.6, .5, .4], "room_id": 1,
        "attributes": {"semantic_name": "microwave"},
        "interaction": {"is_interactable": True, "requires_interaction": True,
                        "state": "closed", "state_confidence": .9, "interaction_mode": "open_close"},
    }
    graph = {"nodes": [node], "edges": []}
    candidates = generator.generate({}, graph, (0., 0.), {})
    assert bool(candidates) is expected
    assert all(scope(*allowed).allows_candidate(c) for c in candidates)
