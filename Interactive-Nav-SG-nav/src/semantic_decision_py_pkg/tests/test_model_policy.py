from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.model_policy import (
    ModelCircuitBreaker,
    ModelPolicyClient,
    ModelPolicyConfig,
    compact_graph,
    compact_semantic_graph,
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
        metadata={"target_goal": bool(target_relevance)},
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


def test_semantic_graph_keeps_only_rooms_portals_and_containers() -> None:
    graph = compact_semantic_graph(
        {
            "nodes": [
                {"id": "room_1", "type": "room", "label": "kitchen", "centroid": [0, 0, 0]},
                {
                    "id": "portal_1",
                    "type": "portal",
                    "label": "door",
                    "attributes": {"connected_room_ids": [1, 2]},
                    "interaction": {"state": "closed", "requires_interaction": True},
                },
                {
                    "id": "container_1",
                    "type": "container",
                    "label": "refrigerator",
                    "room_id": 1,
                    "interaction": {"state": "closed", "requires_interaction": True},
                },
                {"id": "object_1", "type": "object", "label": "apple"},
            ],
            "edges": [{"src_id": "room_1", "relation": "contains", "dst_id": "object_1"}],
        },
        robot_context={"robot_xy": [0.1, 0.1]},
    )

    assert graph == {
        "rooms": [
            {
                "id": "room_1",
                "type": "kitchen",
                "anchor_objects": [
                    {"type": "refrigerator", "visible": False}
                ],
            }
        ],
        "portals": [
            {
                "id": "portal_1",
                "type": "door",
                "state": "closed",
                "interaction_available": True,
                "connects": ["room_1", "room_2"],
            }
        ],
        "containers": [
            {
                "id": "container_1",
                "type": "refrigerator",
                "state": "closed",
                "interaction_available": True,
                "room_id": "room_1",
            }
        ],
        "current_room": "room_1",
    }


def test_semantic_graph_exposes_inferred_room_attributes_and_unassigned_anchors() -> None:
    graph = compact_semantic_graph(
        {
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "label": "unknown_room",
                    "centroid": [0.0, 0.0, 0.0],
                    "attributes": {
                        "room_attribute": "kitchen",
                        "room_attribute_confidence": 0.85,
                        "room_attribute_scores": {
                            "kitchen": 2.0,
                            "livingroom": 0.2,
                        },
                    },
                },
                {
                    "id": "container_fridge",
                    "type": "container",
                    "label": "refrigerator",
                    "aabb_size": [1.0, 1.0, 2.0],
                    "is_currently_visible": True,
                },
            ]
        }
    )

    assert graph["rooms"][0]["type"] == "kitchen"
    assert graph["rooms"][0]["observed_type"] == "unknown_room"
    assert graph["rooms"][0]["room_attribute_scores"] == {
        "kitchen": 2.0,
        "livingroom": 0.2,
    }
    assert graph["unassigned_anchor_objects"] == [
        {"type": "refrigerator", "visible": True}
    ]


def test_model_candidate_exposes_unknown_area_and_nearby_semantics() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    candidate = BehaviorCandidate(
        candidate_id="frontier:large_kitchen_side",
        behavior_type="EXPLORE",
        source="test",
        target_id="large_kitchen_side",
        target_name="large_kitchen_side",
        goal_xyyaw=[2.0, 2.0, 0.0],
        features={"distance_m": 4.0},
        metadata={
            "unknown_component_area_m2": 22.75,
            "expected_visible_unknown_area_m2": 14.5,
            "nearby_semantic_nodes": [
                {
                    "label": "refrigerator",
                    "distance_m": 0.8,
                    "visible": True,
                }
            ],
        },
    )

    request = client.build_request([candidate], {"enabled": True, "target_name": "apple"}, {})

    assert request["candidates"][0]["unknown_component_area_m2"] == 22.75
    assert request["candidates"][0]["expected_visible_unknown_area_m2"] == 14.5
    assert request["candidates"][0]["nearby_semantic_nodes"] == [
        {"type": "refrigerator", "distance_m": 0.8, "visible": True}
    ]
    assert "expected_visible_unknown_area_m2" in request["instruction"]
    assert "distance only as a tie-break" in request["instruction"]


def test_model_request_contains_only_semantic_candidate_fields_and_distance() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    interaction = BehaviorCandidate(
        candidate_id="interaction:fridge:open",
        behavior_type="INTERACT",
        source="test",
        target_id="container_fridge",
        target_name="refrigerator",
        goal_xyyaw=[3.0, 4.0, 1.2],
        interaction_command={"action": "open", "joint_names": ["joint_1"]},
        features={"distance_m": 2.345, "visibility_gain": 0.9, "interaction_cost": 1.0},
        metadata={
            "node_type": "container",
            "semantic_name": "refrigerator",
            "debug": {"large": "payload"},
        },
    )

    request = client.build_request(
        [interaction],
        {"enabled": True, "target_name": "apple", "object_labels": ["apple"]},
        {"nodes": []},
        {"robot_xy": [1.0, 2.0], "exploration_context": {"proposal_count": 12}},
    )

    assert request["mission"] == {
        "mode": "object_goal",
        "target": {"name": "apple", "visible": False, "labels": ["apple"]},
    }
    assert request["robot"] == {}
    assert request["candidates"] == [
        {
            "id": "interaction:fridge:open",
            "action": "open",
            "subject_id": "container_fridge",
            "subject_type": "container",
            "effect": "reveal_contents",
            "distance_m": 2.35,
            "subject_name": "refrigerator",
            "subject_semantic_type": "refrigerator",
        }
    ]
    serialized = str(request)
    assert "goal_xyyaw" not in serialized
    assert "joint_names" not in serialized
    assert "visibility_gain" not in serialized
    assert "metadata" not in serialized


def test_exploration_candidates_are_sent_as_concrete_subgoals() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    candidates = [
        BehaviorCandidate(
            candidate_id="frontier:a",
            behavior_type="EXPLORE",
            source="test",
            target_id="a",
            target_name="a",
            goal_xyyaw=[0.5, 0.0, 0.0],
            features={"distance_m": 1.0},
        ),
        BehaviorCandidate(
            candidate_id="frontier:b",
            behavior_type="EXPLORE",
            source="test",
            target_id="b",
            target_name="b",
            goal_xyyaw=[1.0, 0.0, 0.0],
            features={"distance_m": 2.0},
        ),
    ]

    request = client.build_request(
        candidates,
        {"enabled": False},
        {
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "label": "kitchen",
                    "centroid": [0, 0, 0],
                }
            ]
        },
    )

    assert request["candidates"] == [
        {
            "id": "frontier:a",
            "action": "explore",
            "subject_id": "a",
            "subject_type": "frontier",
            "effect": "reveal_space",
            "distance_m": 1.0,
            "room_id": "room_1",
        },
        {
            "id": "frontier:b",
            "action": "explore",
            "subject_id": "b",
            "subject_type": "frontier",
            "effect": "reveal_space",
            "distance_m": 2.0,
            "room_id": "room_1",
        },
    ]


def test_model_selection_returns_the_exact_frontier_id(monkeypatch) -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="mock"))
    candidates = [
        BehaviorCandidate(
            candidate_id="frontier:a",
            behavior_type="EXPLORE",
            source="test",
            target_id="a",
            target_name="a",
            goal_xyyaw=[0.5, 0.0, 0.0],
            features={"distance_m": 1.0},
        ),
        BehaviorCandidate(
            candidate_id="frontier:b",
            behavior_type="EXPLORE",
            source="test",
            target_id="b",
            target_name="b",
            goal_xyyaw=[1.0, 0.0, 0.0],
            features={"distance_m": 2.0},
        ),
    ]
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, metrics_context=None: {
            "ranked_ids": ["frontier:b", "frontier:a"],
            "reason": "INFORMATION_GAIN",
            "confidence": "high",
        },
    )

    selected = client.select(candidates, graph={})

    assert selected is candidates[1]
    assert client.last_selected_candidate_id == "frontier:b"


def test_legacy_room_granularity_remains_available_for_ablation() -> None:
    client = ModelPolicyClient(
        ModelPolicyConfig(mode="disabled", selection_granularity="room")
    )
    candidates = [
        BehaviorCandidate(
            candidate_id="frontier:a",
            behavior_type="EXPLORE",
            source="test",
            target_id="a",
            target_name="a",
            goal_xyyaw=[0.5, 0.0, 0.0],
            features={"distance_m": 1.0},
            metadata={"cell_count": 12, "map_resolution": 0.05},
        ),
        BehaviorCandidate(
            candidate_id="frontier:b",
            behavior_type="EXPLORE",
            source="test",
            target_id="b",
            target_name="b",
            goal_xyyaw=[1.0, 0.0, 0.0],
            features={"distance_m": 2.0},
            metadata={"cell_count": 8, "map_resolution": 0.05},
        ),
    ]

    request = client.build_request(
        candidates,
        {"enabled": False},
        {"nodes": [{"id": "room_1", "type": "room", "centroid": [0, 0, 0]}]},
        {
            "decision_history": [
                {
                    "group_id": "explore:room_1",
                    "candidate_id": "frontier:a",
                    "result": "SUCCEEDED",
                    "steps_ago": 20,
                }
            ],
            "group_history": [
                {
                    "group_id": "explore:room_1",
                    "selection_count": 3,
                    "consecutive_selection_count": 2,
                    "last_selected_steps_ago": 20,
                    "last_result": "SUCCEEDED",
                    "last_frontier_length_delta_m": -0.05,
                    "low_gain_repeat_count": 2,
                }
            ],
        },
    )

    assert request["candidates"][0]["frontier_cell_count"] == 20
    assert request["candidates"][0]["frontier_length_m"] == 1.0
    assert request["candidates"][0]["history"]["low_gain_repeat_count"] == 2
    assert request["recent_decisions"][0]["candidate_id"] == "frontier:a"


def test_candidate_request_includes_spatial_history_without_rule_score() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    candidate = BehaviorCandidate(
        candidate_id="frontier:new_cluster",
        behavior_type="EXPLORE",
        source="test",
        target_id="new_cluster",
        target_name="new_cluster",
        goal_xyyaw=[1.2, 1.1, 0.0],
        features={"distance_m": 2.0},
        metadata={
            "target_room_id": 1,
            "frontier_point": [1.2, 1.1],
            "cell_count": 10,
            "map_resolution": 0.05,
        },
    )

    request = client.build_request(
        [candidate],
        {"enabled": True, "target_name": "apple"},
        {},
        {
            "room_frontier_lengths": {"room_1": 2.0},
            "candidate_history": {
                "explore_region:room_1:1:1": {
                    "selection_count": 2,
                    "last_selected_steps_ago": 15,
                    "last_result": "SUCCEEDED",
                    "low_gain_repeat_count": 1,
                    "last_frontier_shrink_m": 0.05,
                }
            }
        },
    )

    option = request["candidates"][0]
    assert option["id"] == "frontier:new_cluster"
    assert option["frontier_length_m"] == 0.5
    assert option["room_frontier_length_m"] == 2.0
    assert option["history"]["selection_count"] == 2
    assert "rule_score" not in str(request)


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
