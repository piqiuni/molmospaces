from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.model_policy import (
    ModelCircuitBreaker,
    ModelPolicyClient,
    ModelPolicyConfig,
    ROOM_OBJECT_REASONING_MAX_CONTAINERS,
    ROOM_OBJECT_REASONING_MAX_PORTALS,
    ROOM_OBJECT_REASONING_MAX_ROOMS,
    build_room_object_reasoning_context,
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


def test_model_request_exposes_pre_score_and_route_hint_without_geometry() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    portal = BehaviorCandidate(
        candidate_id="interaction:portal_12:open",
        behavior_type="INTERACT",
        source="test",
        target_id="portal_12",
        target_name="door",
        goal_xyyaw=[2.0, 0.0, 1.57],
        interaction_command={"action": "open", "joint_names": ["private_joint"]},
        features={"distance_m": 2.0},
        metadata={"node_type": "portal", "connected_room_ids": [1, 2]},
    )

    request = client.build_request(
        [portal],
        {"enabled": True, "target_name": "apple"},
        {},
        {
            "candidate_pre_scores": {portal.candidate_id: 1.825},
            "candidate_pre_score_terms": {
                portal.candidate_id: {"topology_priority": 1.35}
            },
            "candidate_decision_hints": {
                portal.candidate_id: "NEXT_ROUTE_PORTAL"
            },
        },
    )

    option = request["candidates"][0]
    assert option["pre_score"] == 1.825
    assert option["pre_score_terms"] == {"topology_priority": 1.35}
    assert option["decision_hint"] == "NEXT_ROUTE_PORTAL"
    assert request["schema_version"] == 4
    assert "NEXT_ROUTE_PORTAL before a remote container" in request["instruction"]
    assert "goal_xyyaw" not in str(request)
    assert "private_joint" not in str(request)


def test_pre_score_guard_overrides_model_only_for_strong_route_priority(monkeypatch) -> None:
    client = ModelPolicyClient(
        ModelPolicyConfig(mode="mock", pre_score_guard_margin=0.75)
    )
    frontier = BehaviorCandidate(
        candidate_id="frontier:near",
        behavior_type="EXPLORE",
        source="test",
        target_id="near",
        target_name="near",
        goal_xyyaw=[1.0, 0.0, 0.0],
        features={"distance_m": 1.0},
    )
    portal = BehaviorCandidate(
        candidate_id="interaction:portal_12:open",
        behavior_type="INTERACT",
        source="test",
        target_id="portal_12",
        target_name="door",
        goal_xyyaw=[2.0, 0.0, 0.0],
        interaction_command={"action": "open"},
        features={"distance_m": 2.0},
        metadata={"node_type": "portal"},
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, metrics_context=None: {
            "ranked_ids": [frontier.candidate_id, portal.candidate_id],
            "reason": "DISTANCE_TIEBREAK",
            "confidence": "medium",
        },
    )

    selected = client.select(
        [frontier, portal],
        target_context={"enabled": True, "target_name": "apple"},
        robot_context={
            "candidate_pre_scores": {
                frontier.candidate_id: 0.55,
                portal.candidate_id: 1.55,
            },
            "candidate_decision_hints": {
                portal.candidate_id: "NEXT_ROUTE_PORTAL"
            },
        },
    )

    assert selected is portal
    assert client.last_result_source == "model_pre_score_guard"
    assert client.last_reason == "PRE_SCORE_GUARD_NEXT_ROUTE_PORTAL"
    assert client.last_ranking_ids[0] == portal.candidate_id
    assert "frontier:near->interaction:portal_12:open" in client.last_pre_score_guard


def test_hallucinated_container_id_cannot_escape_curated_frontier_pool(
    monkeypatch,
) -> None:
    """Regression for the ep2664 smoke response after dresser suppression."""
    client = ModelPolicyClient(ModelPolicyConfig(mode="mock"))
    frontiers = [
        BehaviorCandidate(
            candidate_id="frontier:17:14",
            behavior_type="EXPLORE",
            source="test",
            target_id="17:14",
            target_name="17:14",
            goal_xyyaw=[1.0, 0.0, 0.0],
            features={"distance_m": 1.0},
        ),
        BehaviorCandidate(
            candidate_id="frontier:22:11",
            behavior_type="EXPLORE",
            source="test",
            target_id="22:11",
            target_name="22:11",
            goal_xyyaw=[2.0, 0.0, 0.0],
            features={"distance_m": 2.0},
        ),
    ]
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, metrics_context=None: {
            "ranked_ids": ["interaction:container_obj_000003:open"],
            "reason": "REVEAL_TARGET_CONTAINER",
            "confidence": "high",
        },
    )

    selected = client.select(
        frontiers,
        target_context={"enabled": True, "target_name": "Irishpotato"},
        robot_context={
            "candidate_pre_scores": {
                "frontier:17:14": 0.7,
                "frontier:22:11": 0.9,
            }
        },
    )

    assert selected is frontiers[1]
    assert selected.candidate_id in {candidate.candidate_id for candidate in frontiers}
    assert client.last_selected_candidate_id == "frontier:22:11"
    assert client.last_ranking_ids == ["frontier:22:11"]
    assert client.last_rejected_model_ids == [
        "interaction:container_obj_000003:open"
    ]
    assert client.last_result_source == "curated_fallback_invalid_response"
    assert client.last_reason == "CURATED_FALLBACK_INVALID_MODEL_ID"


def test_unknown_ranked_id_is_dropped_before_first_valid_curated_id(
    monkeypatch,
) -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="mock"))
    candidates = [
        BehaviorCandidate(
            candidate_id="frontier:17:14",
            behavior_type="EXPLORE",
            source="test",
            target_id="17:14",
            target_name="17:14",
            goal_xyyaw=[1.0, 0.0, 0.0],
            features={"distance_m": 1.0},
        ),
        BehaviorCandidate(
            candidate_id="frontier:22:11",
            behavior_type="EXPLORE",
            source="test",
            target_id="22:11",
            target_name="22:11",
            goal_xyyaw=[2.0, 0.0, 0.0],
            features={"distance_m": 2.0},
        ),
    ]
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, metrics_context=None: {
            "ranked_ids": [
                "interaction:container_obj_000003:open",
                "frontier:17:14",
                "frontier:22:11",
            ],
            "reason": "INFORMATION_GAIN",
            "confidence": "medium",
        },
    )

    selected = client.select(candidates)

    assert selected is candidates[0]
    assert client.last_ranking_ids == ["frontier:17:14", "frontier:22:11"]
    assert client.last_result_source == "model_filtered_unknown_ids"
    assert client.last_rejected_model_ids == [
        "interaction:container_obj_000003:open"
    ]


def test_post_interaction_traversal_guard_prevents_immediate_container_detour(
    monkeypatch,
) -> None:
    client = ModelPolicyClient(
        ModelPolicyConfig(mode="mock", pre_score_guard_margin=0.75)
    )
    traversal = BehaviorCandidate(
        candidate_id="post_interaction_traversal:portal_12",
        behavior_type="NAVIGATE",
        source="test",
        target_id="portal_12",
        target_name="portal traversal",
        goal_xyyaw=[1.5, 0.0, 0.0],
        features={"distance_m": 1.5},
        metadata={"post_interaction_traversal": True},
    )
    dresser = BehaviorCandidate(
        candidate_id="interaction:dresser:open",
        behavior_type="INTERACT",
        source="test",
        target_id="dresser",
        target_name="dresser",
        goal_xyyaw=[0.5, 0.0, 0.0],
        interaction_command={"action": "open"},
        features={"distance_m": 0.5},
        metadata={"node_type": "container"},
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, metrics_context=None: {
            "ranked_ids": [dresser.candidate_id, traversal.candidate_id],
            "reason": "REVEAL_TARGET_CONTAINER",
            "confidence": "medium",
        },
    )

    selected = client.select(
        [traversal, dresser],
        robot_context={
            "candidate_pre_scores": {
                traversal.candidate_id: 2.5,
                dresser.candidate_id: 1.4,
            },
            "candidate_decision_hints": {
                traversal.candidate_id: "POST_INTERACTION_TRAVERSE",
                dresser.candidate_id: "PLAUSIBLE_TARGET_CONTAINER",
            },
        },
    )

    assert selected is traversal
    assert client.last_reason == "PRE_SCORE_GUARD_POST_INTERACTION_TRAVERSE"
    assert client.last_result_source == "model_pre_score_guard"
    instruction = client.build_request([traversal, dresser], {}, {})["instruction"]
    assert "POST_INTERACTION_TRAVERSE" in instruction
    assert "Only IDs present in the current candidates array" in instruction
    assert "historical context and are forbidden" in instruction


def test_two_stage_room_object_context_links_apple_to_observed_kitchen_fridge() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    fridge = BehaviorCandidate(
        candidate_id="interaction:fridge:open",
        behavior_type="INTERACT",
        source="test",
        target_id="container_fridge",
        target_name="refrigerator",
        goal_xyyaw=[1.0, 0.0, 0.0],
        interaction_command={"action": "open"},
        features={"distance_m": 1.0},
        metadata={"node_type": "container", "target_room_id": "room_kitchen"},
    )
    request = client.build_request(
        [fridge],
        {"enabled": True, "target_name": "apple", "object_labels": ["apple"]},
        {
            "nodes": [
                {
                    "id": "room_kitchen",
                    "type": "room",
                    "label": "unknown_room",
                    "centroid": [0.0, 0.0, 0.0],
                    "attributes": {
                        "room_attribute": "kitchen",
                        "room_attribute_confidence": 0.92,
                    },
                },
                {
                    "id": "room_bedroom",
                    "type": "room",
                    "label": "bedroom",
                    "centroid": [6.0, 0.0, 0.0],
                },
                {
                    "id": "portal_kitchen_bedroom",
                    "type": "portal",
                    "label": "door",
                    "attributes": {
                        "connected_room_ids": ["room_kitchen", "room_bedroom"]
                    },
                    "interaction": {"state": "open", "requires_interaction": True},
                },
                {
                    "id": "container_fridge",
                    "type": "container",
                    "label": "refrigerator",
                    "room_id": "room_kitchen",
                    "interaction": {"state": "closed", "requires_interaction": True},
                },
                {
                    "id": "container_dresser",
                    "type": "container",
                    "label": "dresser",
                    "room_id": "room_bedroom",
                    "interaction": {"state": "closed", "requires_interaction": True},
                },
            ]
        },
        {"robot_xy": [0.1, 0.1]},
    )

    reasoning = request["room_object_reasoning"]
    assert request["schema_version"] == 4
    assert reasoning["stage"] == "observed_room_target_plausibility"
    assert reasoning["target"] == {
        "name": "apple",
        "visible": False,
        "semantic_class": "food",
        "plausible_room_types": ["kitchen", "dining_room"],
        "plausible_container_types": [
            "refrigerator",
            "fridge",
            "cabinet",
            "pantry",
            "cupboard",
        ],
        "labels": ["apple"],
    }
    kitchen = next(room for room in reasoning["observed_rooms"] if room["id"] == "room_kitchen")
    assert kitchen["anchor_objects"] == [{"type": "refrigerator", "visible": False}]
    assert kitchen["target_plausibility"] == {
        "matches_semantic_prior": True,
        "evidence": [
            "room_type:kitchen",
            "anchor_object:refrigerator",
            "container:refrigerator",
        ],
    }
    assert reasoning["observed_portals"] == [
        {
            "id": "portal_kitchen_bedroom",
            "type": "door",
            "state": "open",
            "interaction_available": True,
            "connects": ["room_kitchen", "room_bedroom"],
        }
    ]
    assert "STAGE 1 (OBSERVED_ROOM_OBJECT_PLAUSIBILITY)" in request["instruction"]
    assert "STAGE 2 (EXECUTABLE_CANDIDATE_RANKING)" in request["instruction"]
    assert "return only the final Stage 2 JSON" in request["instruction"]


def test_room_object_reasoning_context_caps_observed_graph_evidence() -> None:
    semantic_graph = {
        "rooms": [
            {"id": f"room_{index}", "type": "kitchen", "anchor_objects": []}
            for index in range(ROOM_OBJECT_REASONING_MAX_ROOMS + 3)
        ],
        "portals": [
            {"id": f"portal_{index}", "type": "door", "connects": []}
            for index in range(ROOM_OBJECT_REASONING_MAX_PORTALS + 3)
        ],
        "containers": [
            {"id": f"container_{index}", "type": "refrigerator"}
            for index in range(ROOM_OBJECT_REASONING_MAX_CONTAINERS + 3)
        ],
    }

    reasoning = build_room_object_reasoning_context(
        {"mode": "object_goal", "target": {"name": "apple"}},
        semantic_graph,
    )

    assert len(reasoning["observed_rooms"]) == ROOM_OBJECT_REASONING_MAX_ROOMS
    assert len(reasoning["observed_portals"]) == ROOM_OBJECT_REASONING_MAX_PORTALS
    assert len(reasoning["observed_containers"]) == ROOM_OBJECT_REASONING_MAX_CONTAINERS
