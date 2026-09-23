import pytest

from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate, CandidateGenerator
from semantic_decision_py_pkg.model_policy import (
    ModelCircuitBreaker,
    ModelPolicyClient,
    ModelPolicyConfig,
    ROOM_OBJECT_REASONING_MAX_CONTAINERS,
    ROOM_OBJECT_REASONING_MAX_PORTALS,
    ROOM_OBJECT_REASONING_MAX_ROOMS,
    SUBGOAL_CONFIDENCE_CODES,
    SUBGOAL_REASON_CODES,
    build_subgoal_selection_response_schema,
    build_room_object_reasoning_context,
    compact_graph,
    compact_semantic_graph,
    public_decision_graph_context,
)
from semantic_mllm_py_pkg.client import MLLMResponse


def test_m2_budget_preserves_mission_candidates_and_input():
    client = ModelPolicyClient(ModelPolicyConfig(context_window_tokens=4096, max_tokens=1536))
    payload = {"instruction": "Rank current candidates", "mission": {"target": "requested"},
               "candidates": [{"id": "current:1", "effect": "approach_target"}],
               "recent_decisions": [{"reason": "history" * 1000}],
               "graph": {"nodes": [{"id": str(index), "detail": "geometry" * 500} for index in range(5)]}}
    context, output_tokens, metrics = client._bounded_http_context(payload)
    assert output_tokens == 512
    assert metrics["m2_input_utf8_bytes_after"] + output_tokens + 1024 <= 4096
    assert context["candidates"] == payload["candidates"]
    assert context["mission"] == payload["mission"]
    assert len(payload["graph"]["nodes"]) == 5
    assert metrics["m2_context_compacted"]


def test_m2_budget_rejects_oversized_mandatory_context():
    client = ModelPolicyClient(ModelPolicyConfig(context_window_tokens=2048))
    with pytest.raises(ValueError, match="mandatory mission/candidates"):
        client._bounded_http_context({"mission": {"instruction": "长" * 3000},
                                      "candidates": [{"id": "current"}]})


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


def test_subgoal_http_schema_is_strict_and_candidate_bounded(monkeypatch) -> None:
    candidate_a = make_candidate("frontier:a", 0.0, 1.0)
    candidate_b = make_candidate("interaction:fridge:open", 0.5, 2.0)
    client = ModelPolicyClient(ModelPolicyConfig(mode="http"))
    payload = client.build_request(
        [candidate_a, candidate_b],
        {"enabled": True, "target_name": "apple"},
        {},
    )
    captured = {}

    def fake_request_json(**kwargs):
        captured.update(kwargs)
        return MLLMResponse(
            payload={
                "ranked_ids": [candidate_a.candidate_id],
                "reason": "INFORMATION_GAIN",
                "confidence": "high",
            },
            latency_s=0.001,
        )

    monkeypatch.setattr(client._mllm_client, "request_json", fake_request_json)
    response = client._request_http(payload)
    schema = captured["response_schema"]
    body_schema = schema["schema"]
    assert response["ranked_ids"] == [candidate_a.candidate_id]
    assert schema["name"] == "subgoal_selection"
    assert schema["strict"] is True
    assert body_schema["additionalProperties"] is False
    assert body_schema["required"] == ["ranked_ids", "reason", "confidence"]
    assert body_schema["properties"]["ranked_ids"]["items"]["enum"] == [
        candidate_a.candidate_id,
        candidate_b.candidate_id,
    ]
    assert body_schema["properties"]["reason"]["enum"] == list(SUBGOAL_REASON_CODES)
    assert body_schema["properties"]["confidence"]["enum"] == list(SUBGOAL_CONFIDENCE_CODES)
    # M2 ranks compact graph/candidate text only.  Visual evidence belongs to
    # M3 after the approach has selected a concrete interaction target.
    assert "images" not in captured


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
                        "state": "blocked",
                        "requires_interaction": False,
                        "traversable": False,
                        "is_interactable": False,
                        "capability": "blocked",
                        "capability_source": "executor_feedback",
                        "failure_reason": "force_target_not_reached",
                    },
                }
            ],
            "edges": [],
        }
    )
    assert graph["nodes"][0]["interaction_state"] == "blocked"
    assert graph["nodes"][0]["interaction_capability"] == "blocked"
    assert graph["nodes"][0]["interaction_failure_reason"] == "force_target_not_reached"
    assert graph["nodes"][0]["connected_room_ids"] == [1, 2]


def test_compact_graph_redacts_portal_source_labels_from_mllm_context() -> None:
    graph = compact_graph(
        {
            "nodes": [
                {
                    "id": "portal_gt_portal_0001",
                    "type": "portal",
                    "label": "doorframe",
                    "name": "doorway_static_42",
                    "attributes": {
                        "source_object_name": "private_doorframe_root",
                        "instance_id": "gt_portal_0001",
                    },
                    "interaction": {"state": "unknown"},
                }
            ]
        }
    )

    node = graph["nodes"][0]
    assert node["id"] == "door_0001"
    assert node["label"] == node["name"] == "door_0001"
    assert "source_object_name" not in node
    assert "doorframe" not in str(graph).casefold()
    assert "doorway" not in str(graph).casefold()
    assert "gt_" not in str(graph).casefold()


def test_compact_semantic_graph_redacts_legacy_portal_identity() -> None:
    graph = compact_semantic_graph(
        {
            "nodes": [
                {
                    "id": "portal_door_x_17",
                    "type": "portal",
                    "label": "doorframe",
                    "name": "door_x_17",
                    "attributes": {
                        "instance_id": "gt_portal_0001",
                        "source_object_name": "private_doorframe_root",
                        "connected_room_ids": [1],
                    },
                    "interaction": {"state": "unknown"},
                }
            ]
        }
    )

    assert graph["portals"][0]["id"] == "door_0001"
    assert graph["portals"][0]["type"] == "door"
    serialized = str(graph).casefold()
    assert "door_x_17" not in serialized
    assert "doorframe" not in serialized
    assert "gt_" not in serialized


def test_semantic_graph_keeps_only_rooms_portals_and_containers() -> None:
    graph = compact_semantic_graph(
        {
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "label": "kitchen",
                    "centroid": [0, 0, 0],
                    "aabb_size": [2.0, 2.0, 1.0],
                },
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
                "centroid_xy": [0.0, 0.0],
                "aabb_center_xy": None,
                "aabb_size_xy": [2.0, 2.0],
                "eligible_frontier_length_m": None,
                "observed_frontier_length_m": None,
                "eligible_frontier_count": None,
                "observed_frontier_count": None,
                "anchor_objects": [
                    {"type": "refrigerator"}
                ],
            }
        ],
        "portals": [
            {
                "id": "portal_1",
                "type": "portal",
                "state": "closed",
                "interaction_available": True,
                "connects": ["room_1", "room_2"],
                "center_xy": None,
            }
        ],
        "containers": [
            {
                "id": "container_1",
                "type": "refrigerator",
                "state": "closed",
                "interaction_available": True,
                "room_id": "room_1",
                "center_xy": None,
            }
        ],
        "current_room": "room_1",
    }


@pytest.mark.parametrize("max_graph_nodes", [1, 80])
def test_model_current_room_uses_same_containment_as_candidates(max_graph_nodes) -> None:
    graph = {
        "nodes": [
            {
                "id": "room_6",
                "type": "room",
                "room_id": 6,
                "centroid": [0.0, 2.5, 0.0],
                "aabb_center": [0.0, 2.5, 0.0],
                "aabb_size": [2.0, 2.0, 1.0],
            },
            {
                "id": "room_7",
                "type": "room",
                "room_id": 7,
                "centroid": [3.0, 0.0, 0.0],
                "aabb_center": [3.0, 0.0, 0.0],
                "aabb_size": [8.0, 2.0, 1.0],
            },
        ]
    }
    robot_xy = (0.0, 0.0)
    candidates = CandidateGenerator().generate(
        {
            "proposals": [
                {
                    "proposal_id": "same_room",
                    "goal_xyyaw": [0.5, 0.0, 0.0],
                    "frontier_point": [0.5, 0.0],
                    "raw_features": {"distance_m": 0.5},
                }
            ]
        },
        graph,
        robot_xy=robot_xy,
    )
    client = ModelPolicyClient(
        ModelPolicyConfig(mode="disabled", max_graph_nodes=max_graph_nodes)
    )
    request = client.build_request(
        candidates,
        {"enabled": True, "target_name": "toilet"},
        graph,
        {"robot_xy": robot_xy},
    )

    assert candidates[0].metadata["robot_room_id"] == 7
    assert request["candidates"][0]["robot_room_id"] == "room_7"
    assert request["robot"]["current_room"] == "room_7"
    assert request["graph"]["current_room"] == "room_7"


@pytest.mark.parametrize("aabb_size", [None, [2.0, 2.0, 1.0]])
def test_model_current_room_does_not_fall_back_to_nearest_centroid(aabb_size) -> None:
    room = {"id": "room_1", "type": "room", "centroid": [0.0, 0.0, 0.0]}
    if aabb_size is not None:
        room["aabb_size"] = aabb_size
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    request = client.build_request(
        [make_candidate("frontier", 0.0, 1.0)],
        {},
        {"nodes": [room]},
        {"robot_xy": [3.0, 0.0]},
    )

    assert "current_room" not in request["robot"]
    assert "current_room" not in request["graph"]


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
    assert "room_attribute_scores" not in graph["rooms"][0]
    assert graph["remembered_objects_without_room"] == [
        {"type": "refrigerator", "id": "container_fridge", "observed_xy": None}
    ]


def test_semantic_graph_marks_graph_only_portal_child_as_unobserved() -> None:
    graph = compact_semantic_graph(
        {
            "nodes": [
                {
                    "id": "room_1000000",
                    "type": "room",
                    "label": "unobserved_portal_room",
                    "centroid": [1.0, 0.0, 0.1],
                    "attributes": {
                        "is_potential_room": True,
                        "observed_free_space": False,
                        "source_portal_id": "portal_door_1",
                    },
                }
            ]
        }
    )

    assert graph["rooms"] == [
        {
            "id": "room_1000000",
            "type": "unobserved_portal_room",
            "potential_room": True,
            "observed_free_space": False,
            "source_portal_id": "portal_door_1",
            "centroid_xy": [1.0, 0.0],
            "aabb_center_xy": None,
            "aabb_size_xy": None,
            "eligible_frontier_length_m": None,
            "observed_frontier_length_m": None,
            "eligible_frontier_count": None,
            "observed_frontier_count": None,
        }
    ]
    reasoning = build_room_object_reasoning_context(
        {"mode": "explore_all"}, graph
    )
    assert reasoning["observed_rooms"][0]["potential_room"] is True
    assert reasoning["observed_rooms"][0]["observed_free_space"] is False


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
        {"type": "refrigerator", "distance_m": 0.8}
    ]
    assert "expected_visible_unknown_area_m2" in request["instruction"]
    assert "distance breaks ties" in request["instruction"]


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
    assert request["robot"] == {
        "current_xy": [1.0, 2.0], "initial_xy": None,
        "position_frame_id": None, "room_visit_history": [],
    }
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
    assert request["mission"] == {"mode": "interaction_coverage_exploration"}
    assert "interaction-coverage exploration mission" in request["instruction"]
    assert "text-only decision" in request["instruction"]
    assert "INTERACTION_COVERAGE" in SUBGOAL_REASON_CODES


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


def test_m2_retries_one_timeout_then_records_success_metrics(monkeypatch) -> None:
    client = ModelPolicyClient(
        ModelPolicyConfig(
            mode="http",
            model="strong-text-model",
            timeout_s=30.0,
            timeout_retry_count=1,
            timeout_retry_backoff_s=0.25,
        )
    )
    candidate = make_candidate("frontier:a", 0.0, 1.0)
    payload = client.build_request([candidate], {}, {})
    calls = []
    sleeps = []

    def fake_request_json(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return MLLMResponse(payload=None, latency_s=30.0, error="timed out")
        return MLLMResponse(
            payload={
                "ranked_ids": [candidate.candidate_id],
                "reason": "INFORMATION_GAIN",
                "confidence": "medium",
            },
            latency_s=0.2,
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        )

    monkeypatch.setattr(client._mllm_client, "request_json", fake_request_json)
    monkeypatch.setattr(
        "semantic_decision_py_pkg.model_policy.time.sleep", sleeps.append
    )

    response = client._request(payload, metrics_context={"episode_index": 7})

    assert response["ranked_ids"] == [candidate.candidate_id]
    assert len(calls) == 2
    assert sleeps == [0.25]
    assert calls[0]["metrics_context"]["m2_request_attempt"] == 1
    assert calls[0]["metrics_context"]["m2_is_timeout_retry"] is False
    assert calls[1]["metrics_context"]["m2_request_attempt"] == 2
    assert calls[1]["metrics_context"]["m2_is_timeout_retry"] is True
    assert client.last_error == ""
    assert client.last_metrics["model"] == "strong-text-model"
    assert client.last_metrics["request_attempts"] == 2
    assert client.last_metrics["timeout_retry_count"] == 1
    assert client.last_metrics["attempts"][0]["timeout"] is True
    assert client.last_metrics["attempts"][1]["timeout"] is False


def test_m2_does_not_retry_non_timeout_error(monkeypatch) -> None:
    client = ModelPolicyClient(
        ModelPolicyConfig(
            mode="http",
            timeout_retry_count=2,
            timeout_retry_backoff_s=0.0,
        )
    )
    candidate = make_candidate("frontier:a", 0.0, 1.0)
    payload = client.build_request([candidate], {}, {})
    calls = []

    def fake_request_json(**kwargs):
        calls.append(kwargs)
        return MLLMResponse(payload=None, latency_s=0.01, error="HTTP 401")

    monkeypatch.setattr(client._mllm_client, "request_json", fake_request_json)

    assert client._request(payload) is None
    assert len(calls) == 1
    assert client.last_error == "HTTP 401"
    assert client.last_metrics["request_attempts"] == 1
    assert client.last_metrics["timeout_retry_count"] == 0


def test_m2_timeout_retries_are_hard_capped(monkeypatch) -> None:
    client = ModelPolicyClient(
        ModelPolicyConfig(
            mode="http",
            timeout_retry_count=99,
            timeout_retry_backoff_s=0.0,
        )
    )
    candidate = make_candidate("frontier:a", 0.0, 1.0)
    payload = client.build_request([candidate], {}, {})
    calls = []

    def fake_request_json(**kwargs):
        calls.append(kwargs)
        return MLLMResponse(payload=None, latency_s=1.0, error="deadline exceeded")

    monkeypatch.setattr(client._mllm_client, "request_json", fake_request_json)

    assert client._request(payload) is None
    assert len(calls) == 4
    assert client.last_metrics["timeout_retry_limit"] == 3
    assert client.last_metrics["timeout_retry_count"] == 3


def test_model_request_exposes_pre_score_and_route_hint_without_geometry() -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled", include_pre_scores=True))
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
    assert request["schema_version"] == 5
    assert "NEXT_ROUTE_PORTAL when observed topology establishes a prerequisite route" in request["instruction"]
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


def test_new_room_payload_and_guard_prefer_unentered_room_without_overriding_postdoor(
    monkeypatch,
) -> None:
    client = ModelPolicyClient(
        ModelPolicyConfig(mode="mock", pre_score_guard_margin=0.75)
    )
    entered = BehaviorCandidate(
        candidate_id="frontier:entered",
        behavior_type="EXPLORE",
        source="test",
        target_id="entered",
        target_name="entered",
        goal_xyyaw=[1.0, 0.0, 0.0],
        features={"distance_m": 1.0},
        metadata={"room_status": "entered_room", "target_room_id": 1},
    )
    new_room = BehaviorCandidate(
        candidate_id="frontier:new_room",
        behavior_type="EXPLORE",
        source="test",
        target_id="new_room",
        target_name="new_room",
        goal_xyyaw=[3.0, 0.0, 0.0],
        features={"distance_m": 3.0},
        metadata={
            "room_status": "unentered_new_room",
            "target_room_id": 2,
            "robot_room_id": 1,
            "potential_room": True,
            "source_portal_id": "portal_1",
            "room_attribute": "kitchen",
            "room_attribute_confidence": 0.9,
            "room_target_affinity": 1.0,
            "room_target_affinity_reason": "room_target_semantic_match",
        },
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, metrics_context=None: {
            "ranked_ids": [entered.candidate_id, new_room.candidate_id],
            "reason": "INFORMATION_GAIN",
            "confidence": "medium",
        },
    )

    selected = client.select(
        [entered, new_room],
        target_context={"enabled": True, "target_name": "apple"},
        robot_context={
            "entered_room_ids": ["room_1"],
            "candidate_pre_scores": {entered.candidate_id: 0.5, new_room.candidate_id: 1.5},
            "candidate_decision_hints": {new_room.candidate_id: "NEW_ROOM_FRONTIER"},
        },
    )

    assert selected is new_room
    assert client.last_reason == "PRE_SCORE_GUARD_NEW_ROOM_FRONTIER"
    option = next(item for item in client.last_candidate_groups if item["id"] == new_room.candidate_id)
    assert option["room_status"] == "unentered_new_room"
    assert option["potential_room"] is True
    assert "room_target_affinity" not in option
    request = client.build_request(
        [new_room],
        {"enabled": True, "target_name": "apple"},
        {},
        {"entered_room_ids": ["room_1"]},
    )
    assert request["robot"]["room_visit_history"] == []
    assert "compatible or unknown unentered rooms" in request["instruction"]

    traversal = BehaviorCandidate(
        candidate_id="traverse:portal_1",
        behavior_type="NAVIGATE",
        source="test",
        target_id="portal_1",
        target_name="door",
        goal_xyyaw=[2.0, 0.0, 0.0],
        features={"distance_m": 2.0},
        metadata={"post_interaction_traversal": True},
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload, metrics_context=None: {
            "ranked_ids": [traversal.candidate_id, new_room.candidate_id],
            "reason": "UNLOCK_ROUTE",
            "confidence": "high",
        },
    )
    selected = client.select(
        [traversal, new_room],
        robot_context={
            "candidate_pre_scores": {traversal.candidate_id: 1.0, new_room.candidate_id: 4.0},
            "candidate_decision_hints": {
                traversal.candidate_id: "POST_INTERACTION_TRAVERSE",
                new_room.candidate_id: "NEW_ROOM_FRONTIER",
            },
        },
    )
    assert selected is traversal


def test_request_leaves_room_target_reasoning_to_model() -> None:
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
    assert request["schema_version"] == 5
    assert reasoning == {}
    kitchen = next(room for room in request["graph"]["rooms"] if room["id"] == "room_kitchen")
    assert kitchen["anchor_objects"] == [{"type": "refrigerator"}]
    assert "target_plausibility" not in kitchen
    assert request["graph"]["portals"] == [
        {
            "id": "portal_kitchen_bedroom",
            "type": "portal",
            "state": "open",
            "interaction_available": True,
            "connects": ["room_kitchen", "room_bedroom"],
            "center_xy": None,
        }
    ]
    stages = ["1. EVIDENCE", "2. DEPENDENCIES", "3. COMPATIBILITY", "4. PROGRESS", "5. VALIDATE"]
    positions = [request["instruction"].index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "return only the final JSON, not the reasoning" in request["instruction"]
    assert "extremely low priority" in request["instruction"]
    assert "NEGATIVE PRIORITY" in request["instruction"]
    assert "POSITIVE CONTAINER PRIORITY" in request["instruction"]
    assert "CLOSED DOOR PRIORITY" in request["instruction"]
    assert "FAILURE MEMORY" in request["instruction"]
    assert "over generic frontiers" in request["instruction"]
    assert "unknown room" in request["instruction"]
    assert "newly accessible, unentered room" in request["instruction"]
    assert "toilet" not in request["instruction"]
    assert "refrigerator" not in request["instruction"]


def test_room_object_reasoning_context_caps_observed_graph_evidence() -> None:
    semantic_graph = {
        "rooms": [
            {"id": f"room_{index}", "type": "kitchen", "anchor_objects": []}
            for index in range(ROOM_OBJECT_REASONING_MAX_ROOMS + 3)
        ],
        "portals": [
            {"id": f"portal_{index}", "type": "portal", "connects": []}
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


def test_public_context_keeps_full_structural_graph_geometry_and_unknown_frontiers() -> None:
    graph = {
        "frame_id": "map", "graph_revision": 42, "capture_step": 100,
        "nodes": [{"id": f"object_{i}", "type": "object", "label": "apple"} for i in range(90)] + [
            {
                "id": "room_1", "type": "room", "room_id": 1,
                "centroid": [2, 3, 0], "aabb_center": [3, 3, 0], "aabb_size": [8, 6, 1],
                "attributes": {"room_attribute": "kitchen", "room_attribute_source": "mllm_room_attribute_inference", "room_attribute_confidence": 0.9, "room_attribute_scores": {"kitchen": 2}},
            },
            {"id": "room_2", "type": "room"},
            {"id": "portal_1", "type": "portal", "centroid": [6, 3, 1]},
            {"id": "container_1", "type": "container", "label": "refrigerator", "room_id": 1},
            {"id": "toilet_memory", "type": "object", "label": "toilet", "centroid": [15, 5, 0], "observation_count": 2, "is_currently_visible": False, "attributes": {"last_observation_frame_index": 19, "max_visible_pixels": 12}},
        ],
    }
    compact = compact_semantic_graph(graph, {
        "robot_xy": [2, 3], "room_frontier_lengths": {"room_1": 1.2},
        "observed_room_frontier_lengths": {"room_1": 3.4},
    }, max_nodes=1)
    room = compact["rooms"][0]
    assert len(compact["rooms"]) == 2
    assert room["centroid_xy"] == [2, 3]
    assert room["aabb_center_xy"] == [3, 3]
    assert room["aabb_size_xy"] == [8, 6]
    assert room["eligible_frontier_length_m"] == 1.2
    assert room["observed_frontier_length_m"] == 3.4
    assert compact["rooms"][1]["eligible_frontier_length_m"] is None
    assert room["room_attribute_source"] == "mllm_room_attribute_inference"
    assert "room_attribute_scores" not in room
    assert room["anchor_objects"] == [{"type": "refrigerator"}]
    assert compact["portals"][0]["center_xy"] == [6, 3]
    memory = compact["remembered_objects_without_room"][0]
    assert memory["observed_xy"] == [15, 5]
    assert memory["observation_count"] == 2
    assert memory["last_observation_frame_index"] == 19
    assert "visible" not in memory
    assert compact["graph_revision"] == 42
    assert compact["frame_id"] == "map"


def test_robot_context_keeps_revisits_and_thirty_actual_decisions() -> None:
    history = [{"decision_id": f"d{i}", "step": i} for i in range(40)]
    visits = [{"room_id": room, "entry_step": step, "entry_xy": [step, 0]} for step, room in enumerate([1, 2, 1])]
    request = ModelPolicyClient().build_request(
        [make_candidate("frontier", 0, 1)], {}, {},
        {"robot_xy": [4, 5], "initial_xy": [1, 2], "initial_position_source": {"kind": "first_candidate_observation", "observation_step": 0}, "room_visit_history": visits, "decision_history": history},
    )
    assert request["robot"]["initial_xy"] == [1, 2]
    assert request["robot"]["current_xy"] == [4, 5]
    assert request["robot"]["position_frame_id"] is None
    assert [v["room_id"] for v in request["robot"]["room_visit_history"]] == ["room_1", "room_2", "room_1"]
    assert request["recent_decisions"] == history[-30:]


def test_pre_score_ablation_never_restores_semantic_guesses() -> None:
    candidate = make_candidate("frontier", 0, 1)
    candidate.metadata.update({"room_target_affinity": 1, "room_target_affinity_reason": "room_target_semantic_match"})
    context = {
        "candidate_pre_scores": {"frontier": 3},
        "candidate_pre_score_terms": {"frontier": {"distance": -1}},
        "candidate_decision_hints": {"frontier": "PLAUSIBLE_TARGET_CONTAINER"},
    }
    requests = [ModelPolicyClient(ModelPolicyConfig(include_pre_scores=enabled)).build_request([candidate], {"enabled": True, "target_name": "toilet"}, {}, context) for enabled in (False, True)]
    assert requests[0]["instruction"] == requests[1]["instruction"]
    for request in requests:
        option = request["candidates"][0]
        assert "room_target_affinity" not in option
        assert "room_target_affinity_reason" not in option
        assert "decision_hint" not in option
        assert request["room_object_reasoning"] == {}
    assert "pre_score" not in requests[0]["candidates"][0]
    assert requests[1]["candidates"][0]["pre_score"] == 3
    context["candidate_decision_hints"]["frontier"] = "NEW_ROOM_FRONTIER_HIGH_CONFIDENCE_MISMATCH"
    candidate.metadata["room_status"] = "unentered_high_confidence_mismatch"
    option = ModelPolicyClient().build_request([candidate], {}, {}, context)["candidates"][0]
    assert option["decision_hint"] == "NEW_ROOM_FRONTIER"
    assert option["room_status"] == "unentered_new_room"


def test_metrics_freeze_exact_public_request_before_network(monkeypatch) -> None:
    client = ModelPolicyClient(ModelPolicyConfig(mode="http"))
    payload = client.build_request([make_candidate("frontier", 0, 1)], {}, {})
    captured = {}
    def fake_request(payload_arg, metrics_context=None):
        captured.update(metrics_context)
        payload_arg["robot"]["current_xy"] = [99, 99]
        return {"ranked_ids": ["frontier"]}
    monkeypatch.setattr(client, "_request_http", fake_request)
    client._request(payload, metrics_context={"candidate_sequence": 3})
    assert captured["public_request"]["robot"]["current_xy"] is None
    assert captured["public_request"]["instruction"] == payload["instruction"]
    assert captured["request_started_ts"] > 0
    assert captured["candidate_sequence"] == 3


def test_online_graph_snapshot_keeps_geometry_after_eighty_objects_and_redacts_private_fields():
    graph = {
        "frame_id": "map", "capture_step": 21,
        "nodes": [{"id": f"o{i}", "type": "object", "label": "apple"} for i in range(90)] + [
            {"id": "room_1", "type": "room", "room_id": 1, "centroid": [0, 0, 0], "aabb_center": [0, 0, 0], "aabb_size": [4, 4, 1], "attributes": {"room_attribute": "kitchen", "room_attribute_confidence": 0.9, "room_attribute_source": "mllm", "private_gt": "secret"}},
            {"id": "portal_gt_door_1", "type": "portal", "label": "private_doorframe", "centroid": [1, 0, 0], "attributes": {"instance_id": "door_1", "source_object_name": "secret"}},
            *[{"id": f"fridge{i}", "type": "object", "label": "refrigerator", "room_id": 1, "aabb_size": [2, 2, 2]} for i in range(7)],
            {"id": "tv", "type": "object", "label": "tv", "room_id": 1},
        ],
    }
    snapshot = public_decision_graph_context(graph)
    request = ModelPolicyClient().build_request([make_candidate("frontier", 0, 1)], {}, snapshot, {"robot_xy": [0, 0]})
    public_graph = request["graph"]
    assert public_graph["current_room"] == "room_1"
    assert public_graph["rooms"][0]["aabb_size_xy"] == [4, 4]
    assert public_graph["rooms"][0]["room_attribute_source"] == "mllm"
    assert public_graph["rooms"][0]["anchor_objects"] == [{"type": "refrigerator"}, {"type": "tv"}]
    assert public_graph["portals"][0]["center_xy"] == [1, 0]
    assert public_graph["frame_id"] == "map"
    assert "secret" not in str(snapshot)
    assert "private_doorframe" not in str(snapshot)


def test_unknown_history_metrics_are_not_fabricated_as_zero_and_frontier_scope_survives():
    candidate = make_candidate("frontier", 0, 1)
    request = ModelPolicyClient().build_request([candidate], {}, {}, {
        "candidate_history": {"frontier": {"selection_count": 2, "last_result": "FAILED", "last_selected_step": 10, "low_gain_repeat_count": None}},
        "frontier_statistics_source": {"scope": "logged_raw_candidate_pool_lower_bound", "complete": False},
    })
    assert request["candidates"][0]["history"] == {"selection_count": 2, "last_result": "FAILED"}
    assert request["graph"]["frontier_statistics"] == {"scope": "logged_raw_candidate_pool_lower_bound", "complete": False}
    request = ModelPolicyClient().build_request([candidate], {}, {}, {
        "candidate_history": {"frontier": {"selection_count": 0, "last_selected_steps_ago": 0, "low_gain_repeat_count": 0, "last_frontier_shrink_m": 0}},
    })
    assert request["candidates"][0]["history"]["last_frontier_shrink_m"] == 0
    assert request["candidates"][0]["history"]["low_gain_repeat_count"] == 0


@pytest.mark.parametrize("graph_xy, expected_room", [([8, -2], "room_2"), (None, None)])
def test_online_graph_pose_controls_robot_room_without_changing_candidate_target_room(graph_xy, expected_room):
    graph = {"frame_id": "map", "nodes": [
        {"id": "room_1", "type": "room", "aabb_center": [1, 2, 0], "aabb_size": [2, 2, 1]},
        {"id": "room_2", "type": "room", "aabb_center": [8, -2, 0], "aabb_size": [2, 2, 1]},
    ]}
    candidate = make_candidate("frontier", 0, 1)
    candidate.metadata.update(robot_room_id=1, target_room_id=1)
    request = ModelPolicyClient().build_request([candidate], {}, graph, {
        "robot_xy": [1, 2], "position_frame_id": "odom", "robot_graph_xy": graph_xy,
        "robot_graph_frame_id": "map", "robot_graph_pose_source": {"source_xy": [1, 2], "source_frame_id": "odom"},
    })
    assert request["robot"].get("current_room") == expected_room
    assert request["robot"]["current_xy"] == graph_xy
    assert request["candidates"][0].get("robot_room_id") == expected_room
    assert request["candidates"][0]["room_id"] == "room_1"
    assert candidate.metadata["robot_room_id"] == 1


def test_explicit_conflicting_frames_without_transform_have_no_robot_room():
    graph = {"frame_id": "map", "nodes": [{"id": "room_1", "type": "room", "aabb_center": [0, 0, 0], "aabb_size": [2, 2, 1]}]}
    request = ModelPolicyClient().build_request([make_candidate("frontier", 0, 1)], {}, graph, {"robot_xy": [0, 0], "position_frame_id": "odom"})
    assert "current_room" not in request["robot"]
    assert "current_room" not in request["graph"]
