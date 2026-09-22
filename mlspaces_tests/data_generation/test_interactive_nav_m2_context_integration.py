"""Pure-Python checks across recovered context and production M2 projection."""

from copy import deepcopy

from scripts.InteractiveNav.evaluation import m2_replay_eval as replay
from scripts.InteractiveNav.evaluation.m2_context_ablation import build_arm_requests, digest
from scripts.InteractiveNav.evaluation.m2_context_dataset import map_pose


def _context_fixture():
    nodes = [
        {"id": f"object_noise_{index}", "type": "object", "label": "book"}
        for index in range(90)
    ]
    nodes.extend([
        {"id": "room_1", "room_id": 1, "type": "room", "centroid": [0, 0, 0],
         "aabb_center": [0, 0, 0], "aabb_size": [10, 10, 3],
         "attributes": {"room_attribute": "living_room", "room_attribute_scores": {"living_room": 2}}},
        {"id": "room_2", "room_id": 2, "type": "room", "centroid": [10, 0, 0],
         "aabb_center": [10, 0, 0], "aabb_size": [10, 10, 3]},
        {"id": "tv", "type": "object", "label": "tv", "room_id": 1,
         "centroid": [2, 2, 0], "aabb_size": [2, 1, 1], "visible": False},
        {"id": "portal_1", "type": "portal", "centroid": [5, 0, 0],
         "connected_room_ids": [1, 2], "interaction": {"state": "closed", "is_interactable": True}},
        {"id": "container_1", "type": "container", "label": "cabinet", "room_id": 1,
         "centroid": [2, 3, 0], "interaction": {"state": "closed", "is_interactable": True}},
    ])
    graph = {"nodes": nodes, "frame_id": "map", "graph_revision": 4, "capture_step": 100}
    raw = [
        {"candidate_id": f"frontier:{index}", "behavior_type": "EXPLORE", "source": "test",
         "target_id": str(index), "target_name": "frontier", "goal_xyyaw": [index, 0, 0],
         "features": {"distance_m": index},
         "metadata": {"room_id": 1, "robot_room_id": 1, "cell_count": 10, "map_resolution": .1,
                      "hard_constraints_passed": True, "room_target_affinity": 1,
                      "room_target_affinity_reason": "room_target_semantic_match"}}
        for index in (1, 2)
    ]
    robot = {
        "initial_xy": [0, 0], "robot_xy": [1, 1], "position_frame_id": "map",
        "initial_position_source": "first_recorded_pose_not_verified_reset_spawn",
        "room_visit_history": [{"room_id": "room_1", "entry_step": 0, "entry_xy": [0, 0]},
                               {"room_id": "room_2", "entry_step": 10, "entry_xy": [6, 0]},
                               {"room_id": "room_1", "entry_step": 20, "entry_xy": [1, 1]}],
        "entered_room_ids": ["room_1", "room_2"],
        "decision_history": [{"decision_id": f"decision_{index}", "candidate_id": "frontier:1",
                              "behavior_type": "EXPLORE", "step": index, "result": "SUCCEEDED"}
                             for index in range(35)],
        "candidate_history": {"frontier:1": {"selection_count": 2, "last_result": "FAILED",
                                                "last_selected_step": 95}},
        "room_frontier_lengths": {"room_1": 2.0},
        "observed_room_frontier_lengths": {"room_1": 3.0},
        "frontier_length_source": "recorded_raw_candidate_pool_lower_bound",
        "frontier_length_complete": False,
    }
    return {
        "case_id": "synthetic", "source": {"recorder_step": 100},
        "raw_candidates": raw, "graph": graph,
        "target_context": {"enabled": True, "target_name": "toilet", "object_labels": ["toilet"]},
        "robot_context": robot, "legacy_request": {"candidates": [{"id": row["candidate_id"]} for row in raw]},
        "reconstruction": {"strict_version_aligned": True},
    }


def _production_request(record):
    symbols = replay._runtime_symbols()
    client = symbols["ModelPolicyClient"](symbols["ModelPolicyConfig"](mode="disabled"))
    return client.build_request(
        [symbols["BehaviorCandidate"](**item) for item in record["raw_candidates"]],
        record["target_context"], record["graph"], record["robot_context"],
    )


def test_production_receives_geometry_route_and_last_thirty_real_decisions():
    request = _production_request(_context_fixture())
    assert request["robot"]["initial_xy"] == [0.0, 0.0]
    assert request["robot"]["current_xy"] == [1.0, 1.0]
    assert request["robot"]["position_frame_id"] == request["graph"]["frame_id"] == "map"
    assert [row["room_id"] for row in request["robot"]["room_visit_history"]] == ["room_1", "room_2", "room_1"]
    assert len(request["recent_decisions"]) == 30
    assert request["recent_decisions"][0]["decision_id"] == "decision_5"
    assert request["recent_decisions"][-1]["decision_id"] == "decision_34"


def test_all_topology_survives_old_node_limit_without_ordinary_object_payloads():
    request = _production_request(_context_fixture())
    graph = request["graph"]
    assert len(graph["rooms"]) == 2
    assert len(graph["portals"]) == len(graph["containers"]) == 1
    room = graph["rooms"][0]
    assert {anchor["type"] for anchor in room["anchor_objects"]} == {"tv", "cabinet"}
    assert all(set(anchor) == {"type"} for anchor in room["anchor_objects"])
    assert "room_attribute_scores" not in room
    assert "objects" not in graph
    assert graph["portals"][0]["center_xy"] == [5.0, 0.0]
    assert room["aabb_size_xy"] == [10.0, 10.0]


def test_frontier_and_candidate_history_reach_actual_model_request():
    request = _production_request(_context_fixture())
    rooms = {room["id"]: room for room in request["graph"]["rooms"]}
    assert rooms["room_1"]["eligible_frontier_length_m"] == 2.0
    assert rooms["room_1"]["observed_frontier_length_m"] == 3.0
    assert rooms["room_2"]["observed_frontier_length_m"] is None
    candidates = {candidate["id"]: candidate for candidate in request["candidates"]}
    assert candidates["frontier:1"]["history"]["selection_count"] == 2
    assert candidates["frontier:1"]["history"]["last_result"] == "FAILED"
    assert "low_gain_repeat_count" not in candidates["frontier:1"]["history"]
    assert "last_frontier_shrink_m" not in candidates["frontier:1"]["history"]
    assert request["room_object_reasoning"] == {}
    for candidate in candidates.values():
        assert "room_target_affinity" not in candidate
        assert "room_target_affinity_reason" not in candidate
        assert "pre_score" not in candidate


def test_recovered_map_coordinates_are_not_reinterpreted_as_odom():
    pose = map_pose({"pose": [1, 2, 0], "stamp_sec": 1, "pose_frame_id": "odom", "graph_frame_id": "map",
                     "tf_map_from_odom": {"source_frame": "odom", "target_frame": "map", "stamp_sec": 2,
                                          "x": 2, "y": 1, "yaw": 0}})
    record = _context_fixture()
    record["robot_context"].update(robot_xy=pose["xy"], initial_xy=pose["xy"], position_frame_id=pose["frame_id"])
    request = _production_request(record)
    assert request["robot"]["current_xy"] == request["robot"]["initial_xy"] == [3.0, 3.0]


def test_context_arms_freeze_prompt_and_membership_without_mutating_source():
    record = _context_fixture()
    original = digest(deepcopy(record))
    arms, _ = build_arm_requests(record, "one frozen instruction")
    ids = [row["id"] for row in arms["full_context"]["candidates"]]
    assert ids
    for name, request in arms.items():
        assert request["instruction"] == "one frozen instruction"
        if name != "legacy_candidate_pool":
            assert [row["id"] for row in request["candidates"]] == ids
    assert digest(record) == original
    assert arms["no_recent_decisions"]["recent_decisions"] == []
    assert "room_visit_history" not in arms["no_room_route"]["robot"]


def test_replay_eligibility_and_history_age_match_current_projection_contract():
    record = _context_fixture()
    record["raw_candidates"][1]["metadata"]["hard_constraints_passed"] = False
    record["raw_candidates"].append({
        "candidate_id": "reobserve_portal:portal_1", "behavior_type": "NAVIGATE", "source": "test",
        "target_id": "portal_1", "target_name": "door", "goal_xyyaw": [4, 0, 0],
        "metadata": {"reobserve_interaction_target": True, "node_type": "portal"},
    })
    record["robot_context"]["room_frontier_lengths"] = {"room_1": 999.0}
    arms, diagnostics = build_arm_requests(record, "frozen")
    request = arms["full_context"]
    assert [row["id"] for row in request["candidates"]] == ["frontier:1"]
    assert request["candidates"][0]["history"]["last_selected_steps_ago"] == 5
    assert request["graph"]["rooms"][0]["eligible_frontier_length_m"] == 1.0
    assert request["graph"]["frontier_statistics"]["complete_map_frontier_inventory"] is False
    assert "reobserve_portal:portal_1" in diagnostics["all_actions_12_frontiers"]["deferred_reobserve_ids"]
    assert diagnostics["all_actions_12_frontiers"]["eligibility_limitations"]


def test_room_frontier_ablation_removes_duplicate_room_totals_but_not_local_frontier_fact():
    arms, _ = build_arm_requests(_context_fixture(), "frozen")
    request = arms["no_room_frontiers"]
    assert all("eligible_frontier_length_m" not in room for room in request["graph"]["rooms"])
    assert all("observed_frontier_length_m" not in room for room in request["graph"]["rooms"])
    assert all("room_frontier_length_m" not in candidate for candidate in request["candidates"])
    assert all(candidate["frontier_length_m"] == 1.0 for candidate in request["candidates"])
