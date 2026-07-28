from __future__ import annotations

import math

from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.post_interaction_traversal import (
    build_post_interaction_traversal_candidate,
    inject_pending_traversal,
    is_terminal_post_interaction_traversal_failure,
    pending_priority_candidate,
    portal_center_xy,
)


def portal_interaction_candidate() -> dict:
    return {
        "candidate_id": "interaction:portal_door_1:open",
        "behavior_type": "INTERACT",
        "target_id": "portal_door_1",
        "target_name": "door_1",
        "goal_xyyaw": [-1.0, 0.0, 0.0],
        "portal_center_xy": [0.0, 0.0],
        "interaction_command": {
            "node_id": "portal_door_1",
            "object_id": "door_1",
            "action": "open",
            "interaction_approach_pose_xyyaw": [-1.0, 0.0, 0.0],
        },
        "features": {"confidence": 0.9},
        "metadata": {
            "node_type": "portal",
            "semantic_name": "door",
            "connected_room_ids": [1, 2],
        },
        "decision_id": "decision_000001",
    }


def test_success_feedback_builds_immediate_traversal_without_open_graph_update() -> None:
    candidate = build_post_interaction_traversal_candidate(
        portal_interaction_candidate(),
        {
            "status": "SUCCEEDED",
            "success": True,
            "detail": {
                "event_id": "object_skill_000001",
                "node_id": "portal_door_1",
                "action": "open",
            },
        },
        robot_xy=(-1.0, 0.0),
        traversal_distance_m=0.9,
    )

    assert candidate is not None
    assert candidate.candidate_id == "traverse:portal_door_1:object_skill_000001"
    assert candidate.goal_xyyaw == [0.9, 0.0, 0.0]
    assert math.isclose(candidate.features["distance_m"], 1.9)
    assert candidate.metadata["post_interaction_traversal"] is True
    assert candidate.metadata["decision_local_transition"] is True


def test_cached_traversal_is_injected_and_selected_before_frontier() -> None:
    pending = build_post_interaction_traversal_candidate(
        portal_interaction_candidate(),
        {
            "status": "SUCCEEDED",
            "detail": {"event_id": "event_1", "action": "open"},
        },
    )
    assert pending is not None
    pending_payload = pending.to_dict()
    pending_payload["metadata"]["source_episode_id"] = "episode_1"
    snapshot = {
        "episode_id": "episode_1",
        "candidate_count": 1,
        "candidates": [
            BehaviorCandidate(
                candidate_id="frontier:1",
                behavior_type="EXPLORE",
                source="test",
                target_id="1",
                target_name="frontier",
                goal_xyyaw=[2.0, 0.0, 0.0],
            ).to_dict()
        ],
    }

    assert inject_pending_traversal(snapshot, pending_payload) is True
    eligible = [BehaviorCandidate(**item) for item in snapshot["candidates"]]
    selected = pending_priority_candidate(eligible, pending.candidate_id)

    assert snapshot["candidate_count"] == 2
    assert selected is not None
    assert selected.candidate_id == pending.candidate_id


def test_graph_published_traversal_deduplicates_cached_candidate() -> None:
    pending = build_post_interaction_traversal_candidate(
        portal_interaction_candidate(),
        {"status": "SUCCEEDED", "detail": {"event_id": "event_1", "action": "open"}},
    )
    assert pending is not None
    snapshot = {"candidate_count": 1, "candidates": [pending.to_dict()]}

    assert inject_pending_traversal(snapshot, pending.to_dict()) is False
    assert snapshot["candidate_count"] == 1


def test_non_open_or_non_portal_feedback_does_not_build_traversal() -> None:
    active = portal_interaction_candidate()
    assert (
        build_post_interaction_traversal_candidate(
            active, {"status": "FAILED", "detail": {"action": "open"}}
        )
        is None
    )
    assert (
        build_post_interaction_traversal_candidate(
            active, {"status": "SUCCEEDED", "detail": {"action": "close"}}
        )
        is None
    )
    active["metadata"]["node_type"] = "container"
    assert (
        build_post_interaction_traversal_candidate(
            active, {"status": "SUCCEEDED", "detail": {"action": "open"}}
        )
        is None
    )


def test_portal_center_prefers_interaction_reference_geometry() -> None:
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "centroid": [5.0, 5.0, 1.0],
                "attributes": {"interaction_reference_aabb_center": [1.0, 2.0, 1.0]},
            }
        ]
    }

    assert portal_center_xy(graph, "portal_1") == [1.0, 2.0]


def test_failed_one_shot_traversal_is_terminal_but_target_preemption_is_not() -> None:
    candidate_id = "traverse:portal_1:open_event_1"
    assert is_terminal_post_interaction_traversal_failure(
        candidate_id, "NAVIGATE", "FAILED"
    )
    assert is_terminal_post_interaction_traversal_failure(
        candidate_id, "NAVIGATE", "REJECTED"
    )
    assert not is_terminal_post_interaction_traversal_failure(
        candidate_id,
        "NAVIGATE",
        "CANCELED",
        {"reason": "preempted_by_target"},
    )
    assert not is_terminal_post_interaction_traversal_failure(
        candidate_id, "NAVIGATE", "SUCCEEDED"
    )
    assert not is_terminal_post_interaction_traversal_failure(
        "frontier:1", "NAVIGATE", "FAILED"
    )
