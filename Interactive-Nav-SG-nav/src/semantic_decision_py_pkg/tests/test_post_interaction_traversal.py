from __future__ import annotations

import math

from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.post_interaction_traversal import (
    PostInteractionRefreshConfig,
    PostInteractionRefreshGate,
    build_post_interaction_traversal_candidate,
    inject_pending_traversal,
    is_terminal_post_interaction_traversal_failure,
    pending_priority_candidate,
    portal_center_xy,
    portal_open_confirmation,
    reproject_post_interaction_traversal_candidate,
)


def refresh_snapshot(
    *,
    sequence: int,
    graph_revision: int,
    observation_step: int | None,
    portal_id: str = "",
    portal_state: str = "",
    traversable: bool | None = None,
    requires_interaction: bool | None = None,
) -> dict:
    snapshot = {
        "episode_id": "episode_1",
        "sequence": sequence,
        "graph_revision": graph_revision,
        "exploration_context": {"observation_step": observation_step},
    }
    if portal_id:
        node = {
            "id": portal_id,
            "type": "portal",
            "interaction_state": portal_state,
            "traversable": traversable,
            "requires_interaction": requires_interaction,
        }
        snapshot["graph_context"] = {"nodes": [node], "edges": []}
    return snapshot


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


def test_post_open_traversal_uses_safe_opposite_interaction_options() -> None:
    active = portal_interaction_candidate()
    active["goal_xyyaw"] = [-1.10, 0.0, 0.0]
    active["interaction_command"]["interaction_approach_pose_xyyaw"] = [
        -1.10,
        0.0,
        0.0,
    ]
    active["metadata"].update(
        {
            "portal_aabb_center_xy": [0.0, 0.0],
            "portal_aabb_size_xy": [0.20, 2.0],
            # The interaction reference is narrower and offset from the full
            # semantic-node AABB.  Clearance must use this full footprint.
            "portal_clearance_aabb_center_xy": [0.50, 0.0],
            "portal_clearance_aabb_size_xy": [1.20, 2.0],
            # These are the original two-sided portal approach options.  The
            # source-side and in-AABB poses must not survive a post-open goal.
            "goal_xyyaw_candidates": [
                [-1.10, 0.0, 0.0],
                [1.10, 0.0, math.pi],
                [1.35, 0.20, math.pi],
                [0.03, 0.0, 0.0],
            ],
        }
    )

    pending = build_post_interaction_traversal_candidate(
        active,
        {"status": "SUCCEEDED", "detail": {"action": "open"}},
        traversal_distance_m=0.9,
    )

    assert pending is not None
    assert pending.goal_xyyaw == [1.35, 0.20, 0.0]
    assert pending.metadata["goal_xyyaw_candidates"] == [[1.35, 0.20, 0.0]]
    assert (
        pending.metadata["post_interaction_traversal_goal_source"]
        == "opposite_interaction_approach_candidates"
    )

    snapshot = refresh_snapshot(
        sequence=11,
        graph_revision=6,
        observation_step=20,
        portal_id="portal_door_1",
        portal_state="open",
        traversable=True,
        requires_interaction=False,
    )
    snapshot["graph_context"]["nodes"][0]["attributes"] = {
        "interaction_reference_aabb_center": [0.2, 0.0, 0.0],
        "interaction_reference_aabb_size": [0.20, 2.0, 2.0],
    }
    snapshot["graph_context"]["nodes"][0]["aabb_center"] = [0.70, 0.0, 0.0]
    snapshot["graph_context"]["nodes"][0]["aabb_size"] = [1.20, 2.0, 2.0]
    projected = reproject_post_interaction_traversal_candidate(
        pending.to_dict(), snapshot
    )

    assert projected is not None
    assert projected["goal_xyyaw"] == [1.55, 0.20, 0.0]
    assert projected["metadata"]["goal_xyyaw_candidates"] == [[1.55, 0.20, 0.0]]


def test_post_open_traversal_reprojects_from_the_fresh_portal_geometry() -> None:
    pending = build_post_interaction_traversal_candidate(
        portal_interaction_candidate(),
        {"status": "SUCCEEDED", "detail": {"event_id": "event_1", "action": "open"}},
        robot_xy=(-1.0, 0.0),
        traversal_distance_m=0.9,
    )
    assert pending is not None
    snapshot = refresh_snapshot(
        sequence=11,
        graph_revision=6,
        observation_step=20,
        portal_id="portal_door_1",
        portal_state="open",
        traversable=True,
        requires_interaction=False,
    )
    snapshot["robot_xy"] = [-1.0, 0.0]
    snapshot["graph_context"]["nodes"][0]["aabb_center"] = [0.2, 0.0, 0.0]

    projected = reproject_post_interaction_traversal_candidate(
        pending.to_dict(), snapshot, traversal_distance_m=0.9
    )

    assert projected is not None
    assert pending.goal_xyyaw == [0.9, 0.0, 0.0]
    assert projected["goal_xyyaw"] == [1.1, 0.0, 0.0]
    assert projected["metadata"]["goal_xyyaw_candidates"] == [[1.1, 0.0, 0.0]]
    assert projected["metadata"]["post_interaction_reprojected"] is True
    assert projected["metadata"]["post_interaction_reprojected_graph_revision"] == 6
    assert projected["metadata"]["post_interaction_reprojected_portal_center_xy"] == [
        0.2,
        0.0,
    ]


def test_post_interaction_refresh_gate_releases_on_one_graph_update_without_rgb() -> None:
    gate = PostInteractionRefreshGate(
        PostInteractionRefreshConfig(
            min_candidate_updates=1,
            timeout_s=12.0,
            require_graph_revision=True,
            require_observation_step=False,
        )
    )
    baseline = refresh_snapshot(sequence=10, graph_revision=5, observation_step=20)
    gate.begin(baseline, portal_id="portal_door_1", now=100.0)

    released = gate.status(
        refresh_snapshot(
            sequence=11,
            graph_revision=6,
            observation_step=20,
            portal_id="portal_door_1",
            portal_state="open",
            traversable=True,
            requires_interaction=False,
        ),
        now=101.0,
    )

    assert released.ready
    assert not released.timed_out
    assert released.graph_updated
    assert not released.observation_updated
    assert released.reason == "fresh_opened_portal_graph"


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


def test_post_interaction_refresh_gate_waits_for_opened_portal_graph_and_observation() -> None:
    gate = PostInteractionRefreshGate(
        PostInteractionRefreshConfig(min_candidate_updates=2, timeout_s=12.0)
    )
    baseline = refresh_snapshot(sequence=10, graph_revision=5, observation_step=20)

    started = gate.begin(baseline, portal_id="portal_door_1", now=100.0)
    assert started.active
    assert not started.ready
    assert started.reason == "awaiting_candidate_updates"

    one_candidate = gate.status(
        refresh_snapshot(sequence=11, graph_revision=6, observation_step=21), now=101.0
    )
    assert not one_candidate.ready
    assert one_candidate.candidate_updates == 1

    stale_graph = gate.status(
        refresh_snapshot(sequence=12, graph_revision=5, observation_step=22), now=102.0
    )
    assert not stale_graph.ready
    assert stale_graph.reason == "awaiting_graph_revision"

    generic_fresh_but_portal_unknown = gate.status(
        refresh_snapshot(
            sequence=12,
            graph_revision=6,
            observation_step=22,
            portal_id="portal_door_1",
            portal_state="unknown",
            traversable=False,
            requires_interaction=True,
        ),
        now=103.0,
    )
    assert not generic_fresh_but_portal_unknown.ready
    assert generic_fresh_but_portal_unknown.reason == "awaiting_portal_open_state"
    assert not generic_fresh_but_portal_unknown.portal_confirmed

    fresh = gate.status(
        refresh_snapshot(
            sequence=13,
            graph_revision=7,
            observation_step=23,
            portal_id="portal_door_1",
            portal_state="open",
            traversable=True,
            requires_interaction=False,
        ),
        now=104.0,
    )
    assert fresh.ready
    assert not fresh.timed_out
    assert fresh.reason == "fresh_opened_portal_graph_observation"
    assert fresh.portal_id == "portal_door_1"
    assert fresh.portal_confirmed


def test_post_interaction_refresh_gate_releases_on_bounded_timeout() -> None:
    gate = PostInteractionRefreshGate(
        PostInteractionRefreshConfig(min_candidate_updates=2, timeout_s=3.0)
    )
    baseline = refresh_snapshot(sequence=10, graph_revision=5, observation_step=20)
    gate.begin(baseline, portal_id="portal_door_1", now=100.0)

    waiting = gate.status(baseline, now=102.9)
    assert not waiting.ready
    assert not waiting.timed_out

    timed_out = gate.status(baseline, now=103.0)
    assert timed_out.ready
    assert timed_out.timed_out
    assert timed_out.reason == "post_open_portal_confirmation_timeout_fallback"


def test_portal_confirmation_is_scoped_to_matching_portal() -> None:
    snapshot = refresh_snapshot(
        sequence=12,
        graph_revision=7,
        observation_step=23,
        portal_id="other_portal",
        portal_state="open",
        traversable=True,
        requires_interaction=False,
    )
    confirmation = portal_open_confirmation(snapshot, "portal_door_1")

    assert not confirmation.observed
    assert not confirmation.ready


def test_portal_confirmation_accepts_matching_connectivity_edge() -> None:
    snapshot = refresh_snapshot(
        sequence=12,
        graph_revision=7,
        observation_step=23,
    )
    snapshot["graph_context"] = {
        "nodes": [],
        "edges": [
            {
                "src_id": "room_1",
                "dst_id": "portal_door_1",
                "attributes": {
                    "portal_node_id": "portal_door_1",
                    "state": "open",
                    "traversable": True,
                    "requires_interaction": False,
                },
            }
        ],
    }

    confirmation = portal_open_confirmation(snapshot, "portal_door_1")

    assert confirmation.observed
    assert confirmation.ready
