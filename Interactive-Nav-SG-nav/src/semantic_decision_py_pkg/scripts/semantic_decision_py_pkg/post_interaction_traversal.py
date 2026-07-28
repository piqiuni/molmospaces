from __future__ import annotations

import copy
import math
from typing import Any, Iterable

from .behavior_candidates import BehaviorCandidate


def portal_center_xy(graph: dict[str, Any], node_id: str) -> list[float] | None:
    """Return the public portal center available in the decision snapshot."""

    for node in graph.get("nodes") or []:
        if str(node.get("id") or "") != str(node_id or ""):
            continue
        attributes = node.get("attributes") or {}
        center = list(
            attributes.get("interaction_reference_aabb_center")
            or node.get("aabb_center")
            or node.get("centroid")
            or []
        )
        if len(center) < 2:
            return None
        return [float(center[0]), float(center[1])]
    return None


def build_post_interaction_traversal_candidate(
    active_candidate: dict[str, Any],
    feedback: dict[str, Any],
    *,
    robot_xy: Iterable[float] | None = None,
    traversal_distance_m: float = 0.9,
) -> BehaviorCandidate | None:
    """Build the immediate one-shot continuation of a successful portal open."""

    metadata = active_candidate.get("metadata") or {}
    command = active_candidate.get("interaction_command") or {}
    detail = feedback.get("detail") or {}
    if str(active_candidate.get("behavior_type") or "").upper() != "INTERACT":
        return None
    if str(metadata.get("node_type") or "").casefold() != "portal":
        return None
    status = str(feedback.get("status") or "").upper()
    success = feedback.get("success")
    if not (status == "SUCCEEDED" or (not status and success is True)):
        return None
    action = str(
        detail.get("action")
        or feedback.get("action")
        or command.get("action")
        or ""
    ).casefold()
    if action != "open":
        return None

    portal_id = str(
        detail.get("node_id")
        or feedback.get("node_id")
        or active_candidate.get("target_id")
        or command.get("node_id")
        or ""
    )
    if not portal_id:
        return None
    approach = list(
        detail.get("approach_goal_xyyaw")
        or command.get("interaction_approach_pose_xyyaw")
        or active_candidate.get("goal_xyyaw")
        or []
    )
    center = list(active_candidate.get("portal_center_xy") or [])
    if len(approach) < 2 or len(center) < 2:
        return None

    through_x = float(center[0]) - float(approach[0])
    through_y = float(center[1]) - float(approach[1])
    through_norm = math.hypot(through_x, through_y)
    if through_norm <= 1e-6:
        return None
    unit_x = through_x / through_norm
    unit_y = through_y / through_norm
    traversal_distance = max(0.0, float(traversal_distance_m))
    goal = [
        float(center[0]) + unit_x * traversal_distance,
        float(center[1]) + unit_y * traversal_distance,
        math.atan2(unit_y, unit_x),
    ]
    robot = list(robot_xy or [])
    distance_m = (
        math.hypot(goal[0] - float(robot[0]), goal[1] - float(robot[1]))
        if len(robot) >= 2
        else math.hypot(goal[0] - float(approach[0]), goal[1] - float(approach[1]))
    )
    event_id = str(
        detail.get("event_id")
        or feedback.get("event_id")
        or command.get("event_id")
        or active_candidate.get("decision_id")
        or "latest_open"
    )
    target_name = str(
        active_candidate.get("target_name")
        or detail.get("object_id")
        or command.get("object_id")
        or portal_id
    )
    confidence = float(
        (active_candidate.get("features") or {}).get("confidence", 1.0) or 1.0
    )
    return BehaviorCandidate(
        candidate_id=f"traverse:{portal_id}:{event_id}",
        behavior_type="NAVIGATE",
        source="post_interaction_feedback",
        target_id=portal_id,
        target_name=target_name,
        goal_xyyaw=goal,
        features={
            "exploration_gain": 1.25,
            "visibility_gain": 1.15,
            "semantic_gain": 1.0,
            "target_relevance": 0.0,
            "distance_m": distance_m,
            "interaction_cost": 0.0,
            "state_age_ratio": 0.0,
            "confidence": confidence,
            "priority": 1.0,
        },
        metadata={
            "node_type": "portal",
            "semantic_name": str(metadata.get("semantic_name") or "door"),
            "state": "open",
            "post_interaction_traversal": True,
            "opened_portal_id": portal_id,
            "source_interaction_event_id": event_id,
            "connected_room_ids": list(metadata.get("connected_room_ids") or []),
            "room_transition_required": True,
            "requires_approach": False,
            "verify_target_visibility": False,
            "goal_xyyaw_candidates": [goal],
            "decision_local_transition": True,
        },
    )


def inject_pending_traversal(
    candidate_snapshot: dict[str, Any],
    pending_candidate: dict[str, Any] | None,
) -> bool:
    """Inject a cached traversal unless the graph already published the same ID."""

    if not pending_candidate:
        return False
    metadata = pending_candidate.get("metadata") or {}
    pending_episode_id = str(metadata.get("source_episode_id") or "")
    snapshot_episode_id = str(candidate_snapshot.get("episode_id") or "")
    if (
        pending_episode_id
        and snapshot_episode_id
        and pending_episode_id != snapshot_episode_id
    ):
        return False
    candidate_id = str(pending_candidate.get("candidate_id") or "")
    if not candidate_id:
        return False
    candidates = list(candidate_snapshot.get("candidates") or [])
    if any(str(candidate.get("candidate_id") or "") == candidate_id for candidate in candidates):
        return False
    candidates.append(copy.deepcopy(pending_candidate))
    candidate_snapshot["candidates"] = candidates
    candidate_snapshot["candidate_count"] = len(candidates)
    return True


def pending_priority_candidate(
    candidates: Iterable[BehaviorCandidate], pending_candidate_id: str
) -> BehaviorCandidate | None:
    if not pending_candidate_id:
        return None
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.candidate_id == pending_candidate_id
            and bool((candidate.metadata or {}).get("post_interaction_traversal"))
        ),
        None,
    )


def is_terminal_post_interaction_traversal_failure(
    candidate_id: str,
    behavior_type: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> bool:
    """Identify a one-shot portal traversal that must not be retried unchanged."""

    normalized_status = str(status or "").upper()
    if (
        str(behavior_type or "").upper() != "NAVIGATE"
        or not str(candidate_id or "").startswith("traverse:")
        or normalized_status not in {"FAILED", "REJECTED", "CANCELED"}
    ):
        return False
    return not (
        normalized_status == "CANCELED"
        and str((detail or {}).get("reason") or "") == "preempted_by_target"
    )
