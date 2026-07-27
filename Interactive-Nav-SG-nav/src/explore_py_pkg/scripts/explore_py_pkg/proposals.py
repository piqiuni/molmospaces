from __future__ import annotations

import time
from typing import Any, Iterable


GEOMETRIC_SCORE_TERMS = {
    "information",
    "distance",
    "previous_subgoal",
    "continuity_cost",
    "far_cluster_penalty",
    "revisit_penalty",
    "failure_penalty",
}


def cluster_to_proposal(cluster: Any, frame_id: str) -> dict[str, Any]:
    score_terms = dict(getattr(cluster, "score_terms", {}) or {})
    return {
        "proposal_id": str(cluster.cluster_id),
        "proposal_type": "frontier_viewpoint",
        "source": "explore_py",
        "frame_id": str(frame_id),
        "cluster_id": str(cluster.cluster_id),
        "goal_xyyaw": [
            float(cluster.subgoal_world[0]),
            float(cluster.subgoal_world[1]),
            float(cluster.subgoal_yaw),
        ],
        "frontier_point": [
            float(cluster.centroid_world[0]),
            float(cluster.centroid_world[1]),
        ],
        "raw_features": {
            "frontier_cell_count": int(len(cluster.cells)),
            "information_gain": float(cluster.information_gain),
            "distance_m": float(cluster.distance_to_robot),
            "unknown_component_area_m2": float(
                getattr(cluster, "unknown_component_area_m2", 0.0) or 0.0
            ),
            "frontier_length_m": float(
                getattr(cluster, "frontier_length_m", 0.0) or 0.0
            ),
            "expected_visible_unknown_area_m2": float(
                getattr(cluster, "expected_visible_unknown_area_m2", 0.0) or 0.0
            ),
        },
        "geometry": {
            "proposal_score": float(cluster.score),
            "proposal_score_terms": {
                key: float(value)
                for key, value in score_terms.items()
                if key in GEOMETRIC_SCORE_TERMS
            },
            "hard_constraints_passed": True,
        },
    }


def build_proposal_snapshot(
    clusters: Iterable[Any],
    *,
    ready: bool,
    frame_id: str,
    robot_xy: tuple[float, float] | None,
    robot_yaw: float | None = None,
    active_cluster_id: str = "",
    initial_scan_complete: bool = True,
    timestamp: float | None = None,
) -> dict[str, Any]:
    proposals = [cluster_to_proposal(cluster, frame_id) for cluster in clusters]
    published_proposals = proposals if initial_scan_complete else []
    return {
        "schema_version": 1,
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "ready": bool(ready),
        "frame_id": str(frame_id),
        "robot_xy": None if robot_xy is None else [float(robot_xy[0]), float(robot_xy[1])],
        "robot_yaw": None if robot_yaw is None else float(robot_yaw),
        "initial_scan_complete": bool(initial_scan_complete),
        "proposal_count": len(published_proposals),
        "frontier_exhausted": bool(
            ready
            and initial_scan_complete
            and not active_cluster_id
            and not published_proposals
        ),
        "active_proposal_id": str(active_cluster_id),
        "proposals": published_proposals,
    }
