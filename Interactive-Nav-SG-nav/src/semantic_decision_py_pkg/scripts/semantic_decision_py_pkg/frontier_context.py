"""Public, observation-only frontier statistics for the M2 context."""
from __future__ import annotations

import math
from typing import Any, Iterable

from .room_context import room_context_for_xy


def _room_id(value: Any) -> str:
    if value in (None, ""):
        return ""
    value = str(value)
    return value if value.startswith("room_") else f"room_{value}"


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(x) for x in point) else None


def _finite_nonnegative(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def observed_room_frontier_summary(
    source: dict[str, Any], graph: dict[str, Any], *, fresh: bool = True,
) -> dict[str, Any]:
    """Measure the published pool, not unfiltered map frontiers or hidden area.

    Status cells allow per-room boundary splitting. Proposal-only data is a
    centroid approximation and explicitly a partial pool. Coordinates are never
    compared across different/unknown frames and no TF is inferred here.
    """
    source_frame = str(source.get("frame_id") or "").lstrip("/")
    graph_frame = str(graph.get("frame_id") or "").lstrip("/")
    frames_match = bool(source_frame and graph_frame and source_frame == graph_frame)
    is_status = isinstance(source.get("frontier_clusters"), list)
    items = source.get("frontier_clusters") if is_status else source.get("proposals")
    if not isinstance(items, list):
        items = source.get("exploration_proposals") or []
        if isinstance(items, dict):
            items = items.get("proposals") or []
    items = [item for item in items if isinstance(item, dict)]
    complete_pool = bool(
        is_status and source.get("ready")
        and source.get("frontier_count") == len(items)
    )
    stats = {
        "source": "explore_py_status_clusters" if is_status else "explore_py_proposals",
        "scope": "published_frontiers_after_explore_py_filters",
        "raw_map_coverage": "partial",
        "complete_published_pool": complete_pool,
        "source_timestamp": source.get("timestamp"),
        "frontier_computed_ts": source.get("frontier_computed_ts"),
        "source_timestamp_kind": "status_publish_time" if is_status else "proposal_snapshot_time",
        "source_frame_id": source.get("frame_id") or None,
        "graph_frame_id": graph.get("frame_id") or None,
        "fresh": bool(fresh),
        "coordinate_frame_status": "matching" if frames_match else "mismatch" if source_frame and graph_frame else "unknown",
        "assignment_method": "frontier_cells_aabb" if is_status else "cluster_centroid_aabb_approximation",
        "length_method": "unique_frontier_cell_count_times_resolution" if is_status else "published_cluster_length",
        "unassigned_cell_count": 0,
        "unassigned_cluster_count": 0,
        "incomplete_cluster_count": 0,
    }
    result = {
        "observed_room_frontier_lengths": {},
        "observed_room_frontier_counts": {},
        "room_frontier_statistics": stats,
    }
    if not fresh or not frames_match or not source.get("ready"):
        stats["complete_published_pool"] = False
        return result
    lengths: dict[str, float] = {}
    clusters_by_room: dict[str, set[str]] = {}
    seen_clusters: set[str] = set()
    seen_cells: set[tuple[float, float]] = set()
    resolution = _finite_nonnegative(source.get("map_resolution"))
    for ordinal, item in enumerate(items):
        center = _xy(item.get("centroid_world") or item.get("frontier_point"))
        key = str(item.get("cluster_id") or item.get("proposal_id") or (f"centroid:{center}" if center else f"unknown_{ordinal}"))
        if key in seen_clusters:
            continue
        seen_clusters.add(key)
        cells = item.get("frontier_cells_world")
        if isinstance(cells, list) and resolution is not None and resolution > 0:
            if item.get("cell_count") is not None and item["cell_count"] != len(cells):
                stats["incomplete_cluster_count"] += 1
            for cell in cells:
                point = _xy(cell)
                if point is None:
                    stats["incomplete_cluster_count"] += 1
                    continue
                point_key = (round(point[0], 6), round(point[1], 6))
                if point_key in seen_cells:
                    continue
                seen_cells.add(point_key)
                room = _room_id(room_context_for_xy(graph, point).get("room_id"))
                if not room:
                    stats["unassigned_cell_count"] += 1
                    continue
                lengths[room] = lengths.get(room, 0.0) + resolution
                clusters_by_room.setdefault(room, set()).add(key)
            continue
        stats["assignment_method"] = "cluster_centroid_aabb_approximation"
        stats["length_method"] = "published_cluster_length_or_cells_times_resolution"
        features = item.get("raw_features") or item
        length = _finite_nonnegative(features.get("frontier_length_m"))
        if length is None or length == 0:
            cell_count = _finite_nonnegative(features.get("frontier_cell_count", features.get("cell_count")))
            if cell_count is not None and resolution is not None and resolution > 0:
                length = cell_count * resolution
        room = _room_id(room_context_for_xy(graph, center).get("room_id")) if center else ""
        if not room:
            stats["unassigned_cluster_count"] += 1
            continue
        clusters_by_room.setdefault(room, set()).add(key)
        if length is None:
            stats["incomplete_cluster_count"] += 1
        else:
            lengths[room] = lengths.get(room, 0.0) + length
    if stats["incomplete_cluster_count"] or stats["unassigned_cell_count"] or stats["unassigned_cluster_count"]:
        stats["complete_published_pool"] = False
    if stats["complete_published_pool"]:
        for node in graph.get("nodes") or []:
            if node.get("type") == "room" and (node.get("attributes") or {}).get("active", True):
                room = _room_id(node.get("room_id") if node.get("room_id") is not None else node.get("id"))
                if room:
                    lengths.setdefault(room, 0.0)
                    clusters_by_room.setdefault(room, set())
    result["observed_room_frontier_lengths"] = {room: round(length, 4) for room, length in lengths.items()}
    result["observed_room_frontier_counts"] = {room: len(keys) for room, keys in clusters_by_room.items()}
    stats["unique_cluster_count"] = len(seen_clusters)
    return result


def eligible_room_frontier_counts(candidates: Iterable[Any], graph: dict[str, Any]) -> dict[str, int]:
    """Count actual execution-eligible candidates, without centroid guesses."""
    counts: dict[str, int] = {
        _room_id(node.get("room_id") if node.get("room_id") is not None else node.get("id")): 0
        for node in graph.get("nodes") or []
        if node.get("type") == "room" and (node.get("attributes") or {}).get("active", True)
    }
    counts.pop("", None)
    seen: set[str] = set()
    for candidate in candidates:
        if str(candidate.behavior_type).upper() != "EXPLORE" or candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        metadata = candidate.metadata or {}
        room = _room_id(metadata.get("target_room_id") if metadata.get("target_room_id") is not None else metadata.get("room_id"))
        if not room:
            room = _room_id(room_context_for_xy(graph, list(candidate.goal_xyyaw or [])[:2]).get("room_id"))
        if room:
            counts[room] = counts.get(room, 0) + 1
    return counts
