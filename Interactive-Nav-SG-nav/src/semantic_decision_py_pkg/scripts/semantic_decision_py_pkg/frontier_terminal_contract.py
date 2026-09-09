"""Classify frontier-filter evidence for the semantic terminal contract.

The explorer intentionally suppresses tiny clusters and unsafe viewpoints before
publishing navigation candidates.  A zero candidate count is therefore not, by
itself, proof that the map has no meaningful frontier left.  This module keeps
the distinction small and explicit for the candidate and completion layers.
"""

from __future__ import annotations

from typing import Any


RETRYABLE_FRONTIER_REASON = "material_frontier_without_safe_viewpoint"


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def summarize_frontier_filtering(frontier_debug: Any) -> dict[str, Any]:
    """Return public terminal evidence without turning tiny noise into work.

    Only a material cluster that survived the tiny-cluster filter but was
    rejected because there is currently no safe viewpoint is retryable.  State
    exclusions (already visited/blacklisted) intentionally do not keep an
    episode alive, and a tiny-only raw frontier remains terminal.
    """

    debug = frontier_debug if isinstance(frontier_debug, dict) else {}
    has_filter_stats = "raw_clusters" in debug and "kept_clusters" in debug
    raw_clusters = _nonnegative_int(debug.get("raw_clusters"))
    frontier_cells = _nonnegative_int(debug.get("frontier_cells"))
    min_cluster_cells = max(1, _nonnegative_int(debug.get("min_cluster_cells")))
    dropped_tiny = min(raw_clusters, _nonnegative_int(debug.get("dropped_tiny")))
    dropped_no_viewpoint = _nonnegative_int(debug.get("dropped_no_viewpoint"))
    dropped_state = _nonnegative_int(debug.get("dropped_state"))
    kept_clusters = _nonnegative_int(debug.get("kept_clusters"))
    material_clusters = max(0, raw_clusters - dropped_tiny)
    retryable = bool(
        has_filter_stats
        and kept_clusters == 0
        and material_clusters > 0
        and dropped_no_viewpoint > 0
    )
    return {
        "raw_frontier_cluster_count": raw_clusters,
        "raw_frontier_cell_count": frontier_cells,
        "raw_frontier_min_cluster_cells": min_cluster_cells,
        "raw_frontier_material_cluster_count": material_clusters,
        "filtered_tiny_frontier_cluster_count": dropped_tiny,
        "filtered_no_viewpoint_frontier_cluster_count": dropped_no_viewpoint,
        "filtered_state_frontier_cluster_count": dropped_state,
        "filtered_frontier_retryable": retryable,
        "filtered_frontier_reason": RETRYABLE_FRONTIER_REASON if retryable else "",
    }
