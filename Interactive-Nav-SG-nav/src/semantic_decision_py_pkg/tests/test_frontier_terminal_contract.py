from semantic_decision_py_pkg.frontier_terminal_contract import (
    RETRYABLE_FRONTIER_REASON,
    summarize_frontier_filtering,
)


def test_material_cluster_without_safe_viewpoint_is_retryable() -> None:
    evidence = summarize_frontier_filtering(
        {
            "frontier_cells": 17,
            "raw_clusters": 2,
            "min_cluster_cells": 3,
            "dropped_tiny": 1,
            "dropped_no_viewpoint": 1,
            "dropped_state": 0,
            "kept_clusters": 0,
        }
    )

    assert evidence["raw_frontier_material_cluster_count"] == 1
    assert evidence["filtered_frontier_retryable"] is True
    assert evidence["filtered_frontier_reason"] == RETRYABLE_FRONTIER_REASON


def test_tiny_or_state_filtered_frontiers_do_not_keep_episode_alive() -> None:
    tiny_only = summarize_frontier_filtering(
        {
            "frontier_cells": 2,
            "raw_clusters": 1,
            "min_cluster_cells": 3,
            "dropped_tiny": 1,
            "dropped_no_viewpoint": 0,
            "dropped_state": 0,
            "kept_clusters": 0,
        }
    )
    state_only = summarize_frontier_filtering(
        {
            "frontier_cells": 12,
            "raw_clusters": 1,
            "min_cluster_cells": 3,
            "dropped_tiny": 0,
            "dropped_no_viewpoint": 0,
            "dropped_state": 1,
            "kept_clusters": 0,
        }
    )

    assert tiny_only["filtered_frontier_retryable"] is False
    assert state_only["filtered_frontier_retryable"] is False
