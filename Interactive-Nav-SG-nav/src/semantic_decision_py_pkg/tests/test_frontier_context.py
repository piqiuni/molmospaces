from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.frontier_context import (
    eligible_room_frontier_counts,
    observed_room_frontier_summary,
)


def graph():
    return {"frame_id": "map", "nodes": [
        {"id": f"room_{i}", "room_id": i, "type": "room", "aabb_center": [x, 0, 0], "aabb_size": [2, 2, 1]}
        for i, x in ((1, 0), (2, 3), (3, 6))
    ]}


def test_status_frontier_cells_split_across_rooms_and_deduplicate():
    cluster = {"cluster_id": "a", "frontier_cells_world": [[0, 0], [0.5, 0], [3, 0]]}
    source = {
        "ready": True, "timestamp": 100, "frame_id": "map", "map_resolution": 0.5,
        "frontier_count": 2, "frontier_clusters": [cluster, cluster],
    }
    result = observed_room_frontier_summary(source, graph())
    assert result["observed_room_frontier_lengths"] == {"room_1": 1, "room_2": 0.5, "room_3": 0}
    assert result["observed_room_frontier_counts"] == {"room_1": 1, "room_2": 1, "room_3": 0}
    stats = result["room_frontier_statistics"]
    assert stats["complete_published_pool"] is True
    assert stats["raw_map_coverage"] == "partial"
    assert stats["source_timestamp"] == 100


def test_proposal_only_statistics_are_partial_not_zero_for_absent_rooms():
    source = {
        "ready": True, "frame_id": "map", "proposal_count": 1,
        "proposals": [{"cluster_id": "a", "frontier_point": [0, 0], "raw_features": {"frontier_length_m": 1.25}}],
    }
    result = observed_room_frontier_summary(source, graph())
    assert result["observed_room_frontier_lengths"] == {"room_1": 1.25}
    assert result["observed_room_frontier_counts"] == {"room_1": 1}
    assert result["room_frontier_statistics"]["complete_published_pool"] is False
    assert result["room_frontier_statistics"]["assignment_method"] == "cluster_centroid_aabb_approximation"


def test_stale_missing_and_mismatched_frames_never_mean_zero_frontiers():
    source = {"ready": True, "frame_id": "map", "frontier_count": 0, "frontier_clusters": []}
    assert observed_room_frontier_summary(source, graph(), fresh=False)["observed_room_frontier_lengths"] == {}
    for frame in (None, "odom"):
        source["frame_id"] = frame
        result = observed_room_frontier_summary(source, graph())
        assert result["observed_room_frontier_lengths"] == {}
        assert result["room_frontier_statistics"]["complete_published_pool"] is False


def test_unassigned_cells_make_room_totals_explicitly_partial():
    source = {
        "ready": True, "frame_id": "map", "map_resolution": 0.5, "frontier_count": 1,
        "frontier_clusters": [{"cluster_id": "a", "frontier_cells_world": [[0, 0], [100, 100]]}],
    }
    result = observed_room_frontier_summary(source, graph())
    assert result["observed_room_frontier_lengths"] == {"room_1": 0.5}
    assert result["room_frontier_statistics"]["unassigned_cell_count"] == 1
    assert result["room_frontier_statistics"]["complete_published_pool"] is False


def test_eligible_frontier_counts_use_actual_pool_and_ignore_duplicate_ids():
    candidate = BehaviorCandidate(candidate_id="a", behavior_type="EXPLORE", source="test", target_id="a", target_name="a", metadata={"target_room_id": 1})
    interaction = BehaviorCandidate(candidate_id="b", behavior_type="INTERACT", source="test", target_id="b", target_name="b")
    assert eligible_room_frontier_counts([candidate, candidate, interaction], graph()) == {"room_1": 1, "room_2": 0, "room_3": 0}
