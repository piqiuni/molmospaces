import gzip
import json

import pytest

from scripts.InteractiveNav.evaluation.m2_context_dataset import (
    candidate_history_from_decisions,
    candidate_map_pose,
    decision_history_at,
    enrich_attempt,
    estimate_request_start,
    map_pose,
    room_history_at,
    recorded_frontier_lengths,
)
from scripts.InteractiveNav.evaluation.m2_replay_eval import DatasetError


def test_request_start_excludes_model_latency():
    assert estimate_request_start({"timestamp": 100, "latency_s": 4}) == 96
    assert estimate_request_start({"timestamp": 100, "latency_s": 4, "request_started_ts": 95}) == 95
    with pytest.raises(DatasetError):
        estimate_request_start({"timestamp": 100})


def test_pose_uses_map_transform_and_independent_availability():
    pose = map_pose({
        "pose": [1, 0, 0], "stamp_sec": 10, "step_index": 3,
        "pose_frame_id": "odom", "graph_frame_id": "map",
        "tf_map_from_odom": {"source_frame": "odom", "target_frame": "map", "stamp_sec": 12,
                             "x": 4, "y": 5, "yaw": 1.5707963267948966},
    })
    assert pose["xy"] == pytest.approx([4, 6])
    assert pose["available_timestamp"] == 12
    assert pose["pose_timestamp"] == 10
    assert map_pose({"pose": [1, 2], "stamp_sec": 10, "pose_frame_id": "odom", "graph_frame_id": "map"}) is None


def test_history_deduplicates_and_never_backfills_future_feedback():
    selections = [
        {"decision_id": "d1", "selected_at": 1, "observation_step": 1},
        {"decision_id": "d1", "selected_at": 1, "observation_step": 5},
        {"decision_id": "d2", "selected_at": 20, "observation_step": 20},
    ]
    feedback = [
        {"decision_id": "d1", "timestamp": 2, "status": "STARTED"},
        {"decision_id": "d1", "timestamp": 15, "status": "SUCCEEDED"},
    ]
    result = decision_history_at(selections, feedback, 10, 10)
    assert len(result) == 1
    assert result[0]["result"] == "STARTED"
    assert result[0]["observation_step"] == 1
    assert result[0]["feedback_timestamp"] == 2


def test_history_retains_only_last_thirty_real_decisions():
    selections = [{"decision_id": f"d{i}", "selected_at": i, "observation_step": i} for i in range(40)]
    result = decision_history_at(selections, [], 100, 100)
    assert len(result) == 30
    assert result[0]["decision_id"] == "d10"
    assert all(row["result"] == "PENDING" for row in result)


def test_room_visits_preserve_returns_and_exclude_future():
    events = [
        {"room_id": room, "available_timestamp": i, "entry_step": i}
        for i, room in enumerate(("room_a", "room_a", "room_b", "room_a", "room_c"))
    ]
    assert [row["room_id"] for row in room_history_at(events, 3)] == ["room_a", "room_b", "room_a"]


def test_candidate_history_recovers_counts_without_inventing_gain():
    result = candidate_history_from_decisions([
        {"candidate_id": "c1", "history_key": "region_a", "observation_step": 1, "result": "FAILED"},
        {"candidate_id": "c2", "history_key": "region_a", "observation_step": 10, "result": "PENDING"},
    ])
    assert result["region_a"]["selection_count"] == 2
    assert result["region_a"]["last_result"] == "PENDING"
    assert result["region_a"]["last_terminal_result"] == "FAILED"
    assert "low_gain_repeat_count" not in result["region_a"]


def test_raw_frontier_lengths_deduplicate_and_do_not_call_unreachable_eligible():
    a = {"candidate_id": "a", "behavior_type": "EXPLORE", "metadata": {"cluster_id": "x", "room_id": 1, "frontier_length_m": 3}}
    b = {"candidate_id": "b", "behavior_type": "EXPLORE", "metadata": {"cluster_id": "y", "room_id": 1, "cell_count": 20, "map_resolution": .1, "room_reachable": False}}
    observed, eligible = recorded_frontier_lengths([a, a, b])
    assert observed == {"room_1": 5}
    assert eligible == {"room_1": 3}


def test_candidate_position_uses_raw_xy_and_only_prior_tf():
    state = {"semantic": {"robot_xy": [3, 4], "timestamp": 10}, "cutoff": 11, "semantic_step": 7,
             "semantic_pose_frame_id": "odom", "semantic_graph_frame_id": "map"}
    poses = [map_pose({"pose": [99, 99], "stamp_sec": ts, "pose_frame_id": "odom", "graph_frame_id": "map",
                       "tf_map_from_odom": {"source_frame": "odom", "target_frame": "map", "stamp_sec": ts, "x": offset, "y": 0, "yaw": 0}})
             for ts, offset in ((9, 2), (10.5, 100))]
    result = candidate_map_pose(state, poses)
    assert result["xy"] == [5, 4]
    assert result["transform_timestamp"] == 9


def _graph(revision, timestamp):
    return {
        "episode_id": "episode_inner", "graph_revision": revision, "timestamp": timestamp,
        "nodes": [{"id": "room_1", "room_id": 1, "type": "room", "aabb_center": [0, 0, 0], "aabb_size": [20, 20, 2]}],
    }


def _boundary(step, timestamp, graph_revision, graph_timestamp, sequence, semantic_timestamp):
    return {
        "step_index": step, "stamp_sec": timestamp, "pose": [step, 0, 0],
        "pose_frame_id": "map", "graph_frame_id": "map",
        "unified_graph": _graph(graph_revision, graph_timestamp),
        "semantic_candidates": {
            "episode_id": "episode_inner", "graph_revision": 10, "sequence": sequence,
            "timestamp": semantic_timestamp, "robot_xy": [step, 0],
            "target_context": {"target_name": "toilet"},
            "candidates": [{"candidate_id": "x", "behavior_type": "EXPLORE"}],
        },
        "semantic_selection": {"decision_id": "d1", "candidate_id": "x", "selected_at": 2},
        "semantic_behavior_feedback": {"decision_id": "d1", "status": "SUCCEEDED", "timestamp": 12},
        "gt_observations": {"never_copy": "private"},
    }


def test_stream_join_prefers_past_graph_and_checks_each_source_timestamp(tmp_path):
    raw = tmp_path / "debug" / "raw"
    raw.mkdir(parents=True)
    # Request cutoff = 10.  Same-boundary graph at 10.5 is in the future;
    # matching candidate snapshot at 9.5 is valid, feedback at 12 is not.
    boundaries = [_boundary(0, 1, 9, 1.2, 3, 1.4), _boundary(1, 9, 11, 10.5, 4, 9.5)]
    with gzip.open(raw / "step_boundaries.jsonl.gz", "wt") as stream:
        for boundary in boundaries:
            stream.write(json.dumps(boundary) + "\n")
    metric = {"role": "subgoal_selection", "episode_id": "episode_inner", "graph_revision": 10,
              "candidate_sequence": 4, "timestamp": 14, "latency_s": 4}
    (tmp_path / "mllm_metrics.jsonl").write_text(json.dumps(metric) + "\n")
    case = {"case_id": "case", "source": {"attempt": "episode_outer/attempt_001", "episode_id": "episode_inner",
            "graph_revision": 10, "candidate_sequence": 4, "recorded_timestamp": 14}, "request": {"candidates": []}}
    record, = enrich_attempt([case], tmp_path)
    assert record["graph"]["graph_revision"] == 9
    assert record["reconstruction"]["graph_revision_lag"] == 1
    assert record["reconstruction"]["candidate_exact_sequence_and_revision"] is True
    assert record["reconstruction"]["strict_version_aligned"] is False
    assert record["robot_context"]["decision_history"][0]["result"] == "PENDING"
    assert "gt_observations" not in json.dumps(record)
    assert record["robot_context"]["current_xy"] == [1.0, 0.0]


def test_exact_graph_at_or_before_cutoff_is_separate_subset(tmp_path):
    raw = tmp_path / "debug" / "raw"
    raw.mkdir(parents=True)
    boundary = _boundary(0, 1, 10, 9, 4, 9.5)
    with gzip.open(raw / "step_boundaries.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(boundary) + "\n")
    metric = {"role": "subgoal_selection", "episode_id": "episode_inner", "graph_revision": 10,
              "candidate_sequence": 4, "timestamp": 14, "latency_s": 4}
    (tmp_path / "mllm_metrics.jsonl").write_text(json.dumps(metric) + "\n")
    case = {"case_id": "case", "source": {"attempt": "episode_outer/attempt_001", "episode_id": "episode_inner",
            "graph_revision": 10, "candidate_sequence": 4, "recorded_timestamp": 14}, "request": {"candidates": []}}
    record, = enrich_attempt([case], tmp_path)
    assert record["reconstruction"]["strict_version_aligned"] is True
    assert record["reconstruction"]["original_http_request_exact"] is False
