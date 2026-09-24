import csv
import gzip
import json

from scripts.InteractiveNav.path_following_diagnostics import path_distance, summarize


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_path_distance_projects_to_segment_instead_of_vertices():
    assert path_distance((0.5, 0.1), [(0.0, 0.0), (1.0, 0.0)]) == 0.1


def test_summary_excludes_post_eval_recording_frames(tmp_path):
    attempt = tmp_path / "attempt_001"
    debug = attempt / "debug"
    debug.mkdir(parents=True)
    write_rows(
        debug / "move_base_plans.csv",
        ["plan_type", "frame_id", "message_index", "step_id", "x", "y"],
        [
            {"plan_type": "global", "frame_id": "tf_frame_map", "message_index": 1,
             "step_id": 0, "x": 0.0, "y": 0.0},
            {"plan_type": "global", "frame_id": "tf_frame_map", "message_index": 1,
             "step_id": 0, "x": 1.0, "y": 0.0},
        ],
    )
    write_rows(
        debug / "move_base_status.csv", ["step_id", "status_name"],
        [{"step_id": 0, "status_name": "ACTIVE"}],
    )
    write_rows(
        debug / "trajectory.csv", ["step_id", "x", "y"],
        [
            {"step_id": 1, "x": 0.2, "y": 0.0},
            {"step_id": 2, "x": 0.4, "y": 0.1},
            {"step_id": 3, "x": 0.6, "y": 0.7},
        ],
    )

    assert summarize(tmp_path)["fraction_over_0_2m"] == 0.333

    evaluation = attempt / "eval"
    evaluation.mkdir()
    (evaluation / "results.json").write_text(
        json.dumps([{"status": "complete", "applied_action_step_count": 2}]),
        encoding="utf-8",
    )

    result = summarize(tmp_path)
    assert result["applied_step_limit"] == 2
    assert result["samples"] == 2
    assert result["fraction_over_0_2m"] == 0.0


def test_summary_clears_published_path_when_recorder_omits_empty_plan(tmp_path):
    debug = tmp_path / "attempt_001" / "debug"
    debug.mkdir(parents=True)
    write_rows(
        debug / "move_base_plans.csv",
        ["plan_type", "frame_id", "message_index", "step_id", "x", "y"],
        [
            {"plan_type": "global", "frame_id": "tf_frame_map", "message_index": 1,
             "step_id": 0, "x": 0.0, "y": 0.0},
            {"plan_type": "global", "frame_id": "tf_frame_map", "message_index": 1,
             "step_id": 0, "x": 1.0, "y": 0.0},
            {"plan_type": "global", "frame_id": "tf_frame_map", "message_index": 3,
             "step_id": 4, "x": 0.0, "y": 0.7},
            {"plan_type": "global", "frame_id": "tf_frame_map", "message_index": 3,
             "step_id": 4, "x": 1.0, "y": 0.7},
        ],
    )
    write_rows(debug / "move_base_status.csv", ["step_id", "status_name"],
               [{"step_id": 0, "status_name": "ACTIVE"}])
    write_rows(
        debug / "trajectory.csv", ["step_id", "x", "y"],
        [{"step_id": step, "x": step / 5, "y": height}
         for step, height in ((1, 0.0), (2, 0.7), (3, 0.7), (4, 0.7))],
    )
    raw = debug / "raw"
    raw.mkdir()
    with gzip.open(raw / "step_boundaries.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"global_plan": {"frame_id": "tf_frame_map",
                                                "message_index": 2, "step_id": 2,
                                                "poses": []}}) + "\n")

    result = summarize(tmp_path)
    assert result["samples"] == 2
    assert result["fraction_over_0_2m"] == 0.0
