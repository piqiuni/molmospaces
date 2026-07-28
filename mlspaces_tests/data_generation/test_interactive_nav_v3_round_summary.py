from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.InteractiveNav.evaluation import v3_round_summary


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_round_summary_combines_evaluator_recorder_and_mllm_metrics(tmp_path: Path) -> None:
    worker = tmp_path / "worker_0_ep_42"
    episode_dir = worker / "eval" / "episodes" / "case"
    _write_json(
        episode_dir / "episode_result.json",
        {
            "status": "complete",
            "result": {
                "episode_index": 42,
                "house_index": 7,
                "status": "complete",
                "success": True,
                "terminal_reason": "task_success",
                "step_count": 250,
                "episode_step_budget": 300,
                "navigation_path_length_m": 2.5,
                "reference_path_length_m": 2.0,
                "target_distance_m": 0.7,
                "target_visibility_fraction": 0.25,
                "correct_interaction_action_count": 1,
                "extra_interaction_action_count": 1,
                "invalid_interaction_action_count": 1,
                "elapsed_seconds": 12.5,
                "interaction_attempts": [
                    {"instance_id": "door", "operation": "open"},
                    {"instance_id": "door", "operation": "open"},
                    {"instance_id": "drawer", "operation": "open"},
                ],
            },
        },
    )
    _write_json(
        episode_dir / "episode_topdown.json",
        {
            "coverage": {
                "source": "static_scene_map_recomputed",
                "exploration_coverage_ratio": 0.8,
            }
        },
    )
    _write_json(
        worker / "debug" / "summary.json",
        {
            "duration_sec": 14.0,
            "step_sync_count": 250,
            "first_person_video_frame_count": 250,
            "step_sync_image_match_count": 249,
            "step_sync_image_reuse_count": 0,
            "step_sync_placeholder_count": 1,
        },
    )
    metrics = [
        {
            "role": "subgoal_selection",
            "candidate_ids": ["frontier:a"],
            "latency_s": 0.2,
            "raw_text": "The prompt documents pre_score_guard but did not apply it.",
        },
        {
            "role": "subgoal_selection",
            "candidate_ids": ["frontier:a"],
            "latency_s": 0.3,
            "pre_score_guard_applied": True,
        },
        {"role": "subgoal_selection", "candidate_ids": ["portal:b"], "latency_s": 0.4},
        {"role": "skill_planning", "candidate_id": "portal:b", "latency_s": 0.1},
    ]
    (worker / "mllm_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metrics) + "\n", encoding="utf-8"
    )

    summary = v3_round_summary.summarise_round(tmp_path)

    assert summary["episode_count"] == 1
    assert summary["success_count"] == 1
    assert summary["parallel_wall_time_estimate_s"] == 12.5
    episode = summary["episodes"][0]
    assert episode["interactions"] == {
        "attempt_count": 3,
        "correct_count": 1,
        "repeat_count": 1,
        "error_count": 2,
        "extra_count": 1,
        "invalid_count": 1,
    }
    assert episode["coverage"]["exploration_ratio"] == 0.8
    assert episode["rgb_step_sync"]["image_match_count"] == 249
    assert episode["rgb_step_sync"]["placeholder_count"] == 1
    assert episode["rgb_step_sync"]["raw_step_sync_complete"] is True
    assert episode["rgb_step_sync"]["step_sync_capture_every"] == 1
    assert episode["rgb_step_sync"]["expected_video_frame_count"] == 250
    assert episode["rgb_step_sync"]["step_sync_capture_count"] == 250
    assert episode["rgb_step_sync"]["step_sync_capture_count_source"] == (
        "video_frame_count_legacy_fallback"
    )
    assert episode["rgb_step_sync"]["step_sync_capture_complete"] is True
    assert episode["rgb_step_sync"]["video_frame_count_complete"] is True
    assert episode["mllm"]["call_count"] == 4
    assert episode["mllm"]["pre_score_guard_count"] == 1
    assert episode["mllm"]["repeated_selected_candidate_count"] == 1
    assert episode["mllm"]["consecutive_repeat_count"] == 1


def test_round_summary_tolerates_missing_optional_artifacts(tmp_path: Path) -> None:
    result_path = tmp_path / "worker_2" / "episode_result.json"
    _write_json(
        result_path,
        {
            "episode_index": 9,
            "success": False,
            "terminal_reason": "max_steps",
            "step_count": 300,
            "episode_step_budget": 300,
        },
    )

    summary = v3_round_summary.summarise_round(tmp_path)
    rendered = v3_round_summary.render_terminal_summary(summary)

    assert summary["worker_count"] == 1
    assert summary["episodes"][0]["rgb_step_sync"]["placeholder_count"] is None
    assert summary["episodes"][0]["coverage"]["exploration_ratio"] is None
    assert "worker_2" in rendered
    assert "300/300" in rendered
    assert "—" in rendered


def test_round_summary_uses_sampled_video_frame_expectation(tmp_path: Path) -> None:
    worker = tmp_path / "worker_1_ep_73"
    _write_json(
        worker / "episode_result.json",
        {
            "episode_index": 73,
            "success": False,
            "terminal_reason": "max_steps",
            "step_count": 701,
            "episode_step_budget": 750,
        },
    )
    _write_json(
        worker / "debug" / "summary.json",
        {
            "step_sync_count": 701,
            "step_sync_capture_every": 2,
            "step_sync_capture_count": 351,
            "first_person_video_frame_count": 351,
            "step_sync_placeholder_count": 0,
        },
    )

    episode = v3_round_summary.summarise_round(tmp_path)["episodes"][0]
    rgb = episode["rgb_step_sync"]

    assert rgb["raw_step_sync_complete"] is True
    assert rgb["expected_video_frame_count"] == 351
    assert rgb["step_sync_capture_count"] == 351
    assert rgb["step_sync_capture_count_source"] == "summary"
    assert rgb["step_sync_capture_complete"] is True
    assert rgb["video_frame_count_complete"] is True
    # This literal legacy field remains false because it compares raw markers
    # with sampled frames, not recorder completeness.
    assert rgb["video_frame_count_matches_step_sync"] is False


def test_round_summary_supports_direct_script_execution(tmp_path: Path) -> None:
    result_path = tmp_path / "worker_0" / "episode_result.json"
    _write_json(
        result_path,
        {
            "episode_index": 3,
            "success": True,
            "terminal_reason": "task_success",
            "step_count": 12,
            "episode_step_budget": 300,
        },
    )
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "InteractiveNav"
        / "evaluation"
        / "v3_round_summary.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--format", "json"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["episode_count"] == 1
    assert payload["success_count"] == 1
