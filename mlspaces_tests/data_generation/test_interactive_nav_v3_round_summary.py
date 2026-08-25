from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

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


def test_round_summary_surfaces_scored_early_stop_diagnostics(tmp_path: Path) -> None:
    result_path = tmp_path / "worker_3" / "episode_result.json"
    _write_json(
        result_path,
        {
            "episode_index": 88,
            "success": False,
            "scoring_eligible": True,
            "terminal_reason": "cross_subgoal_navigation_stall",
            "step_count": 157,
            "episode_step_budget": 500,
            "early_stop": {
                "triggered": True,
                "reason": "cross_subgoal_navigation_stall",
                "trigger_step": 157,
                "failed_subgoal_count": 8,
                "observed_navigation_failure_count": 9,
                "displacement_m": 0.04,
            },
        },
    )

    summary = v3_round_summary.summarise_round(tmp_path)
    rendered = v3_round_summary.render_terminal_summary(summary)

    assert summary["success_count"] == 0
    assert summary["early_stop_count"] == 1
    assert summary["early_stop_reason_counts"] == {
        "cross_subgoal_navigation_stall": 1
    }
    assert summary["episodes"][0]["early_stop"] == {
        "triggered": True,
        "reason": "cross_subgoal_navigation_stall",
        "trigger_step": 157,
        "failed_subgoal_count": 8,
        "observed_navigation_failure_count": 9,
        "displacement_m": 0.04,
    }
    assert "early-stop" in rendered
    assert "cross_subgoal_navigation_stall; n=8@157" in rendered


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


def test_round_summary_aggregates_persisted_paper_metrics(tmp_path: Path) -> None:
    metric_config = {
        "interaction_attempt_cost": 0.3,
        "error_interaction_surcharge": 1.0,
        "failure_penalty": 5.0,
    }
    saved_rows = [
        {
            "episode_index": 1,
            "domains": ["channel"],
            "interaction_requirement": "required",
            # Deliberately disagree with nav_success: paper SR must not read
            # the legacy interaction-conditioned success field.
            "success": False,
            "nav_success": True,
            "reference_path_length_m": 4.0,
            "navigation_path_length_m": 5.0,
            # Deliberately stale: round aggregation must recompute SPL from
            # NavToObj success and the two saved planar path lengths.
            "spl": 0.0,
            "required_interaction_success": True,
            "interaction_precision_episode": 0.5,
            "episode_total_cost": 1.0,
            "interaction_action_count": 9,
            "valid_interaction_attempt_count": 1,
            "error_interaction_attempt_count": 8,
        },
        {
            "episode_index": 2,
            "domains": ["container"],
            "interaction_requirement": "required",
            "success": False,
            "nav_success": False,
            "reference_path_length_m": 3.0,
            "navigation_path_length_m": 2.0,
            "spl": 0.0,
            "required_interaction_success": False,
            "interaction_precision_episode": 0.0,
            "episode_total_cost": 4.0,
            "interaction_action_count": 1,
            "valid_interaction_attempt_count": 0,
            "error_interaction_attempt_count": 1,
        },
        {
            "episode_index": 3,
            "domains": ["channel", "container"],
            "interaction_requirement": "unnecessary",
            "success": False,
            "nav_success": True,
            "reference_path_length_m": 1.0,
            "navigation_path_length_m": 2.0,
            "spl": 0.5,
            # This must not enter ISR's required-only denominator.
            "required_interaction_success": True,
            # IP is an episode macro, not 1 / (9 + 1 + 0).
            "interaction_precision_episode": 1.0,
            "episode_total_cost": 7.0,
            "interaction_action_count": 0,
            "valid_interaction_attempt_count": 0,
            "error_interaction_attempt_count": 0,
        },
    ]
    for index, result in enumerate(saved_rows):
        _write_json(
            tmp_path / f"worker_{index}" / "episode_result.json",
            {
                "result": {
                    **result,
                    "status": "complete",
                    "scoring_eligible": True,
                    "paper_metric_schema_version": (
                        "interactive_nav_v3_paper_metrics_v1"
                    ),
                    "paper_metric_config": metric_config,
                }
            },
        )

    summary = v3_round_summary.summarise_round(tmp_path)
    overall = summary["paper_metrics"]["groups"]["overall"]

    assert overall["sr"] == pytest.approx(2 / 3)
    assert overall["spl"] == pytest.approx(1.3 / 3)
    assert overall["isr"] == pytest.approx(0.5)
    assert overall["ip"] == pytest.approx(0.5)
    assert overall["total_cost"] == pytest.approx(4.0)
    assert overall["success_rate"] == overall["sr"]
    assert overall["mean_spl"] == overall["spl"]
    assert overall["required_interaction_success_rate"] == overall["isr"]
    assert overall["interaction_precision"] == overall["ip"]
    assert overall["mean_total_cost"] == overall["total_cost"]
    assert overall["sr_denominator"] == 3
    assert overall["isr_denominator"] == 2
    assert overall["ip_denominator"] == 3
    assert summary["paper_metrics"]["paper_metric_config"] == metric_config
    assert summary["paper_metrics"]["groups"]["domain/mixed"]["total_cost"] == 7.0
    unnecessary = summary["paper_metrics"]["groups"]["requirement/unnecessary"]
    assert unnecessary["isr"] is None
    assert unnecessary["isr_denominator"] == 0
    assert summary["episodes"][0]["paper_metrics"][
        "valid_interaction_attempt_count"
    ] == 1
    assert "paper=SR=66.7%" in v3_round_summary.render_terminal_summary(summary)


def test_round_summary_refuses_incomplete_or_mixed_paper_metric_records(
    tmp_path: Path,
) -> None:
    shared = {
        "status": "complete",
        "scoring_eligible": True,
        "domains": ["channel"],
        "interaction_requirement": "required",
        "nav_success": True,
        "reference_path_length_m": 1.0,
        "navigation_path_length_m": 1.0,
        "spl": 1.0,
        "required_interaction_success": True,
        "paper_metric_schema_version": "interactive_nav_v3_paper_metrics_v1",
    }
    _write_json(
        tmp_path / "worker_0" / "episode_result.json",
        {
            **shared,
            "interaction_precision_episode": 1.0,
            "episode_total_cost": 2.0,
            "paper_metric_config": {"lambda": 0.3, "mu": 1.0, "kappa": 5.0},
        },
    )
    _write_json(
        tmp_path / "worker_1" / "episode_result.json",
        {
            **shared,
            # Do not silently omit this row from the IP macro average.
            "episode_total_cost": 3.0,
            # A different cost parameter vector makes an aggregate cost
            # scientifically meaningless even though both scalars exist.
            "paper_metric_config": {"lambda": 0.4, "mu": 1.0, "kappa": 5.0},
        },
    )

    summary = v3_round_summary.summarise_round(tmp_path)
    overall = summary["paper_metrics"]["groups"]["overall"]

    assert overall["sr"] == 1.0
    assert overall["ip"] is None
    assert overall["ip_missing_count"] == 1
    assert overall["total_cost"] is None
    assert overall["total_cost_missing_count"] == 0
    assert overall["paper_metric_config_consistent"] is False
    assert any("Total Cost is unavailable" in warning for warning in summary["warnings"])


def test_round_summary_merges_batch_manifest_latest_attempt_and_runtime_diagnostics(
    tmp_path: Path,
) -> None:
    completed_task = tmp_path / "episode_0007"
    incomplete_task = tmp_path / "episode_0008"
    stale_attempt = completed_task / "attempt_001"
    latest_attempt = completed_task / "attempt_002"
    incomplete_attempt = incomplete_task / "attempt_001"
    stale_result = stale_attempt / "eval" / "episodes" / "stale" / "episode_result.json"
    latest_result = latest_attempt / "eval" / "episodes" / "latest" / "episode_result.json"
    _write_json(
        tmp_path / "batch_manifest.json",
        {
            "plans": [
                {
                    "episode_index": 7,
                    "worker_id": 2,
                    "output_dir": str(completed_task),
                },
                {
                    "episode_index": 8,
                    "worker_id": 3,
                    "output_dir": str(incomplete_task),
                },
            ]
        },
    )
    _write_json(
        completed_task / "batch_task_summary.json",
        {
            "completed": True,
            "worker_id": 2,
            # Deliberately stale: the analyzer must use attempt_002 rather
            # than double-counting or trusting the former retry result.
            "attempt_dir": str(stale_attempt),
            "episode_result_path": str(stale_result),
        },
    )
    _write_json(
        stale_result,
        {
            "status": "complete",
            "result": {
                "status": "complete",
                "episode_index": 7,
                "task_success": False,
                "nav_success": False,
                "interaction_requirement": "required",
            },
        },
    )
    _write_json(
        latest_result,
        {
            "status": "complete",
            "result": {
                "status": "complete",
                "episode_index": 7,
                "house_index": 11,
                "domains": ["channel", "container"],
                "interaction_requirement": "required",
                "success": True,
                "task_success": True,
                "nav_success": True,
                "required_interaction_success": True,
                "sequence_success": True,
                "terminal_reason": "target_found",
                "step_count": 21,
                "episode_step_budget": 2000,
                "applied_action_step_count": 13,
                "no_fresh_action_count": 4,
                "policy_termination": {
                    "max_consecutive_no_fresh_action_count": 3,
                    "max_no_fresh_wall_seconds": 1.25,
                    "reason": "ros_bridge_command_starvation",
                    "source": "command_starvation",
                },
                "invalid_interaction_action_count": 1,
                "interaction_attempts": [
                    {
                        "status": "invalid",
                        "result_status": "INVALID",
                        "reason": "unknown_instance_id",
                    }
                ],
                "elapsed_seconds": 12.0,
            },
        },
    )
    (stale_attempt / "mllm_metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (stale_attempt / "mllm_metrics.jsonl").write_text(
        json.dumps({"role": "attribute_inference", "error": "stale_error"}) + "\n",
        encoding="utf-8",
    )
    (latest_attempt / "mllm_metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (latest_attempt / "mllm_metrics.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"role": "attribute_inference", "error": "request_timeout"},
                {"role": "attribute_inference", "error": ""},
                {"role": "subgoal_selection", "candidate_ids": ["frontier:1"], "error": ""},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        incomplete_task / "batch_task_summary.json",
        {
            "completed": False,
            "worker_id": 3,
            "attempt_dir": str(incomplete_attempt),
            "runner_exit_code": 124,
            "timed_out": True,
            "error": "scene timeout",
        },
    )
    _write_json(
        incomplete_attempt / "eval" / "episodes" / "partial" / "episode_result.json",
        {
            "status": "complete",
            "result": {
                "status": "complete",
                "episode_index": 8,
                # A runner-level failure must override these evaluator values
                # for formal scoring while preserving them as diagnostics.
                "scoring_eligible": True,
                "success": True,
                "task_success": True,
                "nav_success": True,
                "interaction_requirement": "required",
                "required_interaction_success": True,
                "sequence_success": True,
                "terminal_reason": "target_found",
            },
        },
    )
    (incomplete_task / "mllm_metrics.jsonl").write_text(
        json.dumps({"role": "attribute_inference", "error": "must_not_count"}) + "\n",
        encoding="utf-8",
    )

    summary = v3_round_summary.summarise_round(tmp_path)
    diagnostics = summary["round_diagnostics"]
    completed_row, incomplete_row = summary["episodes"]

    assert summary["episode_count"] == 2
    assert diagnostics["completion"] == {
        "planned_episode_count": 2,
        "completed_episode_count": 1,
        "incomplete_episode_count": 1,
        "completed_result_episode_count": 1,
        "missing_result_episode_count": 0,
    }
    assert completed_row["worker"] == "worker_2"
    assert completed_row["artifacts"]["episode_result"] == str(latest_result)
    assert completed_row["mllm"]["call_count"] == 3
    assert completed_row["mllm"]["m1_call_count"] == 2
    assert completed_row["mllm"]["m1_error_reason_counts"] == {"request_timeout": 1}
    assert completed_row["outcomes"] == {
        "task_success": True,
        "nav_success": True,
        "required_interaction_success": True,
        "sequence_success": True,
    }
    assert completed_row["policy_termination"] == {
        "no_fresh_action_count": 4,
        "applied_action_step_count": 13,
        "max_consecutive_no_fresh_action_count": 3,
        "max_no_fresh_wall_seconds": 1.25,
        "triggered": None,
        "reason": "ros_bridge_command_starvation",
        "source": "command_starvation",
    }
    assert diagnostics["interactions"] == {
        "invalid_interaction_action_count": 1,
        "unknown_interaction_attempt_count": 1,
        "invalid_reason_counts": {"unknown_instance_id": 1},
        "failed_interaction_attempt_count": 0,
        "failed_reason_counts": {},
        "unknown_reason_counts": {"unknown_instance_id": 1},
    }
    assert diagnostics["ros_policy"]["no_fresh_action_count"] == 4
    assert diagnostics["ros_policy"]["applied_action_step_count"] == 13
    assert diagnostics["m1"] == {
        "metrics_available_episode_count": 1,
        "call_count": 2,
        "error_count": 1,
        "error_reason_counts": {"request_timeout": 1},
    }
    assert incomplete_row["status"] == "incomplete"
    assert incomplete_row["terminal_reason"] == "incomplete"
    assert incomplete_row["scoring_eligible"] is False
    assert incomplete_row["success"] is False
    assert incomplete_row["completion"]["evaluator_success"] is True
    assert incomplete_row["completion"]["evaluator_terminal_reason"] == "target_found"
    assert incomplete_row["mllm"]["metrics_available"] is False
    rendered = v3_round_summary.render_terminal_summary(summary)
    assert "planned=2 completed=1 incomplete=1" in rendered
    assert "M1 err/calls" in rendered


def test_round_summary_attributes_failed_public_preconditions_to_invalid_metric(
    tmp_path: Path,
) -> None:
    """Restricted V3 writes these evaluator-invalid requests as FAILED traces."""

    result_path = tmp_path / "worker_0" / "episode_result.json"
    _write_json(
        result_path,
        {
            "episode_index": 4,
            "success": False,
            "terminal_reason": "policy_exploration_stalled",
            "invalid_interaction_action_count": 2,
            "interaction_attempts": [
                {
                    "status": "FAILED",
                    "result_status": "FAILED",
                    "failure_reason": "interaction_not_visible",
                },
                {
                    "status": "FAILED",
                    "failure_reason": "interaction_too_far",
                },
                {
                    "status": "FAILED",
                    "failure_reason": "force_target_not_reached",
                },
                {
                    "status": "SUCCEEDED",
                    "failure_reason": "interaction_not_visible",
                },
            ],
        },
    )

    summary = v3_round_summary.summarise_round(tmp_path)
    episode = summary["episodes"][0]
    diagnostics = summary["round_diagnostics"]["interactions"]

    assert episode["interaction_diagnostics"] == {
        "invalid_reason_counts": {
            "interaction_not_visible": 1,
            "interaction_too_far": 1,
        },
        "failed_reason_counts": {"force_target_not_reached": 1},
        "failed_interaction_attempt_count": 1,
        "unknown_reason_counts": {},
        "unknown_attempt_count": 0,
    }
    assert diagnostics == {
        "invalid_interaction_action_count": 2,
        "unknown_interaction_attempt_count": 0,
        "invalid_reason_counts": {
            "interaction_not_visible": 1,
            "interaction_too_far": 1,
        },
        "failed_interaction_attempt_count": 1,
        "failed_reason_counts": {"force_target_not_reached": 1},
        "unknown_reason_counts": {},
    }


def test_round_summary_rejects_ambiguous_latest_attempt_results(tmp_path: Path) -> None:
    task = tmp_path / "episode_0009"
    attempt = task / "attempt_001"
    _write_json(
        tmp_path / "batch_manifest.json",
        {"plans": [{"episode_index": 9, "worker_id": 0, "output_dir": str(task)}]},
    )
    _write_json(
        task / "batch_task_summary.json",
        {"completed": True, "worker_id": 0, "attempt_dir": str(attempt)},
    )
    for case in ("duplicate_a", "duplicate_b"):
        _write_json(
            attempt / "eval" / "episodes" / case / "episode_result.json",
            {
                "status": "complete",
                "result": {
                    "status": "complete",
                    "episode_index": 9,
                    "success": True,
                    "task_success": True,
                    "nav_success": True,
                    "interaction_requirement": "required",
                },
            },
        )

    summary = v3_round_summary.summarise_round(tmp_path)
    row = summary["episodes"][0]

    assert row["artifacts"]["episode_result"] is None
    assert row["status"] == "incomplete"
    assert row["completion"]["completed"] is False
    assert row["scoring_eligible"] is False
    assert summary["round_diagnostics"]["completion"]["incomplete_episode_count"] == 1
    assert summary["paper_metrics"]["scoring_eligible_episode_count"] == 0
    assert any("Ambiguous results for planned episode 9" in warning for warning in summary["warnings"])


def test_round_summary_requires_batch_wrapper_completion_record(tmp_path: Path) -> None:
    """An evaluator result alone is not final evidence for a planned batch task."""

    task = tmp_path / "episode_0010"
    attempt = task / "attempt_001"
    _write_json(
        tmp_path / "batch_manifest.json",
        {"plans": [{"episode_index": 10, "worker_id": 0, "output_dir": str(task)}]},
    )
    _write_json(
        attempt / "eval" / "episodes" / "case" / "episode_result.json",
        {
            "status": "complete",
            "result": {
                "status": "complete",
                "episode_index": 10,
                "scoring_eligible": True,
                "success": True,
                "task_success": True,
                "nav_success": True,
                "interaction_requirement": "required",
                "required_interaction_success": True,
                "sequence_success": True,
                "terminal_reason": "target_found",
            },
        },
    )

    summary = v3_round_summary.summarise_round(tmp_path)
    row = summary["episodes"][0]

    assert row["status"] == "incomplete"
    assert row["terminal_reason"] == "incomplete"
    assert row["completion"]["completed"] is False
    assert row["completion"]["evaluator_terminal_reason"] == "target_found"
    assert row["scoring_eligible"] is False
    assert row["success"] is False
    assert summary["success_count"] == 0
    assert summary["round_diagnostics"]["completion"]["completed_result_episode_count"] == 0
    assert summary["paper_metrics"]["scoring_eligible_episode_count"] == 0
