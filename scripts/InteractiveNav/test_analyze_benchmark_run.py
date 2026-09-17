import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).with_name("analyze_benchmark_run.py")
spec = importlib.util.spec_from_file_location("offline_analyzer", SCRIPT)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


@pytest.mark.parametrize(
    ("reason", "extra", "expected"),
    [
        ("target_found", {"success": True}, "verified_success"),
        (
            "target_found",
            {"task_success": True, "required_interaction_success": False},
            "formal_success_blocked_by_interaction_requirement",
        ),
        (
            "target_claim_unverified",
            {"goal_definition_relaxed_success": True},
            "wrong_instance_same_category_target_claim",
        ),
        (
            "target_claim_unverified",
            {"target_distance_m": 1.2},
            "near_target_without_public_visual_evidence",
        ),
        (
            "ros_bridge_observation_turn_limit",
            {
                "step_count": 100,
                "no_fresh_action_count": 20,
                "interaction_attempts": [{"success": False, "failure_reason": "capability_unavailable"}],
            },
            "budget_exhaustion_after_wrong_interaction_binding",
        ),
        (
            "policy_exploration_stalled",
            {
                "policy_termination": {
                    "terminal_detail": {
                        "reported_reason": "no_eligible_candidates_after_bounded_recovery"
                    },
                    "latest_goal_status": {
                        "detail": {
                            "reason": "no_eligible_candidates_after_bounded_recovery",
                            "eligible_candidate_count": 0,
                            "executable_candidate_count": 0,
                            "recovery_scan_count": 2,
                            "exploration_context": {
                                "raw_frontier_cluster_count": 27,
                                "raw_frontier_cell_count": 296,
                                "proposal_count": 0,
                            },
                        }
                    },
                }
            },
            "candidate_generation_exhausted_despite_raw_frontiers",
        ),
        (
            "cross_subgoal_navigation_stall",
            {"early_stop": {"failed_subgoal_count": 8, "displacement_m": 0.01}},
            "repeated_navigation_failures_with_low_displacement",
        ),
    ],
)
def test_terminal_diagnosis_rules(reason, extra, expected):
    result = {"terminal_reason": reason, **extra}
    diagnosis, evidence, recommendations = analyzer.classify_episode(result, {})
    assert diagnosis == expected
    assert evidence
    if expected == "candidate_generation_exhausted_despite_raw_frontiers":
        rendered = " ".join(evidence).lower()
        assert "frontier" in rendered and "27" in rendered and "296" in rendered
    if not result.get("success"):
        assert recommendations


@pytest.mark.parametrize(
    ("detail", "early_stop", "expected"),
    [
        (
            {
                "reason": "semantic_mission_no_progress",
                "failure_reason": "interaction_approach_options_exhausted",
                "semantic_mission_no_progress": True,
                "mission_stalled": True,
                "subgoal_stalled": False,
                "subgoal_key": "portal_obj_81|navigation|10",
                "subgoal_elapsed_task_steps": 0,
                "mission_elapsed_task_steps": 181,
                "subgoal_timeout_task_steps": 60,
                "mission_timeout_task_steps": 180,
                "interaction_approach_options_exhausted": True,
            },
            {"observed_navigation_failure_count": 2, "displacement_m": 0.01},
            "global_mission_timer_triggered_on_new_subgoal",
        ),
        (
            {
                "reason": "semantic_mission_no_progress",
                "failure_reason": "interaction_approach_options_exhausted",
                "semantic_mission_no_progress": True,
                "mission_stalled": False,
                "subgoal_stalled": True,
                "subgoal_key": "portal_obj_42|navigation|3",
                "subgoal_elapsed_task_steps": 61,
                "mission_elapsed_task_steps": 100,
                "subgoal_timeout_task_steps": 60,
                "mission_timeout_task_steps": 180,
                "interaction_approach_options_exhausted": True,
                "local_plan_fresh": False,
            },
            {
                "observed_navigation_failure_count": 4,
                "failed_subgoal_count": 2,
                "displacement_m": 0.02,
                "failure_reason_counts": {"FAILED": 3, "semantic_subgoal_no_progress": 1},
            },
            "portal_approach_exhausted_after_navigation_stagnation",
        ),
    ],
)
def test_semantic_mission_stall_distinguishes_global_timer_from_local_stagnation(
    detail, early_stop, expected
):
    result = {
        "terminal_reason": "policy_exploration_stalled",
        "policy_termination": {
            "terminal_detail": {"reported_reason": "semantic_mission_no_progress"},
            "latest_goal_status": {"detail": detail},
        },
        "early_stop": early_stop,
    }

    diagnosis, evidence, recommendations = analyzer.classify_episode(result, {})

    assert diagnosis == expected
    assert evidence
    assert recommendations
    if expected == "global_mission_timer_triggered_on_new_subgoal":
        rendered = " ".join(evidence + recommendations).lower()
        assert "181" in rendered and "180" in rendered
        assert "suspect" in rendered or "appears" in rendered or "audit" in rendered
        assert "confirmed bug" not in rendered


def make_fixture(tmp_path: Path) -> Path:
    evaluation = tmp_path / "evaluation"
    episode = evaluation / "episode_0007"
    attempt = episode / "attempt_001"
    result_dir = attempt / "eval/episodes/0007_case"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "episode_result.json"
    result_path.write_text(json.dumps({
        "status": "complete",
        "result": {
            "episode_index": 7,
            "case_id": "case_7",
            "house_index": 4,
            "terminal_reason": "target_claim_unverified",
            "success": False,
            "task_success": False,
            "nav_success": False,
            "interaction_conditioned_success": False,
            "required_interaction_success": True,
            "goal_definition_relaxed_success": True,
            "scoring_eligible": True,
            "step_count": 10,
            "applied_action_step_count": 8,
            "no_fresh_action_count": 2,
            "episode_step_budget": 50,
            "target_distance_m": 2.5,
            "target_visibility_fraction": 0,
            "navigation_path_length_m": 3.0,
            "reference_path_length_m": 2.0,
            "spl": 0,
            "elapsed_seconds": 8.0,
            "interaction_attempts": [{"success": True, "status": "SUCCEEDED"}],
            "policy_termination": {
                "terminal_detail": {
                    "observation_turn_count": 10,
                    "observation_turn_limit": 50,
                }
            },
            "early_stop": {},
        },
        "trace": [
            {"decision_step": 0, "action": {"kind": "base"}, "timing_ms": {"total": 1000}},
            {"decision_step": 1, "action": {"kind": "observe"}, "timing_ms": {"total": 3000}},
        ],
    }))
    (attempt / "eval.log").write_text(
        "RosBridgePolicy: action timeout, using noop action.\nmake_plan_unreachable\n"
    )
    (attempt / "runner.log").write_text("runner complete\n")
    (attempt / "offline_video_summary.json").write_text(json.dumps({
        "sim_frame_count": 3,
        "input_step_count": 3,
        "output_frame_count": 3,
        "exact_step_match_count": 3,
        "missing_raw_step_indexes": [],
        "missing_sim_step_indexes": [],
    }))
    (episode / "episode_topdown.json").write_text(json.dumps({
        "coverage": {"exploration_coverage_ratio": 0.5},
        "trajectory_samples": 20,
        "gt_oracle_path": {"complete": True, "length_m": 2.0},
        "gt_interactions": [{"xy": [0, 0]}],
        "actual_interactions": [{"xy": [3, 4]}],
    }))
    summary = {
        "episodes": [{
            "episode_index": 7,
            "attempt": "attempt_001",
            "attempt_dir": str(attempt),
            "episode_result_path": str(result_path),
            "elapsed_sec": 10.0,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:10+00:00",
            "topdown": str(episode / "episode_topdown.png"),
            "six_panel_video": str(episode / "overview_6panel.mp4"),
        }]
    }
    evaluation.mkdir(exist_ok=True)
    (evaluation / "summary.json").write_text(json.dumps(summary))
    return evaluation


def test_cli_builds_reproducible_offline_outputs(tmp_path):
    evaluation = make_fixture(tmp_path)
    output = tmp_path / "analysis"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(evaluation), "--output-dir", str(output),
         "--no-images", "--no-baseline"],
        text=True,
        capture_output=True,
        check=True,
    )
    status = json.loads(completed.stdout)
    assert status["episodes"] == 1
    payload = json.loads((output / "analysis.json").read_text())
    row = payload["episodes"][0]
    assert row["diagnosis"] == "wrong_instance_same_category_target_claim"
    assert row["logs"]["counts"] == {"action_timeout": 1, "make_plan_unreachable": 1}
    assert row["spatial"]["nearest_actual_to_gt_interaction_m"] == 5.0
    assert row["observation_turn_count"] == 10
    assert row["applied_action_step_count"] == 8
    assert row["trace"]["event_count"] == 2
    assert payload["aggregate"]["loop_timing"]["mean_seconds"] == 2.0
    assert (output / "episodes.csv").is_file()
    report = (output / "analysis_report.md").read_text()
    assert "No algorithm source was changed" in report
    assert "target_claim_unverified" in report
    assert "Obs/Applied/Trace/Budget" in report
    assert "10/8/2/50" in report


def _aggregate_row(index: int, *, nav_success: bool, success: bool) -> dict:
    return {
        "episode_index": index,
        "group": "channel",
        "success": success,
        "legacy_interaction_conditioned_success": success,
        "paper_sr_success": nav_success,
        "task_success": nav_success,
        "nav_success": nav_success,
        # Keep this intentionally different from the historical formal `success`
        # field so the compatibility alias cannot silently change its source.
        "interaction_conditioned_success": not success,
        "required_interaction_success": True,
        "goal_definition_relaxed_success": False,
        "spl": 0.0,
        "interaction_precision": 0.0,
        "episode_total_cost": 0.0,
        "step_count": 10,
        "observation_turn_count": 10,
        "spatial": {"coverage_ratio": 0.5},
        "runner_elapsed_seconds": 1.0,
        "terminal_reason": "synthetic",
        "diagnosis": "synthetic",
        "recommendations": [],
        "logs": {"counts": {}},
        "mllm": {
            "call_count": 0,
            "error_count": 0,
            "error_counts": {},
            "role_counts": {},
            "_latencies": [],
            "_queue_lags": [],
        },
        "recording": {
            "output_frame_count": 0,
            "exact_alignment": False,
            "capture_complete": False,
            "encoding_complete": False,
            "complete": False,
            "missing_raw_step_count": 0,
            "missing_sim_step_count": 0,
        },
        "paths": {"video": "/definitely/not/a/video.mp4"},
        "_loop_timings": [],
    }


def test_aggregate_exposes_canonical_paper_sr_and_legacy_ics():
    rows = [
        _aggregate_row(0, nav_success=True, success=False),
        _aggregate_row(1, nav_success=True, success=True),
        _aggregate_row(2, nav_success=False, success=False),
    ]

    group = analyzer.aggregate(rows, {"episodes": []})["groups"]["all"]

    assert group["paper_sr"] == pytest.approx(2 / 3)
    assert group["legacy_ics"] == pytest.approx(1 / 3)
    assert group["paper_sr"] != group["legacy_ics"]


def test_requested_missing_episode_is_rejected(tmp_path):
    evaluation = make_fixture(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(evaluation), "--episode-indices", "8",
         "--no-images", "--no-baseline"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "Requested episodes not present" in completed.stderr


def test_visual_contact_sheet_uses_only_offline_frames(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    evaluation = make_fixture(tmp_path)
    attempt = evaluation / "episode_0007/attempt_001"
    frames = attempt / "sim_step_frames"
    frames.mkdir()
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        Image.new("RGB", (64, 48), color).save(frames / f"step_{index:06d}.png")
    Image.new("RGB", (80, 80), "white").save(evaluation / "episode_0007/episode_topdown.png")
    (evaluation / "episode_0007/overview_6panel.mp4").touch()
    output = tmp_path / "visual-analysis"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(evaluation), "--output-dir", str(output), "--no-baseline"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output / "visual_contact_sheet.jpg").is_file()
    assert (output / "episode_visuals/episode_0007.jpg").is_file()
    payload = json.loads((output / "analysis.json").read_text())
    assert payload["episodes"][0]["visual"]["metrics"]["sample_count"] == 3
    assert payload["episodes"][0]["visual"]["metrics"]["first_last_mean_abs_difference"] > 0


def test_offline_video_completeness_requires_manifest_video_and_all_frame_counts(tmp_path):
    evaluation = make_fixture(tmp_path)
    video = evaluation / "episode_0007/overview_6panel.mp4"
    video.touch()
    summary_path = evaluation / "episode_0007/attempt_001/offline_video_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({
        "sim_frame_count": 10,
        "input_step_count": 10,
        "output_frame_count": 10,
        "exact_step_match_count": 10,
    })
    summary_path.write_text(json.dumps(summary))
    output = tmp_path / "complete-video-analysis"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(evaluation), "--output-dir", str(output),
         "--no-images", "--no-baseline"],
        check=True,
        capture_output=True,
        text=True,
    )
    recording = json.loads((output / "analysis.json").read_text())["episodes"][0]["recording"]
    assert recording["summary_present"] is True
    assert recording["video_present"] is True
    assert recording["sim_frame_count"] == 10
    assert recording["output_frame_count"] == 10
    assert recording["exact_step_match_count"] == 10
    assert recording["exact_alignment"] is True
    assert recording["capture_complete"] is True
    assert recording["encoding_complete"] is True
    assert recording["complete"] is True


def test_offline_video_separates_missing_capture_from_aligned_encoding(tmp_path):
    evaluation = make_fixture(tmp_path)
    video = evaluation / "episode_0007/overview_6panel.mp4"
    video.touch()
    summary_path = evaluation / "episode_0007/attempt_001/offline_video_summary.json"
    summary = json.loads(summary_path.read_text())
    # Mirrors the real 1001 artifact shape: one evaluator/task step was not
    # captured, while every captured frame was encoded with exact alignment.
    summary.update({
        "sim_frame_count": 9,
        "input_step_count": 9,
        "output_frame_count": 9,
        "exact_step_match_count": 9,
    })
    summary_path.write_text(json.dumps(summary))
    output = tmp_path / "capture-gap-analysis"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(evaluation), "--output-dir", str(output),
         "--no-images", "--no-baseline"],
        check=True,
        capture_output=True,
        text=True,
    )
    recording = json.loads((output / "analysis.json").read_text())["episodes"][0]["recording"]
    assert recording["summary_present"] is True
    assert recording["video_present"] is True
    assert recording["exact_alignment"] is True
    assert recording["capture_complete"] is False
    assert recording["encoding_complete"] is True
    assert recording["complete"] is False


def test_offline_video_detects_encoder_count_mismatch(tmp_path):
    evaluation = make_fixture(tmp_path)
    video = evaluation / "episode_0007/overview_6panel.mp4"
    video.touch()
    summary_path = evaluation / "episode_0007/attempt_001/offline_video_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({
        "sim_frame_count": 10,
        "input_step_count": 10,
        "output_frame_count": 9,
        "exact_step_match_count": 9,
    })
    summary_path.write_text(json.dumps(summary))
    output = tmp_path / "encoder-gap-analysis"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(evaluation), "--output-dir", str(output),
         "--no-images", "--no-baseline"],
        check=True,
        capture_output=True,
        text=True,
    )
    recording = json.loads((output / "analysis.json").read_text())["episodes"][0]["recording"]
    assert recording["capture_complete"] is True
    # Compatibility field means all *output* frames were exact matches.  The
    # stricter encoding_complete field additionally detects the dropped frame.
    assert recording["exact_alignment"] is True
    assert recording["encoding_complete"] is False
    assert recording["complete"] is False


def test_offline_video_is_incomplete_when_rendered_video_is_missing(tmp_path):
    evaluation = make_fixture(tmp_path)
    output = tmp_path / "missing-video-analysis"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(evaluation), "--output-dir", str(output),
         "--no-images", "--no-baseline"],
        check=True,
        capture_output=True,
        text=True,
    )
    recording = json.loads((output / "analysis.json").read_text())["episodes"][0]["recording"]
    assert recording["summary_present"] is True
    assert recording["video_present"] is False
    assert recording["exact_alignment"] is True
    assert recording["capture_complete"] is False
    assert recording["encoding_complete"] is True
    assert recording["complete"] is False


def test_shutdown_traceback_and_config_words_are_not_runtime_failures(tmp_path):
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    (attempt / "eval.log").write_text(
        "worker entered evaluation loop\n"
        "Traceback (most recent call last):\n"
        "  File 'benchmark_runner.py', line 99, in run\n"
        "RuntimeError: planner exploded\n"
    )
    (attempt / "runner.log").write_text(
        "[v3-eval] request_timeout_s=30 overflow=drop_oldest\n"
    )
    (attempt / "roslaunch.log").write_text(
        " * /node/object_detection/timeout: 5.0\n"
        "[semantic_mapping_py-6] killing on exit\n"
        "Traceback (most recent call last):\n"
        "  File 'semantic_mapping_node.py', line 1\n"
        "rospy.exceptions.ROSException: publish() to a closed topic\n"
    )
    counts, samples, _files = analyzer.collect_log_evidence(attempt)
    assert counts == {"runtime_traceback": 1, "shutdown_traceback": 1}
    assert "runtime_traceback" in samples
    assert "shutdown_traceback" in samples


def test_graph_revision_summary_is_structured_and_target_aware(tmp_path):
    path = tmp_path / "graph_revision_events.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "event": "NEW_NODE", "graph_revision": 1, "step_id": 2,
                "label": "alarm_clock", "node_id": "object_obj_2",
            }),
            json.dumps({
                "event": "STATE_CHANGED", "graph_revision": 3, "step_id": 9,
                "before": "closed", "after": "open", "node_id": "container_obj_1",
            }),
            json.dumps({
                "event": "NEW_EDGE", "graph_revision": 4, "step_id": 10,
                "relation": "contains", "src_id": "container_obj_1",
                "dst_id": "object_obj_2",
            }),
            "{not-json}",
        ])
    )

    summary = analyzer.collect_graph_revision_summary(
        path, "alarmclock_deadbeef_1_0_2"
    )

    assert summary["event_count"] == 3
    assert summary["event_counts"] == {
        "NEW_NODE": 1, "STATE_CHANGED": 1, "NEW_EDGE": 1,
    }
    assert summary["last_graph_revision"] == 4
    assert summary["state_transition_counts"] == {"closed->open": 1}
    assert summary["target_category_node_ids"] == ["object_obj_2"]
    assert summary["invalid_row_count"] == 1
