import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts.InteractiveNav.evaluation import m2_online_monitor as monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def manifest(root):
    lanes = [{"id": index, "output_dir": str(root / f"lane{index}")} for index in range(3)]
    jobs = [{"job_id": f"G0-mixed{index:04d}", "arm": "G0", "episode_index": 2000 + index,
             "mixed_index": index, "lane_id": index, "output_dir": str(root / f"job{index}")} for index in range(3)]
    value = {"schema_version": "interactive_nav_m2_online_experiment_v1", "experiment_id": "test", "lanes": lanes, "planned_jobs": jobs}
    write(root / "manifest.json", value)
    return value


def test_waiting_manifest_and_missing_lanes_are_not_finished(tmp_path):
    assert monitor.snapshot(tmp_path)["state"] == "waiting_manifest"
    manifest(tmp_path)
    value = monitor.snapshot(tmp_path, now=100)
    assert value["counts"]["queued"] == 3
    assert not value["finished"]
    assert all(row["state"] == "not_started" for row in value["lanes"])
    assert "尚未启动" in monitor.format_status(value)


def test_canonical_success_not_nav_or_aggregate_and_denominators(tmp_path):
    planned = manifest(tmp_path)
    result_path = tmp_path / "job0" / "result.json"
    write(result_path, {"status": "complete", "result": {"episode_index": 2000, "success": False, "nav_success": True, "spl": 0.7,
                                                        "applied_action_step_count": 40, "step_count": 65, "elapsed_seconds": 20}})
    write(Path(planned["planned_jobs"][0]["output_dir"]) / "batch_task_summary.json",
          {"completed": True, "episode_index": 2000, "success": True, "success_rate": 1.0, "elapsed_sec": 25,
           "episode_result_path": str(result_path)})
    write(tmp_path / "lane1" / "lane_status.json", {"lane_id": 1, "experiment_id": "test", "state": "running", "updated_at": 100,
                                                   "active": [{"job_id": "G0-mixed0001"}]})
    value = monitor.snapshot(tmp_path, now=105)
    assert value["counts"]["completed"] == value["counts"]["running"] == value["counts"]["queued"] == 1
    assert value["metrics"]["ics"]["successes"] == 0
    assert value["metrics"]["nav_sr"]["successes"] == 1
    assert value["metrics"]["ics"]["completed_denominator"] == 1
    assert value["metrics"]["ics"]["planned_denominator"] == 3
    assert value["metrics"]["applied_steps"]["mean"] == 40
    assert value["metrics"]["observation_steps"]["mean"] == 65
    assert value["metrics"]["runner_seconds"]["mean"] == 25
    rendered = monitor.format_status(value)
    assert "ICS(有效)=0/1" in rendered and "ICS(计划)=0/3" in rendered
    assert "episode_" not in rendered and "mixed000" not in rendered


def test_stale_active_and_cloud_failed_are_visible_without_secrets(tmp_path):
    manifest(tmp_path)
    write(tmp_path / "lane0" / "lane_status.json", {"lane_id": 0, "state": "running", "updated_at": 1, "active": [{"job_id": "G0-mixed0000"}]})
    write(tmp_path / "task_status.json", {"lane_id": 2, "state": "Failed", "updated_at": 150, "key": "must-not-be-published", "error": "secret error"})
    value = monitor.snapshot(tmp_path, now=200, stale_after=120)
    assert value["counts"]["running_with_stale_heartbeat"] == 1
    assert value["counts"]["queued_blocked_by_lane"] == 1
    text = monitor.format_status(value)
    assert "heartbeat过期" in text and "云端failed" in text
    assert "must-not-be-published" not in json.dumps(value) and "secret error" not in text
    assert not value["finished"]


def test_finished_algorithm_failure_is_not_infra_and_incomplete_retained(tmp_path):
    value = manifest(tmp_path)
    for index, job in enumerate(value["planned_jobs"]):
        write(Path(job["output_dir"]) / "batch_task_summary.json", {"episode_index": job["episode_index"], "completed": index < 2,
                                                                  "success": index == 0, "nav_success": index == 0})
    result = monitor.snapshot(tmp_path)
    assert result["finished"] and result["state"] == "finished_with_incomplete"
    assert result["counts"]["completed"] == 2 and result["counts"]["infra_or_incomplete"] == 1
    assert result["metrics"]["ics"]["completed_rate"] == 0.5
    assert result["metrics"]["ics"]["planned_rate_lower_bound"] == 1 / 3


def test_missing_metrics_identity_and_partial_json_do_not_become_zero(tmp_path):
    value = manifest(tmp_path)
    write(Path(value["planned_jobs"][0]["output_dir"]) / "batch_task_summary.json", {"completed": True, "episode_index": 2000})
    write(Path(value["planned_jobs"][1]["output_dir"]) / "batch_task_summary.json", {"completed": True, "episode_index": 999})
    path = Path(value["planned_jobs"][2]["output_dir"]) / "batch_task_summary.json"
    path.parent.mkdir(parents=True); path.write_text('{"completed":')
    result = monitor.snapshot(tmp_path)
    assert result["counts"]["completed"] == 1
    assert result["metrics"]["ics"]["completed_rate"] is None
    assert result["metrics"]["ics"]["missing_completed_metrics"] == 1
    assert result["metrics"]["spl"] == {"mean": None, "n": 0}
    assert result["data_warnings"]["episode_summary_identity_mismatch"] == 1


def test_atomic_publication_appends_without_episode_spam(tmp_path):
    manifest(tmp_path)
    status = monitor.snapshot(tmp_path, now=100)
    first = monitor.publish(tmp_path, status)
    monitor.publish(tmp_path, status)
    assert json.loads((tmp_path / "overall_status.json").read_text()) == status
    assert (tmp_path / "overall.log").read_text() == (first + "\n") * 2
    assert not list(tmp_path.glob(".overall_status.json.tmp.*"))


def test_restart_epoch_is_published_each_time_without_counting_archive(tmp_path):
    planned = manifest(tmp_path)
    remote = planned["planned_jobs"][2]
    write(Path(remote["output_dir"]) / "batch_task_summary.json",
          {"episode_index": remote["episode_index"], "completed": True, "success": True})
    baseline = monitor.snapshot(tmp_path, now=100)
    archive = tmp_path / "infrastructure_attempts" / "local_pre_cleanup"
    write(archive / "0" / "batch_task_summary.json", {"episode_index": 2000, "completed": True, "success": True})
    write(tmp_path / "restart_epoch.json", {"epoch_id": "local_restart_20260922_0842", "affected_lanes": [0, 1],
          "remote_preserved": True, "archive_root": str(archive), "private_detail": "must-not-copy"})
    status = monitor.snapshot(tmp_path, now=100)
    assert status["restart_epoch"] == {"epoch_id": "local_restart_20260922_0842", "affected_lanes": [0, 1],
                                       "remote_preserved": True, "archive_root": str(archive), "archived_results_included": False}
    assert {key: value for key, value in status.items() if key != "restart_epoch"} == baseline
    assert status["counts"]["completed"] == 1 and status["metrics"]["ics"]["planned_denominator"] == 3
    for _ in range(2):
        text = monitor.publish(tmp_path, status)
        assert "epoch=local_restart_20260922_0842" in text and "affected_lanes=[0,1]" in text
        assert "旧结果已归档、不计入本轮" in text and "remote_preserved=true" in text
    assert (tmp_path / "overall.log").read_text().count("epoch=local_restart_20260922_0842") == 2
    assert "must-not-copy" not in json.dumps(status)


def test_missing_or_invalid_restart_epoch_is_backward_compatible(tmp_path):
    manifest(tmp_path)
    baseline = monitor.snapshot(tmp_path, now=100)
    assert "restart_epoch" not in baseline and "本地重跑" not in monitor.format_status(baseline)
    path = tmp_path / "restart_epoch.json"
    for contents in ('{"epoch_id":', '[]', '{"epoch_id":"bad\\nline"}'):
        path.write_text(contents)
        assert monitor.snapshot(tmp_path, now=100) == baseline


def test_restart_epoch_visible_while_waiting_for_manifest(tmp_path):
    write(tmp_path / "restart_epoch.json", {"epoch_id": "local_restart", "affected_lanes": [0, 1], "remote_preserved": False})
    status = monitor.snapshot(tmp_path, now=100)
    assert status["state"] == "waiting_manifest" and not status["finished"]
    assert "epoch=local_restart" in monitor.format_status(status)
    assert "remote_preserved=false" in monitor.format_status(status)


def test_cloud_banner_status_authoritative_and_terminal_stops_queries(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='Upgrade available [notice]\n[{"Id":"task-1","Name":"secret","Status":"Failed","ExitCode":0}]', stderr="secret stderr")

    monkeypatch.setattr(monitor.subprocess, "run", run)
    poller = monitor.CloudPoller("task-1", 2)
    assert poller.poll(tmp_path, now=100, monotonic=1)
    assert not poller.poll(tmp_path, now=200, monotonic=101)
    value = json.loads((tmp_path / "task_status.json").read_text())
    assert value["state"] == "failed" and value["exit_code"] == 0
    assert value["lane_id"] == 2 and len(calls) == 1
    assert calls[0][0] == ["volc", "ml_task", "get", "-i", "task-1", "--output", "json", "--format", "Id,Name,Status,ExitCode"]
    assert calls[0][1]["timeout"] == 20
    assert "secret" not in json.dumps(value)


def test_cloud_poll_interval_timeout_redaction_and_last_good_age(tmp_path, monkeypatch):
    responses = [SimpleNamespace(returncode=0, stdout='[{"Id":"task-2","Status":"Running"}]'),
                 subprocess.TimeoutExpired(["secret command"], 20, output="secret token", stderr="secret error")]

    def run(*args, **kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(monitor.subprocess, "run", run)
    poller = monitor.CloudPoller("task-2", 2)
    assert poller.poll(tmp_path, now=100, monotonic=1)
    assert not poller.poll(tmp_path, now=130, monotonic=31)
    assert poller.poll(tmp_path, now=161, monotonic=62)
    value = json.loads((tmp_path / "task_status.json").read_text())
    assert value["state"] == "running" and monitor._timestamp(value["updated_at"]) == 100
    assert value["error_count"] == 1 and value["error_type"] == "TimeoutExpired"
    assert "secret" not in json.dumps(value)


def test_cloud_wrong_task_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout='[{"Id":"another-task","Status":"Running","secret":"do not copy"}]'))
    assert monitor.CloudPoller("expected-task", 2).poll(tmp_path, now=100, monotonic=1)
    value = json.loads((tmp_path / "task_status.json").read_text())
    assert "state" not in value and value["error_type"] == "ValueError"
    assert "secret" not in json.dumps(value)


def test_runtime_evidence_is_bounded_and_not_applied_action_count(tmp_path):
    planned = manifest(tmp_path)
    write(tmp_path / "lane0" / "lane_status.json", {"state": "running", "updated_at": 100,
                                                   "active": [{"job_id": "G0-mixed0000"}]})
    attempt = Path(planned["planned_jobs"][0]["output_dir"]) / "attempt_001"
    attempt.mkdir(parents=True)
    log = attempt / "roslaunch.log"
    log.write_text("[INFO] SemanticMappingCausalReady source_seq=99 source_stamp=1 capture_step=7 room_latency_ms=1\n")
    tracker = monitor.ProgressTracker()
    first = monitor.snapshot(tmp_path, now=100, progress_tracker=tracker)["runtime_progress"]
    assert first["workers_with_observation_evidence"] == 1
    assert first["applied_action_sum_known_workers"] is None
    assert first["workers_advanced_since_previous_scan"] is None
    assert monitor.snapshot(tmp_path, now=110, progress_tracker=tracker)["runtime_progress"]["workers_advanced_since_previous_scan"] == 0
    log.write_text("[INFO] SemanticMappingCausalReady source_seq=99 source_stamp=2 capture_step=8 room_latency_ms=1\n")
    second = monitor.snapshot(tmp_path, now=120, progress_tracker=tracker)["runtime_progress"]
    assert second["workers_advanced_since_previous_scan"] == 1
    assert monitor._timestamp(second["last_observed_advance_at"]) == 120
    # An older attempt cannot hide a newer empty attempt, and log activity alone
    # does not prove either observations or physical applied actions.
    newer = attempt.parent / "attempt_002"
    newer.mkdir()
    (newer / "eval.log").write_text("model running\n" * 20000)
    row = monitor._runtime_evidence(planned["planned_jobs"][0], {}, tmp_path)
    assert row["observation_step_index"] is None
    assert len(monitor._tail(newer / "eval.log").encode()) <= 131072
    assert "mixed0000" not in monitor.format_status(monitor.snapshot(tmp_path))


def test_starting_qwen_heartbeat_and_failure_redaction(tmp_path):
    manifest(tmp_path)
    write(tmp_path / "lane0" / "lane_status.json", {"state": "starting_qwen", "updated_at": 100})
    write(tmp_path / "lane1" / "lane_status.json", {
        "state": "failed", "updated_at": 1, "error": "RuntimeError: credential=secret https://secret",
        "failure_queue": [{"job_id": "G0-mixed0001", "classification": "incomplete_unclassified", "reason": "secret token"}],
    })
    value = monitor.snapshot(tmp_path, now=105)
    assert value["lanes"][0]["state"] == "starting_qwen" and not value["lanes"][0]["heartbeat_stale"]
    assert value["lanes"][1]["failure_classifications"] == {"incomplete_unclassified": 1}
    assert "RuntimeError" in monitor.format_status(value)
    assert "secret" not in json.dumps(value) and "secret" not in monitor.format_status(value)
    assert "保守暂停" in monitor._safe_lane_error("consecutive_incomplete_threshold; secret")
    assert monitor._safe_lane_error("Qwen model exited (3); see /secret. No restart performed.") == "模型服务退出(code=3)，未重启"


def test_warn_level_slam_frames_and_pose_changes_are_not_action_counts(tmp_path):
    planned = manifest(tmp_path)
    write(tmp_path / "lane0" / "lane_status.json", {"state": "running", "updated_at": 100,
                                                   "active": [{"job_id": "G0-mixed0000"}]})
    attempt = Path(planned["planned_jobs"][0]["output_dir"]) / "attempt_001"
    attempt.mkdir(parents=True)
    log = attempt / "roslaunch.log"
    log.write_text("update frame 11\nLaser Pose= 2.14857 6.21334 0.151184\nupdate frame 12\nLaser Pose= 2.14552 6.22342 0.37515\n")
    tracker = monitor.ProgressTracker()
    first = monitor.snapshot(tmp_path, now=100, progress_tracker=tracker)
    progress = first["runtime_progress"]
    assert progress["workers_with_observation_evidence"] == progress["workers_with_slam_frames"] == 1
    assert progress["workers_with_slam_pose_change_in_tail"] == 1
    assert progress["applied_action_sum_known_workers"] is None
    assert "有观测/SLAM 1/1" in monitor.format_status(first)
    assert monitor.snapshot(tmp_path, now=110, progress_tracker=tracker)["runtime_progress"]["workers_advanced_since_previous_scan"] == 0
    log.write_text("update frame 13\nLaser Pose= 2.1 6.3 0.4\n")
    assert monitor.snapshot(tmp_path, now=120, progress_tracker=tracker)["runtime_progress"]["workers_advanced_since_previous_scan"] == 1
    log.write_text("update frame 13\nLaser Pose= 2.1 6.3 0.5\n")
    assert monitor.snapshot(tmp_path, now=130, progress_tracker=tracker)["runtime_progress"]["workers_advanced_since_previous_scan"] == 1
    # m_count and arbitrary warning activity cannot be mistaken for action steps.
    row = monitor._runtime_evidence(planned["planned_jobs"][0], {}, tmp_path)
    assert row["observation_step_count"] is None and row["applied_action_step_count"] is None


def test_ineligible_kept_in_completion_not_scores_or_means(tmp_path):
    planned = manifest(tmp_path)
    rows = [
        {"success": False, "nav_success": False, "scoring_eligible": False, "spl": 0,
         "applied_action_step_count": 0, "step_count": 0, "elapsed_sec": 1000},
        {"success": False, "nav_success": False, "scoring_eligible": True, "spl": 0,
         "applied_action_step_count": 10, "step_count": 20, "elapsed_sec": 10},
        {"success": True, "nav_success": True, "scoring_eligible": True, "spl": 0.8,
         "applied_action_step_count": 20, "step_count": 30, "elapsed_sec": 20},
    ]
    for job, row in zip(planned["planned_jobs"], rows):
        write(Path(job["output_dir"]) / "batch_task_summary.json", {"episode_index": job["episode_index"], "completed": True, **row})
    status = monitor.snapshot(tmp_path)
    metrics = status["metrics"]
    assert status["counts"]["completed"] == status["planned"] == 3
    assert metrics["total_completed"] == 3 and metrics["valid_completed"] == 2 and metrics["ineligible_completed"] == 1
    assert metrics["ics"]["completed_denominator"] == 2 and metrics["ics"]["completed_rate"] == 0.5
    assert metrics["ics"]["planned_denominator"] == 3 and metrics["ics"]["planned_rate_lower_bound"] == 1 / 3
    assert metrics["spl"]["mean"] == 0.4 and metrics["applied_steps"]["mean"] == 15
    assert metrics["observation_steps"]["mean"] == 25 and metrics["runner_seconds"]["mean"] == 15
    assert "有效2/3 ineligible=1" in monitor.format_status(status)


def test_recent_rpc_roles_timeouts_median_and_redaction(tmp_path):
    planned = manifest(tmp_path)
    attempt = Path(planned["planned_jobs"][0]["output_dir"]) / "attempt_001"
    attempt.mkdir(parents=True)
    rows = [
        {"role": "attribute_inference", "timestamp": 900, "latency_s": 30, "error": "request timed out secret-key", "raw_text": "secret-response"},
        {"role": "room_attribute_inference", "timestamp": 950, "latency_s": 10, "error": ""},
        {"role": "subgoal_selection", "timestamp": 980, "latency_s": 76, "error": False, "public_request": {"secret": "context"}},
        {"role": "subgoal_selection", "timestamp": 990, "latency_s": 90, "error": True},
        {"role": "attribute_inference", "timestamp": 879, "latency_s": 1, "error": "timeout"},
        {"role": "ignored", "timestamp": 950, "error": "timeout"},
    ]
    (attempt / "mllm_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    value = monitor.snapshot(tmp_path, now=1000)
    rpc = value["rpc_window"]
    assert rpc["groups"]["M1"]["calls"] == 2 and rpc["groups"]["M1"]["timeouts"] == 1
    assert rpc["groups"]["M1"]["latency_p50_s"] == 20
    assert rpc["groups"]["M2"]["calls"] == 2 and rpc["groups"]["M2"]["errors"] == 1
    assert rpc["groups"]["M2"]["timeouts"] == 0 and rpc["groups"]["M2"]["latency_p50_s"] == 83
    assert rpc["timeout_alert_groups"] == ["M1"]
    assert "请求超时告警(M1)，结果可能受负载影响" in monitor.format_status(value)
    assert not rpc["window_may_be_incomplete"]
    assert "secret" not in json.dumps(value) and "context" not in json.dumps(rpc)


def test_rpc_bounded_tail_partial_window_and_strict_alert_threshold(tmp_path):
    planned = manifest(tmp_path)
    attempt = Path(planned["planned_jobs"][0]["output_dir"]) / "attempt_001"
    attempt.mkdir(parents=True)
    path = attempt / "mllm_metrics.jsonl"
    recent = {"role": "subgoal_selection", "timestamp": 990, "latency_s": 120, "error": "deadline exceeded"}
    path.write_text(json.dumps({"raw_text": "x" * 2000}) + "\n" + json.dumps(recent) + "\n" + '{"role":')
    value = monitor.rpc_window(planned["planned_jobs"], tmp_path, 1000, tail_bytes=256)
    assert value["groups"]["M2"]["timeouts"] == 1 and value["tail_truncated_files"] == 1
    assert value["partial_or_invalid_lines"] == 1 and value["window_may_be_incomplete"]
    rows = [{"role": "subgoal_selection", "timestamp": 990, "latency_s": 1, "error": "TimeoutError" if i == 0 else ""} for i in range(5)]
    rows += [{"role": "attribute_inference", "timestamp": 1001}, {"role": "attribute_inference"}, {"role": ["invalid"]}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value = monitor.rpc_window(planned["planned_jobs"], tmp_path, 1000)
    assert value["groups"]["M2"]["timeout_rate"] == 0.2 and not value["timeout_alert_groups"]
    assert value["future_timestamp_records"] == value["undated_records"] == 1
    assert value["groups"]["M1"]["calls"] == 0
