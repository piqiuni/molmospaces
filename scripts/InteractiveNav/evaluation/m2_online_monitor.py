#!/usr/bin/env python3
"""Read-only lane/episode aggregation; writes only overall monitor artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "interactive_nav_m2_online_monitor_v1"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _restart_epoch(run_dir: Path) -> dict[str, Any]:
    ledger = read_json(run_dir / "restart_epoch.json")
    epoch_id = ledger.get("epoch_id")
    if not isinstance(epoch_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", epoch_id):
        return {}
    lanes = ledger.get("affected_lanes")
    return {"epoch_id": epoch_id,
            "affected_lanes": [lane for lane in lanes if type(lane) is int and lane >= 0] if isinstance(lanes, list) else [],
            "remote_preserved": ledger.get("remote_preserved") if isinstance(ledger.get("remote_preserved"), bool) else None,
            "archive_root": ledger.get("archive_root") if isinstance(ledger.get("archive_root"), str) else None,
            "archived_results_included": False}


def _restart_epoch_line(status: dict[str, Any]) -> str:
    epoch = status.get("restart_epoch") or {}
    if not epoch:
        return ""
    lanes = ",".join(str(lane) for lane in epoch["affected_lanes"]) or "未知"
    remote = "true" if epoch["remote_preserved"] is True else "false" if epoch["remote_preserved"] is False else "unknown"
    return (f"  本地重跑 epoch={epoch['epoch_id']} | affected_lanes=[{lanes}] | "
            f"旧结果已归档、不计入本轮 | remote_preserved={remote}")


def _timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
        except ValueError:
            return None
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _cloud_state(document: dict[str, Any]) -> str:
    raw = str(document.get("state") or document.get("status") or "").strip().casefold()
    return {"queue": "queued", "queued": "queued", "pending": "queued", "running": "running",
            "failed": "failed", "error": "failed", "completed": "completed", "succeeded": "completed",
            "stopped": "stopped", "cancelled": "stopped", "canceled": "stopped"}.get(raw, "unknown")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cloud_record(stdout: str, task_id: str) -> dict[str, Any]:
    """CLI may prepend an upgrade banner; accept only the requested task."""
    decoder = json.JSONDecoder()
    for offset, character in enumerate(stdout):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[offset:])
        except ValueError:
            continue
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if isinstance(row, dict) and str(row.get("Id")) == task_id and isinstance(row.get("Status"), str):
                state = _cloud_state({"status": row["Status"]})
                if state != "unknown":
                    return {"task_id": task_id, "state": state, "exit_code": _number(row.get("ExitCode"))}
    raise ValueError("No recognized task status")


class CloudPoller:
    """Optional, read-only platform polling; never persist raw CLI output."""

    def __init__(self, task_id: str, lane_id: int, *, interval_s: float = 60.0):
        self.task_id, self.lane_id, self.interval_s = task_id, lane_id, interval_s
        self.next_poll = 0.0
        self.terminal = False
        self.error_count = 0

    def poll(self, run_dir: Path, *, now: float | None = None, monotonic: float | None = None) -> bool:
        now = time.time() if now is None else now
        monotonic = time.monotonic() if monotonic is None else monotonic
        if self.terminal or monotonic < self.next_poll or not run_dir.is_dir():
            return False
        self.next_poll = monotonic + self.interval_s
        stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
        previous = read_json(run_dir / "task_status.json")
        document = {key: previous[key] for key in ("state", "exit_code", "updated_at")
                    if previous.get("task_id") == self.task_id and key in previous}
        document.update(task_id=self.task_id, lane_id=self.lane_id, checked_at=stamp)
        try:
            response = subprocess.run(
                ["volc", "ml_task", "get", "-i", self.task_id, "--output", "json", "--format", "Id,Name,Status,ExitCode"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            if response.returncode:
                raise ChildProcessError("Cloud query exited unsuccessfully")
            document.update(_cloud_record(response.stdout, self.task_id), updated_at=stamp)
            self.terminal = document["state"] in {"completed", "failed", "stopped"}
        except (OSError, ValueError, subprocess.TimeoutExpired, ChildProcessError) as error:
            self.error_count += 1
            document["error_type"] = type(error).__name__
        document["error_count"] = self.error_count
        _atomic_json(run_dir / "task_status.json", document)
        return True


def _safe_lane_error(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    if text.startswith("consecutive_incomplete_threshold"):
        return "连续不完整达到阈值，保守暂停出队；非基础设施定性，未重试"
    model_exit = re.match(r"Qwen [^\n]+ exited \((-?\d+)\);", text)
    if model_exit:
        return f"模型服务退出(code={model_exit.group(1)})，未重启"
    exception = re.match(r"([A-Za-z][A-Za-z0-9]*(?:Error|Exception)):", text)
    return f"{exception.group(1)}（详情见lane_status.json）" if exception else "lane异常（详情见lane_status.json）"


def _lane_status(lane: dict[str, Any], root: Path, experiment_id: Any, now: float, stale_after: float,
                 cloud_document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _path(lane["output_dir"], root) / "lane_status.json"
    raw = read_json(path)
    identity_mismatch = bool(raw and (
        str(raw.get("lane_id", lane["id"])) != str(lane["id"])
        or raw.get("experiment_id", experiment_id) != experiment_id
    ))
    if identity_mismatch:
        raw = {}
    stamp = _timestamp(raw.get("updated_at"))
    age = max(0.0, now - stamp) if stamp is not None else None
    state = str(raw.get("state") or "not_started").casefold()
    if state not in {"pending", "starting", "starting_qwen", "running", "completed", "failed", "stopped", "not_started"}:
        state = "unknown"
    terminal = state in {"completed", "failed", "stopped"}
    stale = bool(raw and not terminal and (age is None or age > stale_after))
    cloud = cloud_document if str(cloud_document.get("lane_id")) == str(lane["id"]) else {}
    cloud_stamp = _timestamp(cloud.get("updated_at"))
    cloud_age = max(0.0, now - cloud_stamp) if cloud_stamp is not None else None
    display = {
        "lane_id": lane["id"], "state": state, "status_path": str(path),
        "heartbeat_age_s": age, "heartbeat_stale": stale, "identity_mismatch": identity_mismatch,
        "cloud_state": _cloud_state(cloud) if cloud else None,
        "cloud_status_age_s": cloud_age,
        "cloud_status_stale": bool(cloud and _cloud_state(cloud) not in {"failed", "completed", "stopped"}
                                   and (cloud_age is None or cloud_age > stale_after)),
        "cloud_poll_error_count": cloud.get("error_count", 0),
        "cloud_poll_error_type": cloud.get("error_type"),
        "cloud_task_id": cloud.get("task_id"),
        "error_summary": _safe_lane_error(raw.get("error")),
        "failure_classifications": dict(Counter(
            row.get("classification") if row.get("classification") in {"confirmed_infrastructure", "incomplete_unclassified"} else "unclassified"
            for row in raw.get("failure_queue") or [] if isinstance(row, dict))),
    }
    return display, raw


def _tail(path: Path, limit: int = 131072) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            start = max(0, stream.tell() - limit)
            stream.seek(start)
            value = stream.read(limit).decode("utf-8", errors="replace")
            return value.split("\n", 1)[-1] if start else value
    except OSError:
        return ""


def _latest_attempt(task_dir: Path) -> Path | None:
    try:
        attempts = [path for path in task_dir.iterdir() if re.fullmatch(r"attempt_\d+", path.name) and path.is_dir()]
    except OSError:
        return None
    return max(attempts, key=lambda path: int(path.name.split("_")[-1])) if attempts else None


def rpc_window(jobs: list[dict[str, Any]], root: Path, now: float, *, window_s: float = 120.0,
               tail_bytes: int = 262144) -> dict[str, Any]:
    """Aggregate completed RPC records, never exposing prompts, outputs or errors."""
    role_groups = {"attribute_inference": "M1", "room_attribute_inference": "M1", "subgoal_selection": "M2"}
    grouped = {name: {"calls": 0, "errors": 0, "timeouts": 0, "latencies": []} for name in ("M1", "M2")}
    roles = Counter()
    files = truncated = invalid = undated = read_errors = future = 0
    for job in jobs:
        attempt = _latest_attempt(_path(job["output_dir"], root))
        if attempt is None:
            continue
        path = attempt / "mllm_metrics.jsonl"
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                start = max(0, stream.tell() - tail_bytes)
                stream.seek(start)
                text = stream.read(tail_bytes).decode("utf-8", errors="replace")
                if start:
                    text = text.split("\n", 1)[-1] if "\n" in text else ""
                    truncated += 1
            files += 1
        except FileNotFoundError:
            continue
        except OSError:
            read_errors += 1
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                invalid += 1
                continue
            if not isinstance(row, dict) or not isinstance(row.get("role"), str) or row["role"] not in role_groups:
                continue
            stamp = _timestamp(row.get("timestamp"))
            if stamp is None:
                undated += 1
                continue
            if stamp > now:
                future += 1
                continue
            if stamp < now - window_s:
                continue
            group = grouped[role_groups[row["role"]]]
            group["calls"] += 1
            roles[row["role"]] += 1
            error = row.get("error")
            timeout = row.get("timed_out") is True or row.get("is_timeout") is True
            if isinstance(error, str):
                timeout = timeout or bool(re.search(r"timeout|timed[\s_-]*out|deadline[^\n]*exceeded|超时", error, re.IGNORECASE))
            group["errors"] += bool(error) or timeout
            group["timeouts"] += timeout
            latency = _number(row.get("latency_s"))
            if latency is not None and latency >= 0:
                group["latencies"].append(latency)
    for group in grouped.values():
        latencies = group.pop("latencies")
        group.update(latency_p50_s=statistics.median(latencies) if latencies else None,
                     latency_samples=len(latencies),
                     error_rate=group["errors"] / group["calls"] if group["calls"] else None,
                     timeout_rate=group["timeouts"] / group["calls"] if group["calls"] else None)
    alert_groups = [name for name, values in grouped.items() if values["timeout_rate"] is not None and values["timeout_rate"] > 0.2]
    return {"window_s": window_s, "timestamp_basis": "completed RPC metric timestamp; in-flight requests are not counted",
            "groups": grouped, "calls_by_role": dict(roles), "timeout_alert_groups": alert_groups,
            "metrics_files_read": files, "tail_bytes_per_file": tail_bytes,
            "tail_truncated_files": truncated, "partial_or_invalid_lines": invalid,
            "undated_records": undated, "future_timestamp_records": future, "read_errors": read_errors,
            "window_may_be_incomplete": bool(truncated or invalid or undated or future or read_errors),
            "scope": "Latest attempt of each manifest job. A bounded tail may omit records; absence is not proof of no in-flight requests."}


def _runtime_evidence(job: dict[str, Any], active: dict[str, Any], root: Path) -> dict[str, Any]:
    """Bounded tails only; a worker heartbeat is not a simulator observation."""
    task_dir = _path(job["output_dir"], root)
    attempt = _latest_attempt(task_dir)
    progress = active.get("progress") if isinstance(active.get("progress"), dict) else {}
    observation_count = _number(progress.get("observation_step_count"))
    actions = _number(progress.get("applied_action_step_count"))
    indices, slam_indices, laser_poses = [], [], []
    if attempt is not None:
        for name in ("eval.log", "roslaunch.log"):
            for line in _tail(attempt / name).splitlines():
                # capture_step is observation-envelope metadata, not applied actions.
                match = re.search(r"SemanticMappingCausalReady .*\bcapture_step=(\d+)\b", line)
                if match is None:
                    match = re.search(r"RosBridgePolicy: step-ready bootstrap complete at step=(\d+)\b", line)
                if match:
                    indices.append(int(match.group(1)))
                frame = re.fullmatch(r"\s*update frame (\d+)\s*", line)
                if frame:
                    slam_indices.append(int(frame.group(1)))
                pose = re.fullmatch(r"\s*Laser Pose=\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*", line)
                if pose:
                    try:
                        coordinates = tuple(float(value) for value in pose.groups())
                        if all(math.isfinite(value) for value in coordinates):
                            laser_poses.append(coordinates)
                    except ValueError:
                        pass
    return {"attempt": str(attempt) if attempt else None,
            "observation_step_index": max(indices) if indices else None,
            "slam_frame_index": max(slam_indices) if slam_indices else None,
            "laser_pose": laser_poses[-1] if laser_poses else None,
            "pose_changed_in_tail": len(set(laser_poses)) > 1,
            "observation_step_count": observation_count if observation_count is not None and observation_count >= 0 else None,
            "applied_action_step_count": actions if actions is not None and actions >= 0 else None}


class ProgressTracker:
    def __init__(self):
        self.previous: dict[str, dict[str, Any]] = {}
        self.last_advance: float | None = None
        self.scans = 0

    def aggregate(self, rows: list[tuple[str, dict[str, Any]]], now: float) -> dict[str, Any]:
        observed = advanced = counter_workers = slam_workers = pose_changed = 0
        action_sum = 0.0
        for job_id, row in rows:
            prior = self.previous.get(job_id)
            observed += bool((row["observation_step_index"] is not None)
                             or (row["observation_step_count"] or 0) > 0
                             or row["slam_frame_index"] is not None or row["laser_pose"] is not None)
            slam_workers += row["slam_frame_index"] is not None
            pose_changed += row["pose_changed_in_tail"]
            if row["applied_action_step_count"] is not None:
                counter_workers += 1
                action_sum += row["applied_action_step_count"]
            same_attempt = prior is not None and prior["attempt"] == row["attempt"]
            counters = ("observation_step_index", "observation_step_count", "applied_action_step_count", "slam_frame_index")
            if same_attempt and (any(row[key] is not None and prior[key] is not None and row[key] > prior[key] for key in counters)
                                 or (row["laser_pose"] is not None and prior["laser_pose"] is not None
                                     and row["laser_pose"] != prior["laser_pose"])):
                advanced += 1
                self.last_advance = now
            elif prior is not None and not same_attempt:
                # A new attempt resets indices; never report this as applied progress.
                pass
            elif prior is not None and any(prior[key] is None and row[key] is not None
                                           for key in (*counters, "laser_pose")):
                advanced += 1
                self.last_advance = now
            self.previous[job_id] = row
        self.scans += 1
        return {"running_workers": len(rows), "workers_with_observation_evidence": observed,
                "workers_without_observation_evidence": len(rows) - observed,
                "workers_with_slam_frames": slam_workers, "workers_with_slam_pose_change_in_tail": pose_changed,
                "workers_advanced_since_previous_scan": advanced if self.scans > 1 else None,
                "applied_action_counter_workers": counter_workers,
                "applied_action_sum_known_workers": action_sum if counter_workers else None,
                "last_observed_advance_at": datetime.fromtimestamp(self.last_advance, timezone.utc).isoformat() if self.last_advance else None,
                "evidence_scope": "Latest attempt, at most 128 KiB tail per eval.log and roslaunch.log. Missing evidence is unknown, not zero. Observation indices, SLAM frames and estimated pose updates are not applied action counts."}


def _episode_summary(job: dict[str, Any], root: Path, experiment_id: Any) -> tuple[dict[str, Any], str | None]:
    task_dir = _path(job["output_dir"], root)
    summary = read_json(task_dir / "batch_task_summary.json")
    if not summary:
        return {}, None
    if (summary.get("episode_index") not in (None, job["episode_index"])
        or summary.get("experiment_id", experiment_id) != experiment_id):
        return {}, "episode_summary_identity_mismatch"
    # Canonical per-episode result wins over flattened summaries; never read an
    # aggregate success_rate, which has a different (NavSR) meaning.
    result_path = summary.get("episode_result_path")
    canonical = read_json(_path(result_path, task_dir)) if isinstance(result_path, str) and result_path else {}
    result = canonical.get("result", canonical)
    if isinstance(result, dict) and result and canonical.get("status") == "complete":
        if result.get("episode_index") not in (None, job["episode_index"]):
            return {}, "episode_result_identity_mismatch"
        for key in ("success", "nav_success", "spl", "applied_action_step_count", "step_count", "elapsed_seconds", "scoring_eligible"):
            if key in result:
                summary[key] = result[key]
    return summary, None


def _mean(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [number for row in rows if (number := _number(row.get(key))) is not None]
    return {"mean": sum(values) / len(values) if values else None, "n": len(values)}


def _rate(rows: list[dict[str, Any]], key: str, planned: int) -> dict[str, Any]:
    values = [row[key] for row in rows if isinstance(row.get(key), bool)]
    successes = sum(values)
    return {"successes": successes, "completed_denominator": len(rows), "valid_metric_denominator": len(values),
            "missing_completed_metrics": len(rows) - len(values),
            "completed_rate": successes / len(rows) if rows and len(values) == len(rows) else None,
            "valid_metric_rate": successes / len(values) if values else None,
            "planned_denominator": planned, "planned_rate_lower_bound": successes / planned if planned else None}


def _metrics(rows: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get("scoring_eligible") is not False]
    return {"ics": _rate(valid, "success", planned), "nav_sr": _rate(valid, "nav_success", planned),
            "spl": _mean(valid, "spl"), "applied_steps": _mean(valid, "applied_action_step_count"),
            "observation_steps": _mean(valid, "step_count"), "runner_seconds": _mean(valid, "elapsed_sec"),
            "evaluator_seconds": _mean(valid, "elapsed_seconds"),
            "total_completed": len(rows), "valid_completed": len(valid),
            "eligible_completed": sum(row.get("scoring_eligible") is True for row in rows),
            "unknown_eligibility_completed": sum(not isinstance(row.get("scoring_eligible"), bool) for row in rows),
            "ineligible_completed": sum(row.get("scoring_eligible") is False for row in rows)}


def snapshot(run_dir: Path, *, now: float | None = None, stale_after: float = 120.0,
             progress_tracker: ProgressTracker | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    manifest = read_json(run_dir / "manifest.json")
    output: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                              "experiment_id": manifest.get("experiment_id"), "state": "waiting_manifest", "finished": False}
    restart_epoch = _restart_epoch(run_dir)
    if restart_epoch:
        output["restart_epoch"] = restart_epoch
    if not manifest:
        return output
    jobs = manifest.get("planned_jobs") or []
    lanes = manifest.get("lanes") or []
    ids = [job.get("job_id") for job in jobs]
    if not jobs or any(not isinstance(jid, str) or not jid for jid in ids) or len(ids) != len(set(ids)):
        output.update(state="invalid_manifest", warning="planned_jobs missing/empty/duplicate IDs")
        return output
    cloud = read_json(run_dir / "task_status.json")
    lane_rows, lane_raw = {}, {}
    for lane in lanes:
        display, raw = _lane_status(lane, run_dir, manifest.get("experiment_id"), now, stale_after, cloud)
        lane_rows[str(lane["id"])], lane_raw[str(lane["id"])] = display, raw
    active = {str(item.get("job_id")): item for raw in lane_raw.values() for item in raw.get("active") or [] if isinstance(item, dict)}
    failed = {str(item.get("job_id")) for raw in lane_raw.values() for item in raw.get("failure_queue") or [] if isinstance(item, dict)}
    arms: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    complete_rows, arm_completed = [], {}
    warnings: Counter[str] = Counter()
    running_progress = []
    for job in jobs:
        arm = str(job["arm"])
        arm_state = arms.setdefault(arm, {"planned": 0, "completed": 0, "running": 0, "queued": 0, "infra_or_incomplete": 0})
        arm_completed.setdefault(arm, [])
        arm_state["planned"] += 1
        summary, warning = _episode_summary(job, run_dir, manifest.get("experiment_id"))
        if warning:
            warnings[warning] += 1
        lane = lane_rows.get(str(job["lane_id"]), {})
        if summary.get("completed") is True:
            state = "completed"
            complete_rows.append(summary)
            arm_completed[arm].append(summary)
        elif summary.get("completed") is False or job["job_id"] in failed:
            state = "infra_or_incomplete"
        elif job["job_id"] in active:
            state = "running"
            running_progress.append((job["job_id"], _runtime_evidence(job, active[job["job_id"]], run_dir)))
            if lane.get("heartbeat_stale"):
                counts["running_with_stale_heartbeat"] += 1
        else:
            state = "queued"
            if lane.get("state") in {"failed", "stopped"} or lane.get("cloud_state") in {"failed", "stopped"}:
                counts["queued_blocked_by_lane"] += 1
        arm_state[state] += 1
        counts[state] += 1
    for arm, values in arms.items():
        values["metrics"] = _metrics(arm_completed[arm], values["planned"])
    finished = counts["completed"] + counts["infra_or_incomplete"] == len(jobs)
    output.update(state="completed" if finished and not counts["infra_or_incomplete"] else "finished_with_incomplete" if finished else "running",
                  finished=finished, planned=len(jobs), counts={key: counts[key] for key in (
                      "completed", "running", "queued", "infra_or_incomplete", "running_with_stale_heartbeat", "queued_blocked_by_lane")},
                  lanes=list(lane_rows.values()), by_arm=arms, metrics=_metrics(complete_rows, len(jobs)),
                  runtime_progress=(progress_tracker or ProgressTracker()).aggregate(running_progress, now),
                  rpc_window=rpc_window(jobs, run_dir, now),
                  data_warnings=dict(warnings),
                  metric_notes=["ICS uses per-episode success, never aggregate success_rate.",
                                "Metric completed denominators use scoring_eligible != false only, including valid algorithm failures. Ineligible completions remain in completion/planned counts, not score or mean denominators.",
                                "Missing scoring_eligible remains included for older summaries and is counted separately as unknown eligibility.",
                                "Planned denominator includes queued/running/infra/missing jobs; progress rates are provisional lower bounds.",
                                "infra_or_incomplete is a wrapper-completion class, not a proven infrastructure root cause.",
                                "Applied action steps and observation steps are separate; missing metrics are never silently zero."])
    return output


def _ratio(rate: dict[str, Any], *, planned: bool = False) -> str:
    denominator = rate["planned_denominator"] if planned else rate["completed_denominator"]
    value = rate["planned_rate_lower_bound"] if planned else rate["completed_rate"]
    if not planned and rate["missing_completed_metrics"]:
        return f"N/A(有效n={denominator},缺{rate['missing_completed_metrics']})"
    return f"{rate['successes']}/{denominator}({value:.1%})" if value is not None else f"N/A(n={denominator})"


def _average(value: dict[str, Any], precision: int = 1) -> str:
    return f"{value['mean']:.{precision}f}(n={value['n']})" if value["mean"] is not None else "N/A(n=0)"


def format_status(status: dict[str, Any], timezone_name: str = "Asia/Hong_Kong") -> str:
    stamp = datetime.fromisoformat(status["updated_at"]).astimezone(ZoneInfo(timezone_name)).isoformat(timespec="seconds")
    restart_line = _restart_epoch_line(status)
    if "counts" not in status:
        message = f"[{stamp}] 总体监控：{status['state']}（等待manifest/任务启动；尚无可聚合结果）"
        return message + ("\n" + restart_line if restart_line else "")
    count = status["counts"]
    lines = [f"[{stamp}] 总体 完成 {count['completed']}/{status['planned']} | 运行 {count['running']} | 排队 {count['queued']} | infra/不完整 {count['infra_or_incomplete']} | 状态 {status['state']}"]
    if restart_line:
        lines.append(restart_line)
    if count["running_with_stale_heartbeat"] or count["queued_blocked_by_lane"]:
        lines.append(f"  告警：运行中heartbeat过期 {count['running_with_stale_heartbeat']}；排队中lane阻塞 {count['queued_blocked_by_lane']}")
    for lane in status["lanes"]:
        label = "尚未启动" if lane["state"] == "not_started" else lane["state"]
        heartbeat = " | heartbeat过期" if lane["heartbeat_stale"] else ""
        identity = " | 身份不符" if lane["identity_mismatch"] else ""
        cloud = f" | 云端{lane['cloud_state']}" if lane["cloud_state"] else ""
        if lane["cloud_state"] in {"failed", "stopped"} and lane["cloud_task_id"]:
            cloud += f"(task={lane['cloud_task_id']})"
        if lane["cloud_status_stale"]:
            cloud += "(状态过期)"
        if lane["cloud_poll_error_type"]:
            cloud += f" | 云查询错误{lane['cloud_poll_error_type']}(累计{lane['cloud_poll_error_count']})"
        lines.append(f"  lane{lane['lane_id']}: {label}{heartbeat}{identity}{cloud}")
        if lane["error_summary"] or lane["failure_classifications"]:
            classified = ",".join(f"{name}={number}" for name, number in lane["failure_classifications"].items())
            lines.append(f"    原因：{lane['error_summary'] or '存在不完整结果'}；分类：{classified or '无'}；详情 {lane['status_path']}")
    progress = status["runtime_progress"]
    advanced = progress["workers_advanced_since_previous_scan"]
    action_sum = progress["applied_action_sum_known_workers"]
    lines.append(f"  仿真证据：有观测/SLAM {progress['workers_with_observation_evidence']}/{progress['running_workers']}运行worker（SLAM帧{progress['workers_with_slam_frames']}，日志尾位姿变化{progress['workers_with_slam_pose_change_in_tail']}）；无证据(未知) {progress['workers_without_observation_evidence']}；本轮采样推进 {advanced if advanced is not None else 'N/A(首采样)'}；动作累计 {action_sum if action_sum is not None else '未知'}(有计数worker={progress['applied_action_counter_workers']})；最近发现推进 {progress['last_observed_advance_at'] or '尚无跨采样证据'}")
    rpc = status["rpc_window"]
    rpc_parts = []
    for name, group in rpc["groups"].items():
        median = f"{group['latency_p50_s']:.1f}s" if group["latency_p50_s"] is not None else "N/A"
        timeout_rate = f"{group['timeout_rate']:.1%}" if group["timeout_rate"] is not None else "N/A"
        rpc_parts.append(f"{name} 调用{group['calls']}/错误{group['errors']}/超时{group['timeouts']}({timeout_rate})/耗时p50={median}")
    completeness = (f"；窗口可能不完整(裁剪文件{rpc['tail_truncated_files']},坏/半行{rpc['partial_or_invalid_lines']},无时戳{rpc['undated_records']},未来时戳{rpc['future_timestamp_records']},读失败{rpc['read_errors']})"
                    if rpc["window_may_be_incomplete"] else "")
    alert = f"；请求超时告警({','.join(rpc['timeout_alert_groups'])})，结果可能受负载影响" if rpc["timeout_alert_groups"] else ""
    lines.append(f"  RPC近{rpc['window_s']:.0f}s已记录完成请求：" + " | ".join(rpc_parts) + completeness + alert)
    overall = status["metrics"]
    lines.append(f"  有效评测 {overall['valid_completed']}/{overall['total_completed']}已完成；ineligible={overall['ineligible_completed']}（eligibility未知={overall['unknown_eligibility_completed']}）；总ICS：有效完成分母 {_ratio(overall['ics'])}；计划分母 {_ratio(overall['ics'], planned=True)}（临时下界，保留未完成/无效任务）")
    for arm, row in sorted(status["by_arm"].items()):
        metrics = row["metrics"]
        lines.append(f"  {arm} 完成{row['completed']}/{row['planned']} 有效{metrics['valid_completed']}/{row['completed']} ineligible={metrics['ineligible_completed']} | ICS(有效)={_ratio(metrics['ics'])} | NavSR(有效)={_ratio(metrics['nav_sr'])} | ICS(计划)={_ratio(metrics['ics'], planned=True)} | SPL={_average(metrics['spl'], 3)} | 动作步={_average(metrics['applied_steps'])} | 观测步={_average(metrics['observation_steps'])} | runner耗时秒={_average(metrics['runner_seconds'])}")
    if status["data_warnings"]:
        lines.append("  数据告警：" + ", ".join(f"{key}={value}" for key, value in status["data_warnings"].items()))
    return "\n".join(lines)


def publish(run_dir: Path, status: dict[str, Any], *, timezone_name: str = "Asia/Hong_Kong") -> str:
    text = format_status(status, timezone_name)
    _atomic_json(run_dir / "overall_status.json", status)
    with (run_dir / "overall.log").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--stale-after-s", type=float, default=120.0)
    parser.add_argument("--timezone", default="Asia/Hong_Kong")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cloud-task-id", help="Optional existing task; read-only volc query every 60 seconds")
    parser.add_argument("--cloud-lane-id", type=int, default=2)
    args = parser.parse_args()
    if args.interval_s <= 0 or args.stale_after_s <= 0:
        parser.error("interval and stale threshold must be positive")
    run_dir = args.run_dir.resolve()
    cloud_poller = CloudPoller(args.cloud_task_id, args.cloud_lane_id) if args.cloud_task_id else None
    progress_tracker = ProgressTracker()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    lock = None
    try:
        while not stop.is_set():
            if run_dir.is_dir() and lock is None:
                lock = (run_dir / ".overall_monitor.lock").open("a")
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print("Another overall monitor already holds this run directory", file=sys.stderr)
                    return 2
            if cloud_poller:
                cloud_poller.poll(run_dir)
            status = snapshot(run_dir, stale_after=args.stale_after_s, progress_tracker=progress_tracker)
            message = publish(run_dir, status, timezone_name=args.timezone) if run_dir.is_dir() else format_status(status, args.timezone)
            print(message, flush=True)
            if args.once or status["finished"]:
                return 1 if status["state"] == "finished_with_incomplete" else 0
            stop.wait(args.interval_s)
    finally:
        if lock is not None:
            lock.close()
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
