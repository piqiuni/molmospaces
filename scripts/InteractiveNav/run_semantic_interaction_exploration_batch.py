#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = (
    REPO_ROOT
    / "scripts"
    / "InteractiveNav"
    / "run_house7_semantic_exploration_ros_test.zsh"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-scene frontier and semantic-interaction exploration."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--house-inds", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-master-port", type=int, default=12420)
    parser.add_argument("--task-horizon", type=int, default=1000)
    parser.add_argument(
        "--method",
        choices=["container_exploration", "object_goal_runtime"],
        default="object_goal_runtime",
    )
    parser.add_argument("--scene-timeout-s", type=float, default=1500.0)
    parser.add_argument("--memory-sample-interval-s", type=float, default=2.0)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def terminate_group(process: subprocess.Popen[bytes], grace_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    for signum, wait_s in (
        (signal.SIGINT, grace_s),
        (signal.SIGTERM, 5.0),
        (signal.SIGKILL, 1.0),
    ):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def host_memory_usage() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        values[key] = float(raw_value.strip().split()[0]) / 1024.0
    return {
        "host_mem_used_mb": values["MemTotal"] - values["MemAvailable"],
        "host_mem_available_mb": values["MemAvailable"],
        "host_swap_used_mb": values["SwapTotal"] - values["SwapFree"],
    }


def process_group_usage(process_group_id: int) -> dict[str, float | int]:
    process_count = 0
    rss_mb = 0.0
    recorder_rss_mb = 0.0
    simulator_rss_mb = 0.0
    other_rss_mb = 0.0
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            remainder = stat_text[stat_text.rfind(")") + 2 :].split()
            parent_pid = int(remainder[1])
            status_lines = (entry / "status").read_text(encoding="utf-8").splitlines()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        processes[int(entry.name)] = (parent_pid, status_lines, command)

    descendants = {int(process_group_id)}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _status_lines, _command) in processes.items():
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True

    for pid in descendants:
        record = processes.get(pid)
        if record is None:
            continue
        _parent_pid, status_lines, command = record
        process_count += 1
        for line in status_lines:
            if line.startswith("VmRSS:"):
                process_rss_mb = float(line.split()[1]) / 1024.0
                rss_mb += process_rss_mb
                if "record_explore_debug.py" in command:
                    recorder_rss_mb += process_rss_mb
                elif "run_nav_ros_sim.py" in command:
                    simulator_rss_mb += process_rss_mb
                else:
                    other_rss_mb += process_rss_mb
                break
    return {
        "worker_process_count": process_count,
        "worker_rss_mb": rss_mb,
        "recorder_rss_mb": recorder_rss_mb,
        "simulator_rss_mb": simulator_rss_mb,
        "other_rss_mb": other_rss_mb,
    }


def current_sim_step(scene_dir: Path, previous_step: int) -> int:
    manifest = scene_dir / "sim_step_frames" / "manifest.jsonl"
    try:
        step_count = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
    except OSError:
        step_count = 0
    if step_count <= 0:
        frames_csv = scene_dir / "debug" / "video_frames.csv"
        try:
            step_count = max(0, sum(1 for _line in frames_csv.open(encoding="utf-8")) - 1)
        except OSError:
            step_count = 0
    return max(previous_step, step_count)


def monitor_scene_memory(
    stop_event: threading.Event,
    process_group_id: int,
    worker_id: int,
    house_ind: int,
    scene_dir: Path,
    interval_s: float,
    started: float,
) -> None:
    path = scene_dir / "memory_by_step.csv"
    fieldnames = [
        "wall_time",
        "elapsed_sec",
        "worker_id",
        "house_ind",
        "sim_step",
        "host_mem_used_mb",
        "host_mem_available_mb",
        "host_swap_used_mb",
        "worker_rss_mb",
        "recorder_rss_mb",
        "simulator_rss_mb",
        "other_rss_mb",
        "worker_process_count",
    ]
    last_step = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            last_step = current_sim_step(scene_dir, last_step)
            row = {
                "wall_time": time.time(),
                "elapsed_sec": time.monotonic() - started,
                "worker_id": worker_id,
                "house_ind": house_ind,
                "sim_step": last_step,
                **host_memory_usage(),
                **process_group_usage(process_group_id),
            }
            writer.writerow(row)
            handle.flush()
            if stop_event.wait(max(0.25, interval_s)):
                break


def summarize_scene_memory(path: Path) -> dict[str, float | int | None]:
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError:
        rows = []
    if not rows:
        return {"memory_samples": 0}

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) not in {None, ""}]

    worker_rss = values("worker_rss_mb")
    recorder_rss = values("recorder_rss_mb")
    simulator_rss = values("simulator_rss_mb")
    other_rss = values("other_rss_mb")
    host_used = values("host_mem_used_mb")
    host_available = values("host_mem_available_mb")
    swap_used = values("host_swap_used_mb")
    return {
        "memory_samples": len(rows),
        "memory_final_step": int(float(rows[-1].get("sim_step") or 0)),
        "worker_rss_start_mb": worker_rss[0] if worker_rss else None,
        "worker_rss_end_mb": worker_rss[-1] if worker_rss else None,
        "peak_worker_rss_mb": max(worker_rss) if worker_rss else None,
        "peak_recorder_rss_mb": max(recorder_rss) if recorder_rss else None,
        "peak_simulator_rss_mb": max(simulator_rss) if simulator_rss else None,
        "peak_other_rss_mb": max(other_rss) if other_rss else None,
        "peak_host_mem_used_mb": max(host_used) if host_used else None,
        "min_host_mem_available_mb": min(host_available) if host_available else None,
        "peak_host_swap_used_mb": max(swap_used) if swap_used else None,
    }


def run_scene(worker_id: int, house_ind: int, args: argparse.Namespace) -> dict[str, Any]:
    scene_id = f"house_{house_ind:04d}"
    scene_dir = args.output_dir / scene_id
    result_path = scene_dir / "semantic_exploration_result.json"
    if args.resume and result_path.exists():
        result = read_json(result_path)
        if result:
            result.update({"worker_id": worker_id, "resumed": True, "exit_code": 0})
            return result
    scene_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "METHOD": args.method,
            "HOUSE_IND": str(house_ind),
            "ROUTE_ID": scene_id,
            "USE_FIXED_ROUTE": "false",
            "SCENE_SEED": str(house_ind),
            "TASK_HORIZON": str(args.task_horizon),
            "SIM_TIMEOUT_S": str(int(args.scene_timeout_s)),
            "ROS_MASTER_URI": f"http://127.0.0.1:{args.base_master_port + worker_id}",
            "ROUTE_NAV_CONFIG": "",
            "INITIAL_DOOR_STATE": "closed",
            "FORCE_CLOSE_CONTAINERS": "true",
            "CLEAN_INTERMEDIATE": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = ["zsh", str(args.runner.resolve()), str(scene_dir), scene_id]
    if args.dry_run:
        return {
            "worker_id": worker_id,
            "house_ind": house_ind,
            "output_dir": str(scene_dir),
            "command": command,
            "ros_master_uri": environment["ROS_MASTER_URI"],
            "dry_run": True,
        }
    started = time.monotonic()
    with (scene_dir / "batch_task.log").open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitor_stop = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_scene_memory,
            args=(
                monitor_stop,
                process.pid,
                worker_id,
                house_ind,
                scene_dir,
                args.memory_sample_interval_s,
                started,
            ),
            daemon=True,
        )
        monitor_thread.start()
        try:
            exit_code = process.wait(timeout=args.scene_timeout_s + 360.0)
        except subprocess.TimeoutExpired:
            terminate_group(process)
            exit_code = 124
        finally:
            monitor_stop.set()
            monitor_thread.join(timeout=max(5.0, args.memory_sample_interval_s + 1.0))
    result = read_json(result_path)
    memory_summary = summarize_scene_memory(scene_dir / "memory_by_step.csv")
    step_timing = result.get("step_timing") or {}
    result.update(
        {
            "worker_id": worker_id,
            "house_ind": house_ind,
            "output_dir": str(scene_dir),
            "ros_master_uri": environment["ROS_MASTER_URI"],
            "exit_code": exit_code,
            "elapsed_sec": time.monotonic() - started,
            "resumed": False,
            "step_timing_loop_ms_avg": step_timing.get("loop_ms_avg"),
            "step_timing_policy_ms_avg": step_timing.get("policy_ms_avg"),
            "step_timing_task_ms_avg": step_timing.get("task_ms_avg"),
            **memory_summary,
        }
    )
    (scene_dir / "batch_task_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_worker(
    worker_id: int, houses: list[int], args: argparse.Namespace
) -> list[dict[str, Any]]:
    return [run_scene(worker_id, house_ind, args) for house_ind in houses]


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    results = sorted(results, key=lambda row: int(row.get("house_ind", -1)))
    (output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_results = [
        row
        for row in results
        if (row.get("target_selection") or {}).get("target_context", {}).get("enabled")
    ]

    def mean_numeric(key: str) -> float | None:
        values = [float(row[key]) for row in results if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    aggregate = {
        "scene_count": len(results),
        "completed_scene_count": sum(bool(row.get("overall_success")) for row in results),
        "overall_success_rate": (
            sum(bool(row.get("overall_success")) for row in results) / len(results)
            if results else 0.0
        ),
        "target_selected_count": len(selected_results),
        "target_container_interaction_success_rate": (
            sum(bool(row.get("target_container_interaction_success")) for row in selected_results)
            / len(selected_results)
            if selected_results else 0.0
        ),
        "target_object_visible_navigation_success_rate": (
            sum(bool(row.get("target_object_visible_navigation_success")) for row in selected_results)
            / len(selected_results)
            if selected_results else 0.0
        ),
        "target_goal_success_rate": (
            sum(bool(row.get("target_goal_success")) for row in selected_results)
            / len(selected_results)
            if selected_results else 0.0
        ),
        "mean_coverage_ratio": mean_numeric("coverage_ratio"),
        "mean_mapped_free_coverage_ratio": mean_numeric("mapped_free_coverage_ratio"),
        "mean_elapsed_sec": mean_numeric("elapsed_sec"),
        "mean_offline_video_elapsed_sec": mean_numeric("offline_video_elapsed_sec"),
        "mean_offline_analysis_elapsed_sec": mean_numeric("offline_analysis_elapsed_sec"),
        "mean_loop_ms": mean_numeric("step_timing_loop_ms_avg"),
        "peak_worker_rss_mb": max(
            (float(row["peak_worker_rss_mb"]) for row in results if row.get("peak_worker_rss_mb") is not None),
            default=None,
        ),
        "peak_simulator_rss_mb": max(
            (float(row["peak_simulator_rss_mb"]) for row in results if row.get("peak_simulator_rss_mb") is not None),
            default=None,
        ),
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "house_ind",
        "worker_id",
        "exit_code",
        "elapsed_sec",
        "completed_early",
        "completion_reason",
        "sim_step_frames",
        "video_frames",
        "coverage_ratio",
        "mapped_free_coverage_ratio",
        "decision_count",
        "successful_behavior_count",
        "interaction_count",
        "contains_edge_count",
        "container_with_children_count",
        "target_goal_success",
        "overall_success",
        "target_container_interaction_success",
        "target_object_visible_navigation_success",
        "offline_video_elapsed_sec",
        "offline_analysis_elapsed_sec",
        "step_timing_loop_ms_avg",
        "step_timing_policy_ms_avg",
        "step_timing_task_ms_avg",
        "valid_step_video",
        "memory_samples",
        "memory_final_step",
        "worker_rss_start_mb",
        "worker_rss_end_mb",
        "peak_worker_rss_mb",
        "peak_recorder_rss_mb",
        "peak_simulator_rss_mb",
        "peak_other_rss_mb",
        "peak_host_mem_used_mb",
        "min_host_mem_available_mb",
        "peak_host_swap_used_mb",
        "video",
        "output_dir",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.runner = args.runner.expanduser().resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not args.runner.exists():
        raise FileNotFoundError(args.runner)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = [args.house_inds[index :: args.workers] for index in range(args.workers)]
    if args.dry_run:
        rows = [
            run_scene(worker_id, house_ind, args)
            for worker_id, houses in enumerate(groups)
            for house_ind in houses
        ]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_worker, worker_id, houses, args): worker_id
            for worker_id, houses in enumerate(groups)
            if houses
        }
        for future in as_completed(futures):
            results.extend(future.result())
            write_summary(args.output_dir, results)
    write_summary(args.output_dir, results)
    failures = [result for result in results if int(result.get("exit_code", 1)) != 0]
    print(json.dumps({"results": results, "failure_count": len(failures)}, ensure_ascii=False, indent=2))
    return 0 if args.allow_failures or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
