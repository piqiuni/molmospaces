#!/usr/bin/env python3
"""Run comparable House 7 exploration methods with at most two ROS workers."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_CONFIG = (
    REPO_ROOT
    / "scripts"
    / "InteractiveNav"
    / "configs"
    / "semantic_decision"
    / "house7_force_routes.yaml"
)
DEFAULT_RUNNER = REPO_ROOT / "scripts" / "InteractiveNav" / "run_house7_semantic_exploration_ros_test.zsh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE_CONFIG)
    parser.add_argument("--route-ids", nargs="*", default=[])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "frontier_only",
            "interactive_rule",
            "object_goal_rule",
            "object_goal_model_mock",
        ),
        default=["frontier_only", "interactive_rule"],
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-master-port", type=int, default=11520)
    parser.add_argument("--task-horizon", type=int, default=1000)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--route-timeout-s", type=float, default=1500.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_routes(path: Path, selected: list[str]) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    available = [str(route["route_id"]) for route in payload.get("routes") or []]
    if not selected:
        return available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown route ids: {unknown}")
    selected_set = set(selected)
    return [route_id for route_id in available if route_id in selected_set]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def terminate_group(process: subprocess.Popen[bytes], grace_s: float = 20.0) -> None:
    if process.poll() is not None:
        return
    for signum, wait_s in ((signal.SIGINT, grace_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def run_task(
    worker_id: int,
    method: str,
    route_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_dir = args.output_dir / method / route_id
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / "semantic_exploration_result.json"
    if args.resume and result_path.exists():
        result = read_json(result_path)
        if result:
            result.update({"worker_id": worker_id, "resumed": True, "exit_code": 0})
            return result

    environment = os.environ.copy()
    environment.update(
        {
            "METHOD": method,
            "ROUTE_ID": route_id,
            "ROUTE_CONFIG": str(args.route_config.resolve()),
            "TASK_HORIZON": str(args.task_horizon),
            "ROS_MASTER_URI": f"http://127.0.0.1:{args.base_master_port + worker_id}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = ["zsh", str(args.runner.resolve()), str(task_dir), route_id]
    if args.dry_run:
        return {
            "worker_id": worker_id,
            "method": method,
            "route_id": route_id,
            "output_dir": str(task_dir),
            "command": command,
            "ros_master_uri": environment["ROS_MASTER_URI"],
            "dry_run": True,
        }

    started = time.monotonic()
    log_path = task_dir / "batch_task.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=args.route_timeout_s)
        except subprocess.TimeoutExpired:
            terminate_group(process)
            exit_code = 124

    result = read_json(result_path)
    result.update(
        {
            "worker_id": worker_id,
            "method": method,
            "route_id": route_id,
            "output_dir": str(task_dir),
            "ros_master_uri": environment["ROS_MASTER_URI"],
            "exit_code": exit_code,
            "elapsed_sec": time.monotonic() - started,
            "resumed": False,
        }
    )
    (task_dir / "batch_task_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_worker_group(
    worker_id: int,
    tasks: list[tuple[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return [run_task(worker_id, method, route_id, args) for method, route_id in tasks]


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    results = sorted(results, key=lambda row: (str(row.get("method")), str(row.get("route_id"))))
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "method",
        "route_id",
        "worker_id",
        "exit_code",
        "elapsed_sec",
        "sim_step_frames",
        "video_frames",
        "coverage_ratio",
        "mapped_free_coverage_ratio",
        "decision_count",
        "successful_behavior_count",
        "interaction_count",
        "target_goal_success",
        "valid_step_video",
        "output_dir",
    ]
    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.route_config = args.route_config.expanduser().resolve()
    args.runner = args.runner.expanduser().resolve()
    if args.workers < 1 or args.workers > 2:
        raise ValueError("--workers must be in [1, 2]")
    if not args.route_config.exists():
        raise FileNotFoundError(args.route_config)
    if not args.runner.exists():
        raise FileNotFoundError(args.runner)
    route_ids = load_routes(args.route_config, args.route_ids)
    if not route_ids:
        raise ValueError("No routes selected")
    tasks = [(method, route_id) for route_id in route_ids for method in args.methods]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for index, (method, route_id) in enumerate(tasks):
            print(json.dumps({"worker_id": index % args.workers, "method": method, "route_id": route_id}))
        return 0

    task_groups = [[] for _ in range(args.workers)]
    for index, task in enumerate(tasks):
        task_groups[index % args.workers].append(task)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_worker_group, worker_id, group, args): worker_id
            for worker_id, group in enumerate(task_groups)
            if group
        }
        for future in as_completed(futures):
            for result in future.result():
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    write_summary(args.output_dir, results)
    failures = [result for result in results if int(result.get("exit_code", 1)) != 0 or not result.get("valid_step_video", False)]
    return 0 if args.allow_failures or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
