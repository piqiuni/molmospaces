#!/usr/bin/env python3
"""Run frozen House 7 force-interaction routes with at most two ROS workers."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
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
DEFAULT_ROUTE_RUNNER = REPO_ROOT / "scripts" / "InteractiveNav" / "run_house7_force_route_ros_test.zsh"


def validate_worker_count(worker_count: int) -> int:
    count = int(worker_count)
    if count < 1:
        raise ValueError("--workers must be at least 1")
    if count > 2:
        raise ValueError("--workers cannot exceed 2 because two simulators approach the memory limit")
    return count


def split_round_robin(items: list[str], worker_count: int) -> list[list[str]]:
    return [items[index::worker_count] for index in range(worker_count)]


def load_route_ids(path: Path, selected: list[str] | None = None) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    available = [str(route["route_id"]) for route in payload.get("routes") or []]
    if not selected:
        return available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown route ids: {unknown}")
    return [route_id for route_id in available if route_id in set(selected)]


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def terminate_process_group(process: subprocess.Popen, grace_s: float = 20.0) -> None:
    if process.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGINT, grace_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def route_command(args: argparse.Namespace, route_dir: Path, route_id: str) -> list[str]:
    return ["zsh", str(args.route_runner), str(route_dir), route_id]


class RouteBatchWorker:
    def __init__(self, worker_id: int, route_ids: list[str], args: argparse.Namespace) -> None:
        self.worker_id = int(worker_id)
        self.route_ids = list(route_ids)
        self.args = args
        self.master_port = int(args.base_master_port) + self.worker_id
        self.master_uri = f"http://127.0.0.1:{self.master_port}"
        self.results: list[dict[str, Any]] = []

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "ROS_MASTER_URI": self.master_uri,
                "ROUTE_CONFIG": str(self.args.route_config),
                "TASK_HORIZON": str(self.args.task_horizon),
                "ROUTE_READY_TIMEOUT_S": str(self.args.ready_timeout_s),
                "ROUTE_NAVIGATION_TIMEOUT_S": str(self.args.navigation_timeout_s),
                "ROUTE_INTERACTION_TIMEOUT_S": str(self.args.interaction_timeout_s),
                "ROUTE_GRAPH_TIMEOUT_S": str(self.args.graph_timeout_s),
                "VIDEO_FPS": str(self.args.video_fps),
                "VIDEO_PANEL_WIDTH_PX": str(self.args.video_panel_width_px),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def run(self) -> None:
        environment = self._environment()
        for route_id in self.route_ids:
            route_dir = self.args.output_dir / route_id
            existing = read_json(route_dir / "route_result.json")
            if self.args.resume and bool((existing.get("result") or {}).get("success")):
                self.results.append(self._summarize(route_id, route_dir, 0, 0.0, resumed=True))
                continue
            if route_dir.exists() and any(route_dir.iterdir()):
                raise FileExistsError(
                    f"Route output is not empty: {route_dir}; use --resume or a new output directory"
                )
            route_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            command = route_command(self.args, route_dir, route_id)
            with (route_dir / "batch_runner.log").open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    exit_code = process.wait(timeout=float(self.args.route_timeout_s))
                except subprocess.TimeoutExpired:
                    terminate_process_group(process)
                    exit_code = 124
            self.results.append(
                self._summarize(
                    route_id,
                    route_dir,
                    exit_code,
                    time.monotonic() - started,
                    resumed=False,
                )
            )

    def _summarize(
        self,
        route_id: str,
        route_dir: Path,
        exit_code: int,
        elapsed_s: float,
        resumed: bool,
    ) -> dict[str, Any]:
        route_payload = read_json(route_dir / "route_result.json")
        route_result = route_payload.get("result") or {}
        video_summary = read_json(route_dir / "offline_video_summary.json")
        return {
            "worker_id": self.worker_id,
            "route_id": route_id,
            "ros_master_uri": self.master_uri,
            "output_dir": str(route_dir),
            "exit_code": int(exit_code),
            "elapsed_s": float(elapsed_s),
            "resumed": bool(resumed),
            "success": bool(route_result.get("success")) and int(exit_code) == 0,
            "route_status": route_result.get("status", "UNKNOWN"),
            "route_duration_s": route_result.get("duration_s"),
            "approach_duration_s": (route_result.get("approach_navigation") or {}).get("duration_s"),
            "final_duration_s": (route_result.get("final_navigation") or {}).get("duration_s"),
            "final_room_count": (route_result.get("final_graph") or {}).get("room_count"),
            "video": video_summary.get("video", ""),
            "video_frame_count": video_summary.get("output_frame_count", 0),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE_CONFIG)
    parser.add_argument("--route-runner", type=Path, default=DEFAULT_ROUTE_RUNNER)
    parser.add_argument("--route-ids", nargs="*")
    parser.add_argument("--max-routes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-master-port", type=int, default=11450)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-horizon", type=int, default=1000)
    parser.add_argument("--ready-timeout-s", type=float, default=90.0)
    parser.add_argument("--navigation-timeout-s", type=float, default=180.0)
    parser.add_argument("--interaction-timeout-s", type=float, default=30.0)
    parser.add_argument("--graph-timeout-s", type=float, default=30.0)
    parser.add_argument("--route-timeout-s", type=float, default=600.0)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--video-panel-width-px", type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.route_config = args.route_config.expanduser().resolve()
    args.route_runner = args.route_runner.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    worker_count = validate_worker_count(args.workers)
    if not args.route_config.exists():
        raise FileNotFoundError(args.route_config)
    if not args.route_runner.exists():
        raise FileNotFoundError(args.route_runner)
    route_ids = load_route_ids(args.route_config, args.route_ids)
    if args.max_routes > 0:
        route_ids = route_ids[: int(args.max_routes)]
    if not route_ids:
        raise ValueError("No routes selected")
    worker_count = min(worker_count, len(route_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = split_round_robin(route_ids, worker_count)
    workers = [RouteBatchWorker(index, shard, args) for index, shard in enumerate(shards)]
    plan = {
        "route_config": str(args.route_config),
        "route_ids": route_ids,
        "workers": [
            {
                "worker_id": worker.worker_id,
                "route_ids": worker.route_ids,
                "ros_master_uri": worker.master_uri,
            }
            for worker in workers
        ],
    }
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    occupied = [worker.master_port for worker in workers if not port_available(worker.master_port)]
    if occupied:
        raise RuntimeError(f"ROS master ports are occupied: {occupied}")

    errors: list[BaseException] = []

    def run_worker(worker: RouteBatchWorker) -> None:
        try:
            worker.run()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_worker, args=(worker,)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    results = [result for worker in workers for result in worker.results]
    summary = {
        "success": all(result["success"] for result in results),
        "route_count": len(results),
        "success_count": sum(1 for result in results if result["success"]),
        "failure_count": sum(1 for result in results if not result["success"]),
        "max_workers": worker_count,
        "results": sorted(results, key=lambda result: result["route_id"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
