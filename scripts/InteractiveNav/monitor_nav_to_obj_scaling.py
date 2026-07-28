#!/usr/bin/env python3
"""Bounded, observable scale test for isolated native nav_to_obj workers.

This starts with a deliberately large worker pool and adds a fixed increment
only while host CPU is below the requested utilisation.  The test is a
controller around ``nav_to_obj_batch_manager.py``: all work still goes through
the lease ledger and every extra worker has a distinct ROS master port.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean


TIMING_RE = re.compile(
    r"SimLoop timing over (?P<count>\d+) steps: policy=(?P<policy>[0-9.]+)ms, "
    r"task=(?P<task>[0-9.]+)ms \(physics=(?P<physics>[0-9.]+)ms "
    r"sensors=(?P<sensors>[0-9.]+)ms\), loop=(?P<loop>[0-9.]+)ms"
)


def _cpu_snapshot() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    return sum(values), values[3] + values[4]


def cpu_percent(sample_seconds: float = 1.0) -> float:
    total_before, idle_before = _cpu_snapshot()
    time.sleep(sample_seconds)
    total_after, idle_after = _cpu_snapshot()
    total_delta = total_after - total_before
    if total_delta <= 0:
        return 0.0
    return 100.0 * (1.0 - (idle_after - idle_before) / total_delta)


def gpu_snapshot() -> list[dict[str, float | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, float | int]] = []
    for line in completed.stdout.splitlines():
        index, total_mb, used_mb, utilization = (part.strip() for part in line.split(","))
        rows.append(
            {
                "index": int(index),
                "memory_total_mb": int(total_mb),
                "memory_used_mb": int(used_mb),
                "memory_percent": round(100.0 * int(used_mb) / int(total_mb), 2),
                "gpu_util_percent": int(utilization),
            }
        )
    return rows


def _tail_text(path: Path, max_bytes: int = 128 * 1024) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        return stream.read().decode("utf-8", errors="replace")


def timing_snapshot(run_root: Path) -> dict[str, float | int | None]:
    windows: list[dict[str, float]] = []
    for path in run_root.glob("episode_*/attempt_*/stdout.log"):
        try:
            matches = list(TIMING_RE.finditer(_tail_text(path)))
        except OSError:
            continue
        if matches:
            windows.append({key: float(value) for key, value in matches[-1].groupdict().items() if key != "count"})
    result: dict[str, float | int | None] = {"timing_worker_count": len(windows)}
    for key in ("policy", "task", "physics", "sensors", "loop"):
        result[f"{key}_ms_mean"] = round(fmean(row[key] for row in windows), 3) if windows else None
    return result


def manager_status(manager: Path, run_root: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(manager), "status", "--run-root", str(run_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manager", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-workers", type=int, default=50)
    parser.add_argument("--worker-increment", type=int, default=10)
    parser.add_argument("--max-increments", type=int, default=10)
    parser.add_argument("--target-cpu-percent", type=float, default=80.0)
    parser.add_argument("--warmup-seconds", type=float, default=90.0)
    parser.add_argument("--stage-seconds", type=float, default=60.0)
    parser.add_argument("--gpu-devices", default="1,2,3,4,5,7")
    args = parser.parse_args()
    if args.initial_workers < 1 or args.worker_increment < 1 or args.max_increments < 0:
        raise ValueError("worker counts must be positive and max increments non-negative")
    devices = [value.strip() for value in args.gpu_devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("--gpu-devices must name at least one GPU")

    args.run_root.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_root / "scale_metrics.jsonl"
    controller_path = args.run_root / "scale_controller.json"
    controller_path.write_text(
        json.dumps({"pid": __import__("os").getpid(), "started_at": time.time(), "args": vars(args)}, default=str, indent=2)
        + "\n",
        encoding="utf-8",
    )
    children: list[subprocess.Popen[bytes]] = []
    launched_workers = 0

    def launch(worker_count: int) -> None:
        nonlocal launched_workers
        bindings = ",".join(devices[index % len(devices)] for index in range(launched_workers, launched_workers + worker_count))
        command = [
            sys.executable,
            str(args.manager),
            "run",
            "--run-root",
            str(args.run_root),
            "--workers",
            str(worker_count),
            "--worker-slot-start",
            str(launched_workers),
            "--worker-id-prefix",
            "scale",
            "--cuda-visible-devices-list",
            bindings,
        ]
        children.append(subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        launched_workers += worker_count

    launch(args.initial_workers)
    for stage in range(args.max_increments + 1):
        delay = args.warmup_seconds if stage == 0 else args.stage_seconds
        time.sleep(delay)
        cpu = cpu_percent()
        row: dict[str, object] = {
            "timestamp": time.time(),
            "stage": stage,
            "launched_workers": launched_workers,
            "cpu_percent": round(cpu, 3),
            "gpu": gpu_snapshot(),
            "timing": timing_snapshot(args.run_root),
            "manager": manager_status(args.manager, args.run_root),
            "controller_children_alive": sum(child.poll() is None for child in children),
        }
        append_jsonl(metrics_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if cpu >= args.target_cpu_percent or stage == args.max_increments:
            break
        launch(args.worker_increment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
