#!/usr/bin/env python3
"""Record host and GPU resource use while an arbitrary batch PID is alive.

This monitor is intentionally independent of a particular runner.  It can be
started beside recording-heavy house batches as well as evaluator-only V3
batches, and writes only small CSV/JSON files below the supplied output
directory.

Example::

    bash scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py ... &
    batch_pid=$!
    /home/ldl/conda_envs/mlspaces/bin/python \\
      scripts/InteractiveNav/monitor_batch_resources.py \\
      --pid "${batch_pid}" --output-dir /home/ldl/outputs/interactive-nav/run_name &
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample host CPU/RAM and per-GPU memory/utilization until a batch PID exits."
    )
    parser.add_argument("--pid", type=int, required=True, help="Top-level batch process PID.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path (default: <output-dir>/resource_telemetry.csv).",
    )
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=0.0,
        help="Optional positive wall-clock cap; 0 means follow PID until it exits.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def host_memory_usage() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            values[key] = float(fields[0]) / 1024.0
        except ValueError:
            continue
    result: dict[str, float] = {}
    if "MemTotal" in values and "MemAvailable" in values:
        result["host_mem_used_mb"] = values["MemTotal"] - values["MemAvailable"]
        result["host_mem_available_mb"] = values["MemAvailable"]
    if "SwapTotal" in values and "SwapFree" in values:
        result["host_swap_used_mb"] = values["SwapTotal"] - values["SwapFree"]
    return result


def host_cpu_counters() -> tuple[float, float] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not fields or fields[0] != "cpu":
        return None
    try:
        counters = [float(value) for value in fields[1:]]
    except ValueError:
        return None
    if not counters:
        return None
    return sum(counters), (counters[3] if len(counters) > 3 else 0.0) + (counters[4] if len(counters) > 4 else 0.0)


def gpu_usage() -> list[dict[str, float | int]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    result: list[dict[str, float | int]] = []
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            result.append(
                {
                    "index": int(fields[0]),
                    "memory_used_mb": float(fields[1]),
                    "memory_total_mb": float(fields[2]),
                    "utilization_gpu_percent": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return result


def process_records() -> dict[int, tuple[int, float, int]]:
    """Return pid -> (ppid, RSS MB, user+system ticks) for readable processes."""

    records: dict[int, tuple[int, float, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            tail = stat[stat.rfind(")") + 2 :].split()
            # tail starts at Linux field 3 (state): ppid=field4 -> index1,
            # utime/stime=fields14/15 -> indexes11/12.
            parent_pid = int(tail[1])
            ticks = int(tail[11]) + int(tail[12])
            rss_mb = 0.0
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
        records[int(entry.name)] = (parent_pid, rss_mb, ticks)
    return records


def process_tree_usage(root_pid: int) -> tuple[int, float, int]:
    records = process_records()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _rss_mb, _ticks) in records.items():
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    selected = [records[pid] for pid in descendants if pid in records]
    return len(selected), sum(record[1] for record in selected), sum(record[2] for record in selected)


def pid_alive(pid: int) -> bool:
    # A short-lived shell child can remain a zombie until its parent reaps it;
    # treating that as alive would make a monitor run forever after the batch
    # has already stopped.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2 :].split()
        if tail and tail[0] == "Z":
            return False
    except (FileNotFoundError, PermissionError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def summarize(path: Path) -> dict[str, Any]:
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError:
        rows = []

    def values(key: str) -> list[float]:
        output: list[float] = []
        for row in rows:
            raw = row.get(key)
            if raw in {None, ""}:
                continue
            try:
                output.append(float(raw))
            except ValueError:
                continue
        return output

    gpu_peaks: dict[int, dict[str, float | int]] = {}
    for row in rows:
        try:
            metrics = json.loads(row.get("gpu_metrics_json") or "[]")
        except ValueError:
            metrics = []
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            try:
                index = int(metric["index"])
                memory = float(metric["memory_used_mb"])
                total = float(metric["memory_total_mb"])
                utilization = float(metric["utilization_gpu_percent"])
            except (KeyError, TypeError, ValueError):
                continue
            peak = gpu_peaks.setdefault(
                index,
                {
                    "index": index,
                    "peak_memory_used_mb": 0.0,
                    "memory_total_mb": total,
                    "peak_utilization_gpu_percent": 0.0,
                },
            )
            peak["peak_memory_used_mb"] = max(float(peak["peak_memory_used_mb"]), memory)
            peak["memory_total_mb"] = total
            peak["peak_utilization_gpu_percent"] = max(float(peak["peak_utilization_gpu_percent"]), utilization)
    host_cpu = values("host_cpu_percent")
    batch_cpu = values("batch_cpu_percent")
    batch_rss = values("batch_rss_mb")
    host_memory = values("host_mem_used_mb")
    host_available = values("host_mem_available_mb")
    return {
        "resource_telemetry_path": str(path),
        "sample_count": len(rows),
        "mean_host_cpu_percent": sum(host_cpu) / len(host_cpu) if host_cpu else None,
        "peak_host_cpu_percent": max(host_cpu) if host_cpu else None,
        "peak_batch_cpu_percent": max(batch_cpu) if batch_cpu else None,
        "peak_batch_rss_mb": max(batch_rss) if batch_rss else None,
        "peak_host_mem_used_mb": max(host_memory) if host_memory else None,
        "min_host_mem_available_mb": min(host_available) if host_available else None,
        "gpu_peaks": [gpu_peaks[index] for index in sorted(gpu_peaks)],
    }


def main() -> int:
    args = parse_args()
    if args.pid < 1:
        raise ValueError("--pid must be positive")
    if args.interval_s <= 0.0:
        raise ValueError("--interval-s must be positive")
    if args.max_duration_s < 0.0:
        raise ValueError("--max-duration-s must be non-negative")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (args.output or args.output_dir / "resource_telemetry.csv").expanduser().resolve()
    if output.parent != args.output_dir and args.output is None:
        raise AssertionError("default output must live under --output-dir")
    output.parent.mkdir(parents=True, exist_ok=True)

    stopping = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    started = time.monotonic()
    previous_host_cpu = host_cpu_counters()
    previous_batch_ticks: int | None = None
    previous_sample_at: float | None = None
    clock_ticks = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    fields = [
        "wall_time",
        "elapsed_sec",
        "monitored_pid",
        "batch_process_count",
        "batch_rss_mb",
        "batch_cpu_percent",
        "host_cpu_percent",
        "host_mem_used_mb",
        "host_mem_available_mb",
        "host_swap_used_mb",
        "gpu_metrics_json",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while not stopping:
            now = time.monotonic()
            if args.max_duration_s > 0.0 and now - started >= args.max_duration_s:
                break
            alive = pid_alive(args.pid)
            count, rss_mb, ticks = process_tree_usage(args.pid)
            current_host_cpu = host_cpu_counters()
            host_cpu_percent: float | None = None
            if previous_host_cpu is not None and current_host_cpu is not None:
                total_delta = current_host_cpu[0] - previous_host_cpu[0]
                idle_delta = current_host_cpu[1] - previous_host_cpu[1]
                if total_delta > 0.0:
                    host_cpu_percent = 100.0 * max(0.0, min(1.0, 1.0 - idle_delta / total_delta))
            previous_host_cpu = current_host_cpu
            batch_cpu_percent: float | None = None
            if previous_batch_ticks is not None and previous_sample_at is not None and now > previous_sample_at:
                batch_cpu_percent = 100.0 * (ticks - previous_batch_ticks) / clock_ticks / (now - previous_sample_at)
            previous_batch_ticks = ticks
            previous_sample_at = now
            writer.writerow(
                {
                    "wall_time": time.time(),
                    "elapsed_sec": now - started,
                    "monitored_pid": args.pid,
                    "batch_process_count": count,
                    "batch_rss_mb": rss_mb,
                    "batch_cpu_percent": batch_cpu_percent,
                    "host_cpu_percent": host_cpu_percent,
                    "gpu_metrics_json": json.dumps(gpu_usage(), separators=(",", ":")),
                    **host_memory_usage(),
                }
            )
            handle.flush()
            if not alive:
                break
            time.sleep(args.interval_s)

    summary = summarize(output)
    summary.update(
        {
            "generated_at": utc_now(),
            "monitored_pid": args.pid,
            "stopped_by_signal": stopping,
        }
    )
    summary_path = output.with_name(f"{output.stem}_summary.json")
    atomic_json(summary_path, summary)
    print(json.dumps({"resource_telemetry": str(output), "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
