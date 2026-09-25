#!/usr/bin/env python3
"""Summarize live/completed V3 batch model timeouts, bridge no-ops, and speed."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import statistics


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[position], 3)


def _model_metrics(attempts: list[Path]):
    roles = defaultdict(lambda: {"calls": 0, "errors": 0, "timeouts": 0, "latency_s": [], "queue_lag_s": []})
    m1_objects: Counter[tuple[str, str]] = Counter()
    for attempt in attempts:
        path = attempt / "mllm_metrics.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # Another worker may still be appending this line.
                if not isinstance(row, dict):
                    continue
                role = str(row.get("role") or "unknown")
                group = roles[role]
                group["calls"] += 1
                error = str(row.get("error") or "")
                timed_out = (
                    row.get("timed_out") is True
                    or row.get("is_timeout") is True
                    or bool(re.search(r"timeout|timed[\s_-]*out|deadline[^\n]*exceeded|超时", error, re.I))
                )
                group["errors"] += bool(error) or timed_out
                group["timeouts"] += timed_out
                for field, destination in (("latency_s", "latency_s"), ("queue_lag_sec", "queue_lag_s")):
                    value = row.get(field)
                    if isinstance(value, (int, float)):
                        group[destination].append(float(value))
                if role == "attribute_inference" and row.get("object_id"):
                    m1_objects[(str(attempt), str(row["object_id"]))] += 1
    return {
        "by_role": {
            role: {
                "calls": values["calls"],
                "errors": values["errors"],
                "timeouts": values["timeouts"],
                "latency_p50_s": percentile(values["latency_s"], 0.5),
                "latency_p95_s": percentile(values["latency_s"], 0.95),
                "queue_lag_p95_s": percentile(values["queue_lag_s"], 0.95),
            }
            for role, values in sorted(roles.items())
        },
        "m1_max_calls_per_object": max(m1_objects.values(), default=0),
        "m1_objects_above_ten_calls": sum(calls > 10 for calls in m1_objects.values()),
    }


def _resource_metrics(path: Path):
    if not path.is_file():
        return {}
    gpu_samples: dict[int, list[dict]] = defaultdict(list)
    cpu = []
    elapsed = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    elapsed.append(float(row["elapsed_sec"]))
                    cpu.append(float(row["host_cpu_percent"]))
                    for gpu in json.loads(row.get("gpu_metrics_json") or "[]"):
                        gpu_samples[int(gpu["index"])].append(gpu)
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return {
        "duration_s": round(max(elapsed), 1) if elapsed else None,
        "mean_host_cpu_percent": round(statistics.mean(cpu), 2) if cpu else None,
        "gpus": {
            str(index): {
                "peak_memory_gib": round(max(item["memory_used_mb"] for item in samples) / 1024, 2),
                "peak_gpu_percent": max(item["utilization_gpu_percent"] for item in samples),
            }
            for index, samples in sorted(gpu_samples.items())
        },
    }


def summarize(directory: Path) -> dict:
    results = []
    attempts = []
    bridge_timeouts = bridge_actions = 0
    for episode in sorted(directory.glob("episode_*")):
        if not episode.is_dir():
            continue
        for attempt in sorted(episode.glob("attempt_*"), reverse=True):
            if attempt.is_dir():
                attempts.append(attempt)
                document = read_json(attempt / "eval" / "results.json")
                if isinstance(document, list) and len(document) == 1 and isinstance(document[0], dict):
                    row = document[0]
                    results.append(row)
                    trace_path = row.get("trace_path")
                    trace = read_json(Path(trace_path)) if trace_path else None
                    for event in (trace or {}).get("trace", []):
                        action = event.get("action")
                        if isinstance(action, dict):
                            bridge_actions += 1
                            bridge_timeouts += bool((action.get("metadata") or {}).get("bridge_action_timed_out"))
                    break
    launch_config = read_json(directory / "launch_config.json") or {}
    episodes_expected = launch_config.get("episode_indices")
    if episodes_expected is None:
        episodes_expected = [
            index
            for lo, hi in launch_config.get("episode_ranges", [])
            for index in range(lo, hi + 1)
        ]
    time_total_ms = step_count = 0
    for row in results:
        timing = ((row.get("timing_summary") or {}).get("phases") or {}).get("step_total") or {}
        time_total_ms += float(timing.get("total") or 0)
        step_count += int(timing.get("count") or 0)
    domains = defaultdict(lambda: {"completed": 0, "success": 0})
    for row in results:
        domain = "+".join(row.get("domains") or []) or "unknown"
        domains[domain]["completed"] += 1
        domains[domain]["success"] += bool(row.get("success"))
    resources = _resource_metrics(directory / "resource_telemetry.csv")
    applied = sum(int(row.get("applied_action_step_count") or 0) for row in results)
    scoring_eligible = [
        row for row in results
        if row.get("terminal_reason") != "runtime_consistency_ineligible"
    ]
    resource_seconds = resources.get("duration_s") or 0
    return {
        "expected": len(episodes_expected),
        "completed": len(results),
        "scoring_eligible_completed": len(scoring_eligible),
        "scoring_eligible_success_rate": (
            round(sum(bool(row.get("success")) for row in scoring_eligible) / len(scoring_eligible), 4)
            if scoring_eligible else None
        ),
        "by_domain": dict(sorted(domains.items())),
        "terminal_reasons": dict(Counter(row.get("terminal_reason") or "unknown" for row in results)),
        "success": sum(bool(row.get("success")) for row in results),
        "nav_success": sum(bool(row.get("nav_success")) for row in results),
        "required_interaction_success": sum(bool(row.get("required_interaction_success")) for row in results),
        "applied_action_steps": applied,
        "no_fresh_actions": sum(int(row.get("no_fresh_action_count") or 0) for row in results),
        "ros_bridge_action_timeouts": bridge_timeouts,
        "ros_bridge_actions": bridge_actions,
        "ros_bridge_timeout_rate": round(bridge_timeouts / bridge_actions, 4) if bridge_actions else None,
        "weighted_seconds_per_step": round(time_total_ms / 1000 / step_count, 3) if step_count else None,
        "completed_steps_per_wall_second": round(applied / resource_seconds, 3) if resource_seconds else None,
        "model": _model_metrics(attempts),
        "resources": resources,
        "deployment": read_json(directory / "qwen-service" / "deployment.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = summarize(args.evaluation_dir)
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(formatted + "\n", encoding="utf-8")
    print(formatted)


if __name__ == "__main__":
    main()
