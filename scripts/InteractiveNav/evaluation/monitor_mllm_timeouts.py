"""Read-only, periodic MLLM timeout report for a custom benchmark task."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time


TIMEOUT = re.compile(r"timeout|timed[\s_-]*out|deadline[^\n]*exceeded|超时", re.I)
ENGINE = re.compile(r"Engine\s+\d+: Avg prompt throughput: ([\d.]+) tokens/s, "
                    r"Avg generation throughput: ([\d.]+) tokens/s, "
                    r"Running: (\d+) reqs, Waiting: (\d+) reqs")
ROLE_GROUPS = {
    "attribute_inference": "M1-objects",
    "room_attribute_inference": "M1-rooms",
    "subgoal_selection": "M2",
    "visual_verification": "M3",
}
TERMINAL = {"Success", "Failed", "Cancelled", "Killed", "Exception"}


class BackendLogMonitor:
    """Incrementally read all vLLM replica logs and count completed HTTP calls."""

    def __init__(self, directory):
        self.directory = directory
        self.offsets = {}
        self.pending = {}
        self.responses = Counter()
        self.previous_responses = Counter()
        self.latest = {}

    def report(self):
        snapshots = []
        names = sorted(
            (path.name.removesuffix(".launcher.log")
             for path in self.directory.glob("*.launcher.log")
             if path.name == "vllm.launcher.log" or path.name.startswith("gpu")),
            key=lambda name: (0, int(name[3:])) if name[3:].isdigit() else (1, name),
        )
        for name in names:
            path = self.directory / f"{name}.launcher.log"
            try:
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    size = stream.tell()
                    if size < self.offsets.get(name, 0):
                        self.offsets[name] = 0
                        self.pending[name] = b""
                        self.responses[name] = 0
                        self.previous_responses[name] = 0
                        self.latest.pop(name, None)
                    stream.seek(self.offsets.get(name, 0))
                    chunk = stream.read()
                    self.offsets[name] = stream.tell()
            except OSError:
                snapshots.append(f"{name}: 尚未启动")
                continue
            lines = (self.pending.get(name, b"") + chunk).split(b"\n")
            self.pending[name] = lines.pop()
            for raw in lines:
                line = raw.decode("utf-8", errors="replace")
                if 'POST /v1/chat/completions HTTP/1.1" 200 OK' in line:
                    self.responses[name] += 1
                match = ENGINE.search(line)
                if match:
                    self.latest[name] = (float(match[1]), float(match[2]), int(match[3]), int(match[4]))
            delta = self.responses[name] - self.previous_responses[name]
            self.previous_responses[name] = self.responses[name]
            stats = self.latest.get(name)
            if stats:
                prompt, generation, running, waiting = stats
                snapshots.append(f"{name}: HTTP200累计={self.responses[name]} 增量={delta} "
                                 f"running={running} waiting={waiting} "
                                 f"prefill={prompt:.1f}tok/s gen={generation:.1f}tok/s")
            else:
                state = "已就绪/待请求" if (self.directory / "deployment.json").is_file() else "启动中"
                snapshots.append(f"{name}: HTTP200累计={self.responses[name]} 增量={delta} {state}")
        if names and all(name in self.latest for name in names):
            total = sum(self.responses[name] for name in names)
            snapshots.append(
                "后端完成比例 " + " ".join(
                    f"{name}={100 * self.responses[name] / total:.1f}%"
                    for name in names
                ) if total else "后端尚无完成请求"
            )
        return "\n  ".join(snapshots)


def task_status(task_id):
    try:
        result = subprocess.run(
            ["volc", "ml_task", "get", "-i", task_id, "--output", "json"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return json.loads(result.stdout[result.stdout.index("["):])[0]["Status"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        return f"status-unavailable:{type(exc).__name__}"


def summarize(root, window_s):
    now = time.time()
    groups = defaultdict(lambda: {"calls": 0, "errors": 0, "timeouts": 0, "latencies": []})
    recent = defaultdict(lambda: {"calls": 0, "errors": 0, "timeouts": 0, "latencies": []})
    attempts = 0
    malformed = 0
    for path in sorted(root.glob("episode_*/attempt_*/mllm_metrics.jsonl")):
        attempts += 1
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        malformed += 1  # A final line can be in the middle of a write.
                        continue
                    if not isinstance(row, dict):
                        continue
                    group = ROLE_GROUPS.get(row.get("role"), row.get("role") or "unknown")
                    error = row.get("error")
                    timed_out = (row.get("timed_out") is True or row.get("is_timeout") is True
                                 or isinstance(error, str) and bool(TIMEOUT.search(error)))
                    for bucket, include in ((groups, True), (recent, isinstance(row.get("timestamp"), (int, float))
                                             and now - window_s <= row["timestamp"] <= now + 5)):
                        if not include:
                            continue
                        target = bucket[group]
                        target["calls"] += 1
                        target["errors"] += bool(error) or timed_out
                        target["timeouts"] += timed_out
                        if isinstance(row.get("latency_s"), (int, float)):
                            target["latencies"].append(row["latency_s"])
        except OSError:
            continue

    completed = succeeded = 0
    for path in root.glob("episode_*/batch_task_summary.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if str(row.get("completed", "")).lower() == "true":
            completed += 1
            succeeded += str(row.get("task_success", "")).lower() == "true"
    return groups, recent, attempts, malformed, completed, succeeded


def format_groups(groups):
    parts = []
    for name in ("M1-objects", "M1-rooms", "M2", "M3"):
        entry = groups[name]
        values = sorted(entry["latencies"])
        p95 = values[min(len(values) - 1, int(len(values) * .95))] if values else None
        parts.append(f"{name} {entry['calls']}次/超时{entry['timeouts']}/其他错误"
                     f"{entry['errors'] - entry['timeouts']}/p95={p95:.1f}s" if p95 is not None else
                     f"{name} {entry['calls']}次/超时{entry['timeouts']}/其他错误"
                     f"{entry['errors'] - entry['timeouts']}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--qwen-dir", type=Path, help="Qwen replica service log directory")
    args = parser.parse_args()
    backends = BackendLogMonitor(args.qwen_dir) if args.qwen_dir else None
    launch_config = args.evaluation_dir / "launch_config.json"
    try:
        expected = len(json.loads(launch_config.read_text())["episode_indices"])
    except (OSError, ValueError, KeyError, TypeError):
        expected = "?"
    while True:
        status = task_status(args.task_id)
        total, recent, attempts, malformed, completed, succeeded = summarize(args.evaluation_dir, args.window)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"[{stamp}] task={status} 已完成={completed}/{expected} 成功={succeeded} "
              f"metrics文件={attempts} 不完整行={malformed}", flush=True)
        print(f"  最近{args.window}s: {format_groups(recent)}", flush=True)
        print(f"  累计: {format_groups(total)}", flush=True)
        if backends:
            print(f"  模型后端: {backends.report()}", flush=True)
        if status in TERMINAL:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
