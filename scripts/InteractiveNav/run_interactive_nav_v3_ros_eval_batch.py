#!/usr/bin/env python3
"""Run isolated single-episode InteractiveNav V3 ROS evaluations in parallel.

The V3 ROS runner owns one ROS master, simulator, recorder, evaluator, and
offline visual artefact set.  It must therefore be invoked once per episode;
this wrapper only schedules those independent invocations.  A task directory
keeps every attempt, so ``--resume`` can retry an incomplete episode without
overwriting its previous ROS/evaluator evidence.

Example (outputs and caches should live on the large /home/ldl volume)::

    /home/ldl/conda_envs/mlspaces/bin/python \
      scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py \
      --benchmark scripts/InteractiveNav/output/.../benchmark/mixed.json \
      --output-dir /home/ldl/outputs/interactive-nav/v3_mixed_10x \
      --episode-indices 0 1 2 3 4 5 6 7 8 9 \
      --workers 10 --base-master-port 12600 --max-steps 1500 \
      --model-endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
      --no-recording --scene-timeout-s 7200 --allow-failures

``--no-recording`` (also available as ``--fast-eval``) retains the canonical
episode-result JSON and batch telemetry but starts no recorder, writes no
per-step frames, and does not wait for a recorder acknowledgement.  This is
the intended mode for the large V3 success-rate runs.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "scripts" / "InteractiveNav" / "run_interactive_nav_v3_ros_eval_test.zsh"
SUMMARY_SCHEMA_VERSION = "interactive_nav_v3_ros_batch_v2"


@dataclass(frozen=True)
class EpisodePlan:
    """One independently runnable V3 episode task."""

    ordinal: int
    worker_id: int
    episode_index: int
    master_port: int
    task_dir: Path

    @property
    def ros_master_uri(self) -> str:
        return f"http://127.0.0.1:{self.master_port}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated V3 ROS object-goal benchmark episodes in a worker pool."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Explicit frozen V3 benchmark JSON (for example channel.json, container.json, or mixed.json).",
    )
    parser.add_argument("--episode-indices", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-master-port", type=int, default=12600)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--scene-timeout-s",
        type=float,
        default=7200.0,
        help="Wall-clock cap for one full runner lifecycle, including recorder drain/rendering.",
    )
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--runner-shell",
        choices=("bash", "zsh"),
        default="bash",
        help="Shell used to launch --runner; the maintained runner is Bash-compatible.",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Conda environment name or absolute prefix passed to the single-episode runner.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=None,
        help="MolmoSpaces Python executable passed to the single-episode runner.",
    )
    parser.add_argument(
        "--model-endpoints",
        nargs="+",
        metavar="URL",
        help=(
            "Local MLLM endpoint URLs. Logical workers are assigned URLs round-robin; "
            "each episode receives a derived dotenv with its assigned endpoint."
        ),
    )
    parser.add_argument(
        "--semantic-model-env-file",
        type=Path,
        help=(
            "Base semantic-model dotenv. With --model-endpoints it is copied into "
            "each attempt and only SEMANTIC_MODEL_ENDPOINT is overridden."
        ),
    )
    parser.add_argument(
        "--mujoco-egl-devices",
        nargs="+",
        metavar="DEVICE_ID",
        help="Optional EGL GPU IDs assigned to logical workers round-robin.",
    )
    parser.add_argument(
        "--no-recording",
        "--fast-eval",
        dest="fast_eval",
        action="store_true",
        help=(
            "Run evaluator-only mode: no recorder, no frame queue/ack barrier, no MP4/topdown requirement."
        ),
    )
    parser.add_argument(
        "--resource-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample batch-wide host CPU/RAM and GPU utilization/memory (default: enabled).",
    )
    parser.add_argument(
        "--resource-sample-interval-s",
        type=float,
        default=2.0,
        help="Resource telemetry sampling period in seconds.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def assigned_model_endpoint(worker_id: int, args: argparse.Namespace) -> str | None:
    endpoints = args.model_endpoints
    if not endpoints:
        return None
    return str(endpoints[worker_id % len(endpoints)])


def assigned_mujoco_egl_device(worker_id: int, args: argparse.Namespace) -> str | None:
    devices = args.mujoco_egl_devices
    if not devices:
        return None
    return str(devices[worker_id % len(devices)])


def derive_episode_model_env_file(
    attempt_dir: Path,
    source_env_file: Path,
    endpoint: str,
) -> Path:
    """Copy the local model config and append an episode-specific endpoint.

    ``python-dotenv`` loads this file with ``override=True`` in the semantic
    stack, so the final setting wins without replacing the model name, timeout,
    or any local-only configuration in the source file.
    """

    source_text = source_env_file.read_text(encoding="utf-8")
    if source_text and not source_text.endswith("\n"):
        source_text += "\n"
    derived_path = attempt_dir / "semantic_model.env"
    derived_text = (
        source_text
        + "\n# Generated by run_interactive_nav_v3_ros_eval_batch.py; do not edit mid-run.\n"
        + f"SEMANTIC_MODEL_ENDPOINT={shlex.quote(endpoint)}\n"
    )
    temporary = derived_path.with_name(f".{derived_path.name}.{os.getpid()}.tmp")
    temporary.write_text(derived_text, encoding="utf-8")
    temporary.replace(derived_path)
    return derived_path


def host_memory_usage() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        fields = raw_value.strip().split()
        if not fields:
            continue
        try:
            values[key] = float(fields[0]) / 1024.0
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    result: dict[str, float] = {}
    if total is not None and available is not None:
        result["host_mem_used_mb"] = total - available
        result["host_mem_available_mb"] = available
    if swap_total is not None and swap_free is not None:
        result["host_swap_used_mb"] = swap_total - swap_free
    return result


def host_cpu_counters() -> tuple[float, float] | None:
    """Return total and idle Linux CPU jiffies, or ``None`` when unavailable."""

    try:
        first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not first or first[0] != "cpu":
        return None
    try:
        counters = [float(value) for value in first[1:]]
    except ValueError:
        return None
    if not counters:
        return None
    idle = counters[3] if len(counters) > 3 else 0.0
    iowait = counters[4] if len(counters) > 4 else 0.0
    return sum(counters), idle + iowait


def gpu_usage() -> list[dict[str, float | int]]:
    """Read GPU utilization without requiring a Python CUDA dependency."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, float | int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "memory_used_mb": float(fields[1]),
                    "memory_total_mb": float(fields[2]),
                    "utilization_gpu_percent": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return rows


class BatchResourceTelemetry:
    """One lightweight sampler for the whole batch, not one ``nvidia-smi`` per worker."""

    fieldnames = (
        "wall_time",
        "elapsed_sec",
        "active_episode_count",
        "active_episode_indices",
        "host_cpu_percent",
        "host_mem_used_mb",
        "host_mem_available_mb",
        "host_swap_used_mb",
        "gpu_metrics_json",
    )

    def __init__(self, output_dir: Path, interval_s: float) -> None:
        self.path = output_dir / "resource_telemetry.csv"
        self.interval_s = max(0.25, float(interval_s))
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active: dict[int, int] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="v3-batch-resource-monitor", daemon=True)
        self._thread.start()

    def mark_started(self, plan: EpisodePlan) -> None:
        with self._lock:
            self._active[plan.ordinal] = plan.episode_index

    def mark_finished(self, plan: EpisodePlan) -> None:
        with self._lock:
            self._active.pop(plan.ordinal, None)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s + 1.0))
        return summarize_resource_telemetry(self.path)

    def _run(self) -> None:
        previous_cpu = host_cpu_counters()
        # ``--resume`` may start the wrapper after previous episodes have
        # already produced useful peak telemetry.  Append rather than truncate
        # that evidence; the summary naturally aggregates all samples.
        append = self.path.is_file() and self.path.stat().st_size > 0
        with self.path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not append:
                writer.writeheader()
            while True:
                current_cpu = host_cpu_counters()
                host_cpu_percent: float | None = None
                if previous_cpu is not None and current_cpu is not None:
                    total_delta = current_cpu[0] - previous_cpu[0]
                    idle_delta = current_cpu[1] - previous_cpu[1]
                    if total_delta > 0.0:
                        host_cpu_percent = 100.0 * max(0.0, min(1.0, 1.0 - idle_delta / total_delta))
                previous_cpu = current_cpu
                with self._lock:
                    active_indexes = sorted(self._active.values())
                row: dict[str, Any] = {
                    "wall_time": time.time(),
                    "elapsed_sec": time.monotonic() - self._started,
                    "active_episode_count": len(active_indexes),
                    "active_episode_indices": json.dumps(active_indexes),
                    "host_cpu_percent": host_cpu_percent,
                    "gpu_metrics_json": json.dumps(gpu_usage(), separators=(",", ":")),
                    **host_memory_usage(),
                }
                writer.writerow(row)
                handle.flush()
                if self._stop.wait(self.interval_s):
                    break


def summarize_resource_telemetry(path: Path) -> dict[str, Any]:
    """Return compact peak metrics for the batch summary while retaining raw CSV."""

    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError:
        rows = []
    if not rows:
        return {"resource_telemetry_path": str(path), "resource_sample_count": 0}

    def numeric_values(key: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            raw = row.get(key)
            if raw in {None, ""}:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        return values

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
                memory_used = float(metric["memory_used_mb"])
                memory_total = float(metric["memory_total_mb"])
                utilization = float(metric["utilization_gpu_percent"])
            except (KeyError, TypeError, ValueError):
                continue
            peak = gpu_peaks.setdefault(
                index,
                {
                    "index": index,
                    "peak_memory_used_mb": 0.0,
                    "memory_total_mb": memory_total,
                    "peak_utilization_gpu_percent": 0.0,
                },
            )
            peak["peak_memory_used_mb"] = max(float(peak["peak_memory_used_mb"]), memory_used)
            peak["memory_total_mb"] = memory_total
            peak["peak_utilization_gpu_percent"] = max(
                float(peak["peak_utilization_gpu_percent"]), utilization
            )

    cpu = numeric_values("host_cpu_percent")
    memory_used = numeric_values("host_mem_used_mb")
    memory_available = numeric_values("host_mem_available_mb")
    swap_used = numeric_values("host_swap_used_mb")
    active = numeric_values("active_episode_count")
    return {
        "resource_telemetry_path": str(path),
        "resource_sample_count": len(rows),
        "peak_active_episode_count": max(active) if active else None,
        "mean_host_cpu_percent": sum(cpu) / len(cpu) if cpu else None,
        "peak_host_cpu_percent": max(cpu) if cpu else None,
        "peak_host_mem_used_mb": max(memory_used) if memory_used else None,
        "min_host_mem_available_mb": min(memory_available) if memory_available else None,
        "peak_host_swap_used_mb": max(swap_used) if swap_used else None,
        "gpu_peaks": [gpu_peaks[index] for index in sorted(gpu_peaks)],
    }


def terminate_group(process: subprocess.Popen[bytes], grace_s: float = 30.0) -> None:
    """Stop the runner and all children without leaving a ROS master behind."""

    if process.poll() is not None:
        return
    for signum, wait_s in (
        (signal.SIGINT, grace_s),
        (signal.SIGTERM, 10.0),
        (signal.SIGKILL, 2.0),
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


def infer_conda_prefix() -> str | None:
    """Infer an activated prefix only when this wrapper runs inside a conda env."""

    active = os.environ.get("CONDA_PREFIX")
    if active and (Path(active) / "bin" / "python").is_file():
        return active
    executable = Path(sys.executable).resolve()
    candidate = executable.parent.parent
    if executable.parent.name == "bin" and (candidate / "conda-meta").is_dir():
        return str(candidate)
    return None


def artifact_paths(attempt_dir: Path, episode_index: int) -> dict[str, Path | None]:
    episode_results = sorted(
        (attempt_dir / "eval" / "episodes").glob(
            f"{episode_index:04d}_*/episode_result.json"
        )
    )
    result_path = episode_results[0] if len(episode_results) == 1 else None
    episode_dir = result_path.parent if result_path is not None else None
    return {
        "episode_result": result_path,
        # The maintained V3 runner records raw artifacts under ``debug/`` and
        # builds the six-panel MP4 offline into the attempt-level videos dir.
        "six_panel_video": attempt_dir / "videos" / "overview_6panel.mp4",
        "topdown": None if episode_dir is None else episode_dir / "episode_topdown.png",
    }


def load_episode_result(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {}, None
    document = read_json(path)
    result = document.get("result")
    if isinstance(result, dict):
        status = document.get("status")
        return result, str(status) if status is not None else None
    status = document.get("status")
    return document, str(status) if status is not None else None


def video_is_valid(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def episode_result_is_complete(path: Path | None) -> bool:
    result, document_status = load_episode_result(path)
    return document_status == "complete" and result.get("status") == "complete"


def existing_completed_summary(task_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    summary = read_json(task_dir / "batch_task_summary.json")
    if not summary.get("completed"):
        return None
    # Do not let a recording-mode resume silently satisfy a fast-eval request,
    # or vice versa: their evidence contracts are intentionally different.
    if bool(summary.get("fast_eval", False)) != bool(args.fast_eval):
        return None
    episode_result = summary.get("episode_result_path")
    if not isinstance(episode_result, str) or not episode_result_is_complete(Path(episode_result)):
        return None
    if not args.fast_eval:
        video = summary.get("six_panel_video")
        if not isinstance(video, str) or not video_is_valid(Path(video)):
            return None
    return summary


def next_attempt_dir(task_dir: Path) -> Path:
    existing_numbers: list[int] = []
    for candidate in task_dir.glob("attempt_*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("attempt_")
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    return task_dir / f"attempt_{max(existing_numbers, default=0) + 1:03d}"


def plan_to_dict(plan: EpisodePlan, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ordinal": plan.ordinal,
        "worker_id": plan.worker_id,
        "episode_index": plan.episode_index,
        "ros_master_uri": plan.ros_master_uri,
        "output_dir": str(plan.task_dir),
        "model_endpoint": assigned_model_endpoint(plan.worker_id, args),
        "mujoco_egl_device": assigned_mujoco_egl_device(plan.worker_id, args),
    }


def append_task_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def encode_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def make_environment(args: argparse.Namespace, plan: EpisodePlan, attempt_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    cache_root = args.output_dir / "_runtime_cache"
    temporary_root = args.output_dir / "_tmp"
    for path in (cache_root, temporary_root, attempt_dir / "mplconfig"):
        path.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "MAX_STEPS": str(args.max_steps),
            "BENCHMARK": str(args.benchmark),
            "ROS_MASTER_URI": plan.ros_master_uri,
            "MPLCONFIGDIR": str(attempt_dir / "mplconfig"),
            "TMPDIR": str(temporary_root),
            "XDG_CACHE_HOME": str(cache_root),
            "PYTHONUNBUFFERED": "1",
            "FAST_EVAL": "true" if args.fast_eval else "false",
            "RECORD_HEAD_CAMERA": "false" if args.fast_eval else environment.get("RECORD_HEAD_CAMERA", "false"),
        }
    )
    conda_env = args.conda_env or infer_conda_prefix()
    if conda_env:
        environment["CONDA_ENV"] = conda_env
    if args.python_bin is not None:
        environment["PYTHON_BIN"] = str(args.python_bin)
    elif conda_env and Path(conda_env).is_absolute():
        candidate = Path(conda_env) / "bin" / "python"
        if candidate.is_file():
            environment["PYTHON_BIN"] = str(candidate)
    model_endpoint = assigned_model_endpoint(plan.worker_id, args)
    if model_endpoint is not None:
        assert args.semantic_model_env_file is not None
        environment["SEMANTIC_MODEL_ENV_FILE"] = str(
            derive_episode_model_env_file(attempt_dir, args.semantic_model_env_file, model_endpoint)
        )
    elif args.semantic_model_env_file is not None:
        environment["SEMANTIC_MODEL_ENV_FILE"] = str(args.semantic_model_env_file)
    mujoco_egl_device = assigned_mujoco_egl_device(plan.worker_id, args)
    if mujoco_egl_device is not None:
        environment["MUJOCO_EGL_DEVICE_ID"] = mujoco_egl_device
    return environment


def base_summary(plan: EpisodePlan, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "episode_index": plan.episode_index,
        "worker_id": plan.worker_id,
        "ros_master_uri": plan.ros_master_uri,
        "output_dir": str(plan.task_dir),
        "runner": str(args.runner),
        "benchmark": str(args.benchmark),
        "max_steps": args.max_steps,
        "fast_eval": bool(args.fast_eval),
        "recording_enabled": not bool(args.fast_eval),
        "model_endpoint": assigned_model_endpoint(plan.worker_id, args),
        "mujoco_egl_device": assigned_mujoco_egl_device(plan.worker_id, args),
    }


def run_episode(
    plan: EpisodePlan,
    args: argparse.Namespace,
    telemetry: BatchResourceTelemetry | None = None,
) -> dict[str, Any]:
    """Run one episode and leave a self-contained summary in its task directory."""

    task_dir = plan.task_dir
    if args.dry_run:
        result = base_summary(plan, args)
        result.update(
            {
                "command": [args.runner_shell, str(args.runner), str(task_dir / "attempt_001"), str(plan.episode_index)],
                "completed": False,
                "dry_run": True,
                "resumed": False,
                "fast_eval": bool(args.fast_eval),
                "model_endpoint": assigned_model_endpoint(plan.worker_id, args),
                "mujoco_egl_device": assigned_mujoco_egl_device(plan.worker_id, args),
            }
        )
        return result

    previous = existing_completed_summary(task_dir, args) if args.resume else None
    if previous is not None:
        result = dict(previous)
        result.update({"worker_id": plan.worker_id, "resumed": True, "resume_skipped": True})
        return result

    if task_dir.exists() and not args.resume and (task_dir / "batch_task_summary.json").exists():
        result = base_summary(plan, args)
        result.update(
            {
                "completed": False,
                "runner_exit_code": 2,
                "exit_code": 2,
                "error": "existing task output; use --resume or select a new --output-dir",
                "resumed": False,
            }
        )
        return result

    task_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = next_attempt_dir(task_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    environment = make_environment(args, plan, attempt_dir)
    command = [args.runner_shell, str(args.runner), str(attempt_dir), str(plan.episode_index)]
    task_log = task_dir / "batch_task.log"
    runner_log = attempt_dir / "runner.log"
    started_wall_time = utc_now()
    started = time.monotonic()
    append_task_log(
        task_log,
        f"[{started_wall_time}] attempt={attempt_dir.name} start command={shlex.join(command)}",
    )

    timed_out = False
    exception_text: str | None = None
    if telemetry is not None:
        telemetry.mark_started(plan)
    try:
        with runner_log.open("wb") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                runner_exit_code = process.wait(timeout=args.scene_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_group(process)
                runner_exit_code = 124
    except (OSError, subprocess.SubprocessError) as exc:
        runner_exit_code = 127
        exception_text = f"{type(exc).__name__}: {exc}"
    finally:
        if telemetry is not None:
            telemetry.mark_finished(plan)

    elapsed_sec = time.monotonic() - started
    paths = artifact_paths(attempt_dir, plan.episode_index)
    episode_result, document_status = load_episode_result(paths["episode_result"])
    six_panel = paths["six_panel_video"]
    topdown = paths["topdown"]
    episode_result_complete = (
        document_status == "complete" and episode_result.get("status") == "complete"
    )
    completed = bool(
        runner_exit_code == 0
        and episode_result_complete
        and (args.fast_eval or video_is_valid(six_panel))
    )
    result = base_summary(plan, args)
    result.update(
        {
            "attempt_dir": str(attempt_dir),
            "attempt": attempt_dir.name,
            "runner_log": str(runner_log),
            "command": command,
            "started_at": started_wall_time,
            "finished_at": utc_now(),
            "elapsed_sec": elapsed_sec,
            "timed_out": timed_out,
            "runner_exit_code": runner_exit_code,
            "exit_code": runner_exit_code,
            "completed": completed,
            "resumed": False,
            "episode_result_path": None if paths["episode_result"] is None else str(paths["episode_result"]),
            "episode_document_status": document_status,
            "episode_status": episode_result.get("status"),
            "artifact_validation_mode": (
                "episode_result_only_fast_eval" if args.fast_eval else "episode_result_and_six_panel_video"
            ),
            "six_panel_video": None if args.fast_eval else str(six_panel),
            "six_panel_video_bytes": (
                0 if args.fast_eval or not video_is_valid(six_panel) else six_panel.stat().st_size
            ),
            "topdown": None if args.fast_eval or topdown is None else str(topdown),
            "topdown_exists": (
                False
                if args.fast_eval
                else topdown is not None and topdown.is_file() and topdown.stat().st_size > 0
            ),
            "semantic_model_env_file": environment.get("SEMANTIC_MODEL_ENV_FILE"),
            "error": exception_text or episode_result.get("error"),
        }
    )
    for key in (
        "case_id",
        "house_index",
        "domains",
        "recipe",
        "interaction_types",
        "interaction_requirement",
        "success",
        "task_success",
        "interaction_conditioned_success",
        "nav_success",
        "required_interaction_success",
        "sequence_success",
        "terminal_reason",
        "step_count",
        "navigation_step_count",
        "view_action_count",
        "interaction_action_count",
        "correct_interaction_action_count",
        "invalid_interaction_action_count",
        "navigation_path_length_m",
        "reference_path_length_m",
        "spl",
        "elapsed_seconds",
        "target_distance_m",
        "target_visibility_fraction",
        "scoring_eligible",
        "early_stop",
        "timing_summary",
    ):
        if key in episode_result:
            result[key] = episode_result[key]
    atomic_json(attempt_dir / "wrapper_summary.json", result)
    atomic_json(task_dir / "batch_task_summary.json", result)
    append_task_log(
        task_log,
        f"[{result['finished_at']}] attempt={attempt_dir.name} exit_code={runner_exit_code} "
        f"completed={completed} elapsed_sec={elapsed_sec:.1f}",
    )
    return result


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def write_summary(
    args: argparse.Namespace,
    plans: list[EpisodePlan],
    results: list[dict[str, Any]],
    resource_summary: dict[str, Any] | None = None,
) -> None:
    ordered_results = sorted(results, key=lambda row: int(row.get("episode_index", -1)))
    completed = [row for row in ordered_results if bool(row.get("completed"))]
    failures = [row for row in ordered_results if not bool(row.get("completed"))]
    aggregate = {
        "planned_episode_count": len(plans),
        "reported_episode_count": len(ordered_results),
        "completed_episode_count": len(completed),
        "failed_or_incomplete_episode_count": len(failures),
        "six_panel_video_count": sum(video_is_valid(Path(str(row["six_panel_video"]))) for row in ordered_results if row.get("six_panel_video")),
        "formal_success_count": sum(bool(row.get("success")) for row in completed),
        "task_success_count": sum(bool(row.get("task_success")) for row in completed),
        "nav_success_count": sum(bool(row.get("nav_success")) for row in completed),
        "mean_runner_elapsed_sec": numeric_mean(completed, "elapsed_sec"),
        "mean_evaluator_elapsed_sec": numeric_mean(completed, "elapsed_seconds"),
        "mean_step_count": numeric_mean(completed, "step_count"),
        "mean_spl": numeric_mean(completed, "spl"),
    }
    if resource_summary is not None:
        aggregate["resource_telemetry"] = resource_summary
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "config": {
            "output_dir": str(args.output_dir),
            "benchmark": str(args.benchmark),
            "runner": str(args.runner),
            "runner_shell": args.runner_shell,
            "workers": args.workers,
            "base_master_port": args.base_master_port,
            "max_steps": args.max_steps,
            "scene_timeout_s": args.scene_timeout_s,
            "fast_eval": bool(args.fast_eval),
            "recording_enabled": not bool(args.fast_eval),
            "model_endpoints": args.model_endpoints,
            "mujoco_egl_devices": args.mujoco_egl_devices,
            "semantic_model_env_file": (
                None if args.semantic_model_env_file is None else str(args.semantic_model_env_file)
            ),
            "resource_telemetry": bool(args.resource_telemetry),
            "resource_sample_interval_s": args.resource_sample_interval_s,
            "resume": args.resume,
        },
        "plans": [plan_to_dict(plan, args) for plan in plans],
        "aggregate": aggregate,
        "episodes": ordered_results,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "aggregate_metrics.json", aggregate)
    fields = [
        "episode_index",
        "worker_id",
        "model_endpoint",
        "mujoco_egl_device",
        "semantic_model_env_file",
        "fast_eval",
        "artifact_validation_mode",
        "completed",
        "runner_exit_code",
        "elapsed_sec",
        "episode_status",
        "case_id",
        "house_index",
        "domains",
        "interaction_types",
        "interaction_requirement",
        "success",
        "task_success",
        "nav_success",
        "required_interaction_success",
        "terminal_reason",
        "step_count",
        "interaction_action_count",
        "spl",
        "six_panel_video",
        "topdown",
        "runner_log",
        "output_dir",
        "error",
    ]
    temporary = args.output_dir / f".summary.csv.{os.getpid()}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ordered_results:
            writer.writerow({key: encode_csv_value(row.get(key)) for key in fields})
    temporary.replace(args.output_dir / "summary.csv")


def validate_args(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.benchmark = args.benchmark.expanduser().resolve()
    args.runner = args.runner.expanduser().resolve()
    if args.python_bin is not None:
        args.python_bin = args.python_bin.expanduser().resolve()
    if args.semantic_model_env_file is not None:
        args.semantic_model_env_file = args.semantic_model_env_file.expanduser().resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.scene_timeout_s <= 0:
        raise ValueError("--scene-timeout-s must be positive")
    if args.resource_sample_interval_s <= 0.0:
        raise ValueError("--resource-sample-interval-s must be positive")
    if not 1 <= args.base_master_port <= 65535:
        raise ValueError("--base-master-port must be in [1, 65535]")
    if any(index < 0 for index in args.episode_indices):
        raise ValueError("--episode-indices must be non-negative")
    if len(set(args.episode_indices)) != len(args.episode_indices):
        raise ValueError("--episode-indices must not contain duplicates")
    if args.base_master_port + args.workers - 1 > 65535:
        raise ValueError("base master port range exceeds 65535")
    if not args.benchmark.is_file():
        raise FileNotFoundError(args.benchmark)
    if not args.runner.is_file():
        raise FileNotFoundError(args.runner)
    if args.python_bin is not None and not args.python_bin.is_file():
        raise FileNotFoundError(args.python_bin)
    if args.model_endpoints:
        args.model_endpoints = [endpoint.strip() for endpoint in args.model_endpoints]
        if any(not endpoint for endpoint in args.model_endpoints):
            raise ValueError("--model-endpoints must not contain an empty URL")
    if args.mujoco_egl_devices:
        args.mujoco_egl_devices = [device.strip() for device in args.mujoco_egl_devices]
        if any(not device.isdigit() for device in args.mujoco_egl_devices):
            raise ValueError("--mujoco-egl-devices values must be non-negative integer IDs")
    if args.semantic_model_env_file is None and args.model_endpoints:
        inherited = os.environ.get("SEMANTIC_MODEL_ENV_FILE")
        args.semantic_model_env_file = (
            Path(inherited).expanduser().resolve()
            if inherited
            else (REPO_ROOT / ".env").resolve()
        )
    if args.semantic_model_env_file is not None and not args.semantic_model_env_file.is_file():
        raise FileNotFoundError(args.semantic_model_env_file)


def failed_plan_result(plan: EpisodePlan, args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    """Persist one unexpected wrapper failure without stopping the other workers."""

    result = base_summary(plan, args)
    result.update(
        {
            "completed": False,
            "runner_exit_code": 127,
            "exit_code": 127,
            "error": f"wrapper exception: {type(exc).__name__}: {exc}",
            "resumed": False,
        }
    )
    plan.task_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(plan.task_dir / "batch_task_summary.json", result)
    return result


def run_worker(
    worker_id: int,
    plans: list[EpisodePlan],
    args: argparse.Namespace,
    telemetry: BatchResourceTelemetry | None,
) -> list[dict[str, Any]]:
    """Run a worker's plans serially so its ROS master port is never shared."""

    del worker_id  # The worker identity is already frozen into every plan.
    results: list[dict[str, Any]] = []
    for plan in plans:
        try:
            results.append(run_episode(plan, args, telemetry))
        except Exception as exc:  # Preserve the rest of an overnight shard.
            results.append(failed_plan_result(plan, args, exc))
    return results


def main() -> int:
    args = parse_args()
    validate_args(args)
    plans = [
        EpisodePlan(
            ordinal=ordinal,
            worker_id=ordinal % args.workers,
            episode_index=episode_index,
            # A logical worker owns this port and executes its shard serially.
            # Assigning one port per *episode* is unsafe once a thread pool
            # starts later tasks out of order.
            master_port=args.base_master_port + (ordinal % args.workers),
            task_dir=args.output_dir / f"episode_{episode_index:04d}",
        )
        for ordinal, episode_index in enumerate(args.episode_indices)
    ]
    if args.dry_run:
        rows = [run_episode(plan, args) for plan in plans]
        print(json.dumps({"plans": rows}, ensure_ascii=False, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "batch_manifest.json",
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "created_at": utc_now(),
            "config": {
                "runner": str(args.runner),
                "runner_shell": args.runner_shell,
                "benchmark": str(args.benchmark),
                "workers": args.workers,
                "base_master_port": args.base_master_port,
                "max_steps": args.max_steps,
                "scene_timeout_s": args.scene_timeout_s,
                "fast_eval": bool(args.fast_eval),
                "recording_enabled": not bool(args.fast_eval),
                "model_endpoints": args.model_endpoints,
                "mujoco_egl_devices": args.mujoco_egl_devices,
                "semantic_model_env_file": (
                    None if args.semantic_model_env_file is None else str(args.semantic_model_env_file)
                ),
                "resource_telemetry": bool(args.resource_telemetry),
                "resource_sample_interval_s": args.resource_sample_interval_s,
            },
            "plans": [plan_to_dict(plan, args) for plan in plans],
        },
    )

    results: list[dict[str, Any]] = []
    shards = [
        [plan for plan in plans if plan.worker_id == worker_id]
        for worker_id in range(args.workers)
    ]
    telemetry = (
        BatchResourceTelemetry(args.output_dir, args.resource_sample_interval_s)
        if args.resource_telemetry
        else None
    )
    if telemetry is not None:
        telemetry.start()
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(plans))) as executor:
            futures = {
                executor.submit(run_worker, worker_id, shard, args, telemetry): worker_id
                for worker_id, shard in enumerate(shards)
                if shard
            }
            for future in as_completed(futures):
                worker_results = future.result()
                results.extend(worker_results)
                write_summary(args, plans, results)
    finally:
        resource_summary = telemetry.stop() if telemetry is not None else None

    write_summary(args, plans, results, resource_summary)
    failure_count = sum(not bool(result.get("completed")) for result in results)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "episode_count": len(results),
                "failure_count": failure_count,
                "summary": str(args.output_dir / "summary.json"),
                "resource_telemetry": (
                    None if resource_summary is None else resource_summary.get("resource_telemetry_path")
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if args.allow_failures or failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
