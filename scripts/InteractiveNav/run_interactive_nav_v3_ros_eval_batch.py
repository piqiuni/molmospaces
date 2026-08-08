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
      --output-dir /home/ldl/outputs/interactive-nav/v3_mixed_10x \
      --episode-indices 0 1 2 3 4 5 6 7 8 9 \
      --workers 10 --base-master-port 12600 --max-steps 1500 \
      --scene-timeout-s 7200 --allow-failures
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
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "scripts" / "InteractiveNav" / "run_interactive_nav_v3_ros_eval_test.zsh"
SUMMARY_SCHEMA_VERSION = "interactive_nav_v3_ros_batch_v1"


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


def existing_completed_summary(task_dir: Path) -> dict[str, Any] | None:
    summary = read_json(task_dir / "batch_task_summary.json")
    if not summary.get("completed"):
        return None
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


def plan_to_dict(plan: EpisodePlan) -> dict[str, Any]:
    return {
        "ordinal": plan.ordinal,
        "worker_id": plan.worker_id,
        "episode_index": plan.episode_index,
        "ros_master_uri": plan.ros_master_uri,
        "output_dir": str(plan.task_dir),
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
            "ROS_MASTER_URI": plan.ros_master_uri,
            "MPLCONFIGDIR": str(attempt_dir / "mplconfig"),
            "TMPDIR": str(temporary_root),
            "XDG_CACHE_HOME": str(cache_root),
            "PYTHONUNBUFFERED": "1",
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
    return environment


def base_summary(plan: EpisodePlan, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "episode_index": plan.episode_index,
        "worker_id": plan.worker_id,
        "ros_master_uri": plan.ros_master_uri,
        "output_dir": str(plan.task_dir),
        "runner": str(args.runner),
        "max_steps": args.max_steps,
    }


def run_episode(plan: EpisodePlan, args: argparse.Namespace) -> dict[str, Any]:
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
            }
        )
        return result

    previous = existing_completed_summary(task_dir) if args.resume else None
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

    elapsed_sec = time.monotonic() - started
    paths = artifact_paths(attempt_dir, plan.episode_index)
    episode_result, document_status = load_episode_result(paths["episode_result"])
    six_panel = paths["six_panel_video"]
    topdown = paths["topdown"]
    completed = bool(
        runner_exit_code == 0
        and document_status == "complete"
        and episode_result.get("status") == "complete"
        and video_is_valid(six_panel)
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
            "six_panel_video": str(six_panel),
            "six_panel_video_bytes": six_panel.stat().st_size if video_is_valid(six_panel) else 0,
            "topdown": None if topdown is None else str(topdown),
            "topdown_exists": topdown is not None and topdown.is_file() and topdown.stat().st_size > 0,
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


def write_summary(args: argparse.Namespace, plans: list[EpisodePlan], results: list[dict[str, Any]]) -> None:
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
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "config": {
            "output_dir": str(args.output_dir),
            "runner": str(args.runner),
            "runner_shell": args.runner_shell,
            "workers": args.workers,
            "base_master_port": args.base_master_port,
            "max_steps": args.max_steps,
            "scene_timeout_s": args.scene_timeout_s,
            "resume": args.resume,
        },
        "plans": [plan_to_dict(plan) for plan in plans],
        "aggregate": aggregate,
        "episodes": ordered_results,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "aggregate_metrics.json", aggregate)
    fields = [
        "episode_index",
        "worker_id",
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
    args.runner = args.runner.expanduser().resolve()
    if args.python_bin is not None:
        args.python_bin = args.python_bin.expanduser().resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.scene_timeout_s <= 0:
        raise ValueError("--scene-timeout-s must be positive")
    if not 1 <= args.base_master_port <= 65535:
        raise ValueError("--base-master-port must be in [1, 65535]")
    if any(index < 0 for index in args.episode_indices):
        raise ValueError("--episode-indices must be non-negative")
    if len(set(args.episode_indices)) != len(args.episode_indices):
        raise ValueError("--episode-indices must not contain duplicates")
    if args.base_master_port + len(args.episode_indices) - 1 > 65535:
        raise ValueError("base master port range exceeds 65535")
    if not args.runner.is_file():
        raise FileNotFoundError(args.runner)
    if args.python_bin is not None and not args.python_bin.is_file():
        raise FileNotFoundError(args.python_bin)


def main() -> int:
    args = parse_args()
    validate_args(args)
    plans = [
        EpisodePlan(
            ordinal=ordinal,
            worker_id=ordinal % args.workers,
            episode_index=episode_index,
            master_port=args.base_master_port + ordinal,
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
                "workers": args.workers,
                "base_master_port": args.base_master_port,
                "max_steps": args.max_steps,
                "scene_timeout_s": args.scene_timeout_s,
            },
            "plans": [plan_to_dict(plan) for plan in plans],
        },
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(plans))) as executor:
        futures = {executor.submit(run_episode, plan, args): plan for plan in plans}
        for future in as_completed(futures):
            plan = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Keep other overnight workers alive.
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
            results.append(result)
            write_summary(args, plans, results)

    write_summary(args, plans, results)
    failure_count = sum(not bool(result.get("completed")) for result in results)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "episode_count": len(results),
                "failure_count": failure_count,
                "summary": str(args.output_dir / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if args.allow_failures or failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
