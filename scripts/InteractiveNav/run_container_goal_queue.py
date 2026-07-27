#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any

from container_scene_probe import close_context, load_scene_context
from run_semantic_interaction_exploration_batch import (
    monitor_scene_memory,
    read_json,
    summarize_scene_memory,
    terminate_group,
)
from runtime_target_selection import select_far_container_target


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNNER = SCRIPT_DIR / "run_house7_semantic_exploration_ros_test.zsh"
DEFAULT_DECISION_OVERRIDE = (
    SCRIPT_DIR
    / "configs"
    / "semantic_decision"
    / "full_mllm_object_goal_runtime.yaml"
)
DEFAULT_MAPPING_OVERRIDE = (
    SCRIPT_DIR / "configs" / "semantic_decision" / "full_mllm_mapping.yaml"
)
STOP = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover objects inside articulated containers, queue one goal per scene, "
            "and start navigation workers after a scene-scan warmup."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--house-inds", nargs="+", type=int)
    parser.add_argument("--house-start", type=int, default=0)
    parser.add_argument("--house-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--warmup-scenes", type=int, default=5)
    parser.add_argument("--base-master-port", type=int, default=12840)
    parser.add_argument("--task-horizon", type=int, default=1000)
    parser.add_argument("--scene-timeout-s", type=float, default=1200.0)
    parser.add_argument("--memory-sample-interval-s", type=float, default=2.0)
    parser.add_argument("--selection-top-k", type=int, default=3)
    parser.add_argument("--gt-step-interval", type=int, default=5)
    parser.add_argument("--gt-max-distance-m", type=float, default=6.0)
    parser.add_argument("--gt-min-visible-pixels", type=int, default=16)
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--robot", default="rby1", choices=["rby1", "droid", "rum"])
    parser.add_argument("--variant", default="base")
    parser.add_argument("--method", default="semantic_interaction_object_goal")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--decision-override", type=Path, default=DEFAULT_DECISION_OVERRIDE)
    parser.add_argument("--mapping-override", type=Path, default=DEFAULT_MAPPING_OVERRIDE)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--attribute-model-name", default="qwen3.6-35b-a3b")
    parser.add_argument("--recording", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reveal-container-context", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def hidden_target_context(context: dict[str, Any]) -> dict[str, Any]:
    public_context = dict(context)
    for key in (
        "target_container_name",
        "target_container_labels",
        "target_container_source_object_name",
        "target_container_instance_id",
        "target_container_requires_interaction",
    ):
        public_context.pop(key, None)
    public_context["selection_mode"] = "queued_hidden_container_object"
    public_context["require_interaction"] = False
    public_context["completion_requires_visibility"] = True
    return public_context


def discover_goal(args: argparse.Namespace, house_ind: int) -> dict[str, Any]:
    scan_args = argparse.Namespace(
        seed=int(house_ind),
        scene_dataset=args.scene_dataset,
        data_split=args.data_split,
        robot=args.robot,
        variant=args.variant,
        output_dir=args.output_dir,
    )
    started = time.monotonic()
    context = load_scene_context(scan_args, house_ind)
    try:
        target_context, selection = select_far_container_target(
            SimpleNamespace(env=context.env),
            selection_seed=int(house_ind),
            top_k=args.selection_top_k,
        )
    finally:
        close_context(context)
    selection = dict(selection)
    selection["house_ind"] = int(house_ind)
    selection["scan_elapsed_sec"] = time.monotonic() - started
    selection["target_context"] = (
        dict(target_context)
        if args.reveal_container_context
        else hidden_target_context(target_context)
    )
    return selection


def task_id(selection: dict[str, Any]) -> str:
    return f"house_{int(selection['house_ind']):04d}_goal_00"


def run_task(
    worker_id: int,
    selection: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    house_ind = int(selection["house_ind"])
    run_id = task_id(selection)
    scene_dir = args.output_dir / "runs" / run_id
    result_path = scene_dir / "semantic_exploration_result.json"
    if args.resume and result_path.exists():
        result = read_json(result_path)
        if result:
            return {
                **result,
                "task_id": run_id,
                "worker_id": worker_id,
                "resumed": True,
                "exit_code": 0,
            }

    scene_dir.mkdir(parents=True, exist_ok=True)
    selection_input_path = scene_dir / "target_selection_input.json"
    write_json(selection_input_path, selection)
    environment = os.environ.copy()
    environment.update(
        {
            "METHOD": args.method,
            "HOUSE_IND": str(house_ind),
            "ROUTE_ID": run_id,
            "USE_FIXED_ROUTE": "false",
            "SCENE_SEED": str(house_ind),
            "TASK_HORIZON": str(args.task_horizon),
            "SIM_TIMEOUT_S": str(int(args.scene_timeout_s)),
            "ROS_MASTER_URI": f"http://127.0.0.1:{args.base_master_port + worker_id}",
            "ROUTE_NAV_CONFIG": "",
            "INITIAL_DOOR_STATE": "closed",
            "FORCE_CLOSE_CONTAINERS": "true",
            "RUNTIME_TARGET_MODE": "fixed_container_object",
            "RUNTIME_TARGET_SELECTION_INPUT_PATH": str(selection_input_path),
            "SEMANTIC_DECISION_OVERRIDE": str(args.decision_override),
            "SEMANTIC_MAPPING_OVERRIDE": str(args.mapping_override),
            "ENABLE_ATTRIBUTE_INFERENCE": "true",
            "SEMANTIC_ATTRIBUTE_MODEL_NAME": args.attribute_model_name,
            "SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S": "8.0",
            "SEMANTIC_MODEL_TIMEOUT_S": "3.0",
            "ENABLE_RECORDING": "true" if args.recording else "false",
            "VIDEO_PANEL_WIDTH_PX": "480",
            "VIDEO_ENCODER_PRESET": "ultrafast",
            "INTERACTION_EXECUTION_MODE": "fast",
            "DRAWER_EXECUTION_MODE": "fast",
            "DRAWER_OBSERVATION_STEPS": "6",
            "GT_STEP_INTERVAL": str(max(1, int(args.gt_step_interval))),
            "GT_MAX_DISTANCE_M": str(max(0.0, float(args.gt_max_distance_m))),
            "GT_MIN_VISIBLE_PIXELS": str(max(1, int(args.gt_min_visible_pixels))),
            "SKIP_COVERAGE": "true",
            "CLEAN_INTERMEDIATE": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if args.env_file is not None:
        environment["SEMANTIC_MODEL_ENV_FILE"] = str(args.env_file)
    command = ["zsh", str(args.runner), str(scene_dir), run_id]
    if args.dry_run:
        return {
            "task_id": run_id,
            "worker_id": worker_id,
            "house_ind": house_ind,
            "selection": selection,
            "command": command,
            "ros_master_uri": environment["ROS_MASTER_URI"],
            "dry_run": True,
            "exit_code": 0,
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
    result.update(
        {
            "task_id": run_id,
            "worker_id": worker_id,
            "house_ind": house_ind,
            "target_name": selection.get("target_name"),
            "target_category": selection.get("target_category"),
            "container_name": selection.get("container_name"),
            "container_category": selection.get("container_category"),
            "output_dir": str(scene_dir),
            "ros_master_uri": environment["ROS_MASTER_URI"],
            "exit_code": exit_code,
            "elapsed_sec": time.monotonic() - started,
            "resumed": False,
            **summarize_scene_memory(scene_dir / "memory_by_step.csv"),
        }
    )
    write_json(scene_dir / "batch_task_summary.json", result)
    return result


def write_experiment_summary(
    output_dir: Path,
    scan_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    write_json(output_dir / "scan_results.json", scan_rows)
    write_json(output_dir / "results.json", results)
    write_json(output_dir / "failures.json", failures)
    write_json(output_dir / "queue_state.json", state)
    aggregate = {
        "scanned_scene_count": int(state.get("scanned_scene_count", 0) or 0),
        "discovered_goal_count": len(scan_rows),
        "completed_task_count": len(results),
        "failed_task_count": len(
            [row for row in results if int(row.get("exit_code", 1)) != 0]
        ),
        "target_goal_success_count": sum(
            bool(row.get("target_goal_success")) for row in results
        ),
        "target_goal_success_rate": (
            sum(bool(row.get("target_goal_success")) for row in results) / len(results)
            if results
            else 0.0
        ),
        "container_interaction_success_count": sum(
            bool(row.get("target_container_interaction_success")) for row in results
        ),
        "workers": state.get("worker_count"),
        "workers_started_after_scene_count": state.get("workers_started_after_scene_count"),
    }
    write_json(output_dir / "aggregate_metrics.json", aggregate)
    fields = [
        "task_id",
        "worker_id",
        "house_ind",
        "target_category",
        "container_category",
        "exit_code",
        "elapsed_sec",
        "target_goal_success",
        "target_container_interaction_success",
        "interaction_count",
        "decision_count",
        "mllm_request_count",
        "valid_step_video",
        "video",
        "output_dir",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: int(row.get("house_ind", -1))))


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.runner = args.runner.expanduser().resolve()
    args.decision_override = args.decision_override.expanduser().resolve()
    args.mapping_override = args.mapping_override.expanduser().resolve()
    if args.env_file is not None:
        args.env_file = args.env_file.expanduser().resolve()
    houses = (
        list(dict.fromkeys(args.house_inds))
        if args.house_inds
        else list(range(args.house_start, args.house_start + args.house_count))
    )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.warmup_scenes < 1:
        raise ValueError("--warmup-scenes must be positive")
    if not houses:
        raise ValueError("No houses requested")
    for path in (args.runner, args.decision_override, args.mapping_override):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.env_file is not None and not args.env_file.exists():
        raise FileNotFoundError(args.env_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
    lock = threading.Lock()
    previous_state = read_json(args.output_dir / "queue_state.json") if args.resume else {}
    scan_rows: list[dict[str, Any]] = (
        list(read_json(args.output_dir / "scan_results.json") or []) if args.resume else []
    )
    results: list[dict[str, Any]] = (
        list(read_json(args.output_dir / "results.json") or []) if args.resume else []
    )
    failures: list[dict[str, Any]] = (
        list(read_json(args.output_dir / "failures.json") or []) if args.resume else []
    )
    previous_requested_houses = list(previous_state.get("requested_houses") or [])
    previous_scanned_count = min(
        len(previous_requested_houses),
        max(0, int(previous_state.get("scanned_scene_count", 0) or 0)),
    )
    scanned_houses = set(previous_requested_houses[:previous_scanned_count])
    completed_task_ids = {str(row.get("task_id") or "") for row in results}
    for selection in scan_rows:
        if task_id(selection) not in completed_task_ids:
            task_queue.put(selection)
    worker_threads: list[threading.Thread] = []
    state: dict[str, Any] = {
        "status": "scanning",
        "worker_count": args.workers,
        "workers_started": False,
        "workers_started_after_scene_count": None,
        "requested_houses": houses,
        "scanned_scene_count": previous_scanned_count,
        "discovered_goal_count": len(scan_rows),
        "queued_task_count": len(scan_rows),
        "running_tasks": {},
        "completed_task_count": len(results),
    }

    def persist() -> None:
        write_experiment_summary(args.output_dir, scan_rows, results, failures, state)

    def worker_loop(worker_id: int) -> None:
        while True:
            selection = task_queue.get()
            if selection is STOP:
                task_queue.task_done()
                return
            assert isinstance(selection, dict)
            run_id = task_id(selection)
            with lock:
                state["running_tasks"][str(worker_id)] = run_id
                persist()
            print(f"worker {worker_id} starting {run_id}", flush=True)
            try:
                result = run_task(worker_id, selection, args)
            except Exception as exc:
                result = {
                    "task_id": run_id,
                    "worker_id": worker_id,
                    "house_ind": int(selection["house_ind"]),
                    "target_category": selection.get("target_category"),
                    "container_category": selection.get("container_category"),
                    "exit_code": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with lock:
                results.append(result)
                state["running_tasks"].pop(str(worker_id), None)
                state["completed_task_count"] = len(results)
                persist()
            print(
                f"worker {worker_id} finished {run_id}: "
                f"exit={result.get('exit_code')} success={result.get('target_goal_success')}",
                flush=True,
            )
            task_queue.task_done()

    def start_workers(scanned_count: int) -> None:
        if worker_threads:
            return
        state["workers_started"] = True
        state["workers_started_after_scene_count"] = scanned_count
        state["status"] = "scanning_and_running"
        for worker_id in range(args.workers):
            thread = threading.Thread(target=worker_loop, args=(worker_id,), daemon=False)
            thread.start()
            worker_threads.append(thread)
        persist()
        print(
            f"started {args.workers} workers after scanning {scanned_count} scenes",
            flush=True,
        )

    persist()
    for scan_index, house_ind in enumerate(houses, start=1):
        if house_ind in scanned_houses:
            print(f"house {house_ind} already scanned; resuming", flush=True)
            if scan_index >= min(args.warmup_scenes, len(houses)):
                with lock:
                    start_workers(scan_index)
            continue
        print(f"scanning house {house_ind} ({scan_index}/{len(houses)})", flush=True)
        try:
            selection = discover_goal(args, house_ind)
        except Exception as exc:
            with lock:
                failures.append(
                    {
                        "stage": "scan",
                        "house_ind": house_ind,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                state["scanned_scene_count"] = scan_index
                persist()
            print(f"house {house_ind} scan failed: {exc}", flush=True)
        else:
            with lock:
                state["scanned_scene_count"] = scan_index
                if bool((selection.get("target_context") or {}).get("enabled")):
                    scan_rows.append(selection)
                    task_queue.put(selection)
                    state["discovered_goal_count"] = len(scan_rows)
                    state["queued_task_count"] = len(scan_rows)
                    print(
                        f"queued {task_id(selection)}: {selection.get('target_category')} "
                        f"inside {selection.get('container_category')}",
                        flush=True,
                    )
                else:
                    print(f"house {house_ind}: no container goal", flush=True)
                persist()
        if scan_index >= min(args.warmup_scenes, len(houses)):
            with lock:
                start_workers(scan_index)

    with lock:
        if not worker_threads:
            start_workers(len(houses))
        state["status"] = "draining_queue"
        persist()
    for _ in worker_threads:
        task_queue.put(STOP)
    for thread in worker_threads:
        thread.join()
    with lock:
        state["status"] = "complete"
        persist()

    failed_results = [row for row in results if int(row.get("exit_code", 1)) != 0]
    print(
        json.dumps(
            {
                "scanned_scene_count": state["scanned_scene_count"],
                "discovered_goal_count": len(scan_rows),
                "completed_task_count": len(results),
                "failure_count": len(failures) + len(failed_results),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if args.allow_failures or (not failures and not failed_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
