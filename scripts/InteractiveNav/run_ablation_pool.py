#!/usr/bin/env python3
"""Run selected Full/ablation variants in one work-conserving pool."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import run_benchmark_eval as baseline
import run_interactive_nav_v3_ros_eval_batch as batch
from ablations import DESIGN_REVISION, VARIANTS
from ablations.launch import render_artifacts, runner_path, write_artifacts, artifact_digests
from ablations.preflight import check_model_endpoints


def select_scenes(source, count, *, episode_start=2000, episode_stride=1,
                  selection_mode="rank"):
    """Select benchmark rows either by historical ranking or fixed stride.

    The stride mode is used for the requested mixed-1000 diagnostic sample:
    the mixed block begins at episode 2000, so stride 10 yields 2000..2990.
    """
    if episode_stride < 1 or episode_start < 0:
        raise ValueError("episode_start and episode_stride must be positive")
    if selection_mode == "explicit":
        rows = [r for r in source["episodes"]
                if r.get("completed", True) and r.get("scoring_eligible", True)]
        indices = [r["episode_index"] for r in rows]
        if len(indices) != count or len(indices) != len(set(indices)):
            raise ValueError("explicit selection must contain exactly count unique eligible episodes")
        if any(index < 2000 or index >= 3000 for index in indices):
            raise ValueError("explicit selection must contain mixed episode indices 2000..2999")
        return sorted(rows, key=lambda row: row["episode_index"])
    wanted = {episode_start + episode_stride * i for i in range(count)}
    rows = [r for r in source["episodes"]
            if r.get("episode_index") in wanted
            and r.get("completed", True) and r.get("scoring_eligible", True)]
    if selection_mode == "stride":
        return sorted(rows, key=lambda row: row["episode_index"])
    if selection_mode != "rank":
        raise ValueError(selection_mode)
    def rank(row):
        correct = row.get("correct_interaction_action_count", 0)
        attempts = row.get("interaction_action_count", 0)
        return (-bool(row.get("success")), -bool(row.get("required_interaction_success")),
                -bool(row.get("task_success")), -correct / max(1, attempts), -correct,
                row["episode_index"])
    return sorted(rows, key=rank)[:count]


def drain_pool(jobs, workers, execute, stop):
    """A fixed worker/ROS port immediately takes the next job of any variant."""
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    def worker(worker_id):
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                execute(worker_id, job)
            finally:
                pending.task_done()
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [executor.submit(worker, i) for i in range(min(workers, len(jobs)))]
        for future in futures:
            future.result()


def native_args(config, output, runner):
    command, _ = baseline.build_command(config, output)
    previous = sys.argv
    try:
        sys.argv = [command[2], *command[3:], "--runner", str(runner)]
        args = batch.parse_args()
    finally:
        sys.argv = previous
    batch.validate_args(args)
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--base-master-port", type=int, default=18000)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--episode-start", type=int, default=2000)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--paper-cost-budget", type=float)
    parser.add_argument("--selection-mode", choices=("rank", "stride", "explicit"), default="rank")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    output = cli.output_dir.resolve()
    if not output.is_relative_to(Path("/home/ldl")):
        parser.error("output must be under /home/ldl")
    if cli.workers < 1 or cli.scenes < 1 or cli.base_master_port + cli.workers > 65536:
        parser.error("invalid worker/scene count or port range")
    source = json.loads(cli.selection_source.read_text())
    if isinstance(source, list):
        # A benchmark JSON is also a valid deterministic selection source.  Its
        # list position is the canonical episode index; no historical scores
        # are consulted in fixed-stride mode.
        source = {"episodes": [{"episode_index": i} for i in range(len(source))]}
    selected = select_scenes(source, cli.scenes, episode_start=cli.episode_start,
                             episode_stride=cli.episode_stride,
                             selection_mode=cli.selection_mode)
    if len(selected) != cli.scenes:
        parser.error("not enough eligible mixed scenes")
    config = json.loads(cli.config.read_text())
    config.update(workers=cli.workers, base_master_port=cli.base_master_port,
                  episode_indices=sorted(r["episode_index"] for r in selected),
                  recording=False, max_steps=2000, step_budget_mode="dynamic", resume=False)
    config["paper_cost_budget"] = float(
        cli.paper_cost_budget if cli.paper_cost_budget is not None
        else config.get("paper_cost_budget", 30.0)
    )
    if not math.isfinite(config["paper_cost_budget"]) or config["paper_cost_budget"] <= 0:
        parser.error("paper_cost_budget must be finite and positive")
    variants = tuple(dict.fromkeys(cli.variants))
    # Launch historically long scenes first; interleave variants instead of
    # reserving fixed worker quotas for each ablation.
    ordered = sorted(selected, key=lambda r: -float(r.get("elapsed_sec") or 0))
    jobs = [(variant, row["episode_index"]) for row in ordered for variant in variants]
    manifest = {"design_revision": DESIGN_REVISION, "created_at": batch.utc_now(),
                "config": config, "selection_source": str(cli.selection_source.resolve()),
                "selection_rule": ("fixed mixed stride" if cli.selection_mode == "stride"
                                   else "explicit mixed indices from selection source" if cli.selection_mode == "explicit"
                                   else "eligible mixed; success, required interaction, task success, correct/attempt, correct count, index"),
                "selected_full_rows": selected, "jobs": jobs, "full_launched": "full" in variants,
                "scheduling": "one shared queue; fixed port per slot; immediate refill across variants",
                "m1_refresh_profile": config.get("m1_refresh_profile", "continuous")}
    artifacts_by_variant = {}
    for variant in variants:
        directory = output / variant
        artifacts = render_artifacts(baseline.REPO, directory / "ablation", variant,
                                     config["python_bin"], manifest["m1_refresh_profile"])
        runner = artifacts.get(directory / "ablation/runner.sh")
        if runner is not None:
            subprocess.run(["bash", "-n"], input=runner, text=True, check=True)
        artifacts_by_variant[variant] = artifacts
    if cli.dry_run:
        print(json.dumps({k: v for k, v in manifest.items() if k != "selected_full_rows"}, indent=2))
        return 0
    baseline.check_mujoco_gpu_inventory(config)
    check_model_endpoints(config)
    for port in range(cli.base_master_port, cli.base_master_port + cli.workers):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    output.mkdir(parents=True, exist_ok=False)
    args_by_variant, plans, results = {}, {}, {v: [] for v in variants}
    launch_gate = batch.SceneLaunchGate(float(config.get("scene_start_interval_s", 10.0)))
    for variant in variants:
        directory = output / variant
        artifacts = artifacts_by_variant[variant]
        write_artifacts(artifacts)
        runner = runner_path(baseline.REPO, directory / "ablation", variant,
                             manifest["m1_refresh_profile"])
        args_by_variant[variant] = native_args(config, directory, runner)
        args_by_variant[variant]._scene_launch_gate = launch_gate
        batch.atomic_json(directory / "ablation_manifest.json", {
            "variant": variant, "design_revision": DESIGN_REVISION,
            "m1_refresh_profile": manifest["m1_refresh_profile"], "config": config,
            "artifact_sha256": artifact_digests(artifacts), "shared_pool_manifest": str(output / "pool_manifest.json")})
    for ordinal, (variant, index) in enumerate(jobs):
        plans[(variant, index)] = batch.EpisodePlan(ordinal, -1, index, 0,
                                                   output / variant / f"episode_{index:04d}")
    manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["git_status"] = subprocess.check_output(["git", "status", "--short"], text=True)
    batch.atomic_json(output / "pool_manifest.json", manifest)
    for runtime_dir in (output / "tmp", output / "cache"):
        runtime_dir.mkdir()
    os.environ.update({baseline.OWNER_KEY: str(output), "MIN_STEPS": str(config.get("min_steps", 200)),
                       "PAPER_COST_BUDGET": str(config["paper_cost_budget"]),
                       "TMPDIR": str(output / "tmp"), "XDG_CACHE_HOME": str(output / "cache"),
                       "HF_HOME": "/home/ldl/.cache/huggingface",
                       "TORCH_HOME": "/home/ldl/.cache/torch",
                       "NLTK_DATA": "/home/ldl/nltk_data", "PYTHONDONTWRITEBYTECODE": "1"})
    stop, lock = threading.Event(), threading.Lock()
    finished = queue.Queue()
    telemetry = batch.BatchResourceTelemetry(output, 5.)
    active = {}
    def request_stop(*_):
        stop.set()
        baseline.cleanup(str(output))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    def event(kind, **fields):
        with lock:
            with (output / "scheduling.jsonl").open("a") as handle:
                handle.write(json.dumps({"event": kind, "time": batch.utc_now(), **fields}) + "\n")
    def execute(worker_id, job):
        variant, index = job
        plan = replace(plans[job], worker_id=worker_id, master_port=cli.base_master_port + worker_id)
        with lock:
            plans[job] = plan
            active[worker_id] = job
        event("start", worker_id=worker_id, variant=variant, episode_index=index)
        try:
            row = batch.run_episode(plan, args_by_variant[variant], telemetry)
        except Exception as exc:
            row = batch.failed_plan_result(plan, args_by_variant[variant], exc)
        row["ablation_variant"] = variant
        finished.put((variant, row))
        with lock:
            active.pop(worker_id, None)
        event("finish", worker_id=worker_id, variant=variant, episode_index=index,
              completed=row.get("completed"), success=row.get("success"))
    error = []
    def run():
        try:
            drain_pool(jobs, cli.workers, execute, stop)
        except BaseException as exc:
            error.append(repr(exc))
    thread = threading.Thread(target=run, name="shared-ablation-pool")
    started = time.monotonic()
    telemetry.start()
    thread.start()
    try:
        while thread.is_alive() or not finished.empty():
            try:
                variant, row = finished.get(timeout=10.)
                results[variant].append(row)
                with lock:
                    variant_plans = [p for (v, _), p in plans.items() if v == variant]
                batch.write_summary(args_by_variant[variant], variant_plans, results[variant])
            except queue.Empty:
                pass
            with lock:
                running = dict(active)
            count = sum(map(len, results.values()))
            state = {"reported": count, "planned": len(jobs), "active": running,
                     "queued": len(jobs) - count - len(running), "elapsed_sec": time.monotonic() - started,
                     "per_variant": {v: {"reported": len(rows), "success": sum(bool(r.get("success")) for r in rows)}
                                     for v, rows in results.items()}}
            batch.atomic_json(output / "pool_status.json", state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
        thread.join()
    finally:
        resource = telemetry.stop()
        baseline.cleanup(str(output))
    for variant in variants:
        variant_plans = [p for (v, _), p in plans.items() if v == variant]
        aggregate = batch.write_summary(args_by_variant[variant], variant_plans, results[variant], resource)
        batch.atomic_json(output / variant / "aggregate_metrics.json", aggregate)
    batch.atomic_json(output / "pool_result.json", {"elapsed_sec": time.monotonic() - started,
                      "errors": error, "stopped": stop.is_set(), "results": results})
    return 130 if stop.is_set() else int(bool(error) or sum(map(len, results.values())) != len(jobs))


if __name__ == "__main__":
    raise SystemExit(main())
