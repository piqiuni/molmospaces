#!/usr/bin/env python3
"""Run isolated occupancy-only exploration episodes and aggregate coverage results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

from run_parallel_ros_episodes import (
    pids_for_master,
    port_available,
    process_usage,
    read_gpu_samples,
    read_host_sample,
    terminate_group,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETUP = REPO_ROOT / "Interactive-Nav-SG-nav" / "devel" / "setup.zsh"
DEFAULT_RECORDER = (
    REPO_ROOT
    / "Interactive-Nav-SG-nav"
    / "src"
    / "explore_py_pkg"
    / "scripts"
    / "record_explore_debug.py"
)
DEFAULT_COVERAGE = REPO_ROOT / "scripts" / "InteractiveNav" / "evaluate_exploration_coverage.py"
DEFAULT_OFFLINE_VIDEO_BUILDER = (
    REPO_ROOT / "scripts" / "InteractiveNav" / "build_exploration_video_offline.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-start", type=int, default=0)
    parser.add_argument("--house-count", type=int, default=100)
    parser.add_argument(
        "--repeat-house-ind",
        type=int,
        help="Run house-count independent episodes of the same house index.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-master-port", type=int, default=12411)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--task-horizon", type=int, default=300)
    parser.add_argument("--scene-timeout-s", type=float, default=900.0)
    parser.add_argument("--recorder-shutdown-grace-s", type=float, default=300.0)
    parser.add_argument("--coverage-timeout-s", type=float, default=180.0)
    parser.add_argument("--resource-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--min-memory-available-mb",
        type=float,
        default=8192.0,
        help="Abort all workers when host available memory falls below this value; use 0 to disable.",
    )
    parser.add_argument("--setup-file", type=Path, default=DEFAULT_SETUP)
    parser.add_argument("--recorder-script", type=Path, default=DEFAULT_RECORDER)
    parser.add_argument("--coverage-script", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--offline-video-builder", type=Path, default=DEFAULT_OFFLINE_VIDEO_BUILDER)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--action-timeout-s", type=float, default=0.5)
    parser.add_argument("--explore-config-override-file", type=Path)
    parser.add_argument("--nav-config-override-file", type=Path)
    parser.add_argument(
        "--base-local-planner",
        default="dwa_local_planner/DWAPlannerROS",
        help="nav_core local planner plugin class passed to move_base.",
    )
    parser.add_argument(
        "--local-planner-namespace",
        default="DWAPlannerROS",
        help="move_base local planner namespace used by the debug recorder.",
    )
    parser.add_argument("--max-scene-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shell_command(setup_file: Path, command: list[str]) -> list[str]:
    return [
        "/bin/zsh",
        "-lc",
        f"source {shlex.quote(str(setup_file))} && exec {shlex.join(command)}",
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def load_scene_metrics(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.glob("worker_*/scene_*/scene_metrics.json")):
        row = read_json(path)
        if row:
            rows.append(row)
    return rows


def iter_events(path: Path):
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def ensure_h264(debug_dir: Path, log_path: Path) -> tuple[str, int]:
    video_path = debug_dir / "videos" / "first_person.mp4"
    full_step_frame_dir = debug_dir / "videos" / "full_step_composite_frames"
    frame_dir = full_step_frame_dir if full_step_frame_dir.exists() else debug_dir / "videos" / "composite_frames"
    frames = sorted(frame_dir.glob("*.png")) if frame_dir.exists() else []
    if video_path.exists() and video_path.stat().st_size > 0:
        return str(video_path), len(frames)
    if not frames:
        return "", 0
    command = [
        "ffmpeg", "-y", "-framerate", "15", "-pattern_type", "glob",
        "-i", str(frame_dir / "*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-movflags", "+faststart", str(video_path),
    ]
    with log_path.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode != 0 or not video_path.exists():
        return "", len(frames)
    return str(video_path), len(frames)


def build_full_step_video(
    builder: Path,
    scene_dir: Path,
    debug_dir: Path,
    log_path: Path,
) -> tuple[int, dict[str, Any]]:
    command = [
        sys.executable,
        str(builder),
        "--scene-dir",
        str(scene_dir),
        "--debug-dir",
        str(debug_dir),
        "--fps",
        "15",
    ]
    with log_path.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    return result.returncode, read_json(debug_dir / "full_step_video_summary.json")


def subgoal_repeat_metrics(subgoal_path: Path) -> dict[str, Any]:
    if not subgoal_path.exists():
        return {"subgoal_count": 0}
    with subgoal_path.open() as stream:
        rows = list(csv.DictReader(stream))
    points = [(float(row["x"]), float(row["y"])) for row in rows]
    repeated_05 = 0
    repeated_075 = 0
    consecutive_05 = 0
    for index, point in enumerate(points):
        if index == 0:
            continue
        minimum = min(math.dist(point, previous) for previous in points[:index])
        repeated_05 += minimum <= 0.5 + 1e-6
        repeated_075 += minimum <= 0.75 + 1e-6
        consecutive_05 += math.dist(point, points[index - 1]) <= 0.5 + 1e-6
    denominator = max(1, len(points) - 1)
    return {
        "subgoal_count": len(points),
        "viewpoint_repeat_le_0_5m": repeated_05,
        "viewpoint_repeat_le_0_5m_ratio": repeated_05 / denominator,
        "viewpoint_repeat_le_0_75m": repeated_075,
        "consecutive_viewpoint_le_0_5m": consecutive_05,
    }


def frontier_repeat_metrics(events_path: Path) -> dict[str, Any]:
    goals: list[tuple[float, float]] = []
    last_key = None
    for event in iter_events(events_path) or []:
        if event.get("type") != "explore_status":
            continue
        active = event.get("payload", {}).get("state", {}).get("active_goal")
        if not isinstance(active, dict):
            continue
        point = active.get("point")
        frontier = active.get("frontier_point")
        if not point or not frontier:
            continue
        key = (
            active.get("cluster_id"),
            round(float(point[0]), 3),
            round(float(point[1]), 3),
        )
        if key == last_key:
            continue
        last_key = key
        goals.append((float(frontier[0]), float(frontier[1])))
    repeated = 0
    for index, point in enumerate(goals):
        if index and min(math.dist(point, previous) for previous in goals[:index]) <= 0.5 + 1e-6:
            repeated += 1
    return {
        "frontier_goal_transitions": len(goals),
        "frontier_repeat_le_0_5m": repeated,
        "frontier_repeat_le_0_5m_ratio": repeated / max(1, len(goals) - 1),
    }


def simulation_step_count(scene_dir: Path) -> int:
    try:
        import h5py
    except ImportError:
        return 0
    counts = []
    for path in scene_dir.glob("sim/house_*/trajectories_batch_*.h5"):
        try:
            with h5py.File(path, "r") as handle:
                for key in handle:
                    actions = handle[key].get("actions/commanded_action")
                    if actions is not None:
                        counts.append(int(actions.shape[0]))
        except (OSError, KeyError, ValueError):
            continue
    return max(counts, default=0)


def final_explorer_metrics(events_path: Path) -> dict[str, Any]:
    last_status: dict[str, Any] = {}
    seen_positive = False
    completion = None
    max_step = 0
    for event in iter_events(events_path) or []:
        max_step = max(max_step, int(event.get("step_id") or 0))
        if event.get("type") != "explore_status":
            continue
        payload = event.get("payload", {})
        last_status = payload
        frontier_count = payload.get("frontier_count")
        if isinstance(frontier_count, int) and frontier_count > 0:
            seen_positive = True
            # A transient zero-frontier state is not completion if frontiers return later.
            completion = None
        elif seen_positive and frontier_count == 0 and completion is None:
            completion = {
                "elapsed_sec": float(event.get("elapsed_sec") or 0.0),
                "step_id": int(event.get("step_id") or 0),
            }
    state = last_status.get("state", {}) if isinstance(last_status, dict) else {}
    frontier_debug = last_status.get("frontier_debug", {}) if isinstance(last_status, dict) else {}
    return {
        "final_step_id": max_step,
        "frontier_count": last_status.get("frontier_count"),
        "raw_frontier_cells": frontier_debug.get("frontier_cells"),
        "kept_frontier_clusters": frontier_debug.get("kept_clusters"),
        "unreachable_frontiers": state.get("unreachable_frontiers"),
        "active_goal": state.get("active_goal") is not None,
        "last_explorer_event": state.get("last_event"),
        "completion": completion,
    }


def build_scene_metrics(
    scene_dir: Path,
    worker_id: int,
    scene_index: int,
    house_ind: int,
    launch_exit: int | None,
    recorder_exit: int | None,
    elapsed_sec: float,
    task_horizon: int,
) -> dict[str, Any]:
    debug_dir = scene_dir / "debug"
    summary = read_json(debug_dir / "summary.json")
    full_step_video = read_json(debug_dir / "full_step_video_summary.json")
    coverage = read_json(debug_dir / "exploration_coverage.json")
    metrics: dict[str, Any] = {
        "worker_id": worker_id,
        "scene_index": scene_index,
        "house_ind": house_ind,
        "scene_dir": str(scene_dir),
        "launch_exit": launch_exit,
        "recorder_exit": recorder_exit,
        "elapsed_sec": elapsed_sec,
        "distance_m": summary.get("distance_m"),
        "trajectory_samples": summary.get("trajectory_samples"),
        "recorder_video_frame_count": summary.get("first_person_video_frame_count"),
        "video_frame_count": (
            full_step_video.get("output_frame_count")
            or summary.get("first_person_video_frame_count")
        ),
        "full_step_video_frame_count": full_step_video.get("output_frame_count"),
        "sim_step_frame_count": full_step_video.get("sim_frame_count"),
        "exact_step_match_count": full_step_video.get("exact_step_match_count"),
        "missing_video_step_count": full_step_video.get("missing_step_count"),
        "video_max_stamp_delta_sec": full_step_video.get("max_stamp_delta_sec"),
        "finalization_complete": summary.get("finalization_complete"),
        "stuck": (debug_dir / "stuck_exit.json").exists(),
        "coverage_ratio": coverage.get("exploration_coverage_ratio"),
        "mapped_free_coverage_ratio": coverage.get("mapped_free_coverage_ratio"),
        "mapped_occupied_on_gt_free_ratio": coverage.get("mapped_occupied_on_gt_free_ratio"),
        "gt_navigable_area_m2": coverage.get("gt_navigable_area_m2"),
        "observed_gt_area_m2": coverage.get("observed_gt_area_m2"),
    }
    metrics.update(subgoal_repeat_metrics(debug_dir / "subgoals.csv"))
    metrics.update(frontier_repeat_metrics(debug_dir / "events.jsonl"))
    metrics.update(final_explorer_metrics(debug_dir / "events.jsonl"))
    metrics["simulation_frames"] = simulation_step_count(scene_dir)
    metrics["required_simulation_frames"] = task_horizon + 1
    metrics["required_video_frames"] = task_horizon
    metrics["video_path"] = str(debug_dir / "videos" / "first_person.mp4")
    metrics["final_occ_path"] = str(debug_dir / "final_map_trajectory_crop.png")
    metrics["valid"] = bool(
        (debug_dir / "final_occ_map.yaml").exists()
        and (debug_dir / "videos" / "first_person.mp4").exists()
        and coverage
        and metrics["simulation_frames"] >= metrics["required_simulation_frames"]
        and metrics["full_step_video_frame_count"] == metrics["sim_step_frame_count"]
        and metrics["exact_step_match_count"] == metrics["sim_step_frame_count"]
        and int(metrics["missing_video_step_count"] or 0) == 0
        and float(metrics["video_max_stamp_delta_sec"] or 0.0) <= 0.05
        and int(metrics["full_step_video_frame_count"] or 0) >= metrics["required_video_frames"]
    )
    metrics["status"] = "success" if metrics["valid"] else "failed"
    return metrics


def write_analysis(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    columns = [
        "worker_id", "scene_index", "house_ind", "status", "valid", "elapsed_sec",
        "simulation_frames", "required_simulation_frames", "required_video_frames", "sim_step_frame_count",
        "full_step_video_frame_count", "exact_step_match_count", "missing_video_step_count",
        "video_max_stamp_delta_sec", "raw_composite_frame_count",
        "output_size_bytes", "final_step_id", "distance_m", "subgoal_count", "coverage_ratio",
        "mapped_free_coverage_ratio", "mapped_occupied_on_gt_free_ratio",
        "frontier_count", "unreachable_frontiers", "frontier_repeat_le_0_5m_ratio",
        "viewpoint_repeat_le_0_5m_ratio", "active_goal", "last_explorer_event", "stuck",
        "scene_dir", "video_path", "final_occ_path",
    ]
    with (output_dir / "analysis.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["scene_index"]))

    valid = [row for row in rows if row.get("valid")]

    def values(key: str, subset: list[dict[str, Any]] | None = None) -> list[float]:
        source = valid if subset is None else subset
        return [float(row[key]) for row in source if row.get(key) is not None]

    def average(key: str) -> float | None:
        samples = values(key)
        return statistics.fmean(samples) if samples else None

    def distribution(key: str) -> dict[str, float | None]:
        samples = values(key)
        if not samples:
            return {name: None for name in ("min", "q25", "median", "q75", "max", "mean", "std")}
        array = np.asarray(samples, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "q25": float(np.percentile(array, 25)),
            "median": float(np.median(array)),
            "q75": float(np.percentile(array, 75)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
        }

    worker_stats = []
    for worker_id in sorted({int(row["worker_id"]) for row in rows}):
        worker_rows = [row for row in rows if int(row["worker_id"]) == worker_id]
        worker_valid = [row for row in worker_rows if row.get("valid")]
        coverage = values("coverage_ratio", worker_valid)
        elapsed = values("elapsed_sec", worker_valid)
        worker_stats.append({
            "worker_id": worker_id,
            "scene_count": len(worker_rows),
            "valid_count": len(worker_valid),
            "mean_coverage": statistics.fmean(coverage) if coverage else None,
            "median_coverage": statistics.median(coverage) if coverage else None,
            "elapsed_scene_sum_sec": sum(elapsed),
        })

    resource_stats = []
    resource_samples_by_worker: dict[int, list[dict[str, Any]]] = {}
    gpu_samples: list[dict[str, Any]] = []
    for worker_id in sorted({int(row["worker_id"]) for row in rows}):
        samples = list(iter_events(output_dir / f"worker_{worker_id:02d}" / "resources.jsonl") or [])
        resource_samples_by_worker[worker_id] = samples
        rss = [float(sample.get("process_usage", {}).get("rss_mb") or 0.0) for sample in samples]
        process_counts = [int(sample.get("process_usage", {}).get("process_count") or 0) for sample in samples]
        for sample in samples:
            gpu_samples.extend(sample.get("gpus") or [])
        resource_stats.append({
            "worker_id": worker_id,
            "sample_count": len(samples),
            "peak_process_rss_mb": max(rss, default=0.0),
            "mean_process_rss_mb": statistics.fmean(rss) if rss else 0.0,
            "peak_process_count": max(process_counts, default=0),
        })

    rss_buckets: dict[int, dict[int, float]] = {}
    host_memory_used: list[float] = []
    host_memory_available: list[float] = []
    swap_used: list[float] = []
    for worker_id, samples in resource_samples_by_worker.items():
        for sample in samples:
            try:
                timestamp = datetime.fromisoformat(str(sample["timestamp"])).timestamp()
            except (KeyError, TypeError, ValueError):
                continue
            bucket = math.floor(timestamp / 10.0) * 10
            worker_rss = float(sample.get("process_usage", {}).get("rss_mb") or 0.0)
            bucket_workers = rss_buckets.setdefault(bucket, {})
            bucket_workers[worker_id] = max(worker_rss, bucket_workers.get(worker_id, 0.0))
            total_mb = float(sample.get("mem_total_mb") or 0.0)
            available_mb = float(sample.get("mem_available_mb") or 0.0)
            if total_mb and available_mb:
                host_memory_used.append(total_mb - available_mb)
                host_memory_available.append(available_mb)
            swap_used.append(float(sample.get("swap_used_mb") or 0.0))
    worker_ids = set(resource_samples_by_worker)
    aggregate_rss = [
        sum(worker_values.values())
        for worker_values in rss_buckets.values()
        if set(worker_values) == worker_ids
    ]

    gpu_memory = [float(sample.get("memory_used_mb") or 0.0) for sample in gpu_samples]
    gpu_utilization = [float(sample.get("utilization_gpu_pct") or 0.0) for sample in gpu_samples]
    gpu_power = [float(sample.get("power_draw_w") or 0.0) for sample in gpu_samples]
    coverage_distribution = distribution("coverage_ratio")
    ordered_coverage = sorted(
        valid,
        key=lambda row: float(row.get("coverage_ratio") or 0.0),
    )
    summary = {
        "scene_count": len(rows),
        "valid_count": len(valid),
        "failed_count": len(rows) - len(valid),
        "simulation_frames_min": min((int(row.get("simulation_frames") or 0) for row in valid), default=0),
        "simulation_frames_max": max((int(row.get("simulation_frames") or 0) for row in valid), default=0),
        "raw_composite_frame_count": sum(int(row.get("raw_composite_frame_count") or 0) for row in valid),
        "scene_output_size_bytes": sum(int(row.get("output_size_bytes") or 0) for row in valid),
        "elapsed_scene_sum_sec": sum(values("elapsed_sec")),
        "coverage": coverage_distribution,
        "mapped_free_coverage": distribution("mapped_free_coverage_ratio"),
        "mapped_occupied_on_gt_free": distribution("mapped_occupied_on_gt_free_ratio"),
        "distance_m": distribution("distance_m"),
        "subgoal_count": distribution("subgoal_count"),
        "coverage_threshold_counts": {
            f"ge_{int(threshold * 100)}pct": sum(
                float(row.get("coverage_ratio") or 0.0) >= threshold for row in valid
            )
            for threshold in (0.5, 0.8, 0.9, 0.95)
        },
        "frontier_zero_count": sum(int(row.get("frontier_count") or 0) == 0 for row in valid),
        "active_goal_count": sum(bool(row.get("active_goal")) for row in valid),
        "stuck_count": sum(bool(row.get("stuck")) for row in valid),
        "worker_stats": worker_stats,
        "resource_stats": resource_stats,
        "host_resources": {
            "peak_aligned_worker_rss_mb": max(aggregate_rss, default=0.0),
            "peak_host_memory_used_mb": max(host_memory_used, default=0.0),
            "minimum_host_memory_available_mb": min(host_memory_available, default=0.0),
            "peak_swap_used_mb": max(swap_used, default=0.0),
        },
        "gpu": {
            "peak_memory_used_mb": max(gpu_memory, default=0.0),
            "mean_memory_used_mb": statistics.fmean(gpu_memory) if gpu_memory else 0.0,
            "peak_utilization_pct": max(gpu_utilization, default=0.0),
            "mean_utilization_pct": statistics.fmean(gpu_utilization) if gpu_utilization else 0.0,
            "peak_power_w": max(gpu_power, default=0.0),
        },
        "lowest_coverage_scenes": [
            {"scene_index": row["scene_index"], "house_ind": row["house_ind"], "coverage_ratio": row.get("coverage_ratio")}
            for row in ordered_coverage[:10]
        ],
        "highest_coverage_scenes": [
            {"scene_index": row["scene_index"], "house_ind": row["house_ind"], "coverage_ratio": row.get("coverage_ratio")}
            for row in reversed(ordered_coverage[-10:])
        ],
    }

    lines = [
        "# Exploration Coverage Evaluation",
        "",
        f"- Scenes requested: {len(rows)}",
        f"- Valid scenes: {len(valid)}",
        f"- Failed scenes: {len(rows) - len(valid)}",
        f"- Mean coverage: {average('coverage_ratio') or 0.0:.4f}",
        f"- Median coverage: {float(coverage_distribution['median'] or 0.0):.4f}",
        f"- Coverage Q25/Q75: {float(coverage_distribution['q25'] or 0.0):.4f} / {float(coverage_distribution['q75'] or 0.0):.4f}",
        f"- Mean mapped-free coverage: {average('mapped_free_coverage_ratio') or 0.0:.4f}",
        f"- Mean distance: {average('distance_m') or 0.0:.3f} m",
        f"- Mean subgoals: {average('subgoal_count') or 0.0:.2f}",
        f"- Mean frontier repeat <=0.5m: {average('frontier_repeat_le_0_5m_ratio') or 0.0:.4f}",
        f"- Mean viewpoint repeat <=0.5m: {average('viewpoint_repeat_le_0_5m_ratio') or 0.0:.4f}",
        f"- Coverage >=50/80/90/95%: {summary['coverage_threshold_counts']['ge_50pct']} / {summary['coverage_threshold_counts']['ge_80pct']} / {summary['coverage_threshold_counts']['ge_90pct']} / {summary['coverage_threshold_counts']['ge_95pct']}",
        f"- Simulation frames min/max: {summary['simulation_frames_min']} / {summary['simulation_frames_max']}",
        f"- Preserved composite PNG frames: {summary['raw_composite_frame_count']}",
        f"- Successful scene data size: {summary['scene_output_size_bytes'] / (1024 ** 3):.2f} GiB",
        f"- Peak GPU memory/utilization: {summary['gpu']['peak_memory_used_mb']:.0f} MiB / {summary['gpu']['peak_utilization_pct']:.1f}%",
        f"- Peak aligned worker RSS: {summary['host_resources']['peak_aligned_worker_rss_mb'] / 1024.0:.2f} GiB",
        f"- Minimum host memory available / peak swap: {summary['host_resources']['minimum_host_memory_available_mb'] / 1024.0:.2f} / {summary['host_resources']['peak_swap_used_mb'] / 1024.0:.2f} GiB",
        "",
        "## Per-worker summary",
        "",
        "| Worker | Scenes | Valid | Mean coverage | Median coverage | Scene time sum h | Peak RSS MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for worker in worker_stats:
        resource = next(item for item in resource_stats if item["worker_id"] == worker["worker_id"])
        lines.append(
            f"| {worker['worker_id']} | {worker['scene_count']} | {worker['valid_count']} | "
            f"{float(worker['mean_coverage'] or 0.0):.3f} | {float(worker['median_coverage'] or 0.0):.3f} | "
            f"{worker['elapsed_scene_sum_sec'] / 3600.0:.2f} | {resource['peak_process_rss_mb']:.0f} |"
        )
    lines.extend([
        "",
        "## Scene results",
        "",
        "| Worker | Scene | House | Status | Frames | PNG | Coverage | Free coverage | Distance m | Subgoals | Frontier |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(rows, key=lambda item: item["scene_index"]):
        lines.append(
            f"| {row['worker_id']} | {row['scene_index']} | {row['house_ind']} | {row['status']} | "
            f"{int(row.get('simulation_frames') or 0)} | {int(row.get('raw_composite_frame_count') or 0)} | "
            f"{float(row.get('coverage_ratio') or 0.0):.3f} | "
            f"{float(row.get('mapped_free_coverage_ratio') or 0.0):.3f} | "
            f"{float(row.get('distance_m') or 0.0):.2f} | {int(row.get('subgoal_count') or 0)} | "
            f"{row.get('frontier_count')} |"
        )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n")
    write_json(output_dir / "analysis.json", rows)
    write_json(output_dir / "analysis_summary.json", summary)


def render_worker_contact_sheet(
    worker_dir: Path,
    rows: list[dict[str, Any]],
    *,
    gt_coverage: bool = True,
    output_name: str | None = None,
) -> Path:
    panel_w, panel_h = 520, 420
    cols = 5
    rows_count = max(1, math.ceil(len(rows) / cols))
    header_h = 62 if gt_coverage else 0
    canvas = np.full((header_h + rows_count * panel_h, cols * panel_w, 3), 245, dtype=np.uint8)
    if gt_coverage:
        legend = [
            ((46, 42, 38), "GT non-navigable"),
            ((224, 202, 183), "GT free, unexplored"),
            ((80, 175, 76), "explored free"),
            ((95, 183, 246), "observed non-free"),
            ((47, 47, 211), "false occupied"),
            ((255, 170, 0), "trajectory"),
            ((180, 35, 128), "pending frontier"),
        ]
        x = 18
        for color, label in legend:
            cv2.rectangle(canvas, (x, 18), (x + 24, 42), color, -1)
            cv2.rectangle(canvas, (x, 18), (x + 24, 42), (25, 25, 25), 1)
            cv2.putText(
                canvas,
                label,
                (x + 32, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            x += 245
    for index, row in enumerate(sorted(rows, key=lambda item: item["scene_index"])):
        if gt_coverage:
            image_path = Path(row["scene_dir"]) / "debug" / "exploration_coverage.png"
        else:
            image_path = Path(row["final_occ_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if gt_coverage and image is not None:
            # The per-scene evaluator adds its own text header; the sheet supplies one compact legend.
            coverage_metadata = read_json(Path(row["scene_dir"]) / "debug" / "exploration_coverage.json")
            coverage_header_h = int(coverage_metadata.get("render_header_height_px", 92))
            if image.shape[0] > coverage_header_h:
                image = image[coverage_header_h:, :]
        if image is None:
            image = np.full((320, 500, 3), 220, dtype=np.uint8)
            cv2.putText(image, "MISSING MAP", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2)
        available_h = panel_h - 62
        scale = min(panel_w / image.shape[1], available_h / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
        x = (panel_w - resized.shape[1]) // 2
        y = 58 + (available_h - resized.shape[0]) // 2
        panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        coverage = row.get("coverage_ratio")
        label = f"scene={row['scene_index']:03d} house={row['house_ind']:03d} {row['status']}"
        metric = f"coverage={coverage:.1%}" if isinstance(coverage, (int, float)) else "coverage=N/A"
        cv2.putText(panel, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(panel, metric, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 90, 20), 1, cv2.LINE_AA)
        row_index, col_index = divmod(index, cols)
        canvas[
            header_h + row_index * panel_h:header_h + (row_index + 1) * panel_h,
            col_index * panel_w:(col_index + 1) * panel_w,
        ] = panel
    if output_name is None:
        output_name = "final_occ_contact_sheet.png" if gt_coverage else "final_occ_observed_contact_sheet.png"
    output = worker_dir / output_name
    cv2.imwrite(str(output), canvas)
    return output


class BatchWorker:
    def __init__(
        self,
        worker_id: int,
        scenes: list[tuple[int, int]],
        args: argparse.Namespace,
        global_abort: threading.Event,
    ):
        self.worker_id = worker_id
        self.scenes = scenes
        self.args = args
        self.global_abort = global_abort
        self.port = args.base_master_port + worker_id
        self.master_uri = f"http://127.0.0.1:{self.port}"
        self.worker_dir = args.output_dir / f"worker_{worker_id:02d}"
        self.ros_home = Path("/tmp/molmospaces_explore100") / f"worker_{worker_id:02d}"
        self.rows: list[dict[str, Any]] = []
        self.monitor_stop = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.stop_requested = threading.Event()
        self.current_roscore: subprocess.Popen | None = None
        self.current_roslaunch: subprocess.Popen | None = None
        self.current_recorder: subprocess.Popen | None = None

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "ROS_MASTER_URI": self.master_uri,
            "ROS_HOME": str(self.ros_home),
            "ROS_LOG_DIR": str(self.ros_home / "log"),
            "ROS_HOSTNAME": "127.0.0.1",
            "CUDA_VISIBLE_DEVICES": str(self.args.gpu_id),
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib_explore100",
        })
        return env

    def log(self, message: str) -> None:
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        with (self.worker_dir / "worker.log").open("a") as stream:
            stream.write(f"{utc_now()} {message}\n")

    def monitor(self) -> None:
        path = self.worker_dir / "resources.jsonl"
        while not self.monitor_stop.wait(self.args.resource_interval_s):
            sample = read_host_sample()
            sample.update({
                "worker_id": self.worker_id,
                "ros_master_uri": self.master_uri,
                "process_usage": process_usage(pids_for_master(self.master_uri)),
                "gpus": read_gpu_samples(),
            })
            with path.open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            minimum_mb = max(0.0, float(self.args.min_memory_available_mb))
            available_mb = float(sample.get("mem_available_mb") or 0.0)
            if minimum_mb > 0.0 and available_mb < minimum_mb:
                reason = {
                    "timestamp": sample["timestamp"],
                    "worker_id": self.worker_id,
                    "reason": "low_host_memory",
                    "mem_available_mb": available_mb,
                    "minimum_required_mb": minimum_mb,
                    "swap_used_mb": float(sample.get("swap_used_mb") or 0.0),
                }
                write_json(self.worker_dir / "resource_abort.json", reason)
                self.log(
                    f"resource abort available={available_mb:.0f}MB "
                    f"threshold={minimum_mb:.0f}MB"
                )
                self.global_abort.set()
                self.stop()
                return

    def wait_for_master(self, roscore: subprocess.Popen) -> None:
        deadline = time.monotonic() + 30.0
        command = shell_command(self.args.setup_file, ["rosparam", "list"])
        while time.monotonic() < deadline:
            if roscore.poll() is not None:
                raise RuntimeError(f"roscore exited: {roscore.returncode}")
            result = subprocess.run(command, env=self.environment(), capture_output=True, timeout=5)
            if result.returncode == 0:
                return
            time.sleep(0.25)
        raise TimeoutError(f"ROS master did not start on {self.master_uri}")

    def run_coverage(self, scene_dir: Path, house_ind: int) -> int | None:
        debug_dir = scene_dir / "debug"
        if not (debug_dir / "final_occ_map.yaml").exists():
            return None
        command = [
            sys.executable,
            str(self.args.coverage_script),
            "--run-dir", str(debug_dir),
            "--robot", self.args.robot,
            "--scene-dataset", self.args.scene_dataset,
            "--data-split", self.args.data_split,
            "--house-ind", str(house_ind),
            "--gt-agent-radius-m", "0.10",
        ]
        with (scene_dir / "coverage.log").open("w") as stream:
            try:
                result = subprocess.run(
                    command,
                    env=self.environment(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=self.args.coverage_timeout_s,
                )
                return result.returncode
            except subprocess.TimeoutExpired:
                stream.write("\ncoverage timeout\n")
                return None

    def run_scene(self, scene_index: int, house_ind: int) -> dict[str, Any]:
        scene_dir = self.worker_dir / f"scene_{scene_index:03d}_house_{house_ind:03d}"
        metrics_path = scene_dir / "scene_metrics.json"
        if self.args.resume:
            existing = read_json(metrics_path)
            if existing.get("valid"):
                self.log(f"resume skip scene={scene_index} house={house_ind}")
                return existing
        if self.args.retry_invalid and scene_dir.exists():
            archive_root = self.worker_dir / "failed_attempts"
            archive_root.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while True:
                archive = archive_root / f"{scene_dir.name}_attempt_{attempt:02d}"
                if not archive.exists():
                    scene_dir.rename(archive)
                    self.log(f"archived invalid scene={scene_index} house={house_ind} to={archive}")
                    break
                attempt += 1

        scene_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = scene_dir / "debug"
        sim_dir = scene_dir / "sim"
        sim_step_frame_dir = scene_dir / "sim_step_frames"
        debug_dir.mkdir(exist_ok=True)
        sim_dir.mkdir(exist_ok=True)
        env = self.environment()
        (self.ros_home / "log").mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        roscore = roslaunch = recorder = None
        launch_exit = recorder_exit = None
        try:
            with (scene_dir / "roscore.log").open("w") as log:
                roscore = subprocess.Popen(
                    shell_command(self.args.setup_file, ["roscore", "-p", str(self.port)]),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.current_roscore = roscore
                self.wait_for_master(roscore)

            launch_command = [
                "roslaunch", "nav_pkg", "molmospaces_nav_system.launch",
                f"robot:={self.args.robot}",
                f"scene_dataset:={self.args.scene_dataset}",
                f"data_split:={self.args.data_split}",
                f"house_ind:={house_ind}",
                f"house_inds:={house_ind}",
                "target_types:=",
                f"task_horizon:={self.args.task_horizon}",
                f"scene_timeout_s:={self.args.scene_timeout_s}",
                f"max_scene_attempts:={self.args.max_scene_attempts}",
                "max_consecutive_action_timeouts:=0",
                "exploration_only:=true",
                "start_explore_py:=true",
                "start_semantic_mapping:=false",
                f"explore_py_config_override_file:={self.args.explore_config_override_file or ''}",
                f"nav_config_override_file:={self.args.nav_config_override_file or ''}",
                f"base_local_planner:={self.args.base_local_planner}",
                "publish_debug_front_camera:=true",
                f"output_dir:={sim_dir}",
                "sim_extra_args:=--samples_per_house 1 --cmd_vel_linear_gain 3.0 "
                f"--action_timeout_s {self.args.action_timeout_s:g} "
                f"--step_frame_dir {sim_step_frame_dir} "
                "--debug_front_camera_offset=-1.4,0.0,1.35 "
                "--debug_front_camera_lookat_offset=0.0,0.0,0.35 --step_log_every_n_steps 50",
            ]
            recorder_command = [
                sys.executable,
                str(self.args.recorder_script),
                "--output-dir", str(debug_dir),
                "--stall-snapshot-sec", "30",
                "--stall-snapshot-distance-m", "0.15",
                "--stall-snapshot-cooldown-sec", "45",
                "--external-image-topic", "/molmo_spaces/debug_front_camera/image",
                "--no-external-video",
                "--first-person-video-with-map",
                "--first-person-video-fps", "15",
                "--first-person-video-capture-mode", "step",
                "--video-step-sync-topic", "/molmo_spaces/step_sync",
                "--step-sync-queue-size", str(max(512, self.args.task_horizon + 32)),
                "--image-queue-size", "8",
                "--video-frame-job-queue-size", str(max(512, self.args.task_horizon + 32)),
                "--video-history-size", "64",
                "--no-video-save-panel-frames",
                "--first-person-video-h264-preset", "ultrafast",
                "--first-person-video-h264-timeout-sec", "300",
                "--overlay-contact-sheet-columns", "4",
                "--async-artifact-writes",
                "--artifact-write-queue-size", str(max(512, self.args.task_horizon + 32)),
                "--local-global-plan-topic",
                f"/move_base/{self.args.local_planner_namespace}/global_plan",
                "--local-plan-topic",
                f"/move_base/{self.args.local_planner_namespace}/local_plan",
            ]
            with (scene_dir / "recorder.log").open("w") as recorder_log:
                recorder = subprocess.Popen(
                    shell_command(self.args.setup_file, recorder_command),
                    env=env,
                    stdout=recorder_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.current_recorder = recorder
            # Register recorder subscribers before the simulator publishes step 0.
            time.sleep(1.0)
            with (scene_dir / "roslaunch.log").open("w") as launch_log:
                roslaunch = subprocess.Popen(
                    shell_command(self.args.setup_file, launch_command),
                    env=env,
                    stdout=launch_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.current_roslaunch = roslaunch
            try:
                launch_exit = roslaunch.wait(timeout=self.args.scene_timeout_s + 180.0)
            except subprocess.TimeoutExpired:
                self.log(f"scene timeout scene={scene_index} house={house_ind}")
                terminate_group(roslaunch, 15.0)
                launch_exit = roslaunch.returncode
        except Exception as exc:
            self.log(f"scene exception scene={scene_index} house={house_ind}: {exc!r}")
        finally:
            terminate_group(recorder, self.args.recorder_shutdown_grace_s)
            recorder_exit = None if recorder is None else recorder.returncode
            terminate_group(roslaunch, 15.0)
            terminate_group(roscore, 10.0)
            self.current_recorder = None
            self.current_roslaunch = None
            self.current_roscore = None

        full_step_video_exit = None
        try:
            full_step_video_exit, full_step_summary = build_full_step_video(
                self.args.offline_video_builder,
                scene_dir,
                debug_dir,
                scene_dir / "full_step_video_build.log",
            )
            if full_step_video_exit != 0:
                self.log(
                    f"full-step video failed scene={scene_index} house={house_ind} "
                    f"exit={full_step_video_exit}"
                )
            elif full_step_summary.get("output_frame_count") != full_step_summary.get("sim_frame_count"):
                self.log(
                    f"full-step video incomplete scene={scene_index} house={house_ind} "
                    f"frames={full_step_summary.get('output_frame_count')}/"
                    f"{full_step_summary.get('sim_frame_count')}"
                )
            elif full_step_summary.get("exact_step_match_count") != full_step_summary.get("sim_frame_count"):
                self.log(
                    f"full-step exact matching failed scene={scene_index} house={house_ind} "
                    f"matches={full_step_summary.get('exact_step_match_count')}/"
                    f"{full_step_summary.get('sim_frame_count')}"
                )
        except (OSError, subprocess.SubprocessError) as exc:
            self.log(f"full-step video exception scene={scene_index} house={house_ind}: {exc!r}")
        video_path, frame_count = ensure_h264(debug_dir, scene_dir / "ffmpeg_fallback.log")
        coverage_exit = self.run_coverage(scene_dir, house_ind)
        elapsed = time.monotonic() - start
        metrics = build_scene_metrics(
            scene_dir,
            self.worker_id,
            scene_index,
            house_ind,
            launch_exit,
            recorder_exit,
            elapsed,
            self.args.task_horizon,
        )
        metrics["coverage_exit"] = coverage_exit
        metrics["full_step_video_exit"] = full_step_video_exit
        metrics["video_path"] = video_path
        metrics["raw_composite_frame_count"] = frame_count
        metrics["output_size_bytes"] = sum(
            path.stat().st_size for path in scene_dir.rglob("*") if path.is_file()
        )
        write_json(metrics_path, metrics)
        self.log(
            f"scene done scene={scene_index} house={house_ind} status={metrics['status']} "
            f"coverage={metrics.get('coverage_ratio')} elapsed={elapsed:.1f}s"
        )
        return metrics

    def run(self) -> None:
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()
        try:
            for scene_index, house_ind in self.scenes:
                if self.stop_requested.is_set():
                    break
                self.log(f"scene start scene={scene_index} house={house_ind}")
                self.rows.append(self.run_scene(scene_index, house_ind))
                write_json(self.worker_dir / "worker_metrics.json", self.rows)
        finally:
            self.monitor_stop.set()
            if self.monitor_thread is not None:
                self.monitor_thread.join(timeout=10)
            worker_rows = [
                row for row in load_scene_metrics(self.args.output_dir)
                if int(row.get("worker_id", -1)) == self.worker_id
            ]
            contact_sheet = render_worker_contact_sheet(self.worker_dir, worker_rows)
            observed_contact_sheet = render_worker_contact_sheet(
                self.worker_dir,
                worker_rows,
                gt_coverage=False,
            )
            write_json(
                self.worker_dir / "worker_summary.json",
                {
                    "worker_id": self.worker_id,
                    "scene_count": len(worker_rows),
                    "valid_count": sum(bool(row.get("valid")) for row in worker_rows),
                    "contact_sheet": str(contact_sheet),
                    "observed_contact_sheet": str(observed_contact_sheet),
                    "completed_at": utc_now(),
                },
            )

    def stop(self) -> None:
        self.stop_requested.set()
        self.monitor_stop.set()
        terminate_group(self.current_recorder, 5.0)
        terminate_group(self.current_roslaunch, 5.0)
        terminate_group(self.current_roscore, 5.0)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if args.house_count < 1:
        raise ValueError("--house-count must be positive")
    for path in (
        args.setup_file,
        args.recorder_script,
        args.coverage_script,
        args.offline_video_builder,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_abort = threading.Event()

    if args.retry_invalid:
        existing_rows = load_scene_metrics(args.output_dir)
        invalid_rows = [row for row in existing_rows if not row.get("valid")]
        worker_scenes: dict[int, list[tuple[int, int]]] = {}
        for row in invalid_rows:
            worker_scenes.setdefault(int(row["worker_id"]), []).append(
                (int(row["scene_index"]), int(row["house_ind"]))
            )
        workers = [
            BatchWorker(worker_id, sorted(worker_scenes[worker_id]), args, global_abort)
            for worker_id in sorted(worker_scenes)
        ]
        scenes = [scene for worker in workers for scene in worker.scenes]
        if not workers:
            write_analysis(existing_rows, args.output_dir)
            print("No invalid scenes to retry.")
            return 0
    else:
        scenes = [
            (
                index,
                args.repeat_house_ind
                if args.repeat_house_ind is not None
                else args.house_start + index,
            )
            for index in range(args.house_count)
        ]
        shards = [scenes[index::args.num_workers] for index in range(args.num_workers)]
        workers = [
            BatchWorker(index, shard, args, global_abort)
            for index, shard in enumerate(shards)
        ]
    plan = {
        "created_at": utc_now(),
        "house_start": args.house_start,
        "house_count": args.house_count,
        "repeat_house_ind": args.repeat_house_ind,
        "num_workers": args.num_workers,
        "task_horizon": args.task_horizon,
        "action_timeout_s": args.action_timeout_s,
        "min_memory_available_mb": args.min_memory_available_mb,
        "explore_config_override_file": (
            str(args.explore_config_override_file.resolve())
            if args.explore_config_override_file else ""
        ),
        "nav_config_override_file": (
            str(args.nav_config_override_file.resolve())
            if args.nav_config_override_file else ""
        ),
        "base_local_planner": args.base_local_planner,
        "local_planner_namespace": args.local_planner_namespace,
        "offline_video_builder": str(args.offline_video_builder.resolve()),
        "workers": [
            {
                "worker_id": worker.worker_id,
                "master_uri": worker.master_uri,
                "scenes": worker.scenes,
                "output_dir": str(worker.worker_dir),
            }
            for worker in workers
        ],
    }
    write_json(args.output_dir / ("retry_plan.json" if args.retry_invalid else "plan.json"), plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    occupied = [worker.port for worker in workers if not port_available(worker.port)]
    if occupied:
        raise RuntimeError(f"ROS master ports occupied: {occupied}")

    threads = [threading.Thread(target=worker.run, name=f"explore-worker-{worker.worker_id}") for worker in workers]
    abort_handled = False
    try:
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            if global_abort.is_set() and not abort_handled:
                abort_handled = True
                for worker in workers:
                    worker.stop()
            for thread in threads:
                thread.join(timeout=1.0)
            rows = load_scene_metrics(args.output_dir)
            write_analysis(rows, args.output_dir)
    except KeyboardInterrupt:
        for worker in workers:
            worker.stop()
        for thread in threads:
            thread.join(timeout=20.0)
        return 130

    all_rows = load_scene_metrics(args.output_dir)
    write_analysis(all_rows, args.output_dir)
    final = {
        "completed_at": utc_now(),
        "scene_count": len(all_rows),
        "valid_count": sum(bool(row.get("valid")) for row in all_rows),
        "failed_count": sum(not bool(row.get("valid")) for row in all_rows),
        "resource_abort": global_abort.is_set(),
        "worker_contact_sheets": [
            str(worker_dir / "final_occ_contact_sheet.png")
            for worker_dir in sorted(args.output_dir.glob("worker_*"))
            if worker_dir.is_dir()
        ],
        "analysis_csv": str(args.output_dir / "analysis.csv"),
        "analysis_md": str(args.output_dir / "analysis.md"),
    }
    write_json(args.output_dir / "summary.json", final)
    return 0 if final["failed_count"] == 0 else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
