from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import explore_molmo_interactions as emi


DEFAULT_OUTPUT_DIR = Path(
    "/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/output/benchmark_door_state_scan"
)


def single_door_plot_path(episode_dir: Path, door_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", door_name)
    return episode_dir / f"single_door_close_{safe_name}.png"


def make_episode_args(args: argparse.Namespace, episode: dict[str, Any], output_dir: Path) -> argparse.Namespace:
    robot_name = episode.get("robot", {}).get("robot_name", args.robot)
    return argparse.Namespace(
        command="benchmark-door-state-scan",
        scene_dataset=episode["scene_dataset"],
        data_split=episode["data_split"],
        house_ind=episode["house_index"],
        variant=args.variant,
        robot=robot_name,
        seed=args.seed,
        target_types=None,
        output_json=output_dir / "result.json",
        px_per_m=args.px_per_m,
        downscale=args.downscale,
        open_threshold=args.open_threshold,
        benchmark_episode=episode,
    )


def path_changed(open_path: np.ndarray | None, closed_path: np.ndarray | None, atol: float = 1e-4) -> bool:
    if open_path is None and closed_path is None:
        return False
    if (open_path is None) != (closed_path is None):
        return True
    assert open_path is not None and closed_path is not None
    if open_path.shape != closed_path.shape:
        return True
    return not np.allclose(open_path, closed_path, atol=atol, rtol=0.0)


def path_length_m(path: np.ndarray | None) -> float | None:
    return emi.path_length(path)


def path_length_delta(open_path: np.ndarray | None, closed_path: np.ndarray | None) -> float | None:
    open_len = path_length_m(open_path)
    closed_len = path_length_m(closed_path)
    if open_len is None or closed_len is None:
        return None
    return float(closed_len - open_len)


def resample_path_by_arclength(path: np.ndarray, num_samples: int) -> np.ndarray:
    if len(path) == 0:
        return path
    if len(path) == 1:
        return np.repeat(path, num_samples, axis=0)

    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 1e-8:
        return np.repeat(path[:1], num_samples, axis=0)

    sample_ds = np.linspace(0.0, total, num_samples)
    out = np.empty((num_samples, 2), dtype=float)
    for i, d in enumerate(sample_ds):
        seg_idx = int(np.searchsorted(cum, d, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(path) - 2)
        seg_start = cum[seg_idx]
        seg_end = cum[seg_idx + 1]
        alpha = 0.0 if seg_end <= seg_start else (d - seg_start) / (seg_end - seg_start)
        out[i] = (1.0 - alpha) * path[seg_idx] + alpha * path[seg_idx + 1]
    return out


def path_distance_stats(
    open_path: np.ndarray | None,
    closed_path: np.ndarray | None,
    num_samples: int = 64,
) -> dict[str, float | None]:
    if open_path is None or closed_path is None:
        return {
            "mean_point_distance_m": None,
            "max_point_distance_m": None,
        }
    open_rs = resample_path_by_arclength(np.asarray(open_path, dtype=float), num_samples)
    closed_rs = resample_path_by_arclength(np.asarray(closed_path, dtype=float), num_samples)
    dists = np.linalg.norm(open_rs - closed_rs, axis=1)
    return {
        "mean_point_distance_m": float(np.mean(dists)),
        "max_point_distance_m": float(np.max(dists)),
    }


def path_changed_strict(
    open_path: np.ndarray | None,
    closed_path: np.ndarray | None,
    distance_threshold_m: float,
    max_distance_threshold_m: float,
    length_delta_threshold_m: float,
) -> tuple[bool, dict[str, float | None]]:
    if open_path is None and closed_path is None:
        stats = {
            "mean_point_distance_m": 0.0,
            "max_point_distance_m": 0.0,
            "path_length_delta_m": 0.0,
        }
        return False, stats
    if (open_path is None) != (closed_path is None):
        stats = {
            **path_distance_stats(open_path, closed_path),
            "path_length_delta_m": path_length_delta(open_path, closed_path),
        }
        return True, stats

    stats = {
        **path_distance_stats(open_path, closed_path),
        "path_length_delta_m": path_length_delta(open_path, closed_path),
    }
    mean_dist = stats["mean_point_distance_m"] or 0.0
    max_dist = stats["max_point_distance_m"] or 0.0
    length_delta = abs(stats["path_length_delta_m"] or 0.0)
    changed = (
        mean_dist > distance_threshold_m
        or max_dist > max_distance_threshold_m
        or length_delta > length_delta_threshold_m
    )
    return changed, stats


def densify_polyline(path: np.ndarray, step_m: float) -> np.ndarray:
    if path is None or len(path) == 0:
        return np.empty((0, 2), dtype=float)
    if len(path) == 1:
        return np.asarray(path, dtype=float)

    dense_points = [np.asarray(path[0], dtype=float)]
    for a, b in zip(path[:-1], path[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-8:
            continue
        n = max(1, int(np.ceil(seg_len / step_m)))
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            dense_points.append(a + t * seg)
    return np.asarray(dense_points, dtype=float)


def path_traverses_door_region(
    path: np.ndarray | None,
    door_record: dict[str, Any],
    padding_m: float,
    sample_step_m: float,
) -> bool:
    return bool(
        path_door_crossing_details(
            path,
            door_record,
            padding_m=padding_m,
            sample_step_m=sample_step_m,
        )["traverses"]
    )


def door_portal_geometry(
    door_record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if "portal_center_xy" in door_record:
        center = np.asarray(door_record["portal_center_xy"], dtype=float)[:2]
        tangent = np.asarray(door_record["portal_tangent_xy"], dtype=float)[:2]
        normal = np.asarray(door_record["portal_normal_xy"], dtype=float)[:2]
        half_width = float(door_record["portal_half_width_m"])
        half_thickness = float(door_record["portal_half_thickness_m"])
    else:
        center = np.asarray(door_record["aabb_center"], dtype=float)[:2]
        size = np.asarray(door_record["aabb_size"], dtype=float)[:2]
        major_axis = int(np.argmax(size))
        tangent = np.zeros(2, dtype=float)
        tangent[major_axis] = 1.0
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        half_width = float(size[major_axis] / 2.0)
        half_thickness = float(size[1 - major_axis] / 2.0)

    tangent_norm = float(np.linalg.norm(tangent))
    normal_norm = float(np.linalg.norm(normal))
    if tangent_norm <= 1e-8 or normal_norm <= 1e-8:
        raise ValueError(f"Invalid portal basis for door {door_record.get('name')}")
    return (
        center,
        tangent / tangent_norm,
        normal / normal_norm,
        max(half_width, 0.12),
        max(half_thickness, 0.02),
    )


def path_door_crossing_details(
    path: np.ndarray | None,
    door_record: dict[str, Any],
    *,
    padding_m: float,
    sample_step_m: float,
) -> dict[str, Any]:
    if path is None or len(path) == 0:
        return {
            "traverses": False,
            "start_inside": False,
            "ignored_initial_region": False,
            "entry_index": None,
            "crossing_index": None,
        }
    center, tangent, normal, half_width, half_thickness = door_portal_geometry(
        door_record
    )
    dense = densify_polyline(np.asarray(path, dtype=float), step_m=sample_step_m)
    if len(dense) == 0:
        return {
            "traverses": False,
            "start_inside": False,
            "ignored_initial_region": False,
            "entry_index": None,
            "crossing_index": None,
        }

    relative = dense - center[None, :]
    tangent_offsets = relative @ tangent
    normal_offsets = relative @ normal
    padded_half_width = half_width + float(padding_m)
    padded_half_thickness = half_thickness + float(padding_m)
    inside = np.logical_and(
        np.abs(tangent_offsets) <= padded_half_width,
        np.abs(normal_offsets) <= padded_half_thickness,
    )
    start_inside = bool(inside[0])
    search_start = 0
    ignored_initial_region = False
    if start_inside:
        outside_indices = np.flatnonzero(~inside)
        if not len(outside_indices):
            return {
                "traverses": False,
                "start_inside": True,
                "ignored_initial_region": True,
                "entry_index": None,
                "crossing_index": None,
            }
        search_start = int(outside_indices[0])
        ignored_initial_region = True

    side_standoff = max(half_thickness + min(float(padding_m), 0.1), 0.08)
    for segment_index in range(search_start, len(dense) - 1):
        first_normal = float(normal_offsets[segment_index])
        second_normal = float(normal_offsets[segment_index + 1])
        if first_normal == second_normal:
            continue
        if first_normal * second_normal > 0.0:
            continue
        ratio = -first_normal / (second_normal - first_normal)
        if ratio < 0.0 or ratio > 1.0:
            continue
        crossing_tangent = float(
            tangent_offsets[segment_index]
            + ratio
            * (tangent_offsets[segment_index + 1] - tangent_offsets[segment_index])
        )
        if abs(crossing_tangent) > padded_half_width:
            continue

        before_index = segment_index
        while before_index > search_start and abs(normal_offsets[before_index]) < side_standoff:
            before_index -= 1
        after_index = segment_index + 1
        while after_index < len(dense) - 1 and abs(normal_offsets[after_index]) < side_standoff:
            after_index += 1
        before_normal = float(normal_offsets[before_index])
        after_normal = float(normal_offsets[after_index])
        if abs(before_normal) < side_standoff or abs(after_normal) < side_standoff:
            continue
        if before_normal * after_normal >= 0.0:
            continue

        entry_index = segment_index
        while entry_index > search_start and inside[entry_index - 1]:
            entry_index -= 1
        return {
            "traverses": True,
            "start_inside": start_inside,
            "ignored_initial_region": ignored_initial_region,
            "entry_index": int(entry_index),
            "crossing_index": int(segment_index + 1),
            "crossing_xy": np.asarray(
                dense[segment_index]
                + ratio * (dense[segment_index + 1] - dense[segment_index]),
                dtype=float,
            ),
            "crossing_tangent_offset_m": crossing_tangent,
        }

    return {
        "traverses": False,
        "start_inside": start_inside,
        "ignored_initial_region": ignored_initial_region,
        "entry_index": None,
        "crossing_index": None,
    }


def traversed_interactive_doors_on_path(
    env,
    doorway_analysis: dict[str, Any] | None,
    path: np.ndarray | None,
    padding_m: float,
    sample_step_m: float,
) -> list[dict[str, Any]]:
    if doorway_analysis is None or path is None or len(path) == 0:
        return []

    traversed = []
    for rec in emi.collect_interactive_door_root_object_records(env, doorway_analysis):
        if path_traverses_door_region(path, rec, padding_m=padding_m, sample_step_m=sample_step_m):
            traversed.append(rec)
    traversed.sort(key=lambda item: item["name"])
    return traversed


def analyze_single_door_detours(
    ctx,
    args: argparse.Namespace,
    episode_dir: Path,
    start_xy: np.ndarray,
    nav_goal_xy: np.ndarray,
    open_path: np.ndarray | None,
    open_scene_map,
    open_analysis: dict[str, Any] | None,
    open_focus_records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    traversed_doors = traversed_interactive_doors_on_path(
        ctx.env,
        open_analysis,
        open_path,
        padding_m=args.door_on_path_padding_m,
        sample_step_m=args.path_region_sample_step_m,
    )
    candidate_doors = []
    for rec in traversed_doors:
        candidate_doors.append(
            {
                "door_name": rec["name"],
                "hinge_body_names": list(rec.get("children", [])),
                "aabb_center": np.asarray(rec["aabb_center"], dtype=float).tolist(),
                "aabb_size": np.asarray(rec["aabb_size"], dtype=float).tolist(),
            }
        )

    qualifying_results = []
    all_results = []
    for door_info in candidate_doors:
        emi.open_all_doors(ctx.env)
        transition = emi.set_door_root_state(
            ctx.env, open_analysis, door_info["door_name"], "closed"
        )
        single_closed_map, single_closed_analysis = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        single_closed_path = emi.compute_path_from_map(
            single_closed_map, start_xy, nav_goal_xy, downscale_factor=args.downscale
        )
        detour_changed, detour_stats = path_changed_strict(
            open_path,
            single_closed_path,
            distance_threshold_m=args.path_mean_distance_threshold_m,
            max_distance_threshold_m=args.path_max_distance_threshold_m,
            length_delta_threshold_m=args.path_length_delta_threshold_m,
        )
        result = {
            **door_info,
            "transition": transition,
            "closed_path_found": single_closed_path is not None,
            "closed_path_length_m": emi.path_length(single_closed_path),
            "path_changed_vs_open_strict": detour_changed,
            **detour_stats,
            "waypoints": None if single_closed_path is None else single_closed_path.tolist(),
            "plot_path": None,
        }

        if single_closed_path is not None and detour_changed:
            single_closed_focus_records = emi.dedupe_plot_records(
                emi.collect_interactive_door_root_object_records(ctx.env, single_closed_analysis)
                + emi.collect_non_interactive_doorway_object_records(ctx.env, single_closed_analysis)
            )
            plot_path = single_door_plot_path(episode_dir, door_info["door_name"])
            emi.save_door_path_figure(
                out_path=plot_path,
                scene_map=single_closed_map,
                door_records=[],
                object_records=single_closed_focus_records,
                selected_doors=[door_info["door_name"]],
                highlighted_object_names=[],
                start_xy=start_xy,
                goal_xy=nav_goal_xy,
                primary_path=single_closed_path,
                primary_label=f"path with {door_info['door_name']} closed",
                secondary_path=open_path,
                secondary_label="benchmark/original GT path",
                title=f"Benchmark Door State Scan | close {door_info['door_name']}",
            )
            result["plot_path"] = str(plot_path)
            qualifying_results.append(result)

        all_results.append(result)

    emi.open_all_doors(ctx.env)
    return [item["door_name"] for item in candidate_doors], all_results, qualifying_results


def run_single_episode(args: argparse.Namespace, episode_index: int, episode: dict[str, Any], root_output_dir: Path) -> dict[str, Any]:
    episode_dir = root_output_dir / f"benchmark_ep_{episode_index:04d}_house_{episode['house_index']}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_args = make_episode_args(args, episode, episode_dir)
    ctx = emi.load_context(episode_args, task_mode="nav_task")
    try:
        start_xy = emi.get_robot_xy(ctx.env)
        open_scene_map, open_analysis = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=episode_args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=episode_args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        nav_goal, nav_goal_source, nav_goal_sampling_error = emi.sample_navigation_goal(
            ctx.task, open_scene_map
        )
        open_path = emi.compute_path_from_map(
            open_scene_map, start_xy, nav_goal[:2], downscale_factor=episode_args.downscale
        )
        open_all_transitions = emi.open_all_doors(ctx.env)
        all_door_names = emi.interactive_door_root_names(open_analysis)

        open_focus_records = emi.dedupe_plot_records(
            emi.collect_interactive_door_root_object_records(ctx.env, open_analysis)
            + emi.collect_non_interactive_doorway_object_records(ctx.env, open_analysis)
        )
        open_plot_path = episode_dir / "all_open.png"
        emi.save_door_path_figure(
            out_path=open_plot_path,
            scene_map=open_scene_map,
            door_records=[],
            object_records=open_focus_records,
            selected_doors=[],
            highlighted_object_names=[],
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=open_path,
            primary_label="benchmark/original GT path",
            title=f"Benchmark Door State Scan | original/open | target={ctx.task.config.task_config.pickup_obj_name}",
        )

        traversed_door_names, single_door_results, qualifying_single_door_results = analyze_single_door_detours(
            ctx=ctx,
            args=args,
            episode_dir=episode_dir,
            start_xy=start_xy,
            nav_goal_xy=nav_goal[:2],
            open_path=open_path,
            open_scene_map=open_scene_map,
            open_analysis=open_analysis,
            open_focus_records=open_focus_records,
        )

        close_all_transitions = emi.close_all_doors(ctx.env)
        closed_scene_map, closed_analysis = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=episode_args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=episode_args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        closed_path = emi.compute_path_from_map(
            closed_scene_map, start_xy, nav_goal[:2], downscale_factor=episode_args.downscale
        )
        closed_focus_records = emi.dedupe_plot_records(
            emi.collect_interactive_door_root_object_records(ctx.env, closed_analysis)
            + emi.collect_non_interactive_doorway_object_records(ctx.env, closed_analysis)
        )
        closed_plot_path = episode_dir / "all_closed.png"
        emi.save_door_path_figure(
            out_path=closed_plot_path,
            scene_map=closed_scene_map,
            door_records=[],
            object_records=closed_focus_records,
            selected_doors=list(all_door_names),
            highlighted_object_names=[],
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=closed_path,
            primary_label="all-closed GT path",
            secondary_path=open_path,
            secondary_label="benchmark/original GT path",
            title="Benchmark Door State Scan | all interactive doors closed",
        )

        all_closed_changed_strict, all_closed_change_stats = path_changed_strict(
            open_path,
            closed_path,
            distance_threshold_m=args.path_mean_distance_threshold_m,
            max_distance_threshold_m=args.path_max_distance_threshold_m,
            length_delta_threshold_m=args.path_length_delta_threshold_m,
        )
        legacy_changed = path_changed(open_path, closed_path)
        result = {
            "benchmark_episode_index": episode_index,
            "benchmark_source_traj_key": episode.get("source", {}).get("traj_key"),
            "scene_path": str(ctx.env.current_model_path),
            "scene_dataset": episode["scene_dataset"],
            "data_split": episode["data_split"],
            "house_index": episode["house_index"],
            "target_object": ctx.task.config.task_config.pickup_obj_name,
            "target_candidates": ctx.task.config.task_config.pickup_obj_candidates,
            "robot_xy": start_xy.tolist(),
            "nav_goal": nav_goal.tolist(),
            "nav_goal_source": nav_goal_source,
            "nav_goal_sampling_error": nav_goal_sampling_error,
            "interactive_door_names": list(all_door_names),
            "interactive_door_count": len(all_door_names),
            "open_all_doors_transitions": open_all_transitions,
            "open_path_traversed_interactive_doors": traversed_door_names,
            "single_door_on_path_padding_m": args.door_on_path_padding_m,
            "single_door_path_candidates": single_door_results,
            "single_door_detour_results": qualifying_single_door_results,
            "single_door_detour_found": len(qualifying_single_door_results) > 0,
            "close_all_doors_transitions": close_all_transitions,
            "open_path_found": open_path is not None,
            "open_path_length_m": emi.path_length(open_path),
            "closed_path_found": closed_path is not None,
            "closed_path_length_m": emi.path_length(closed_path),
            "path_length_delta_m": path_length_delta(open_path, closed_path),
            "path_changed": legacy_changed,
            "path_changed_strict": all_closed_changed_strict,
            **all_closed_change_stats,
            "open_waypoints": None if open_path is None else open_path.tolist(),
            "closed_waypoints": None if closed_path is None else closed_path.tolist(),
            "open_plot_path": str(open_plot_path),
            "closed_plot_path": str(closed_plot_path),
            "episode_output_dir": str(episode_dir),
        }
        (episode_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return result
    finally:
        emi.close_context(ctx)


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "benchmark_episode_index",
        "benchmark_source_traj_key",
        "scene_dataset",
        "data_split",
        "house_index",
        "target_object",
        "interactive_door_count",
        "open_path_found",
        "open_path_length_m",
        "closed_path_found",
        "closed_path_length_m",
        "path_length_delta_m",
        "path_changed",
        "path_changed_strict",
        "mean_point_distance_m",
        "max_point_distance_m",
        "single_door_detour_found",
        "single_door_detour_count",
        "episode_output_dir",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["single_door_detour_count"] = len(row.get("single_door_detour_results", []))
            writer.writerow(csv_row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lightweight door-state GT path scans for benchmark nav episodes."
    )
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene_dataset", default="procthor-10k")
    parser.add_argument("--data_split", default="train")
    parser.add_argument("--house_ind", type=int, default=1)
    parser.add_argument("--variant", default="ceiling")
    parser.add_argument("--robot", default="rby1", choices=["rby1", "droid", "rum"])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--px_per_m", type=int, default=200)
    parser.add_argument("--downscale", type=int, default=5)
    parser.add_argument("--open_threshold", type=float, default=1e-3)
    parser.add_argument("--door_on_path_padding_m", type=float, default=0.2)
    parser.add_argument("--path_region_sample_step_m", type=float, default=0.05)
    parser.add_argument("--path_mean_distance_threshold_m", type=float, default=0.35)
    parser.add_argument("--path_max_distance_threshold_m", type=float, default=0.75)
    parser.add_argument("--path_length_delta_threshold_m", type=float, default=0.25)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    episodes = emi.load_benchmark_episodes(args.benchmark_dir)
    end = min(len(episodes), args.start_idx + args.max_episodes)
    selected = episodes[args.start_idx:end]

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for episode_index, episode in enumerate(selected, start=args.start_idx):
        try:
            results.append(run_single_episode(args, episode_index, episode, run_dir))
        except Exception as exc:
            failures.append(
                {
                    "benchmark_episode_index": episode_index,
                    "house_index": episode["house_index"],
                    "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                    "error": str(exc),
                }
            )

    summary = {
        "benchmark_dir": str(args.benchmark_dir),
        "start_idx": args.start_idx,
        "max_episodes": args.max_episodes,
        "processed_episode_count": len(results),
        "failed_episode_count": len(failures),
        "output_dir": str(run_dir),
        "results": results,
        "failures": failures,
        "path_changed_episode_indices": [
            row["benchmark_episode_index"] for row in results if row["path_changed"]
        ],
        "path_unchanged_episode_indices": [
            row["benchmark_episode_index"] for row in results if not row["path_changed"]
        ],
        "path_changed_strict_episode_indices": [
            row["benchmark_episode_index"] for row in results if row.get("path_changed_strict", False)
        ],
        "single_door_detour_episode_indices": [
            row["benchmark_episode_index"] for row in results if row.get("single_door_detour_found", False)
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_summary_csv(results, run_dir / "summary.csv")


if __name__ == "__main__":
    main()
