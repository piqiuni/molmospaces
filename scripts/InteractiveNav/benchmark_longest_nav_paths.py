from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav import benchmark_door_state_scan as door_scan
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.tasks.util_samplers.navgoal_sampler import NavGoalSampler
from molmo_spaces.utils.pose import pos_quat_to_pose_mat


DEFAULT_BENCHMARK_DIR = Path(
    REPO_ROOT
    / "assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts/InteractiveNav/output/benchmark_longest_nav_paths"


def make_episode_args(args: argparse.Namespace, episode: dict[str, Any], output_dir: Path) -> argparse.Namespace:
    robot_name = episode.get("robot", {}).get("robot_name", args.robot)
    return argparse.Namespace(
        command="benchmark-longest-nav-paths",
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


def target_category(target_object: str | None) -> str | None:
    if not target_object:
        return None
    return target_object.split("_", 1)[0]


def swap_benchmark_alias_prefix(object_name: str) -> str | None:
    aliases = {
        "trashcan_": "ashcan_",
        "ashcan_": "trashcan_",
    }
    for src, dst in aliases.items():
        if object_name.startswith(src):
            return dst + object_name[len(src) :]
    return None


def normalize_object_name(object_name: str, valid_names: set[str]) -> str:
    if object_name in valid_names:
        return object_name
    alias = swap_benchmark_alias_prefix(object_name)
    if alias is not None and alias in valid_names:
        return alias
    return object_name


def load_existing_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def result_path_for(output_dir: Path, episode_index: int, house_index: int) -> Path:
    return output_dir / "episodes" / f"benchmark_ep_{episode_index:04d}_house_{house_index}" / "result.json"


def set_episode_robot_pose(env, episode: dict[str, Any]) -> np.ndarray:
    robot_base_pose = np.asarray(episode["task"]["robot_base_pose"], dtype=float)
    env.current_robot.robot_view.base.pose = pos_quat_to_pose_mat(
        robot_base_pose[:3], robot_base_pose[3:7]
    )
    mujoco.mj_forward(env.current_model, env.current_data)
    return np.asarray(env.current_robot.robot_view.base.pose[:2, 3], dtype=float)


def episode_nav_objects(env, episode: dict[str, Any]) -> tuple[list[MlSpacesObject], list[str], str]:
    om = env.object_managers[env.current_batch_index]
    valid_names = set(env.current_scene_metadata.get("objects", {}).keys())

    target_name = normalize_object_name(episode["task"]["pickup_obj_name"], valid_names)
    candidate_names = episode["task"].get("pickup_obj_candidates") or [target_name]
    candidate_names = [normalize_object_name(name, valid_names) for name in candidate_names]

    target_category_name = None
    target_synset = None
    try:
        target_category_name = om.category_from_name(target_name)
        target_synset = om.get_annotation_synset(target_name)
    except Exception:
        pass

    filtered_names = []
    for name in candidate_names:
        if name in filtered_names:
            continue
        try:
            if target_category_name is None:
                filtered_names.append(name)
                continue
            obj_category = om.category_from_name(name)
            obj_synset = om.get_annotation_synset(name)
            if obj_category == target_category_name or (
                target_synset is not None and obj_synset == target_synset
            ):
                filtered_names.append(name)
        except Exception:
            continue
    if not filtered_names:
        filtered_names = candidate_names

    nav_objects = []
    resolved_names = []
    for name in filtered_names:
        try:
            nav_objects.append(MlSpacesObject(data=env.current_data, object_name=name))
            resolved_names.append(name)
        except Exception:
            continue
    if not nav_objects:
        raise ValueError(f"No valid nav target candidates for {target_name}")
    return nav_objects, resolved_names, target_name


def nearest_nav_object(env, nav_objects: list[MlSpacesObject]) -> MlSpacesObject:
    robot_xy = np.asarray(env.current_robot.robot_view.base.pose[:2, 3], dtype=float)
    return min(nav_objects, key=lambda obj: float(np.linalg.norm(obj.position[:2] - robot_xy)))


def sample_nav_goal_for_episode(
    env,
    scene_map,
    episode: dict[str, Any],
) -> tuple[np.ndarray, str, str | None, str, list[str]]:
    nav_objects, resolved_candidates, target_name = episode_nav_objects(env, episode)
    target_obj = nearest_nav_object(env, nav_objects)
    sampler = NavGoalSampler(scene_map, check_target_in_view=False, camera_name="head_camera")
    sampler.set_target(target_obj)
    sampler.set_robot_view(env.current_robot.robot_view)
    goal = sampler.sample()
    if goal is not None:
        return (
            emi.normalize_point3d(goal),
            "nav_goal_sampler",
            None,
            target_name,
            resolved_candidates,
        )

    target_pos = emi.normalize_point3d(target_obj.position)
    nearest_free_xy_goal = emi.nearest_free_point_xy(scene_map, target_pos[:2])
    if nearest_free_xy_goal is not None:
        fallback_goal = np.array(
            [nearest_free_xy_goal[0], nearest_free_xy_goal[1], float(target_pos[2])],
            dtype=float,
        )
        return (
            fallback_goal,
            "nearest_free_point_fallback",
            "Failed to sample a nav goal near target object",
            target_name,
            resolved_candidates,
        )

    return (
        target_pos,
        "target_object_center_fallback",
        "Failed to sample a nav goal and nearest free fallback",
        target_name,
        resolved_candidates,
    )


def run_single_episode(
    args: argparse.Namespace,
    episode_index: int,
    episode: dict[str, Any],
    output_dir: Path,
    save_plot: bool,
) -> dict[str, Any]:
    episode_dir = output_dir / "episodes" / f"benchmark_ep_{episode_index:04d}_house_{episode['house_index']}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_args = make_episode_args(args, episode, episode_dir)
    started_at = time.perf_counter()

    ctx = emi.load_context(episode_args, task_mode="nav_task")
    try:
        start_xy = emi.get_robot_xy(ctx.env)
        open_scene_map, doorway_analysis = emi.build_live_procthor_map(
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
        path = emi.compute_path_from_map(
            open_scene_map,
            start_xy,
            nav_goal[:2],
            downscale_factor=episode_args.downscale,
        )
        path_len = emi.path_length(path)
        euclidean_distance = float(np.linalg.norm(np.asarray(start_xy) - np.asarray(nav_goal[:2])))
        target_object = ctx.task.config.task_config.pickup_obj_name
        target_candidates = ctx.task.config.task_config.pickup_obj_candidates
        interactive_door_names = emi.interactive_door_root_names(doorway_analysis)
        plot_path = episode_dir / "open_gt_path.png"

        if save_plot:
            focus_records = emi.dedupe_plot_records(
                emi.collect_interactive_door_root_object_records(ctx.env, doorway_analysis)
                + emi.collect_non_interactive_doorway_object_records(ctx.env, doorway_analysis)
            )
            emi.save_door_path_figure(
                out_path=plot_path,
                scene_map=open_scene_map,
                door_records=[],
                object_records=focus_records,
                selected_doors=[],
                highlighted_object_names=[],
                start_xy=start_xy,
                goal_xy=nav_goal[:2],
                primary_path=path,
                primary_label="open-state GT path",
                title=(
                    f"Nav benchmark ep={episode_index} house={episode['house_index']} "
                    f"len={path_len:.2f}m target={target_category(target_object)}"
                ),
            )

        result = {
            "benchmark_episode_index": episode_index,
            "benchmark_source_traj_key": episode.get("source", {}).get("traj_key"),
            "benchmark_source_episode_length": episode.get("source", {}).get("episode_length"),
            "scene_path": str(ctx.env.current_model_path),
            "scene_dataset": episode["scene_dataset"],
            "data_split": episode["data_split"],
            "house_index": episode["house_index"],
            "target_object": target_object,
            "target_category": target_category(target_object),
            "target_candidates": target_candidates,
            "target_candidate_count": len(target_candidates or []),
            "robot_xy": np.asarray(start_xy).tolist(),
            "nav_goal": np.asarray(nav_goal).tolist(),
            "nav_goal_source": nav_goal_source,
            "nav_goal_sampling_error": nav_goal_sampling_error,
            "euclidean_distance_m": euclidean_distance,
            "open_path_found": path is not None,
            "open_path_length_m": path_len,
            "open_waypoint_count": None if path is None else int(len(path)),
            "interactive_door_count": len(interactive_door_names),
            "interactive_door_names": list(interactive_door_names),
            "episode_elapsed_sec": time.perf_counter() - started_at,
            "plot_path": str(plot_path) if save_plot else None,
            "episode_output_dir": str(episode_dir),
        }
        (episode_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return result
    finally:
        emi.close_context(ctx)


def run_episode_in_loaded_house(
    args: argparse.Namespace,
    ctx,
    scene_map,
    doorway_analysis: dict[str, Any] | None,
    episode_index: int,
    episode: dict[str, Any],
    output_dir: Path,
    save_plot: bool,
) -> dict[str, Any]:
    episode_dir = output_dir / "episodes" / f"benchmark_ep_{episode_index:04d}_house_{episode['house_index']}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    start_xy = set_episode_robot_pose(ctx.env, episode)
    rng_state = np.random.get_state()
    np.random.seed((int(args.seed) * 1_000_003 + int(episode_index)) % (2**32 - 1))
    try:
        nav_goal, nav_goal_source, nav_goal_sampling_error, target_object, resolved_candidates = (
            sample_nav_goal_for_episode(ctx.env, scene_map, episode)
        )
    finally:
        np.random.set_state(rng_state)
    path = emi.compute_path_from_map(
        scene_map,
        start_xy,
        nav_goal[:2],
        downscale_factor=args.downscale,
    )
    path_len = emi.path_length(path)
    euclidean_distance = float(np.linalg.norm(np.asarray(start_xy) - np.asarray(nav_goal[:2])))
    interactive_door_names = emi.interactive_door_root_names(doorway_analysis)
    plot_path = episode_dir / "open_gt_path.png"

    if save_plot:
        focus_records = emi.dedupe_plot_records(
            emi.collect_interactive_door_root_object_records(ctx.env, doorway_analysis)
            + emi.collect_non_interactive_doorway_object_records(ctx.env, doorway_analysis)
        )
        emi.save_door_path_figure(
            out_path=plot_path,
            scene_map=scene_map,
            door_records=[],
            object_records=focus_records,
            selected_doors=[],
            highlighted_object_names=[],
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=path,
            primary_label="open-state GT path",
            title=(
                f"Nav benchmark ep={episode_index} house={episode['house_index']} "
                f"len={path_len:.2f}m target={target_category(target_object)}"
            ),
        )

    result = {
        "benchmark_episode_index": episode_index,
        "benchmark_source_traj_key": episode.get("source", {}).get("traj_key"),
        "benchmark_source_episode_length": episode.get("source", {}).get("episode_length"),
        "scene_path": str(ctx.env.current_model_path),
        "scene_dataset": episode["scene_dataset"],
        "data_split": episode["data_split"],
        "house_index": episode["house_index"],
        "target_object": target_object,
        "target_category": target_category(target_object),
        "target_candidates": resolved_candidates,
        "target_candidate_count": len(resolved_candidates or []),
        "robot_xy": np.asarray(start_xy).tolist(),
        "nav_goal": np.asarray(nav_goal).tolist(),
        "nav_goal_source": nav_goal_source,
        "nav_goal_sampling_error": nav_goal_sampling_error,
        "euclidean_distance_m": euclidean_distance,
        "open_path_found": path is not None,
        "open_path_length_m": path_len,
        "open_waypoint_count": None if path is None else int(len(path)),
        "interactive_door_count": len(interactive_door_names),
        "interactive_door_names": list(interactive_door_names),
        "episode_elapsed_sec": time.perf_counter() - started_at,
        "plot_path": str(plot_path) if save_plot else None,
        "episode_output_dir": str(episode_dir),
    }
    (episode_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def run_house_group(
    args: argparse.Namespace,
    house_index: int,
    indexed_episodes: list[tuple[int, dict[str, Any]]],
    output_dir: Path,
    save_plot_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    save_plot_indices = save_plot_indices or set()
    ctx = None
    results = []
    failures = []
    first_episode = indexed_episodes[0][1]
    try:
        episode_args = make_episode_args(args, first_episode, output_dir)
        ctx = emi.load_context(episode_args, task_mode="scene_only")
        scene_map, doorway_analysis = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=episode_args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=episode_args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        for episode_index, episode in indexed_episodes:
            try:
                result = run_episode_in_loaded_house(
                    args=args,
                    ctx=ctx,
                    scene_map=scene_map,
                    doorway_analysis=doorway_analysis,
                    episode_index=episode_index,
                    episode=episode,
                    output_dir=output_dir,
                    save_plot=episode_index in save_plot_indices,
                )
                results.append(result)
                print(
                    f"[ok] ep={episode_index} house={house_index} "
                    f"path={result['open_path_length_m']} elapsed={result['episode_elapsed_sec']:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "benchmark_episode_index": episode_index,
                        "house_index": house_index,
                        "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                        "benchmark_source_episode_length": episode.get("source", {}).get("episode_length"),
                        "error": str(exc),
                    }
                )
                print(f"[fail] ep={episode_index} house={house_index} error={exc}", flush=True)
    except Exception as exc:
        for episode_index, episode in indexed_episodes:
            failures.append(
                {
                    "benchmark_episode_index": episode_index,
                    "house_index": house_index,
                    "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                    "benchmark_source_episode_length": episode.get("source", {}).get("episode_length"),
                    "error": f"house_load_failed: {exc}",
                }
            )
        print(f"[fail] house={house_index} error={exc}", flush=True)
    finally:
        if ctx is not None:
            emi.close_context(ctx)
    return results, failures


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "rank_by_open_path_length",
        "benchmark_episode_index",
        "house_index",
        "target_category",
        "target_object",
        "open_path_found",
        "open_path_length_m",
        "euclidean_distance_m",
        "open_waypoint_count",
        "benchmark_source_episode_length",
        "benchmark_source_traj_key",
        "nav_goal_source",
        "interactive_door_count",
        "target_candidate_count",
        "plot_path",
        "episode_output_dir",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {key: row.get(key) for key in fieldnames}
            out["rank_by_open_path_length"] = rank
            writer.writerow(out)


def make_montage(rows: list[dict[str, Any]], output_path: Path, columns: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image_rows = [row for row in rows if row.get("plot_path") and Path(row["plot_path"]).exists()]
    if not image_rows:
        return

    images = []
    for rank, row in enumerate(image_rows, start=1):
        img = Image.open(row["plot_path"]).convert("RGB")
        label_h = 72
        canvas = Image.new("RGB", (img.width, img.height + label_h), "white")
        canvas.paste(img, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 19)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        title = (
            f"#{rank} ep {row['benchmark_episode_index']} house {row['house_index']} "
            f"{row['open_path_length_m']:.2f}m"
        )
        subtitle = f"{row.get('target_category')} | source_steps={row.get('benchmark_source_episode_length')}"
        draw.text((16, 10), title, fill=(0, 0, 0), font=font)
        draw.text((16, 42), subtitle[:110], fill=(60, 60, 60), font=small_font)
        images.append(canvas)

    thumb_w = min(args_montage_max_width(output_path), max(img.width for img in images))
    resized = []
    for img in images:
        scale = thumb_w / img.width
        resized.append(img.resize((thumb_w, int(round(img.height * scale))), Image.Resampling.LANCZOS))

    columns = max(1, columns)
    rows_n = int(np.ceil(len(resized) / columns))
    cell_w = max(img.width for img in resized)
    cell_h = max(img.height for img in resized)
    montage = Image.new("RGB", (cell_w * columns, cell_h * rows_n), "white")
    for idx, img in enumerate(resized):
        row = idx // columns
        col = idx % columns
        montage.paste(img, (col * cell_w, row * cell_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)


def args_montage_max_width(_: Path) -> int:
    return 1800


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan nav benchmark episodes and plot the longest open-state GT paths."
    )
    parser.add_argument("--benchmark_dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot_top_only", action="store_true")
    parser.add_argument("--montage_columns", type=int, default=4)
    parser.add_argument("--variant", default="ceiling")
    parser.add_argument("--robot", default="rby1", choices=["rby1", "droid", "rum"])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--px_per_m", type=int, default=200)
    parser.add_argument("--downscale", type=int, default=5)
    parser.add_argument("--open_threshold", type=float, default=1e-3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    episodes = emi.load_benchmark_episodes(args.benchmark_dir)
    end_idx = len(episodes) if args.max_episodes is None else min(len(episodes), args.start_idx + args.max_episodes)
    selected = episodes[args.start_idx:end_idx]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for episode_index, episode in enumerate(selected, start=args.start_idx):
        cached = load_existing_json(result_path_for(args.output_dir, episode_index, episode["house_index"]))
        if args.resume and cached is not None and not args.plot_top_only:
            results.append(cached)
            continue
        grouped.setdefault(episode["house_index"], []).append((episode_index, episode))

    for house_index in sorted(grouped):
        house_results, house_failures = run_house_group(
            args=args,
            house_index=house_index,
            indexed_episodes=grouped[house_index],
            output_dir=args.output_dir,
        )
        results.extend(house_results)
        failures.extend(house_failures)

    path_results = [row for row in results if row.get("open_path_length_m") is not None]
    ranked = sorted(path_results, key=lambda row: row["open_path_length_m"], reverse=True)
    top_rows = ranked[: args.top_n]

    top_grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    top_indices = {row["benchmark_episode_index"] for row in top_rows}
    for row in top_rows:
        episode_index = row["benchmark_episode_index"]
        episode = episodes[episode_index]
        top_grouped.setdefault(episode["house_index"], []).append((episode_index, episode))

    plotted_top_rows: list[dict[str, Any]] = []
    for house_index in sorted(top_grouped):
        house_results, house_failures = run_house_group(
            args=args,
            house_index=house_index,
            indexed_episodes=top_grouped[house_index],
            output_dir=args.output_dir,
            save_plot_indices=top_indices,
        )
        plotted_top_rows.extend(house_results)
        for failure in house_failures:
            failure["error"] = f"plot_top_failed: {failure['error']}"
        failures.extend(house_failures)
    if plotted_top_rows:
        plotted_by_index = {row["benchmark_episode_index"]: row for row in plotted_top_rows}
        top_rows = [plotted_by_index.get(row["benchmark_episode_index"], row) for row in top_rows]

    total_elapsed = time.perf_counter() - started_at
    summary = {
        "benchmark_dir": str(args.benchmark_dir),
        "output_dir": str(args.output_dir),
        "start_idx": args.start_idx,
        "end_idx_exclusive": end_idx,
        "episode_count_requested": len(selected),
        "processed_episode_count": len(results),
        "path_found_episode_count": len(path_results),
        "failed_episode_count": len(failures),
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec_per_processed_episode": (
            float(np.mean([row.get("episode_elapsed_sec", 0.0) for row in results])) if results else None
        ),
        "top_n": args.top_n,
        "top_by_open_path_length": top_rows,
        "failures": failures,
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "all_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
    write_csv(ranked, args.output_dir / "all_results_ranked_by_open_path_length.csv")
    write_csv(top_rows, args.output_dir / f"top_{args.top_n:02d}_open_path_length.csv")
    make_montage(
        top_rows,
        args.output_dir / f"top_{args.top_n:02d}_open_path_length_montage.png",
        columns=args.montage_columns,
    )

    print(
        json.dumps(
            {
                "processed": len(results),
                "path_found": len(path_results),
                "failed": len(failures),
                "total_elapsed_sec": total_elapsed,
                "top_csv": str(args.output_dir / f"top_{args.top_n:02d}_open_path_length.csv"),
                "montage": str(args.output_dir / f"top_{args.top_n:02d}_open_path_length_montage.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
