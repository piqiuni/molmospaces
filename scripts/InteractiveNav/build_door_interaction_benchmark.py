from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for lightweight environments.
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import benchmark_door_state_scan as door_scan
from scripts.InteractiveNav import benchmark_longest_nav_paths as nav_paths
from scripts.InteractiveNav import explore_molmo_interactions as emi


DEFAULT_BENCHMARK_DIR = Path(
    "/home/user/ldl/molmospaces/assets/benchmarks/molmospaces-bench-v2/"
    "procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/output/"
    "door_interaction_benchmark_preview"
)


def progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def safe_case_id(episode_index: int, house_index: int) -> str:
    return f"ep_{episode_index:04d}_house_{house_index}"


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return slug[:max_len] if len(slug) > max_len else slug


def episode_dir_for(output_dir: Path, episode_index: int, house_index: int) -> Path:
    return output_dir / "episodes" / safe_case_id(episode_index, house_index)


def result_path_for(output_dir: Path, episode_index: int, house_index: int) -> Path:
    return episode_dir_for(output_dir, episode_index, house_index) / "critical_preview.json"


def build_result_path_for(output_dir: Path, episode_index: int, house_index: int) -> Path:
    return episode_dir_for(output_dir, episode_index, house_index) / "scan_result.json"


def load_existing_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_episode_args(args: argparse.Namespace, episode: dict[str, Any], output_dir: Path) -> argparse.Namespace:
    robot_name = episode.get("robot", {}).get("robot_name", args.robot)
    return argparse.Namespace(
        command="door-interaction-benchmark-critical-preview",
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


def group_episodes(episodes: list[dict[str, Any]], start_idx: int, max_episodes: int | None) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    end_idx = len(episodes) if max_episodes is None else min(len(episodes), start_idx + max_episodes)
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for episode_index, episode in enumerate(episodes[start_idx:end_idx], start=start_idx):
        grouped.setdefault(int(episode["house_index"]), []).append((episode_index, episode))
    return grouped


def door_record_to_preview_dict(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "door_name": rec["name"],
        "hinge_body_names": list(rec.get("children", [])),
        "aabb_center": np.asarray(rec["aabb_center"], dtype=float).tolist(),
        "aabb_size": np.asarray(rec["aabb_size"], dtype=float).tolist(),
    }


def unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def door_distance_to_start(record: dict[str, Any], start_xy: np.ndarray) -> float:
    center = np.asarray(record.get("aabb_center", record.get("position", [np.inf, np.inf])), dtype=float)
    return float(np.linalg.norm(center[:2] - np.asarray(start_xy, dtype=float)[:2]))


def sort_door_records_by_start_distance(
    records: list[dict[str, Any]],
    start_xy: np.ndarray,
) -> list[dict[str, Any]]:
    return sorted(records, key=lambda rec: (door_distance_to_start(rec, start_xy), rec.get("name", "")))


def door_distance_lookup(
    records: list[dict[str, Any]],
    start_xy: np.ndarray,
) -> dict[str, float]:
    return {rec["name"]: door_distance_to_start(rec, start_xy) for rec in records}


def load_existing_scan_index(output_dir: Path) -> dict[int, dict[str, Any]]:
    index_path = output_dir / "scan_index.jsonl"
    if not index_path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with open(index_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = row.get("parent_benchmark_episode_index")
            if episode_index is not None:
                rows[int(episode_index)] = row
    return rows


def case_dir_for(output_dir: Path, case_id: str) -> Path:
    return output_dir / "samples" / case_id


def sample_json_path_for(output_dir: Path, case_id: str) -> Path:
    return case_dir_for(output_dir, case_id) / "sample.json"


def sample_plot_path_for(output_dir: Path, case_id: str) -> Path:
    return case_dir_for(output_dir, case_id) / "path.png"


def compute_path_for_current_state(args: argparse.Namespace, ctx, goal_xy: np.ndarray):
    scene_map, analysis = emi.build_live_procthor_map(
        ctx.env.current_model,
        ctx.env.current_data,
        model_path=str(ctx.env.current_model_path),
        px_per_m=args.px_per_m,
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
        open_threshold=args.open_threshold,
        treat_all_non_interactive_doorways_as_open=True,
        return_doorway_analysis=True,
    )
    start_xy = emi.get_robot_xy(ctx.env)
    path = emi.compute_path_from_map(
        scene_map,
        start_xy,
        goal_xy,
        downscale_factor=args.downscale,
    )
    return scene_map, analysis, path


def make_interactive_nav_payload(
    args: argparse.Namespace,
    *,
    case_id: str,
    case_type: str,
    parent_episode_index: int,
    closed_doors: list[str],
    open_doors: list[str],
    required_open_doors: list[str],
    distractor_closed_doors: list[str],
    sampling: dict[str, Any] | None,
    open_path,
    initial_path,
    oracle_path,
    all_closed_path,
    all_closed_changed_strict: bool,
    all_closed_change_stats: dict[str, Any],
    critical_door_names: list[str],
    noncritical_door_names: list[str],
    plot_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": "door_interaction_nav_v1",
        "benchmark_type": "door_interaction_nav",
        "case_id": case_id,
        "case_type": case_type,
        "parent_benchmark_episode_index": parent_episode_index,
        "door_state": {
            "closed_doors": list(closed_doors),
            "open_doors": list(open_doors),
        },
        "oracle": {
            "required_open_doors": list(required_open_doors),
            "distractor_closed_doors": list(distractor_closed_doors),
            "expected_static_path_found": initial_path is not None,
            "expected_after_oracle_path_found": oracle_path is not None,
        },
        "paths": {
            "all_open_path_length_m": emi.path_length(open_path),
            "initial_state_path_found": initial_path is not None,
            "initial_state_path_length_m": emi.path_length(initial_path),
            "all_closed_path_found": all_closed_path is not None,
            "all_closed_path_length_m": emi.path_length(all_closed_path),
            "oracle_restored_path_found": oracle_path is not None,
            "oracle_restored_path_length_m": emi.path_length(oracle_path),
        },
        "diagnostics": {
            "all_closed_path_changed_strict": all_closed_changed_strict,
            **all_closed_change_stats,
            "critical_door_names": list(critical_door_names),
            "noncritical_interactive_door_names": list(noncritical_door_names),
            "critical_door_definition": "interactive door root boxes traversed by P_open",
            "door_on_path_padding_m": args.door_on_path_padding_m,
            "path_region_sample_step_m": args.path_region_sample_step_m,
        },
        "sampling": sampling,
        "plot_path": None if plot_path is None else str(plot_path),
    }


def write_sample_json(
    output_dir: Path,
    original_episode: dict[str, Any],
    interactive_nav: dict[str, Any],
) -> Path:
    case_id = interactive_nav["case_id"]
    out_path = sample_json_path_for(output_dir, case_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = copy.deepcopy(original_episode)
    sample["interactive_nav"] = interactive_nav
    out_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n")
    return out_path


def maybe_save_case_plot(
    args: argparse.Namespace,
    *,
    ctx,
    out_path: Path,
    scene_map,
    doorway_analysis: dict[str, Any] | None,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    path,
    open_path,
    case_type: str,
    case_id: str,
    required_open_doors: list[str],
) -> Path | None:
    if not args.save_plots:
        return None
    if args.plot_positive_only and case_type == "distractor_doors_closed":
        return None
    focus_records = emi.dedupe_plot_records(
        emi.collect_interactive_door_root_object_records(ctx.env, doorway_analysis)
        + emi.collect_non_interactive_doorway_object_records(ctx.env, doorway_analysis)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emi.save_door_path_figure(
        out_path=out_path,
        scene_map=scene_map,
        door_records=[],
        object_records=focus_records,
        selected_doors=[],
        highlighted_object_names=required_open_doors,
        start_xy=start_xy,
        goal_xy=goal_xy,
        primary_path=path,
        primary_label=f"{case_type} initial path",
        secondary_path=open_path,
        secondary_label="all-open GT path",
        title=f"{case_type} | {case_id} | required={len(required_open_doors)}",
    )
    return out_path


def select_nearby_doors(
    pool: list[str],
    k_min: int,
    k_max: int,
    sample_index: int,
) -> tuple[list[str], dict[str, Any]]:
    if not pool:
        return [], {
            "method": "nearest_by_robot_start_distance",
            "candidate_pool": [],
            "requested_k_min": k_min,
            "requested_k_max": k_max,
            "selected_k": 0,
            "skip_reason": "empty_candidate_pool",
        }
    effective_k_min = min(max(k_min, 0), len(pool))
    effective_k_max = min(max(k_max, effective_k_min), len(pool))
    if effective_k_max <= 0:
        return [], {
            "method": "nearest_by_robot_start_distance",
            "candidate_pool": list(pool),
            "requested_k_min": k_min,
            "requested_k_max": k_max,
            "selected_k": 0,
            "skip_reason": "non_positive_k_range",
        }
    selected_k = min(effective_k_min + max(sample_index, 0), effective_k_max)
    selected = list(pool[:selected_k])
    return selected, {
        "method": "nearest_by_robot_start_distance",
        "candidate_pool": list(pool),
        "requested_k_min": k_min,
        "requested_k_max": k_max,
        "effective_k_min": effective_k_min,
        "effective_k_max": effective_k_max,
        "sample_index": sample_index,
        "selected_k": selected_k,
        "selected_doors": selected,
    }


def make_case_id(episode_index: int, house_index: int, case_type: str, suffix: str) -> str:
    suffix_slug = safe_slug(suffix)
    return f"ep_{episode_index:04d}_house_{house_index}_{case_type}_{suffix_slug}"


def stable_case_seed(base_seed: int, episode_index: int, case_serial: int) -> int:
    return int((base_seed * 1_000_003 + episode_index * 9_176 + case_serial * 101 + 17) % (2**32 - 1))


def apply_closed_door_state(
    ctx,
    doorway_analysis: dict[str, Any] | None,
    closed_doors: list[str],
) -> list[dict[str, Any]]:
    transitions = emi.open_all_doors(ctx.env)
    for door_name in closed_doors:
        transitions.append(emi.set_door_root_state(ctx.env, doorway_analysis, door_name, "closed"))
    return transitions


def build_case_sample(
    args: argparse.Namespace,
    *,
    ctx,
    original_episode: dict[str, Any],
    episode_index: int,
    output_dir: Path,
    case_id: str,
    case_type: str,
    closed_doors: list[str],
    required_open_doors: list[str],
    distractor_closed_doors: list[str],
    sampling: dict[str, Any] | None,
    all_interactive_door_names: list[str],
    critical_door_names: list[str],
    noncritical_door_names: list[str],
    goal_xy: np.ndarray,
    open_path,
    all_closed_path,
    all_closed_changed_strict: bool,
    all_closed_change_stats: dict[str, Any],
) -> dict[str, Any]:
    house_index = int(original_episode["house_index"])
    closed_doors = unique_preserve_order(closed_doors)
    required_open_doors = unique_preserve_order(required_open_doors)
    distractor_closed_doors = unique_preserve_order(distractor_closed_doors)
    open_doors = [name for name in all_interactive_door_names if name not in set(closed_doors)]

    apply_closed_door_state(ctx, None, [])
    current_analysis = emi.collect_runtime_doorway_analysis(ctx.env)
    apply_closed_door_state(ctx, current_analysis, closed_doors)
    start_xy = nav_paths.set_episode_robot_pose(ctx.env, original_episode)
    initial_scene_map, initial_analysis, initial_path = compute_path_for_current_state(args, ctx, goal_xy)

    oracle_closed_doors = [name for name in closed_doors if name not in set(required_open_doors)]
    apply_closed_door_state(ctx, initial_analysis, oracle_closed_doors)
    nav_paths.set_episode_robot_pose(ctx.env, original_episode)
    _, _, oracle_path = compute_path_for_current_state(args, ctx, goal_xy)

    plot_path = maybe_save_case_plot(
        args,
        ctx=ctx,
        out_path=sample_plot_path_for(output_dir, case_id),
        scene_map=initial_scene_map,
        doorway_analysis=initial_analysis,
        start_xy=start_xy,
        goal_xy=goal_xy,
        path=initial_path,
        open_path=open_path,
        case_type=case_type,
        case_id=case_id,
        required_open_doors=required_open_doors,
    )
    interactive_nav = make_interactive_nav_payload(
        args,
        case_id=case_id,
        case_type=case_type,
        parent_episode_index=episode_index,
        closed_doors=closed_doors,
        open_doors=open_doors,
        required_open_doors=required_open_doors,
        distractor_closed_doors=distractor_closed_doors,
        sampling=sampling,
        open_path=open_path,
        initial_path=initial_path,
        oracle_path=oracle_path,
        all_closed_path=all_closed_path,
        all_closed_changed_strict=all_closed_changed_strict,
        all_closed_change_stats=all_closed_change_stats,
        critical_door_names=critical_door_names,
        noncritical_door_names=noncritical_door_names,
        plot_path=plot_path,
    )
    sample_path = write_sample_json(output_dir, original_episode, interactive_nav)
    return {
        "case_id": case_id,
        "case_type": case_type,
        "sample_path": str(sample_path),
        "plot_path": None if plot_path is None else str(plot_path),
        "closed_door_count": len(closed_doors),
        "required_open_door_count": len(required_open_doors),
        "initial_path_found": initial_path is not None,
        "oracle_restored_path_found": oracle_path is not None,
        "initial_path_length_m": emi.path_length(initial_path),
        "oracle_restored_path_length_m": emi.path_length(oracle_path),
        "house_index": house_index,
    }


def deterministic_nav_goal_for_episode(
    args: argparse.Namespace,
    ctx,
    scene_map,
    episode_index: int,
    episode: dict[str, Any],
) -> tuple[np.ndarray, str, str | None, str, list[str]]:
    rng_state = np.random.get_state()
    np.random.seed((int(args.seed) * 1_000_003 + int(episode_index)) % (2**32 - 1))
    try:
        return nav_paths.sample_nav_goal_for_episode(ctx.env, scene_map, episode)
    finally:
        np.random.set_state(rng_state)


def run_episode_build(
    args: argparse.Namespace,
    ctx,
    open_scene_map,
    open_analysis: dict[str, Any] | None,
    episode_index: int,
    episode: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    house_index = int(episode["house_index"])
    episode_dir = episode_dir_for(output_dir, episode_index, house_index)
    episode_dir.mkdir(parents=True, exist_ok=True)

    emi.open_all_doors(ctx.env)
    start_xy = nav_paths.set_episode_robot_pose(ctx.env, episode)
    nav_goal, nav_goal_source, nav_goal_sampling_error, target_object, resolved_candidates = (
        deterministic_nav_goal_for_episode(args, ctx, open_scene_map, episode_index, episode)
    )
    open_path = emi.compute_path_from_map(
        open_scene_map,
        start_xy,
        nav_goal[:2],
        downscale_factor=args.downscale,
    )

    all_interactive_records = sort_door_records_by_start_distance(
        emi.collect_interactive_door_root_object_records(ctx.env, open_analysis),
        start_xy,
    )
    all_interactive_door_names = [rec["name"] for rec in all_interactive_records]
    critical_records = door_scan.traversed_interactive_doors_on_path(
        ctx.env,
        open_analysis,
        open_path,
        padding_m=args.door_on_path_padding_m,
        sample_step_m=args.path_region_sample_step_m,
    )
    critical_records = sort_door_records_by_start_distance(critical_records, start_xy)
    critical_door_names = [rec["name"] for rec in critical_records]
    critical_set = set(critical_door_names)
    noncritical_door_names = [name for name in all_interactive_door_names if name not in critical_set]
    interactive_door_distances_m = door_distance_lookup(all_interactive_records, start_xy)

    emi.close_all_doors(ctx.env)
    nav_paths.set_episode_robot_pose(ctx.env, episode)
    all_closed_scene_map, all_closed_analysis, all_closed_path = compute_path_for_current_state(
        args, ctx, nav_goal[:2]
    )
    all_closed_changed_strict, all_closed_change_stats = door_scan.path_changed_strict(
        open_path,
        all_closed_path,
        distance_threshold_m=args.path_mean_distance_threshold_m,
        max_distance_threshold_m=args.path_max_distance_threshold_m,
        length_delta_threshold_m=args.path_length_delta_threshold_m,
    )

    has_interaction = bool(open_path is not None and all_closed_changed_strict and critical_door_names)
    skip_reason = None
    if open_path is None:
        skip_reason = "open_path_missing"
    elif not all_closed_changed_strict:
        skip_reason = "all_closed_path_unchanged_strict"
    elif not critical_door_names:
        skip_reason = "no_critical_door_on_open_path"

    case_summaries: list[dict[str, Any]] = []
    built_all_closed = False
    built_partial_count = 0
    seen_closed_state_keys: set[tuple[str, ...]] = set()

    if has_interaction:
        all_closed_case_id = make_case_id(episode_index, house_index, "all_closed", "all")
        seen_closed_state_keys.add(tuple(sorted(all_interactive_door_names)))
        case_summaries.append(
            build_case_sample(
                args,
                ctx=ctx,
                original_episode=episode,
                episode_index=episode_index,
                output_dir=output_dir,
                case_id=all_closed_case_id,
                case_type="all_closed",
                closed_doors=all_interactive_door_names,
                required_open_doors=critical_door_names,
                distractor_closed_doors=noncritical_door_names,
                sampling={
                    "sampling_seed": stable_case_seed(args.sampling_seed, episode_index, 0),
                    "method": "deterministic_all_interactive_doors",
                },
                all_interactive_door_names=all_interactive_door_names,
                critical_door_names=critical_door_names,
                noncritical_door_names=noncritical_door_names,
                goal_xy=nav_goal[:2],
                open_path=open_path,
                all_closed_path=all_closed_path,
                all_closed_changed_strict=all_closed_changed_strict,
                all_closed_change_stats=all_closed_change_stats,
            )
        )
        built_all_closed = True

        for door_name in critical_door_names:
            closed_key = tuple(sorted([door_name]))
            if closed_key in seen_closed_state_keys:
                continue
            seen_closed_state_keys.add(closed_key)
            case_id = make_case_id(episode_index, house_index, "single_path_door_closed", door_name)
            case_summaries.append(
                build_case_sample(
                    args,
                    ctx=ctx,
                    original_episode=episode,
                    episode_index=episode_index,
                    output_dir=output_dir,
                    case_id=case_id,
                    case_type="single_path_door_closed",
                    closed_doors=[door_name],
                    required_open_doors=[door_name],
                    distractor_closed_doors=[],
                    sampling={
                        "sampling_seed": stable_case_seed(args.sampling_seed, episode_index, len(case_summaries)),
                        "method": "one_critical_door_from_open_path_nearest_first",
                        "selected_critical_door": door_name,
                        "door_distance_m": interactive_door_distances_m.get(door_name),
                    },
                    all_interactive_door_names=all_interactive_door_names,
                    critical_door_names=critical_door_names,
                    noncritical_door_names=noncritical_door_names,
                    goal_xy=nav_goal[:2],
                    open_path=open_path,
                    all_closed_path=all_closed_path,
                    all_closed_changed_strict=all_closed_changed_strict,
                    all_closed_change_stats=all_closed_change_stats,
                )
            )
            built_partial_count += 1

        distractor_sample_count = args.num_distractor_samples_per_episode if critical_door_names else 0
        for sample_idx in range(distractor_sample_count):
            seed = stable_case_seed(args.sampling_seed, episode_index, 10_000 + sample_idx)
            distractors, sampling = select_nearby_doors(
                noncritical_door_names,
                args.distractor_k_min,
                args.distractor_k_max,
                sample_idx,
            )
            sampling["sampling_seed"] = seed
            sampling["door_distances_m"] = {
                name: interactive_door_distances_m.get(name) for name in distractors
            }
            if not distractors and args.distractor_k_min > 0:
                continue
            closed_key = tuple(sorted(distractors))
            if closed_key in seen_closed_state_keys:
                continue
            seen_closed_state_keys.add(closed_key)
            case_id = make_case_id(
                episode_index,
                house_index,
                "distractor_doors_closed",
                f"s{sample_idx}_k{len(distractors)}",
            )
            case_summaries.append(
                build_case_sample(
                    args,
                    ctx=ctx,
                    original_episode=episode,
                    episode_index=episode_index,
                    output_dir=output_dir,
                    case_id=case_id,
                    case_type="distractor_doors_closed",
                    closed_doors=distractors,
                    required_open_doors=[],
                    distractor_closed_doors=distractors,
                    sampling=sampling,
                    all_interactive_door_names=all_interactive_door_names,
                    critical_door_names=critical_door_names,
                    noncritical_door_names=noncritical_door_names,
                    goal_xy=nav_goal[:2],
                    open_path=open_path,
                    all_closed_path=all_closed_path,
                    all_closed_changed_strict=all_closed_changed_strict,
                    all_closed_change_stats=all_closed_change_stats,
                )
            )
            built_partial_count += 1

        for door_name in critical_door_names:
            for sample_idx in range(args.num_mixed_samples_per_critical_door):
                seed = stable_case_seed(
                    args.sampling_seed,
                    episode_index,
                    20_000 + len(case_summaries) * 17 + sample_idx,
                )
                distractors, sampling = select_nearby_doors(
                    noncritical_door_names,
                    args.distractor_k_min,
                    args.distractor_k_max,
                    sample_idx,
                )
                sampling["sampling_seed"] = seed
                sampling["selected_critical_door"] = door_name
                sampling["selected_critical_door_distance_m"] = interactive_door_distances_m.get(door_name)
                sampling["door_distances_m"] = {
                    name: interactive_door_distances_m.get(name) for name in [door_name] + distractors
                }
                if not distractors and args.distractor_k_min > 0:
                    continue
                closed = unique_preserve_order([door_name] + distractors)
                closed_key = tuple(closed)
                if closed_key in seen_closed_state_keys:
                    continue
                seen_closed_state_keys.add(closed_key)
                case_id = make_case_id(
                    episode_index,
                    house_index,
                    "mixed_critical_and_distractor_closed",
                    f"{door_name}_s{sample_idx}_k{len(distractors)}",
                )
                case_summaries.append(
                    build_case_sample(
                        args,
                        ctx=ctx,
                        original_episode=episode,
                        episode_index=episode_index,
                        output_dir=output_dir,
                        case_id=case_id,
                        case_type="mixed_critical_and_distractor_closed",
                        closed_doors=closed,
                        required_open_doors=[door_name],
                        distractor_closed_doors=distractors,
                        sampling=sampling,
                        all_interactive_door_names=all_interactive_door_names,
                        critical_door_names=critical_door_names,
                        noncritical_door_names=noncritical_door_names,
                        goal_xy=nav_goal[:2],
                        open_path=open_path,
                        all_closed_path=all_closed_path,
                        all_closed_changed_strict=all_closed_changed_strict,
                        all_closed_change_stats=all_closed_change_stats,
                    )
                )
                built_partial_count += 1

    row = {
        "schema_version": "door_interaction_nav_scan_v1",
        "mode": "build",
        "parent_benchmark_episode_index": episode_index,
        "case_id": safe_case_id(episode_index, house_index),
        "scene_dataset": episode["scene_dataset"],
        "data_split": episode["data_split"],
        "house_index": house_index,
        "target_object": target_object,
        "target_category": nav_paths.target_category(target_object),
        "target_candidates": resolved_candidates,
        "robot_xy": np.asarray(start_xy).tolist(),
        "nav_goal": np.asarray(nav_goal).tolist(),
        "nav_goal_source": nav_goal_source,
        "nav_goal_sampling_error": nav_goal_sampling_error,
        "open_path_found": open_path is not None,
        "open_path_length_m": emi.path_length(open_path),
        "all_closed_path_found": all_closed_path is not None,
        "all_closed_path_length_m": emi.path_length(all_closed_path),
        "all_closed_path_changed_strict": all_closed_changed_strict,
        **all_closed_change_stats,
        "has_interaction": has_interaction,
        "skip_reason": skip_reason,
        "interactive_door_count": len(all_interactive_door_names),
        "critical_door_count": len(critical_door_names),
        "critical_door_names": critical_door_names,
        "noncritical_interactive_door_count": len(noncritical_door_names),
        "noncritical_interactive_door_names": noncritical_door_names,
        "interactive_door_distances_m": interactive_door_distances_m,
        "door_order_rule": "interactive and critical doors are ordered by distance from the benchmark robot start pose",
        "built_all_closed_record": built_all_closed,
        "built_partial_record_count": built_partial_count,
        "built_case_count": len(case_summaries),
        "built_case_ids": [case["case_id"] for case in case_summaries],
        "case_summaries": case_summaries,
        "path_change_thresholds": {
            "mean_distance_m": args.path_mean_distance_threshold_m,
            "max_distance_m": args.path_max_distance_threshold_m,
            "length_delta_m": args.path_length_delta_threshold_m,
        },
        "sampling_config": {
            "sampling_seed": args.sampling_seed,
            "num_distractor_samples_per_episode": args.num_distractor_samples_per_episode,
            "num_mixed_samples_per_critical_door": args.num_mixed_samples_per_critical_door,
            "distractor_k_min": args.distractor_k_min,
            "distractor_k_max": args.distractor_k_max,
        },
        "episode_output_dir": str(episode_dir),
        "elapsed_sec": time.perf_counter() - started_at,
    }
    build_result_path_for(output_dir, episode_index, house_index).write_text(
        json.dumps(row, indent=2, ensure_ascii=False) + "\n"
    )
    return row


def run_episode_critical_preview(
    args: argparse.Namespace,
    ctx,
    scene_map,
    doorway_analysis: dict[str, Any] | None,
    episode_index: int,
    episode: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    episode_dir = episode_dir_for(output_dir, episode_index, int(episode["house_index"]))
    episode_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    emi.open_all_doors(ctx.env)
    start_xy = nav_paths.set_episode_robot_pose(ctx.env, episode)

    rng_state = np.random.get_state()
    np.random.seed((int(args.seed) * 1_000_003 + int(episode_index)) % (2**32 - 1))
    try:
        nav_goal, nav_goal_source, nav_goal_sampling_error, target_object, resolved_candidates = (
            nav_paths.sample_nav_goal_for_episode(ctx.env, scene_map, episode)
        )
    finally:
        np.random.set_state(rng_state)

    open_path = emi.compute_path_from_map(
        scene_map,
        start_xy,
        nav_goal[:2],
        downscale_factor=args.downscale,
    )
    all_interactive_door_records = emi.collect_interactive_door_root_object_records(
        ctx.env, doorway_analysis
    )
    all_noninteractive_door_records = emi.collect_non_interactive_doorway_object_records(
        ctx.env, doorway_analysis
    )
    critical_records = door_scan.traversed_interactive_doors_on_path(
        ctx.env,
        doorway_analysis,
        open_path,
        padding_m=args.door_on_path_padding_m,
        sample_step_m=args.path_region_sample_step_m,
    )
    critical_door_names = [rec["name"] for rec in critical_records]
    noncritical_door_names = [
        rec["name"] for rec in all_interactive_door_records if rec["name"] not in set(critical_door_names)
    ]

    plot_path = episode_dir / "critical_doors_preview.png"
    focus_records = emi.dedupe_plot_records(
        all_interactive_door_records + all_noninteractive_door_records
    )
    emi.save_door_path_figure(
        out_path=plot_path,
        scene_map=scene_map,
        door_records=[],
        object_records=focus_records,
        selected_doors=[],
        highlighted_object_names=critical_door_names,
        start_xy=start_xy,
        goal_xy=nav_goal[:2],
        primary_path=open_path,
        primary_label="all-open GT path",
        title=(
            f"Critical Door Preview | ep={episode_index} house={episode['house_index']} "
            f"critical={len(critical_door_names)} target={nav_paths.target_category(target_object)}"
        ),
    )

    result = {
        "schema_version": "door_interaction_benchmark_preview_v1",
        "mode": "critical_preview",
        "parent_benchmark_episode_index": episode_index,
        "case_id": safe_case_id(episode_index, int(episode["house_index"])),
        "scene_dataset": episode["scene_dataset"],
        "data_split": episode["data_split"],
        "house_index": episode["house_index"],
        "target_object": target_object,
        "target_category": nav_paths.target_category(target_object),
        "target_candidates": resolved_candidates,
        "robot_xy": np.asarray(start_xy).tolist(),
        "nav_goal": np.asarray(nav_goal).tolist(),
        "nav_goal_source": nav_goal_source,
        "nav_goal_sampling_error": nav_goal_sampling_error,
        "open_path_found": open_path is not None,
        "open_path_length_m": emi.path_length(open_path),
        "open_waypoint_count": None if open_path is None else int(len(open_path)),
        "interactive_door_count": len(all_interactive_door_records),
        "critical_door_count": len(critical_door_names),
        "critical_door_names": critical_door_names,
        "critical_door_records": [door_record_to_preview_dict(rec) for rec in critical_records],
        "noncritical_interactive_door_names": noncritical_door_names,
        "path_door_rule": {
            "definition": "critical doors are interactive door root boxes traversed by P_open",
            "door_on_path_padding_m": args.door_on_path_padding_m,
            "path_region_sample_step_m": args.path_region_sample_step_m,
        },
        "plot_path": str(plot_path),
        "episode_output_dir": str(episode_dir),
        "elapsed_sec": time.perf_counter() - started_at,
    }
    result_path_for(output_dir, episode_index, int(episode["house_index"])).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result


def run_house_group(
    args: argparse.Namespace,
    house_index: int,
    indexed_episodes: list[tuple[int, dict[str, Any]]],
    output_dir: Path,
    progress=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_episode = indexed_episodes[0][1]
    episode_args = build_episode_args(args, first_episode, output_dir)
    results = []
    failures = []
    ctx = None
    try:
        ctx = emi.load_context(episode_args, task_mode="scene_only")
        emi.open_all_doors(ctx.env)
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
                result = run_episode_critical_preview(
                    args=args,
                    ctx=ctx,
                    scene_map=scene_map,
                    doorway_analysis=doorway_analysis,
                    episode_index=episode_index,
                    episode=episode,
                    output_dir=output_dir,
                )
                results.append(result)
                progress_write(
                    f"[ok] ep={episode_index} house={house_index} "
                    f"path={result['open_path_length_m']} critical={result['critical_door_count']} "
                    f"elapsed={result['elapsed_sec']:.2f}s"
                )
            except Exception as exc:
                failure = {
                    "parent_benchmark_episode_index": episode_index,
                    "house_index": house_index,
                    "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                    "error": str(exc),
                }
                failures.append(failure)
                progress_write(f"[fail] ep={episode_index} house={house_index} error={exc}")
            finally:
                if progress is not None:
                    progress.update(1)
    except Exception as exc:
        for episode_index, episode in indexed_episodes:
            failures.append(
                {
                    "parent_benchmark_episode_index": episode_index,
                    "house_index": house_index,
                    "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                    "error": f"house_load_failed: {exc}",
                }
            )
        progress_write(f"[fail] house={house_index} error={exc}")
        if progress is not None:
            progress.update(len(indexed_episodes))
    finally:
        if ctx is not None:
            emi.close_context(ctx)
    return results, failures


def run_house_group_build(
    args: argparse.Namespace,
    house_index: int,
    indexed_episodes: list[tuple[int, dict[str, Any]]],
    output_dir: Path,
    progress=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_episode = indexed_episodes[0][1]
    episode_args = build_episode_args(args, first_episode, output_dir)
    results = []
    failures = []
    ctx = None
    try:
        ctx = emi.load_context(episode_args, task_mode="scene_only")
        emi.open_all_doors(ctx.env)
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
        for episode_index, episode in indexed_episodes:
            try:
                result = run_episode_build(
                    args=args,
                    ctx=ctx,
                    open_scene_map=open_scene_map,
                    open_analysis=open_analysis,
                    episode_index=episode_index,
                    episode=episode,
                    output_dir=output_dir,
                )
                results.append(result)
                progress_write(
                    f"[ok] ep={episode_index} house={house_index} "
                    f"has_interaction={result['has_interaction']} "
                    f"cases={result['built_case_count']} "
                    f"critical={result['critical_door_count']} "
                    f"elapsed={result['elapsed_sec']:.2f}s"
                )
            except Exception as exc:
                failure = {
                    "parent_benchmark_episode_index": episode_index,
                    "house_index": house_index,
                    "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                    "error": str(exc),
                }
                failures.append(failure)
                progress_write(f"[fail] ep={episode_index} house={house_index} error={exc}")
            finally:
                if progress is not None:
                    progress.update(1)
    except Exception as exc:
        for episode_index, episode in indexed_episodes:
            failures.append(
                {
                    "parent_benchmark_episode_index": episode_index,
                    "house_index": house_index,
                    "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                    "error": f"house_load_failed: {exc}",
                }
            )
        progress_write(f"[fail] house={house_index} error={exc}")
        if progress is not None:
            progress.update(len(indexed_episodes))
    finally:
        if ctx is not None:
            emi.close_context(ctx)
    return results, failures


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "parent_benchmark_episode_index",
        "case_id",
        "house_index",
        "target_category",
        "target_object",
        "open_path_found",
        "open_path_length_m",
        "interactive_door_count",
        "critical_door_count",
        "critical_door_names",
        "plot_path",
        "episode_output_dir",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            out["critical_door_names"] = json.dumps(row.get("critical_door_names", []), ensure_ascii=False)
            writer.writerow(out)


def write_scan_index_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "parent_benchmark_episode_index",
        "case_id",
        "house_index",
        "target_category",
        "target_object",
        "open_path_found",
        "open_path_length_m",
        "all_closed_path_found",
        "all_closed_path_length_m",
        "all_closed_path_changed_strict",
        "mean_point_distance_m",
        "max_point_distance_m",
        "path_length_delta_m",
        "has_interaction",
        "skip_reason",
        "interactive_door_count",
        "critical_door_count",
        "critical_door_names",
        "built_all_closed_record",
        "built_partial_record_count",
        "built_case_count",
        "built_case_ids",
        "episode_output_dir",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            out["critical_door_names"] = json.dumps(row.get("critical_door_names", []), ensure_ascii=False)
            out["built_case_ids"] = json.dumps(row.get("built_case_ids", []), ensure_ascii=False)
            writer.writerow(out)


def write_summary_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/preview door interaction benchmark data from MolmoSpaces nav episodes."
    )
    parser.add_argument("--benchmark_dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input_mode", choices=["original", "existing"], default="original")
    parser.add_argument("--mode", choices=["critical-preview", "build"], default="critical-preview")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
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
    parser.add_argument("--path_length_delta_threshold_m", type=float, default=0.5)
    parser.add_argument("--num_distractor_samples_per_episode", type=int, default=1)
    parser.add_argument("--num_mixed_samples_per_critical_door", type=int, default=1)
    parser.add_argument("--distractor_k_min", type=int, default=1)
    parser.add_argument("--distractor_k_max", type=int, default=5)
    parser.add_argument("--sampling_seed", type=int, default=20260708)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_positive_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = emi.load_benchmark_episodes(args.benchmark_dir)
    grouped = group_episodes(episodes, args.start_idx, args.max_episodes)
    total_episode_count = sum(len(indexed_episodes) for indexed_episodes in grouped.values())

    results = []
    failures = []
    started_at = time.perf_counter()
    existing_scan_index = load_existing_scan_index(args.output_dir) if args.input_mode == "existing" else {}

    if args.mode == "critical-preview":
        if args.input_mode != "original":
            raise NotImplementedError("--mode critical-preview currently supports --input_mode original only.")
        progress = (
            tqdm(
                total=total_episode_count,
                desc="critical-preview episodes",
                unit="ep",
                dynamic_ncols=True,
                mininterval=5.0,
                file=sys.stdout,
            )
            if tqdm is not None
            else None
        )
        for house_index in sorted(grouped):
            todo = []
            for episode_index, episode in grouped[house_index]:
                cached = load_existing_json(result_path_for(args.output_dir, episode_index, int(episode["house_index"])))
                if args.resume and cached is not None:
                    results.append(cached)
                    if progress is not None:
                        progress.update(1)
                else:
                    todo.append((episode_index, episode))
            if not todo:
                continue
            house_results, house_failures = run_house_group(
                args,
                house_index,
                todo,
                args.output_dir,
                progress=progress,
            )
            results.extend(house_results)
            failures.extend(house_failures)
        if progress is not None:
            progress.close()

        results.sort(key=lambda row: row["parent_benchmark_episode_index"])
        summary = {
            "schema_version": "door_interaction_benchmark_preview_summary_v1",
            "mode": args.mode,
            "input_mode": args.input_mode,
            "benchmark_dir": str(args.benchmark_dir),
            "output_dir": str(args.output_dir),
            "start_idx": args.start_idx,
            "max_episodes": args.max_episodes,
            "processed_episode_count": len(results),
            "failed_episode_count": len(failures),
            "path_found_episode_count": sum(1 for row in results if row.get("open_path_found")),
            "episodes_with_critical_doors": [
                row["parent_benchmark_episode_index"] for row in results if row.get("critical_door_count", 0) > 0
            ],
            "episodes_without_critical_doors": [
                row["parent_benchmark_episode_index"] for row in results if row.get("critical_door_count", 0) == 0
            ],
            "total_elapsed_sec": time.perf_counter() - started_at,
            "failures": failures,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        write_summary_csv(results, args.output_dir / "critical_preview_index.csv")
        write_summary_jsonl(results, args.output_dir / "critical_preview_index.jsonl")
        (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.mode != "build":
        raise ValueError(f"Unsupported mode: {args.mode}")

    progress = (
        tqdm(
            total=total_episode_count,
            desc="build episodes",
            unit="ep",
            dynamic_ncols=True,
            mininterval=5.0,
            file=sys.stdout,
        )
        if tqdm is not None
        else None
    )
    for house_index in sorted(grouped):
        todo = []
        for episode_index, episode in grouped[house_index]:
            cached = load_existing_json(build_result_path_for(args.output_dir, episode_index, int(episode["house_index"])))
            if cached is None:
                cached = existing_scan_index.get(episode_index)
            if args.resume and cached is not None:
                results.append(cached)
                if progress is not None:
                    progress.update(1)
            else:
                todo.append((episode_index, episode))
        if not todo:
            continue
        house_results, house_failures = run_house_group_build(
            args,
            house_index,
            todo,
            args.output_dir,
            progress=progress,
        )
        results.extend(house_results)
        failures.extend(house_failures)
    if progress is not None:
        progress.close()

    results.sort(key=lambda row: row["parent_benchmark_episode_index"])
    benchmark_samples = []
    missing_sample_paths = []
    for row in results:
        for case in row.get("case_summaries", []):
            sample_path = Path(case["sample_path"])
            if not sample_path.exists():
                missing_sample_paths.append(str(sample_path))
                continue
            benchmark_samples.append(json.loads(sample_path.read_text()))

    summary = {
        "schema_version": "door_interaction_benchmark_build_summary_v1",
        "mode": args.mode,
        "input_mode": args.input_mode,
        "benchmark_dir": str(args.benchmark_dir),
        "output_dir": str(args.output_dir),
        "start_idx": args.start_idx,
        "max_episodes": args.max_episodes,
        "processed_episode_count": len(results),
        "failed_episode_count": len(failures),
        "path_found_episode_count": sum(1 for row in results if row.get("open_path_found")),
        "interaction_episode_count": sum(1 for row in results if row.get("has_interaction")),
        "skipped_no_interaction_episode_count": sum(
            1 for row in results if row.get("skip_reason") == "all_closed_path_unchanged_strict"
        ),
        "built_sample_count": len(benchmark_samples),
        "built_all_closed_record_count": sum(1 for row in results if row.get("built_all_closed_record")),
        "built_partial_record_count": sum(int(row.get("built_partial_record_count", 0)) for row in results),
        "case_type_counts": {
            case_type: sum(
                1
                for row in results
                for case in row.get("case_summaries", [])
                if case.get("case_type") == case_type
            )
            for case_type in [
                "all_closed",
                "single_path_door_closed",
                "distractor_doors_closed",
                "mixed_critical_and_distractor_closed",
            ]
        },
        "sampling_config": {
            "sampling_seed": args.sampling_seed,
            "num_distractor_samples_per_episode": args.num_distractor_samples_per_episode,
            "num_mixed_samples_per_critical_door": args.num_mixed_samples_per_critical_door,
            "distractor_k_min": args.distractor_k_min,
            "distractor_k_max": args.distractor_k_max,
            "save_plots": args.save_plots,
            "plot_positive_only": args.plot_positive_only,
        },
        "missing_sample_paths": missing_sample_paths,
        "total_elapsed_sec": time.perf_counter() - started_at,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_scan_index_csv(results, args.output_dir / "scan_index.csv")
    write_summary_jsonl(results, args.output_dir / "scan_index.jsonl")
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(benchmark_samples, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
