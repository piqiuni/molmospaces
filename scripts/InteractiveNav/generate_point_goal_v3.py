"""Generate PointGoal InteractiveNav V3 demo episodes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import interactive_nav_v3 as v3
from scripts.InteractiveNav.interactive_nav_grounded_plan import (
    load_episodes,
    path_length,
    select_episode,
)


PathFn = Callable[[np.ndarray], np.ndarray | None]


def select_point_goal_candidate(
    candidates: Iterable[Iterable[float]],
    *,
    start_xy: Iterable[float],
    open_path_fn: PathFn,
    closed_path_fn: PathFn | None,
    rng: np.random.Generator,
    min_distance_m: float,
    max_distance_m: float | None,
    interaction_aware: bool,
    min_path_delta_m: float = 0.5,
    max_attempts: int = 2048,
) -> dict[str, Any]:
    points = np.asarray(list(candidates), dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        raise ValueError("PointGoal sampling requires at least one XY candidate")
    start = np.asarray(list(start_xy), dtype=float)[:2]
    order = rng.permutation(len(points))[: min(len(points), int(max_attempts))]
    failures: dict[str, int] = {}
    attempted_count = 0
    for candidate_index in order:
        attempted_count += 1
        goal = np.asarray(points[candidate_index, :2], dtype=float)
        straight_distance = float(np.linalg.norm(goal - start))
        if straight_distance < min_distance_m:
            failures["below_min_distance"] = failures.get("below_min_distance", 0) + 1
            continue
        if max_distance_m is not None and straight_distance > max_distance_m:
            failures["above_max_distance"] = failures.get("above_max_distance", 0) + 1
            continue
        open_path = open_path_fn(goal)
        if open_path is None:
            failures["open_path_missing"] = failures.get("open_path_missing", 0) + 1
            continue
        closed_path = closed_path_fn(goal) if closed_path_fn is not None else open_path
        open_length = float(path_length(open_path) or 0.0)
        closed_length = path_length(closed_path)
        if interaction_aware:
            if closed_path is not None and float(closed_length or 0.0) - open_length < min_path_delta_m:
                failures["interaction_not_needed"] = failures.get("interaction_not_needed", 0) + 1
                continue
        elif closed_path is None:
            failures["initial_path_missing"] = failures.get("initial_path_missing", 0) + 1
            continue
        return {
            "goal_xy": goal.tolist(),
            "straight_line_distance_m": straight_distance,
            "open_path": np.asarray(open_path, dtype=float).tolist(),
            "open_path_length_m": open_length,
            "closed_path": (
                None if closed_path is None else np.asarray(closed_path, dtype=float).tolist()
            ),
            "closed_path_length_m": closed_length,
            "interaction_aware": bool(interaction_aware),
            "min_path_delta_m": float(min_path_delta_m),
            "interaction_requirement": (
                "required"
                if interaction_aware and closed_path is None
                else "beneficial" if interaction_aware else "unnecessary"
            ),
            "candidate_index": int(candidate_index),
            "attempted_candidate_count": attempted_count,
            "failure_counts": failures,
        }
    raise RuntimeError(
        "No PointGoal candidate passed the requested constraints: "
        + json.dumps(failures, sort_keys=True)
    )


def _point_oracle_plan(
    source_episode: dict[str, Any],
    *,
    goal_point: list[float],
    goal_yaw: float,
    interaction_aware: bool,
    position_tolerance_m: float,
) -> dict[str, Any]:
    source_plan = source_episode.get("interactive_nav", {}).get("oracle_plan", {})
    if not interaction_aware:
        return {
            "plan_id": "oracle_0",
            "required_interaction_ids": [],
            "steps": [
                {
                    "type": "navigate",
                    "interaction_id": None,
                    "goal_point": goal_point,
                    "goal_yaw": float(goal_yaw),
                    "position_tolerance_m": float(position_tolerance_m),
                    "yaw_tolerance_rad": 0.35,
                    "reason": "satisfy_nav_to_point_success",
                }
            ],
        }

    steps = copy.deepcopy(source_plan.get("steps") or [])
    navigate_indices = [index for index, step in enumerate(steps) if step.get("type") == "navigate"]
    if not navigate_indices:
        raise ValueError("Interaction-aware PointGoal source has no navigate step")
    terminal_index = navigate_indices[-1]
    output_steps = []
    for index, step in enumerate(steps):
        if step.get("type") in {"observe_target", "set_view"}:
            continue
        if index == terminal_index:
            output_steps.append(
                {
                    "type": "navigate",
                    "interaction_id": None,
                    "goal_point": goal_point,
                    "goal_yaw": float(goal_yaw),
                    "position_tolerance_m": float(position_tolerance_m),
                    "yaw_tolerance_rad": 0.35,
                    "reason": "satisfy_nav_to_point_success",
                }
            )
        else:
            output_steps.append(step)
    return {
        "plan_id": "oracle_0",
        "required_interaction_ids": list(source_plan.get("required_interaction_ids") or []),
        "steps": output_steps,
    }


def build_point_goal_episode(
    source_episode: dict[str, Any],
    *,
    sample: dict[str, Any],
    source_episode_index: int | None,
    sampling_source: str,
    clearance_m: float,
    success_threshold_m: float = 0.25,
    goal_yaw: float | None = None,
) -> dict[str, Any]:
    """Create a backward-compatible V3 PointGoal episode from a V3 source."""

    interaction_aware = bool(sample["interaction_aware"])
    source_nav = source_episode.get("interactive_nav", {})
    source_domains = list(source_nav.get("interaction_domains") or [])
    if interaction_aware and source_domains != ["channel"]:
        raise ValueError("Interaction-aware PointGoal currently requires a channel V3 source")
    goal_xy = [float(value) for value in sample["goal_xy"]]
    goal_point = [goal_xy[0], goal_xy[1], 0.0]
    if goal_yaw is None:
        open_path = sample.get("open_path") or []
        if len(open_path) >= 2:
            delta = np.asarray(open_path[-1], dtype=float)[:2] - np.asarray(
                open_path[-2], dtype=float
            )[:2]
            goal_yaw = float(math.atan2(delta[1], delta[0]))
        else:
            goal_yaw = 0.0
    oracle_plan = _point_oracle_plan(
        source_episode,
        goal_point=goal_point,
        goal_yaw=float(goal_yaw),
        interaction_aware=interaction_aware,
        position_tolerance_m=success_threshold_m,
    )
    episode = copy.deepcopy(source_episode)
    episode["source"] = None
    source_case_id = str(source_nav.get("case_id") or f"episode_{source_episode_index}")
    mode_slug = "interactive" if interaction_aware else "reachable"
    episode["task"] = {
        "task_cls": "molmo_spaces.tasks.point_nav_task.PointNavTask",
        "task_type": "nav_to_point",
        "robot_base_pose": list(source_episode.get("task", {}).get("robot_base_pose") or []),
        "goal_point": goal_point,
        "goal_yaw": None,
        "succ_pos_threshold": float(success_threshold_m),
        "require_goal_yaw": False,
        "succ_yaw_threshold": None,
    }
    episode["language"] = {
        "task_description": "Navigate to the designated point.",
        "instruction_type": "point_goal",
        "locale": "en",
        "interaction_disclosure": "hidden",
        "referral_expressions": {},
        "referral_expressions_priority": {},
        "task_input_mode": "goal_spec",
        "generation_mode": "point_goal_generator_v1",
    }
    if interaction_aware:
        interactions = copy.deepcopy(source_nav.get("interactions") or [])
        if not interactions:
            raise ValueError("Interaction-aware PointGoal source has no channel interactions")
        requirement = str(
            sample.get("interaction_requirement")
            or ("required" if sample.get("closed_path") is None else "beneficial")
        )
        if requirement == "beneficial":
            for interaction in interactions:
                effects = [
                    value
                    for value in interaction.get("effect_types") or []
                    if value != "restore_reachability"
                ]
                if "reduce_navigation_cost" not in effects:
                    effects.append("reduce_navigation_cost")
                interaction["effect_types"] = effects
        initial_state = copy.deepcopy(source_nav.get("initial_state") or {})
        relevant_objects = sorted(
            {
                str(row.get("object_name"))
                for row in interactions
                if row.get("object_name")
            }
        )
    else:
        interactions = []
        initial_state = {"interaction_states": [], "task_success_without_interaction": False}
        relevant_objects = []
        requirement = "unnecessary"
    episode["task_relevant_objects"] = relevant_objects
    generation_validation = {
        "navigation_validation": {
            "sampling_source": sampling_source,
            "goal_point": goal_point,
            "inflated_map_clearance_m": float(clearance_m),
            "straight_line_distance_m": sample["straight_line_distance_m"],
            "initial_state_path_found": sample.get("closed_path") is not None,
            "initial_state_path_length_m": sample.get("closed_path_length_m"),
            "oracle_restored_path_found": sample.get("open_path") is not None,
            "oracle_restored_path_length_m": sample.get("open_path_length_m"),
            "path_length_delta_m": (
                None
                if sample.get("closed_path_length_m") is None
                else float(sample["closed_path_length_m"])
                - float(sample["open_path_length_m"])
            ),
            "path_length_ratio_delta": (
                None
                if sample.get("closed_path_length_m") is None
                else (
                    float(sample["closed_path_length_m"])
                    - float(sample["open_path_length_m"])
                )
                / max(float(sample["open_path_length_m"]), 1e-6)
            ),
            "shortcut_verified": requirement == "beneficial",
            "shortcut_thresholds": {
                "min_delta_m": float(sample.get("min_path_delta_m", 0.5))
            },
            "gt_path_waypoints": sample.get("open_path"),
        },
        "interaction_validations": [],
        "oracle_prefixes": [
            {
                "plan_id": "oracle_0",
                "completed_step_count": 0,
                "robot_reachable_to_next_goal": True,
                "target_distance_passed": False,
                "target_visibility_fraction": None,
                "target_visible_pixels": None,
                "task_success": False,
                "opened_interaction_ids": [],
            },
            {
                "plan_id": "oracle_0",
                "completed_step_count": len(oracle_plan["steps"]),
                "robot_reachable_to_next_goal": True,
                "target_distance_passed": True,
                "target_visibility_fraction": None,
                "target_visible_pixels": None,
                "task_success": True,
                "opened_interaction_ids": list(oracle_plan["required_interaction_ids"]),
            },
        ],
        "success_evidence": {
            "status": "passed",
            "validation_mode": "path_feasibility_only",
            "target_object_name": None,
            "target_point": goal_point,
            "planar_distance_m": 0.0,
            "distance_threshold_m": float(success_threshold_m),
            "distance_passed": True,
            "visibility_passed": None,
            "expected_task_success": True,
        },
        "minimal_plan_verified": True if interaction_aware else None,
    }
    episode["interactive_nav"] = {
        "schema_version": "interactive_nav_v3",
        "case_id": f"{source_case_id}__point_goal_{mode_slug}",
        "parent_benchmark_episode_index": source_episode_index,
        "interaction_domains": ["channel"],
        "interaction_requirement": requirement,
        "target": v3.build_point_target(
            goal_point=goal_point,
            goal_yaw=None,
            sampling_source=sampling_source,
            clearance_m=clearance_m,
            grounding={
                "unique": True,
                "matching_instance_count": 1,
                "description": "fixed world-frame navigation point",
                "attributes": {},
            },
        ),
        "success_criteria": v3.build_nav_to_point_success_criteria(
            success_threshold_m
        ),
        "initial_state": initial_state,
        "interactions": interactions,
        "oracle_plan": oracle_plan,
        "oracle_plans": [copy.deepcopy(oracle_plan)],
        "generation_validation": generation_validation,
        "task_generation": {
            "generator": "generate_point_goal_v3.py",
            "generator_version": "point_goal_generator_v1",
            "source_case_id": source_case_id,
            "source_episode_index": source_episode_index,
            "sampling": {
                key: value
                for key, value in sample.items()
                if key not in {"open_path", "closed_path"}
            },
        },
    }
    return v3.validate_point_goal_v3_episode(episode)


def _runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        scene_dataset=args.scene_dataset,
        data_split=args.data_split,
        robot=args.robot,
        variant=args.variant,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def sample_from_v3_runtime(
    source_episode: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], Any, float]:
    from scripts.InteractiveNav import container_scene_probe as probe
    from scripts.InteractiveNav import explore_molmo_interactions as emi

    ctx = probe.load_scene_context(_runtime_args(args), int(source_episode["house_index"]))
    try:
        probe.apply_episode_scene_state(ctx.env, source_episode)
        radius = float(ctx.cfg.task_sampler_config.robot_safety_radius)
        closed_map = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            px_per_m=args.px_per_m,
            agent_radius=radius,
            treat_all_non_interactive_doorways_as_open=True,
        )
        start_xy = np.asarray(source_episode["task"]["robot_base_pose"][:2], dtype=float)
        if args.interaction_aware:
            plan = source_episode["interactive_nav"]["oracle_plan"]
            required = set(plan.get("required_interaction_ids") or [])
            interactions = {
                row["interaction_id"]: row
                for row in source_episode["interactive_nav"].get("interactions") or []
            }
            for interaction_id in required:
                interaction = interactions[interaction_id]
                if not str(interaction.get("type", "")).startswith("channel_"):
                    raise ValueError("PointGoal runtime only opens channel interactions")
                emi.set_articulation_fraction(
                    ctx.env,
                    interaction["object_name"],
                    int(interaction["joint_index"]),
                    float(interaction["target_state"]["joint_fraction"]),
                )
            open_map = emi.build_live_procthor_map(
                ctx.env.current_model,
                ctx.env.current_data,
                px_per_m=args.px_per_m,
                agent_radius=radius,
                treat_all_non_interactive_doorways_as_open=True,
            )
        else:
            open_map = closed_map
        rng = np.random.default_rng(args.seed)
        sample = select_point_goal_candidate(
            np.asarray(open_map.get_free_points(), dtype=float),
            start_xy=start_xy,
            open_path_fn=lambda goal: emi.compute_path_from_map(
                open_map, start_xy, goal, downscale_factor=1
            ),
            closed_path_fn=lambda goal: emi.compute_path_from_map(
                closed_map, start_xy, goal, downscale_factor=1
            ),
            rng=rng,
            min_distance_m=args.min_distance_m,
            max_distance_m=args.max_distance_m,
            interaction_aware=args.interaction_aware,
            min_path_delta_m=args.min_path_delta_m,
            max_attempts=args.max_attempts,
        )
        return sample, open_map, radius
    finally:
        ctx.sampler.close()


def sample_from_raw_scene_runtime(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Any, float]:
    from molmo_spaces.utils.pose import pose_mat_to_7d
    from scripts.InteractiveNav import container_scene_probe as probe
    from scripts.InteractiveNav import explore_molmo_interactions as emi
    from scripts.InteractiveNav.collection.seed_builder import load_template_episode

    if args.house_ind is None:
        raise ValueError("scene_split PointGoal generation requires --house-ind")
    ctx = probe.load_scene_context(_runtime_args(args), int(args.house_ind))
    try:
        radius = float(ctx.cfg.task_sampler_config.robot_safety_radius)
        scene_map = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            px_per_m=args.px_per_m,
            agent_radius=radius,
            treat_all_non_interactive_doorways_as_open=True,
        )
        free_points = np.asarray(scene_map.get_free_points(), dtype=float)
        if len(free_points) < 2:
            raise RuntimeError("Raw scene contains fewer than two free PointGoal cells")
        rng = np.random.default_rng(args.seed)
        start_order = rng.permutation(len(free_points))[: min(len(free_points), 128)]
        sample = None
        start_xy = None
        for start_index in start_order:
            candidate_start = np.asarray(free_points[start_index, :2], dtype=float)
            try:
                sample = select_point_goal_candidate(
                    free_points,
                    start_xy=candidate_start,
                    open_path_fn=lambda goal, start=candidate_start: emi.compute_path_from_map(
                        scene_map, start, goal, downscale_factor=1
                    ),
                    closed_path_fn=None,
                    rng=rng,
                    min_distance_m=args.min_distance_m,
                    max_distance_m=args.max_distance_m,
                    interaction_aware=False,
                    max_attempts=args.max_attempts,
                )
            except RuntimeError:
                continue
            start_xy = candidate_start
            break
        if sample is None or start_xy is None:
            raise RuntimeError("Failed to sample a connected raw-scene PointGoal pair")
        goal_xy = np.asarray(sample["goal_xy"], dtype=float)
        yaw = math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
        robot_pose = probe.make_robot_pose_from_xy(
            ctx.env.current_robot.robot_view, start_xy, yaw
        )
        source = load_template_episode()
        source.update(
            {
                "source": None,
                "house_index": int(args.house_ind),
                "scene_dataset": args.scene_dataset,
                "data_split": args.data_split,
                "seed": int(args.seed),
            }
        )
        source["task"] = {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": pose_mat_to_7d(robot_pose).astype(float).tolist(),
            "selection_mode": "specific_instance",
            "pickup_obj_name": "point_goal_placeholder",
            "pickup_obj_candidates": ["point_goal_placeholder"],
            "succ_pos_threshold": 1.5,
        }
        source["task_relevant_objects"] = []
        source["language"] = {
            "task_description": "Navigate to the designated point.",
            "instruction_type": "point_goal",
            "locale": "en",
            "interaction_disclosure": "hidden",
            "referral_expressions": {},
            "referral_expressions_priority": {},
        }
        source["interactive_nav"] = {
            "case_id": f"raw_house_{int(args.house_ind)}_seed_{int(args.seed)}"
        }
        return sample, source, scene_map, radius
    finally:
        ctx.sampler.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def save_topdown(path: Path, scene_map, sample: dict[str, Any], start_xy: Iterable[float]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.asarray(scene_map.occupancy, dtype=float), cmap="gray", origin="upper")

    def pixels(points):
        rows = []
        for point in points:
            rows.append(scene_map.pos_m_to_px(np.asarray([point[0], point[1], 0.0])))
        return np.asarray(rows)

    for key, color, label in (
        ("closed_path", "tab:red", "initial path"),
        ("open_path", "tab:green", "oracle path"),
    ):
        if sample.get(key):
            px = pixels(sample[key])
            ax.plot(px[:, 1], px[:, 0], color=color, linewidth=2, label=label)
    start_px = pixels([start_xy])[0]
    goal_px = pixels([sample["goal_xy"]])[0]
    ax.scatter([start_px[1]], [start_px[0]], c="tab:blue", s=60, label="start")
    ax.scatter([goal_px[1]], [goal_px[0]], c="gold", edgecolors="black", s=80, label="goal")
    ax.legend()
    ax.set_title("PointGoal V3 sampling")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, nargs="?")
    parser.add_argument("--source-mode", choices=["v3", "scene_split"], default="v3")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interaction-aware", action="store_true")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--min-distance-m", type=float, default=2.0)
    parser.add_argument("--max-distance-m", type=float)
    parser.add_argument("--min-path-delta-m", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=2048)
    parser.add_argument("--px-per-m", type=int, default=50)
    parser.add_argument("--success-threshold-m", type=float, default=0.25)
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--house-ind", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.source_mode == "v3":
        if args.benchmark is None:
            raise ValueError("V3 PointGoal generation requires a benchmark path")
        episodes = load_episodes(args.benchmark)
        episode_index, source = select_episode(
            episodes, episode_index=args.episode_index, case_id=args.case_id
        )
        if source.get("interactive_nav", {}).get("interaction_domains") != ["channel"]:
            raise ValueError("PointGoal demo currently selects a channel V3 episode")
        args.scene_dataset = source.get("scene_dataset", args.scene_dataset)
        args.data_split = source.get("data_split", args.data_split)
        sample, scene_map, radius = sample_from_v3_runtime(source, args)
        sampling_source = (
            "post_interaction_inflated_occupancy_free_cell"
            if args.interaction_aware
            else "initial_inflated_occupancy_free_cell"
        )
    else:
        if args.interaction_aware:
            raise ValueError("Raw scene_split demo currently generates reachable PointGoals only")
        episode_index = None
        sample, source, scene_map, radius = sample_from_raw_scene_runtime(args)
        sampling_source = "raw_scene_inflated_occupancy_free_cell"
    generated = build_point_goal_episode(
        source,
        sample=sample,
        source_episode_index=episode_index,
        sampling_source=sampling_source,
        clearance_m=radius,
        success_threshold_m=args.success_threshold_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "benchmark.json", [generated])
    write_json(args.output_dir / "sampling.json", sample)
    save_topdown(
        args.output_dir / "topdown.png",
        scene_map,
        sample,
        source["task"]["robot_base_pose"][:2],
    )
    print(args.output_dir / "benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
