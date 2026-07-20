from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.utils.pose import pose_mat_to_7d
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_BENCHMARK = REPO_ROOT / (
    "assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json"
)


def load_template_episode(path: Path = DEFAULT_TEMPLATE_BENCHMARK) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    template = copy.deepcopy(episodes[0])
    template["source"] = None
    template["scene_modifications"] = {
        "added_objects": {},
        "object_poses": {},
        "removed_objects": [],
        "articulation_states": [],
    }
    template.pop("interactive_nav", None)
    return template


def _target_center(record: dict[str, Any]) -> np.ndarray:
    value = record.get("aabb_center", record.get("position"))
    return np.asarray(value, dtype=float)[:3]


def _candidate_start_pose(
    ctx: probe.LoadedContext,
    scene_map,
    free_points: np.ndarray,
    target_xy: np.ndarray,
    rng: np.random.Generator,
    candidate_pool: int,
    component_labels: np.ndarray,
) -> np.ndarray | None:
    if len(free_points) == 0:
        return None
    sample_count = min(len(free_points), candidate_pool)
    sampled = free_points[
        rng.choice(len(free_points), size=sample_count, replace=False)
    ]
    goal_px = np.asarray(
        scene_map.pos_m_to_px(np.asarray([target_xy[0], target_xy[1], 0.0], dtype=float)),
        dtype=int,
    )[:2]
    goal_label = int(component_labels[int(goal_px[0]), int(goal_px[1])])
    if goal_label <= 0:
        return None
    sampled_px = np.asarray(scene_map.pos_m_to_px(sampled), dtype=int)
    same_component = component_labels[sampled_px[:, 0], sampled_px[:, 1]] == goal_label
    sampled = sampled[same_component]
    if len(sampled) == 0:
        return None
    order = np.argsort(np.linalg.norm(sampled[:, :2] - target_xy[None, :2], axis=1))[::-1]
    robot_view = ctx.env.current_robot.robot_view
    for index in order[: min(32, len(order))]:
        xy = np.asarray(sampled[index, :2], dtype=float)
        yaw = math.atan2(target_xy[1] - xy[1], target_xy[0] - xy[0])
        pose = probe.make_robot_pose_from_xy(robot_view, xy, yaw)
        if ctx.env.check_if_robot_collision_at_base_pose(robot_view, pose):
            continue
        return pose
    return None


def build_house_seed_episodes(
    *,
    house_index: int,
    scene_dataset: str,
    data_split: str,
    variant: str,
    robot: str,
    seed: int,
    seeds_per_house: int,
    candidate_pool: int,
    template: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    template = copy.deepcopy(template or load_template_episode())
    args = argparse.Namespace(
        scene_dataset=scene_dataset,
        data_split=data_split,
        robot=robot,
        variant=variant,
        seed=seed,
        output_dir=None,
    )
    ctx = probe.load_scene_context(args, house_index)
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        logging.getLogger("molmo_spaces.env.env").setLevel(logging.WARNING)
        emi.ensure_runtime_dependencies()
        records, _containers = probe.collect_scene_records(ctx)
        targets = [record for record in records if probe.is_target_like(record)]
        rng = np.random.default_rng(seed + house_index * 1009)
        targets.sort(key=lambda record: (str(record.get("category", "")), record["name"]))
        rng.shuffle(targets)
        scene_map = ctx.env.get_thormap(
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius
        )
        free_points = np.asarray(scene_map.get_free_points(), dtype=float)
        component_labels, _component_count = connected_components(
            np.asarray(scene_map.occupancy, dtype=bool)
        )
        used_categories: set[str] = set()
        for record in targets:
            if len(episodes) >= seeds_per_house:
                break
            category = str(record.get("category") or record["name"].split("_", 1)[0])
            if category in used_categories and len(targets) >= seeds_per_house:
                continue
            target_center = _target_center(record)
            nearest_goal = emi.nearest_free_point_xy(scene_map, target_center[:2])
            if nearest_goal is None:
                failures.append(
                    {"house_index": house_index, "target": record["name"], "reason": "no_target_goal"}
                )
                continue
            pose = _candidate_start_pose(
                ctx,
                scene_map,
                free_points,
                np.asarray(nearest_goal, dtype=float),
                rng,
                candidate_pool,
                component_labels,
            )
            if pose is None:
                failures.append(
                    {"house_index": house_index, "target": record["name"], "reason": "no_valid_start_pose"}
                )
                continue
            episode = copy.deepcopy(template)
            episode.update(
                {
                    "source": None,
                    "house_index": house_index,
                    "scene_dataset": scene_dataset,
                    "data_split": data_split,
                    "seed": int(seed + house_index * 1009 + len(episodes)),
                }
            )
            episode["task"] = {
                "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
                "task_type": "nav_to_obj",
                "selection_mode": "specific_instance",
                "robot_base_pose": pose_mat_to_7d(pose).astype(float).tolist(),
                "pickup_obj_name": record["name"],
                "pickup_obj_candidates": [record["name"]],
                "pickup_obj_category": category,
                "pickup_obj_start_pose": None,
                "receptacle_name": None,
                "succ_pos_threshold": 1.5,
            }
            episode["task_relevant_objects"] = [record["name"]]
            episode["language"] = {
                "task_description": f"find the {category.lower()}.",
                "instruction_type": "object_goal",
                "locale": "en",
                "interaction_disclosure": "hidden",
                "referral_expressions": {"object_name": category},
                "referral_expressions_priority": {},
            }
            episode["seed_generation"] = {
                "schema_version": "interactive_nav_train_seed_v1",
                "target_center": target_center.astype(float).tolist(),
                "goal_xy": np.asarray(nearest_goal, dtype=float).tolist(),
                "strategy": "farthest_collision_free_reachable_point",
            }
            EpisodeSpec.model_validate(episode)
            episodes.append(episode)
            used_categories.add(category)
    finally:
        ctx.sampler.close()
    return episodes, failures
