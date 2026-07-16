#!/usr/bin/env python3
"""Sample reproducible House 7 navigation-force-interaction-navigation routes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TARGET_ROOT = "doorway_ada234694d8669f8c477500ae8f01b1a_1_0_4"
DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "scripts"
    / "InteractiveNav"
    / "configs"
    / "semantic_decision"
    / "house7_force_routes.yaml"
)


def normalized(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(vector), dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        raise ValueError("Expected a non-zero vector")
    return value / norm


def parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = [int(part) for part in token.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid seed range: {token}")
            start, stop = parts[:2]
            step = parts[2] if len(parts) == 3 else 1
            seeds.extend(range(start, stop, step))
        else:
            seeds.append(int(token))
    return list(dict.fromkeys(seeds))


def point_to_pixel(scene_map, point_xy: Iterable[float]) -> tuple[int, int]:
    point = np.asarray(list(point_xy), dtype=float)
    pixel = np.asarray(
        scene_map.pos_m_to_px(np.array([point[0], point[1], 0.0]))
    ).astype(int)
    return int(pixel[0]), int(pixel[1])


def supported_room_id(
    scene_map, point_xy: Iterable[float], radius_m: float = 0.35
) -> int | None:
    room_map = np.asarray(scene_map.room_map)
    occupancy = np.asarray(scene_map.occupancy).astype(bool)
    row, col = point_to_pixel(scene_map, point_xy)
    radius_px = max(1, int(round(float(radius_m) * float(scene_map.px_per_m))))
    row_min = max(0, row - radius_px)
    row_max = min(room_map.shape[0], row + radius_px + 1)
    col_min = max(0, col - radius_px)
    col_max = min(room_map.shape[1], col + radius_px + 1)
    if row_min >= row_max or col_min >= col_max:
        return None
    room_patch = room_map[row_min:row_max, col_min:col_max]
    free_patch = occupancy[row_min:row_max, col_min:col_max]
    values = room_patch[(room_patch > 0) & free_patch]
    if not values.size:
        values = room_patch[room_patch > 0]
    if not values.size:
        return None
    counts = Counter(int(value) for value in values.tolist())
    return counts.most_common(1)[0][0]


def portal_room_sides(
    scene_map,
    portal_center_xy: Iterable[float],
    portal_normal_xy: Iterable[float],
    portal_half_width_m: float,
) -> dict[int, dict[str, Any]]:
    center = np.asarray(list(portal_center_xy), dtype=float)
    normal = normalized(portal_normal_xy)
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    tangent_span = min(max(float(portal_half_width_m) * 0.35, 0.10), 0.35)
    sides: dict[int, dict[str, Any]] = {}
    for sign in (-1, 1):
        observations = []
        for distance_m in (0.55, 0.75, 1.00, 1.25):
            for tangent_offset_m in (-tangent_span, 0.0, tangent_span):
                point = center + sign * normal * distance_m + tangent * tangent_offset_m
                room_id = supported_room_id(scene_map, point, radius_m=0.20)
                if room_id is not None:
                    observations.append((room_id, point))
        counts = Counter(room_id for room_id, _point in observations)
        room_id = counts.most_common(1)[0][0] if counts else None
        representatives = [
            point for candidate_room, point in observations if candidate_room == room_id
        ]
        sides[sign] = {
            "sign": sign,
            "room_id": room_id,
            "support": int(counts.get(room_id, 0)) if room_id is not None else 0,
            "sample_xy": representatives[len(representatives) // 2]
            if representatives
            else None,
        }
    return sides


def route_yaw(source_xy: Iterable[float], target_xy: Iterable[float]) -> float:
    source = np.asarray(list(source_xy), dtype=float)
    target = np.asarray(list(target_xy), dtype=float)
    delta = target - source
    return float(math.atan2(delta[1], delta[0]))


def candidate_approach_points(
    portal_center_xy: Iterable[float],
    portal_normal_xy: Iterable[float],
    portal_half_width_m: float,
    side_sign: int,
) -> list[np.ndarray]:
    center = np.asarray(list(portal_center_xy), dtype=float)
    normal = normalized(portal_normal_xy)
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    tangent_span = min(max(float(portal_half_width_m) * 0.20, 0.08), 0.25)
    candidates = []
    for distance_m in (0.70, 0.85, 1.00, 1.15):
        for tangent_offset_m in (0.0, -tangent_span, tangent_span):
            candidates.append(
                center + int(side_sign) * normal * distance_m + tangent * tangent_offset_m
            )
    return candidates


def far_goal_candidates(
    scene_map,
    room_id: int,
    portal_center_xy: Iterable[float],
    connected_from_xy: Iterable[float],
    min_portal_distance_m: float = 2.5,
    min_clearance_m: float = 0.45,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    occupancy = np.asarray(scene_map.occupancy).astype(bool)
    room_map = np.asarray(scene_map.room_map)
    start_row, start_col = point_to_pixel(scene_map, connected_from_xy)
    component_count, components = cv2.connectedComponents(
        occupancy.astype(np.uint8), connectivity=8
    )
    if not (
        0 <= start_row < occupancy.shape[0] and 0 <= start_col < occupancy.shape[1]
    ):
        return []
    component_id = int(components[start_row, start_col])
    if component_id <= 0 or component_count <= 1:
        return []
    clearance = cv2.distanceTransform(
        occupancy.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ) / float(scene_map.px_per_m)
    valid = (
        occupancy
        & (room_map == int(room_id))
        & (components == component_id)
        & (clearance >= float(min_clearance_m))
    )
    stride = max(1, int(round(float(scene_map.px_per_m) * 0.15)))
    sampled = np.zeros_like(valid)
    sampled[::stride, ::stride] = True
    pixels = np.argwhere(valid & sampled)
    if not len(pixels):
        return []
    points = np.asarray(scene_map.pos_px_to_m(pixels), dtype=float)[:, :2]
    portal_center = np.asarray(list(portal_center_xy), dtype=float)
    portal_distances = np.linalg.norm(points - portal_center[None, :], axis=1)
    clearances = clearance[pixels[:, 0], pixels[:, 1]]
    keep = portal_distances >= float(min_portal_distance_m)
    pixels = pixels[keep]
    points = points[keep]
    portal_distances = portal_distances[keep]
    clearances = clearances[keep]
    if not len(points):
        return []
    scores = portal_distances + 0.35 * clearances
    order = np.argsort(scores)[::-1][: int(max_candidates)]
    return [
        {
            "xy": points[index],
            "pixel": pixels[index],
            "portal_distance_m": float(portal_distances[index]),
            "clearance_m": float(clearances[index]),
            "score": float(scores[index]),
        }
        for index in order
    ]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sample_route_for_seed(spec: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from molmo_spaces.utils.pose import pose_mat_to_7d

    from scripts.InteractiveNav.explore_molmo_interactions import (
        build_live_procthor_map,
        close_context,
        collect_interactive_door_root_object_records,
        compute_path_from_map,
        get_robot_xy,
        load_context,
        path_length,
        set_door_root_state,
    )
    from scripts.InteractiveNav.force_interaction_runtime import set_all_door_roots_closed

    seed = int(spec["seed"])
    args = SimpleNamespace(
        seed=seed,
        robot=spec["robot"],
        scene_dataset=spec["scene_dataset"],
        data_split=spec["data_split"],
        house_ind=int(spec["house_ind"]),
        variant=spec["variant"],
        target_types=None,
        benchmark_episode=None,
    )
    context = None
    debug: dict[str, Any] = {}
    try:
        context = load_context(args, task_mode="nav_task")
        env = context.env
        set_all_door_roots_closed(env)
        start_pose_matrix = env.current_robot.robot_view.base.pose.copy()
        start_xy = get_robot_xy(env)
        debug["start_xy"] = start_xy
        collision = bool(env.check_robot_collision_in_current_pose())
        debug["start_collision"] = collision
        if collision:
            raise RuntimeError("Default sampler returned a colliding robot pose")

        map_kwargs = {
            "model_path": str(env.current_model_path),
            "px_per_m": int(spec["px_per_m"]),
            "agent_radius": float(context.cfg.task_sampler_config.robot_safety_radius),
            "open_threshold": 1e-3,
            "treat_all_non_interactive_doorways_as_open": True,
            "return_doorway_analysis": True,
            "include_wall_collision_slices": bool(spec["include_wall_collision_slices"]),
        }
        closed_map, closed_analysis = build_live_procthor_map(
            env.current_model, env.current_data, **map_kwargs
        )
        target_records = [
            record
            for record in collect_interactive_door_root_object_records(env, closed_analysis)
            if record["name"] == spec["target_root"]
        ]
        if not target_records:
            raise RuntimeError(f"Target doorway root not found: {spec['target_root']}")
        portal = target_records[0]
        debug["portal_center_xy"] = portal["portal_center_xy"]
        debug["portal_normal_xy"] = portal["portal_normal_xy"]
        debug["portal_half_width_m"] = float(portal["portal_half_width_m"])
        sides = portal_room_sides(
            closed_map,
            portal["portal_center_xy"],
            portal["portal_normal_xy"],
            portal["portal_half_width_m"],
        )
        debug["portal_sides"] = sides
        if sides[-1]["room_id"] is None or sides[1]["room_id"] is None:
            raise RuntimeError(f"Could not identify both portal rooms: {jsonable(sides)}")
        if sides[-1]["room_id"] == sides[1]["room_id"]:
            raise RuntimeError(f"Portal samples resolve to one room: {jsonable(sides)}")
        start_room_id = supported_room_id(closed_map, start_xy)
        debug["start_room_id"] = start_room_id
        matching_sides = [
            sign for sign, side in sides.items() if side["room_id"] == start_room_id
        ]
        if len(matching_sides) != 1:
            raise RuntimeError(
                f"Start room {start_room_id} is not exactly one portal side: {jsonable(sides)}"
            )
        start_side_sign = matching_sides[0]
        goal_room_id = int(sides[-start_side_sign]["room_id"])
        debug["start_side_sign"] = start_side_sign
        debug["goal_room_id"] = goal_room_id

        best_approach = None
        approach_checks = []
        for approach_xy in candidate_approach_points(
            portal["portal_center_xy"],
            portal["portal_normal_xy"],
            portal["portal_half_width_m"],
            start_side_sign,
        ):
            approach_room_id = supported_room_id(closed_map, approach_xy, radius_m=0.18)
            check = {
                "xy": approach_xy,
                "room_id": approach_room_id,
            }
            if approach_room_id != start_room_id:
                check["rejected"] = "wrong_room"
                approach_checks.append(check)
                continue
            path = compute_path_from_map(
                closed_map,
                start_xy,
                approach_xy,
                downscale_factor=int(spec["downscale"]),
                max_start_goal_distance=8,
            )
            length_m = path_length(path)
            check["path_found"] = path is not None
            check["path_length_m"] = length_m
            if path is None or length_m is None or length_m < float(spec["min_first_leg_m"]):
                check["rejected"] = "missing_or_short_path"
                approach_checks.append(check)
                continue
            check["accepted"] = True
            approach_checks.append(check)
            candidate = {"xy": approach_xy, "path": path, "path_length_m": length_m}
            if best_approach is None or length_m < best_approach["path_length_m"]:
                best_approach = candidate
        debug["approach_checks"] = approach_checks
        if best_approach is None:
            raise RuntimeError("No valid closed-door path from sampled start to door approach")

        set_door_root_state(env, closed_analysis, spec["target_root"], "open")
        open_map, _open_analysis = build_live_procthor_map(
            env.current_model, env.current_data, **map_kwargs
        )
        far_candidates = far_goal_candidates(
            open_map,
            goal_room_id,
            portal["portal_center_xy"],
            best_approach["xy"],
            min_portal_distance_m=float(spec["min_far_goal_distance_m"]),
            min_clearance_m=float(spec["min_far_goal_clearance_m"]),
        )
        debug["far_candidate_count"] = len(far_candidates)
        selected_far = None
        far_checks = []
        for far_candidate in far_candidates:
            open_path = compute_path_from_map(
                open_map,
                best_approach["xy"],
                far_candidate["xy"],
                downscale_factor=int(spec["downscale"]),
                max_start_goal_distance=8,
            )
            open_length_m = path_length(open_path)
            total_route_length_m = (
                None
                if open_length_m is None
                else float(best_approach["path_length_m"] + open_length_m)
            )
            far_check = {
                "xy": far_candidate["xy"],
                "portal_distance_m": far_candidate["portal_distance_m"],
                "clearance_m": far_candidate["clearance_m"],
                "open_path_found": open_path is not None,
                "open_path_length_m": open_length_m,
                "total_route_length_m": total_route_length_m,
            }
            if open_path is None or open_length_m is None:
                far_check["rejected"] = "open_path_missing"
                far_checks.append(far_check)
                continue
            if total_route_length_m < float(spec["min_total_route_m"]):
                far_check["rejected"] = "total_route_too_short"
                far_checks.append(far_check)
                continue
            closed_direct_path = compute_path_from_map(
                closed_map,
                start_xy,
                far_candidate["xy"],
                downscale_factor=int(spec["downscale"]),
                max_start_goal_distance=8,
            )
            far_check["closed_path_found"] = closed_direct_path is not None
            if closed_direct_path is not None:
                far_check["rejected"] = "closed_path_still_available"
                far_checks.append(far_check)
                continue
            far_check["accepted"] = True
            far_checks.append(far_check)
            selected_far = {
                **far_candidate,
                "path": open_path,
                "path_length_m": open_length_m,
            }
            break
        debug["far_checks"] = far_checks
        if selected_far is None:
            raise RuntimeError(
                "No opposite-room far goal that is blocked when the target door is closed"
            )

        portal_center = np.asarray(portal["portal_center_xy"], dtype=float)
        start_yaw = math.atan2(
            float(start_pose_matrix[1, 0]), float(start_pose_matrix[0, 0])
        )
        approach_yaw = route_yaw(best_approach["xy"], portal_center)
        far_yaw = (
            route_yaw(selected_far["path"][-2], selected_far["path"][-1])
            if len(selected_far["path"]) >= 2
            else approach_yaw
        )
        return jsonable(
            {
                "valid": True,
                "seed": seed,
                "pickup_obj_name": context.task.config.task_config.pickup_obj_name,
                "robot_base_pose": pose_mat_to_7d(start_pose_matrix),
                "start_xyyaw": [start_xy[0], start_xy[1], start_yaw],
                "door_approach_xyyaw": [
                    best_approach["xy"][0],
                    best_approach["xy"][1],
                    approach_yaw,
                ],
                "far_goal_xyyaw": [
                    selected_far["xy"][0],
                    selected_far["xy"][1],
                    far_yaw,
                ],
                "target_root": spec["target_root"],
                "portal_center_xy": portal_center,
                "portal_normal_xy": portal["portal_normal_xy"],
                "portal_half_width_m": float(portal["portal_half_width_m"]),
                "start_room_id": int(start_room_id),
                "goal_room_id": int(goal_room_id),
                "start_room_name": closed_map.room_ids_to_name.get(int(start_room_id)),
                "goal_room_name": closed_map.room_ids_to_name.get(int(goal_room_id)),
                "start_side_sign": int(start_side_sign),
                "first_leg_path_length_m": float(best_approach["path_length_m"]),
                "third_leg_path_length_m": float(selected_far["path_length_m"]),
                "total_route_path_length_m": float(
                    best_approach["path_length_m"] + selected_far["path_length_m"]
                ),
                "far_goal_portal_distance_m": float(selected_far["portal_distance_m"]),
                "far_goal_clearance_m": float(selected_far["clearance_m"]),
                "closed_start_to_far_path_found": False,
                "open_approach_to_far_path_found": True,
                "start_collision": False,
            }
        )
    except Exception as exc:
        return {
            "valid": False,
            "seed": seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "debug": jsonable(debug),
        }
    finally:
        if context is not None:
            close_context(context)


def build_route_config(spec: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, Any]:
    frozen_routes = []
    for index, route in enumerate(sorted(routes, key=lambda item: int(item["seed"]))):
        frozen_routes.append(
            {
                "route_id": f"house7_force_route_{index + 1:02d}",
                "seed": int(route["seed"]),
                "pickup_obj_name": route["pickup_obj_name"],
                "robot_base_pose": route["robot_base_pose"],
                "start_xyyaw": route["start_xyyaw"],
                "door_approach_xyyaw": route["door_approach_xyyaw"],
                "far_goal_xyyaw": route["far_goal_xyyaw"],
                "interaction": {
                    "backend": "force",
                    "action": "open",
                    "source_object_name": spec["target_root"],
                    "atomic_sim_steps": 1,
                    "assume_success": True,
                },
                "validation": {
                    key: route[key]
                    for key in (
                        "start_room_id",
                        "goal_room_id",
                        "start_room_name",
                        "goal_room_name",
                        "start_side_sign",
                        "first_leg_path_length_m",
                        "third_leg_path_length_m",
                        "total_route_path_length_m",
                        "far_goal_portal_distance_m",
                        "far_goal_clearance_m",
                        "closed_start_to_far_path_found",
                        "open_approach_to_far_path_found",
                        "start_collision",
                    )
                },
            }
        )
    return {
        "schema_version": 1,
        "description": (
            "House 7 routes sampled from the normal NavToObj initial placement. "
            "All doors start closed; the target double door is opened with the force backend."
        ),
        "scene": {
            "scene_dataset": spec["scene_dataset"],
            "data_split": spec["data_split"],
            "house_ind": int(spec["house_ind"]),
            "variant": spec["variant"],
            "robot": spec["robot"],
        },
        "target_door_root": spec["target_root"],
        "sampling": {
            "px_per_m": int(spec["px_per_m"]),
            "downscale": int(spec["downscale"]),
            "min_first_leg_m": float(spec["min_first_leg_m"]),
            "min_total_route_m": float(spec["min_total_route_m"]),
            "min_far_goal_distance_m": float(spec["min_far_goal_distance_m"]),
            "min_far_goal_clearance_m": float(spec["min_far_goal_clearance_m"]),
        },
        "routes": frozen_routes,
    }


def load_reused_results(
    paths: Iterable[Path], spec: dict[str, Any], requested_seeds: set[int]
) -> dict[int, dict[str, Any]]:
    reused: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        cached_spec = payload.get("spec", {})
        mismatches = {
            key: (cached_spec.get(key), value)
            for key, value in spec.items()
            if cached_spec.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Diagnostics spec mismatch for {path}: {mismatches}")
        for result in payload.get("results", []):
            seed = int(result["seed"])
            if seed in requested_seeds:
                reused[seed] = result
    return reused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--house-ind", type=int, default=7)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--target-root", default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--seeds", default="0:20")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-routes", type=int, default=6)
    parser.add_argument("--px-per-m", type=int, default=80)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--min-first-leg-m", type=float, default=1.25)
    parser.add_argument("--min-total-route-m", type=float, default=5.0)
    parser.add_argument("--min-far-goal-distance-m", type=float, default=2.5)
    parser.add_argument("--min-far-goal-clearance-m", type=float, default=0.45)
    parser.add_argument(
        "--include-wall-collision-slices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument(
        "--reuse-diagnostics",
        type=Path,
        action="append",
        default=[],
        help="Reuse matching per-seed results and run only missing seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_spec(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    workers = max(1, min(int(args.workers), 2))
    spec = {
        "scene_dataset": args.scene_dataset,
        "data_split": args.data_split,
        "house_ind": args.house_ind,
        "variant": args.variant,
        "robot": args.robot,
        "target_root": args.target_root,
        "px_per_m": args.px_per_m,
        "downscale": args.downscale,
        "min_first_leg_m": args.min_first_leg_m,
        "min_total_route_m": args.min_total_route_m,
        "min_far_goal_distance_m": args.min_far_goal_distance_m,
        "min_far_goal_clearance_m": args.min_far_goal_clearance_m,
        "include_wall_collision_slices": args.include_wall_collision_slices,
    }
    requested_seed_set = set(seeds)
    reused = load_reused_results(args.reuse_diagnostics, spec, requested_seed_set)
    results_by_seed = dict(reused)
    if reused:
        print(f"Reused {len(reused)} seed results from diagnostics.", flush=True)
    pending_seeds = [seed for seed in seeds if seed not in results_by_seed]
    if pending_seeds:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_seed = {
                executor.submit(sample_route_for_seed, {**spec, "seed": seed}): seed
                for seed in pending_seeds
            }
            completed = len(results_by_seed)
            for future in as_completed(future_to_seed):
                result = future.result()
                results_by_seed[int(result["seed"])] = result
                completed += 1
                status = "VALID" if result.get("valid") else "REJECT"
                print(
                    f"[{completed:02d}/{len(seeds):02d}] seed={result['seed']} {status} "
                    f"{result.get('error', '')}",
                    flush=True,
                )
    results = [results_by_seed[seed] for seed in seeds]
    valid_routes = sorted(
        [result for result in results if result.get("valid")],
        key=lambda item: int(item["seed"]),
    )[: int(args.max_routes)]
    diagnostics_path = args.diagnostics or args.output.with_suffix(".diagnostics.json")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(
            {
                "spec": spec,
                "requested_seeds": seeds,
                "valid_count": len(valid_routes),
                "results": sorted(results, key=lambda item: int(item["seed"])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if len(valid_routes) < int(args.max_routes):
        raise RuntimeError(
            f"Only {len(valid_routes)} valid routes found; requested {args.max_routes}. "
            f"See {diagnostics_path}"
        )
    route_config = build_route_config(spec, valid_routes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(route_config, sort_keys=False, allow_unicode=True))
    print(f"Wrote {len(valid_routes)} frozen routes to {args.output}")
    print(f"Diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
