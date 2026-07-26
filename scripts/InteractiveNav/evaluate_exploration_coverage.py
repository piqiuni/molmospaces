#!/usr/bin/env python3
"""Compare a final ROS occupancy grid with the static scene navigable area."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap
from read_scene_room_properties import build_scene_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--robot", default="rby1", choices=["droid", "rby1", "rum"])
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--house-ind", type=int, required=True)
    parser.add_argument("--variant", default="base", choices=["base", "ceiling", "map"])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--gt-agent-radius-m", type=float, default=0.10)
    parser.add_argument("--gt-px-per-m", type=int, default=100)
    return parser.parse_args()


def load_ros_map(run_dir: Path) -> tuple[np.ndarray, float, np.ndarray, float]:
    yaml_path = run_dir / "final_occ_map.yaml"
    metadata = yaml.safe_load(yaml_path.read_text())
    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    origin = np.asarray(metadata["origin"], dtype=float)
    return image, float(metadata["resolution"]), origin[:2], float(origin[2])


def sample_ros_map(
    image: np.ndarray,
    resolution: float,
    origin_xy: np.ndarray,
    origin_yaw: float,
    world_xy: np.ndarray,
) -> np.ndarray:
    delta = world_xy - origin_xy[None, :]
    c = math.cos(origin_yaw)
    s = math.sin(origin_yaw)
    local_x = c * delta[:, 0] + s * delta[:, 1]
    local_y = -s * delta[:, 0] + c * delta[:, 1]
    cell_x = np.floor(local_x / resolution).astype(np.int64)
    cell_y = np.floor(local_y / resolution).astype(np.int64)
    row = image.shape[0] - 1 - cell_y
    valid = (
        (cell_x >= 0)
        & (cell_x < image.shape[1])
        & (row >= 0)
        & (row < image.shape[0])
    )
    sampled = np.full(len(world_xy), 205, dtype=np.uint8)
    sampled[valid] = image[row[valid], cell_x[valid]]
    return sampled


def load_trajectory(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "trajectory.csv"
    if not path.exists():
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    points = []
    yaws = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                points.append((float(row["x"]), float(row["y"])))
                yaws.append(float(row["yaw"]))
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(points, dtype=float), np.asarray(yaws, dtype=float)


def load_recorded_frontiers(run_dir: Path) -> np.ndarray:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return np.empty((0, 2), dtype=float)
    latest = None
    with path.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "explore_status":
                latest = event.get("payload", {})
    points = []
    for cluster in (latest or {}).get("frontier_clusters", []):
        for point in cluster.get("frontier_cells_world", []):
            if isinstance(point, list) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    return np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float)


def reconstruct_frontiers(
    image: np.ndarray,
    resolution: float,
    origin_xy: np.ndarray,
    origin_yaw: float,
    min_cluster_cells: int = 3,
) -> np.ndarray:
    free = image >= 250
    unknown = (image > 50) & (image < 250)
    neighbor_unknown = np.zeros_like(unknown)
    neighbor_unknown[1:, :] |= unknown[:-1, :]
    neighbor_unknown[:-1, :] |= unknown[1:, :]
    neighbor_unknown[:, 1:] |= unknown[:, :-1]
    neighbor_unknown[:, :-1] |= unknown[:, 1:]
    frontier = (free & neighbor_unknown).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(frontier, connectivity=8)
    keep = np.zeros_like(frontier, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_cluster_cells:
            keep |= labels == label
    rows, cols = np.nonzero(keep)
    if not len(rows):
        return np.empty((0, 2), dtype=float)
    cell_x = cols.astype(float)
    cell_y = (image.shape[0] - 1 - rows).astype(float)
    local_x = (cell_x + 0.5) * resolution
    local_y = (cell_y + 0.5) * resolution
    c = math.cos(origin_yaw)
    s = math.sin(origin_yaw)
    world_x = origin_xy[0] + c * local_x - s * local_y
    world_y = origin_xy[1] + s * local_x + c * local_y
    return np.column_stack((world_x, world_y))


def world_to_scene_pixels(scene_map, world_xy: np.ndarray) -> np.ndarray:
    if not len(world_xy):
        return np.empty((0, 2), dtype=np.int32)
    world_xyz = np.column_stack((world_xy, np.zeros(len(world_xy), dtype=float)))
    return np.asarray(scene_map.pos_m_to_px(world_xyz), dtype=np.int32)


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    ros_image, resolution, origin_xy, origin_yaw = load_ros_map(args.run_dir)

    cfg = build_scene_config(args)
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        sampler.update_scene(variant=args.variant)
        env = sampler.env
        map_cls = iTHORMap if "ithor" in args.scene_dataset.lower() else ProcTHORMap
        scene_map = map_cls.from_mj_model_path(
            model_path=env.current_model_path,
            agent_radius=args.gt_agent_radius_m,
            px_per_m=args.gt_px_per_m,
            device_id=None,
        )
    finally:
        sampler.close()

    gt_free_rc = np.argwhere(scene_map.occupancy)
    gt_world = scene_map.pos_px_to_m(gt_free_rc[:, :2])[:, :2]
    sampled = sample_ros_map(ros_image, resolution, origin_xy, origin_yaw, gt_world)
    observed = sampled != 205
    mapped_free = sampled >= 250
    mapped_occupied = sampled <= 50

    cell_area_m2 = 1.0 / float(scene_map.px_per_m) ** 2
    gt_count = int(len(gt_free_rc))
    observed_count = int(np.count_nonzero(observed))
    mapped_free_count = int(np.count_nonzero(mapped_free))
    mapped_occupied_count = int(np.count_nonzero(mapped_occupied))
    trajectory_world, trajectory_yaw = load_trajectory(args.run_dir)
    frontier_world = load_recorded_frontiers(args.run_dir)
    frontier_source = "explorer_status"
    if not len(frontier_world):
        frontier_world = reconstruct_frontiers(
            ros_image,
            resolution,
            origin_xy,
            origin_yaw,
        )
        frontier_source = "final_occ_reconstruction"

    result = {
        "scene_dataset": args.scene_dataset,
        "data_split": args.data_split,
        "house_ind": args.house_ind,
        "gt_agent_radius_m": args.gt_agent_radius_m,
        "gt_px_per_m_requested": args.gt_px_per_m,
        "gt_px_per_m_effective": float(scene_map.px_per_m),
        "gt_navigable_cells": gt_count,
        "gt_navigable_area_m2": gt_count * cell_area_m2,
        "observed_gt_cells": observed_count,
        "observed_gt_area_m2": observed_count * cell_area_m2,
        "exploration_coverage_ratio": observed_count / gt_count if gt_count else 0.0,
        "mapped_free_gt_cells": mapped_free_count,
        "mapped_free_gt_area_m2": mapped_free_count * cell_area_m2,
        "mapped_free_coverage_ratio": mapped_free_count / gt_count if gt_count else 0.0,
        "mapped_occupied_on_gt_free_cells": mapped_occupied_count,
        "mapped_occupied_on_gt_free_ratio": mapped_occupied_count / gt_count if gt_count else 0.0,
        "ros_map_resolution_m": resolution,
        "ros_map_origin": [float(origin_xy[0]), float(origin_xy[1]), origin_yaw],
        "trajectory_samples": int(len(trajectory_world)),
        "final_frontier_cells": int(len(frontier_world)),
        "final_frontier_source": frontier_source,
        "render_header_height_px": 122,
    }
    result_path = args.run_dir / "exploration_coverage.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    coverage = np.zeros(scene_map.occupancy.shape, dtype=np.uint8)
    coverage[scene_map.occupancy] = 1
    coverage[gt_free_rc[observed, 0], gt_free_rc[observed, 1]] = 2
    coverage[gt_free_rc[mapped_free, 0], gt_free_rc[mapped_free, 1]] = 3
    coverage[gt_free_rc[mapped_occupied, 0], gt_free_rc[mapped_occupied, 1]] = 4
    colors = np.asarray(
        [
            [38, 42, 46],
            [183, 202, 224],
            [246, 183, 95],
            [76, 175, 80],
            [211, 47, 47],
        ],
        dtype=np.uint8,
    )
    rgb = colors[coverage]
    trajectory_px = world_to_scene_pixels(scene_map, trajectory_world)
    valid_trajectory = (
        (trajectory_px[:, 0] >= 0)
        & (trajectory_px[:, 0] < rgb.shape[0])
        & (trajectory_px[:, 1] >= 0)
        & (trajectory_px[:, 1] < rgb.shape[1])
    ) if len(trajectory_px) else np.zeros((0,), dtype=bool)
    trajectory_px = trajectory_px[valid_trajectory]
    if len(trajectory_px) >= 2:
        cv2.polylines(
            rgb,
            [trajectory_px[:, [1, 0]].reshape((-1, 1, 2))],
            False,
            (0, 170, 255),
            max(2, int(round(scene_map.px_per_m * 0.035))),
            cv2.LINE_AA,
        )
    frontier_px = world_to_scene_pixels(scene_map, frontier_world)
    frontier_radius = max(2, int(round(scene_map.px_per_m * 0.035)))
    for row, col in frontier_px:
        if 0 <= row < rgb.shape[0] and 0 <= col < rgb.shape[1]:
            cv2.circle(rgb, (int(col), int(row)), frontier_radius, (128, 35, 180), -1, cv2.LINE_AA)
    if len(trajectory_px):
        end_row, end_col = trajectory_px[-1]
        cv2.circle(
            rgb,
            (int(end_col), int(end_row)),
            max(4, int(round(scene_map.px_per_m * 0.07))),
            (0, 90, 255),
            -1,
            cv2.LINE_AA,
        )
    scale = max(1.0, min(4.0, 1200.0 / max(rgb.shape[:2])))
    rendered = cv2.resize(
        rgb,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    header_height = int(result["render_header_height_px"])
    output_width = max(rendered.shape[1], 1050)
    header = np.full((header_height, output_width, 3), 255, dtype=np.uint8)
    if rendered.shape[1] < output_width:
        padded = np.full((rendered.shape[0], output_width, 3), colors[0], dtype=np.uint8)
        offset = (output_width - rendered.shape[1]) // 2
        padded[:, offset:offset + rendered.shape[1]] = rendered
        rendered = padded
    cv2.putText(
        header,
        f"House {args.house_ind}: observed={result['exploration_coverage_ratio']:.1%}  "
        f"mapped_free={result['mapped_free_coverage_ratio']:.1%}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "blue=unobserved GT free  orange=observed  green=mapped free  red=false occupied",
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"cyan=trajectory  blue=end pose  purple=pending frontier ({result['final_frontier_source']})",
        (15, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    output_rgb = np.vstack([header, rendered])
    cv2.imwrite(
        str(args.run_dir / "exploration_coverage.png"),
        cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
