#!/usr/bin/env python3
"""Compare a final ROS occupancy grid with the static scene navigable area."""

from __future__ import annotations

import argparse
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
    scale = max(1.0, min(4.0, 1200.0 / max(rgb.shape[:2])))
    rendered = cv2.resize(
        rgb,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    header = np.full((92, rendered.shape[1], 3), 255, dtype=np.uint8)
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
    output_rgb = np.vstack([header, rendered])
    cv2.imwrite(
        str(args.run_dir / "exploration_coverage.png"),
        cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
