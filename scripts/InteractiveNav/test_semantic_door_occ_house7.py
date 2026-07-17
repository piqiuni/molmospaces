#!/usr/bin/env python3
"""Exercise semantic door-state and occupancy overlay logic on a real scene model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_SCRIPTS = REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_mapping_py_pkg" / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SEMANTIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_SCRIPTS))

from molmo_spaces.env.data_views import Door
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.semantic_occ_overlay import SemanticOccupancyOverlay

from force_interaction_runtime import ForceDriveConfig, open_door_root_with_force
from read_scene_room_properties import build_scene_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house-ind", type=int, default=7)
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--robot", default="rby1", choices=["droid", "rby1", "rum"])
    parser.add_argument("--variant", default="base", choices=["base", "ceiling", "map"])
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--clear-padding-m", type=float, default=0.1)
    parser.add_argument("--interaction-mode", choices=["direct", "force"], default="direct")
    parser.add_argument("--force-max-physics-substeps", type=int, default=2500)
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic_door_occ_house7.json"))
    return parser.parse_args()


def scene_args(args):
    return SimpleNamespace(
        seed=2,
        robot=args.robot,
        scene_dataset=args.scene_dataset,
        data_split=args.data_split,
        house_ind=args.house_ind,
    )


def closed_value(joint_range):
    lower, upper = float(joint_range[0]), float(joint_range[1])
    if lower <= 0.0 <= upper:
        return 0.0
    return lower if abs(lower) <= abs(upper) else upper


def open_value(joint_range, closed):
    lower, upper = float(joint_range[0]), float(joint_range[1])
    return lower if abs(lower - closed) >= abs(upper - closed) else upper


def collect_door_groups(env):
    model = env.current_model
    data = env.current_data
    object_manager = env.object_managers[env.current_batch_index]
    groups = {}
    for door_name in object_manager.find_door_names():
        try:
            door = Door(door_name, data)
            hinge_index = door.get_hinge_joint_index()
        except (KeyError, ValueError):
            continue
        body_id = int(model.body(door_name).id)
        root_id = int(model.body(body_id).rootid[0])
        groups.setdefault(root_id, []).append((door_name, door, hinge_index))
    return groups


def root_observation(env, root_id, leaves):
    model = env.current_model
    data = env.current_data
    root_name = str(model.body(root_id).name or f"door_root_{root_id}")
    try:
        center, size = body_aabb(model, data, root_id, visual_only=True)
    except Exception:
        center = data.xpos[root_id]
        size = [0.1, 0.1, 0.1]
    joint_infos = []
    for _door_name, door, hinge_index in leaves:
        joint_infos.append(
            {
                "joint_name": str(door.joint_names[hinge_index]),
                "joint_type": "hinge",
                "joint_range": [float(value) for value in door.get_joint_range(hinge_index)],
                "joint_value": float(door.get_joint_position(hinge_index)),
            }
        )
    primary = joint_infos[0]
    return {
        "observation_id": f"house7_{root_name}",
        "instance_id": root_name,
        "semantic_name": "door",
        "category": "Door",
        "confidence": 1.0,
        "position": [float(value) for value in center],
        "aabb_center": [float(value) for value in center],
        "aabb_size": [float(value) for value in size],
        "room_id": None,
        "parent": None,
        "children": [name for name, _door, _hinge in leaves],
        "is_receptacle": False,
        "is_pickup_candidate": False,
        "is_articulable": True,
        "is_door": True,
        "is_movable_door": True,
        "joint_infos": joint_infos,
        "primary_joint_name": primary["joint_name"],
        "joint_type": primary["joint_type"],
        "joint_range": primary["joint_range"],
        "joint_value": primary["joint_value"],
        "source": "house7_integration_test",
        "name": root_name,
    }


def make_grid_info(center, size, resolution):
    extent_x = max(float(size[0]) * 0.5 + 1.0, 1.5)
    extent_y = max(float(size[1]) * 0.5 + 1.0, 1.5)
    width = max(4, int(math.ceil(2.0 * extent_x / resolution)))
    height = max(4, int(math.ceil(2.0 * extent_y / resolution)))
    origin = SimpleNamespace(
        position=SimpleNamespace(
            x=float(center[0]) - extent_x,
            y=float(center[1]) - extent_y,
            z=0.0,
        ),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return SimpleNamespace(width=width, height=height, resolution=float(resolution), origin=origin)


def set_group_state(env, leaves, use_open_state):
    for _door_name, door, hinge_index in leaves:
        limits = door.get_joint_range(hinge_index)
        closed = closed_value(limits)
        target = open_value(limits, closed) if use_open_state else closed
        door.set_joint_position(hinge_index, target)
    mujoco.mj_forward(env.current_model, env.current_data)


def run_group(env, root_id, leaves, args):
    originals = [float(door.get_joint_position(hinge_index)) for _name, door, hinge_index in leaves]
    try:
        set_group_state(env, leaves, use_open_state=False)
        closed_observation = root_observation(env, root_id, leaves)
        store = InteractionGraphStore(scene_id=f"house_{args.house_ind}")
        store.update_observations([closed_observation], source_mode="realtime_gt_observation")
        closed_graph = store.as_graph_dict()
        closed_portal = next(node for node in closed_graph["nodes"] if node["type"] == "portal")
        if closed_portal["interaction"]["state"] != "closed":
            raise AssertionError(f"Expected closed state, got {closed_portal['interaction']['state']}")

        overlay = SemanticOccupancyOverlay(clear_padding_m=args.clear_padding_m)
        overlay.update_graph(closed_graph)
        grid_info = make_grid_info(
            closed_observation["aabb_center"],
            closed_observation["aabb_size"],
            args.resolution,
        )
        raw = [100] * (int(grid_info.width) * int(grid_info.height))
        closed_stats = overlay.apply(grid_info, raw)[2]
        if closed_stats["cleared_cells"] != 0:
            raise AssertionError("Closed portal unexpectedly cleared occupancy")

        force_result = None
        if args.interaction_mode == "force":
            force_result = open_door_root_with_force(
                env,
                closed_observation["instance_id"],
                config=ForceDriveConfig(
                    max_physics_substeps=args.force_max_physics_substeps,
                ),
            )
        else:
            set_group_state(env, leaves, use_open_state=True)
        open_observation = root_observation(env, root_id, leaves)
        store.update_observations([open_observation], source_mode="realtime_gt_observation")
        open_graph = store.as_graph_dict()
        open_portal = next(node for node in open_graph["nodes"] if node["type"] == "portal")
        if open_portal["interaction"]["state"] != "open":
            raise AssertionError(f"Expected open state, got {open_portal['interaction']['state']}")
        overlay.update_graph(open_graph)
        planning, _mask, open_stats = overlay.apply(grid_info, raw)
        if open_stats["cleared_cells"] <= 0 or min(planning) != 0:
            raise AssertionError("Open portal did not clear its cached closed AABB")

        return {
            "root_body_name": closed_observation["instance_id"],
            "leaf_body_names": [name for name, _door, _hinge in leaves],
            "leaf_count": len(leaves),
            "closed_state": closed_portal["interaction"]["state"],
            "open_state": open_portal["interaction"]["state"],
            "open_fraction": open_portal["interaction"].get("open_fraction"),
            "closed_aabb_center": closed_observation["aabb_center"],
            "closed_aabb_size": closed_observation["aabb_size"],
            "cleared_cells": int(open_stats["cleared_cells"]),
            "joint_infos_closed": closed_observation["joint_infos"],
            "joint_infos_open": open_observation["joint_infos"],
            "interaction_mode": args.interaction_mode,
            "force_result": force_result,
        }
    finally:
        for original, (_door_name, door, hinge_index) in zip(originals, leaves):
            door.set_joint_position(hinge_index, original)
        mujoco.mj_forward(env.current_model, env.current_data)


def main():
    args = parse_args()
    cfg = build_scene_config(scene_args(args))
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        sampler.update_scene(variant=args.variant)
        env = sampler.env
        groups = collect_door_groups(env)
        if not groups:
            raise RuntimeError(f"No controllable doorway roots found in house {args.house_ind}")
        results = [run_group(env, root_id, leaves, args) for root_id, leaves in sorted(groups.items())]
        payload = {
            "scene_dataset": args.scene_dataset,
            "data_split": args.data_split,
            "house_ind": args.house_ind,
            "interaction_mode": args.interaction_mode,
            "doorway_root_count": len(results),
            "single_door_root_count": sum(result["leaf_count"] == 1 for result in results),
            "multi_leaf_root_count": sum(result["leaf_count"] > 1 for result in results),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"House {args.house_ind} semantic door OCC test passed: {args.output}")
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
