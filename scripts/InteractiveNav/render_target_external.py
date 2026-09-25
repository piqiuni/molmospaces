"""Render an offline, roofless front-above view of a frozen benchmark target."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def camera_settings(episode: dict, distance: float, elevation: float, azimuth: float | None) -> dict:
    target = episode["interactive_nav"]["target"]
    center = list(target.get("container_aabb_center") or target["object_aabb_center"])
    source = "explicit_azimuth" if azimuth is not None else "fallback_positive_y"
    if azimuth is None:
        azimuth = 270.0
        for step in episode["interactive_nav"].get("oracle_plan", {}).get("steps", []):
            if step.get("reason") != "approach_container_interaction":
                continue
            point = step["goal_point"]
            delta_x, delta_y = point[0] - center[0], point[1] - center[1]
            if math.hypot(delta_x, delta_y) > 1e-6:
                azimuth = math.degrees(math.atan2(delta_y, delta_x)) + 180.0
                source = "frozen_container_approach_direction"
                break
    return {"lookat": center, "distance": distance, "elevation": elevation, "azimuth": azimuth % 360.0, "direction_source": source}


def structural_hidden(name: str, hide_walls: bool) -> bool:
    lowered = name.lower()
    return lowered.startswith(("roof", "ceiling")) or (hide_walls and lowered.startswith("wall_"))


def render(args: argparse.Namespace) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw

    payload = json.loads(args.benchmark.read_text())
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if not isinstance(episodes, list):
        raise ValueError("Benchmark episodes must be a list")
    episode = episodes[args.episode_index]
    target = episode["interactive_nav"]["target"]
    scene_path = args.scene_model
    if scene_path is None:
        scene_path = args.scenes_root / f"{episode['scene_dataset']}-{episode['data_split']}" / f"{episode['data_split']}_{episode['house_index']}.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    critical = {target["selected_instance"], target.get("container_name")} - {None}
    for name in critical:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"Missing task-critical object: {name}")
    skipped = []
    applied_poses = 0
    modifications = episode.get("scene_modifications", {})
    if modifications.get("added_objects") or modifications.get("removed_objects"):
        raise ValueError("This renderer does not support added/removed benchmark objects")
    for name, pose in modifications.get("object_poses", {}).items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = int(model.body_jntadr[body_id]) if body_id >= 0 else -1
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            if name in critical:
                raise ValueError(f"Cannot restore task-critical free pose: {name}")
            skipped.append(name)
            continue
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address:address + 7] = pose
        applied_poses += 1
    for state in modifications.get("articulation_states", []):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, state["joint_name"])
        if joint_id < 0:
            raise ValueError(f"Missing recorded joint: {state['joint_name']}")
        data.qpos[model.jnt_qposadr[joint_id]] = state["position"]
    mujoco.mj_forward(model, data)
    hidden = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        if structural_hidden(name, args.hide_walls) or structural_hidden(body_name, args.hide_walls):
            model.geom_rgba[geom_id, 3] = 0.0
            hidden.append(name or body_name)
    settings = camera_settings(episode, args.distance, args.elevation, args.azimuth)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = settings["lookat"]
    camera.distance = settings["distance"]
    camera.elevation = settings["elevation"]
    camera.azimuth = settings["azimuth"]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    model.vis.quality.shadowsize = min(model.vis.quality.shadowsize, 2048)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        renderer.update_scene(data, camera=camera)
        pixels = np.asarray(renderer.render()).copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    finally:
        renderer.close()
    geom_pixels = segmentation[..., 0]
    pixels[geom_pixels < 0] = 255
    focus_name = target.get("container_name") or target["selected_instance"]
    focus_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, focus_name)
    focus_geoms = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0 and body_id != focus_body:
            body_id = int(model.body_parentid[body_id])
        if body_id == focus_body:
            focus_geoms.append(geom_id)
    focus_mask = np.isin(geom_pixels, focus_geoms) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(focus_mask.astype(np.uint8), 8)
    if component_count > 1:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        focus_mask = labels == component
    rows, columns = np.nonzero(focus_mask)
    focus_bbox = None
    image = Image.fromarray(pixels)
    if len(rows):
        focus_bbox = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
        draw = ImageDraw.Draw(image)
        draw.rectangle(focus_bbox, outline=(255, 150, 0), width=3)
        draw.text((focus_bbox[0], max(0, focus_bbox[1] - 16)), "TARGET CONTAINER" if target.get("container_name") else "TARGET", fill=(255, 100, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        "schema_version": "target_external_view_v1",
        "evaluator_private": True,
        "episode_index": args.episode_index,
        "house_index": episode["house_index"],
        "scene_model": str(scene_path.resolve()),
        "target": target,
        "camera": settings,
        "state": "frozen_initial_object_poses_and_articulations_no_physics_rollout",
        "not_policy_observation": True,
        "robot_present": False,
        "hide_walls": args.hide_walls,
        "hidden_structural_geoms": hidden,
        "focus_visible_pixels": int(len(rows)),
        "focus_bbox_xyxy": focus_bbox,
        "applied_object_pose_count": applied_poses,
        "skipped_noncritical_pose_names": skipped,
        "image": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--scene-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument("--elevation", type=float, default=-35.0)
    parser.add_argument("--azimuth", type=float)
    parser.add_argument("--hide-walls", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()
    if args.distance <= 0 or args.width <= 0 or args.height <= 0 or not -90 < args.elevation < 0:
        parser.error("Positive distance/resolution and -90 < elevation < 0 required")
    if args.episode_index < 0:
        parser.error("episode-index must be nonnegative")
    print(json.dumps(render(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
