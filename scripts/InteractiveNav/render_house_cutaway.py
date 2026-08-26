from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.utils.pose import pos_quat_to_pose_mat
from scripts.InteractiveNav import build_container_interaction_benchmark as container_builder
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi


DEFAULT_CATALOG = REPO_ROOT / (
    "scripts/InteractiveNav/output/mixed_rough_catalog_occfix_all_strict_v2_20260718/"
    "mixed_rough_catalog.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "scripts/InteractiveNav/output/mixed_shortcut_visualizations_20260721/"
    "house_783_roofless_oblique.png"
)
DEFAULT_CASE_PREFIX = "mixed_h783__refrigerator"


def load_candidate(catalog_path: Path, case_prefix: str) -> tuple[dict, dict]:
    payload = json.loads(catalog_path.read_text())
    candidate = next(
        row for row in payload["candidates"] if row["case_id"].startswith(case_prefix)
    )
    return candidate, payload


def load_context(candidate: dict, payload: dict):
    episodes = container_builder.load_benchmark_episodes(Path(payload["benchmark_dir"]))
    episode_index = int(candidate["source_episode_indices"][0])
    episode = episodes[episode_index]
    args = argparse.Namespace(
        robot="rby1",
        variant="base",
        seed=0,
        output_dir=DEFAULT_OUTPUT.parent,
    )
    return container_builder.load_episode_context(args, episode)


def scene_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    emi.ensure_runtime_dependencies()
    floor_ids = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("room_") and "visual" in name:
            floor_ids.append(geom_id)
    if not floor_ids:
        raise RuntimeError("No room visual geoms found")
    center, size = emi.geom_aabb(model, data, floor_ids, tight_mesh=False)
    return np.asarray(center, dtype=float), np.asarray(size, dtype=float)


def render_house(
    output_path: Path,
    *,
    candidate: dict,
    payload: dict,
    width: int,
    height: int,
    azimuth: float,
    elevation: float,
    distance_scale: float,
    show_robot: bool,
    open_target_container: bool,
) -> dict:
    ctx = load_context(candidate, payload)
    try:
        # Open channel doors so the room layout is visible. Keep containers closed
        # because this is the original-house architectural view, not a state panel.
        container_builder.open_all_available_doors(ctx)
        _, containers = probe.collect_scene_records(ctx)
        container_builder.close_all_containers(ctx.env, containers)
        if open_target_container:
            target_container = next(
                row for row in containers if row["name"] == candidate["container_name"]
            )
            target_joint_indices = set(int(index) for index in candidate["joint_sequence"])
            for joint in target_container["joints"]:
                if int(joint["joint_index"]) in target_joint_indices:
                    probe.set_articulation_state_by_record(
                        ctx.env,
                        target_container,
                        int(joint["joint_index"]),
                        float(joint["open_value"]),
                    )
        robot_pose = np.asarray(candidate["source_robot_base_pose"], dtype=float)
        ctx.env.current_robot.robot_view.base.pose = pos_quat_to_pose_mat(robot_pose)
        # Hide the infinite structural ground plane; it is outside the house and
        # creates a distracting textured foreground in an oblique cutaway render.
        for geom_id in range(ctx.env.current_model.ngeom):
            name = mujoco.mj_id2name(
                ctx.env.current_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if name == "floor":
                ctx.env.current_model.geom_rgba[geom_id, 3] = 0.0
        mujoco.mj_forward(ctx.env.current_model, ctx.env.current_data)

        center, size = scene_bounds(ctx.env.current_model, ctx.env.current_data)
        bounds_min = center[:2] - size[:2] / 2.0
        bounds_max = center[:2] + size[:2] / 2.0
        horizontal = float(max(size[0], size[1]))
        target = center.copy()
        target[2] = 0.9
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = target
        camera.distance = horizontal * float(distance_scale)
        camera.azimuth = float(azimuth)
        camera.elevation = float(elevation)

        # MuJoCo's default offscreen framebuffer is often only 1280 px wide.
        # Increase it before constructing the renderer for a paper-resolution image.
        ctx.env.current_model.vis.global_.offwidth = max(
            int(ctx.env.current_model.vis.global_.offwidth), int(width)
        )
        ctx.env.current_model.vis.global_.offheight = max(
            int(ctx.env.current_model.vis.global_.offheight), int(height)
        )
        renderer = mujoco.Renderer(ctx.env.current_model, height=height, width=width)
        renderer.update_scene(ctx.env.current_data, camera=camera)
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()
        image = renderer.render()
        # Make the cutaway presentation paper-friendly: remove skybox/exterior
        # ground pixels while preserving all rendered house geometry.
        seg_geom = np.asarray(segmentation)[..., 0]
        background = seg_geom < 0
        if floor_ids := [
            geom_id
            for geom_id in range(ctx.env.current_model.ngeom)
            if mujoco.mj_id2name(
                ctx.env.current_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            == "floor"
        ]:
            background |= np.isin(seg_geom, floor_ids)
        # Remove robot or other helper geometry that is outside the house
        # footprint (the benchmark episode places the robot near an exterior
        # boundary, which is useful for navigation but not for an architectural
        # paper figure).
        if not show_robot:
            outside_geom_ids = []
            for geom_id in range(ctx.env.current_model.ngeom):
                xy = np.asarray(ctx.env.current_data.geom_xpos[geom_id, :2], dtype=float)
                if np.any(xy < bounds_min - 0.35) or np.any(xy > bounds_max + 0.35):
                    outside_geom_ids.append(geom_id)
            background |= np.isin(seg_geom, outside_geom_ids)
        image = np.asarray(image).copy()
        image[background] = np.asarray([255, 255, 255], dtype=np.uint8)
        renderer.close()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(np.asarray(image)).save(output_path)
        output_path.chmod(0o644)
        metadata = {
            "schema_version": "house_roofless_oblique_render_v1",
            "house_index": int(candidate["house_index"]),
            "case_id": candidate["case_id"],
            "scene_model": str(ctx.env.current_model_path),
            "roof_mode": "original_proc_thor_room_geometry_without_explicit_roof_mesh",
            "door_state": "all_interactive_doors_open",
            "container_state": (
                "target_container_open_all_other_containers_closed"
                if open_target_container
                else "all_containers_closed"
            ),
            "robot": {
                "visible": bool(show_robot),
                "pose_source": "mixed_rough_candidate.source_robot_base_pose",
                "pose": robot_pose.tolist(),
            },
            "camera": {
                "width": width,
                "height": height,
                "lookat": target.tolist(),
                "distance": float(camera.distance),
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
            },
            "image": output_path.name,
        }
        output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    finally:
        probe.close_context(ctx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--case_prefix", default=DEFAULT_CASE_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=1700)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-38.0)
    parser.add_argument("--distance_scale", type=float, default=1.20)
    parser.add_argument(
        "--show_robot", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--open_target_container",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    candidate, payload = load_candidate(args.catalog, args.case_prefix)
    metadata = render_house(
        args.output,
        candidate=candidate,
        payload=payload,
        width=args.width,
        height=args.height,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance_scale=args.distance_scale,
        show_robot=args.show_robot,
        open_target_container=args.open_target_container,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
