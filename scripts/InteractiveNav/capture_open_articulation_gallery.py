"""Capture static external-camera images of opened articulated objects.

The gallery intentionally writes named joint states directly (GT setup) and
does not run an opening policy or a long simulation rollout.  Each selected
scene is loaded once, the robot is placed at a pose that is collision-free for
the requested open sequence, and one RGB frame is rendered.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.utils.pose import (
    compute_lookat_forward_up,
    pose_mat_to_7d,
    pos_quat_to_pose_mat,
)
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi


DEFAULT_OUTPUT = REPO_ROOT / "scripts/InteractiveNav/output/open_articulation_gallery_20260720_v1"


@dataclass(frozen=True)
class GalleryTarget:
    gallery_id: str
    kind: str
    house_index: int
    asset_id: str
    joint_indices: tuple[int, ...] = ()
    camera_profile: str = "rear_shoulder"


TARGETS = (
    GalleryTarget("door_single", "door", 1, "Doorway_10"),
    GalleryTarget("door_double", "door", 1, "Doorway_Double_1"),
    GalleryTarget("fridge_5", "container", 1, "Fridge_5", (0,)),
    GalleryTarget("fridge_14", "container", 7, "Fridge_14", (1, 2)),
    GalleryTarget("fridge_15", "container", 10, "Fridge_15", (0,)),
    GalleryTarget("fridge_19", "container", 0, "Fridge_19", (2, 3)),
    GalleryTarget("drawer_4", "container", 1, "Dresser_218_1", (0,), "high_overhead"),
    GalleryTarget(
        "drawer_6", "container", 36, "RoboTHOR_dresser_birkeland", (0,), "high_overhead"
    ),
    GalleryTarget("drawer_12", "container", 31, "Dresser_224_1", (0, 6, 11), "high_overhead"),
    GalleryTarget("box", "container", 3, "Box_23", (1, 2, 3, 4), "topdown_midpoint"),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")
    path.chmod(0o644)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower() or "item"


def scene_args(seed: int, variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        scene_dataset="procthor-10k",
        data_split="train",
        robot="rby1",
        variant=variant,
    )


def yaw_to_face(source_xy: np.ndarray, target_xy: np.ndarray) -> float:
    delta = np.asarray(target_xy, dtype=float) - np.asarray(source_xy, dtype=float)
    return float(math.atan2(delta[1], delta[0]))


def make_pose(robot_view: Any, xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    return probe.make_robot_pose_from_xy(robot_view, xy, yaw_to_face(xy, target_xy))


def find_container(containers: list[dict[str, Any]], asset_id: str) -> dict[str, Any]:
    matches = [row for row in containers if row.get("asset_id") == asset_id]
    if not matches:
        raise ValueError(f"No articulated container with asset_id={asset_id}")
    return matches[0]


def find_door(
    records: list[dict[str, Any]],
    doorway_records: list[dict[str, Any]],
    asset_id: str,
) -> dict[str, Any]:
    roots = {
        row["name"]: row
        for row in doorway_records
        if row.get("is_movable_door") and row.get("name")
    }
    candidates = [
        row
        for row in records
        if row.get("asset_id") == asset_id and row.get("name") in roots
    ]
    if not candidates:
        raise ValueError(f"No interactive doorway with asset_id={asset_id}")
    result = dict(roots[candidates[0]["name"]])
    result.update({"asset_id": asset_id, "scene_record": candidates[0]})
    return result


def set_container_open(
    ctx: probe.LoadedContext,
    rec: dict[str, Any],
    joint_indices: tuple[int, ...],
) -> None:
    probe.set_all_articulation_joints_closed(ctx.env, rec, rec["joints"])
    by_index = {int(row["joint_index"]): row for row in rec["joints"]}
    for index in joint_indices:
        if index not in by_index:
            raise ValueError(f"Joint index {index} missing from {rec['name']}")
        probe.set_articulation_state_by_record(
            ctx.env, rec, index, float(by_index[index]["open_value"])
        )


def stage_box_in_open_area(
    ctx: probe.LoadedContext,
    rec: dict[str, Any],
) -> dict[str, Any]:
    env = ctx.env
    original_pose = probe.free_joint_pose(env, rec["name"])
    if original_pose is None:
        raise ValueError(f"Box does not expose a free-joint pose: {rec['name']}")
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    clearance = distance_transform_edt(np.asarray(thormap.occupancy, dtype=bool)) / float(
        thormap.px_per_m
    )
    stage_px = np.asarray(np.unravel_index(int(np.argmax(clearance)), clearance.shape))
    stage_xy = np.asarray(thormap.pos_px_to_m(stage_px[None, :])[0, :2], dtype=float)
    staged_pose = original_pose.copy()
    staged_pose[:2, 3] = stage_xy
    staged_pose[2, 3] = max(0.16, 0.5 * float(rec["aabb_size"][2]) + 0.02)
    if not probe.set_free_joint_pose(env, rec["name"], staged_pose):
        raise ValueError(f"Failed to stage Box free joint: {rec['name']}")
    center, size = probe.safe_body_aabb(
        env.current_model,
        env.current_data,
        int(rec["body_id"]),
    )
    rec["aabb_center"] = center
    rec["aabb_size"] = size
    staging = {
        "method": "gt_free_joint_open_area_staging",
        "original_pose": original_pose,
        "staged_pose": staged_pose,
        "selected_map_pixel": stage_px,
        "selected_clearance_m": float(clearance[tuple(stage_px)]),
    }
    rec["staging"] = staging
    return staging


def choose_container_pose(
    ctx: probe.LoadedContext,
    rec: dict[str, Any],
    joint_indices: tuple[int, ...],
    camera_profile: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    axis = probe.container_approach_axis(ctx.env, rec)
    by_index = {int(row["joint_index"]): row for row in rec["joints"]}
    target_center, _ = probe.joint_target_geometry(
        ctx.env,
        rec,
        by_index[joint_indices[-1]],
    )
    free_points = ctx.env.get_thormap(
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius
    ).get_free_points()

    def camera_clearance(robot_pose: np.ndarray) -> float:
        if camera_profile == "topdown_midpoint":
            return 0.0
        preview = camera_pose(
            pose_mat_to_7d(robot_pose),
            target_center,
            camera_profile,
        )
        nearest = probe.nearest_free_point(free_points, preview["position"][:2])
        if nearest is None:
            return float("inf")
        return float(np.linalg.norm(nearest[:2] - preview["position"][:2]))

    valid = probe.valid_robot_poses_for_joint_sequence(
        ctx,
        rec,
        list(joint_indices),
        desired_distance=0.95,
        max_poses=24,
        front_axis_xy=axis,
    )
    for robot_pose, pose_meta in valid:
        clearance = camera_clearance(robot_pose)
        if clearance <= 0.22:
            pose_meta["camera_free_distance_m"] = clearance
            return robot_pose, pose_meta
    if len(joint_indices) == 1:
        joint = by_index[joint_indices[0]]
        fallback_pose, fallback_meta = probe.choose_pose_valid_for_joint_states(
            ctx,
            rec,
            joint,
            float(joint["closed_value"]),
            float(joint["open_value"]),
            desired_dist=1.05,
            min_clearance_m=0.05,
            max_center_distance_m=1.85,
            allow_back_approach=True,
        )
        if fallback_pose is not None and camera_clearance(fallback_pose) <= 0.22:
            fallback_meta["search_mode"] = "extended_single_joint_fallback"
            fallback_meta["camera_free_distance_m"] = camera_clearance(fallback_pose)
            return fallback_pose, fallback_meta
    raise ValueError(
        f"No collision-free robot/camera pose for {rec['asset_id']} {joint_indices}"
    )


def choose_door_pose(
    ctx: probe.LoadedContext,
    doorway_analysis: dict[str, Any],
    rec: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center = np.asarray(rec["portal_center_xy"], dtype=float)
    normal = np.asarray(rec["portal_normal_xy"], dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    free_points = env.get_thormap(
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius
    ).get_free_points()
    if free_points.size == 0:
        raise ValueError("No free points in scene map")

    qpos_before = env.current_data.qpos.copy()
    pose_before = robot_view.base.pose.copy()
    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    try:
        for side in (-1.0, 1.0):
            for distance in (0.75, 0.95, 1.15, 1.35):
                for lateral in (0.0, 0.18, -0.18):
                    desired = center + side * normal * distance + tangent * lateral
                    free = probe.nearest_free_point(free_points, desired)
                    if free is None:
                        continue
                    pose = make_pose(robot_view, np.asarray(free[:2], dtype=float), center)
                    emi.set_door_root_state(env, doorway_analysis, rec["name"], "closed")
                    closed_collision = bool(env.check_if_robot_collision_at_base_pose(robot_view, pose))
                    emi.set_door_root_state(env, doorway_analysis, rec["name"], "open")
                    open_collision = bool(env.check_if_robot_collision_at_base_pose(robot_view, pose))
                    if closed_collision or open_collision:
                        continue
                    pose_7d = pose_mat_to_7d(pose)
                    preview_camera = camera_pose(
                        pose_7d,
                        np.asarray([center[0], center[1], 1.0], dtype=float),
                        "rear_shoulder",
                    )
                    camera_free = probe.nearest_free_point(
                        free_points,
                        np.asarray(preview_camera["position"][:2], dtype=float),
                    )
                    if camera_free is None:
                        continue
                    camera_free_distance = float(
                        np.linalg.norm(camera_free[:2] - preview_camera["position"][:2])
                    )
                    if camera_free_distance > 0.22:
                        continue
                    score = (
                        abs(distance - 0.95)
                        + abs(lateral)
                        + float(np.linalg.norm(free[:2] - desired))
                        + camera_free_distance
                    )
                    candidates.append(
                        (
                            score,
                            pose.copy(),
                            {
                                "desired_xy": desired.tolist(),
                                "free_point_xy": np.asarray(free[:2], dtype=float).tolist(),
                                "closed_collision": closed_collision,
                                "open_collision": open_collision,
                                "camera_free_distance_m": camera_free_distance,
                            },
                        )
                    )
    finally:
        env.current_data.qpos[:] = qpos_before
        robot_view.base.pose = pose_before
        mujoco.mj_forward(env.current_model, env.current_data)
    if not candidates:
        raise ValueError(f"No collision-free robot pose for door {rec['name']}")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def camera_pose(
    robot_pose_7d: np.ndarray,
    target_xyz: np.ndarray,
    profile: str,
) -> dict[str, np.ndarray]:
    pose = np.asarray(robot_pose_7d, dtype=float)
    w, x, y, z = pose[3:7]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
    right = np.array([math.sin(yaw), -math.cos(yaw)], dtype=float)
    base = pose[:3]
    if profile == "rear_shoulder":
        behind, lateral, height, look_z, fov = 1.20, 0.85, 1.75, 0.95, 66.0
    elif profile == "rear_shoulder_close":
        behind, lateral, height, look_z, fov = 0.58, 0.48, 1.62, 0.78, 72.0
    elif profile == "right_side_rear":
        # Stronger camera displacement toward the robot's right side.  With
        # the robot facing the target, this keeps the robot on the left of the
        # frame and the articulated object on the right.
        behind, lateral, height, look_z, fov = 0.85, 1.35, 1.75, 0.95, 66.0
    elif profile == "right_side_rear_close":
        # Drawer_4 sits close to a wall; keep the right-side bias while moving
        # the camera inward so the dresser is not hidden behind the wall.
        behind, lateral, height, look_z, fov = 0.58, 0.72, 1.62, 0.78, 72.0
    elif profile == "high_overhead":
        behind, lateral, height, fov = 0.85, 0.52, 3.20, 70.0
        look_z = float(np.clip(target_xyz[2], 0.35, 1.05))
    elif profile == "topdown_midpoint":
        midpoint_xy = 0.35 * base[:2] + 0.65 * np.asarray(target_xyz[:2], dtype=float)
        position = np.asarray(
            [midpoint_xy[0] - 0.18, midpoint_xy[1] - 0.15, base[2] + 2.80],
            dtype=np.float32,
        )
        target = np.asarray(
            [midpoint_xy[0], midpoint_xy[1], float(np.clip(target_xyz[2], 0.35, 1.05))],
            dtype=np.float32,
        )
        forward_3d, up = compute_lookat_forward_up(position, target)
        return {
            "position": position,
            "target": target,
            "forward": np.asarray(forward_3d, dtype=np.float32),
            "up": np.asarray(up, dtype=np.float32),
            "fov_deg": np.float32(64.0),
        }
    else:
        raise ValueError(f"Unknown camera profile: {profile}")
    position = np.array(
        [
            base[0] - behind * forward[0] + lateral * right[0],
            base[1] - behind * forward[1] + lateral * right[1],
            base[2] + height,
        ],
        dtype=np.float32,
    )
    target = np.asarray(target_xyz, dtype=float).copy()
    target[2] = float(look_z)
    forward_3d, up = compute_lookat_forward_up(position, target)
    return {
        "position": position,
        "target": target.astype(np.float32),
        "forward": np.asarray(forward_3d, dtype=np.float32),
        "up": np.asarray(up, dtype=np.float32),
        "fov_deg": np.float32(fov),
    }


def render_target(
    ctx: probe.LoadedContext,
    target: GalleryTarget,
    rec: dict[str, Any],
    robot_pose: np.ndarray,
    output_path: Path,
    collision_meta: dict[str, Any],
) -> dict[str, Any]:
    env = ctx.env
    if target.kind == "door":
        target_xyz = np.asarray(rec["portal_center_xy"].tolist() + [1.0], dtype=float)
    elif target.asset_id.startswith("Box_"):
        target_xyz = np.asarray(rec["aabb_center"], dtype=float)
    else:
        target_xyz = np.asarray(rec["aabb_center"], dtype=float)
    pose_7d = pose_mat_to_7d(robot_pose)
    env.current_robot.robot_view.base.pose = pos_quat_to_pose_mat(pose_7d)
    mujoco.mj_forward(env.current_model, env.current_data)
    env.camera_manager.registry.update_all_cameras(env)
    camera = camera_pose(pose_7d, target_xyz, target.camera_profile)
    camera_name = f"gallery_{slug(target.gallery_id)}"
    env.camera_manager.add_camera(
        camera_name,
        camera["position"],
        camera["forward"],
        camera["up"],
        fov=float(camera["fov_deg"]),
    )
    rgb = env.render_rgb_frame(camera_name)
    probe.save_rgb_image(output_path, rgb)
    output_path.chmod(0o644)
    return {
        "gallery_id": target.gallery_id,
        "kind": target.kind,
        "house_index": target.house_index,
        "asset_id": target.asset_id,
        "object_name": rec["name"],
        "joint_indices": list(target.joint_indices),
        "joint_readback": (
            [
                {
                    "joint_name": row["joint_name"],
                    "value": probe.joint_value_by_name(env, row["joint_name"]),
                }
                for row in rec.get("joints", [])
                if int(row["joint_index"]) in set(target.joint_indices)
            ]
            if target.kind != "door"
            else [
                {
                    "joint_name": transition["joint_name"],
                    "value": probe.joint_value_by_name(env, transition["joint_name"]),
                }
                for transition in rec.get("door_transition", {}).get("transitions", [])
            ]
        ),
        "robot_pose_7d": pose_7d.tolist(),
        "robot_collision": bool(
            env.check_if_robot_collision_at_base_pose(
                env.current_robot.robot_view,
                robot_pose,
            )
        ),
        "collision_pose_search": collision_meta,
        "staging": rec.get("staging"),
        "camera_profile": target.camera_profile,
        "camera": {
            "name": camera_name,
            "position": camera["position"].tolist(),
            "target": camera["target"].tolist(),
            "forward": camera["forward"].tolist(),
            "up": camera["up"].tolist(),
            "fov_deg": float(camera["fov_deg"]),
        },
        "image": output_path.name,
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
    }


def capture_group(
    house_index: int,
    targets: list[GalleryTarget],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    ctx = None
    rows: list[dict[str, Any]] = []
    try:
        ctx = probe.load_scene_context(scene_args(args.seed, args.variant), house_index)
        records, containers = probe.collect_scene_records(ctx)
        doorway_analysis = None
        doorway_records: list[dict[str, Any]] = []
        if any(target.kind == "door" for target in targets):
            emi.ensure_runtime_dependencies()
            doorway_analysis = emi.collect_runtime_doorway_analysis(ctx.env)
            doorway_records = emi.collect_interactive_door_root_object_records(
                ctx.env,
                doorway_analysis,
            )
        for target in targets:
            try:
                if target.kind == "door":
                    if doorway_analysis is None:
                        raise RuntimeError("Doorway analysis was not initialized")
                    rec = find_door(records, doorway_records, target.asset_id)
                    robot_pose, pose_meta = choose_door_pose(ctx, doorway_analysis, rec)
                    rec["door_transition"] = emi.set_door_root_state(
                        ctx.env,
                        doorway_analysis,
                        rec["name"],
                        "open",
                    )
                else:
                    rec = find_container(containers, target.asset_id)
                    if target.asset_id.startswith("Box_"):
                        stage_box_in_open_area(ctx, rec)
                    robot_pose, pose_meta = choose_container_pose(
                        ctx,
                        rec,
                        target.joint_indices,
                        target.camera_profile,
                    )
                    set_container_open(ctx, rec, target.joint_indices)
                output_path = output_dir / (
                    f"{target.gallery_id}__h{house_index}__{slug(target.asset_id)}.png"
                )
                row = render_target(ctx, target, rec, robot_pose, output_path, pose_meta)
                row["scene_dataset"] = "procthor-10k"
                row["data_split"] = "train"
                rows.append(row)
                print(f"captured {target.gallery_id}: {output_path}", flush=True)
            except Exception as exc:  # keep the rest of the small gallery running
                rows.append(
                    {
                        "gallery_id": target.gallery_id,
                        "kind": target.kind,
                        "house_index": house_index,
                        "asset_id": target.asset_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"failed {target.gallery_id}: {exc}", flush=True)
    finally:
        if ctx is not None:
            probe.close_context(ctx)
    return rows


def save_contact_sheet(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("image")]
    if not ok_rows:
        return
    cols = 3
    rows_n = int(math.ceil(len(ok_rows) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(15, 4.4 * rows_n), squeeze=False)
    for ax, row in zip(axes.reshape(-1), ok_rows, strict=False):
        image = plt.imread(output_dir / row["image"])
        ax.imshow(image)
        ax.set_title(f"{row['gallery_id']} · {row['asset_id']}", fontsize=10)
        ax.axis("off")
    for ax in axes.reshape(-1)[len(ok_rows) :]:
        ax.axis("off")
    fig.suptitle("Opened articulation gallery · external cameras", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = output_dir / "contact_sheet.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    path.chmod(0o644)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--only", nargs="*", help="Capture only selected gallery IDs")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Replace selected rows in an existing manifest and preserve all other rows",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [target.gallery_id for target in TARGETS])
    targets = [target for target in TARGETS if target.gallery_id in selected]
    grouped: dict[int, list[GalleryTarget]] = {}
    for target in targets:
        grouped.setdefault(target.house_index, []).append(target)
    all_rows: list[dict[str, Any]] = []
    for house_index in sorted(grouped):
        all_rows.extend(capture_group(house_index, grouped[house_index], args, output_dir))
    manifest_path = output_dir / "manifest.json"
    if args.append and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        selected_ids = {target.gallery_id for target in targets}
        all_rows = [
            row for row in existing.get("rows", []) if row.get("gallery_id") not in selected_ids
        ] + all_rows
        order = {target.gallery_id: index for index, target in enumerate(TARGETS)}
        all_rows.sort(key=lambda row: order.get(row.get("gallery_id"), len(order)))
    write_json(
        manifest_path,
        {
            "schema_version": "open_articulation_gallery_v1",
            "description": "Static GT-opened articulation images from external cameras",
            "no_policy_rollout": True,
            "targets_requested": [row.get("gallery_id") for row in all_rows],
            "rows": all_rows,
        },
    )
    save_contact_sheet(output_dir, all_rows)
    ok = sum(bool(row.get("image")) for row in all_rows)
    failed = sum(row.get("status") == "failed" for row in all_rows)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "captured": ok, "failed": failed},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
