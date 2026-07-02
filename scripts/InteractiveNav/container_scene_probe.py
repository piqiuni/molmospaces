from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np

from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    RBY1GoProD455CameraSystem,
)
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig, FrankaRobotConfig, RBY1Config
from molmo_spaces.env.data_views import Door, MlSpacesArticulationObject
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.molmo_spaces_constants import get_resource_manager
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.constants.object_constants import RECEPTACLE_TYPES_THOR
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.rendering_utils import get_geom_seg_mask
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_OUTPUT_DIR = (
    Path("/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/output/container_scene_probe")
)
WRITABLE_ASSET_MIRROR = Path("/tmp/container_scene_probe_assets")
CONTAINER_TOKENS = (
    "drawer",
    "cabinet",
    "fridge",
    "refrigerator",
    "microwave",
    "oven",
    "dishwasher",
    "box",
    "chestofdrawers",
)
PORTABLE_PREFERRED_TOKENS = (
    "apple",
    "mug",
    "bowl",
    "atomizer",
    "alarmclock",
    "vase",
    "laptop",
    "basketball",
)
DEFAULT_LEFT_ARM_QPOS = np.array([0.28, 0.0, 0.0, -0.64, 0.39, -0.26, -0.04], dtype=np.float32)
DEFAULT_RIGHT_ARM_QPOS = np.array([0.28, 0.0, 0.0, -0.64, 0.39, -0.26, -0.04], dtype=np.float32)
DEFAULT_HEAD_QPOS = np.array([0.0, 0.6], dtype=np.float32)


@dataclass
class LoadedContext:
    cfg: Any
    sampler: Any
    task: Any | None
    initial_head_qpos: np.ndarray | None = None

    @property
    def env(self):
        return self.sampler.env


class SceneOnlyTaskSampler(BaseMujocoTaskSampler):
    """Load a MolmoSpaces scene without sampling a concrete task."""

    def init_scene(self, env) -> None:
        return None

    def randomize_scene(self, env, robot_view) -> None:
        return None

    def _sample_task(self, env) -> BaseMujocoTask:
        raise NotImplementedError("SceneOnlyTaskSampler only loads scenes.")


def build_scene_config(args: argparse.Namespace) -> NavToObjBaseConfig:
    cfg = NavToObjBaseConfig()
    cfg.seed = args.seed
    cfg.task_type = "nav_to_obj"
    cfg.scene_dataset = args.scene_dataset
    cfg.data_split = args.data_split
    cfg.num_workers = 1
    cfg.num_threads = 1
    cfg.use_passive_viewer = False
    cfg.use_filament = False
    cfg.record_videos = False
    cfg.task_sampler_config.task_sampler_class = SceneOnlyTaskSampler
    cfg.task_sampler_config.house_inds = []
    cfg.task_sampler_config.samples_per_house = 1
    cfg.task_sampler_config.randomize_lighting = False
    cfg.task_sampler_config.randomize_textures = False
    cfg.task_sampler_config.randomize_dynamics = False
    cfg.policy_config = AStarNavToObjPolicyConfig()

    if args.robot == "droid":
        cfg.robot_config = FrankaRobotConfig()
        cfg.camera_config = FrankaDroidCameraSystem()
        cfg.camera_config.img_resolution = (320, 240)
    elif args.robot == "rby1":
        cfg.robot_config = RBY1Config()
        cfg.camera_config = RBY1GoProD455CameraSystem()
    elif args.robot == "rum":
        cfg.robot_config = FloatingRUMRobotConfig()
    else:
        raise ValueError(f"Unsupported robot: {args.robot}")

    # For debug / visibility analysis we want deterministic camera poses across
    # repeated setup_cameras() calls, so disable all camera randomization.
    for cam in cfg.camera_config.cameras:
        if hasattr(cam, "pos_noise_range"):
            cam.pos_noise_range = None
        if hasattr(cam, "orientation_noise_degrees"):
            cam.orientation_noise_degrees = None
        if hasattr(cam, "fov_noise_degrees"):
            cam.fov_noise_degrees = None

    return cfg


def load_scene_context(args: argparse.Namespace, house_ind: int) -> LoadedContext:
    import molmo_spaces.tasks.task_sampler as task_sampler_module

    manager = get_resource_manager()
    manager.symlink_lock = False
    manager.cache_lock = False
    task_sampler_module.install_scene_with_objects_and_grasps_from_path = lambda *a, **k: {}
    cfg = build_scene_config(args)
    cfg.task_sampler_config.house_inds = [house_ind]
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=house_ind)
        original_scene_path = Path(sampler._current_house_scene_path(variant=args.variant))
        scene_path = prepare_writable_scene_path(original_scene_path)
        sampler.update_scene(scene_path=scene_path, variant=args.variant)
        env = sampler.env
        if env.current_scene_metadata is None:
            env._scene_metadata = get_scene_metadata(original_scene_path)
        initial_head_qpos = get_head_joint_position(env)
        apply_default_arm_pose(env)
        env.camera_manager.setup_cameras(env, cfg.camera_config)
        return LoadedContext(
            cfg=cfg,
            sampler=sampler,
            task=None,
            initial_head_qpos=None if initial_head_qpos is None else initial_head_qpos.copy(),
        )
    except Exception:
        sampler.close()
        raise


def prepare_writable_scene_path(scene_path: Path) -> str:
    assets_root = Path(os.environ.get("MLSPACES_ASSETS_DIR", "/home/user/ldl/molmospaces/assets"))
    mirror_root = WRITABLE_ASSET_MIRROR
    mirror_root.mkdir(parents=True, exist_ok=True)

    for top_level in ("objects", "robots", "grasps"):
        src = assets_root / top_level
        dst = mirror_root / top_level
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    refs_src = assets_root / "scenes" / "refs"
    refs_dst = mirror_root / "scenes" / "refs"
    refs_dst.parent.mkdir(parents=True, exist_ok=True)
    if refs_src.exists() and not refs_dst.exists():
        refs_dst.symlink_to(refs_src)

    rel_scene = scene_path.relative_to(assets_root / "scenes")
    dst_scene = mirror_root / "scenes" / rel_scene
    dst_scene.parent.mkdir(parents=True, exist_ok=True)
    if not dst_scene.exists():
        dst_scene.symlink_to(scene_path)

    scene_assets_dir = scene_path.parent / f"{scene_path.stem}_assets"
    if scene_assets_dir.exists():
        dst_assets_dir = dst_scene.parent / scene_assets_dir.name
        if not dst_assets_dir.exists():
            dst_assets_dir.symlink_to(scene_assets_dir)
    return str(dst_scene)


def close_context(ctx: LoadedContext) -> None:
    ctx.sampler.close()


def apply_default_arm_pose(env) -> None:
    robot_view = env.current_robot.robot_view
    if "left_arm" in robot_view.move_group_ids():
        left = robot_view.get_move_group("left_arm")
        if np.asarray(left.joint_pos).shape == DEFAULT_LEFT_ARM_QPOS.shape:
            left.joint_pos = DEFAULT_LEFT_ARM_QPOS.copy()
    if "right_arm" in robot_view.move_group_ids():
        right = robot_view.get_move_group("right_arm")
        if np.asarray(right.joint_pos).shape == DEFAULT_RIGHT_ARM_QPOS.shape:
            right.joint_pos = DEFAULT_RIGHT_ARM_QPOS.copy()
    mujoco.mj_forward(env.current_model, env.current_data)


def get_default_head_qpos(env, fallback: np.ndarray | None = None) -> np.ndarray | None:
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float32).reshape(-1)
        if fallback.shape == DEFAULT_HEAD_QPOS.shape:
            return fallback.copy()
    current = get_head_joint_position(env)
    if current is not None and current.shape == DEFAULT_HEAD_QPOS.shape:
        return current.copy()
    try:
        head_qpos = env.current_robot.exp_config.robot_config.init_qpos.get("head")
    except Exception:
        head_qpos = None
    if head_qpos is None:
        return DEFAULT_HEAD_QPOS.copy()
    head_qpos = np.asarray(head_qpos, dtype=np.float32).reshape(-1)
    if head_qpos.shape != DEFAULT_HEAD_QPOS.shape:
        return DEFAULT_HEAD_QPOS.copy()
    return head_qpos.copy()


def get_head_joint_position(env) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "head" not in robot_view.move_group_ids():
        return None
    return np.asarray(robot_view.get_move_group("head").joint_pos, dtype=np.float32).copy()


def set_head_joint_position(env, head_qpos: np.ndarray | list[float]) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "head" not in robot_view.move_group_ids():
        return None
    head_qpos = np.asarray(head_qpos, dtype=np.float32).reshape(-1)
    head_group = robot_view.get_move_group("head")
    limits = np.asarray(head_group.joint_pos_limits, dtype=np.float32)
    if head_qpos.shape[0] != limits.shape[0]:
        raise ValueError(f"Expected head qpos shape {(limits.shape[0],)}, got {head_qpos.shape}")
    clipped = np.clip(head_qpos, limits[:, 0], limits[:, 1]).astype(np.float32)
    head_group.joint_pos = clipped.copy()
    head_group.ctrl = head_group.noop_ctrl
    mujoco.mj_forward(env.current_model, env.current_data)
    return clipped


def apply_default_head_pose(env, default_qpos: np.ndarray | None = None) -> np.ndarray | None:
    default_qpos = get_default_head_qpos(env, fallback=default_qpos)
    return set_head_joint_position(env, default_qpos)


def lower_head_for_drawer_view(env, tilt_delta: float = 0.35, pan: float | None = None) -> np.ndarray | None:
    current = get_head_joint_position(env)
    if current is None:
        return None
    target = current.copy()
    if pan is not None:
        target[0] = float(pan)
    target[1] = current[1] + float(tilt_delta)
    return set_head_joint_position(env, target)


def raise_head_to_pose(env, head_qpos: np.ndarray | list[float] | None = None) -> np.ndarray | None:
    target = get_default_head_qpos(env) if head_qpos is None else np.asarray(head_qpos, dtype=np.float32)
    return set_head_joint_position(env, target)


def token_match(value: str, tokens: tuple[str, ...]) -> bool:
    lower = value.lower()
    return any(token in lower for token in tokens)


def safe_body_aabb(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return body_aabb(model, data, body_id, visual_only=True)
    except Exception:
        return data.xpos[body_id].copy(), np.zeros(3)


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    if quat.shape != (4,):
        return np.eye(3)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.eye(3)
    quat = quat / norm
    out = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(out, quat)
    return out.reshape(3, 3)


def aabb_bounds(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=float) / 2.0
    center = np.asarray(center, dtype=float)
    return center - half, center + half


def compute_box_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    offsets = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    return center + offsets * half


def add_box_to_ax(ax, center: np.ndarray, size: np.ndarray, color: str, label: str, alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    corners = compute_box_corners(center, size)
    faces = [
        [corners[idx] for idx in face]
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (4, 7, 3, 0),
        )
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidths=0.8)
    ax.add_collection3d(poly)
    ax.text(center[0], center[1], center[2], label, color=color, fontsize=7)


def save_relation_plot(
    output_path: Path,
    container_rec: dict[str, Any],
    object_rec: dict[str, Any],
    relation: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    add_box_to_ax(
        ax,
        np.asarray(container_rec["aabb_center"], dtype=float),
        np.asarray(container_rec["aabb_size"], dtype=float),
        "tab:blue",
        f"container:{container_rec['name']}",
        0.15,
    )
    add_box_to_ax(
        ax,
        np.asarray(object_rec["aabb_center"], dtype=float),
        np.asarray(object_rec["aabb_size"], dtype=float),
        "tab:red",
        f"object:{object_rec['name']}",
        0.45,
    )

    c_center = np.asarray(container_rec["aabb_center"], dtype=float)
    o_center = np.asarray(object_rec["aabb_center"], dtype=float)
    ax.plot(
        [c_center[0], o_center[0]],
        [c_center[1], o_center[1]],
        [c_center[2], o_center[2]],
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_title(
        f"{relation['label']} | score={relation['score']:.2f}\n"
        f"{container_rec['name']} -> {object_rec['name']}"
    )
    all_pts = np.vstack(
        [
            compute_box_corners(container_rec["aabb_center"], container_rec["aabb_size"]),
            compute_box_corners(object_rec["aabb_center"], object_rec["aabb_size"]),
        ]
    )
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    spans = np.maximum(maxs - mins, 0.1)
    max_span = float(np.max(spans)) * 0.6
    mid = (mins + maxs) / 2.0
    ax.set_xlim(mid[0] - max_span, mid[0] + max_span)
    ax.set_ylim(mid[1] - max_span, mid[1] + max_span)
    ax.set_zlim(max(0.0, mid[2] - max_span), mid[2] + max_span)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_")


def save_rgb_image(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_segmentation_preview(path: Path, seg_frame: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    seg = np.asarray(seg_frame[..., :2], dtype=np.int32)
    geom_ids = seg[..., 0]
    obj_types = seg[..., 1]
    preview = np.zeros((*geom_ids.shape, 3), dtype=np.uint8)
    preview[..., 0] = (geom_ids * 37) % 255
    preview[..., 1] = (geom_ids * 67) % 255
    preview[..., 2] = (obj_types * 97) % 255
    Image.fromarray(preview).save(path)


def save_object_mask_preview(path: Path, mask: np.ndarray | None) -> None:
    from PIL import Image

    if mask is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def scene_category(name: str, category: str | None) -> str:
    lower_name = name.lower()
    lower_cat = (category or "").lower()
    if "door" in lower_name or "door" in lower_cat or "gate" in lower_name:
        return "portal"
    if token_match(lower_name, CONTAINER_TOKENS) or token_match(lower_cat, CONTAINER_TOKENS):
        return "container"
    return "other"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def collect_scene_records(ctx: LoadedContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env = ctx.env
    om = env.object_managers[env.current_batch_index]
    objects_meta = env.current_scene_metadata.get("objects", {})
    records: list[dict[str, Any]] = []
    containers: list[dict[str, Any]] = []

    for object_name, meta in objects_meta.items():
        try:
            body_id = om.get_object_body_id(object_name)
        except Exception:
            continue

        center, size = safe_body_aabb(env.current_model, env.current_data, body_id)
        category = meta.get("category") or om.get_annotation_category(object_name)
        support = om.get_support_below(object_name, RECEPTACLE_TYPES_THOR)
        room = om.infer_room_name(object_name, RECEPTACLE_TYPES_THOR)
        parent_chain = om.get_parent_chain_names(object_name)
        is_articulable = om.is_object_articulable(object_name)
        is_structural = om.is_structural(object_name)
        has_free_joint = om.has_free_joint(object_name)

        rec = {
            "name": object_name,
            "body_id": int(body_id),
            "category": category,
            "position": env.current_data.xpos[body_id].copy(),
            "quat": env.current_data.xquat[body_id].copy(),
            "aabb_center": center.copy(),
            "aabb_size": size.copy(),
            "parent": meta.get("parent"),
            "parent_chain": parent_chain,
            "support_below": support,
            "room": room,
            "is_structural": is_structural,
            "is_articulable": is_articulable,
            "has_free_joint": has_free_joint,
            "interaction_group": scene_category(object_name, category),
            "is_receptacle": om.has_receptacle_site(object_name),
            "asset_id": meta.get("asset_id"),
        }
        records.append(rec)

        if rec["interaction_group"] == "container" and is_articulable:
            art_obj = om.get_object_by_name(object_name)
            if not isinstance(art_obj, MlSpacesArticulationObject):
                continue
            joints = []
            for joint_index, joint_name in enumerate(art_obj.joint_names):
                joint_range = [float(v) for v in art_obj.get_joint_range(joint_index)]
                joints.append(
                    {
                        "joint_index": joint_index,
                        "joint_name": joint_name,
                        "joint_type": str(art_obj.get_joint_type(joint_index)).split(".")[-1],
                        "joint_range": joint_range,
                        "current_value": float(art_obj.get_joint_position(joint_index)),
                        "closed_value": float(min(joint_range)),
                        "open_value": float(max(joint_range)),
                    }
                )
            container_rec = copy.deepcopy(rec)
            container_rec["joints"] = joints
            containers.append(container_rec)

    return records, containers


def collect_door_records(ctx: LoadedContext) -> list[dict[str, Any]]:
    env = ctx.env
    om = env.object_managers[env.current_batch_index]
    door_records: list[dict[str, Any]] = []
    for door_name in om.find_door_names():
        try:
            door = Door(door_name, env.current_data)
            hinge_idx = door.get_hinge_joint_index()
            joint_range = [float(v) for v in door.get_joint_range(hinge_idx)]
            center, size = safe_body_aabb(env.current_model, env.current_data, door.body_id)
            door_records.append(
                {
                    "name": door_name,
                    "interaction_group": "portal",
                    "body_id": int(door.body_id),
                    "category": "Door",
                    "position": env.current_data.xpos[door.body_id].copy(),
                    "quat": env.current_data.xquat[door.body_id].copy(),
                    "aabb_center": center.copy(),
                    "aabb_size": size.copy(),
                    "hinge_joint_index": hinge_idx,
                    "hinge_joint_name": door.joint_names[hinge_idx],
                    "hinge_joint_range": joint_range,
                    "closed_value": float(min(joint_range)),
                    "open_value": float(max(joint_range)),
                }
            )
        except Exception:
            continue
    return door_records


def set_articulation_state_by_record(
    env,
    rec: dict[str, Any],
    joint_index: int,
    joint_value: float,
) -> None:
    if rec.get("interaction_group") == "portal":
        door = Door(rec["name"], env.current_data)
        door.set_joint_position(joint_index, float(joint_value))
    else:
        om = env.object_managers[env.current_batch_index]
        obj = om.get_object_by_name(rec["name"])
        if not isinstance(obj, MlSpacesArticulationObject):
            raise ValueError(f"{rec['name']} is not an articulable container.")
        obj.set_joint_position(joint_index, float(joint_value))
    mujoco.mj_forward(env.current_model, env.current_data)


def set_container_joint_fraction(
    env,
    container_name: str,
    joint_index: int,
    open_fraction: float,
) -> dict[str, Any]:
    om = env.object_managers[env.current_batch_index]
    container_obj = om.get_object_by_name(container_name)
    if not isinstance(container_obj, MlSpacesArticulationObject):
        raise ValueError(f"{container_name} is not an articulable container.")
    joint_range = [float(v) for v in container_obj.get_joint_range(joint_index)]
    lo, hi = min(joint_range), max(joint_range)
    target = lo + float(open_fraction) * (hi - lo)
    container_obj.set_joint_position(joint_index, target)
    mujoco.mj_forward(env.current_model, env.current_data)
    return {
        "container_name": container_name,
        "joint_index": joint_index,
        "joint_range": joint_range,
        "target_value": float(target),
        "open_fraction": float(open_fraction),
    }


def open_container_joint(env, container_name: str, joint_index: int) -> dict[str, Any]:
    return set_container_joint_fraction(env, container_name, joint_index, 1.0)


def close_container_joint(env, container_name: str, joint_index: int) -> dict[str, Any]:
    return set_container_joint_fraction(env, container_name, joint_index, 0.0)


def set_door_open_fraction(env, door_name: str, open_fraction: float) -> dict[str, Any]:
    door = Door(door_name, env.current_data)
    hinge_idx = door.get_hinge_joint_index()
    joint_range = [float(v) for v in door.get_joint_range(hinge_idx)]
    lo, hi = min(joint_range), max(joint_range)
    target = lo + float(open_fraction) * (hi - lo)
    door.set_joint_position(hinge_idx, target)
    mujoco.mj_forward(env.current_model, env.current_data)
    return {
        "door_name": door_name,
        "hinge_joint_index": hinge_idx,
        "hinge_joint_name": door.joint_names[hinge_idx],
        "hinge_joint_range": joint_range,
        "target_value": float(target),
        "open_fraction": float(open_fraction),
    }


def open_door_space(env, door_name: str) -> dict[str, Any]:
    return set_door_open_fraction(env, door_name, 1.0)


def close_door_space(env, door_name: str) -> dict[str, Any]:
    return set_door_open_fraction(env, door_name, 0.0)


def is_target_like(rec: dict[str, Any]) -> bool:
    if rec["is_structural"] or rec["interaction_group"] in {"container", "portal"}:
        return False
    if rec["has_free_joint"]:
        return True
    category = str(rec["category"] or "").lower()
    name = rec["name"].lower()
    return token_match(category, PORTABLE_PREFERRED_TOKENS) or token_match(name, PORTABLE_PREFERRED_TOKENS)


def compute_relation(container_rec: dict[str, Any], object_rec: dict[str, Any]) -> dict[str, Any]:
    c_name = container_rec["name"]
    c_min, c_max = aabb_bounds(container_rec["aabb_center"], container_rec["aabb_size"])
    o_min, o_max = aabb_bounds(object_rec["aabb_center"], object_rec["aabb_size"])
    inside = bool(np.all(o_min >= c_min) and np.all(o_max <= c_max))
    overlap = np.maximum(0.0, np.minimum(o_max, c_max) - np.maximum(o_min, c_min))
    overlap_vol = float(np.prod(overlap))
    o_size = np.asarray(object_rec["aabb_size"], dtype=float)
    o_vol = max(float(np.prod(np.maximum(o_size, 1e-4))), 1e-6)
    overlap_ratio = overlap_vol / o_vol
    support = object_rec.get("support_below")
    parent = object_rec.get("parent")
    parent_chain = object_rec.get("parent_chain") or []
    parent_chain_hit = c_name in parent_chain
    support_hit = bool(support and (support == c_name or support.startswith(c_name)))
    parent_hit = bool(parent and (parent == c_name or parent.startswith(c_name)))
    distance = float(
        np.linalg.norm(np.asarray(object_rec["aabb_center"], dtype=float) - np.asarray(container_rec["aabb_center"], dtype=float))
    )
    score = 0.0
    if inside:
        score += 3.0
    score += 2.0 * overlap_ratio
    if parent_chain_hit:
        score += 1.5
    if parent_hit:
        score += 1.0
    if support_hit:
        score += 1.0
    score += max(0.0, 1.0 - distance / 2.0)

    # Keep "inside" strict: require strong geometric evidence instead of
    # structural metadata such as parent/support, which often marks objects
    # resting on top of furniture rather than inside a cavity.
    if inside or overlap_ratio > 0.8:
        label = "inside"
    elif parent_hit or support_hit:
        label = "attached_to_container"
    elif parent_chain_hit or overlap_ratio > 0.25:
        label = "likely_inside"
    elif distance < 0.75:
        label = "near_container"
    else:
        label = "unrelated"

    return {
        "container_name": c_name,
        "object_name": object_rec["name"],
        "inside_aabb": inside,
        "overlap_ratio": overlap_ratio,
        "parent_chain_hit": parent_chain_hit,
        "parent_hit": parent_hit,
        "support_hit": support_hit,
        "distance_to_container_center": distance,
        "label": label,
        "score": score,
    }


def yaw_to_face(source_xy: np.ndarray, target_xy: np.ndarray) -> float:
    delta = np.asarray(target_xy, dtype=float) - np.asarray(source_xy, dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def nearest_free_point(free_points: np.ndarray, target_xy: np.ndarray) -> np.ndarray | None:
    if free_points.size == 0:
        return None
    dists = np.linalg.norm(free_points[:, :2] - target_xy[None, :2], axis=1)
    return free_points[int(np.argmin(dists))]


def container_front_axis(container_rec: dict[str, Any]) -> np.ndarray:
    size = np.asarray(container_rec["aabb_size"], dtype=float)
    quat = np.asarray(container_rec.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
    rot = quat_to_rotmat(quat)
    horizontal_extents = [
        (abs(rot[0, 0]) * size[0] + abs(rot[1, 0]) * size[1], rot[:2, 0]),
        (abs(rot[0, 1]) * size[0] + abs(rot[1, 1]) * size[1], rot[:2, 1]),
    ]
    horizontal_extents.sort(key=lambda item: item[0])
    front_axis_xy = horizontal_extents[0][1]
    norm = np.linalg.norm(front_axis_xy)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return front_axis_xy / norm


def make_robot_pose_from_xy(robot_view, xy: np.ndarray, yaw: float) -> np.ndarray:
    pose = robot_view.base.pose.copy()
    pose[:3, 3] = np.array([xy[0], xy[1], 0.0], dtype=float)
    pose[:3, :3] = quat_to_rotmat(yaw_to_quat(yaw))
    return pose


def choose_collision_free_pose(
    env,
    robot_view,
    center_xy: np.ndarray,
    free_points: np.ndarray,
    candidate_xy: np.ndarray,
    penalty: float = 0.0,
) -> tuple[np.ndarray, float] | None:
    free_pt = nearest_free_point(free_points, candidate_xy)
    if free_pt is None:
        return None
    yaw = yaw_to_face(free_pt[:2], center_xy)
    pose = make_robot_pose_from_xy(robot_view, free_pt[:2], yaw)
    if env.check_if_robot_collision_at_base_pose(robot_view, pose):
        return None
    score = float(np.linalg.norm(free_pt[:2] - candidate_xy[:2]) + penalty)
    return pose, score


def choose_pose_valid_for_joint_states(
    ctx: LoadedContext,
    articulation_rec: dict[str, Any],
    joint: dict[str, Any],
    closed_val: float,
    open_val: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center, _size = joint_target_geometry(env, articulation_rec, joint)
    front_axis_xy = container_front_axis(articulation_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)

    desired_dist = 0.8
    primary_front = center[:2] + front_axis_xy * desired_dist
    fallback_front = center[:2] - front_axis_xy * desired_dist

    # Prefer positions close to the drawer/fridge center, while still trying
    # the natural front-facing slot first. Retreat distance is penalized.
    candidate_specs: list[tuple[np.ndarray, float, str]] = []
    for base_xy, base_penalty, side_name in (
        (primary_front, 0.0, "front"),
        (fallback_front, 1.5, "back"),
    ):
        candidate_specs.extend(
            [
                (base_xy, base_penalty + 0.00, side_name),
                (base_xy + front_axis_xy * 0.15, base_penalty + 0.15, side_name),
                (base_xy + front_axis_xy * 0.30, base_penalty + 0.30, side_name),
                (base_xy + lateral_xy * 0.20, base_penalty + 0.20, f"{side_name}_left"),
                (base_xy - lateral_xy * 0.20, base_penalty + 0.20, f"{side_name}_right"),
                (
                    base_xy + front_axis_xy * 0.15 + lateral_xy * 0.20,
                    base_penalty + 0.35,
                    f"{side_name}_left",
                ),
                (
                    base_xy + front_axis_xy * 0.15 - lateral_xy * 0.20,
                    base_penalty + 0.35,
                    f"{side_name}_right",
                ),
            ]
        )

    qpos_before = env.current_data.qpos.copy()
    robot_pose_before = robot_view.base.pose.copy()
    best_pose = None
    best_score = float("inf")
    best_meta: dict[str, Any] = {}

    try:
        for candidate_xy, penalty, label in candidate_specs:
            free_pt = nearest_free_point(free_points, candidate_xy)
            if free_pt is None:
                continue
            yaw = yaw_to_face(free_pt[:2], center[:2])
            pose = make_robot_pose_from_xy(robot_view, free_pt[:2], yaw)

            # Must be collision-free in the closed state.
            env.current_data.qpos[:] = qpos_before
            mujoco.mj_forward(env.current_model, env.current_data)
            set_articulation_state_by_record(
                env, articulation_rec, joint["joint_index"], closed_val
            )
            if env.check_if_robot_collision_at_base_pose(robot_view, pose):
                continue

            # Must also be collision-free in the open state.
            env.current_data.qpos[:] = qpos_before
            mujoco.mj_forward(env.current_model, env.current_data)
            set_articulation_state_by_record(
                env, articulation_rec, joint["joint_index"], open_val
            )
            if env.check_if_robot_collision_at_base_pose(robot_view, pose):
                continue

            center_dist = float(np.linalg.norm(free_pt[:2] - center[:2]))
            score = center_dist + penalty
            if score < best_score:
                best_score = score
                best_pose = pose.copy()
                best_meta = {
                    "candidate_label": label,
                    "candidate_target_xy": candidate_xy.tolist(),
                    "free_point_xy": free_pt[:2].tolist(),
                    "center_distance": center_dist,
                    "score": score,
                }
    finally:
        env.current_data.qpos[:] = qpos_before
        robot_view.base.pose = robot_pose_before
        mujoco.mj_forward(env.current_model, env.current_data)

    return best_pose, best_meta


def joint_target_geometry(env, container_rec: dict[str, Any], joint: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model = env.current_model
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint["joint_name"])
    if joint_id < 0:
        return np.asarray(container_rec["aabb_center"], dtype=float), np.asarray(
            container_rec["aabb_size"], dtype=float
        )
    body_id = int(model.jnt_bodyid[joint_id])
    center, size = safe_body_aabb(model, env.current_data, body_id)
    return np.asarray(center, dtype=float), np.asarray(size, dtype=float)


def place_robot_in_front_of_container(ctx: LoadedContext, container_rec: dict[str, Any]) -> bool:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center = np.asarray(container_rec["aabb_center"], dtype=float)
    front_axis_xy = container_front_axis(container_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    desired_dist = 0.8
    lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)
    candidate_specs: list[tuple[np.ndarray, float]] = []
    for sign in (1.0, -1.0):
        front_xy = center[:2] + sign * front_axis_xy * desired_dist
        candidate_specs.append((front_xy, 0.0))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.25, 0.25))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.50, 0.50))
        candidate_specs.append((front_xy + lateral_xy * 0.30, 0.30))
        candidate_specs.append((front_xy - lateral_xy * 0.30, 0.30))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.25 + lateral_xy * 0.30, 0.55))
        candidate_specs.append((front_xy + sign * front_axis_xy * 0.25 - lateral_xy * 0.30, 0.55))

    best_pose = None
    best_rank = float("inf")
    for candidate_xy, penalty in candidate_specs:
        result = choose_collision_free_pose(
            env, robot_view, center[:2], free_points, candidate_xy, penalty=penalty
        )
        if result is None:
            continue
        pose, score = result
        if score < best_rank:
            best_rank = score
            best_pose = pose.copy()

    if best_pose is None:
        return False

    robot_view.base.pose = best_pose
    mujoco.mj_forward(env.current_model, env.current_data)
    apply_default_arm_pose(env)
    apply_default_head_pose(env, ctx.initial_head_qpos)
    env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
    return True


def place_robot_for_container_joint(
    ctx: LoadedContext, container_rec: dict[str, Any], joint: dict[str, Any]
) -> tuple[bool, np.ndarray | None]:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center, _size = joint_target_geometry(env, container_rec, joint)
    front_axis_xy = container_front_axis(container_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    desired_dist = 0.8
    lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)

    primary_front = center[:2] + front_axis_xy * desired_dist
    fallback_front = center[:2] - front_axis_xy * desired_dist
    candidate_specs: list[tuple[np.ndarray, float]] = []
    for base_xy, base_penalty in ((primary_front, 0.0), (fallback_front, 1.5)):
        candidate_specs.extend(
            [
                (base_xy, base_penalty),
                (base_xy + front_axis_xy * 0.25, base_penalty + 0.25),
                (base_xy + front_axis_xy * 0.50, base_penalty + 0.50),
                (base_xy + lateral_xy * 0.30, base_penalty + 0.30),
                (base_xy - lateral_xy * 0.30, base_penalty + 0.30),
                (base_xy + front_axis_xy * 0.25 + lateral_xy * 0.30, base_penalty + 0.55),
                (base_xy + front_axis_xy * 0.25 - lateral_xy * 0.30, base_penalty + 0.55),
                (base_xy + front_axis_xy * 0.50 + lateral_xy * 0.30, base_penalty + 0.80),
                (base_xy + front_axis_xy * 0.50 - lateral_xy * 0.30, base_penalty + 0.80),
            ]
        )

    best_pose = None
    best_rank = float("inf")
    for candidate_xy, penalty in candidate_specs:
        result = choose_collision_free_pose(
            env, robot_view, center[:2], free_points, candidate_xy, penalty=penalty
        )
        if result is None:
            continue
        pose, score = result
        if score < best_rank:
            best_pose = pose.copy()
            best_rank = score

    if best_pose is None:
        return False, None

    robot_view.base.pose = best_pose
    mujoco.mj_forward(env.current_model, env.current_data)
    apply_default_arm_pose(env)
    apply_default_head_pose(env, ctx.initial_head_qpos)
    env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
    return True, best_pose.copy()


def save_debug_head_image(env, path: Path) -> None:
    rgb = env.render_rgb_frame("head_camera")
    save_rgb_image(path, rgb)


def command_debug_container_view(args: argparse.Namespace) -> int:
    house_dir = args.output_dir / f"debug_container_view_{args.house_ind}"
    house_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        records, containers = collect_scene_records(ctx)
        container_rec = next(
            (rec for rec in containers if rec["name"] == args.container_name),
            None,
        )
        if container_rec is None:
            raise RuntimeError(f"Container not found: {args.container_name}")

        env = ctx.env
        om = env.object_managers[env.current_batch_index]
        container_obj = om.get_object_by_name(container_rec["name"])
        if not isinstance(container_obj, MlSpacesArticulationObject):
            raise RuntimeError("Selected container is not articulable")

        thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
        free_points = thormap.get_free_points()
        center = np.asarray(container_rec["aabb_center"], dtype=float)
        front_axis_xy = container_front_axis(container_rec)
        lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)
        desired_dist = 0.8
        base_front = center[:2] + front_axis_xy * desired_dist
        viewpoint_targets = {
            "front": base_front,
            "left": base_front + lateral_xy * 0.35,
            "right": base_front - lateral_xy * 0.35,
        }

        qpos_before = env.current_data.qpos.copy()
        robot_before = env.current_robot.robot_view.base.pose.copy()
        head_before = get_head_joint_position(env)
        images = {}
        pose_rows = []
        head_pose_rows = []
        try:
            container_obj.set_joint_position(args.joint_index, float(args.joint_value))
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            apply_default_head_pose(env, ctx.initial_head_qpos)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

            for name, candidate_xy in viewpoint_targets.items():
                pose = None
                # try direct, back, and lateral retreats for this requested viewpoint
                candidates = [
                    (candidate_xy, 0.0),
                    (candidate_xy + front_axis_xy * 0.25, 0.25),
                    (candidate_xy + front_axis_xy * 0.50, 0.50),
                    (candidate_xy + lateral_xy * 0.20, 0.20),
                    (candidate_xy - lateral_xy * 0.20, 0.20),
                    (candidate_xy + front_axis_xy * 0.25 + lateral_xy * 0.20, 0.45),
                    (candidate_xy + front_axis_xy * 0.25 - lateral_xy * 0.20, 0.45),
                ]
                best = None
                best_score = float("inf")
                for cand_xy, penalty in candidates:
                    result = choose_collision_free_pose(
                        env, env.current_robot.robot_view, center[:2], free_points, cand_xy, penalty
                    )
                    if result is None:
                        continue
                    cur_pose, cur_score = result
                    if cur_score < best_score:
                        best = cur_pose
                        best_score = cur_score
                if best is None:
                    pose_rows.append({"view": name, "success": False})
                    continue
                env.current_robot.robot_view.base.pose = best
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_arm_pose(env)
                apply_default_head_pose(env, ctx.initial_head_qpos)
                env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                img_path = house_dir / f"{name}_head_rgb.png"
                save_debug_head_image(env, img_path)
                images[name] = str(img_path)
                pose_rows.append(
                    {
                        "view": name,
                        "success": True,
                        "robot_pose": best.tolist(),
                        "image_path": str(img_path),
                    }
                )
                if name == "front":
                    default_head = (
                        None if ctx.initial_head_qpos is None else ctx.initial_head_qpos.copy()
                    )
                    if default_head is None:
                        default_head = get_head_joint_position(env)
                    if default_head is not None:
                        head_states = [
                            ("front_head_default", default_head.copy()),
                            (
                                "front_head_look_down",
                                np.asarray(
                                    [default_head[0], default_head[1] + args.head_tilt_delta],
                                    dtype=np.float32,
                                ),
                            ),
                            ("front_head_restored", default_head.copy()),
                        ]
                    else:
                        head_states = []
                    for tag, target_head_qpos in head_states:
                        head_qpos = set_head_joint_position(env, target_head_qpos)
                        if head_qpos is None:
                            continue
                        env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                        head_img_path = house_dir / f"{tag}_rgb.png"
                        save_debug_head_image(env, head_img_path)
                        images[tag] = str(head_img_path)
                        head_pose_rows.append(
                            {
                                "tag": tag,
                                "head_qpos": np.asarray(head_qpos, dtype=float).tolist(),
                                "image_path": str(head_img_path),
                            }
                        )
        finally:
            env.current_data.qpos[:] = qpos_before
            env.current_robot.robot_view.base.pose = robot_before
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            if head_before is not None:
                set_head_joint_position(env, head_before)
            else:
                apply_default_head_pose(env, ctx.initial_head_qpos)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

        payload = {
            "house_ind": args.house_ind,
            "container_name": args.container_name,
            "joint_index": args.joint_index,
            "joint_value": args.joint_value,
            "images": images,
            "poses": pose_rows,
            "head_pose_images": head_pose_rows,
            "head_tilt_delta": args.head_tilt_delta,
        }
        write_json(house_dir / "debug_view_result.json", payload)
        return 0
    finally:
        close_context(ctx)


def command_debug_door_view(args: argparse.Namespace) -> int:
    out_dir = args.output_dir / f"debug_door_view_{args.house_ind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        door_records = collect_door_records(ctx)
        if args.door_name:
            door_rec = next((rec for rec in door_records if rec["name"] == args.door_name), None)
            if door_rec is None:
                raise RuntimeError(f"Door not found: {args.door_name}")
        else:
            if not door_records:
                raise RuntimeError(f"No interactive doors found in house {args.house_ind}")
            door_rec = door_records[0]

        env = ctx.env
        qpos_before = env.current_data.qpos.copy()
        robot_pose_before = env.current_robot.robot_view.base.pose.copy()
        images = {}
        try:
            close_meta = close_door_space(env, door_rec["name"])
            shared_pose, shared_meta = choose_pose_valid_for_joint_states(
                ctx,
                {
                    "name": door_rec["name"],
                    "interaction_group": "portal",
                    "aabb_center": door_rec["aabb_center"],
                    "aabb_size": door_rec["aabb_size"],
                    "quat": door_rec["quat"],
                },
                {
                    "joint_index": door_rec["hinge_joint_index"],
                    "joint_name": door_rec["hinge_joint_name"],
                },
                close_meta["target_value"],
                door_rec["open_value"],
            )
            if shared_pose is None:
                raise RuntimeError("Could not find a shared robot pose for door closed/open states.")

            for tag, target_value in (
                ("closed", close_meta["target_value"]),
                ("open", door_rec["open_value"]),
            ):
                env.current_robot.robot_view.base.pose = shared_pose
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_arm_pose(env)
                set_articulation_state_by_record(
                    env, door_rec, door_rec["hinge_joint_index"], target_value
                )
                env.current_robot.robot_view.base.pose = shared_pose
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_arm_pose(env)
                env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                img_path = out_dir / f"{tag}_head_rgb.png"
                seg_path = out_dir / f"{tag}_head_seg.png"
                save_debug_head_image(env, img_path)
                save_segmentation_preview(seg_path, env.render_segmentation_frame("head_camera"))
                images[tag] = {"rgb": str(img_path), "seg": str(seg_path)}
        finally:
            env.current_data.qpos[:] = qpos_before
            env.current_robot.robot_view.base.pose = robot_pose_before
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

        payload = {
            "house_ind": args.house_ind,
            "door_name": door_rec["name"],
            "hinge_joint_index": door_rec["hinge_joint_index"],
            "hinge_joint_range": door_rec["hinge_joint_range"],
            "shared_robot_pose": shared_pose.tolist(),
            "shared_pose_meta": shared_meta,
            "images": images,
        }
        write_json(out_dir / "debug_door_result.json", payload)
        return 0
    finally:
        close_context(ctx)


def measure_container_visibility(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    object_names: list[str],
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    env = ctx.env
    om = env.object_managers[env.current_batch_index]
    container_obj = om.get_object_by_name(container_rec["name"])
    if not isinstance(container_obj, MlSpacesArticulationObject):
        return []

    qpos_before = env.current_data.qpos.copy()
    robot_pose_before = env.current_robot.robot_view.base.pose.copy()
    results = []
    for joint in container_rec["joints"]:
        closed_val = joint["closed_value"]
        open_val = joint["open_value"]
        shared_pose, shared_pose_meta = choose_pose_valid_for_joint_states(
            ctx, container_rec, joint, closed_val, open_val
        )
        closed_robot_pose_actual = None
        open_robot_pose_actual = None
        try:
            if shared_pose is None:
                continue
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
            container_obj.set_joint_position(joint["joint_index"], closed_val)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            closed_robot_pose_actual = env.current_robot.robot_view.base.pose.copy()
            closed_vis = env.check_visibility("head_camera", *object_names)
            closed_rgb = env.render_rgb_frame("head_camera")
            closed_seg = env.render_segmentation_frame("head_camera")
            container_obj.set_joint_position(joint["joint_index"], open_val)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.current_robot.robot_view.base.pose = shared_pose
            mujoco.mj_forward(env.current_model, env.current_data)
            open_robot_pose_actual = env.current_robot.robot_view.base.pose.copy()
            open_vis = env.check_visibility("head_camera", *object_names)
            open_rgb = env.render_rgb_frame("head_camera")
            open_seg = env.render_segmentation_frame("head_camera")
        finally:
            env.current_data.qpos[:] = qpos_before
            mujoco.mj_forward(env.current_model, env.current_data)
            env.current_robot.robot_view.base.pose = robot_pose_before
            mujoco.mj_forward(env.current_model, env.current_data)
            apply_default_arm_pose(env)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)

        if isinstance(closed_vis, float):
            closed_map = {object_names[0]: float(closed_vis)}
        else:
            closed_map = {k: float(v) for k, v in closed_vis.items()}
        if isinstance(open_vis, float):
            open_map = {object_names[0]: float(open_vis)}
        else:
            open_map = {k: float(v) for k, v in open_vis.items()}

        for object_name in object_names:
            cval = closed_map.get(object_name, 0.0)
            oval = open_map.get(object_name, 0.0)
            image_paths = {}
            if output_dir is not None:
                img_dir = output_dir / "visibility_images"
                base = (
                    f"{sanitize_name(container_rec['name'])}__joint{joint['joint_index']}__"
                    f"{sanitize_name(object_name)}"
                )
                closed_rgb_path = img_dir / f"{base}__closed_rgb.png"
                open_rgb_path = img_dir / f"{base}__open_rgb.png"
                closed_seg_path = img_dir / f"{base}__closed_seg.png"
                open_seg_path = img_dir / f"{base}__open_seg.png"
                closed_mask_path = img_dir / f"{base}__closed_mask.png"
                open_mask_path = img_dir / f"{base}__open_mask.png"
                save_rgb_image(closed_rgb_path, closed_rgb)
                save_rgb_image(open_rgb_path, open_rgb)
                save_segmentation_preview(closed_seg_path, closed_seg)
                save_segmentation_preview(open_seg_path, open_seg)
                try:
                    body_id = env.current_model.body(object_name).id
                    closed_mask = get_geom_seg_mask(env.current_model, closed_seg[..., :2], body_id)
                    open_mask = get_geom_seg_mask(env.current_model, open_seg[..., :2], body_id)
                except KeyError:
                    closed_mask = None
                    open_mask = None
                save_object_mask_preview(closed_mask_path, closed_mask)
                save_object_mask_preview(open_mask_path, open_mask)
                image_paths = {
                    "closed_rgb": str(closed_rgb_path),
                    "open_rgb": str(open_rgb_path),
                    "closed_seg": str(closed_seg_path),
                    "open_seg": str(open_seg_path),
                    "closed_mask": str(closed_mask_path),
                    "open_mask": str(open_mask_path),
                }
            results.append(
                {
                    "container_name": container_rec["name"],
                    "joint_index": joint["joint_index"],
                    "joint_name": joint["joint_name"],
                    "object_name": object_name,
                    "closed_visibility": cval,
                    "open_visibility": oval,
                    "delta_visibility": oval - cval,
                    "became_visible": cval <= 1e-4 and oval > 1e-4,
                    "shared_pose_ok": shared_pose is not None,
                    "shared_robot_pose": shared_pose.tolist() if shared_pose is not None else None,
                    "closed_robot_pose_actual": closed_robot_pose_actual.tolist()
                    if shared_pose is not None
                    else None,
                    "open_robot_pose_actual": open_robot_pose_actual.tolist()
                    if shared_pose is not None
                    else None,
                    "shared_pose_meta": shared_pose_meta,
                    "image_paths": image_paths,
                }
            )
    return results


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def build_scan_output_dir(args: argparse.Namespace) -> Path:
    house_tag = f"{min(args.house_inds)}_{max(args.house_inds)}" if args.house_inds else "empty"
    return args.output_dir / f"scan_{args.scene_dataset}_{args.data_split}_{house_tag}"


def command_scan_houses(args: argparse.Namespace) -> int:
    out_dir = build_scan_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_summary = []

    for house_ind in args.house_inds:
        house_dir = out_dir / f"house_{house_ind}"
        house_dir.mkdir(parents=True, exist_ok=True)
        ctx = load_scene_context(args, house_ind)
        try:
            records, containers = collect_scene_records(ctx)
            target_records = [rec for rec in records if is_target_like(rec)]
            write_json(house_dir / "all_objects.json", records)
            write_json(house_dir / "containers.json", containers)
            write_json(house_dir / "target_objects.json", target_records)

            relations = []
            visibility_results = []
            top_candidates = []
            for container_rec in containers:
                per_container_rel = [
                    compute_relation(container_rec, object_rec) for object_rec in target_records
                ]
                per_container_rel.sort(key=lambda item: (-item["score"], item["object_name"]))
                relations.extend(per_container_rel)

                related_names = [
                    rel["object_name"]
                    for rel in per_container_rel
                    if rel["label"] in {"inside", "likely_inside"} and rel["score"] >= 1.5
                ][: args.max_visibility_objects_per_container]
                if related_names:
                    vis_rows = measure_container_visibility(
                        ctx, container_rec, related_names, output_dir=house_dir
                    )
                    visibility_results.extend(vis_rows)

                for rel in per_container_rel[: args.max_plots_per_container]:
                    if rel["label"] == "unrelated":
                        continue
                    object_rec = next(rec for rec in target_records if rec["name"] == rel["object_name"])
                    plot_name = (
                        f"relation__{container_rec['name'].replace('/', '_')}__"
                        f"{object_rec['name'].replace('/', '_')}.png"
                    )
                    save_relation_plot(house_dir / plot_name, container_rec, object_rec, rel)

            vis_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in visibility_results:
                vis_lookup.setdefault((row["container_name"], row["object_name"]), []).append(row)

            for rel in relations:
                rows = vis_lookup.get((rel["container_name"], rel["object_name"]), [])
                best_delta = max((row["delta_visibility"] for row in rows), default=0.0)
                became_visible = any(row["became_visible"] for row in rows)
                candidate = {
                    **rel,
                    "best_delta_visibility": best_delta,
                    "became_visible": became_visible,
                    "interaction_value_score": rel["score"] + 5.0 * max(best_delta, 0.0),
                }
                if rel["label"] != "unrelated":
                    top_candidates.append(candidate)
            top_candidates.sort(
                key=lambda item: (
                    -item["interaction_value_score"],
                    -item["best_delta_visibility"],
                    -item["score"],
                    item["container_name"],
                    item["object_name"],
                )
            )

            write_json(house_dir / "relations.json", relations)
            write_json(house_dir / "visibility.json", visibility_results)
            write_json(house_dir / "top_candidates.json", top_candidates[:50])

            overall_summary.append(
                {
                    "house_ind": house_ind,
                    "scene_path": str(ctx.env.current_model_path),
                    "num_objects": len(records),
                    "num_target_objects": len(target_records),
                    "num_containers": len(containers),
                    "num_relations": len(relations),
                    "num_visibility_rows": len(visibility_results),
                    "top_candidates": top_candidates[:10],
                }
            )
        finally:
            close_context(ctx)

    write_json(out_dir / "scan_report_all_houses.json", overall_summary)
    log.info("Wrote scan report to %s", out_dir)
    return 0


def choose_existing_asset(ctx: LoadedContext) -> tuple[str, str, str]:
    env = ctx.env
    objects_meta = env.current_scene_metadata.get("objects", {})
    for object_name, meta in objects_meta.items():
        category = str(meta.get("category", "")).lower()
        asset_id = meta.get("asset_id")
        if not asset_id:
            continue
        if token_match(category, PORTABLE_PREFERRED_TOKENS):
            object_xml = install_uid(asset_id)
            try:
                object_ref = str(object_xml.relative_to(Path("/home/user/ldl/molmospaces/assets")))
            except ValueError:
                object_ref = str(object_xml)
            return object_ref, asset_id, category
    raise RuntimeError("Could not auto-select an existing portable asset from the scene.")


def build_added_object_pose(container_rec: dict[str, Any], current_name: str) -> list[float]:
    center = np.asarray(container_rec["aabb_center"], dtype=float).copy()
    size = np.asarray(container_rec["aabb_size"], dtype=float)
    center[2] = max(center[2], 0.05)
    if "drawer" in container_rec["name"].lower() or "drawer" in str(container_rec["category"]).lower():
        center[2] = center[2] - 0.1 * size[2]
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return np.concatenate([center, quat]).tolist()


def build_added_episode_spec(
    base_ctx: LoadedContext,
    args: argparse.Namespace,
    object_name: str,
    object_relpath: str,
    object_pose: list[float],
) -> EpisodeSpec:
    env = base_ctx.env
    robot_init_qpos = {
        group_name: np.asarray(qpos, dtype=float).tolist()
        for group_name, qpos in base_ctx.cfg.robot_config.init_qpos.items()
    }
    camera_specs = [
        {
            "name": "probe_exocentric_camera",
            "type": "exocentric",
            "pos": [0.0, -6.0, 6.0],
            "up": [0.0, 0.0, 1.0],
            "forward": [0.0, 0.7, -0.7],
            "fov": 60.0,
            "record_depth": False,
        }
    ]
    episode = {
        "source": {
            "h5_file": "container_scene_probe",
            "traj_key": "synthetic",
            "episode_length": 0,
            "camera_system_class": type(base_ctx.cfg.camera_config).__name__,
            "source_data_date": None,
            "benchmark_created_date": None,
        },
        "house_index": args.house_ind,
        "scene_dataset": args.scene_dataset,
        "data_split": args.data_split,
        "seed": args.seed,
        "robot": {
            "robot_name": base_ctx.cfg.robot_config.name,
            "init_qpos": robot_init_qpos,
        },
        "img_resolution": list(base_ctx.cfg.camera_config.img_resolution),
        "cameras": camera_specs,
        "scene_modifications": {
            "added_objects": {object_name: object_relpath},
            "object_poses": {object_name: object_pose},
            "removed_objects": [],
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": object_name,
            "pickup_obj_candidates": [object_name],
            "succ_pos_threshold": 1.5,
        },
        "language": {
            "task_description": f"Navigate to the added object {object_name.split('/')[-1]}",
            "referral_expressions": {"object_name": object_name.split("/")[-1]},
            "referral_expressions_priority": {},
        },
    }
    return EpisodeSpec.model_validate(episode)


def build_added_object_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / f"add_object_{args.scene_dataset}_{args.data_split}_{args.house_ind}"


def command_add_object_test(args: argparse.Namespace) -> int:
    out_dir = build_added_object_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ctx = load_scene_context(args, args.house_ind)
    try:
        records, containers = collect_scene_records(base_ctx)
        if not containers:
            raise RuntimeError(f"No containers found in house {args.house_ind}")
        if args.container_name:
            container_rec = next(
                (rec for rec in containers if rec["name"] == args.container_name),
                None,
            )
            if container_rec is None:
                raise RuntimeError(f"Container '{args.container_name}' not found in house {args.house_ind}")
        else:
            container_rec = max(containers, key=lambda rec: len(rec["joints"]))

        if args.object_relpath:
            object_relpath = args.object_relpath
            object_uid = Path(object_relpath).stem
            object_label = object_uid
        else:
            object_relpath, object_uid, object_label = choose_existing_asset(base_ctx)

        object_body_name = args.object_name or f"custom_probe_{Path(object_relpath).stem}"
        object_name = f"custom_object/{object_body_name}"
        object_pose = build_added_object_pose(container_rec, object_name)
        episode_spec = build_added_episode_spec(
            base_ctx,
            args,
            object_name=object_name,
            object_relpath=object_relpath,
            object_pose=object_pose,
        )
        write_json(out_dir / "synthetic_episode.json", episode_spec.model_dump())
    finally:
        close_context(base_ctx)

    import molmo_spaces.tasks.task_sampler as task_sampler_module

    manager = get_resource_manager()
    manager.symlink_lock = False
    manager.cache_lock = False
    task_sampler_module.install_scene_with_objects_and_grasps_from_path = lambda *a, **k: {}

    cfg = build_scene_config(args)
    task_sampler = JsonEvalTaskSampler(cfg, episode_spec)
    try:
        task_sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=args.house_ind)
        original_scene_path = Path(task_sampler._current_house_scene_path(variant=args.variant))
        scene_path = prepare_writable_scene_path(original_scene_path)
        task_sampler.update_scene(scene_path=scene_path, variant=args.variant)
        env = task_sampler.env
        if env.current_scene_metadata is None:
            env._scene_metadata = get_scene_metadata(original_scene_path)
        env.camera_manager.setup_cameras(env, cfg.camera_config)
        om = env.object_managers[env.current_batch_index]
        added_obj = om.get_object_by_name(object_name)
        center, size = safe_body_aabb(env.current_model, env.current_data, added_obj.body_id)
        container_obj = om.get_object_by_name(container_rec["name"])
        if isinstance(container_obj, MlSpacesArticulationObject) and container_rec["joints"]:
            first_joint = container_rec["joints"][0]
            container_obj.set_joint_position(first_joint["joint_index"], first_joint["open_value"])
            mujoco.mj_forward(env.current_model, env.current_data)
            env.camera_manager.setup_cameras(env, cfg.camera_config)
        result = {
            "house_ind": args.house_ind,
            "scene_path": str(env.current_model_path),
            "container_name": container_rec["name"],
            "container_category": container_rec["category"],
            "added_object_name": object_name,
            "asset_relpath": object_relpath,
            "asset_uid": object_uid,
            "asset_label": object_label,
            "requested_pose": object_pose,
            "readback_aabb_center": center.tolist(),
            "readback_aabb_size": size.tolist(),
            "success": True,
        }
        write_json(out_dir / "add_object_result.json", result)
        added_rec = {
            "name": object_name,
            "aabb_center": center.tolist(),
            "aabb_size": size.tolist(),
        }
        save_relation_plot(
            out_dir / "add_object_debug.png",
            container_rec,
            added_rec,
            {"label": "added_object", "score": 1.0},
        )
    finally:
        task_sampler.close()

    log.info("Added object test output written to %s", out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe container geometry and visibility in MolmoSpaces scenes.")
    parser.add_argument("--scene_dataset", default="procthor-10k")
    parser.add_argument("--data_split", default="train")
    parser.add_argument("--robot", default="rby1", choices=["rby1", "droid", "rum"])
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan-houses")
    scan_parser.add_argument("--house_inds", nargs="+", type=int, required=True)
    scan_parser.add_argument("--max_plots_per_container", type=int, default=4)
    scan_parser.add_argument("--max_visibility_objects_per_container", type=int, default=8)
    scan_parser.set_defaults(func=command_scan_houses)

    add_parser = subparsers.add_parser("add-object-test")
    add_parser.add_argument("--house_ind", type=int, required=True)
    add_parser.add_argument("--container_name", type=str)
    add_parser.add_argument("--object_relpath", type=str)
    add_parser.add_argument("--object_name", type=str)
    add_parser.set_defaults(func=command_add_object_test)

    debug_parser = subparsers.add_parser("debug-container-view")
    debug_parser.add_argument("--house_ind", type=int, required=True)
    debug_parser.add_argument("--container_name", type=str, required=True)
    debug_parser.add_argument("--joint_index", type=int, required=True)
    debug_parser.add_argument("--joint_value", type=float, required=True)
    debug_parser.add_argument("--head_tilt_delta", type=float, default=0.35)
    debug_parser.set_defaults(func=command_debug_container_view)

    debug_door_parser = subparsers.add_parser("debug-door-view")
    debug_door_parser.add_argument("--house_ind", type=int, required=True)
    debug_door_parser.add_argument("--door_name", type=str)
    debug_door_parser.set_defaults(func=command_debug_door_view)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
