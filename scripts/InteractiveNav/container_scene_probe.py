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
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    RBY1GoProD455CameraSystem,
)
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig, FrankaRobotConfig, RBY1Config
from molmo_spaces.env.data_views import Door, MlSpacesArticulationObject, MlSpacesFreeJointBody
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
DEFAULT_NAV_BENCHMARK_JSON = Path(
    "/home/user/ldl/molmospaces/assets/benchmarks/molmospaces-bench-v2/"
    "procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json"
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
FORCE_DRIVE_TOLERANCE = 0.01


@dataclass
class LoadedContext:
    cfg: Any
    sampler: Any
    task: Any | None
    initial_head_qpos: np.ndarray | None = None
    initial_torso_qpos: np.ndarray | None = None

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
            env._scene_metadata = get_scene_metadata(scene_path) or get_scene_metadata(original_scene_path)
        initial_head_qpos = get_head_joint_position(env)
        initial_torso_qpos = get_torso_joint_position(env)
        apply_default_torso_pose(env, initial_torso_qpos)
        apply_default_arm_pose(env)
        env.camera_manager.setup_cameras(env, cfg.camera_config)
        return LoadedContext(
            cfg=cfg,
            sampler=sampler,
            task=None,
            initial_head_qpos=None if initial_head_qpos is None else initial_head_qpos.copy(),
            initial_torso_qpos=None if initial_torso_qpos is None else initial_torso_qpos.copy(),
        )
    except Exception:
        sampler.close()
        raise


def prepare_writable_scene_path(scene_path: Path) -> str:
    assets_root = Path(os.environ.get("MLSPACES_ASSETS_DIR", "/home/user/ldl/molmospaces/assets"))
    cache_root = Path(os.environ.get("MLSPACES_CACHE_DIR", "/home/user/.cache/molmo-spaces-resources"))
    local_assets_root = REPO_ROOT / "assets"
    tmp_asset_roots = [
        root
        for root in (
            [Path("/tmp/molmo-spaces-assets-proxy")]
            + sorted(Path("/tmp").glob("container_probe_assets_*"))
        )
        if (root / "scenes").exists()
    ]
    resource_roots = [
        cache_root,
        *[
            root
            for root in sorted(Path("/tmp").glob("container_probe_resources_*"))
            if (root / "scenes").exists()
        ],
    ]
    asset_roots = [
        assets_root,
        Path("/home/user/ldl/molmospaces/assets"),
        local_assets_root,
    ] + tmp_asset_roots
    mirror_root = WRITABLE_ASSET_MIRROR
    mirror_root.mkdir(parents=True, exist_ok=True)

    def usable_source(src: Path) -> Path:
        default_cache = Path("/home/user/.cache/molmo-spaces-resources")

        def cache_scene_match(path: Path) -> Path | None:
            for root in asset_roots:
                try:
                    rel = path.relative_to(root / "scenes")
                except ValueError:
                    continue
                for candidate_root in asset_roots:
                    candidate = candidate_root / "scenes" / rel
                    if candidate.exists():
                        return candidate
                parts = rel.parts
                if len(parts) < 2:
                    continue
                dataset = parts[0]
                scene_rel = Path(*parts[1:])
                for resource_root in resource_roots:
                    for candidate in (resource_root / "scenes" / dataset).glob(
                        "*/" + str(scene_rel)
                    ):
                        if candidate.exists():
                            return candidate
            return None

        matched = cache_scene_match(src)
        if matched is not None:
            return matched
        cur = src
        seen: set[Path] = set()
        for _ in range(8):
            matched = cache_scene_match(cur)
            if matched is not None:
                return matched
            if cur.exists() and not cur.is_symlink():
                return cur
            if not cur.is_symlink() or cur in seen:
                break
            seen.add(cur)
            target = Path(os.readlink(cur))
            if not target.is_absolute():
                target = cur.parent / target
            try:
                rel = target.relative_to(default_cache)
            except ValueError:
                rel = None
            if rel is not None:
                redirected = cache_root / rel
                if redirected.exists():
                    return redirected
            cur = target
        if src.exists():
            return src
        return src

    def ensure_symlink(dst: Path, src: Path) -> None:
        src = usable_source(src)
        if dst.is_symlink() and (not dst.exists() or Path(os.readlink(dst)) != src):
            dst.unlink()
        if src.exists() and not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src)

    for top_level in ("objects", "robots", "grasps"):
        src = assets_root / top_level
        dst = mirror_root / top_level
        ensure_symlink(dst, src)

    refs_src = assets_root / "scenes" / "refs"
    refs_dst = mirror_root / "scenes" / "refs"
    refs_dst.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(refs_dst, refs_src)

    rel_scene = None
    for root in asset_roots:
        try:
            rel_scene = scene_path.relative_to(root / "scenes")
            break
        except ValueError:
            continue
    if rel_scene is None:
        raise ValueError(f"Scene path is not under a known assets/scenes root: {scene_path}")
    dst_scene = mirror_root / "scenes" / rel_scene
    dst_scene.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(dst_scene, scene_path)

    for sibling in scene_path.parent.glob(f"{scene_path.stem}*"):
        if sibling == scene_path:
            continue
        if sibling.is_dir() and not sibling.name.endswith("_assets"):
            continue
        sibling_dst = dst_scene.parent / sibling.name
        ensure_symlink(sibling_dst, sibling)

    scene_assets_dir = scene_path.parent / f"{scene_path.stem}_assets"
    if scene_assets_dir.exists():
        dst_assets_dir = dst_scene.parent / scene_assets_dir.name
        ensure_symlink(dst_assets_dir, scene_assets_dir)
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


def get_torso_joint_position(env) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "torso" not in robot_view.move_group_ids():
        return None
    return np.asarray(robot_view.get_move_group("torso").joint_pos, dtype=np.float32).copy()


def get_default_torso_qpos(env, fallback: np.ndarray | None = None) -> np.ndarray | None:
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float32).reshape(-1)
        current = get_torso_joint_position(env)
        if current is not None and fallback.shape == current.shape:
            return fallback.copy()
    try:
        torso_qpos = env.current_robot.exp_config.robot_config.init_qpos.get("torso")
    except Exception:
        torso_qpos = None
    if torso_qpos is None:
        current = get_torso_joint_position(env)
        return None if current is None else np.zeros_like(current, dtype=np.float32)
    torso_qpos = np.asarray(torso_qpos, dtype=np.float32).reshape(-1)
    current = get_torso_joint_position(env)
    if current is not None and torso_qpos.shape != current.shape:
        return np.zeros_like(current, dtype=np.float32)
    return torso_qpos.copy()


def set_torso_joint_position(env, torso_qpos: np.ndarray | list[float]) -> np.ndarray | None:
    robot_view = env.current_robot.robot_view
    if "torso" not in robot_view.move_group_ids():
        return None
    torso_group = robot_view.get_move_group("torso")
    torso_qpos = np.asarray(torso_qpos, dtype=np.float32).reshape(-1)
    limits = np.asarray(torso_group.joint_pos_limits, dtype=np.float32)
    if torso_qpos.shape[0] != limits.shape[0]:
        raise ValueError(f"Expected torso qpos shape {(limits.shape[0],)}, got {torso_qpos.shape}")
    clipped = np.clip(torso_qpos, limits[:, 0], limits[:, 1]).astype(np.float32)
    torso_group.joint_pos = clipped.copy()
    torso_group.ctrl = torso_group.noop_ctrl
    mujoco.mj_forward(env.current_model, env.current_data)
    return clipped


def apply_default_torso_pose(env, default_qpos: np.ndarray | None = None) -> np.ndarray | None:
    default_qpos = get_default_torso_qpos(env, fallback=default_qpos)
    if default_qpos is None:
        return None
    return set_torso_joint_position(env, default_qpos)


def lean_torso_for_drawer_view(
    env,
    default_qpos: np.ndarray | None = None,
    pitch_delta: float = 0.35,
) -> np.ndarray | None:
    base = get_default_torso_qpos(env, fallback=default_qpos)
    if base is None:
        return None
    target = base.copy()
    if target.shape[0] < 2:
        return set_torso_joint_position(env, target)
    # RBY1 torso_1 is the lower pitch joint; positive values move the head
    # forward in the robot frame, giving a "lean over the drawer" viewpoint.
    target[1] = target[1] + float(pitch_delta)
    return set_torso_joint_position(env, target)


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


def body_pose_mat(env, body_id: int) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = env.current_data.xmat[body_id].reshape(3, 3).copy()
    pose[:3, 3] = env.current_data.xpos[body_id].copy()
    return pose


def pos_quat_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = R.from_quat(np.asarray(quat, dtype=float), scalar_first=True).as_matrix()
    pose[:3, 3] = np.asarray(pos, dtype=float)
    return pose


def mat_to_pos_quat(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=float)
    pos = pose[:3, 3].copy()
    quat = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
    return pos, quat


def aabb_bounds(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=float) / 2.0
    center = np.asarray(center, dtype=float)
    return center - half, center + half


def aabb_overlap_metrics(
    a_center: np.ndarray,
    a_size: np.ndarray,
    b_center: np.ndarray,
    b_size: np.ndarray,
) -> dict[str, Any]:
    a_min, a_max = aabb_bounds(np.asarray(a_center, dtype=float), np.asarray(a_size, dtype=float))
    b_min, b_max = aabb_bounds(np.asarray(b_center, dtype=float), np.asarray(b_size, dtype=float))
    inter_size = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    inter_vol = float(np.prod(inter_size))
    a_vol = max(float(np.prod(np.maximum(np.asarray(a_size, dtype=float), 1e-6))), 1e-9)
    b_vol = max(float(np.prod(np.maximum(np.asarray(b_size, dtype=float), 1e-6))), 1e-9)
    return {
        "inter_size": inter_size,
        "inter_vol": inter_vol,
        "ratio_of_a": inter_vol / a_vol,
        "ratio_of_b": inter_vol / b_vol,
    }


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


def set_axes_equal_3d(ax) -> None:
    x_limits = np.asarray(ax.get_xlim3d(), dtype=float)
    y_limits = np.asarray(ax.get_ylim3d(), dtype=float)
    z_limits = np.asarray(ax.get_zlim3d(), dtype=float)
    centers = np.array([x_limits.mean(), y_limits.mean(), z_limits.mean()], dtype=float)
    radius = 0.5 * max(
        float(np.ptp(x_limits)),
        float(np.ptp(y_limits)),
        float(np.ptp(z_limits)),
        1e-3,
    )
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(max(0.0, centers[2] - radius), centers[2] + radius)


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


def joint_name_for_record(env, rec: dict[str, Any], joint_index: int) -> str:
    if rec.get("interaction_group") == "portal":
        door = Door(rec["name"], env.current_data)
        return door.joint_names[joint_index]
    om = env.object_managers[env.current_batch_index]
    obj = om.get_object_by_name(rec["name"])
    if not isinstance(obj, MlSpacesArticulationObject):
        raise ValueError(f"{rec['name']} is not an articulable container.")
    return obj.joint_names[joint_index]


def joint_value_by_name(env, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
    return float(env.current_data.qpos[qpos_addr])


def drive_joint_to_value_with_force(
    env,
    joint_name: str,
    target_value: float,
    *,
    max_steps: int = 1500,
    tolerance: float = FORCE_DRIVE_TOLERANCE,
) -> dict[str, Any]:
    """Move a 1-DoF articulation joint by external body force/torque instead of writing qpos."""
    model = env.current_model
    data = env.current_data
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    joint_type = int(model.jnt_type[joint_id])
    if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError(f"Force drive only supports hinge/slide joints, got type={joint_type}")

    qpos_addr = int(model.jnt_qposadr[joint_id])
    dof_addr = int(model.jnt_dofadr[joint_id])
    body_id = int(model.jnt_bodyid[joint_id])
    local_axis = np.asarray(model.jnt_axis[joint_id], dtype=float)
    joint_range = [float(v) for v in model.jnt_range[joint_id]]
    lo, hi = min(joint_range), max(joint_range)
    target = float(np.clip(target_value, lo, hi))
    is_slide = joint_type == mujoco.mjtJoint.mjJNT_SLIDE
    kp = 600.0 if is_slide else 90.0
    kd = 80.0 if is_slide else 12.0
    max_force = 160.0 if is_slide else 45.0
    stable_steps = 0
    reached = False

    try:
        for step in range(int(max_steps)):
            current = float(data.qpos[qpos_addr])
            velocity = float(data.qvel[dof_addr])
            error = target - current
            effort = float(np.clip(kp * error - kd * velocity, -max_force, max_force))
            world_axis = data.xmat[body_id].reshape(3, 3) @ local_axis
            if is_slide:
                data.xfrc_applied[body_id, :3] = world_axis * effort
            else:
                data.xfrc_applied[body_id, 3:] = world_axis * effort
            mujoco.mj_step(model, data)
            if abs(error) <= tolerance and abs(velocity) <= tolerance:
                stable_steps += 1
                if stable_steps >= 8:
                    reached = True
                    break
            else:
                stable_steps = 0
    finally:
        data.xfrc_applied[body_id, :] = 0.0
        data.qvel[dof_addr] = 0.0
        mujoco.mj_forward(model, data)

    final_value = float(data.qpos[qpos_addr])
    final_error = float(target - final_value)
    reached = reached or abs(final_error) <= tolerance
    if not reached:
        log.warning(
            "Force drive did not fully converge for %s: target=%.4f final=%.4f error=%.4f",
            joint_name,
            target,
            final_value,
            final_error,
        )
    return {
        "method": "xfrc_applied_pd",
        "force_application": "xfrc_applied_body_force_or_torque",
        "joint_name": joint_name,
        "joint_type": "slide" if is_slide else "hinge",
        "body_id": body_id,
        "joint_range": joint_range,
        "target_value": target,
        "final_value": final_value,
        "final_error": final_error,
        "steps": step + 1,
        "reached": reached,
        "tolerance": float(tolerance),
    }


def drive_articulation_state_by_record(
    env,
    rec: dict[str, Any],
    joint_index: int,
    joint_value: float,
) -> dict[str, Any]:
    joint_name = joint_name_for_record(env, rec, joint_index)
    return drive_joint_to_value_with_force(env, joint_name, joint_value)


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
    drive_meta = drive_joint_to_value_with_force(env, container_obj.joint_names[joint_index], target)
    return {
        "container_name": container_name,
        "joint_index": joint_index,
        "joint_range": joint_range,
        "target_value": float(target),
        "open_fraction": float(open_fraction),
        "drive": drive_meta,
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
    drive_meta = drive_joint_to_value_with_force(env, door.joint_names[hinge_idx], target)
    return {
        "door_name": door_name,
        "hinge_joint_index": hinge_idx,
        "hinge_joint_name": door.joint_names[hinge_idx],
        "hinge_joint_range": joint_range,
        "target_value": float(target),
        "open_fraction": float(open_fraction),
        "drive": drive_meta,
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
    desired_dist: float = 0.8,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center, _size = joint_target_geometry(env, articulation_rec, joint)
    front_axis_xy = container_front_axis(articulation_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
    lateral_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)

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


def joint_body_id(env, joint: dict[str, Any]) -> int:
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint["joint_name"])
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint['joint_name']}")
    return int(env.current_model.jnt_bodyid[joint_id])


def joint_mujoco_type_name(env, joint: dict[str, Any]) -> str:
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint["joint_name"])
    if joint_id < 0:
        return str(joint.get("joint_type"))
    joint_type = int(env.current_model.jnt_type[joint_id])
    if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        return "slide"
    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return "hinge"
    return str(joint_type)


def articulation_joint_records(rec: dict[str, Any]) -> list[dict[str, Any]]:
    if rec.get("joints"):
        return list(rec["joints"])
    if rec.get("interaction_group") == "portal":
        return [
            {
                "joint_index": int(rec["hinge_joint_index"]),
                "joint_name": rec["hinge_joint_name"],
                "joint_type": "hinge",
                "joint_range": rec["hinge_joint_range"],
                "current_value": None,
                "closed_value": float(rec["closed_value"]),
                "open_value": float(rec["open_value"]),
            }
        ]
    return []


def set_all_articulation_joints_closed(
    env,
    articulation_rec: dict[str, Any],
    joints: list[dict[str, Any]],
) -> None:
    for joint in joints:
        set_articulation_state_by_record(
            env,
            articulation_rec,
            int(joint["joint_index"]),
            float(joint["closed_value"]),
        )


def collect_joint_box_state_records(
    env,
    articulation_rec: dict[str, Any],
    joints: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    qpos_before = env.current_data.qpos.copy()
    records: dict[int, dict[str, Any]] = {}
    try:
        set_all_articulation_joints_closed(env, articulation_rec, joints)
        for joint in joints:
            joint_index = int(joint["joint_index"])
            center, size = joint_target_geometry(env, articulation_rec, joint)
            records[joint_index] = {
                "joint_index": joint_index,
                "joint_name": joint["joint_name"],
                "joint_type": joint.get("joint_type"),
                "mujoco_joint_type": joint_mujoco_type_name(env, joint),
                "body_id": joint_body_id(env, joint),
                "closed_value": float(joint["closed_value"]),
                "open_value": float(joint["open_value"]),
                "closed_box": {"center": center, "size": size},
            }

        for joint in joints:
            joint_index = int(joint["joint_index"])
            set_all_articulation_joints_closed(env, articulation_rec, joints)
            set_articulation_state_by_record(
                env,
                articulation_rec,
                joint_index,
                float(joint["open_value"]),
            )
            center, size = joint_target_geometry(env, articulation_rec, joint)
            closed_center = np.asarray(records[joint_index]["closed_box"]["center"], dtype=float)
            records[joint_index]["open_box"] = {"center": center, "size": size}
            records[joint_index]["open_delta"] = np.asarray(center, dtype=float) - closed_center
    finally:
        env.current_data.qpos[:] = qpos_before
        mujoco.mj_forward(env.current_model, env.current_data)
    return records


def infer_front_axis_from_joint_motion(
    container_rec: dict[str, Any],
    joint_box_records: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, str]:
    slide_dirs = []
    for rec in joint_box_records.values():
        if rec.get("mujoco_joint_type") != "slide":
            continue
        delta_xy = np.asarray(rec.get("open_delta", [0.0, 0.0, 0.0]), dtype=float)[:2]
        norm = float(np.linalg.norm(delta_xy))
        if norm > 1e-3:
            slide_dirs.append(delta_xy / norm)

    if slide_dirs:
        front_axis = np.mean(np.stack(slide_dirs, axis=0), axis=0)
        norm = float(np.linalg.norm(front_axis))
        if norm > 1e-6:
            return front_axis / norm, "slide_open_delta"

    front_axis = container_front_axis(container_rec)
    motion_votes = []
    for rec in joint_box_records.values():
        delta_xy = np.asarray(rec.get("open_delta", [0.0, 0.0, 0.0]), dtype=float)[:2]
        if np.linalg.norm(delta_xy) > 1e-3:
            motion_votes.append(float(np.dot(delta_xy, front_axis)))
    if motion_votes and sum(motion_votes) < 0.0:
        return -front_axis, "container_front_axis_flipped_by_open_delta"
    return front_axis, "container_front_axis"


def box_front_projection(
    center: np.ndarray,
    size: np.ndarray,
    front_axis_xy: np.ndarray,
) -> dict[str, Any]:
    front_axis_xy = np.asarray(front_axis_xy, dtype=float)
    front_axis_xy = front_axis_xy / max(float(np.linalg.norm(front_axis_xy)), 1e-9)
    lateral_axis_xy = np.array([-front_axis_xy[1], front_axis_xy[0]], dtype=float)
    corners = compute_box_corners(np.asarray(center, dtype=float), np.asarray(size, dtype=float))
    xy = corners[:, :2]
    depth = xy @ front_axis_xy
    lateral = xy @ lateral_axis_xy
    z_vals = corners[:, 2]
    return {
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_center": float(np.asarray(center, dtype=float)[:2] @ front_axis_xy),
        "lateral_min": float(lateral.min()),
        "lateral_max": float(lateral.max()),
        "z_min": float(z_vals.min()),
        "z_max": float(z_vals.max()),
    }


def front_projection_overlap_metrics(
    target_proj: dict[str, Any],
    candidate_proj: dict[str, Any],
) -> dict[str, Any]:
    lateral_overlap = max(
        0.0,
        min(target_proj["lateral_max"], candidate_proj["lateral_max"])
        - max(target_proj["lateral_min"], candidate_proj["lateral_min"]),
    )
    z_overlap = max(
        0.0,
        min(target_proj["z_max"], candidate_proj["z_max"])
        - max(target_proj["z_min"], candidate_proj["z_min"]),
    )
    overlap_area = float(lateral_overlap * z_overlap)
    target_area = max(
        float(
            (target_proj["lateral_max"] - target_proj["lateral_min"])
            * (target_proj["z_max"] - target_proj["z_min"])
        ),
        1e-9,
    )
    candidate_area = max(
        float(
            (candidate_proj["lateral_max"] - candidate_proj["lateral_min"])
            * (candidate_proj["z_max"] - candidate_proj["z_min"])
        ),
        1e-9,
    )
    return {
        "overlap_lateral": lateral_overlap,
        "overlap_z": z_overlap,
        "overlap_area": overlap_area,
        "ratio_of_target_projection": overlap_area / target_area,
        "ratio_of_candidate_projection": overlap_area / candidate_area,
    }


def union_aabb(
    a_center: np.ndarray,
    a_size: np.ndarray,
    b_center: np.ndarray,
    b_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a_min, a_max = aabb_bounds(np.asarray(a_center, dtype=float), np.asarray(a_size, dtype=float))
    b_min, b_max = aabb_bounds(np.asarray(b_center, dtype=float), np.asarray(b_size, dtype=float))
    out_min = np.minimum(a_min, b_min)
    out_max = np.maximum(a_max, b_max)
    return (out_min + out_max) / 2.0, out_max - out_min


def box_projection_along_axis(
    center: np.ndarray,
    size: np.ndarray,
    axis_xy: np.ndarray,
) -> dict[str, Any]:
    axis_xy = np.asarray(axis_xy, dtype=float)
    axis_xy = axis_xy / max(float(np.linalg.norm(axis_xy)), 1e-9)
    lateral_axis_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=float)
    corners = compute_box_corners(np.asarray(center, dtype=float), np.asarray(size, dtype=float))
    xy = corners[:, :2]
    depth = xy @ axis_xy
    lateral = xy @ lateral_axis_xy
    z_vals = corners[:, 2]
    return {
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "lateral_min": float(lateral.min()),
        "lateral_max": float(lateral.max()),
        "z_min": float(z_vals.min()),
        "z_max": float(z_vals.max()),
    }


def interval_overlap_and_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> tuple[float, float]:
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    gap = max(0.0, max(a_min, b_min) - min(a_max, b_max))
    return overlap, gap


def slide_hinge_blocker_candidates(
    target_rec: dict[str, Any],
    joint_box_records: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    delta_xy = np.asarray(target_rec.get("open_delta", [0.0, 0.0, 0.0]), dtype=float)[:2]
    delta_norm = float(np.linalg.norm(delta_xy))
    if delta_norm <= 1e-4:
        return []
    slide_axis_xy = delta_xy / delta_norm
    swept_center, swept_size = union_aabb(
        target_rec["closed_box"]["center"],
        target_rec["closed_box"]["size"],
        target_rec["open_box"]["center"],
        target_rec["open_box"]["size"],
    )
    swept_proj = box_projection_along_axis(swept_center, swept_size, slide_axis_xy)
    swept_depth = max(swept_proj["depth_max"] - swept_proj["depth_min"], 1e-9)
    swept_lateral = max(swept_proj["lateral_max"] - swept_proj["lateral_min"], 1e-9)
    swept_z = max(swept_proj["z_max"] - swept_proj["z_min"], 1e-9)
    lateral_gap_allowance = max(0.12, 1.5 * swept_lateral)

    candidates = []
    for candidate_rec in joint_box_records.values():
        if candidate_rec["joint_index"] == target_rec["joint_index"]:
            continue
        if candidate_rec.get("mujoco_joint_type") != "hinge":
            continue
        candidate_proj = box_projection_along_axis(
            candidate_rec["closed_box"]["center"],
            candidate_rec["closed_box"]["size"],
            slide_axis_xy,
        )
        depth_overlap, depth_gap = interval_overlap_and_gap(
            swept_proj["depth_min"],
            swept_proj["depth_max"],
            candidate_proj["depth_min"],
            candidate_proj["depth_max"],
        )
        lateral_overlap, lateral_gap = interval_overlap_and_gap(
            swept_proj["lateral_min"],
            swept_proj["lateral_max"],
            candidate_proj["lateral_min"],
            candidate_proj["lateral_max"],
        )
        z_overlap, z_gap = interval_overlap_and_gap(
            swept_proj["z_min"],
            swept_proj["z_max"],
            candidate_proj["z_min"],
            candidate_proj["z_max"],
        )
        depth_ratio = depth_overlap / swept_depth
        lateral_ratio = lateral_overlap / swept_lateral
        z_ratio = z_overlap / swept_z
        lateral_near = lateral_gap <= lateral_gap_allowance
        valid = z_ratio >= 0.5 and depth_ratio >= 0.1 and (lateral_overlap > 0.0 or lateral_near)
        if not valid:
            continue

        volume_overlap = aabb_overlap_metrics(
            swept_center,
            swept_size,
            candidate_rec["closed_box"]["center"],
            candidate_rec["closed_box"]["size"],
        )
        lateral_score = lateral_ratio if lateral_overlap > 0.0 else 0.2 * (1.0 - lateral_gap / lateral_gap_allowance)
        score = 3.0 * z_ratio + depth_ratio + max(0.0, lateral_score)
        candidates.append(
            {
                "joint_index": int(candidate_rec["joint_index"]),
                "score": float(score),
                "has_lateral_overlap": bool(lateral_overlap > 0.0),
                "depth_overlap": float(depth_overlap),
                "depth_gap": float(depth_gap),
                "depth_ratio": float(depth_ratio),
                "lateral_overlap": float(lateral_overlap),
                "lateral_gap": float(lateral_gap),
                "lateral_ratio": float(lateral_ratio),
                "lateral_gap_allowance": float(lateral_gap_allowance),
                "z_overlap": float(z_overlap),
                "z_gap": float(z_gap),
                "z_ratio": float(z_ratio),
                "slide_axis_xy": slide_axis_xy,
                "swept_box": {"center": swept_center, "size": swept_size},
                "swept_projection": swept_proj,
                "candidate_projection_in_slide_frame": candidate_proj,
                "swept_overlap_with_candidate_closed_box": volume_overlap,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["has_lateral_overlap"],
            item["score"],
            item["depth_overlap"],
            -item["lateral_gap"],
        ),
        reverse=True,
    )
    return candidates


def infer_front_axis_from_dependencies(
    dependencies_by_target: dict[int, list[dict[str, Any]]],
    joint_box_records: dict[int, dict[str, Any]],
) -> np.ndarray | None:
    directions = []
    for target_index, candidates in dependencies_by_target.items():
        target_rec = joint_box_records.get(target_index)
        if target_rec is None:
            continue
        target_center = np.asarray(target_rec["closed_box"]["center"], dtype=float)[:2]
        for candidate in candidates:
            prereq_rec = joint_box_records.get(int(candidate["joint_index"]))
            if prereq_rec is None:
                continue
            prereq_center = np.asarray(prereq_rec["closed_box"]["center"], dtype=float)[:2]
            delta = prereq_center - target_center
            norm = float(np.linalg.norm(delta))
            if norm > 1e-3:
                directions.append(delta / norm)
    if not directions:
        return None
    axis = np.mean(np.stack(directions, axis=0), axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return None
    return axis / norm


def infer_joint_open_dependencies(
    env,
    articulation_rec: dict[str, Any],
    *,
    method: str = "front_occlusion",
    min_overlap_volume: float = 1e-6,
    min_overlap_ratio: float = 0.0,
    min_projection_ratio: float = 0.15,
    min_depth_separation: float = 0.01,
) -> list[dict[str, Any]]:
    """Infer prerequisite joints from articulation geometry.

    The default front-occlusion mode treats a candidate joint as a prerequisite
    only when its closed box is in front of the target joint's closed box and
    substantially overlaps the target in the front-view projection.
    """
    joints = articulation_joint_records(articulation_rec)
    joint_box_records = collect_joint_box_state_records(env, articulation_rec, joints)
    blocker_candidates_by_target: dict[int, list[dict[str, Any]]] = {}
    if method == "front_occlusion":
        for rec in joint_box_records.values():
            if rec.get("mujoco_joint_type") != "slide":
                continue
            candidates = slide_hinge_blocker_candidates(rec, joint_box_records)
            if candidates:
                blocker_candidates_by_target[int(rec["joint_index"])] = [candidates[0]]

    dependency_front_axis = infer_front_axis_from_dependencies(
        blocker_candidates_by_target, joint_box_records
    )
    if dependency_front_axis is not None:
        front_axis_xy = dependency_front_axis
        front_axis_source = "slide_hinge_blocker_geometry"
    else:
        front_axis_xy, front_axis_source = infer_front_axis_from_joint_motion(
            articulation_rec, joint_box_records
        )

    if method not in {"front_occlusion", "open_aabb_overlap"}:
        raise ValueError(f"Unsupported dependency inference method: {method}")

    results: list[dict[str, Any]] = []

    for target_joint in joints:
        target_index = int(target_joint["joint_index"])
        target_rec = joint_box_records[target_index]
        target_closed_box = target_rec["closed_box"]
        target_open_box = target_rec["open_box"]
        target_proj = box_front_projection(
            target_closed_box["center"], target_closed_box["size"], front_axis_xy
        )
        prerequisites = []

        for candidate_joint in joints:
            candidate_index = int(candidate_joint["joint_index"])
            if candidate_index == target_index:
                continue
            candidate_rec = joint_box_records[candidate_index]
            candidate_box = candidate_rec["closed_box"]

            if method == "open_aabb_overlap":
                overlap = aabb_overlap_metrics(
                    target_open_box["center"],
                    target_open_box["size"],
                    candidate_box["center"],
                    candidate_box["size"],
                )
                is_prerequisite = (
                    overlap["inter_vol"] > float(min_overlap_volume)
                    and max(overlap["ratio_of_a"], overlap["ratio_of_b"]) >= float(min_overlap_ratio)
                )
                evidence = {"overlap_with_target_open_box": overlap}
            else:
                selected = next(
                    (
                        item
                        for item in blocker_candidates_by_target.get(target_index, [])
                        if int(item["joint_index"]) == candidate_index
                    ),
                    None,
                )
                is_prerequisite = selected is not None
                candidate_proj = box_front_projection(candidate_box["center"], candidate_box["size"], front_axis_xy)
                projection = front_projection_overlap_metrics(target_proj, candidate_proj)
                depth_separation = candidate_proj["depth_center"] - target_proj["depth_center"]
                evidence = {
                    "dependency_source": "slide_hinge_blocker_geometry",
                    "front_projection_overlap": projection,
                    "target_front_projection": target_proj,
                    "candidate_front_projection": candidate_proj,
                    "depth_separation": depth_separation,
                    "is_in_front": depth_separation >= float(min_depth_separation),
                    "is_projected_occluder": projection["overlap_area"] > 0.0
                    and projection["ratio_of_target_projection"] >= float(min_projection_ratio),
                    "slide_hinge_blocker_evidence": selected,
                }

            if not is_prerequisite:
                continue

            prerequisites.append(
                {
                    "joint_index": candidate_index,
                    "joint_name": candidate_joint["joint_name"],
                    "joint_type": candidate_joint.get("joint_type"),
                    "mujoco_joint_type": candidate_rec.get("mujoco_joint_type"),
                    "body_id": candidate_rec["body_id"],
                    "closed_box": candidate_box,
                    **evidence,
                }
            )

        results.append(
            {
                "joint_index": target_index,
                "joint_name": target_joint["joint_name"],
                "joint_type": target_joint.get("joint_type"),
                "mujoco_joint_type": target_rec.get("mujoco_joint_type"),
                "body_id": target_rec["body_id"],
                "closed_value": float(target_joint["closed_value"]),
                "open_value": float(target_joint["open_value"]),
                "closed_box": target_closed_box,
                "open_box": target_open_box,
                "open_delta": target_rec.get("open_delta"),
                "front_axis_xy": front_axis_xy,
                "front_axis_source": front_axis_source,
                "dependency_method": method,
                "prerequisite_joint_indices": [item["joint_index"] for item in prerequisites],
                "prerequisite_joints": prerequisites,
            }
        )
    return results


def save_joint_dependency_plot(
    output_path: Path,
    articulation_rec: dict[str, Any],
    joint_dependencies: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]

    front_axis = np.array([1.0, 0.0], dtype=float)
    if joint_dependencies:
        front_axis = np.asarray(joint_dependencies[0].get("front_axis_xy", [1.0, 0.0]), dtype=float)
    front_axis = front_axis / max(float(np.linalg.norm(front_axis)), 1e-9)
    lateral_axis = np.array([-front_axis[1], front_axis[0]], dtype=float)
    origin_xy = np.asarray(articulation_rec["aabb_center"], dtype=float)[:2]

    def local_box(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        corners = compute_box_corners(center, size)
        xy = corners[:, :2] - origin_xy[None, :]
        local_corners = np.column_stack(
            [
                xy @ lateral_axis,
                xy @ front_axis,
                corners[:, 2],
            ]
        )
        bmin = local_corners.min(axis=0)
        bmax = local_corners.max(axis=0)
        return (bmin + bmax) / 2.0, bmax - bmin

    def local_point(point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        xy = point[:2] - origin_xy
        return np.array([float(xy @ lateral_axis), float(xy @ front_axis), float(point[2])], dtype=float)

    all_centers = []
    all_sizes = []
    for dep in joint_dependencies:
        color = colors[int(dep["joint_index"]) % len(colors)]
        closed_box = dep["closed_box"]
        open_box = dep["open_box"]
        closed_center_world = np.asarray(closed_box["center"], dtype=float)
        closed_size = np.asarray(closed_box["size"], dtype=float)
        open_center_world = np.asarray(open_box["center"], dtype=float)
        open_size = np.asarray(open_box["size"], dtype=float)
        closed_center, closed_size_local = local_box(closed_center_world, closed_size)
        open_center, open_size_local = local_box(open_center_world, open_size)
        all_centers.extend([closed_center, open_center])
        all_sizes.extend([closed_size_local, open_size_local])
        add_box_to_ax(
            ax,
            closed_center,
            closed_size_local,
            color,
            f"j{dep['joint_index']} closed",
            0.20,
        )
        add_box_to_ax(
            ax,
            open_center,
            open_size_local,
            color,
            f"j{dep['joint_index']} open",
            0.06,
        )
        ax.plot(
            [closed_center[0], open_center[0]],
            [closed_center[1], open_center[1]],
            [closed_center[2], open_center[2]],
            color=color,
            linestyle="--",
            linewidth=1.0,
        )

    dep_by_index = {int(dep["joint_index"]): dep for dep in joint_dependencies}
    for dep in joint_dependencies:
        target_center = local_point(np.asarray(dep["closed_box"]["center"], dtype=float))
        for prereq in dep["prerequisite_joints"]:
            prereq_dep = dep_by_index.get(int(prereq["joint_index"]))
            if prereq_dep is None:
                continue
            prereq_center = local_point(np.asarray(prereq_dep["closed_box"]["center"], dtype=float))
            delta = target_center - prereq_center
            ax.quiver(
                prereq_center[0],
                prereq_center[1],
                prereq_center[2],
                delta[0],
                delta[1],
                delta[2],
                color="black",
                arrow_length_ratio=0.12,
                linewidth=1.5,
            )

    if joint_dependencies:
        container_center = local_point(np.asarray(articulation_rec["aabb_center"], dtype=float))
        arrow_start = container_center.copy()
        arrow_start[2] = max(arrow_start[2], 0.05)
        ax.quiver(
            arrow_start[0],
            arrow_start[1],
            arrow_start[2],
            0.0,
            0.6,
            0.0,
            color="magenta",
            arrow_length_ratio=0.18,
            linewidth=2.0,
        )
        ax.text(
            arrow_start[0],
            arrow_start[1] + 0.65,
            arrow_start[2],
            "front",
            color="magenta",
            fontsize=10,
        )

    if all_centers:
        mins = []
        maxs = []
        for center, size in zip(all_centers, all_sizes):
            bmin, bmax = aabb_bounds(center, size)
            mins.append(bmin)
            maxs.append(bmax)
        mins_arr = np.min(np.stack(mins, axis=0), axis=0)
        maxs_arr = np.max(np.stack(maxs, axis=0), axis=0)
        margin = 0.25
        ax.set_xlim(float(mins_arr[0] - margin), float(maxs_arr[0] + margin))
        ax.set_ylim(float(mins_arr[1] - margin), float(maxs_arr[1] + margin))
        ax.set_zlim(max(0.0, float(mins_arr[2] - margin)), float(maxs_arr[2] + margin))
        set_axes_equal_3d(ax)

    ax.set_xlabel("lateral")
    ax.set_ylabel("front_depth")
    ax.set_zlabel("z")
    title = f"{articulation_rec.get('asset_id') or articulation_rec.get('category')} joint boxes"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def aabb_contains_object(
    box_center: np.ndarray,
    box_size: np.ndarray,
    object_rec: dict[str, Any],
    padding: float = 0.0,
) -> bool:
    padded_size = np.asarray(box_size, dtype=float) + 2.0 * float(padding)
    bmin, bmax = aabb_bounds(np.asarray(box_center, dtype=float), padded_size)
    omin, omax = aabb_bounds(
        np.asarray(object_rec["aabb_center"], dtype=float),
        np.asarray(object_rec["aabb_size"], dtype=float),
    )
    return bool(np.all(omin >= bmin) and np.all(omax <= bmax))


def free_joint_pose(env, object_name: str) -> np.ndarray | None:
    try:
        body = MlSpacesFreeJointBody(env.current_data, object_name)
    except AssertionError:
        return None
    return pos_quat_to_mat(body.position, body.quat)


def set_free_joint_pose(env, object_name: str, pose: np.ndarray) -> bool:
    try:
        body = MlSpacesFreeJointBody(env.current_data, object_name)
    except AssertionError:
        return False
    pos, quat = mat_to_pos_quat(pose)
    body.position = pos
    body.quat = quat
    mujoco.mj_forward(env.current_model, env.current_data)
    return True


def drawer_joint_contained_objects(
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    joint: dict[str, Any],
    object_records: list[dict[str, Any]],
    padding: float = 0.0,
) -> list[dict[str, Any]]:
    env = ctx.env
    qpos_before = env.current_data.qpos.copy()
    try:
        set_articulation_state_by_record(
            env, container_rec, joint["joint_index"], joint["closed_value"]
        )
        center, size = joint_target_geometry(env, container_rec, joint)
        out = []
        for object_rec in object_records:
            if not object_rec.get("has_free_joint"):
                continue
            if object_rec["name"] == container_rec["name"]:
                continue
            body_id = int(object_rec["body_id"])
            cur_center, cur_size = safe_body_aabb(env.current_model, env.current_data, body_id)
            cur_rec = {**object_rec, "aabb_center": cur_center, "aabb_size": cur_size}
            if aabb_contains_object(center, size, cur_rec, padding=padding):
                out.append(cur_rec)
        return out
    finally:
        env.current_data.qpos[:] = qpos_before
        mujoco.mj_forward(env.current_model, env.current_data)


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
    ctx: LoadedContext,
    container_rec: dict[str, Any],
    joint: dict[str, Any],
    desired_dist: float = 0.8,
) -> tuple[bool, np.ndarray | None]:
    env = ctx.env
    robot_view = env.current_robot.robot_view
    center, _size = joint_target_geometry(env, container_rec, joint)
    front_axis_xy = container_front_axis(container_rec)
    thormap = env.get_thormap(agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius)
    free_points = thormap.get_free_points()
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


def command_debug_drawer_bound_object(args: argparse.Namespace) -> int:
    out_dir = args.output_dir / f"debug_drawer_bound_object_house_{args.house_ind}_joint_{args.joint_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        records, containers = collect_scene_records(ctx)
        container_rec = next((rec for rec in containers if rec["name"] == args.container_name), None)
        if container_rec is None:
            raise RuntimeError(f"Container not found: {args.container_name}")
        joint = next((item for item in container_rec["joints"] if item["joint_index"] == args.joint_index), None)
        if joint is None:
            raise RuntimeError(f"Joint index {args.joint_index} not found on {args.container_name}")

        env = ctx.env
        object_records = [rec for rec in records if is_target_like(rec)]
        contained = drawer_joint_contained_objects(
            ctx, container_rec, joint, object_records, padding=args.box_padding
        )
        if args.object_name:
            object_rec = next((rec for rec in records if rec["name"] == args.object_name), None)
            if object_rec is None:
                raise RuntimeError(f"Object not found: {args.object_name}")
        elif contained:
            contained.sort(key=lambda rec: (str(rec["category"]), rec["name"]))
            object_rec = contained[0]
        else:
            raise RuntimeError(
                f"No free-joint target-like object found inside joint {args.joint_index} "
                f"AABB with padding={args.box_padding}."
            )
        object_name = object_rec["name"]

        qpos_before = env.current_data.qpos.copy()
        robot_pose_before = env.current_robot.robot_view.base.pose.copy()
        head_before = get_head_joint_position(env)
        torso_before = get_torso_joint_position(env)
        try:
            set_articulation_state_by_record(
                env, container_rec, joint["joint_index"], joint["closed_value"]
            )
            closed_joint_body_pose = body_pose_mat(env, joint_body_id(env, joint))
            closed_object_pose = free_joint_pose(env, object_name)
            if closed_object_pose is None:
                raise RuntimeError(f"{object_name} does not have a free joint pose.")
            closed_object_pos = env.current_data.xpos[int(object_rec["body_id"])].copy()

            set_articulation_state_by_record(
                env, container_rec, joint["joint_index"], joint["open_value"]
            )
            open_joint_body_pose = body_pose_mat(env, joint_body_id(env, joint))
            drawer_delta = open_joint_body_pose @ np.linalg.inv(closed_joint_body_pose)
            moved_object_pose = drawer_delta @ closed_object_pose

            shared_pose, shared_meta = choose_pose_valid_for_joint_states(
                ctx,
                container_rec,
                joint,
                joint["closed_value"],
                joint["open_value"],
                desired_dist=args.view_distance,
            )
            if shared_pose is None:
                ok, placed_pose = place_robot_for_container_joint(
                    ctx, container_rec, joint, desired_dist=args.view_distance
                )
                if not ok or placed_pose is None:
                    raise RuntimeError("Could not place robot for drawer view.")
                shared_pose = placed_pose
                shared_meta = {"fallback": "place_robot_for_container_joint"}

            def render_case(
                tag: str,
                joint_value: float,
                object_pose: np.ndarray,
                *,
                force_drive: bool = False,
                known_issue: str | None = None,
            ) -> dict[str, Any]:
                env.current_data.qpos[:] = qpos_before
                mujoco.mj_forward(env.current_model, env.current_data)
                if force_drive:
                    set_articulation_state_by_record(
                        env, container_rec, joint["joint_index"], joint["closed_value"]
                    )
                    set_free_joint_pose(env, object_name, object_pose)
                    drive_meta = drive_articulation_state_by_record(
                        env, container_rec, joint["joint_index"], joint_value
                    )
                else:
                    set_articulation_state_by_record(
                        env, container_rec, joint["joint_index"], joint_value
                    )
                    set_free_joint_pose(env, object_name, object_pose)
                    drive_meta = {"method": "direct_set_joint_position"}
                env.current_robot.robot_view.base.pose = shared_pose
                mujoco.mj_forward(env.current_model, env.current_data)
                apply_default_torso_pose(env, ctx.initial_torso_qpos)
                torso_qpos = lean_torso_for_drawer_view(
                    env, ctx.initial_torso_qpos, pitch_delta=args.torso_lean_delta
                )
                apply_default_arm_pose(env)
                apply_default_head_pose(env, ctx.initial_head_qpos)
                lower_head_for_drawer_view(env, tilt_delta=args.head_tilt_delta)
                env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
                rgb_path = out_dir / f"{tag}_rgb.png"
                seg_path = out_dir / f"{tag}_seg.png"
                save_debug_head_image(env, rgb_path)
                save_segmentation_preview(seg_path, env.render_segmentation_frame("head_camera"))
                visibility = env.check_visibility("head_camera", object_name)
                if isinstance(visibility, dict):
                    vis_value = float(visibility.get(object_name, 0.0))
                else:
                    vis_value = float(visibility)
                body_id = int(object_rec["body_id"])
                center, size = safe_body_aabb(env.current_model, env.current_data, body_id)
                return {
                    "tag": tag,
                    "joint_value": float(joint_value),
                    "actual_joint_value": joint_value_by_name(env, joint["joint_name"]),
                    "drive": drive_meta,
                    "known_issue": known_issue,
                    "rgb": str(rgb_path),
                    "seg": str(seg_path),
                    "visibility": vis_value,
                    "object_xpos": env.current_data.xpos[body_id].copy().tolist(),
                    "object_aabb_center": center.tolist(),
                    "object_aabb_size": size.tolist(),
                    "head_qpos": get_head_joint_position(env).tolist()
                    if get_head_joint_position(env) is not None
                    else None,
                    "torso_qpos": None if torso_qpos is None else torso_qpos.tolist(),
                }

            cases = [
                render_case("closed_look_down", joint["closed_value"], closed_object_pose),
                render_case("open_without_move_look_down", joint["open_value"], closed_object_pose),
                render_case(
                    "open_with_moved_object_look_down",
                    joint["open_value"],
                    moved_object_pose,
                    known_issue=(
                        "Known-bad workaround: the object is moved by the joint body's global "
                        "delta, but this can mismatch drawer compartments and make an object "
                        "from one drawer appear to fly out with another drawer."
                    ),
                ),
                render_case(
                    "open_with_force_look_down",
                    joint["open_value"],
                    closed_object_pose,
                    force_drive=True,
                ),
            ]
            moved_pos = np.asarray(cases[2]["object_xpos"], dtype=float)
            force_pos = np.asarray(cases[-1]["object_xpos"], dtype=float)
            payload = {
                "house_ind": args.house_ind,
                "container_name": container_rec["name"],
                "joint": joint,
                "box_padding": args.box_padding,
                "view_distance": args.view_distance,
                "head_tilt_delta": args.head_tilt_delta,
                "torso_lean_delta": args.torso_lean_delta,
                "contained_candidates": [
                    {
                        "name": rec["name"],
                        "category": rec["category"],
                        "aabb_center": np.asarray(rec["aabb_center"], dtype=float).tolist(),
                        "aabb_size": np.asarray(rec["aabb_size"], dtype=float).tolist(),
                    }
                    for rec in contained
                ],
                "selected_object": {
                    "name": object_name,
                    "category": object_rec["category"],
                    "body_id": int(object_rec["body_id"]),
                },
                "shared_robot_pose": shared_pose.tolist(),
                "shared_pose_meta": shared_meta,
                "closed_object_xpos": closed_object_pos.tolist(),
                "moved_object_xpos_delta": (moved_pos - closed_object_pos).tolist(),
                "force_drive_object_xpos_delta": (force_pos - closed_object_pos).tolist(),
                "manual_move_known_issue": (
                    "Do not treat open_with_moved_object_look_down as a valid interaction result. "
                    "It was kept only as a diagnostic for the old workaround; it can attach an "
                    "object to the wrong drawer space."
                ),
                "drawer_delta": drawer_delta.tolist(),
                "cases": cases,
            }
            write_json(out_dir / "drawer_bound_object_result.json", payload)
            return 0
        finally:
            env.current_data.qpos[:] = qpos_before
            env.current_robot.robot_view.base.pose = robot_pose_before
            mujoco.mj_forward(env.current_model, env.current_data)
            if torso_before is not None:
                set_torso_joint_position(env, torso_before)
            else:
                apply_default_torso_pose(env, ctx.initial_torso_qpos)
            apply_default_arm_pose(env)
            if head_before is not None:
                set_head_joint_position(env, head_before)
            else:
                apply_default_head_pose(env, ctx.initial_head_qpos)
            env.camera_manager.setup_cameras(env, ctx.cfg.camera_config)
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


def benchmark_entries_by_house(path: Path, max_episodes: int) -> dict[int, list[dict[str, Any]]]:
    with open(path) as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    out: dict[int, list[dict[str, Any]]] = {}
    for episode_index, episode in enumerate(episodes[:max_episodes]):
        house_ind = int(episode["house_index"])
        task = episode.get("task", {})
        candidates = task.get("pickup_obj_candidates") or []
        target = task.get("pickup_obj_name")
        out.setdefault(house_ind, []).append(
            {
                "episode_index": episode_index,
                "pickup_obj_name": target,
                "pickup_obj_candidates": candidates,
                "task_description": episode.get("language", {}).get("task_description"),
            }
        )
    return out


def resolve_benchmark_json(path: Path) -> Path:
    if path.exists():
        return path
    rel = Path(
        "benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/"
        "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json"
    )
    candidates = [
        Path(os.environ.get("MLSPACES_ASSETS_DIR", "/home/user/ldl/molmospaces/assets")) / rel,
        Path(os.environ.get("MLSPACES_CACHE_DIR", "/home/user/.cache/molmo-spaces-resources"))
        / "benchmarks/molmospaces-bench-v2/20240407/procthor-10k/NavToObjDataGenConfig/"
        "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Benchmark JSON not found: {path}")


def command_scan_container_target_overlap(args: argparse.Namespace) -> int:
    benchmark_json = resolve_benchmark_json(args.benchmark_json)
    benchmark_by_house = benchmark_entries_by_house(benchmark_json, args.max_episodes)
    house_inds = args.house_inds or sorted(benchmark_by_house)
    out_dir = args.output_dir / f"container_target_overlap_first_{args.max_episodes}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {
        "benchmark_json": str(benchmark_json),
        "max_episodes": args.max_episodes,
        "num_houses": len(house_inds),
        "num_inside_objects": 0,
        "num_exact_target_inside_objects": 0,
        "num_candidate_target_inside_objects": 0,
        "houses_with_inside_objects": 0,
        "houses_with_exact_target_inside": 0,
        "houses_with_candidate_target_inside": 0,
        "per_house": [],
    }

    for house_ind in house_inds:
        log.info("Scanning container-target overlap for house %s", house_ind)
        ctx = load_scene_context(args, house_ind)
        try:
            records, containers = collect_scene_records(ctx)
            target_records = [rec for rec in records if is_target_like(rec)]
            house_episodes = benchmark_by_house.get(house_ind, [])
            exact_targets = {
                ep["pickup_obj_name"] for ep in house_episodes if ep.get("pickup_obj_name")
            }
            candidate_targets = {
                name
                for ep in house_episodes
                for name in (ep.get("pickup_obj_candidates") or [])
            }
            house_rows = []
            for container_rec in containers:
                for object_rec in target_records:
                    rel = compute_relation(container_rec, object_rec)
                    if rel["label"] != "inside":
                        continue
                    row = {
                        "house_ind": house_ind,
                        "container_name": container_rec["name"],
                        "container_category": container_rec["category"],
                        "object_name": object_rec["name"],
                        "object_category": object_rec["category"],
                        "is_exact_benchmark_target": object_rec["name"] in exact_targets,
                        "is_candidate_benchmark_target": object_rec["name"] in candidate_targets,
                        "benchmark_episode_indices": [
                            ep["episode_index"]
                            for ep in house_episodes
                            if object_rec["name"] == ep.get("pickup_obj_name")
                            or object_rec["name"] in (ep.get("pickup_obj_candidates") or [])
                        ],
                        "relation": rel,
                    }
                    rows.append(row)
                    house_rows.append(row)

            exact_count = sum(1 for row in house_rows if row["is_exact_benchmark_target"])
            candidate_count = sum(1 for row in house_rows if row["is_candidate_benchmark_target"])
            summary["per_house"].append(
                {
                    "house_ind": house_ind,
                    "num_benchmark_episodes": len(house_episodes),
                    "exact_targets": sorted(exact_targets),
                    "candidate_targets": sorted(candidate_targets),
                    "num_inside_objects": len(house_rows),
                    "num_exact_target_inside_objects": exact_count,
                    "num_candidate_target_inside_objects": candidate_count,
                }
            )
            summary["num_inside_objects"] += len(house_rows)
            summary["num_exact_target_inside_objects"] += exact_count
            summary["num_candidate_target_inside_objects"] += candidate_count
            summary["houses_with_inside_objects"] += int(bool(house_rows))
            summary["houses_with_exact_target_inside"] += int(exact_count > 0)
            summary["houses_with_candidate_target_inside"] += int(candidate_count > 0)
            write_json(out_dir / "inside_objects.partial.json", rows)
            write_json(out_dir / "summary.partial.json", summary)
        finally:
            close_context(ctx)

    write_json(out_dir / "inside_objects.json", rows)
    write_json(out_dir / "summary.json", summary)
    return 0


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


def command_scan_joint_dependencies(args: argparse.Namespace) -> int:
    out_dir = args.output_dir / f"joint_dependencies_house_{args.house_ind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_scene_context(args, args.house_ind)
    try:
        _records, containers = collect_scene_records(ctx)
        doors = collect_door_records(ctx)
        articulation_records = containers + doors
        if args.articulation_name:
            articulation_records = [
                rec for rec in articulation_records if rec["name"] == args.articulation_name
            ]
            if not articulation_records:
                raise RuntimeError(f"Articulation not found: {args.articulation_name}")

        rows = []
        for rec in articulation_records:
            joint_dependencies = infer_joint_open_dependencies(
                ctx.env,
                rec,
                method=args.dependency_method,
                min_overlap_volume=args.min_overlap_volume,
                min_overlap_ratio=args.min_overlap_ratio,
                min_projection_ratio=args.min_projection_ratio,
                min_depth_separation=args.min_depth_separation,
            )
            plot_path = None
            if args.save_plots:
                safe_name = rec["name"].replace("/", "_")
                plot_path = out_dir / "joint_dependency_plots" / f"{safe_name}.png"
                save_joint_dependency_plot(plot_path, rec, joint_dependencies)
            rows.append(
                {
                    "house_ind": args.house_ind,
                    "name": rec["name"],
                    "category": rec.get("category"),
                    "asset_id": rec.get("asset_id"),
                    "interaction_group": rec.get("interaction_group"),
                    "num_joints": len(articulation_joint_records(rec)),
                    "plot_path": None if plot_path is None else str(plot_path),
                    "joint_dependencies": joint_dependencies,
                }
            )

        payload = {
            "house_ind": args.house_ind,
            "dependency_method": args.dependency_method,
            "min_overlap_volume": args.min_overlap_volume,
            "min_overlap_ratio": args.min_overlap_ratio,
            "min_projection_ratio": args.min_projection_ratio,
            "min_depth_separation": args.min_depth_separation,
            "articulations": rows,
        }
        out_path = out_dir / "joint_dependencies.json"
        write_json(out_path, payload)
        log.info("Wrote joint dependency report to %s", out_path)
        return 0
    finally:
        close_context(ctx)


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

    overlap_parser = subparsers.add_parser("scan-container-target-overlap")
    overlap_parser.add_argument("--benchmark_json", type=Path, default=DEFAULT_NAV_BENCHMARK_JSON)
    overlap_parser.add_argument("--max_episodes", type=int, default=100)
    overlap_parser.add_argument("--house_inds", nargs="+", type=int)
    overlap_parser.set_defaults(func=command_scan_container_target_overlap)

    dep_parser = subparsers.add_parser("scan-joint-dependencies")
    dep_parser.add_argument("--house_ind", type=int, required=True)
    dep_parser.add_argument("--articulation_name", type=str)
    dep_parser.add_argument(
        "--dependency_method",
        choices=["front_occlusion", "open_aabb_overlap"],
        default="front_occlusion",
    )
    dep_parser.add_argument("--min_overlap_volume", type=float, default=1e-6)
    dep_parser.add_argument("--min_overlap_ratio", type=float, default=0.0)
    dep_parser.add_argument("--min_projection_ratio", type=float, default=0.15)
    dep_parser.add_argument("--min_depth_separation", type=float, default=0.01)
    dep_parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    dep_parser.set_defaults(func=command_scan_joint_dependencies)

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

    drawer_bound_parser = subparsers.add_parser("debug-drawer-bound-object")
    drawer_bound_parser.add_argument("--house_ind", type=int, required=True)
    drawer_bound_parser.add_argument("--container_name", type=str, required=True)
    drawer_bound_parser.add_argument("--joint_index", type=int, required=True)
    drawer_bound_parser.add_argument("--object_name", type=str)
    drawer_bound_parser.add_argument("--box_padding", type=float, default=0.05)
    drawer_bound_parser.add_argument("--view_distance", type=float, default=0.45)
    drawer_bound_parser.add_argument("--head_tilt_delta", type=float, default=0.25)
    drawer_bound_parser.add_argument("--torso_lean_delta", type=float, default=0.45)
    drawer_bound_parser.set_defaults(func=command_debug_drawer_bound_object)

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
