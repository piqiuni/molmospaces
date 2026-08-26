from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    import cv2 as _cv2
    import networkx as _nx
    import numpy as _np

cv2 = None
nx = None
np = None
dtutils = None
NavToObjBaseConfig = None
FrankaDroidCameraSystem = None
RBY1GoProD455CameraSystem = None
AStarNavToObjPolicyConfig = None
FloatingRUMRobotConfig = None
FrankaRobotConfig = None
RBY1Config = None
Door = None
MlSpacesArticulationObject = None
MjOpenGLRenderer = None
BaseMujocoTaskSampler = None
NavGoalSampler = None
inverse_homogeneous_matrix = None
geom_aabb = None
body_aabb = None
descendant_geoms = None
ProcTHORMap = None
circular_kernel = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import mujoco
except ModuleNotFoundError as exc:
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None

DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts/InteractiveNav/output"
_WALL_COLLISION_SLICE_CACHE: dict[
    tuple[str, float], tuple[np.ndarray, dict[str, Any]]
] = {}
_WALL_COLLISION_SLICE_CACHE_MAX_SCENES = 8


@dataclass
class LoadedContext:
    cfg: Any
    sampler: Any
    task: Any | None

    @property
    def env(self):
        return self.sampler.env


def ensure_runtime_dependencies() -> None:
    global cv2, nx, np, dtutils
    global NavToObjBaseConfig, FrankaDroidCameraSystem, RBY1GoProD455CameraSystem
    global AStarNavToObjPolicyConfig, FloatingRUMRobotConfig, FrankaRobotConfig, RBY1Config
    global Door, MlSpacesArticulationObject, MjOpenGLRenderer, BaseMujocoTaskSampler
    global NavGoalSampler, inverse_homogeneous_matrix, geom_aabb, body_aabb, descendant_geoms
    global ProcTHORMap, circular_kernel

    if mujoco is None:
        raise RuntimeError(
            "mujoco is not available in the current Python environment. "
            "Please activate the MolmoSpaces environment first, e.g. `conda activate mlspaces`, "
            "then re-run this script."
        ) from _MUJOCO_IMPORT_ERROR

    if np is None:
        import cv2 as _cv2
        import networkx as _nx
        import numpy as _np

        import molmo_spaces.utils.distance_transform_utils as _dtutils
        from molmo_spaces.configs.base_nav_to_obj_config import (
            NavToObjBaseConfig as _NavToObjBaseConfig,
        )
        from molmo_spaces.configs.camera_configs import (
            FrankaDroidCameraSystem as _FrankaDroidCameraSystem,
            RBY1GoProD455CameraSystem as _RBY1GoProD455CameraSystem,
        )
        from molmo_spaces.configs.policy_configs import (
            AStarNavToObjPolicyConfig as _AStarNavToObjPolicyConfig,
        )
        from molmo_spaces.configs.robot_configs import (
            FloatingRUMRobotConfig as _FloatingRUMRobotConfig,
            FrankaRobotConfig as _FrankaRobotConfig,
            RBY1Config as _RBY1Config,
        )
        from molmo_spaces.env.data_views import (
            Door as _Door,
            MlSpacesArticulationObject as _MlSpacesArticulationObject,
        )
        from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer as _MjOpenGLRenderer
        from molmo_spaces.tasks.task_sampler import (
            BaseMujocoTaskSampler as _BaseMujocoTaskSampler,
        )
        from molmo_spaces.tasks.util_samplers.navgoal_sampler import (
            NavGoalSampler as _NavGoalSampler,
        )
        from molmo_spaces.utils.linalg_utils import (
            inverse_homogeneous_matrix as _inverse_homogeneous_matrix,
        )
        from molmo_spaces.utils.mj_model_and_data_utils import (
            body_aabb as _body_aabb,
            descendant_geoms as _descendant_geoms,
            geom_aabb as _geom_aabb,
        )
        from molmo_spaces.utils.scene_maps import (
            ProcTHORMap as _ProcTHORMap,
            circular_kernel as _circular_kernel,
        )

        cv2 = _cv2
        nx = _nx
        np = _np
        dtutils = _dtutils
        NavToObjBaseConfig = _NavToObjBaseConfig
        FrankaDroidCameraSystem = _FrankaDroidCameraSystem
        RBY1GoProD455CameraSystem = _RBY1GoProD455CameraSystem
        AStarNavToObjPolicyConfig = _AStarNavToObjPolicyConfig
        FloatingRUMRobotConfig = _FloatingRUMRobotConfig
        FrankaRobotConfig = _FrankaRobotConfig
        RBY1Config = _RBY1Config
        Door = _Door
        MlSpacesArticulationObject = _MlSpacesArticulationObject
        MjOpenGLRenderer = _MjOpenGLRenderer
        BaseMujocoTaskSampler = _BaseMujocoTaskSampler
        NavGoalSampler = _NavGoalSampler
        inverse_homogeneous_matrix = _inverse_homogeneous_matrix
        geom_aabb = _geom_aabb
        body_aabb = _body_aabb
        descendant_geoms = _descendant_geoms
        ProcTHORMap = _ProcTHORMap
        circular_kernel = _circular_kernel


def make_scene_only_task_sampler():
    ensure_runtime_dependencies()

    class SceneOnlyTaskSampler(BaseMujocoTaskSampler):
        """Load a MolmoSpaces scene without sampling a concrete task."""

        def init_scene(self, env) -> None:
            return None

        def randomize_scene(self, env, robot_view) -> None:
            return None

        def _sample_task(self, env):
            raise NotImplementedError("SceneOnlyTaskSampler only loads scenes.")

    return SceneOnlyTaskSampler


def build_config(args: argparse.Namespace, task_mode: str):
    ensure_runtime_dependencies()
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
    cfg.task_sampler_config.house_inds = [args.house_ind]
    cfg.task_sampler_config.samples_per_house = 1
    cfg.task_sampler_config.randomize_lighting = False
    cfg.task_sampler_config.randomize_textures = False
    cfg.task_sampler_config.randomize_dynamics = False
    cfg.policy_config = AStarNavToObjPolicyConfig()

    if task_mode == "scene_only":
        cfg.task_sampler_config.task_sampler_class = make_scene_only_task_sampler()
    if args.target_types:
        cfg.task_sampler_config.pickup_types = args.target_types.split(",")
    benchmark_episode = getattr(args, "benchmark_episode", None)
    if benchmark_episode is not None:
        task_spec = benchmark_episode["task"]
        cfg.scene_dataset = benchmark_episode["scene_dataset"]
        cfg.data_split = benchmark_episode["data_split"]
        cfg.task_sampler_config.house_inds = [benchmark_episode["house_index"]]
        cfg.task_config.robot_base_pose = task_spec.get("robot_base_pose")
        cfg.task_config.pickup_obj_name = task_spec.get("pickup_obj_name")
        cfg.task_config.pickup_obj_candidates = task_spec.get("pickup_obj_candidates")
        succ_pos_threshold = task_spec.get("succ_pos_threshold")
        if succ_pos_threshold is not None:
            cfg.task_config.succ_pos_threshold = succ_pos_threshold

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

    return cfg


def load_context(args: argparse.Namespace, task_mode: str) -> LoadedContext:
    ensure_runtime_dependencies()
    # InteractiveNav collectors operate on already-versioned local scene assets.
    # Disabling resource-manager locks avoids attempts to create cache lock files
    # in read-only/shared caches and mirrors container_scene_probe.load_scene_context.
    import molmo_spaces.tasks.task_sampler as task_sampler_module
    from molmo_spaces.molmo_spaces_constants import get_resource_manager

    manager = get_resource_manager()
    manager.symlink_lock = False
    manager.cache_lock = False
    task_sampler_module.install_scene_with_objects_and_grasps_from_path = lambda *a, **k: {}
    cfg = build_config(args, task_mode=task_mode)
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    task = None
    try:
        if task_mode == "scene_only":
            sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=args.house_ind)
            scene_path = sampler._current_house_scene_path(variant=args.variant)
            if not Path(scene_path).exists():
                from scripts.InteractiveNav.container_scene_probe import (
                    prepare_writable_scene_path,
                )

                scene_path = prepare_writable_scene_path(Path(scene_path))
            sampler.update_scene(scene_path=scene_path, variant=args.variant)
        else:
            task = sampler.sample_task(house_index=args.house_ind, variant=args.variant)
            if task is None:
                raise RuntimeError("Failed to sample nav_to_obj task")
        return LoadedContext(cfg=cfg, sampler=sampler, task=task)
    except Exception:
        sampler.close()
        raise


def close_context(ctx: LoadedContext) -> None:
    ctx.sampler.close()


def scene_category(name: str, category: str | None) -> str:
    lower_name = name.lower()
    lower_cat = (category or "").lower()
    if "door" in lower_name or "door" in lower_cat or "gate" in lower_name:
        return "portal"
    if any(token in lower_name or token in lower_cat for token in ("drawer", "cabinet", "fridge", "refrigerator", "microwave", "oven", "dishwasher", "box")):
        return "container"
    if any(token in lower_name or token in lower_cat for token in ("lamp", "light", "switch", "knob", "button", "curtain", "window")):
        return "device_or_fixture"
    return "other_articulated"


def articulation_summary(obj: MlSpacesArticulationObject, object_manager) -> dict[str, Any]:
    category = object_manager.get_annotation_category(obj.name)
    joints = []
    for idx, joint_name in enumerate(obj.joint_names):
        joints.append(
            {
                "joint_index": idx,
                "joint_name": joint_name,
                "joint_type": str(obj.get_joint_type(idx)).split(".")[-1],
                "joint_range": [float(v) for v in obj.get_joint_range(idx)],
                "joint_position": float(obj.get_joint_position(idx)),
            }
        )
    return {
        "name": obj.name,
        "category": category,
        "interaction_group": scene_category(obj.name, category),
        "joint_count": obj.njoints,
        "joints": joints,
    }


def list_articulated_objects(ctx: LoadedContext) -> dict[str, Any]:
    env = ctx.env
    om = env.object_managers[env.current_batch_index]
    door_names = set(om.find_door_names())
    articulations = []

    for obj in om.list_top_level_objects():
        if not om.is_object_articulable(obj.name):
            continue
        art_obj = om.get_object_by_name(obj.name)
        if not isinstance(art_obj, MlSpacesArticulationObject):
            continue
        rec = articulation_summary(art_obj, om)
        rec["is_named_door"] = art_obj.name in door_names
        articulations.append(rec)

    light_info = []
    for light_id in range(env.current_model.nlight):
        light_info.append(
            {
                "light_id": light_id,
                "light_name": mujoco.mj_id2name(
                    env.current_model, mujoco.mjtObj.mjOBJ_LIGHT, light_id
                ),
                "active": int(env.current_model.light_active[light_id]),
                "position": env.current_model.light_pos[light_id].tolist(),
            }
        )

    return {
        "scene_dataset": ctx.cfg.scene_dataset,
        "data_split": ctx.cfg.data_split,
        "house_ind": ctx.sampler.current_house_index,
        "scene_path": str(env.current_model_path),
        "door_names": sorted(door_names),
        "articulated_objects": articulations,
        "lights": light_info,
    }


def _collect_open_door_root_ids(model: mujoco.MjModel, data: mujoco.MjData, open_threshold: float) -> tuple[set[int], set[int]]:
    parent_to_child: dict[int, list[int]] = {}
    for body_id in range(model.nbody):
        root_body = model.body(model.body(body_id).rootid.item())
        root_body_id = int(root_body.id)
        root_body_name = root_body.name
        if root_body_name and (root_body_name.startswith("door_") or root_body_name.startswith("doorway_")):
            parent_to_child.setdefault(root_body_id, []).append(body_id)

    open_door_ids: set[int] = set()
    doorway_ids: set[int] = set()
    for root_body_id, children in parent_to_child.items():
        for door_body_id in children:
            door_body = model.body(door_body_id)
            jntadr = door_body.jntadr.item()
            if jntadr >= 0 and model.joint(jntadr).type == mujoco.mjtJoint.mjJNT_HINGE:
                qposadr = model.joint(jntadr).qposadr.item()
                if abs(float(data.qpos[qposadr])) > open_threshold:
                    open_door_ids.add(door_body_id)
                    doorway_ids.update(children)
            elif jntadr < 0 and len(children) == 2:
                doorway_ids.add(door_body_id)
    return open_door_ids, doorway_ids


def _collect_doorway_analysis(
    model: mujoco.MjModel, data: mujoco.MjData, open_threshold: float
) -> dict[str, Any]:
    parent_to_child: dict[int, list[int]] = {}
    for body_id in range(model.nbody):
        root_body = model.body(model.body(body_id).rootid.item())
        root_body_id = int(root_body.id)
        root_body_name = root_body.name
        if root_body_name and (
            root_body_name.startswith("door_")
            or root_body_name.startswith("doorway_")
            or root_body_name.startswith("doorframe_")
        ):
            parent_to_child.setdefault(root_body_id, []).append(body_id)

    open_door_ids: set[int] = set()
    doorway_root_ids: set[int] = set()
    non_interactive_root_ids: set[int] = set()
    interactive_door_body_ids: set[int] = set()
    fixed_opening_root_ids: set[int] = set()
    root_records: list[dict[str, Any]] = []

    for root_body_id, children in sorted(parent_to_child.items()):
        root_body = model.body(root_body_id)
        root_body_name = root_body.name
        if root_body_name.startswith("doorframe_"):
            root_kind = "doorframe"
        elif root_body_name.startswith("doorway_"):
            root_kind = "doorway"
        else:
            root_kind = "door"

        hinge_body_ids: list[int] = []
        open_hinge_body_ids: list[int] = []
        no_joint_body_ids: list[int] = []

        for body_id in children:
            body = model.body(body_id)
            jntadr = int(body.jntadr.item())
            if jntadr >= 0 and model.joint(jntadr).type == mujoco.mjtJoint.mjJNT_HINGE:
                hinge_body_ids.append(body_id)
                qposadr = int(model.joint(jntadr).qposadr.item())
                if abs(float(data.qpos[qposadr])) > open_threshold:
                    open_hinge_body_ids.append(body_id)
            elif jntadr < 0:
                no_joint_body_ids.append(body_id)

        interactive = len(hinge_body_ids) > 0
        pipeline_static_passage = (root_kind == "doorway") and (len(children) == 2) and (not interactive)
        fixed_opening = pipeline_static_passage or (
            root_kind == "doorframe" and len(children) == 2 and not interactive
        )
        if interactive:
            interactive_door_body_ids.update(hinge_body_ids)
            if open_hinge_body_ids:
                open_door_ids.update(open_hinge_body_ids)
                doorway_root_ids.add(root_body_id)
        else:
            non_interactive_root_ids.add(root_body_id)
            if fixed_opening:
                doorway_root_ids.add(root_body_id)
                fixed_opening_root_ids.add(root_body_id)

        root_records.append(
            {
                "root_body_id": root_body_id,
                "root_body_name": root_body_name,
                "root_kind": root_kind,
                "interactive": interactive,
                "child_body_names": [model.body(body_id).name for body_id in children],
                "hinge_body_names": [model.body(body_id).name for body_id in hinge_body_ids],
                "open_hinge_body_names": [model.body(body_id).name for body_id in open_hinge_body_ids],
                "no_joint_body_names": [model.body(body_id).name for body_id in no_joint_body_ids],
                "pipeline_static_passage": pipeline_static_passage,
                "fixed_opening": fixed_opening,
            }
        )

    return {
        "open_door_ids": open_door_ids,
        "doorway_root_ids": doorway_root_ids,
        "non_interactive_root_ids": non_interactive_root_ids,
        "interactive_door_body_ids": interactive_door_body_ids,
        "fixed_opening_root_ids": fixed_opening_root_ids,
        "root_records": root_records,
    }


def collect_runtime_doorway_analysis(
    env,
    open_threshold: float = 1e-3,
) -> dict[str, Any]:
    return _collect_doorway_analysis(env.current_model, env.current_data, open_threshold)


def _compile_model_without_ceiling_geoms(model_path: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(model_path)

    ceiling_geoms = []

    def collect_ceiling_geoms_recursively(body_spec: mujoco.MjsBody) -> None:
        for geom in body_spec.geoms:
            geom_name = geom.name
            if geom_name and "ceiling" in geom_name.lower():
                ceiling_geoms.append(geom)
        for child_body in body_spec.bodies:
            collect_ceiling_geoms_recursively(child_body)

    collect_ceiling_geoms_recursively(spec.worldbody)
    for geom in ceiling_geoms:
        spec.delete(geom)

    try:
        return spec.compile()
    finally:
        del spec


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _copy_joint_positions_by_name(
    src_model: mujoco.MjModel,
    src_data: mujoco.MjData,
    dst_model: mujoco.MjModel,
    dst_data: mujoco.MjData,
) -> None:
    for src_joint_id in range(src_model.njnt):
        joint_name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_JOINT, src_joint_id)
        if not joint_name:
            continue

        dst_joint_id = mujoco.mj_name2id(dst_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if dst_joint_id < 0:
            continue

        src_width = _joint_qpos_width(int(src_model.jnt_type[src_joint_id]))
        dst_width = _joint_qpos_width(int(dst_model.jnt_type[dst_joint_id]))
        width = min(src_width, dst_width)
        src_adr = int(src_model.jnt_qposadr[src_joint_id])
        dst_adr = int(dst_model.jnt_qposadr[dst_joint_id])
        dst_data.qpos[dst_adr : dst_adr + width] = src_data.qpos[src_adr : src_adr + width]


def _move_root_free_joint_far_away(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_name: str,
    translation_xyz: np.ndarray | None = None,
) -> bool:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if body_id < 0:
        return False

    if translation_xyz is None:
        translation_xyz = np.array([1000.0, 1000.0, -100.0], dtype=float)

    body = model.body(body_id)
    jntadr = int(body.jntadr.item())
    if jntadr < 0:
        return False
    joint = model.joint(jntadr)
    if joint.type != mujoco.mjtJoint.mjJNT_FREE:
        return False

    qposadr = int(joint.qposadr.item())
    data.qpos[qposadr : qposadr + 3] = np.asarray(translation_xyz, dtype=float)
    data.qpos[qposadr + 3 : qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return True


def _triangle_horizontal_slice_segment(
    triangle_xyz: np.ndarray,
    height_m: float,
    epsilon: float = 1e-6,
) -> np.ndarray | None:
    """Return the longest XY segment where a triangle intersects a Z plane."""
    triangle = np.asarray(triangle_xyz, dtype=float)
    if triangle.shape != (3, 3):
        raise ValueError(f"Expected a (3, 3) triangle, got {triangle.shape}")

    signed = triangle[:, 2] - float(height_m)
    points: list[np.ndarray] = []
    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = triangle[start_index]
        end = triangle[end_index]
        start_signed = float(signed[start_index])
        end_signed = float(signed[end_index])
        start_on_plane = abs(start_signed) <= epsilon
        end_on_plane = abs(end_signed) <= epsilon

        if start_on_plane:
            points.append(start[:2].copy())
        if end_on_plane:
            points.append(end[:2].copy())
        if start_signed * end_signed < -(epsilon * epsilon):
            ratio = start_signed / (start_signed - end_signed)
            points.append((start + ratio * (end - start))[:2])

    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) <= epsilon for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None

    best_pair = None
    best_distance = 0.0
    for first_index in range(len(unique) - 1):
        for second_index in range(first_index + 1, len(unique)):
            distance = float(np.linalg.norm(unique[first_index] - unique[second_index]))
            if distance > best_distance:
                best_distance = distance
                best_pair = (unique[first_index], unique[second_index])
    if best_pair is None or best_distance <= epsilon:
        return None
    return np.asarray(best_pair, dtype=float)


def _mesh_geom_world_vertices(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return None
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None

    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    local_vertices = np.asarray(
        model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=float
    )
    faces = np.asarray(
        model.mesh_face[face_start : face_start + face_count], dtype=np.int64
    )
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    translation = np.asarray(data.geom_xpos[geom_id], dtype=float)
    world_vertices = local_vertices @ rotation.T + translation
    return world_vertices, faces


def _geom_world_vertices_for_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    mesh_geometry = _mesh_geom_world_vertices(model, data, geom_id)
    if mesh_geometry is not None:
        return mesh_geometry[0]

    corners = np.asarray(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    local_aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
    local_corners = local_aabb[:3] + corners * local_aabb[3:]
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    translation = np.asarray(data.geom_xpos[geom_id], dtype=float)
    return local_corners @ rotation.T + translation


def oriented_xy_bounds_for_geoms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
) -> dict[str, Any] | None:
    ensure_runtime_dependencies()
    if not geom_ids:
        return None
    points_xy = np.concatenate(
        [
            _geom_world_vertices_for_bounds(model, data, geom_id)[:, :2]
            for geom_id in geom_ids
        ],
        axis=0,
    )
    if len(points_xy) < 2:
        return None

    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    if tangent[0] < -1e-8 or (abs(tangent[0]) <= 1e-8 and tangent[1] < 0.0):
        tangent = -tangent
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)

    tangent_projection = points_xy @ tangent
    normal_projection = points_xy @ normal
    tangent_min = float(tangent_projection.min())
    tangent_max = float(tangent_projection.max())
    normal_min = float(normal_projection.min())
    normal_max = float(normal_projection.max())
    center_xy = (
        tangent * ((tangent_min + tangent_max) / 2.0)
        + normal * ((normal_min + normal_max) / 2.0)
    )
    return {
        "center_xy": center_xy,
        "tangent_xy": tangent,
        "normal_xy": normal,
        "width_m": tangent_max - tangent_min,
        "thickness_m": normal_max - normal_min,
    }


def collect_wall_collision_slice_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height_m: float = 0.45,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Slice thin ProcTHOR wall collision meshes without filling their door holes."""
    ensure_runtime_dependencies()
    segments: list[np.ndarray] = []
    wall_geom_count = 0
    sliced_triangle_count = 0
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name or ""
        if not body_name.startswith("wall_") or "collision" not in geom_name.lower():
            continue
        mesh_geometry = _mesh_geom_world_vertices(model, data, geom_id)
        if mesh_geometry is None:
            continue
        wall_geom_count += 1
        world_vertices, faces = mesh_geometry
        for face in faces:
            segment = _triangle_horizontal_slice_segment(
                world_vertices[np.asarray(face, dtype=np.int64)], height_m
            )
            if segment is None:
                continue
            segments.append(segment)
            sliced_triangle_count += 1

    segment_array = (
        np.asarray(segments, dtype=float)
        if segments
        else np.empty((0, 2, 2), dtype=float)
    )
    return segment_array, {
        "height_m": float(height_m),
        "wall_collision_geom_count": int(wall_geom_count),
        "slice_segment_count": int(len(segment_array)),
        "sliced_triangle_count": int(sliced_triangle_count),
    }


def cached_wall_collision_slice_segments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    model_path: str | None,
    height_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_key = None
    if model_path is not None:
        cache_key = (str(Path(model_path).resolve()), round(float(height_m), 4))
        cached = _WALL_COLLISION_SLICE_CACHE.get(cache_key)
        if cached is not None:
            segments, stats = cached
            return segments, {**stats, "cache_hit": True}

    segments, stats = collect_wall_collision_slice_segments(
        model, data, height_m=height_m
    )
    if cache_key is not None:
        if len(_WALL_COLLISION_SLICE_CACHE) >= _WALL_COLLISION_SLICE_CACHE_MAX_SCENES:
            oldest_key = next(iter(_WALL_COLLISION_SLICE_CACHE))
            _WALL_COLLISION_SLICE_CACHE.pop(oldest_key)
        _WALL_COLLISION_SLICE_CACHE[cache_key] = (segments, dict(stats))
    return segments, {**stats, "cache_hit": False}


def rasterize_world_xy_segments(
    segments_xy: np.ndarray,
    world_to_map: np.ndarray,
    shape: tuple[int, int],
    *,
    height_m: float = 0.0,
    thickness_px: int = 1,
) -> np.ndarray:
    ensure_runtime_dependencies()
    mask = np.zeros(shape, dtype=np.uint8)
    for segment in np.asarray(segments_xy, dtype=float):
        homogeneous = np.column_stack(
            [segment, np.full(2, float(height_m)), np.ones(2, dtype=float)]
        )
        pixels = homogeneous @ np.asarray(world_to_map, dtype=float).T
        start = (int(round(float(pixels[0, 1]))), int(round(float(pixels[0, 0]))))
        end = (int(round(float(pixels[1, 1]))), int(round(float(pixels[1, 0]))))
        cv2.line(
            mask,
            start,
            end,
            color=1,
            thickness=max(1, int(thickness_px)),
            lineType=cv2.LINE_8,
        )
    return mask.astype(bool)


def build_live_procthor_map(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    model_path: str | None = None,
    px_per_m: int = 200,
    agent_radius: float | None = None,
    open_threshold: float = 1e-3,
    device_id: int | None = None,
    treat_all_non_interactive_doorways_as_open: bool = False,
    return_doorway_analysis: bool = False,
    ignored_root_body_names: set[str] | None = None,
    include_wall_collision_slices: bool = True,
    wall_slice_height_m: float = 0.45,
    wall_slice_thickness_px: int = 1,
    doorway_clearance_m: float = 0.30,
) -> ProcTHORMap | tuple[ProcTHORMap, dict[str, Any] | None]:
    ensure_runtime_dependencies()
    work_model = model
    work_data = data
    owns_work_model = False

    if model_path is not None:
        work_model = _compile_model_without_ceiling_geoms(model_path)
        work_data = mujoco.MjData(work_model)
        _copy_joint_positions_by_name(model, data, work_model, work_data)
        if ignored_root_body_names:
            ignored_moved = []
            for root_body_name in ignored_root_body_names:
                if _move_root_free_joint_far_away(work_model, work_data, root_body_name):
                    ignored_moved.append(root_body_name)
            if ignored_moved:
                log.info("Moved %d ignored movable roots out of scene for occupancy: %s", len(ignored_moved), ignored_moved[:8])
        mujoco.mj_forward(work_model, work_data)
        owns_work_model = True

    floor_ids = []
    room_ids_to_name = {}
    for geom_id in range(work_model.ngeom):
        geom_name = mujoco.mj_id2name(work_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and (geom_name.startswith("room|") or geom_name.startswith("room_")):
            floor_ids.append(geom_id)
            room_body_id = work_model.geom(geom_id).bodyid.item()
            room_ids_to_name[geom_id + 1] = work_model.body(room_body_id).name

    if not floor_ids:
        raise ValueError("No floors found in the live model.")

    doorway_analysis: dict[str, Any] | None = None
    if treat_all_non_interactive_doorways_as_open:
        doorway_analysis = _collect_doorway_analysis(work_model, work_data, open_threshold)
        open_door_ids = doorway_analysis["open_door_ids"]
        doorway_ids = doorway_analysis["doorway_root_ids"]
    else:
        open_door_ids, doorway_ids = _collect_open_door_root_ids(
            work_model, work_data, open_threshold
        )

    doorframe_geom_ids = []
    door_geom_ids = []
    for geom_id in range(work_model.ngeom):
        body_id = work_model.geom(geom_id).bodyid.item()
        parent_body_id = work_model.body(body_id).parentid.item()
        if body_id in open_door_ids or parent_body_id in open_door_ids:
            door_geom_ids.append(geom_id)
        root_body_id = work_model.body(body_id).rootid.item()
        if root_body_id in doorway_ids:
            doorframe_geom_ids.append(geom_id)

    aabb_center, aabb_size = geom_aabb(work_model, work_data, floor_ids, tight_mesh=False)
    aabb_size += np.array([2, 2, 0])

    def render_occupancy(cam_distance: float):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = aabb_center
        cam.distance = cam_distance
        cam.azimuth = 0
        cam.elevation = -90
        cam.orthographic = 1

        h = round(px_per_m * aabb_size[0])
        w = round(px_per_m * aabb_size[1])
        effective_px = h / aabb_size[0]

        renderer = MjOpenGLRenderer(model=work_model, height=h, width=w, device_id=device_id)
        renderer.update(work_data, cam)
        for camera in renderer.scene.camera:
            camera.orthographic = 1
            camera.frustum_bottom = -aabb_size[0] / 2
            camera.frustum_top = aabb_size[0] / 2

        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        seg_geom = seg[..., 0]
        cam_to_world = None
        if cam_distance == 5.0:
            cam_to_world = np.eye(4)
            cam_to_world[:3, 3] = renderer.scene.camera[0].pos
            camera_x_ax = np.cross(renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
            cam_to_world[:3, :3] = np.column_stack(
                (camera_x_ax, renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
            )
        renderer.close()

        occ_room_floor = np.zeros_like(seg_geom, dtype=int)
        for fid in floor_ids:
            occ_room_floor[seg_geom == fid] = fid + 1

        occ_floor = np.ones_like(seg_geom, dtype=bool)
        for fid in floor_ids:
            occ_floor &= seg_geom != fid

        occ_door = np.zeros_like(seg_geom, dtype=bool)
        for did in door_geom_ids:
            occ_door[seg_geom == did] = True

        occ_doorframe = np.zeros_like(seg_geom, dtype=bool)
        for did in doorframe_geom_ids:
            occ_doorframe[seg_geom == did] = True

        occ_door_path = occ_doorframe & ~occ_door
        doorway_clearance_px = max(1, int(round(doorway_clearance_m * effective_px)))
        occ_door_path = cv2.dilate(
            occ_door_path.astype(np.uint8),
            circular_kernel(doorway_clearance_px),
        ).astype(bool)

        occ = occ_floor.copy()
        occ[occ_door_path == 1] = False
        # The carve removes the top-down door-frame/lintel projection. Keep the
        # actually swung-open leaf as a collision obstacle if it overlaps the
        # carved portal region.
        occ[occ_door] = True

        if cam_distance == 5.0:
            return occ, occ_room_floor, effective_px, (h, w), cam_to_world
        return occ, occ_room_floor, effective_px, (h, w)

    occ_map_5, room_map_5, effective_px, (h, w), cam_to_world = render_occupancy(5.0)
    occ_final = occ_map_5.copy()
    room_map_final = room_map_5.copy()

    cam_to_map = np.array([[0, -effective_px, 0, h / 2], [effective_px, 0, 0, w / 2]])
    world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)

    map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
    centered_to_cam = np.array([[1 / effective_px, 0, 0], [0, 1 / effective_px, 0], [0, 0, 1]])
    cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
    cam_to_world_floor[2, 2] = 0
    map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered

    wall_slice_mask = np.zeros_like(occ_final, dtype=bool)
    wall_slice_stats = {
        "enabled": bool(include_wall_collision_slices),
        "height_m": float(wall_slice_height_m),
        "wall_collision_geom_count": 0,
        "slice_segment_count": 0,
        "sliced_triangle_count": 0,
        "rasterized_pixel_count": 0,
    }
    if include_wall_collision_slices:
        wall_segments, collected_stats = cached_wall_collision_slice_segments(
            work_model,
            work_data,
            model_path=model_path,
            height_m=wall_slice_height_m,
        )
        wall_slice_mask = rasterize_world_xy_segments(
            wall_segments,
            world_to_map,
            occ_final.shape,
            height_m=wall_slice_height_m,
            thickness_px=wall_slice_thickness_px,
        )
        occ_final |= wall_slice_mask
        room_map_final[wall_slice_mask] = 0
        wall_slice_stats.update(collected_stats)
        wall_slice_stats["rasterized_pixel_count"] = int(wall_slice_mask.sum())

    if agent_radius is not None:
        rad_px = int(agent_radius * effective_px)
        kernel = circular_kernel(rad_px)
        occ_final = cv2.dilate(occ_final.astype(np.uint8), kernel).astype(bool)
        room_map_final[occ_final] = 0

    occ_final = ~occ_final
    if not np.any(occ_final) or np.all(occ_final):
        raise RuntimeError(
            "build_live_procthor_map produced a degenerate occupancy map "
            f"(all_free={bool(np.all(occ_final))}, all_blocked={bool(not np.any(occ_final))})."
        )

    instance = ProcTHORMap(
        occupancy=occ_final,
        room_map=room_map_final,
        room_ids_to_name=room_ids_to_name,
        world_to_map=world_to_map,
        map_to_world=map_to_world,
        px_per_m=effective_px,
    )
    instance.occupancy_rendered_base = occ_map_5
    instance.occupancy_wall_slice_mask = wall_slice_mask
    instance.occupancy_base = occ_map_5 | wall_slice_mask
    instance.wall_slice_stats = wall_slice_stats

    if owns_work_model:
        del work_data
        del work_model

    gc.collect()
    if return_doorway_analysis:
        return instance, doorway_analysis
    return instance


def render_topdown_geom_mask(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: list[int],
    px_per_m: int = 200,
    device_id: int | None = None,
) -> tuple[np.ndarray, float]:
    ensure_runtime_dependencies()
    floor_ids = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and (geom_name.startswith("room|") or geom_name.startswith("room_")):
            floor_ids.append(geom_id)
    if not floor_ids:
        raise ValueError("No floors found in the live model.")

    aabb_center, aabb_size = geom_aabb(model, data, floor_ids, tight_mesh=False)
    aabb_size += np.array([2, 2, 0])

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = aabb_center
    cam.distance = 5.0
    cam.azimuth = 0
    cam.elevation = -90
    cam.orthographic = 1

    h = round(px_per_m * aabb_size[0])
    w = round(px_per_m * aabb_size[1])
    effective_px = h / aabb_size[0]

    renderer = MjOpenGLRenderer(model=model, height=h, width=w, device_id=device_id)
    renderer.update(data, cam)
    for camera in renderer.scene.camera:
        camera.orthographic = 1
        camera.frustum_bottom = -aabb_size[0] / 2
        camera.frustum_top = aabb_size[0] / 2
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.close()

    seg_geom = seg[..., 0]
    mask = np.zeros_like(seg_geom, dtype=bool)
    for geom_id in geom_ids:
        mask |= seg_geom == geom_id
    return mask, effective_px


def compute_path_from_map(
    scene_map: ProcTHORMap,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    downscale_factor: int = 5,
    max_start_goal_distance: int = 40,
) -> np.ndarray | None:
    grid = scene_map.occupancy.astype(bool)
    padded = np.zeros(
        (
            grid.shape[0] + (downscale_factor - grid.shape[0] % downscale_factor),
            grid.shape[1] + (downscale_factor - grid.shape[1] % downscale_factor),
        ),
        dtype=bool,
    )
    padded[: grid.shape[0], : grid.shape[1]] = grid
    downscaled_grid = (
        padded.reshape(
            padded.shape[0] // downscale_factor,
            downscale_factor,
            padded.shape[1] // downscale_factor,
            downscale_factor,
        )
        .min(axis=1)
        .min(axis=-1)
    )

    grid_spacing = downscale_factor / scene_map.px_per_m
    dt = dtutils.make_distance_transform(downscaled_grid, grid_spacing)
    graph = dtutils.make_grid_graph(downscaled_grid, dt, weight_exp=2)

    def discretize(location_xy: np.ndarray) -> tuple[int, int]:
        px = scene_map.pos_m_to_px(np.array([location_xy[0], location_xy[1], 0.0]))
        rc = np.floor(px / downscale_factor).astype(np.int32)
        return int(rc[0]), int(rc[1])

    def find_close(missing: tuple[int, int]) -> tuple[int, int] | None:
        for search_range in range(1, max_start_goal_distance + 1):
            for shiftr in range(-search_range, search_range + 1):
                for shiftc in range(-search_range, search_range + 1):
                    if shiftr != search_range and shiftc != search_range:
                        continue
                    candidate = (missing[0] + shiftr, missing[1] + shiftc)
                    if candidate in graph:
                        return candidate
        return None

    start = discretize(start_xy)
    goal = discretize(goal_xy)
    if start not in graph:
        start = find_close(start)
    if goal not in graph:
        goal = find_close(goal)
    if start is None or goal is None:
        return None

    try:
        waypoints, _, _ = dtutils.make_discrete_path(
            graph, start[0], start[1], goal[0], goal[1], dt, 3, grid_spacing, 0.6
        )
    except nx.NetworkXUnfeasible:
        return None

    pixel_waypoints = np.array(waypoints) * downscale_factor
    return scene_map.pos_px_to_m(pixel_waypoints)[:, :2]


def safe_body_aabb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return body_aabb(model, data, body_id, visual_only=True)
    except Exception as exc:
        log.debug("Failed to compute visual AABB for body %s: %s", body_id, exc)
        return np.asarray(data.xpos[body_id]).copy(), np.zeros(3, dtype=float)


def joint_type_name(joint_type: Any) -> str:
    text = str(joint_type).lower()
    if "hinge" in text:
        return "hinge"
    if "slide" in text:
        return "slide"
    if "free" in text:
        return "free"
    return "none"


def scene_object_record(env, om, object_name: str, meta: dict[str, Any], force_door: bool = False) -> dict[str, Any]:
    model = env.current_model
    data = env.current_data
    body_id = model.body(object_name).id
    center, size = safe_body_aabb(model, data, body_id)
    joints = meta.get("name_map", {}).get("joints", {})
    sites = meta.get("name_map", {}).get("sites", {})
    joint_infos = []
    for joint_name in sorted(joints.keys()):
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            joint_infos.append(
                {
                    "joint_name": str(joint_name),
                    "joint_type": joint_type_name(model.joint(joint_id).type),
                    "joint_range": [float(v) for v in model.joint(joint_id).range],
                }
            )
        except Exception as exc:
            log.debug("Failed to gather joint info for %s on %s: %s", joint_name, object_name, exc)
            continue

    return {
        "name": object_name,
        "body_id": int(body_id),
        "_model": model,
        "_data": data,
        "object_id": meta.get("object_id"),
        "asset_id": meta.get("asset_id"),
        "category": "Door" if force_door else meta.get("category"),
        "room_id": meta.get("room_id"),
        "parent": meta.get("parent") or None,
        "children": meta.get("children", []),
        "is_static": bool(meta.get("is_static", False)),
        "is_structural": False if force_door else bool(om.is_structural(object_name)),
        "is_receptacle": bool(om.has_receptacle_site(object_name)),
        "is_pickup_candidate": bool(om.has_free_joint(object_name)),
        "is_articulable": bool(om.is_object_articulable(object_name)),
        "is_door": bool(force_door or str(object_name).lower().startswith(("door_", "doorway_", "doorframe_"))),
        "is_movable_door": bool(force_door),
        "joint_infos": joint_infos,
        "position": np.asarray(data.xpos[body_id]).copy(),
        "aabb_center": np.asarray(center).copy(),
        "aabb_size": np.asarray(size).copy(),
    }


def door_parent_metadata(objects_meta: dict[str, dict[str, Any]], door_body_name: str) -> dict[str, Any]:
    for object_name, meta in objects_meta.items():
        bodies = meta.get("name_map", {}).get("bodies", {})
        if door_body_name in bodies:
            door_meta = dict(meta)
            door_meta["parent"] = object_name
            door_meta["children"] = []
            return door_meta
    return {
        "category": "Door",
        "object_id": door_body_name,
        "asset_id": None,
        "room_id": None,
        "parent": None,
        "children": [],
        "is_static": True,
        "name_map": {"joints": {}, "sites": {}},
    }


def collect_scene_plot_records(env) -> list[dict[str, Any]]:
    om = env.object_managers[env.current_batch_index]
    objects_meta = env.current_scene_metadata.get("objects", {})
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for object_name, meta in objects_meta.items():
        try:
            env.current_model.body(object_name)
        except KeyError:
            continue
        records.append(scene_object_record(env, om, object_name, meta))
        seen_names.add(object_name)

    try:
        doorway_analysis = collect_runtime_doorway_analysis(env)
        for rec in collect_interactive_door_root_object_records(env, doorway_analysis):
            if rec["name"] in seen_names:
                continue
            records.append(rec)
            seen_names.add(rec["name"])
    except Exception as exc:
        log.warning("Could not enumerate interactive door roots for plotting: %s", exc)

    records.sort(key=lambda item: (str(item.get("room_id")), str(item.get("category")), item["name"]))
    return records


def collect_non_interactive_doorway_object_records(
    env,
    doorway_analysis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if doorway_analysis is None:
        return []

    records: list[dict[str, Any]] = []
    for rec in doorway_analysis.get("root_records", []):
        if rec.get("interactive", True):
            continue
        root_body_name = rec.get("root_body_name")
        if not root_body_name:
            continue
        try:
            body_id = env.current_model.body(root_body_name).id
        except KeyError:
            continue

        center, size = safe_body_aabb(env.current_model, env.current_data, body_id)
        root_kind = str(rec.get("root_kind", "doorway"))
        category = "Doorframe" if root_kind == "doorframe" else "Doorway"
        records.append(
            {
                "name": root_body_name,
                "body_id": int(body_id),
                "_model": env.current_model,
                "_data": env.current_data,
                "object_id": root_body_name,
                "asset_id": None,
                "category": category,
                "room_id": None,
                "parent": None,
                "children": rec.get("child_body_names", []),
                "hinge_body_names": rec.get("hinge_body_names", []),
                "is_static": True,
                "is_structural": False,
                "is_receptacle": False,
                "is_pickup_candidate": False,
                "is_articulable": False,
                "is_door": True,
                "is_movable_door": False,
                "is_fixed_opening": bool(rec.get("fixed_opening", False)),
                "is_noninteractive_doorway": True,
                "record_type": f"non_interactive_{root_kind}",
                "position": np.asarray(env.current_data.xpos[body_id]).copy(),
                "aabb_center": np.asarray(center).copy(),
                "aabb_size": np.asarray(size).copy(),
            }
        )

    records.sort(key=lambda item: item["name"])
    return records


def collect_interactive_door_root_object_records(
    env,
    doorway_analysis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if doorway_analysis is None:
        return []

    records: list[dict[str, Any]] = []
    for rec in doorway_analysis.get("root_records", []):
        if not rec.get("interactive", False):
            continue
        root_body_name = rec.get("root_body_name")
        if not root_body_name:
            continue
        try:
            body_id = env.current_model.body(root_body_name).id
        except KeyError:
            continue

        center, size = safe_body_aabb(env.current_model, env.current_data, body_id)
        frame_candidates = []
        for frame_body_name in rec.get("no_joint_body_names", []):
            try:
                frame_body_id = env.current_model.body(frame_body_name).id
            except KeyError:
                continue
            direct_geom_ids = [
                geom_id
                for geom_id in range(env.current_model.ngeom)
                if int(env.current_model.geom_bodyid[geom_id]) == int(frame_body_id)
            ]
            visual_geom_ids = [
                geom_id
                for geom_id in direct_geom_ids
                if "visual"
                in (
                    mujoco.mj_id2name(
                        env.current_model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    )
                    or ""
                ).lower()
            ]
            selected_geom_ids = visual_geom_ids or direct_geom_ids
            if not selected_geom_ids:
                continue
            frame_center, frame_size = geom_aabb(
                env.current_model,
                env.current_data,
                selected_geom_ids,
                tight_mesh=True,
            )
            oriented = oriented_xy_bounds_for_geoms(
                env.current_model,
                env.current_data,
                selected_geom_ids,
            )
            if oriented is None:
                continue
            major_size = float(oriented["width_m"])
            minor_size = float(oriented["thickness_m"])
            if major_size < 0.4:
                continue
            frame_candidates.append(
                {
                    "body_name": frame_body_name,
                    "center": np.asarray(frame_center, dtype=float),
                    "size": np.asarray(frame_size, dtype=float),
                    "portal": oriented,
                    "score": (
                        major_size / max(minor_size, 0.02),
                        major_size,
                        -minor_size,
                    ),
                }
            )

        if frame_candidates:
            portal_frame = max(frame_candidates, key=lambda item: item["score"])
            portal_center_xy = portal_frame["portal"]["center_xy"]
            portal_tangent_xy = portal_frame["portal"]["tangent_xy"]
            portal_normal_xy = portal_frame["portal"]["normal_xy"]
            portal_width_m = float(portal_frame["portal"]["width_m"])
            portal_thickness_m = float(portal_frame["portal"]["thickness_m"])
            portal_frame_body_name = str(portal_frame["body_name"])
        else:
            portal_center_xy = np.asarray(center, dtype=float)[:2]
            portal_size_xy = np.asarray(size, dtype=float)[:2]
            portal_major_axis = int(np.argmax(portal_size_xy))
            portal_tangent_xy = np.zeros(2, dtype=float)
            portal_tangent_xy[portal_major_axis] = 1.0
            portal_normal_xy = np.asarray(
                [-portal_tangent_xy[1], portal_tangent_xy[0]], dtype=float
            )
            portal_width_m = float(portal_size_xy[portal_major_axis])
            portal_thickness_m = float(portal_size_xy[1 - portal_major_axis])
            portal_frame_body_name = None
        records.append(
            {
                "name": root_body_name,
                "body_id": int(body_id),
                "_model": env.current_model,
                "_data": env.current_data,
                "object_id": root_body_name,
                "asset_id": None,
                "category": "Door",
                "room_id": None,
                "parent": None,
                "children": rec.get("child_body_names", []),
                "hinge_body_names": rec.get("hinge_body_names", []),
                "is_static": True,
                "is_structural": False,
                "is_receptacle": False,
                "is_pickup_candidate": False,
                "is_articulable": True,
                "is_door": True,
                "is_movable_door": True,
                "interactive_root_name": root_body_name,
                "position": np.asarray(env.current_data.xpos[body_id]).copy(),
                "aabb_center": np.asarray(center).copy(),
                "aabb_size": np.asarray(size).copy(),
                "portal_frame_body_name": portal_frame_body_name,
                "portal_center_xy": np.asarray(portal_center_xy, dtype=float),
                "portal_tangent_xy": portal_tangent_xy,
                "portal_normal_xy": portal_normal_xy,
                "portal_half_width_m": portal_width_m / 2.0,
                "portal_half_thickness_m": portal_thickness_m / 2.0,
            }
        )

    records.sort(key=lambda item: item["name"])
    return records


def dedupe_plot_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records:
        name = str(rec.get("name", ""))
        if name in seen:
            continue
        seen.add(name)
        deduped.append(rec)
    return deduped


def path_length(path: np.ndarray | None) -> float | None:
    if path is None or len(path) < 2:
        return None
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def image_output_paths(output_json: Path) -> tuple[Path, Path, Path]:
    stem = output_json.stem
    return (
        output_json.with_name(f"{stem}_map.png"),
        output_json.with_name(f"{stem}_baseline.png"),
        output_json.with_name(f"{stem}_compare.png"),
    )


def map_compare_output_paths(output_json: Path) -> tuple[Path, Path, Path]:
    stem = output_json.stem
    return (
        output_json.with_name(f"{stem}_closed.png"),
        output_json.with_name(f"{stem}_cached.png"),
        output_json.with_name(f"{stem}_open.png"),
    )


def movable_clear_output_path(output_json: Path) -> Path:
    stem = output_json.stem
    return output_json.with_name(f"{stem}_movable-cleared.png")


def door_path_study_output_paths(output_json: Path, door_names: list[str]) -> tuple[Path, list[Path]]:
    stem = output_json.stem
    baseline_path = output_json.with_name(f"{stem}_all-open.png")
    compare_paths = []
    for idx, door_name in enumerate(door_names):
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", door_name)
        compare_paths.append(output_json.with_name(f"{stem}_close-{idx:02d}_{safe_name}.png"))
    return baseline_path, compare_paths


def door_path_study_all_closed_output_path(output_json: Path) -> Path:
    stem = output_json.stem
    return output_json.with_name(f"{stem}_all-closed.png")


def local_patch_debug_output_path(output_json: Path, object_name: str) -> Path:
    stem = output_json.stem
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", object_name)
    return output_json.with_name(f"{stem}_local-patch_{safe_name}.png")


def points_xy_to_px(scene_map: ProcTHORMap, points_xy: np.ndarray | None) -> np.ndarray | None:
    if points_xy is None:
        return None
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or len(points_xy) == 0:
        return None
    points_xyz = np.column_stack([points_xy, np.zeros(len(points_xy), dtype=float)])
    return scene_map.pos_m_to_px(points_xyz)


def make_scene_plot_background(scene_map: ProcTHORMap) -> np.ndarray:
    free_mask = np.asarray(scene_map.occupancy).astype(bool)
    bg = np.ones(free_mask.shape + (3,), dtype=float)
    bg[:] = np.array([0.96, 0.95, 0.92], dtype=float)
    bg[~free_mask] = np.array([0.18, 0.20, 0.22], dtype=float)

    room_map = getattr(scene_map, "room_map", None)
    if room_map is not None:
        contours = np.zeros(free_mask.shape, dtype=bool)
        contours[1:, :] |= room_map[1:, :] != room_map[:-1, :]
        contours[:, 1:] |= room_map[:, 1:] != room_map[:, :-1]
        contours &= room_map > 0
        bg[contours] = np.array([0.72, 0.76, 0.82], dtype=float)

    return bg


def scene_visual_kind(rec: dict[str, Any]) -> str:
    if rec.get("is_movable_door", False):
        return "movable_door"
    if rec.get("is_fixed_opening", False):
        return "fixed_opening"
    if rec.get("is_noninteractive_doorway", False):
        return "noninteractive_doorway"
    if rec.get("is_pickup_candidate", False):
        return "pickup"
    if rec.get("is_receptacle", False):
        return "receptacle"
    if rec.get("is_articulable", False):
        return "articulable"
    if rec.get("is_structural", False):
        return "structural"
    return "plain"


def scene_plot_color(kind: str) -> str:
    return {
        "structural": "#6b7280",
        "receptacle": "#2563eb",
        "pickup": "#f97316",
        "articulable": "#dc2626",
        "movable_door": "#7c3aed",
        "fixed_opening": "#0ea5e9",
        "noninteractive_doorway": "#f97316",
        "plain": "#0f766e",
    }[kind]


def scene_plot_label(rec: dict[str, Any]) -> str:
    category = str(rec.get("category") or rec["name"])
    if rec.get("is_movable_door", False):
        return f"door: {category}"
    if rec.get("is_fixed_opening", False):
        return f"doorframe: {category}"
    if rec.get("is_noninteractive_doorway", False):
        return f"doorway: {category}"
    if rec.get("is_pickup_candidate", False):
        return f"pickup: {category}"
    if rec.get("is_receptacle", False):
        return f"receptacle: {category}"
    if rec.get("is_articulable", False):
        return category
    return category


def object_box_to_px(scene_map: ProcTHORMap, rec: dict[str, Any]) -> tuple[float, float, float, float] | None:
    center = np.asarray(rec["aabb_center"], dtype=float)
    size = np.asarray(rec["aabb_size"], dtype=float)
    if np.any(size[:2] <= 1e-6):
        center = np.asarray(rec["position"], dtype=float)
        size = np.array([0.15, 0.15, 0.0], dtype=float)
    corners_xy = np.asarray(
        [
            [center[0] - size[0] / 2.0, center[1] - size[1] / 2.0],
            [center[0] + size[0] / 2.0, center[1] - size[1] / 2.0],
            [center[0] + size[0] / 2.0, center[1] + size[1] / 2.0],
            [center[0] - size[0] / 2.0, center[1] + size[1] / 2.0],
        ],
        dtype=float,
    )
    corners_px = points_xy_to_px(scene_map, corners_xy)
    if corners_px is None or len(corners_px) != 4:
        return None
    rows = corners_px[:, 0]
    cols = corners_px[:, 1]
    col_min = float(np.min(cols))
    col_max = float(np.max(cols))
    row_min = float(np.min(rows))
    row_max = float(np.max(rows))
    return col_min, row_min, col_max - col_min, row_max - row_min


def pick_plot_scene_map(primary_map: ProcTHORMap, fallback_map: ProcTHORMap | None, use_fallback: bool = False) -> ProcTHORMap:
    primary_occ = np.asarray(primary_map.occupancy).astype(bool)
    if 0.0 < float(np.mean(primary_occ)) < 1.0:
        return primary_map
    if fallback_map is not None and use_fallback:
        fallback_occ = np.asarray(fallback_map.occupancy).astype(bool)
        if 0.0 < float(np.mean(fallback_occ)) < 1.0:
            return fallback_map
    return primary_map


def nearest_free_point_xy(
    scene_map: ProcTHORMap, target_xy: np.ndarray, max_radius_px: int = 300
) -> np.ndarray | None:
    occ = np.asarray(scene_map.occupancy).astype(bool)
    target_px = points_xy_to_px(scene_map, np.asarray([target_xy]))
    if target_px is None:
        return None
    row, col = map(int, target_px[0])
    h, w = occ.shape
    if 0 <= row < h and 0 <= col < w and occ[row, col]:
        return np.asarray(target_xy, dtype=float)

    for radius in range(1, max_radius_px + 1):
        r0 = max(0, row - radius)
        r1 = min(h, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(w, col + radius + 1)

        candidates = []
        if r0 < r1 and c0 < c1:
            top = np.argwhere(occ[r0, c0:c1]) + np.array([r0, c0])
            bottom = np.argwhere(occ[r1 - 1, c0:c1]) + np.array([r1 - 1, c0])
            left = np.argwhere(occ[r0:r1, c0]) + np.array([r0, c0])
            right = np.argwhere(occ[r0:r1, c1 - 1]) + np.array([r0, c1 - 1])
            for arr in (top, bottom, left, right):
                if len(arr) > 0:
                    candidates.append(arr)
        if not candidates:
            continue
        all_candidates = np.unique(np.concatenate(candidates, axis=0), axis=0)
        dists = np.linalg.norm(all_candidates - np.array([row, col]), axis=1)
        best_px = all_candidates[int(np.argmin(dists))]
        best_world = scene_map.pos_px_to_m(best_px.reshape(1, 2))[0]
        return np.asarray(best_world[:2], dtype=float)
    return None


def removable_obstacle_overlap_stats(
    scene_map: ProcTHORMap,
    rec: dict[str, Any],
    padding_px: int = 4,
) -> dict[str, Any] | None:
    box = object_box_to_px(scene_map, rec)
    if box is None:
        return None
    col, row, width, height = box
    occ_free = np.asarray(scene_map.occupancy).astype(bool)
    h, w = occ_free.shape
    r0 = max(0, int(np.floor(row)) - padding_px)
    r1 = min(h, int(np.ceil(row + height)) + padding_px)
    c0 = max(0, int(np.floor(col)) - padding_px)
    c1 = min(w, int(np.ceil(col + width)) + padding_px)
    if r0 >= r1 or c0 >= c1:
        return None

    free_window = occ_free[r0:r1, c0:c1]
    blocked_mask = ~free_window
    blocked_pixels = int(np.count_nonzero(blocked_mask))
    total_pixels = int(blocked_mask.size)
    if blocked_pixels == 0:
        return {
            "bbox_px": [c0, r0, c1, r1],
            "blocked_pixels": 0,
            "blocked_ratio": 0.0,
            "neighbor_component_count": 0,
            "neighbor_component_labels": [],
        }

    dilated = cv2.dilate(blocked_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    neighbor_free = free_window & (~dilated.astype(bool))
    num_labels, labels = cv2.connectedComponents(neighbor_free.astype(np.uint8))
    touching_labels: set[int] = set()
    for rr, cc in np.argwhere(blocked_mask):
        rr0 = max(0, rr - 1)
        rr1 = min(labels.shape[0], rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(labels.shape[1], cc + 2)
        touching_labels.update(int(v) for v in np.unique(labels[rr0:rr1, cc0:cc1]) if int(v) > 0)

    return {
        "bbox_px": [c0, r0, c1, r1],
        "blocked_pixels": blocked_pixels,
        "blocked_ratio": float(blocked_pixels / max(total_pixels, 1)),
        "neighbor_component_count": len(touching_labels),
        "neighbor_component_labels": sorted(touching_labels),
    }


def local_patch_connectivity_stats(
    scene_map: ProcTHORMap,
    rec: dict[str, Any],
    margin_px: int = 24,
    dilation_px: int = 6,
) -> dict[str, Any] | None:
    box = object_box_to_px(scene_map, rec)
    if box is None:
        return None

    col, row, width, height = box
    occ_free = np.asarray(scene_map.occupancy).astype(bool)
    h, w = occ_free.shape

    obj_c0 = max(0, int(np.floor(col)))
    obj_c1 = min(w, int(np.ceil(col + width)))
    obj_r0 = max(0, int(np.floor(row)))
    obj_r1 = min(h, int(np.ceil(row + height)))
    if obj_r0 >= obj_r1 or obj_c0 >= obj_c1:
        return None

    r0 = max(0, obj_r0 - margin_px)
    r1 = min(h, obj_r1 + margin_px)
    c0 = max(0, obj_c0 - margin_px)
    c1 = min(w, obj_c1 + margin_px)
    if r0 >= r1 or c0 >= c1:
        return None

    patch_free = occ_free[r0:r1, c0:c1].copy()
    object_mask = np.zeros_like(patch_free, dtype=bool)
    object_mask[(obj_r0 - r0) : (obj_r1 - r0), (obj_c0 - c0) : (obj_c1 - c0)] = True
    removable_mask = object_mask & (~patch_free)
    blocked_pixels = int(np.count_nonzero(removable_mask))
    if blocked_pixels == 0:
        return None

    kernel_size = max(1, int(dilation_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    neighbor_mask = cv2.dilate(removable_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    neighbor_free_before = patch_free & neighbor_mask

    _, before_labels = cv2.connectedComponents(patch_free.astype(np.uint8))
    before_touching_labels = sorted(
        int(v) for v in np.unique(before_labels[neighbor_free_before]) if int(v) > 0
    )

    patch_opened = patch_free.copy()
    patch_opened[removable_mask] = True
    _, after_labels = cv2.connectedComponents(patch_opened.astype(np.uint8))
    neighbor_free_after = patch_opened & neighbor_mask
    after_touching_labels = sorted(
        int(v) for v in np.unique(after_labels[neighbor_free_after]) if int(v) > 0
    )

    bridge_detected = len(before_touching_labels) >= 2 and len(after_touching_labels) == 1
    component_reduction = max(0, len(before_touching_labels) - len(after_touching_labels))

    return {
        "bbox_px": [obj_c0, obj_r0, obj_c1, obj_r1],
        "patch_bbox_px": [c0, r0, c1, r1],
        "blocked_pixels": blocked_pixels,
        "before_touching_component_count": len(before_touching_labels),
        "before_touching_component_labels": before_touching_labels,
        "after_touching_component_count": len(after_touching_labels),
        "after_touching_component_labels": after_touching_labels,
        "component_reduction": component_reduction,
        "bridge_detected": bool(bridge_detected),
    }


def _empty_component_merge_stats(
    bbox_px: list[int],
    patch_bbox_px: list[int],
    freed_pixels: int = 0,
) -> dict[str, Any]:
    return {
        "bbox_px": bbox_px,
        "patch_bbox_px": patch_bbox_px,
        "freed_pixels": int(freed_pixels),
        "before_touching_component_count": 0,
        "before_touching_component_labels": [],
        "after_touching_component_count": 0,
        "after_touching_component_labels": [],
        "component_reduction": 0,
        "connectivity_changed": False,
    }


def component_merge_stats_from_patches(
    before_patch: np.ndarray,
    after_patch: np.ndarray,
    freed_patch: np.ndarray,
    bbox_px: list[int],
    patch_bbox_px: list[int],
    touch_dilation_px: int = 8,
) -> dict[str, Any]:
    freed_pixels = int(np.count_nonzero(freed_patch))
    if freed_pixels == 0:
        return _empty_component_merge_stats(bbox_px, patch_bbox_px)

    kernel_size = max(1, int(touch_dilation_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    touch_band = cv2.dilate(freed_patch.astype(np.uint8), kernel, iterations=1).astype(bool)
    touch_band &= ~freed_patch

    _, before_labels = cv2.connectedComponents(before_patch.astype(np.uint8), connectivity=8)
    _, after_labels = cv2.connectedComponents(after_patch.astype(np.uint8), connectivity=8)

    before_touch_labels = sorted(
        int(v) for v in np.unique(before_labels[touch_band & before_patch]) if int(v) > 0
    )
    after_touch_region = (touch_band | freed_patch) & after_patch
    after_touch_labels = sorted(
        int(v) for v in np.unique(after_labels[after_touch_region]) if int(v) > 0
    )
    component_reduction = max(0, len(before_touch_labels) - len(after_touch_labels))
    connectivity_changed = len(before_touch_labels) >= 2 and component_reduction > 0

    return {
        "bbox_px": bbox_px,
        "patch_bbox_px": patch_bbox_px,
        "freed_pixels": freed_pixels,
        "before_touching_component_count": len(before_touch_labels),
        "before_touching_component_labels": before_touch_labels,
        "after_touching_component_count": len(after_touch_labels),
        "after_touching_component_labels": after_touch_labels,
        "component_reduction": component_reduction,
        "connectivity_changed": bool(connectivity_changed),
    }


def local_component_merge_approx_from_bbox(
    scene_map: ProcTHORMap,
    rec: dict[str, Any],
    margin_px: int = 120,
    release_expansion_px: int = 60,
    touch_dilation_px: int = 8,
) -> dict[str, Any] | None:
    box = object_box_to_px(scene_map, rec)
    if box is None:
        return None

    before_free = np.asarray(scene_map.occupancy).astype(bool)
    h, w = before_free.shape
    col, row, width, height = box
    obj_c0 = max(0, int(np.floor(col)))
    obj_c1 = min(w, int(np.ceil(col + width)))
    obj_r0 = max(0, int(np.floor(row)))
    obj_r1 = min(h, int(np.ceil(row + height)))
    if obj_r0 >= obj_r1 or obj_c0 >= obj_c1:
        return None

    r0 = max(0, obj_r0 - margin_px)
    r1 = min(h, obj_r1 + margin_px)
    c0 = max(0, obj_c0 - margin_px)
    c1 = min(w, obj_c1 + margin_px)
    if r0 >= r1 or c0 >= c1:
        return None

    before_patch = before_free[r0:r1, c0:c1]
    release_r0 = max(r0, max(0, obj_r0 - release_expansion_px)) - r0
    release_r1 = min(r1, min(h, obj_r1 + release_expansion_px)) - r0
    release_c0 = max(c0, max(0, obj_c0 - release_expansion_px)) - c0
    release_c1 = min(c1, min(w, obj_c1 + release_expansion_px)) - c0
    release_mask = np.zeros_like(before_patch, dtype=bool)
    release_mask[release_r0:release_r1, release_c0:release_c1] = True
    freed_patch = release_mask & (~before_patch)
    after_patch = before_patch.copy()
    after_patch[freed_patch] = True

    return component_merge_stats_from_patches(
        before_patch=before_patch,
        after_patch=after_patch,
        freed_patch=freed_patch,
        bbox_px=[obj_c0, obj_r0, obj_c1, obj_r1],
        patch_bbox_px=[c0, r0, c1, r1],
        touch_dilation_px=touch_dilation_px,
    )


def local_connectivity_change_between_maps(
    before_map: ProcTHORMap,
    after_map: ProcTHORMap,
    rec: dict[str, Any],
    margin_px: int = 120,
) -> dict[str, Any] | None:
    box = object_box_to_px(before_map, rec)
    if box is None:
        return None

    before_free = np.asarray(before_map.occupancy).astype(bool)
    after_free = np.asarray(after_map.occupancy).astype(bool)
    if before_free.shape != after_free.shape:
        return None

    col, row, width, height = box
    h, w = before_free.shape
    obj_c0 = max(0, int(np.floor(col)))
    obj_c1 = min(w, int(np.ceil(col + width)))
    obj_r0 = max(0, int(np.floor(row)))
    obj_r1 = min(h, int(np.ceil(row + height)))
    if obj_r0 >= obj_r1 or obj_c0 >= obj_c1:
        return None

    r0 = max(0, obj_r0 - margin_px)
    r1 = min(h, obj_r1 + margin_px)
    c0 = max(0, obj_c0 - margin_px)
    c1 = min(w, obj_c1 + margin_px)
    if r0 >= r1 or c0 >= c1:
        return None

    before_patch = before_free[r0:r1, c0:c1]
    after_patch = after_free[r0:r1, c0:c1]
    freed_patch = after_patch & (~before_patch)
    merge_stats = component_merge_stats_from_patches(
        before_patch=before_patch,
        after_patch=after_patch,
        freed_patch=freed_patch,
        bbox_px=[obj_c0, obj_r0, obj_c1, obj_r1],
        patch_bbox_px=[c0, r0, c1, r1],
    )

    _, before_labels = cv2.connectedComponents(before_patch.astype(np.uint8), connectivity=8)
    _, after_labels = cv2.connectedComponents(after_patch.astype(np.uint8), connectivity=8)

    obj_r0p = obj_r0 - r0
    obj_r1p = obj_r1 - r0
    obj_c0p = obj_c0 - c0
    obj_c1p = obj_c1 - c0

    col_pad = max(8, (obj_c1p - obj_c0p) // 2)
    row_pad = max(8, (obj_r1p - obj_r0p) // 2)
    top_mask = np.zeros_like(before_patch, dtype=bool)
    bottom_mask = np.zeros_like(before_patch, dtype=bool)
    left_mask = np.zeros_like(before_patch, dtype=bool)
    right_mask = np.zeros_like(before_patch, dtype=bool)

    cc0 = max(0, obj_c0p - col_pad)
    cc1 = min(before_patch.shape[1], obj_c1p + col_pad)
    rr0 = max(0, obj_r0p - row_pad)
    rr1 = min(before_patch.shape[0], obj_r1p + row_pad)
    top_mask[:obj_r0p, cc0:cc1] = True
    bottom_mask[obj_r1p:, cc0:cc1] = True
    left_mask[rr0:rr1, :obj_c0p] = True
    right_mask[rr0:rr1, obj_c1p:] = True

    def label_set(labels: np.ndarray, side_mask: np.ndarray) -> set[int]:
        values = labels[side_mask]
        return {int(v) for v in np.unique(values) if int(v) > 0}

    def connected(labels: np.ndarray, a_mask: np.ndarray, b_mask: np.ndarray) -> bool:
        a = label_set(labels, a_mask)
        b = label_set(labels, b_mask)
        return bool(a and b and (a & b))

    vertical_before = connected(before_labels, top_mask, bottom_mask)
    vertical_after = connected(after_labels, top_mask, bottom_mask)
    horizontal_before = connected(before_labels, left_mask, right_mask)
    horizontal_after = connected(after_labels, left_mask, right_mask)

    return {
        **merge_stats,
        "vertical_connected_before": bool(vertical_before),
        "vertical_connected_after": bool(vertical_after),
        "horizontal_connected_before": bool(horizontal_before),
        "horizontal_connected_after": bool(horizontal_after),
        "top_component_labels_before": sorted(label_set(before_labels, top_mask)),
        "bottom_component_labels_before": sorted(label_set(before_labels, bottom_mask)),
        "left_component_labels_before": sorted(label_set(before_labels, left_mask)),
        "right_component_labels_before": sorted(label_set(before_labels, right_mask)),
        "top_component_labels_after": sorted(label_set(after_labels, top_mask)),
        "bottom_component_labels_after": sorted(label_set(after_labels, bottom_mask)),
        "left_component_labels_after": sorted(label_set(after_labels, left_mask)),
        "right_component_labels_after": sorted(label_set(after_labels, right_mask)),
    }


def compute_local_patch_connectivity_debug(
    scene_map: ProcTHORMap,
    rec: dict[str, Any],
    margin_px: int = 24,
    dilation_px: int = 6,
    removal_expansion_px: int = 0,
) -> dict[str, Any] | None:
    box = object_box_to_px(scene_map, rec)
    if box is None:
        return None

    col, row, width, height = box
    occ_free = np.asarray(scene_map.occupancy).astype(bool)
    h, w = occ_free.shape

    obj_c0 = max(0, int(np.floor(col)))
    obj_c1 = min(w, int(np.ceil(col + width)))
    obj_r0 = max(0, int(np.floor(row)))
    obj_r1 = min(h, int(np.ceil(row + height)))
    if obj_r0 >= obj_r1 or obj_c0 >= obj_c1:
        return None

    r0 = max(0, obj_r0 - margin_px)
    r1 = min(h, obj_r1 + margin_px)
    c0 = max(0, obj_c0 - margin_px)
    c1 = min(w, obj_c1 + margin_px)
    if r0 >= r1 or c0 >= c1:
        return None

    patch_free = occ_free[r0:r1, c0:c1].copy()
    object_mask = np.zeros_like(patch_free, dtype=bool)
    object_mask[(obj_r0 - r0) : (obj_r1 - r0), (obj_c0 - c0) : (obj_c1 - c0)] = True
    removable_mask = object_mask & (~patch_free)
    blocked_pixels = int(np.count_nonzero(removable_mask))
    if blocked_pixels == 0:
        return None

    removal_mask = removable_mask.copy()
    if removal_expansion_px > 0:
        expand_kernel = np.ones((removal_expansion_px, removal_expansion_px), dtype=np.uint8)
        removal_mask = cv2.dilate(removal_mask.astype(np.uint8), expand_kernel, iterations=1).astype(bool)
        removal_mask &= ~patch_free

    kernel_size = max(1, int(dilation_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    neighbor_mask = cv2.dilate(removal_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    neighbor_free_before = patch_free & neighbor_mask

    before_count, before_labels = cv2.connectedComponents(patch_free.astype(np.uint8))
    before_touching_labels = sorted(
        int(v) for v in np.unique(before_labels[neighbor_free_before]) if int(v) > 0
    )

    patch_opened = patch_free.copy()
    patch_opened[removal_mask] = True
    after_count, after_labels = cv2.connectedComponents(patch_opened.astype(np.uint8))
    neighbor_free_after = patch_opened & neighbor_mask
    after_touching_labels = sorted(
        int(v) for v in np.unique(after_labels[neighbor_free_after]) if int(v) > 0
    )

    return {
        "bbox_px": [obj_c0, obj_r0, obj_c1, obj_r1],
        "patch_bbox_px": [c0, r0, c1, r1],
        "patch_free": patch_free,
        "object_mask": object_mask,
        "removable_mask": removable_mask,
        "removal_mask": removal_mask,
        "neighbor_mask": neighbor_mask,
        "before_labels": before_labels,
        "after_labels": after_labels,
        "before_component_count": int(max(0, before_count - 1)),
        "after_component_count": int(max(0, after_count - 1)),
        "before_touching_component_labels": before_touching_labels,
        "after_touching_component_labels": after_touching_labels,
        "before_touching_component_count": len(before_touching_labels),
        "after_touching_component_count": len(after_touching_labels),
        "bridge_detected": bool(len(before_touching_labels) >= 2 and len(after_touching_labels) == 1),
    }


def save_local_patch_connectivity_figure(
    out_path: Path,
    scene_map: ProcTHORMap,
    rec: dict[str, Any],
    margin_px: int = 24,
    dilation_px: int = 6,
    removal_expansion_px: int = 0,
) -> dict[str, Any] | None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    debug = compute_local_patch_connectivity_debug(
        scene_map,
        rec,
        margin_px=margin_px,
        dilation_px=dilation_px,
        removal_expansion_px=removal_expansion_px,
    )
    if debug is None:
        return None

    patch_free = debug["patch_free"]
    patch_occ = ~patch_free
    removal_mask = debug["removal_mask"]
    removable_mask = debug["removable_mask"]
    neighbor_mask = debug["neighbor_mask"]
    patch_opened = patch_free.copy()
    patch_opened[removal_mask] = True

    before_rgb = np.zeros((*patch_free.shape, 3), dtype=float)
    before_rgb[patch_free] = np.array([0.95, 0.95, 0.92])
    before_rgb[patch_occ] = np.array([0.18, 0.20, 0.22])
    before_rgb[neighbor_mask] = before_rgb[neighbor_mask] * 0.6 + np.array([0.2, 0.4, 1.0]) * 0.4
    before_rgb[removable_mask] = np.array([0.96, 0.45, 0.12])
    before_rgb[removal_mask & ~removable_mask] = np.array([0.95, 0.75, 0.2])

    after_rgb = np.zeros((*patch_opened.shape, 3), dtype=float)
    after_rgb[patch_opened] = np.array([0.95, 0.95, 0.92])
    after_rgb[~patch_opened] = np.array([0.18, 0.20, 0.22])
    after_rgb[removal_mask] = np.array([0.22, 0.78, 0.33])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = [
        (
            "Before patch",
            f"touching comps={debug['before_touching_component_count']} "
            f"labels={debug['before_touching_component_labels']}",
        ),
        (
            "After local removal",
            f"touching comps={debug['after_touching_component_count']} "
            f"labels={debug['after_touching_component_labels']}",
        ),
    ]
    images = [before_rgb, after_rgb]

    for ax, image, (title, subtitle) in zip(axes, images, titles):
        ax.imshow(image, origin="upper")
        obj_c0, obj_r0, obj_c1, obj_r1 = debug["bbox_px"]
        patch_c0, patch_r0, _, _ = debug["patch_bbox_px"]
        rect = Rectangle(
            (obj_c0 - patch_c0, obj_r0 - patch_r0),
            obj_c1 - obj_c0,
            obj_r1 - obj_r0,
            facecolor="none",
            edgecolor="#ef4444",
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Local Patch Connectivity | {rec['name']} | margin={margin_px}px | expansion={removal_expansion_px}px",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "debug_path": str(out_path),
        "bbox_px": debug["bbox_px"],
        "patch_bbox_px": debug["patch_bbox_px"],
        "before_touching_component_count": debug["before_touching_component_count"],
        "after_touching_component_count": debug["after_touching_component_count"],
        "before_touching_component_labels": debug["before_touching_component_labels"],
        "after_touching_component_labels": debug["after_touching_component_labels"],
        "bridge_detected": debug["bridge_detected"],
        "removal_expansion_px": removal_expansion_px,
    }


def save_global_object_inflation_overlay_figure(
    out_path: Path,
    scene_map: ProcTHORMap,
    rec: dict[str, Any],
    inflation_px: int,
    title: str | None = None,
) -> dict[str, Any] | None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    box = object_box_to_px(scene_map, rec)
    if box is None:
        return None

    occ_free = np.asarray(scene_map.occupancy).astype(bool)
    h, w = occ_free.shape
    obj_c0 = max(0, int(np.floor(box[0])))
    obj_c1 = min(w, int(np.ceil(box[0] + box[2])))
    obj_r0 = max(0, int(np.floor(box[1])))
    obj_r1 = min(h, int(np.ceil(box[1] + box[3])))

    removable_mask = None
    try:
        body_id = int(rec["body_id"])
        geom_ids = descendant_geoms(rec["_model"], body_id, visual_only=False)
        rendered_mask, effective_px = render_topdown_geom_mask(
            rec["_model"],
            rec["_data"],
            geom_ids,
            px_per_m=int(round(scene_map.px_per_m)),
        )
        if rendered_mask.shape == occ_free.shape:
            removable_mask = rendered_mask & (~occ_free)
    except Exception as exc:
        log.warning("Falling back to bbox seed for %s overlay: %s", rec["name"], exc)

    if removable_mask is None:
        object_mask = np.zeros((h, w), dtype=bool)
        object_mask[obj_r0:obj_r1, obj_c0:obj_c1] = True
        removable_mask = object_mask & (~occ_free)

    kernel_size = max(1, int(inflation_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    inflated_mask = cv2.dilate(removable_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    inflated_mask &= ~occ_free

    bg = make_scene_plot_background(scene_map)
    overlay = bg.copy()
    overlay[inflated_mask] = 0.55 * overlay[inflated_mask] + 0.45 * np.array([0.20, 0.78, 0.33])
    overlay[removable_mask] = np.array([0.96, 0.45, 0.12])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay, origin="upper")
    rect = Rectangle(
        (obj_c0, obj_r0),
        obj_c1 - obj_c0,
        obj_r1 - obj_r0,
        facecolor="none",
        edgecolor="#ef4444",
        linewidth=2.0,
    )
    ax.add_patch(rect)
    ax.set_title(
        title
        or f"Global Object Inflation Overlay | {rec['name']} | inflation={inflation_px}px"
    )
    ax.set_xlabel("map x (px)")
    ax.set_ylabel("map y (px)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "overlay_path": str(out_path),
        "bbox_px": [obj_c0, obj_r0, obj_c1, obj_r1],
        "inflation_px": inflation_px,
        "removable_pixels": int(np.count_nonzero(removable_mask)),
        "inflated_pixels": int(np.count_nonzero(inflated_mask)),
    }


def analyze_local_blocking_movable_obstacles(
    base_scene_map: ProcTHORMap,
    candidate_records: list[dict[str, Any]],
    baseline_path: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    baseline_path_px = points_xy_to_px(base_scene_map, baseline_path) if baseline_path is not None else None
    analyzed = []
    blocking_names: list[str] = []

    for rec in candidate_records:
        if not rec.get("is_pickup_candidate", False):
            continue
        if rec.get("is_structural", False) or rec.get("is_door", False):
            continue

        overlap_stats = removable_obstacle_overlap_stats(base_scene_map, rec)
        local_stats = local_patch_connectivity_stats(base_scene_map, rec)
        if overlap_stats is None or local_stats is None:
            continue

        center_dist_to_path_px = None
        if baseline_path_px is not None and len(baseline_path_px) > 0:
            center_px = np.array(
                [
                    0.5 * (local_stats["bbox_px"][1] + local_stats["bbox_px"][3]),
                    0.5 * (local_stats["bbox_px"][0] + local_stats["bbox_px"][2]),
                ],
                dtype=float,
            )
            center_dist_to_path_px = float(
                np.min(np.linalg.norm(np.asarray(baseline_path_px, dtype=float) - center_px, axis=1))
            )

        score = float(local_stats["component_reduction"])
        if local_stats["bridge_detected"]:
            score += 3.0
        if center_dist_to_path_px is not None and center_dist_to_path_px <= 120.0:
            score += 1.0

        analyzed.append(
            {
                "name": rec["name"],
                "category": rec.get("category"),
                "room_id": rec.get("room_id"),
                "blocked_pixels": overlap_stats["blocked_pixels"],
                "blocked_ratio": overlap_stats["blocked_ratio"],
                "neighbor_component_count": overlap_stats["neighbor_component_count"],
                "bbox_px": local_stats["bbox_px"],
                "patch_bbox_px": local_stats["patch_bbox_px"],
                "before_touching_component_count": local_stats["before_touching_component_count"],
                "after_touching_component_count": local_stats["after_touching_component_count"],
                "component_reduction": local_stats["component_reduction"],
                "bridge_detected": local_stats["bridge_detected"],
                "center_dist_to_path_px": center_dist_to_path_px,
                "local_blocking_score": score,
            }
        )
        if local_stats["bridge_detected"]:
            blocking_names.append(rec["name"])

    analyzed.sort(
        key=lambda item: (
            item["local_blocking_score"],
            item["blocked_pixels"],
        ),
        reverse=True,
    )
    return analyzed, blocking_names


def analyze_blocking_movable_obstacles(
    env,
    base_scene_map: ProcTHORMap,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    model_path: str,
    agent_radius: float,
    px_per_m: int,
    open_threshold: float,
    candidate_records: list[dict[str, Any]],
    downscale_factor: int = 5,
) -> tuple[list[dict[str, Any]], ProcTHORMap | None, list[str], np.ndarray | None]:
    margin_px = max(120, int(round(agent_radius * px_per_m * 2)))
    release_expansion_px = max(4, int(round(agent_radius * px_per_m)))
    max_precise_checks = 12
    fast_candidates = []
    for rec in candidate_records:
        if not rec.get("is_pickup_candidate", False):
            continue
        if rec.get("is_structural", False) or rec.get("is_door", False):
            continue
        stats = removable_obstacle_overlap_stats(base_scene_map, rec)
        if stats is None:
            continue
        if stats["blocked_pixels"] <= 64:
            continue
        fast_stats = local_component_merge_approx_from_bbox(
            base_scene_map,
            rec,
            margin_px=margin_px,
            release_expansion_px=release_expansion_px,
        )
        if fast_stats is None:
            continue
        fast_score = (
            10.0 * float(fast_stats["connectivity_changed"])
            + float(fast_stats["component_reduction"])
            + 0.0001 * float(fast_stats["freed_pixels"])
        )
        fast_candidates.append((fast_score, rec, stats, fast_stats))

    fast_candidates.sort(
        key=lambda item: (
            item[3]["connectivity_changed"],
            item[3]["component_reduction"],
            item[3]["before_touching_component_count"],
            item[3]["freed_pixels"],
            item[2]["blocked_pixels"],
        ),
        reverse=True,
    )

    analyzed = []
    blocking_names: list[str] = []
    cleared_map = None
    cleared_path = None
    best_freed_pixels = -1

    selected_names: set[str] = set()
    selected_for_precise = []
    for item in fast_candidates:
        _, rec, _, fast_stats = item
        if fast_stats["connectivity_changed"]:
            selected_for_precise.append(item)
            selected_names.add(rec["name"])
        if len(selected_for_precise) >= max_precise_checks:
            break

    if len(selected_for_precise) < max_precise_checks:
        for item in fast_candidates:
            _, rec, _, fast_stats = item
            if rec["name"] in selected_names:
                continue
            if fast_stats["before_touching_component_count"] < 2:
                continue
            selected_for_precise.append(item)
            selected_names.add(rec["name"])
            if len(selected_for_precise) >= max_precise_checks:
                break

    if len(selected_for_precise) < max_precise_checks:
        for item in fast_candidates:
            _, rec, _, _ = item
            if rec["name"] in selected_names:
                continue
            selected_for_precise.append(item)
            selected_names.add(rec["name"])
            if len(selected_for_precise) >= max_precise_checks:
                break

    precise_by_name: dict[str, tuple[ProcTHORMap, dict[str, Any]]] = {}
    for _, rec, _, _ in selected_for_precise:
        removed_map = build_live_procthor_map(
            env.current_model,
            env.current_data,
            model_path=model_path,
            px_per_m=px_per_m,
            agent_radius=agent_radius,
            open_threshold=open_threshold,
            ignored_root_body_names={rec["name"]},
        )
        connectivity_stats = local_connectivity_change_between_maps(
            base_scene_map,
            removed_map,
            rec,
            margin_px=margin_px,
        )
        if connectivity_stats is None:
            continue
        precise_by_name[rec["name"]] = (removed_map, connectivity_stats)

    for _, rec, stats, fast_stats in fast_candidates:
        precise_item = precise_by_name.get(rec["name"])
        removed_map = None
        connectivity_stats = None
        if precise_item is not None:
            removed_map, connectivity_stats = precise_item

        blocking = bool(
            connectivity_stats is not None and connectivity_stats["connectivity_changed"]
        )
        precise_checked = connectivity_stats is not None
        analyzed.append(
            {
                "name": rec["name"],
                "category": rec.get("category"),
                "room_id": rec.get("room_id"),
                "blocked_pixels": stats["blocked_pixels"],
                "blocked_ratio": stats["blocked_ratio"],
                "neighbor_component_count": stats["neighbor_component_count"],
                "bbox_px": stats["bbox_px"],
                "fast_local_patch_bbox_px": fast_stats["patch_bbox_px"],
                "fast_freed_pixels": fast_stats["freed_pixels"],
                "fast_before_touching_component_count": fast_stats[
                    "before_touching_component_count"
                ],
                "fast_after_touching_component_count": fast_stats[
                    "after_touching_component_count"
                ],
                "fast_component_reduction": fast_stats["component_reduction"],
                "fast_connectivity_changed": fast_stats["connectivity_changed"],
                "precise_checked": bool(precise_checked),
                "local_patch_bbox_px": None
                if connectivity_stats is None
                else connectivity_stats["patch_bbox_px"],
                "freed_pixels_after_removal": None
                if connectivity_stats is None
                else connectivity_stats["freed_pixels"],
                "before_touching_component_count": None
                if connectivity_stats is None
                else connectivity_stats["before_touching_component_count"],
                "after_touching_component_count": None
                if connectivity_stats is None
                else connectivity_stats["after_touching_component_count"],
                "component_reduction": None
                if connectivity_stats is None
                else connectivity_stats["component_reduction"],
                "vertical_connected_before": None
                if connectivity_stats is None
                else connectivity_stats["vertical_connected_before"],
                "vertical_connected_after": None
                if connectivity_stats is None
                else connectivity_stats["vertical_connected_after"],
                "horizontal_connected_before": None
                if connectivity_stats is None
                else connectivity_stats["horizontal_connected_before"],
                "horizontal_connected_after": None
                if connectivity_stats is None
                else connectivity_stats["horizontal_connected_after"],
                "connectivity_changed": False
                if connectivity_stats is None
                else connectivity_stats["connectivity_changed"],
                "top_component_labels_before": []
                if connectivity_stats is None
                else connectivity_stats["top_component_labels_before"],
                "bottom_component_labels_before": []
                if connectivity_stats is None
                else connectivity_stats["bottom_component_labels_before"],
                "left_component_labels_before": []
                if connectivity_stats is None
                else connectivity_stats["left_component_labels_before"],
                "right_component_labels_before": []
                if connectivity_stats is None
                else connectivity_stats["right_component_labels_before"],
                "top_component_labels_after": []
                if connectivity_stats is None
                else connectivity_stats["top_component_labels_after"],
                "bottom_component_labels_after": []
                if connectivity_stats is None
                else connectivity_stats["bottom_component_labels_after"],
                "left_component_labels_after": []
                if connectivity_stats is None
                else connectivity_stats["left_component_labels_after"],
                "right_component_labels_after": []
                if connectivity_stats is None
                else connectivity_stats["right_component_labels_after"],
                "considered_blocking": bool(blocking),
            }
        )
        if blocking:
            blocking_names.append(rec["name"])
            freed_pixels = int(connectivity_stats["freed_pixels"])
            if freed_pixels > best_freed_pixels:
                best_freed_pixels = freed_pixels
                cleared_map = removed_map
                cleared_path = None

    return analyzed, cleared_map, blocking_names, cleared_path


def _collect_numeric_scalars(value: Any, out: list[float]) -> None:
    if value is None:
        return
    if isinstance(value, np.ndarray):
        for item in value.reshape(-1):
            _collect_numeric_scalars(item, out)
        return
    if isinstance(value, np.generic):
        out.append(float(value))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_numeric_scalars(item, out)
        return
    try:
        out.append(float(value))
    except (TypeError, ValueError):
        return


def normalize_point3d(point: Any) -> np.ndarray:
    try:
        arr = np.asarray(point, dtype=float)
        arr = np.squeeze(arr)
        if arr.size >= 3:
            return arr.reshape(-1)[:3].astype(float)
    except (TypeError, ValueError):
        pass

    flat: list[float] = []
    _collect_numeric_scalars(point, flat)
    if len(flat) < 3:
        raise ValueError(f"Expected at least 3 numeric coordinates, got {point!r}")
    return np.asarray(flat[:3], dtype=float)


def collect_door_plot_records(env, door_names: list[str]) -> list[dict[str, Any]]:
    records = []
    for door_name in door_names:
        try:
            door = Door(door_name, env.current_data)
            hinge_idx = door.get_hinge_joint_index()
            records.append(
                {
                    "door_name": door_name,
                    "hinge_xy": door.get_joint_anchor_position(hinge_idx)[:2].copy(),
                    "joint_position": float(door.get_joint_position(hinge_idx)),
                    "joint_range": [float(v) for v in door.get_joint_range(hinge_idx)],
                    "interactive": True,
                    "record_type": "interactive_door",
                }
            )
        except Exception as exc:
            log.warning("Failed to collect plotting data for door %s: %s", door_name, exc)
    return records


def collect_non_interactive_doorway_plot_records(
    env, scene_map: ProcTHORMap, doorway_analysis: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if doorway_analysis is None:
        return []

    records = []
    for rec in doorway_analysis.get("root_records", []):
        if rec.get("interactive", True):
            continue
        root_body_name = rec["root_body_name"]
        try:
            root_body_id = mujoco.mj_name2id(
                env.current_model, mujoco.mjtObj.mjOBJ_BODY, root_body_name
            )
            if root_body_id < 0:
                raise ValueError(f"Body not found in env model: {root_body_name}")
            candidate_xys = [np.asarray(env.current_data.xpos[root_body_id][:2], dtype=float).copy()]
            for child_name in rec.get("child_body_names", []):
                child_body_id = mujoco.mj_name2id(
                    env.current_model, mujoco.mjtObj.mjOBJ_BODY, child_name
                )
                if child_body_id >= 0:
                    candidate_xys.append(
                        np.asarray(env.current_data.xpos[child_body_id][:2], dtype=float).copy()
                    )

            xy = candidate_xys[0]
            if rec.get("fixed_opening", False):
                for candidate_xy in candidate_xys:
                    nearest_xy = nearest_free_point_xy(scene_map, candidate_xy, max_radius_px=120)
                    if nearest_xy is not None:
                        xy = np.asarray(nearest_xy, dtype=float).copy()
                        break
        except Exception as exc:
            log.warning(
                "Failed to collect plotting data for non-interactive doorway %s: %s",
                root_body_name,
                exc,
            )
            continue
        records.append(
            {
                "door_name": root_body_name,
                "hinge_xy": xy,
                "interactive": False,
                "fixed_opening": bool(rec.get("fixed_opening", False)),
                "record_type": f"non_interactive_{rec.get('root_kind', 'doorway')}",
            }
        )
    return records


def save_door_path_figure(
    out_path: Path,
    scene_map: ProcTHORMap,
    door_records: list[dict[str, Any]],
    object_records: list[dict[str, Any]] | None,
    selected_doors: list[str],
    highlighted_object_names: list[str],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    primary_path: np.ndarray | None,
    primary_label: str,
    secondary_path: np.ndarray | None = None,
    secondary_label: str | None = None,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    bg = make_scene_plot_background(scene_map)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bg, origin="upper")

    if object_records:
        for rec in object_records:
            box = object_box_to_px(scene_map, rec)
            if box is None:
                continue
            col, row, width, height = box
            kind = scene_visual_kind(rec)
            color = scene_plot_color(kind)
            is_highlighted = rec["name"] in highlighted_object_names
            rect = Rectangle(
                (col, row),
                max(width, 3.0),
                max(height, 3.0),
                facecolor="none",
                edgecolor="#ef4444" if is_highlighted else color,
                linewidth=2.2 if is_highlighted else (1.0 if kind != "structural" else 0.6),
                linestyle="--" if rec.get("is_pickup_candidate", False) else "-",
                alpha=0.95 if is_highlighted else (0.75 if kind != "structural" else 0.35),
                zorder=3 if is_highlighted else 2,
            )
            ax.add_patch(rect)
            label = scene_plot_label(rec)
            ax.text(
                col + width / 2.0,
                row - 4.0,
                label[:28],
                fontsize=6 if rec.get("is_pickup_candidate", False) else 5,
                color="#ef4444" if is_highlighted else color,
                ha="center",
                va="bottom",
                zorder=4 if is_highlighted else 3,
                bbox={
                    "facecolor": (1.0, 1.0, 1.0, 0.65),
                    "edgecolor": "none",
                    "pad": 0.4,
                },
            )

    for record in door_records:
        marker_px = points_xy_to_px(scene_map, np.asarray([record["hinge_xy"]]))
        if marker_px is None:
            continue
        row, col = marker_px[0]
        is_selected = record["door_name"] in selected_doors
        is_interactive = bool(record.get("interactive", True))
        record_type = record.get("record_type", "")
        if is_interactive:
            base_color = "#7c3aed"
        elif "doorframe" in record_type:
            base_color = "#0ea5e9"
        else:
            base_color = "#f97316"
        ax.scatter(
            col,
            row,
            s=34 if is_selected else (26 if not is_interactive else 20),
            c="#dc2626" if is_selected else base_color,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )

    start_px = points_xy_to_px(scene_map, np.asarray([start_xy]))
    goal_px = points_xy_to_px(scene_map, np.asarray([goal_xy]))
    if start_px is not None:
        ax.scatter(start_px[:, 1], start_px[:, 0], marker="o", s=48, c="#16a34a", edgecolors="black", linewidths=0.8, zorder=6, label="start")
    if goal_px is not None:
        ax.scatter(goal_px[:, 1], goal_px[:, 0], marker="*", s=95, c="#f59e0b", edgecolors="black", linewidths=0.8, zorder=6, label="goal")

    primary_px = points_xy_to_px(scene_map, primary_path)
    if primary_px is not None:
        ax.plot(
            primary_px[:, 1],
            primary_px[:, 0],
            color="#2563eb",
            linewidth=2.4,
            zorder=7,
            label=primary_label,
        )

    secondary_px = points_xy_to_px(scene_map, secondary_path)
    if secondary_px is not None and secondary_label is not None:
        ax.plot(
            secondary_px[:, 1],
            secondary_px[:, 0],
            color="#ea580c",
            linewidth=2.4,
            linestyle="--",
            zorder=7,
            label=secondary_label,
        )

    ax.set_xlim(0, bg.shape[1])
    ax.set_ylim(bg.shape[0], 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x (px)")
    ax.set_ylabel("map y (px)")
    ax.set_title(title or "Door Path Study")
    has_movable_door_boxes = bool(
        object_records and any(rec.get("is_movable_door", False) for rec in object_records)
    )
    has_fixed_opening_boxes = bool(
        object_records and any(rec.get("is_fixed_opening", False) for rec in object_records)
    )
    has_noninteractive_doorway_boxes = bool(
        object_records and any(rec.get("is_noninteractive_doorway", False) for rec in object_records)
    )

    legend_handles = [
        Patch(facecolor="#eeeeee", edgecolor="black", alpha=0.55, label="free space"),
        Patch(facecolor="#2f3437", edgecolor="black", alpha=0.85, label="occupied"),
        Patch(facecolor="none", edgecolor=scene_plot_color("pickup"), linestyle="--", label="pickup / movable"),
        Patch(facecolor="none", edgecolor=scene_plot_color("receptacle"), label="receptacle"),
        Patch(facecolor="none", edgecolor=scene_plot_color("articulable"), label="articulable"),
        Patch(facecolor="none", edgecolor=scene_plot_color("structural"), label="structural"),
    ]
    if has_movable_door_boxes:
        legend_handles.append(
            Patch(facecolor="none", edgecolor=scene_plot_color("movable_door"), linewidth=1.4, label="interactive door")
        )
    if has_fixed_opening_boxes:
        legend_handles.append(
            Patch(facecolor="none", edgecolor=scene_plot_color("fixed_opening"), linewidth=1.4, label="fixed doorframe opening")
        )
    if has_noninteractive_doorway_boxes:
        legend_handles.append(
            Patch(facecolor="none", edgecolor=scene_plot_color("noninteractive_doorway"), linewidth=1.4, label="non-interactive doorway")
        )
    if door_records:
        legend_handles.extend(
            [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#7c3aed", markeredgecolor="white", markersize=6, label="interactive door marker"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#0ea5e9", markeredgecolor="white", markersize=6, label="fixed opening marker"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#f97316", markeredgecolor="white", markersize=6, label="doorway marker"),
            ]
        )
    legend_handles.extend(
        [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markeredgecolor="black", markersize=7, label="start"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#f59e0b", markeredgecolor="black", markersize=10, label="goal"),
        ]
    )
    if primary_px is not None:
        legend_handles.append(Line2D([0], [0], color="#2563eb", linewidth=2.4, label=primary_label))
    if secondary_px is not None and secondary_label is not None:
        legend_handles.append(Line2D([0], [0], color="#ea580c", linewidth=2.4, linestyle="--", label=secondary_label))
    if highlighted_object_names:
        legend_handles.append(Patch(facecolor="none", edgecolor="#ef4444", linewidth=2.2, label="blocking movable obstacle"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.96)
    ax.grid(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def sample_navigation_goal(task: BaseMujocoTask, scene_map: ProcTHORMap) -> tuple[np.ndarray, str, str | None]:
    nav_goal_sampler = NavGoalSampler(scene_map, check_target_in_view=False, camera_name="head_camera")
    batch_idx = task.env.current_batch_index
    target_obj = task.get_nearest_nav_object(batch_idx)
    target_pos = normalize_point3d(target_obj.position)
    nav_goal_sampler.set_target(target_obj)
    nav_goal_sampler.set_robot_view(task.env.current_robot.robot_view)
    goal = nav_goal_sampler.sample()
    if goal is not None:
        return normalize_point3d(goal), "nav_goal_sampler", None

    target_xy = target_pos[:2].copy()
    nearest_free_xy_goal = nearest_free_point_xy(scene_map, target_xy)
    if nearest_free_xy_goal is not None:
        fallback_goal = np.array(
            [nearest_free_xy_goal[0], nearest_free_xy_goal[1], float(target_pos[2])],
            dtype=float,
        )
        return (
            fallback_goal,
            "nearest_free_point_fallback",
            "Failed to sample a nav goal near target object",
        )

    fallback_goal = target_pos.copy()
    return (
        fallback_goal,
        "target_object_center_fallback",
        "Failed to sample a nav goal near target object",
    )


def get_robot_xy(env) -> np.ndarray:
    return env.current_robot.robot_view.base.pose[:2, 3].copy()


def get_open_position_for_joint(joint_range: tuple[float, float]) -> float:
    low, high = float(joint_range[0]), float(joint_range[1])
    return high if abs(high) >= abs(low) else low


def interactive_door_root_names(doorway_analysis: dict[str, Any] | None) -> list[str]:
    if doorway_analysis is None:
        return []
    return [
        rec["root_body_name"]
        for rec in doorway_analysis.get("root_records", [])
        if rec.get("interactive", False)
    ]


def set_door_state(env, door_name: str, state: str) -> dict[str, Any]:
    ensure_runtime_dependencies()
    door = Door(door_name, env.current_data)
    hinge_idx = door.get_hinge_joint_index()
    joint_range = door.get_joint_range(hinge_idx)
    if state == "open":
        target = get_open_position_for_joint(joint_range)
    elif state == "closed":
        target = 0.0
    else:
        raise ValueError(f"Unsupported door state: {state}")
    door.set_joint_position(hinge_idx, target)
    mujoco.mj_forward(env.current_model, env.current_data)
    return {
        "object_name": door_name,
        "door_name": door_name,
        "joint_name": door.joint_names[hinge_idx],
        "joint_index": hinge_idx,
        "joint_range": [float(v) for v in joint_range],
        "joint_position": float(door.get_joint_position(hinge_idx)),
        "open_fraction": 1.0 if state == "open" else 0.0,
        "state": state,
    }


def set_door_root_state(
    env,
    doorway_analysis: dict[str, Any] | None,
    root_door_name: str,
    state: str,
) -> dict[str, Any]:
    if doorway_analysis is None:
        raise ValueError("doorway_analysis is required for root door control")

    matched = None
    for rec in doorway_analysis.get("root_records", []):
        if rec.get("root_body_name") == root_door_name:
            matched = rec
            break
    if matched is None:
        raise ValueError(f"Interactive door root not found: {root_door_name}")

    transitions = []
    skipped_children = []
    skipped_child_errors: dict[str, str] = {}
    for child_name in matched.get("hinge_body_names", []):
        try:
            transitions.append(set_door_state(env, child_name, state))
        except Exception as exc:
            log.debug(
                "Skipping non-door hinge child while setting root %s: %s (%s)",
                root_door_name,
                child_name,
                exc,
            )
            skipped_children.append(child_name)
            skipped_child_errors[child_name] = str(exc)

    if not transitions:
        raise ValueError(
            f"No valid interactive door leaf joints found for root {root_door_name}. "
            f"Candidates were {matched.get('hinge_body_names', [])}; "
            f"errors={skipped_child_errors}"
        )

    return {
        "door_root_name": root_door_name,
        "state": state,
        "hinge_body_names": list(matched.get("hinge_body_names", [])),
        "skipped_hinge_body_names": skipped_children,
        "skipped_hinge_body_errors": skipped_child_errors,
        "transitions": transitions,
    }


def set_articulation_fraction(env, object_name: str, joint_index: int, open_fraction: float) -> dict[str, Any]:
    om = env.object_managers[env.current_batch_index]
    obj = om.get_object_by_name(object_name)
    if not isinstance(obj, MlSpacesArticulationObject):
        raise ValueError(f"{object_name} is not an articulated object")
    joint_range = obj.get_joint_range(joint_index)
    target = float(joint_range[0] + (joint_range[1] - joint_range[0]) * open_fraction)
    obj.set_joint_position(joint_index, target)
    mujoco.mj_forward(env.current_model, env.current_data)
    return {
        "object_name": object_name,
        "joint_index": joint_index,
        "joint_name": obj.joint_names[joint_index],
        "joint_range": [float(v) for v in joint_range],
        "joint_position": float(obj.get_joint_position(joint_index)),
        "open_fraction": float(open_fraction),
    }


def choose_doors_on_path(env, door_names: list[str], path: np.ndarray, top_k: int) -> list[str]:
    if path is None or len(path) == 0:
        return []
    scored = []
    for name in door_names:
        door = Door(name, env.current_data)
        hinge_pos = door.get_joint_anchor_position(door.get_hinge_joint_index())[:2]
        dists = np.linalg.norm(path - hinge_pos[None, :], axis=1)
        scored.append((float(dists.min()), name))
    scored.sort()
    return [name for _, name in scored[:top_k]]


def open_all_doors(env) -> list[dict[str, Any]]:
    doorway_analysis = collect_runtime_doorway_analysis(env)
    transitions = []
    for door_root_name in interactive_door_root_names(doorway_analysis):
        transitions.append(set_door_root_state(env, doorway_analysis, door_root_name, "open"))
    return transitions


def close_all_doors(env) -> list[dict[str, Any]]:
    doorway_analysis = collect_runtime_doorway_analysis(env)
    transitions = []
    for door_root_name in interactive_door_root_names(doorway_analysis):
        transitions.append(set_door_root_state(env, doorway_analysis, door_root_name, "closed"))
    return transitions


def command_door_map_compare(args: argparse.Namespace) -> dict[str, Any]:
    ctx = load_context(args, task_mode="nav_task")
    try:
        closed_plot_path, cached_plot_path, open_plot_path = map_compare_output_paths(args.output_json)
        start_xy = get_robot_xy(ctx.env)
        initial_scene_object_records = collect_scene_plot_records(ctx.env)

        cached_plot_map = getattr(ctx.task, "occupancy_map", None)

        close_transitions = close_all_doors(ctx.env)
        closed_scene_object_records = collect_scene_plot_records(ctx.env)
        closed_candidate_records = [
            rec
            for rec in closed_scene_object_records
            if rec.get("is_pickup_candidate", False) and not rec.get("is_structural", False)
        ]
        closed_map, closed_analysis = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )

        if cached_plot_map is None:
            cached_plot_map = closed_map
        all_door_names = interactive_door_root_names(closed_analysis)
        nav_goal, _, _ = sample_navigation_goal(ctx.task, cached_plot_map)
        raw_live_map = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
        )
        movable_analysis, _, blocking_movable_names, _ = analyze_blocking_movable_obstacles(
            env=ctx.env,
            base_scene_map=raw_live_map,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            model_path=str(ctx.env.current_model_path),
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            px_per_m=args.px_per_m,
            open_threshold=args.open_threshold,
            candidate_records=closed_candidate_records,
            downscale_factor=args.downscale,
        )

        closed_door_object_records = collect_interactive_door_root_object_records(
            ctx.env, closed_analysis
        )
        closed_noninteractive_door_records = collect_non_interactive_doorway_object_records(
            ctx.env, closed_analysis
        )
        closed_blocking_object_records = [
            rec for rec in closed_scene_object_records if rec["name"] in blocking_movable_names
        ]
        closed_focus_records = dedupe_plot_records(
            closed_door_object_records
            + closed_noninteractive_door_records
            + closed_blocking_object_records
        )

        open_transitions = open_all_doors(ctx.env)
        open_scene_object_records = collect_scene_plot_records(ctx.env)
        ignored_blocking_names = set(blocking_movable_names)
        open_map, open_analysis = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
            ignored_root_body_names=ignored_blocking_names if ignored_blocking_names else None,
        )
        open_door_object_records = collect_interactive_door_root_object_records(
            ctx.env, open_analysis
        )
        open_noninteractive_door_records = collect_non_interactive_doorway_object_records(
            ctx.env, open_analysis
        )
        open_blocking_object_records = [
            rec for rec in open_scene_object_records if rec["name"] in blocking_movable_names
        ]
        open_focus_records = dedupe_plot_records(
            open_door_object_records + open_noninteractive_door_records + open_blocking_object_records
        )

        save_door_path_figure(
            out_path=cached_plot_path,
            scene_map=cached_plot_map,
            door_records=[],
            object_records=initial_scene_object_records,
            selected_doors=[],
            highlighted_object_names=[],
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=None,
            primary_label="cached occupancy",
            title="Door Map Compare | cached occupancy + all objects",
        )
        save_door_path_figure(
            out_path=open_plot_path,
            scene_map=open_map,
            door_records=[],
            object_records=open_focus_records,
            selected_doors=[],
            highlighted_object_names=blocking_movable_names,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=None,
            primary_label="open occupancy",
            title="Door Map Compare | all interactive doors open + blocking movable removed",
        )
        save_door_path_figure(
            out_path=closed_plot_path,
            scene_map=closed_map,
            door_records=[],
            object_records=closed_focus_records,
            selected_doors=[],
            highlighted_object_names=blocking_movable_names,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=None,
            primary_label="closed occupancy",
            title="Door Map Compare | all interactive doors closed",
        )

        changed_pixels = int(
            np.count_nonzero(
                np.asarray(closed_map.occupancy).astype(bool)
                != np.asarray(open_map.occupancy).astype(bool)
            )
        )

        return {
            "scene_path": str(ctx.env.current_model_path),
            "target_object": ctx.task.config.task_config.pickup_obj_name,
            "robot_xy": start_xy.tolist(),
            "all_door_names": all_door_names,
            "interactive_door_count": len(all_door_names),
            "interactive_door_names": all_door_names,
            "movable_candidate_names": [rec["name"] for rec in closed_candidate_records],
            "movable_candidate_count": len(closed_candidate_records),
            "blocking_movable_obstacle_names": blocking_movable_names,
            "movable_obstacle_analysis": movable_analysis,
            "movable_fast_analysis_count": len(movable_analysis),
            "movable_precise_check_count": sum(
                1 for item in movable_analysis if item.get("precise_checked", False)
            ),
            "non_interactive_doorway_root_names": []
            if closed_analysis is None
            else [
                rec["root_body_name"]
                for rec in closed_analysis["root_records"]
                if not rec["interactive"] and rec.get("root_kind") == "doorway"
            ],
            "doorframe_root_names": []
            if closed_analysis is None
            else [
                rec["root_body_name"]
                for rec in closed_analysis["root_records"]
                if not rec["interactive"] and rec.get("root_kind") == "doorframe"
            ],
            "doorway_root_records": []
            if closed_analysis is None
            else closed_analysis["root_records"],
            "close_all_doors_transitions": close_transitions,
            "open_all_doors_transitions": open_transitions,
            "closed_plot_path": str(closed_plot_path),
            "cached_plot_path": str(cached_plot_path),
            "open_plot_path": str(open_plot_path),
            "movable_cleared_plot_path": None,
            "closed_free_ratio": float(np.mean(np.asarray(closed_map.occupancy).astype(bool))),
            "cached_free_ratio": float(np.mean(np.asarray(cached_plot_map.occupancy).astype(bool))),
            "open_free_ratio": float(np.mean(np.asarray(open_map.occupancy).astype(bool))),
            "movable_cleared_free_ratio": None,
            "closed_open_changed_pixels": changed_pixels,
            "fixed_opening_changed_pixels": int(
                np.count_nonzero(
                    np.asarray(closed_map.occupancy).astype(bool)
                    != np.asarray(cached_plot_map.occupancy).astype(bool)
                )
            ),
            "cached_object_box_count": len(initial_scene_object_records),
            "closed_focus_object_count": len(closed_focus_records),
            "open_focus_object_count": len(open_focus_records),
            "closed_non_interactive_root_count": 0
            if closed_analysis is None
            else len(closed_analysis["non_interactive_root_ids"]),
            "closed_non_interactive_doorway_count": 0
            if closed_analysis is None
            else sum(
                1
                for rec in closed_analysis["root_records"]
                if not rec["interactive"] and rec.get("root_kind") == "doorway"
            ),
            "closed_doorframe_root_count": 0
            if closed_analysis is None
            else sum(
                1
                for rec in closed_analysis["root_records"]
                if not rec["interactive"] and rec.get("root_kind") == "doorframe"
            ),
            "open_non_interactive_root_count": 0
            if open_analysis is None
            else len(open_analysis["non_interactive_root_ids"]),
            "local_patch_debug": None,
        }
    finally:
        close_context(ctx)


def command_inspect_scene(args: argparse.Namespace) -> dict[str, Any]:
    ctx = load_context(args, task_mode="scene_only")
    try:
        return list_articulated_objects(ctx)
    finally:
        close_context(ctx)


def command_nav_gt(args: argparse.Namespace) -> dict[str, Any]:
    ctx = load_context(args, task_mode="nav_task")
    try:
        live_map = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
        )
        base_result = {
            "scene_path": str(ctx.env.current_model_path),
            "robot_base_pose": ctx.task.env.current_robot.robot_view.base.pose.tolist(),
            "target_object": ctx.task.config.task_config.pickup_obj_name,
            "target_candidates": ctx.task.config.task_config.pickup_obj_candidates,
            "cached_occupancy_available": hasattr(ctx.task, "occupancy_map"),
        }
        nav_goal, nav_goal_source, nav_goal_sampling_error = sample_navigation_goal(ctx.task, live_map)
        path = compute_path_from_map(live_map, get_robot_xy(ctx.env), nav_goal[:2], downscale_factor=args.downscale)
        return {
            **base_result,
            "nav_goal": nav_goal.tolist(),
            "nav_goal_source": nav_goal_source,
            "path_found": path is not None,
            "path_length_m": path_length(path),
            "path_waypoints": None if path is None else path.tolist(),
            "nav_goal_sampling_error": nav_goal_sampling_error,
        }
    finally:
        close_context(ctx)


def run_door_path_study(ctx: LoadedContext, args: argparse.Namespace) -> dict[str, Any]:
        start_xy = get_robot_xy(ctx.env)
        open_transitions = open_all_doors(ctx.env)
        open_scene_object_records = collect_scene_plot_records(ctx.env)
        open_candidate_records = [
            rec
            for rec in open_scene_object_records
            if rec.get("is_pickup_candidate", False) and not rec.get("is_structural", False)
        ]
        raw_open_live_map = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
        )
        start_xy = get_robot_xy(ctx.env)
        base_result = {
            "scene_path": str(ctx.env.current_model_path),
            "target_object": ctx.task.config.task_config.pickup_obj_name,
            "robot_xy": start_xy.tolist(),
            "baseline_plot_path": None,
            "compare_plot_paths": [],
        }

        nav_goal, nav_goal_source, nav_goal_sampling_error = sample_navigation_goal(
            ctx.task, raw_open_live_map
        )
        movable_analysis, _, blocking_movable_names, _ = analyze_blocking_movable_obstacles(
            env=ctx.env,
            base_scene_map=raw_open_live_map,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            model_path=str(ctx.env.current_model_path),
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            px_per_m=args.px_per_m,
            open_threshold=args.open_threshold,
            candidate_records=open_candidate_records,
            downscale_factor=args.downscale,
        )

        ignored_blocking_names = set(blocking_movable_names)
        baseline_map, baseline_analysis = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
            ignored_root_body_names=ignored_blocking_names if ignored_blocking_names else None,
        )
        all_door_names = interactive_door_root_names(baseline_analysis)
        if args.door_names:
            requested = [name.strip() for name in args.door_names.split(",") if name.strip()]
            target_door_names = [name for name in requested if name in set(all_door_names)]
        else:
            target_door_names = list(all_door_names)
        baseline_plot_path, compare_plot_paths = door_path_study_output_paths(
            args.output_json, target_door_names
        )
        all_closed_plot_path = door_path_study_all_closed_output_path(args.output_json)
        baseline_path = compute_path_from_map(
            baseline_map, start_xy, nav_goal[:2], downscale_factor=args.downscale
        )

        baseline_door_object_records = collect_interactive_door_root_object_records(
            ctx.env, baseline_analysis
        )
        baseline_noninteractive_door_records = collect_non_interactive_doorway_object_records(
            ctx.env, baseline_analysis
        )
        baseline_blocking_object_records = [
            rec for rec in open_scene_object_records if rec["name"] in blocking_movable_names
        ]
        baseline_focus_records = dedupe_plot_records(
            baseline_door_object_records
            + baseline_noninteractive_door_records
            + baseline_blocking_object_records
        )

        save_door_path_figure(
            out_path=baseline_plot_path,
            scene_map=baseline_map,
            door_records=[],
            object_records=baseline_focus_records,
            selected_doors=[],
            highlighted_object_names=blocking_movable_names,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=baseline_path,
            primary_label="all-open GT path",
            title=f"Door Path Study | all doors open | target={ctx.task.config.task_config.pickup_obj_name}",
        )

        per_door_results = []
        for door_name, compare_plot_path in zip(target_door_names, compare_plot_paths):
            open_all_doors(ctx.env)
            transition = set_door_root_state(ctx.env, baseline_analysis, door_name, "closed")
            changed_map, changed_analysis = build_live_procthor_map(
                ctx.env.current_model,
                ctx.env.current_data,
                model_path=str(ctx.env.current_model_path),
                px_per_m=args.px_per_m,
                agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
                open_threshold=args.open_threshold,
                treat_all_non_interactive_doorways_as_open=True,
                return_doorway_analysis=True,
                ignored_root_body_names=ignored_blocking_names if ignored_blocking_names else None,
            )
            changed_path = compute_path_from_map(
                changed_map, start_xy, nav_goal[:2], downscale_factor=args.downscale
            )
            changed_door_object_records = collect_interactive_door_root_object_records(
                ctx.env, changed_analysis
            )
            changed_noninteractive_door_records = collect_non_interactive_doorway_object_records(
                ctx.env, changed_analysis
            )
            changed_focus_records = dedupe_plot_records(
                changed_door_object_records
                + changed_noninteractive_door_records
                + baseline_blocking_object_records
            )

            save_door_path_figure(
                out_path=compare_plot_path,
                scene_map=changed_map,
                door_records=[],
                object_records=changed_focus_records,
                selected_doors=[door_name],
                highlighted_object_names=blocking_movable_names,
                start_xy=start_xy,
                goal_xy=nav_goal[:2],
                primary_path=changed_path,
                primary_label=f"path with {door_name} closed",
                secondary_path=baseline_path,
                secondary_label="all-open GT path",
                title=f"Door Path Study | close {door_name}",
            )
            per_door_results.append(
                {
                    "door_name": door_name,
                    "transition": transition,
                    "path_found": changed_path is not None,
                    "path_length_m": path_length(changed_path),
                    "waypoints": None if changed_path is None else changed_path.tolist(),
                    "plot_path": str(compare_plot_path),
                }
            )

        close_all_transitions = close_all_doors(ctx.env)
        all_closed_map, all_closed_analysis = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
            ignored_root_body_names=ignored_blocking_names if ignored_blocking_names else None,
        )
        all_closed_path = compute_path_from_map(
            all_closed_map, start_xy, nav_goal[:2], downscale_factor=args.downscale
        )
        all_closed_door_object_records = collect_interactive_door_root_object_records(
            ctx.env, all_closed_analysis
        )
        all_closed_noninteractive_door_records = collect_non_interactive_doorway_object_records(
            ctx.env, all_closed_analysis
        )
        all_closed_focus_records = dedupe_plot_records(
            all_closed_door_object_records
            + all_closed_noninteractive_door_records
            + baseline_blocking_object_records
        )
        save_door_path_figure(
            out_path=all_closed_plot_path,
            scene_map=all_closed_map,
            door_records=[],
            object_records=all_closed_focus_records,
            selected_doors=list(all_door_names),
            highlighted_object_names=blocking_movable_names,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=all_closed_path,
            primary_label="all-closed GT path",
            secondary_path=baseline_path,
            secondary_label="all-open GT path",
            title="Door Path Study | all interactive doors closed",
        )

        return {
            **base_result,
            "nav_goal": nav_goal.tolist(),
            "nav_goal_source": nav_goal_source,
            "all_door_names": all_door_names,
            "selected_doors": target_door_names,
            "door_transitions": [item["transition"] for item in per_door_results],
            "baseline_path_found": baseline_path is not None,
            "baseline_path_length_m": path_length(baseline_path),
            "baseline_waypoints": None if baseline_path is None else baseline_path.tolist(),
            "changed_path_found": any(item["path_found"] for item in per_door_results),
            "changed_path_length_m": None,
            "changed_waypoints": None,
            "baseline_plot_path": str(baseline_plot_path),
            "compare_plot_path": None,
            "compare_plot_paths": [item["plot_path"] for item in per_door_results],
            "open_all_doors_transitions": open_transitions,
            "close_all_doors_transitions": close_all_transitions,
            "all_closed_path_found": all_closed_path is not None,
            "all_closed_path_length_m": path_length(all_closed_path),
            "all_closed_waypoints": None if all_closed_path is None else all_closed_path.tolist(),
            "all_closed_plot_path": str(all_closed_plot_path),
            "blocking_movable_obstacle_names": blocking_movable_names,
            "movable_obstacle_analysis": movable_analysis,
            "per_door_results": per_door_results,
            "nav_goal_sampling_error": nav_goal_sampling_error,
        }


def command_door_path_study(args: argparse.Namespace) -> dict[str, Any]:
    ctx = load_context(args, task_mode="nav_task")
    try:
        return run_door_path_study(ctx, args)
    except Exception as exc:
        log.error("Error during door path study: %s", exc, exc_info=True)
        raise
    finally:
        close_context(ctx)


def load_benchmark_episodes(benchmark_dir: Path) -> list[dict[str, Any]]:
    benchmark_file = benchmark_dir / "benchmark.json"
    with open(benchmark_file) as f:
        return json.load(f)


def command_benchmark_door_path_study(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_dir = Path(args.benchmark_dir)
    episodes = load_benchmark_episodes(benchmark_dir)
    end = min(len(episodes), args.start_idx + args.max_episodes)
    selected = episodes[args.start_idx:end]
    run_dir = args.output_json.with_suffix("")
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for episode_idx, episode in enumerate(selected, start=args.start_idx):
        episode_args = argparse.Namespace(**vars(args))
        episode_args.scene_dataset = episode["scene_dataset"]
        episode_args.data_split = episode["data_split"]
        episode_args.house_ind = episode["house_index"]
        episode_args.robot = episode.get("robot", {}).get("robot_name", args.robot)
        episode_args.target_types = None
        episode_args.benchmark_episode = episode
        episode_name = f"benchmark_ep_{episode_idx:04d}_house_{episode['house_index']}"
        episode_dir = run_dir / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_args.output_json = episode_dir / f"{episode_name}.json"
        ctx = None
        try:
            ctx = load_context(episode_args, task_mode="nav_task")
            episode_result = run_door_path_study(ctx, episode_args)
            episode_result["benchmark_episode_index"] = episode_idx
            episode_result["benchmark_source_traj_key"] = episode.get("source", {}).get("traj_key")
            episode_result["episode_output_dir"] = str(episode_dir)
            results.append(episode_result)
        except Exception as exc:
            failures.append(
                {
                    "benchmark_episode_index": episode_idx,
                    "house_index": episode["house_index"],
                    "pickup_obj_name": episode.get("task", {}).get("pickup_obj_name"),
                    "error": str(exc),
                }
            )
        finally:
            if ctx is not None:
                close_context(ctx)

    return {
        "benchmark_dir": str(benchmark_dir),
        "start_idx": args.start_idx,
        "max_episodes": args.max_episodes,
        "processed_episode_count": len(results),
        "failed_episode_count": len(failures),
        "output_dir": str(run_dir),
        "results": results,
        "failures": failures,
    }


def command_set_articulation(args: argparse.Namespace) -> dict[str, Any]:
    ctx = load_context(args, task_mode="scene_only")
    try:
        if not args.object_name:
            raise ValueError("--object-name is required for set-articulation")
        result = set_articulation_fraction(
            ctx.env, args.object_name, args.joint_index, args.open_fraction
        )
        return {
            "scene_path": str(ctx.env.current_model_path),
            "result": result,
        }
    finally:
        close_context(ctx)


def command_task_config_template(args: argparse.Namespace) -> dict[str, Any]:
    task_kind = args.task_kind

    common_notes = {
        "nav_to_obj": [
            "robot_base_pose can be filled from a sampled nav_to_obj episode and then frozen.",
            "pickup_obj_name should be a concrete object instance name.",
            "pickup_obj_candidates should contain same-category alternatives if you want eval-style ambiguity.",
            "After task init, you can directly call Door(...).set_joint_position(...) to force door states before recomputing GT path.",
        ],
        "door_opening": [
            "door_body_name and robot_base_pose are the key fields to freeze a reproducible door episode.",
            "articulated_joint_reset_state=[0.0] corresponds to closed in the current door-opening setup.",
            "This template matches DoorOpeningTaskConfig / DoorOpeningTaskSpec fields.",
        ],
        "open_close": [
            "pickup_obj_name should be an articulated object such as drawer/cabinet/fridge/microwave.",
            "joint_index or joint_name identifies which articulated joint to operate on.",
            "joint_start_position can be used to freeze a closed/open initial state before planner execution.",
        ],
    }

    templates = {
        "nav_to_obj": {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": "apple_xxx",
            "pickup_obj_candidates": ["apple_xxx", "apple_yyy"],
            "pickup_obj_category": "apple",
            "succ_pos_threshold": 1.5,
            "post_init_overrides": {
                "doors_to_close": [
                    {
                        "door_body_name": "door|...",
                        "state": "closed",
                        "api": "Door(door_body_name, env.current_data).set_joint_position(hinge_idx, 0.0)",
                    }
                ]
            },
        },
        "door_opening": {
            "task_cls": "molmo_spaces.tasks.opening_tasks.DoorOpeningTask",
            "task_type": "door_opening",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "door_body_name": "door|...",
            "articulated_joint_range": [0.0, 1.57],
            "articulated_joint_reset_state": [0.0],
            "door_openness_threshold": 0.67,
        },
        "open_close": {
            "task_cls": "molmo_spaces.tasks.opening_tasks.OpeningTask",
            "task_type": "open",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": "cabinet_xxx",
            "joint_name": "cabinet_joint_0",
            "joint_index": 0,
            "joint_start_position": 0.0,
            "joint_goal_position": None,
            "task_success_threshold": 0.2,
        },
    }

    return {
        "task_kind": task_kind,
        "template": templates[task_kind],
        "notes": common_notes[task_kind],
        "source_configs": {
            "nav_to_obj": "molmo_spaces.configs.task_configs.NavToObjTaskConfig",
            "door_opening": "molmo_spaces.configs.task_configs.DoorOpeningTaskConfig",
            "open_close": "molmo_spaces.configs.task_configs.OpeningTaskConfig",
        }[task_kind],
    }


def command_action_schema(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    schemas = {
        "navigation": {
            "description": "AStar nav policy emits base waypoint actions plus done.",
            "example_running": {"done": False, "base": [1.25, -0.8, 1.57]},
            "example_done": {"done": True, "base": [0.0, 0.0, 0.0]},
            "notes": [
                "task.step() strips the done field before forwarding controls.",
                "base format follows the robot base move-group control representation.",
            ],
        },
        "door_oracle": {
            "description": "Direct GT/oracle state change for doors, outside learned/planner control.",
            "example_python": {
                "door_ctor": "door = Door(door_body_name, env.current_data)",
                "hinge_index": "hinge_idx = door.get_hinge_joint_index()",
                "close": "door.set_joint_position(hinge_idx, 0.0)",
                "open": "door.set_joint_position(hinge_idx, target_open_joint_pos)",
            },
            "notes": [
                "This is the cleanest way to let navigation call open/close in a GT study.",
                "If you need step-wise physical execution instead, use DoorOpeningTask + opening_solver.py.",
            ],
        },
        "container_oracle": {
            "description": "Direct GT/oracle state change for articulated containers.",
            "example_python": {
                "obj_ctor": "obj = env.object_managers[0].get_object_by_name(object_name)",
                "joint_range": "joint_range = obj.get_joint_range(joint_index)",
                "set_fraction": "obj.set_joint_position(joint_index, joint_range[0] + frac * (joint_range[1] - joint_range[0]))",
            },
            "notes": [
                "Use this for fridge/drawer/cabinet GT studies.",
                "Planner-style execution should go through OpeningTask + OpenClosePlannerPolicy.",
            ],
        },
        "door_planner": {
            "description": "Door opening solver emits multi-move-group actions over phases.",
            "example_shape": {
                "done": False,
                "base": [0.0, 0.0, 0.0],
                "right_arm": ["...arm ctrl..."],
                "right_gripper": ["...gripper ctrl..."],
                "head": ["...head ctrl..."],
            },
            "notes": [
                "Not a single open_door primitive; it is a phased policy.",
                "See opening_solver.py:get_action().",
            ],
        },
        "container_planner": {
            "description": "OpenClosePlannerPolicy emits arm/gripper trajectories for articulated objects.",
            "example_shape": {
                "done": False,
                "arm_or_tcp_group": ["...planner output..."],
                "gripper": ["...gripper ctrl..."],
            },
            "notes": [
                "This is the task-execution path for drawer/cabinet/fridge/microwave.",
                "Not a one-field semantic action like {'open': 'fridge'} in the current codebase.",
            ],
        },
    }
    return {"mode": mode, "schema": schemas[mode]}


def command_benchmark_episode_template(args: argparse.Namespace) -> dict[str, Any]:
    task_kind = args.task_kind
    base_episode = {
        "source": {
            "h5_file": "/path/to/source.h5",
            "traj_key": "traj_0",
            "episode_length": 0,
            "camera_system_class": "RBY1GoProD455CameraSystem",
            "source_data_date": "2026-06-04",
            "benchmark_created_date": "2026-06-04",
        },
        "house_index": args.house_ind,
        "scene_dataset": args.scene_dataset,
        "data_split": args.data_split,
        "seed": args.seed,
        "robot": {
            "name": args.robot,
            "description": "Fill with benchmark robot spec fields used in your eval setup.",
        },
        "img_resolution": [640, 480],
        "cameras": [],
        "scene_modifications": {
            "added_objects": {},
            "object_poses": {},
            "removed_objects": [],
        },
        "task_relevant_objects": [],
        "language": {
            "task_description": "",
            "referral_expressions": {},
            "referral_expressions_priority": {},
        },
    }

    task_templates = {
        "nav_to_obj": {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": "apple_xxx",
            "pickup_obj_candidates": ["apple_xxx", "apple_yyy"],
            "pickup_obj_start_pose": None,
            "receptacle_name": None,
            "succ_pos_threshold": 1.5,
        },
        "door_opening": {
            "task_cls": "molmo_spaces.tasks.opening_tasks.DoorOpeningTask",
            "task_type": "door_opening",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "door_body_name": "door|...",
            "articulated_joint_range": [0.0, 1.57],
            "articulated_joint_reset_state": [0.0],
            "door_openness_threshold": 0.67,
        },
        "open_close": {
            "task_cls": "molmo_spaces.tasks.opening_tasks.OpeningTask",
            "task_type": "open",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": "cabinet_xxx",
            "pickup_obj_start_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "articulation_object_name": None,
            "joint_name": "cabinet_joint_0",
            "joint_index": 0,
            "joint_start_position": 0.0,
            "joint_goal_position": None,
            "task_success_threshold": 0.2,
            "any_inst_of_category": False,
        },
    }

    notes = {
        "nav_to_obj": [
            "This matches benchmark_schema.NavToObjTaskSpec fields.",
            "If you want to study door state changes, freeze robot_base_pose and pickup_obj_name first.",
            "Then apply post-init door overrides and recompute GT path on the live model.",
        ],
        "door_opening": [
            "This matches benchmark_schema.DoorOpeningTaskSpec fields.",
            "Use articulated_joint_reset_state=[0.0] for closed-door initialization in the current setup.",
        ],
        "open_close": [
            "This matches benchmark_schema.OpenCloseTaskSpec fields.",
            "Useful for fridge/drawer/cabinet benchmark-style frozen episodes.",
        ],
    }

    episode = dict(base_episode)
    episode["task"] = task_templates[task_kind]
    return {
        "task_kind": task_kind,
        "episode_spec_template": episode,
        "notes": notes[task_kind],
    }


def command_integration_recipe(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    recipes = {
        "door_oracle_nav_loop": {
            "description": "Navigation loop with direct oracle door open/close overrides.",
            "pseudo_code": [
                "task = sampler.sample_task(...)",
                "obs, info = task.reset()",
                "planner_path = compute_gt_path(...)",
                "if need_interaction(door, planner_path):",
                "    door = Door(door_body_name, env.current_data)",
                "    hinge_idx = door.get_hinge_joint_index()",
                "    door.set_joint_position(hinge_idx, target_open_position)",
                "    planner_path = recompute_gt_path_on_live_map(...)",
                "while not task.is_done():",
                "    waypoint = next_waypoint(planner_path)",
                "    action = {'base': waypoint, 'done': False}",
                "    obs, reward, terminated, truncated, info = task.step(action)",
            ],
            "notes": [
                "This is the cleanest recipe for proving interaction changes reachability/cost.",
                "It avoids mixing navigation research with physical door-manipulation execution.",
            ],
        },
        "container_oracle_nav_loop": {
            "description": "Navigation or exploration loop with direct oracle container articulation.",
            "pseudo_code": [
                "obj = env.object_managers[0].get_object_by_name(object_name)",
                "joint_range = obj.get_joint_range(joint_index)",
                "target = joint_range[0] + open_fraction * (joint_range[1] - joint_range[0])",
                "obj.set_joint_position(joint_index, target)",
                "mujoco.mj_forward(env.current_model, env.current_data)",
                "update_graph_state(object_name, state='open')",
                "continue navigation / active perception",
            ],
            "notes": [
                "Use this for fridge/drawer/cabinet studies where visibility/access changes are the focus.",
                "This is the recommended first step before testing container manipulation execution.",
            ],
        },
        "door_planner_handoff": {
            "description": "How navigation should hand off to the existing door-opening planner task.",
            "pseudo_code": [
                "freeze current nav state (robot pose, chosen door, target object)",
                "construct DoorOpeningTaskConfig with door_body_name + robot_base_pose",
                "door_task = DoorOpeningTask(env, exp_config)",
                "door_policy = OpeningSolver(config, task=door_task)",
                "obs, info = door_task.reset()",
                "while not door_task.is_done():",
                "    action = door_policy.get_action(obs)",
                "    obs, reward, terminated, truncated, info = door_task.step(action)",
                "after success, update nav graph / map state and resume nav_to_obj",
            ],
            "notes": [
                "This is a task handoff, not a single open-door action primitive.",
                "Best used after oracle studies identify which doors matter.",
            ],
        },
        "container_planner_handoff": {
            "description": "How navigation should hand off to the existing open/close planner task.",
            "pseudo_code": [
                "freeze current robot pose and target articulated object",
                "construct OpeningTaskConfig with pickup_obj_name + joint_name/joint_index + joint_start_position",
                "open_task = OpeningTask(env, exp_config)",
                "open_policy = OpenClosePlannerPolicy(config, task=open_task)",
                "obs, info = open_task.reset()",
                "while not open_task.is_done():",
                "    action = open_policy.get_action(obs)",
                "    obs, reward, terminated, truncated, info = open_task.step(action)",
                "after success, update graph state and resume navigation / search",
            ],
            "notes": [
                "This path is appropriate for fridge/drawer/cabinet physical execution studies.",
                "Again, this is a task/policy handoff, not a semantic one-field action.",
            ],
        },
    }
    return {"mode": mode, "recipe": recipes[mode]}


def command_env_check(args: argparse.Namespace) -> dict[str, Any]:
    import importlib.util
    import os

    python_path = sys.executable
    mlspaces_python = os.environ.get(
        "MLSPACES_PYTHON", str(Path.home() / "miniconda3/envs/mlspaces/bin/python")
    )
    default_cache_dir = Path.home() / ".cache/molmo-spaces-resources"
    cache_dir = os.environ.get("MLSPACES_CACHE_DIR", str(default_cache_dir))

    checks = {
        "current_python": python_path,
        "recommended_python": mlspaces_python,
        "using_mlspaces_python": python_path == mlspaces_python,
        "repo_root_on_syspath": str(REPO_ROOT) in sys.path,
        "mujoco_importable": importlib.util.find_spec("mujoco") is not None,
        "networkx_importable": importlib.util.find_spec("networkx") is not None,
        "opencv_importable": importlib.util.find_spec("cv2") is not None,
        "molmo_spaces_importable": importlib.util.find_spec("molmo_spaces") is not None,
        "mlspaces_cache_dir": cache_dir,
        "mlspaces_cache_parent_exists": Path(cache_dir).parent.exists(),
        "mlspaces_cache_dir_exists": Path(cache_dir).exists(),
        "mlspaces_cache_parent_writable": os.access(Path(cache_dir).parent, os.W_OK),
        "tmp_writable": os.access("/tmp", os.W_OK),
    }

    recommendations = []
    if not checks["using_mlspaces_python"]:
        recommendations.append(
            "Set MLSPACES_PYTHON to the environment interpreter or activate the mlspaces conda environment."
        )
    if not checks["mujoco_importable"]:
        recommendations.append("Install or activate an environment containing mujoco.")
    if not checks["networkx_importable"]:
        recommendations.append("Install or activate an environment containing networkx.")
    if not checks["molmo_spaces_importable"]:
        recommendations.append("Run from the repo root or ensure the repository is on PYTHONPATH.")
    if cache_dir.startswith(str(Path.home() / ".cache")):
        recommendations.append(
            "In the current sandbox, prefer MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources or another writable cache path."
        )
    recommendations.append(
        "Real scene-loading commands still require pre-populated local MolmoSpaces resources or outbound network access for remote manifests/assets."
    )

    return {
        "checks": checks,
        "recommendations": recommendations,
    }


def write_output(result: dict[str, Any], output_path: Path | None) -> None:
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
        log.info("Wrote output to %s", output_path)
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore MolmoSpaces interactive objects, GT paths, and articulation interfaces."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--scene_dataset", default="procthor-10k")
        subparser.add_argument("--data_split", default="train")
        subparser.add_argument("--house_ind", type=int, default=1)
        subparser.add_argument("--variant", default="ceiling")
        subparser.add_argument("--robot", default="rby1", choices=["rby1", "droid", "rum"])
        subparser.add_argument("--seed", type=int, default=2)
        subparser.add_argument("--target_types", default=None)
        subparser.add_argument("--output_json", type=Path, default=None)

    inspect_parser = subparsers.add_parser("inspect-scene")
    add_common(inspect_parser)

    nav_parser = subparsers.add_parser("nav-gt")
    add_common(nav_parser)
    nav_parser.add_argument("--px_per_m", type=int, default=200)
    nav_parser.add_argument("--downscale", type=int, default=5)
    nav_parser.add_argument("--open_threshold", type=float, default=1e-3)

    door_parser = subparsers.add_parser("door-path-study")
    add_common(door_parser)
    door_parser.add_argument("--px_per_m", type=int, default=200)
    door_parser.add_argument("--downscale", type=int, default=5)
    door_parser.add_argument("--open_threshold", type=float, default=1e-3)
    door_parser.add_argument("--door_names", default=None)
    door_parser.add_argument("--close_doors_on_path", type=int, default=0)
    door_parser.add_argument(
        "--study_state", choices=["open", "closed"], default="closed"
    )

    bench_door_parser = subparsers.add_parser("benchmark-door-path-study")
    add_common(bench_door_parser)
    bench_door_parser.add_argument("--benchmark_dir", type=Path, required=True)
    bench_door_parser.add_argument("--start_idx", type=int, default=0)
    bench_door_parser.add_argument("--max_episodes", type=int, default=10)
    bench_door_parser.add_argument("--px_per_m", type=int, default=200)
    bench_door_parser.add_argument("--downscale", type=int, default=5)
    bench_door_parser.add_argument("--open_threshold", type=float, default=1e-3)
    bench_door_parser.add_argument("--door_names", default=None)

    door_map_parser = subparsers.add_parser("door-map-compare")
    add_common(door_map_parser)
    door_map_parser.add_argument("--px_per_m", type=int, default=200)
    door_map_parser.add_argument("--open_threshold", type=float, default=1e-3)
    door_map_parser.add_argument("--downscale", type=int, default=5)

    set_parser = subparsers.add_parser("set-articulation")
    add_common(set_parser)
    set_parser.add_argument("--object-name", dest="object_name", required=True)
    set_parser.add_argument("--joint-index", type=int, default=0)
    set_parser.add_argument("--open-fraction", dest="open_fraction", type=float, default=0.0)

    template_parser = subparsers.add_parser("task-config-template")
    add_common(template_parser)
    template_parser.add_argument(
        "--task-kind",
        choices=["nav_to_obj", "door_opening", "open_close"],
        default="nav_to_obj",
    )

    action_parser = subparsers.add_parser("action-schema")
    add_common(action_parser)
    action_parser.add_argument(
        "--mode",
        choices=[
            "navigation",
            "door_oracle",
            "container_oracle",
            "door_planner",
            "container_planner",
        ],
        default="navigation",
    )

    benchmark_parser = subparsers.add_parser("benchmark-episode-template")
    add_common(benchmark_parser)
    benchmark_parser.add_argument(
        "--task-kind",
        choices=["nav_to_obj", "door_opening", "open_close"],
        default="nav_to_obj",
    )

    integration_parser = subparsers.add_parser("integration-recipe")
    add_common(integration_parser)
    integration_parser.add_argument(
        "--mode",
        choices=[
            "door_oracle_nav_loop",
            "container_oracle_nav_loop",
            "door_planner_handoff",
            "container_planner_handoff",
        ],
        default="door_oracle_nav_loop",
    )

    env_check_parser = subparsers.add_parser("env-check")
    add_common(env_check_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.output_json is None:
        default_name = f"{args.command}_{args.scene_dataset}_{args.data_split}_{args.house_ind}.json"
        args.output_json = DEFAULT_OUTPUT_DIR / default_name

    try:
        if args.command == "inspect-scene":
            result = command_inspect_scene(args)
        elif args.command == "nav-gt":
            result = command_nav_gt(args)
        elif args.command == "door-path-study":
            result = command_door_path_study(args)
        elif args.command == "benchmark-door-path-study":
            result = command_benchmark_door_path_study(args)
        elif args.command == "door-map-compare":
            result = command_door_map_compare(args)
        elif args.command == "set-articulation":
            result = command_set_articulation(args)
        elif args.command == "task-config-template":
            result = command_task_config_template(args)
        elif args.command == "action-schema":
            result = command_action_schema(args)
        elif args.command == "benchmark-episode-template":
            result = command_benchmark_episode_template(args)
        elif args.command == "integration-recipe":
            result = command_integration_recipe(args)
        elif args.command == "env-check":
            result = command_env_check(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")

        write_output(result, args.output_json)
    except RuntimeError as exc:
        log.error(str(exc))
        raise SystemExit(1) from exc
    except Exception as exc:
        message = str(exc)
        if "Read-only file system" in message and "molmo-spaces-resources" in message:
            log.error(
                "MolmoSpaces resource cache is pointing to a non-writable location. "
                "Retry with `MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources`."
            )
        elif "r2.dev" in message or "ConnectionError" in type(exc).__name__ or "Max retries exceeded" in message:
            log.error(
                "MolmoSpaces resource manager attempted to fetch remote manifests/assets but network access is unavailable. "
                "In this environment, real scene-loading commands need either pre-populated local resources "
                "or outbound network access."
            )
            log.error(
                "Suggested retry pattern: "
                "`MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources "
                "${MLSPACES_PYTHON:-python} "
                "scripts/InteractiveNav/explore_molmo_interactions.py <subcommand> ...`"
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
