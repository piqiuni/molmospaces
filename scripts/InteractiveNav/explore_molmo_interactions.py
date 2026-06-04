from __future__ import annotations

import argparse
import gc
import json
import logging
import os
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

DEFAULT_OUTPUT_DIR = Path("/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/output")


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
    global NavGoalSampler, inverse_homogeneous_matrix, geom_aabb, ProcTHORMap, circular_kernel

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
        from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb as _geom_aabb
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
    cfg = build_config(args, task_mode=task_mode)
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    task = None
    try:
        if task_mode == "scene_only":
            sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=args.house_ind)
            scene_path = sampler._current_house_scene_path(variant=args.variant)
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


def build_live_procthor_map(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    px_per_m: int = 200,
    agent_radius: float | None = None,
    open_threshold: float = 1e-3,
    device_id: int | None = None,
) -> ProcTHORMap:
    ensure_runtime_dependencies()
    floor_ids = []
    room_ids_to_name = {}
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and (geom_name.startswith("room|") or geom_name.startswith("room_")):
            floor_ids.append(geom_id)
            room_body_id = model.geom(geom_id).bodyid.item()
            room_ids_to_name[geom_id + 1] = model.body(room_body_id).name

    if not floor_ids:
        raise ValueError("No floors found in the live model.")

    open_door_ids, doorway_ids = _collect_open_door_root_ids(model, data, open_threshold)

    doorframe_geom_ids = []
    door_geom_ids = []
    for geom_id in range(model.ngeom):
        body_id = model.geom(geom_id).bodyid.item()
        parent_body_id = model.body(body_id).parentid.item()
        if body_id in open_door_ids or parent_body_id in open_door_ids:
            door_geom_ids.append(geom_id)
        root_body_id = model.body(body_id).rootid.item()
        if root_body_id in doorway_ids:
            doorframe_geom_ids.append(geom_id)

    aabb_center, aabb_size = geom_aabb(model, data, floor_ids, tight_mesh=False)
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

        renderer = MjOpenGLRenderer(model=model, height=h, width=w, device_id=device_id)
        renderer.update(data, cam)
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
        occ_door_path = cv2.dilate(occ_door_path.astype(np.uint8), circular_kernel(15)).astype(bool)

        occ = occ_floor
        occ[occ_door_path == 1] = False

        if cam_distance == 5.0:
            return occ, occ_room_floor, effective_px, (h, w), cam_to_world
        return occ, occ_room_floor, effective_px, (h, w)

    occ_map_5, room_map_5, effective_px, (h, w), cam_to_world = render_occupancy(5.0)
    occ_final = occ_map_5.copy()
    room_map_final = room_map_5.copy()

    if agent_radius is not None:
        rad_px = int(agent_radius * effective_px)
        kernel = circular_kernel(rad_px)
        occ_final = cv2.dilate(occ_final.astype(np.uint8), kernel).astype(bool)
        room_map_final[occ_final] = 0

    cam_to_map = np.array([[0, -effective_px, 0, h / 2], [effective_px, 0, 0, w / 2]])
    world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)

    map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
    centered_to_cam = np.array([[1 / effective_px, 0, 0], [0, 1 / effective_px, 0], [0, 0, 1]])
    cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
    cam_to_world_floor[2, 2] = 0
    map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered

    occ_final = ~occ_final
    instance = ProcTHORMap(
        occupancy=occ_final,
        room_map=room_map_final,
        room_ids_to_name=room_ids_to_name,
        world_to_map=world_to_map,
        map_to_world=map_to_world,
        px_per_m=effective_px,
    )
    instance.occupancy_base = occ_map_5
    gc.collect()
    return instance


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


def path_length(path: np.ndarray | None) -> float | None:
    if path is None or len(path) < 2:
        return None
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def image_output_paths(output_json: Path) -> tuple[Path, Path]:
    stem = output_json.stem
    return (
        output_json.with_name(f"{stem}_baseline.png"),
        output_json.with_name(f"{stem}_compare.png"),
    )


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
                }
            )
        except Exception as exc:
            log.warning("Failed to collect plotting data for door %s: %s", door_name, exc)
    return records


def save_door_path_figure(
    out_path: Path,
    scene_map: ProcTHORMap,
    door_records: list[dict[str, Any]],
    selected_doors: list[str],
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

    bg = make_scene_plot_background(scene_map)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bg, origin="upper")

    for record in door_records:
        marker_px = points_xy_to_px(scene_map, np.asarray([record["hinge_xy"]]))
        if marker_px is None:
            continue
        row, col = marker_px[0]
        is_selected = record["door_name"] in selected_doors
        ax.scatter(
            col,
            row,
            s=34 if is_selected else 20,
            c="#dc2626" if is_selected else "#7c3aed",
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
    ax.legend(loc="upper right")
    ax.grid(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def sample_navigation_goal(task: BaseMujocoTask, scene_map: ProcTHORMap) -> tuple[np.ndarray, str, str | None]:
    nav_goal_sampler = NavGoalSampler(scene_map, check_target_in_view=False, camera_name="head_camera")
    batch_idx = task.env.current_batch_index
    target_obj = task.get_nearest_nav_object(batch_idx)
    nav_goal_sampler.set_target(target_obj)
    nav_goal_sampler.set_robot_view(task.env.current_robot.robot_view)
    goal = nav_goal_sampler.sample()
    if goal is not None:
        return np.asarray(goal), "nav_goal_sampler", None

    target_xy = np.asarray(target_obj.position[:2], dtype=float)
    nearest_free_xy_goal = nearest_free_point_xy(scene_map, target_xy)
    if nearest_free_xy_goal is not None:
        fallback_goal = np.array(
            [nearest_free_xy_goal[0], nearest_free_xy_goal[1], float(target_obj.position[2])],
            dtype=float,
        )
        return (
            fallback_goal,
            "nearest_free_point_fallback",
            "Failed to sample a nav goal near target object",
        )

    fallback_goal = np.asarray(target_obj.position[:3], dtype=float)
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


def set_door_state(env, door_name: str, state: str) -> dict[str, Any]:
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
        "door_name": door_name,
        "joint_index": hinge_idx,
        "joint_range": [float(v) for v in joint_range],
        "joint_position": float(door.get_joint_position(hinge_idx)),
        "state": state,
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


def command_door_path_study(args: argparse.Namespace) -> dict[str, Any]:
    ctx = load_context(args, task_mode="nav_task")
    try:
        baseline_plot_path, compare_plot_path = image_output_paths(args.output_json)
        baseline_map = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
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
            "compare_plot_path": None,
        }
        nav_goal, nav_goal_source, nav_goal_sampling_error = sample_navigation_goal(ctx.task, baseline_map)
        baseline_path = compute_path_from_map(
            baseline_map, start_xy, nav_goal[:2], downscale_factor=args.downscale
        )

        om = ctx.env.object_managers[ctx.env.current_batch_index]
        all_door_names = om.find_door_names()
        baseline_door_records = collect_door_plot_records(ctx.env, all_door_names)
        if args.door_names:
            selected_doors = [name.strip() for name in args.door_names.split(",") if name.strip()]
        elif args.close_doors_on_path > 0:
            selected_doors = choose_doors_on_path(
                ctx.env, all_door_names, baseline_path, args.close_doors_on_path
            )
        else:
            selected_doors = []

        transitions = []
        for door_name in selected_doors:
            transitions.append(set_door_state(ctx.env, door_name, args.study_state))

        changed_map = build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
        )
        changed_path = compute_path_from_map(
            changed_map, start_xy, nav_goal[:2], downscale_factor=args.downscale
        )
        changed_door_records = collect_door_plot_records(ctx.env, all_door_names)
        cached_plot_map = getattr(ctx.task, "occupancy_map", None)
        baseline_plot_map = pick_plot_scene_map(baseline_map, cached_plot_map)
        changed_plot_map = pick_plot_scene_map(changed_map, cached_plot_map)

        save_door_path_figure(
            out_path=baseline_plot_path,
            scene_map=baseline_plot_map,
            door_records=baseline_door_records,
            selected_doors=selected_doors,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=baseline_path,
            primary_label="baseline GT path",
            title=f"Door Path Study Baseline | target={ctx.task.config.task_config.pickup_obj_name}",
        )
        save_door_path_figure(
            out_path=compare_plot_path,
            scene_map=changed_plot_map,
            door_records=changed_door_records,
            selected_doors=selected_doors,
            start_xy=start_xy,
            goal_xy=nav_goal[:2],
            primary_path=changed_path,
            primary_label=f"{args.study_state} door GT path",
            secondary_path=baseline_path,
            secondary_label="baseline GT path",
            title=(
                f"Door Path Study Compare | state={args.study_state} | "
                f"doors={len(selected_doors)}"
            ),
        )

        return {
            **base_result,
            "nav_goal": nav_goal.tolist(),
            "nav_goal_source": nav_goal_source,
            "all_door_names": all_door_names,
            "selected_doors": selected_doors,
            "door_transitions": transitions,
            "baseline_path_found": baseline_path is not None,
            "baseline_path_length_m": path_length(baseline_path),
            "changed_path_found": changed_path is not None,
            "changed_path_length_m": path_length(changed_path),
            "baseline_waypoints": None if baseline_path is None else baseline_path.tolist(),
            "changed_waypoints": None if changed_path is None else changed_path.tolist(),
            "baseline_plot_path": str(baseline_plot_path),
            "compare_plot_path": str(compare_plot_path),
            "nav_goal_sampling_error": nav_goal_sampling_error,
        }
    finally:
        close_context(ctx)


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
    mlspaces_python = "/home/user/miniconda3/envs/mlspaces/bin/python"
    cache_dir = os.environ.get("MLSPACES_CACHE_DIR", "/home/user/.cache/molmo-spaces-resources")

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
            "Use /home/user/miniconda3/envs/mlspaces/bin/python or activate the mlspaces conda environment."
        )
    if not checks["mujoco_importable"]:
        recommendations.append("Install or activate an environment containing mujoco.")
    if not checks["networkx_importable"]:
        recommendations.append("Install or activate an environment containing networkx.")
    if not checks["molmo_spaces_importable"]:
        recommendations.append("Run from the repo root or ensure the repository is on PYTHONPATH.")
    if cache_dir.startswith("/home/user/.cache"):
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
                "/home/user/miniconda3/envs/mlspaces/bin/python "
                "scripts/InteractiveNav/explore_molmo_interactions.py <subcommand> ...`"
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
