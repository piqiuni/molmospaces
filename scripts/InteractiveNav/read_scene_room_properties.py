import argparse
import json
import logging
import signal
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    RBY1GoProD455CameraSystem,
)
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig, FrankaRobotConfig, RBY1Config
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.articulation_utils import gather_joint_info
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_OUTPUT_DIR = Path("/home/user/ldl/molmospaces/scripts/InteractiveNav/output")


class SceneOnlyTaskSampler(BaseMujocoTaskSampler):
    """Load a MolmoSpaces scene without sampling a task.

    This keeps the same XML/asset/robot loading path used by datagen and
    InteractiveNav, while skipping navigation-object sampling and robot placement.
    """

    def init_scene(self, env: BaseMujocoEnv) -> None:
        return None

    def randomize_scene(self, env: BaseMujocoEnv, robot_view) -> None:
        return None

    def _sample_task(self, env: BaseMujocoEnv) -> BaseMujocoTask:
        raise NotImplementedError("SceneOnlyTaskSampler only loads scenes.")


def build_scene_config(args: argparse.Namespace) -> NavToObjBaseConfig:
    cfg = NavToObjBaseConfig()
    cfg.seed = args.seed
    cfg.task_type = "nav_to_obj"
    cfg.scene_dataset = args.scene_dataset
    cfg.data_split = args.data_split
    cfg.num_workers = 1
    cfg.use_passive_viewer = False
    cfg.use_filament = False
    cfg.task_sampler_config.task_sampler_class = SceneOnlyTaskSampler
    cfg.task_sampler_config.house_inds = [args.house_ind]
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

    return cfg


def safe_body_aabb(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return body_aabb(model, data, body_id, visual_only=True)
    except Exception as exc:
        log.debug("Failed to compute visual AABB for body %s: %s", body_id, exc)
        return data.xpos[body_id].copy(), np.zeros(3)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def object_record(
    env,
    om,
    object_name: str,
    meta: dict[str, Any],
    force_door: bool = False,
) -> dict[str, Any]:
    model = env.current_model
    data = env.current_data
    body_id = model.body(object_name).id
    center, size = safe_body_aabb(model, data, body_id)
    joints = meta.get("name_map", {}).get("joints", {})
    sites = meta.get("name_map", {}).get("sites", {})
    is_door = force_door
    joint_infos = []
    for joint_name in sorted(joints.keys()):
        try:
            joint_info = gather_joint_info(model, data, joint_name)
        except Exception as exc:
            log.debug("Failed to gather joint info for %s on %s: %s", joint_name, object_name, exc)
            continue
        joint_infos.append(
            {
                "joint_name": str(joint_name),
                "joint_type": joint_type_name(joint_info.get("joint_type")),
                "joint_range": [float(v) for v in joint_info.get("joint_range", [0.0, 0.0])],
                "joint_value": float(joint_info.get("joint_pos", 0.0)),
            }
        )

    return {
        "name": object_name,
        "body_id": int(body_id),
        "object_id": meta.get("object_id"),
        "asset_id": meta.get("asset_id"),
        "category": "Door" if force_door else meta.get("category"),
        "room_id": meta.get("room_id"),
        "parent": meta.get("parent") or None,
        "children": meta.get("children", []),
        "is_static": meta.get("is_static"),
        "is_structural": False if force_door else om.is_structural(object_name),
        "is_receptacle": om.has_receptacle_site(object_name),
        "is_pickup_candidate": om.has_free_joint(object_name),
        "is_articulable": om.is_object_articulable(object_name),
        "is_door": is_door,
        "is_movable_door": force_door,
        "joint_names": sorted(joints.keys()),
        "joint_infos": joint_infos,
        "site_names": sorted(sites.keys()),
        "position": data.xpos[body_id].copy(),
        "aabb_center": center,
        "aabb_size": size,
    }


def is_door_record(object_name: str, meta: dict[str, Any]) -> bool:
    category = str(meta.get("category", "")).lower()
    object_id = str(meta.get("object_id", "")).lower()
    asset_id = str(meta.get("asset_id", "")).lower()
    return (
        object_name.lower().startswith(("door_", "doorway_", "doorframe_"))
        or object_id.startswith("door|")
        or "door" in category
        or "door" in asset_id
    )


def is_non_movable_door_record(rec: dict[str, Any]) -> bool:
    return (not rec["is_movable_door"]) and is_door_record(
        rec["name"],
        {
            "category": rec.get("category"),
            "object_id": rec.get("object_id"),
            "asset_id": rec.get("asset_id"),
        },
    )


def is_window_record(rec: dict[str, Any]) -> bool:
    category = str(rec.get("category", "")).lower()
    object_id = str(rec.get("object_id", "")).lower()
    asset_id = str(rec.get("asset_id", "")).lower()
    name = str(rec.get("name", "")).lower()
    return name.startswith("window_") or "window" in category or "window" in object_id or "window" in asset_id


def should_plot_foreground_record(rec: dict[str, Any], background_mode: str) -> bool:
    if background_mode != "occupancy":
        return True
    return (
        (not rec["is_structural"])
        or rec["is_movable_door"]
        or is_non_movable_door_record(rec)
        or is_window_record(rec)
    )


def should_label_record(rec: dict[str, Any]) -> bool:
    return (
        (not rec["is_structural"])
        or rec["is_movable_door"]
        or is_non_movable_door_record(rec)
        or is_window_record(rec)
    )


def label_priority(rec: dict[str, Any]) -> tuple[int, int, int, int, float]:
    return (
        int(is_non_movable_door_record(rec)),
        int(is_window_record(rec)),
        int(rec["is_movable_door"]),
        int(rec["is_receptacle"] or rec["is_pickup_candidate"]),
        float(rec["aabb_size"][0] * rec["aabb_size"][1]),
    )


def visual_kind(rec: dict[str, Any]) -> str:
    if rec["is_movable_door"]:
        return "movable_door"
    if rec["is_pickup_candidate"]:
        return "pickup"
    if rec["is_receptacle"]:
        return "receptacle"
    if rec["is_articulable"]:
        return "articulable"
    if rec["is_structural"]:
        return "structural"
    return "plain"


def resolve_room_id(room_selector: str | None, objects_meta: dict[str, dict[str, Any]]) -> int | None:
    if room_selector is None:
        return None
    try:
        return int(room_selector)
    except ValueError:
        pass

    matches = []
    needle = room_selector.lower()
    for meta in objects_meta.values():
        room_id = meta.get("room_id")
        object_id = str(meta.get("object_id", "")).lower()
        category = str(meta.get("category", "")).lower()
        asset_id = str(meta.get("asset_id", "")).lower()
        if needle in object_id or needle in category or needle in asset_id:
            if room_id is not None:
                matches.append(int(room_id))
    return matches[0] if matches else None


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


def collect_scene_records(env, room_selector: str | None) -> tuple[int | None, list[dict[str, Any]], dict[int, list[str]]]:
    om = env.object_managers[env.current_batch_index]
    objects_meta = env.current_scene_metadata.get("objects", {})
    room_id = resolve_room_id(room_selector, objects_meta)

    records = []
    room_to_objects: dict[int, list[str]] = defaultdict(list)
    for object_name, meta in objects_meta.items():
        try:
            env.current_model.body(object_name)
        except KeyError:
            continue
        cur_room_id = meta.get("room_id")
        if cur_room_id is not None:
            room_to_objects[int(cur_room_id)].append(object_name)
        if room_id is not None and cur_room_id != room_id:
            continue
        records.append(object_record(env, om, object_name, meta))

    seen_names = {rec["name"] for rec in records}
    try:
        door_names = om.find_door_names()
    except Exception as exc:
        log.warning("Could not enumerate articulated door child bodies: %s", exc)
        door_names = []
    for door_name in door_names:
        if door_name in seen_names:
            continue
        try:
            env.current_model.body(door_name)
        except KeyError:
            continue
        meta = door_parent_metadata(objects_meta, door_name)
        cur_room_id = meta.get("room_id")
        if cur_room_id is not None:
            room_to_objects[int(cur_room_id)].append(door_name)
        if room_id is not None and cur_room_id != room_id:
            continue
        records.append(object_record(env, om, door_name, meta, force_door=True))

    records.sort(key=lambda item: (str(item["room_id"]), str(item["category"]), item["name"]))
    for names in room_to_objects.values():
        names.sort()
    return room_id, records, dict(sorted(room_to_objects.items()))


def joint_type_name(joint_type: Any) -> str:
    text = str(joint_type).lower()
    if "hinge" in text:
        return "hinge"
    if "slide" in text:
        return "slide"
    return "none"


def derive_room_graph_context(
    env,
    scene_dataset: str,
    records: list[dict[str, Any]],
    px_per_m: int = 40,
) -> tuple[dict[str, list[int]], dict[int, str], list[dict[str, Any]]]:
    from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

    map_cls = iTHORMap if "ithor" in scene_dataset.lower() else ProcTHORMap
    try:
        scene_map = map_cls.from_mj_model_path(
            model_path=env.current_model_path,
            agent_radius=0.0,
            px_per_m=px_per_m,
            device_id=None,
        )
    except Exception as exc:
        log.warning("Could not derive room graph context from room map: %s", exc)
        return {}, {}, []

    room_map = getattr(scene_map, "room_map", None)
    if room_map is None:
        return {}, {}, []

    room_connections = {}
    room_id_to_name = {
        int(room_id): str(room_name)
        for room_id, room_name in getattr(scene_map, "room_ids_to_name", {}).items()
    }
    room_entries = []

    for room_id in sorted(int(room) for room in np.unique(room_map).tolist() if int(room) > 0):
        px_points = np.argwhere(room_map == room_id)
        if len(px_points) == 0:
            continue
        world_points = scene_map.pos_px_to_m(px_points[:, :2])
        xs = [float(point[0]) for point in world_points]
        ys = [float(point[1]) for point in world_points]
        center = [float(np.mean(xs)), float(np.mean(ys)), 0.0]
        size = [
            max(float(max(xs) - min(xs)), 1.0 / float(px_per_m)),
            max(float(max(ys) - min(ys)), 1.0 / float(px_per_m)),
            0.1,
        ]
        room_entries.append(
            {
                "room_id": int(room_id),
                "name": room_id_to_name.get(int(room_id), f"room_{room_id}"),
                "center": center,
                "aabb_center": center,
                "aabb_size": size,
                "cell_count": int(len(px_points)),
            }
        )

    for rec in records:
        if not rec.get("is_door"):
            continue
        center = np.array(rec["aabb_center"], dtype=float)
        px = scene_map.pos_m_to_px(center.reshape(1, 3))[0]
        radius = max(2, int(max(float(rec["aabb_size"][0]), float(rec["aabb_size"][1]), 0.1) * px_per_m * 0.5))
        r0 = max(0, int(px[0]) - radius)
        r1 = min(room_map.shape[0], int(px[0]) + radius + 1)
        c0 = max(0, int(px[1]) - radius)
        c1 = min(room_map.shape[1], int(px[1]) + radius + 1)
        window = room_map[r0:r1, c0:c1]
        room_ids = [int(room_id) for room_id in np.unique(window).tolist() if int(room_id) > 0]
        if room_ids:
            room_connections[rec["name"]] = sorted(room_ids)
    return room_connections, room_id_to_name, room_entries


def export_scene_json(
    out_path: Path,
    scene_id: str,
    records: list[dict[str, Any]],
    room_to_objects: dict[int, list[str]],
    room_id_to_name: dict[int, str],
    rooms: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "scene_id": scene_id,
        "metadata": metadata,
        "room_to_objects": {str(int(room_id)): list(names) for room_id, names in room_to_objects.items()},
        "room_id_to_name": {str(int(room_id)): str(name) for room_id, name in room_id_to_name.items()},
        "rooms": rooms,
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))


def scene_id_from_args(args: argparse.Namespace) -> str:
    return f"{args.scene_dataset}_{args.data_split}_{args.house_ind}_{args.variant}"


def default_scene_json_path(args: argparse.Namespace) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{scene_id_from_args(args)}_scene_full.json"


def fmt_vec(vec: np.ndarray, ndigits: int = 3) -> str:
    return "(" + ", ".join(f"{float(v):.{ndigits}f}" for v in vec) + ")"


def print_room_report(env, room_id: int | None, records: list[dict[str, Any]], room_to_objects: dict[int, list[str]]) -> None:
    print("\n=== Scene ===")
    print(f"model_path: {env.current_model_path}")
    print(f"metadata_objects: {len(env.current_scene_metadata.get('objects', {}))}")
    print(f"available_room_ids: {list(room_to_objects.keys())}")

    title = "ALL ROOMS" if room_id is None else f"ROOM {room_id}"
    print(f"\n=== {title} ===")
    print(f"object_count: {len(records)}")

    counts = defaultdict(int)
    for rec in records:
        counts[rec["category"]] += 1
    print("category_counts:")
    for category, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"  - {category}: {count}")

    print("\nobjects:")
    for rec in records:
        flags = []
        if rec["is_structural"]:
            flags.append("structural")
        if rec["is_door"]:
            flags.append("door")
        if rec["is_movable_door"]:
            flags.append("movable_door")
        if rec["is_receptacle"]:
            flags.append("receptacle")
        if rec["is_pickup_candidate"]:
            flags.append("pickup")
        if rec["is_articulable"]:
            flags.append("articulable")
        if rec["joint_names"]:
            flags.append(f"joints={len(rec['joint_names'])}")
        if rec["site_names"]:
            flags.append(f"sites={len(rec['site_names'])}")
        flag_str = ", ".join(flags) if flags else "plain"
        print(
            f"  - {rec['name']} | category={rec['category']} | room={rec['room_id']} | "
            f"static={rec['is_static']} | {flag_str}"
        )
        print(
            f"      pos={fmt_vec(rec['position'])} aabb_center={fmt_vec(rec['aabb_center'])} "
            f"aabb_size={fmt_vec(rec['aabb_size'])}"
        )


def _plot_record_rect(
    ax,
    rec: dict[str, Any],
    color: str,
    alpha: float,
    linewidth: float,
    linestyle: str = "-",
) -> tuple[np.ndarray, np.ndarray]:
    from matplotlib.patches import Rectangle

    center = rec["aabb_center"]
    size = rec["aabb_size"]
    if np.any(size[:2] <= 1e-6):
        center = rec["position"]
        size = np.array([0.15, 0.15, 0.0])

    bottom_left = center[:2] - size[:2] / 2.0
    rect = Rectangle(
        bottom_left,
        max(float(size[0]), 0.03),
        max(float(size[1]), 0.03),
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    return bottom_left, bottom_left + size[:2]


def plot_occupancy_background(
    ax,
    env,
    scene_dataset: str,
    px_per_m: int,
    agent_radius: float,
    timeout_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    from matplotlib.colors import ListedColormap

    from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

    map_cls = iTHORMap if "ithor" in scene_dataset.lower() else ProcTHORMap

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Occupancy background generation exceeded {timeout_s:.1f}s")

    old_handler = None
    try:
        if timeout_s > 0:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        scene_map = map_cls.from_mj_model_path(
            model_path=env.current_model_path,
            agent_radius=agent_radius,
            px_per_m=px_per_m,
            device_id=None,
        )
    except Exception as exc:
        log.warning("Failed to generate occupancy background: %s", exc)
        return None
    finally:
        if timeout_s > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    occupancy = np.asarray(scene_map.occupancy)
    room_map = getattr(scene_map, "room_map", None)
    h, w = occupancy.shape[:2]

    # Do not use imshow(extent=...) here. The map transform swaps row/col and may
    # flip axes, so draw a downsampled quadrilateral mesh directly in world coords.
    max_mesh_side = 900
    stride = max(1, int(np.ceil(max(h, w) / max_mesh_side)))
    row_edges = np.arange(0, h + 1, stride)
    col_edges = np.arange(0, w + 1, stride)
    if row_edges[-1] != h:
        row_edges = np.append(row_edges, h)
    if col_edges[-1] != w:
        col_edges = np.append(col_edges, w)

    rr, cc = np.meshgrid(row_edges, col_edges, indexing="ij")
    edge_row_col = np.stack([rr.reshape(-1), cc.reshape(-1)], axis=1)
    edge_world = scene_map.pos_px_to_m(edge_row_col).reshape(rr.shape + (3,))
    world_x = edge_world[..., 0]
    world_y = edge_world[..., 1]

    sampled = occupancy[row_edges[:-1][:, None], col_edges[None, :-1]].astype(int)
    cmap = ListedColormap(["#2f3437", "#eeeeee"])
    ax.pcolormesh(
        world_x,
        world_y,
        sampled,
        cmap=cmap,
        shading="flat",
        alpha=0.55,
        zorder=0,
        antialiased=False,
    )

    center_rows = (row_edges[:-1] + row_edges[1:]) / 2.0
    center_cols = (col_edges[:-1] + col_edges[1:]) / 2.0
    center_rr, center_cc = np.meshgrid(center_rows, center_cols, indexing="ij")
    center_row_col = np.stack([center_rr.reshape(-1), center_cc.reshape(-1)], axis=1)
    center_world = scene_map.pos_px_to_m(center_row_col).reshape(center_rr.shape + (3,))

    if room_map is not None:
        sampled_room = room_map[row_edges[:-1][:, None], col_edges[None, :-1]]
        room_ids = [rid for rid in sorted(np.unique(sampled_room).tolist()) if rid != 0]
        for rid in room_ids:
            room_mask = (sampled_room == rid).astype(float)
            if np.count_nonzero(room_mask) < 4:
                continue
            ax.contour(
                center_world[..., 0],
                center_world[..., 1],
                room_mask,
                levels=[0.5],
                colors=["#111827"],
                linewidths=1.2,
                alpha=0.85,
                zorder=1,
            )

            pts = center_world[room_mask.astype(bool), :2]
            label_xy = np.mean(pts, axis=0)
            room_name = getattr(scene_map, "room_ids_to_name", {}).get(int(rid), f"room_{rid}")
            ax.text(
                float(label_xy[0]),
                float(label_xy[1]),
                str(room_name),
                fontsize=9,
                color="#111827",
                ha="center",
                va="center",
                weight="bold",
                zorder=2,
            )

    world_xy = edge_world[..., :2].reshape(-1, 2)
    world_min = np.min(world_xy, axis=0)
    world_max = np.max(world_xy, axis=0)
    log.info(
        "Occupancy background: pixels=(%s,%s), stride=%s, world_bounds=%s -> %s",
        h,
        w,
        stride,
        fmt_vec(world_min),
        fmt_vec(world_max),
    )
    return world_min, world_max


def plot_bounds_background(ax, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    from matplotlib.patches import Rectangle

    if not records:
        return None

    mins = []
    maxs = []
    for rec in records:
        center = rec["aabb_center"]
        size = rec["aabb_size"]
        if np.any(size[:2] <= 1e-6):
            center = rec["position"]
            size = np.array([0.15, 0.15, 0.0])
        mins.append(center[:2] - size[:2] / 2.0)
        maxs.append(center[:2] + size[:2] / 2.0)

    xy_min = np.min(np.stack(mins), axis=0)
    xy_max = np.max(np.stack(maxs), axis=0)
    size = xy_max - xy_min
    ax.add_patch(
        Rectangle(
            xy_min,
            float(size[0]),
            float(size[1]),
            facecolor="none",
            edgecolor="#111827",
            linewidth=2.0,
            linestyle="-",
            zorder=0,
        )
    )
    return xy_min, xy_max


def plot_topdown(
    env,
    records: list[dict[str, Any]],
    out_path: Path,
    max_label_count: int = 200,
    background_mode: str = "bounds",
    scene_dataset: str = "procthor-10k",
    occupancy_px_per_m: int = 100,
    occupancy_agent_radius: float = 0.2,
    occupancy_timeout_s: float = 30.0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not records:
        log.warning("No records to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    color_by_kind = {
        "structural": "#6b7280",
        "receptacle": "#2563eb",
        "pickup": "#f97316",
        "articulable": "#dc2626",
        "movable_door": "#7c3aed",
        "plain": "#0f766e",
    }

    xy_min = np.array([np.inf, np.inf])
    xy_max = np.array([-np.inf, -np.inf])
    foreground_records = (
        [rec for rec in records if should_plot_foreground_record(rec, background_mode)]
        if background_mode == "occupancy"
        else records
    )

    if background_mode == "occupancy":
        bg_bounds = plot_occupancy_background(
            ax,
            env,
            scene_dataset=scene_dataset,
            px_per_m=occupancy_px_per_m,
            agent_radius=occupancy_agent_radius,
            timeout_s=occupancy_timeout_s,
        )
        if bg_bounds is not None:
            xy_min = np.minimum(xy_min, bg_bounds[0])
            xy_max = np.maximum(xy_max, bg_bounds[1])
    elif background_mode == "bounds":
        bg_bounds = plot_bounds_background(ax, records)
        if bg_bounds is not None:
            xy_min = np.minimum(xy_min, bg_bounds[0])
            xy_max = np.maximum(xy_max, bg_bounds[1])

    for rec in foreground_records:
        kind = visual_kind(rec)
        linestyle = "--" if rec["is_pickup_candidate"] else ("-" if rec["is_static"] else ":")
        linewidth = (
            2.2
            if kind == "pickup"
            else (
                1.6
                if kind in {"movable_door", "receptacle"}
                else (1.3 if is_non_movable_door_record(rec) or is_window_record(rec) else 0.9)
            )
        )
        alpha = (
            0.26
            if kind != "structural"
            else (0.16 if is_non_movable_door_record(rec) or is_window_record(rec) else 0.08)
        )

        rec_min, rec_max = _plot_record_rect(
            ax,
            rec,
            color=color_by_kind[kind],
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        xy_min = np.minimum(xy_min, rec_min)
        xy_max = np.maximum(xy_max, rec_max)

    label_records = sorted(
        [
            rec
            for rec in foreground_records
            if should_label_record(rec)
        ],
        key=label_priority,
        reverse=True,
    )[:max_label_count]
    for rec in label_records:
        kind = visual_kind(rec)
        if rec["is_movable_door"]:
            label = f"door: {rec['category'] or rec['name']} [movable]"
        elif is_non_movable_door_record(rec):
            label = f"door: {rec['category'] or rec['name']}"
        elif is_window_record(rec):
            label = f"window: {rec['category'] or rec['name']}"
        elif rec["is_receptacle"]:
            label = f"receptacle: {rec['category'] or rec['name']}"
        elif rec["is_pickup_candidate"]:
            label = f"pickup: {rec['category'] or rec['name']}"
        elif rec["is_articulable"]:
            label = str(rec["category"] or rec["name"])
        else:
            label = str(rec["category"] or rec["name"])

        center = rec["aabb_center"]
        size = rec["aabb_size"]
        if np.any(size[:2] <= 1e-6):
            center = rec["position"]
            size = np.array([0.15, 0.15, 0.0])
        label_x = float(center[0])
        label_y = float(center[1] - size[1] / 2.0 - 0.05)
        if is_non_movable_door_record(rec) or is_window_record(rec):
            label_y = float(center[1])
        ax.text(
            label_x,
            label_y,
            label[:28],
            fontsize=5 if rec["is_pickup_candidate"] else 7,
            color=color_by_kind[kind],
            ha="center",
            va="top",
        )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", alpha=0.35, label=kind)
        for kind, color in color_by_kind.items()
    ]
    if background_mode == "occupancy":
        handles.insert(
            0,
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#eeeeee",
                edgecolor="black",
                alpha=0.55,
                label="scene occupancy background",
            ),
        )
    elif background_mode == "bounds":
        handles.insert(
            0,
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="none",
                edgecolor="#111827",
                linewidth=2.0,
                label="scene/room bounds",
            ),
        )
    ax.legend(handles=handles, loc="upper right")
    margin = 0.5
    ax.set_xlim(float(xy_min[0] - margin), float(xy_max[0] + margin))
    ax.set_ylim(float(xy_min[1] - margin), float(xy_max[1] + margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("MolmoSpaces Room Top-down AABB View")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and print MolmoSpaces room/object ground-truth attributes."
    )
    parser.add_argument("--robot", type=str, default="droid", choices=["droid", "rby1", "rum"])
    parser.add_argument("--scene_dataset", type=str, default="procthor-10k")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--house_ind", type=int, default=0)
    parser.add_argument("--variant", type=str, default="base", choices=["base", "ceiling", "map"])
    parser.add_argument(
        "--room",
        type=str,
        default=None,
        help="Room selector. Use a room_id like '2', or omit to print all rooms.",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Only print attributes; do not write the top-down plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the top-down PNG.",
    )
    parser.add_argument(
        "--background_mode",
        type=str,
        default="occupancy",
        choices=["occupancy", "bounds", "none"],
        help="Top-down plot background: whole-scene occupancy map, bounds rectangle, or blank.",
    )
    parser.add_argument(
        "--occupancy_px_per_m",
        type=int,
        default=100,
        help="Resolution used to generate the occupancy-map background.",
    )
    parser.add_argument(
        "--occupancy_agent_radius",
        type=float,
        default=0.0,
        help="Agent radius in meters used to inflate occupancy-map obstacles. Use 0 to disable inflation.",
    )
    parser.add_argument(
        "--occupancy_timeout_s",
        type=float,
        default=30.0,
        help="Maximum seconds to spend generating occupancy background. Use 0 to disable timeout.",
    )
    parser.add_argument(
        "--export_scene_json",
        type=Path,
        default=None,
        help="Path to export the full scene/object GT JSON. Defaults to output/<scene>_scene_full.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_scene_config(args)
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        sampler.update_scene(variant=args.variant)
        env = sampler.env
        room_id, records, room_to_objects = collect_scene_records(env, args.room)
        connected_room_ids, room_id_to_name, rooms = derive_room_graph_context(env, args.scene_dataset, records)
        for record in records:
            if record["name"] in connected_room_ids:
                record["connected_room_ids"] = connected_room_ids[record["name"]]
        print_room_report(env, room_id, records, room_to_objects)

        scene_json_path = args.export_scene_json or default_scene_json_path(args)
        if scene_json_path is not None:
            scene_id = scene_id_from_args(args)
            export_scene_json(
                scene_json_path,
                scene_id=scene_id,
                records=records,
                room_to_objects=room_to_objects,
                room_id_to_name=room_id_to_name,
                rooms=rooms,
                metadata={
                    "scene_dataset": args.scene_dataset,
                    "data_split": args.data_split,
                    "house_ind": int(args.house_ind),
                    "variant": args.variant,
                    "room_selector": args.room,
                    "selected_room_id": room_id,
                    "model_path": str(env.current_model_path),
                },
            )
            print(f"\nScene JSON saved to: {scene_json_path}")

        if not args.no_plot:
            if args.room is not None and room_id not in room_to_objects:
                log.error(
                    "Requested room %s does not exist in this scene. Available room ids: %s",
                    args.room,
                    list(room_to_objects.keys()),
                )
                return
            if not records:
                log.error("No records to plot; no image will be saved.")
                return
            room_label = "all_rooms" if room_id is None else f"room_{room_id}"
            out_folder_path = args.output
            out_path = out_folder_path / f"{args.scene_dataset}_{args.data_split}_{args.house_ind}_{args.variant}_{room_label}_{args.background_mode}.png"
            plot_topdown(
                env,
                records,
                out_path,
                background_mode=args.background_mode,
                scene_dataset=args.scene_dataset,
                occupancy_px_per_m=args.occupancy_px_per_m,
                occupancy_agent_radius=args.occupancy_agent_radius,
                occupancy_timeout_s=args.occupancy_timeout_s,
            )
            print(f"\nTop-down plot saved to: {out_path}")
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
