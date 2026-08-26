from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.utils.pose import pos_quat_to_pose_mat
from scripts.InteractiveNav import build_container_interaction_benchmark as container_builder
from scripts.InteractiveNav import build_mixed_interaction_benchmark as mixed_builder
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("molmo_spaces.env.camera_manager").setLevel(logging.WARNING)

DEFAULT_BENCHMARK = mixed_builder.DEFAULT_OUTPUT_DIR / "benchmark.json"
DEFAULT_OUTPUT_DIR = mixed_builder.DEFAULT_OUTPUT_DIR / "visualizations"
VISUALIZATION_SCHEMA_VERSION = "mixed_interaction_topdown_visualization_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )
    temporary.chmod(0o644)
    temporary.replace(path)


def safe_slug(value: str, max_len: int = 100) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return (slug or "episode")[:max_len]


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(token.strip()) for token in value.split(",") if token.strip()}


def resolve_benchmark_path(path: Path) -> Path:
    return path / "benchmark.json" if path.is_dir() else path


def load_benchmark_episodes(path: Path) -> list[dict[str, Any]]:
    benchmark_path = resolve_benchmark_path(path)
    payload = json.loads(benchmark_path.read_text())
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    if not isinstance(episodes, list):
        raise ValueError(f"Benchmark does not contain an episode list: {benchmark_path}")
    return list(episodes)


def extract_episode_annotations(episode: dict[str, Any]) -> dict[str, Any]:
    nav = episode.get("interactive_nav")
    if not isinstance(nav, dict) or nav.get("schema_version") != "interactive_nav_v3":
        raise ValueError("Visualization requires an interactive_nav_v3 episode")
    domains = set(nav.get("interaction_domains", []))
    if domains != {"channel", "container"}:
        raise ValueError(f"Visualization requires a mixed episode, got domains={sorted(domains)}")

    target = nav["target"]
    validation = nav["generation_validation"]["navigation_validation"]
    interactions = nav["interactions"]
    channel_interactions = [
        row for row in interactions if str(row.get("type", "")).startswith("channel_")
    ]
    container_interactions = [
        row for row in interactions if str(row.get("type", "")).startswith("container_")
    ]
    if not channel_interactions or not container_interactions:
        raise ValueError("Mixed episode must contain both channel and container interactions")

    required_roots = list(nav["initial_state"].get("required_door_roots_closed", []))
    if not required_roots:
        required_roots = list(
            nav["generation_validation"]
            .get("door_state_validation", {})
            .get("required_closed_root_names", [])
        )
    if not required_roots:
        raise ValueError("Mixed episode does not identify its required closed door root")

    start_pose = episode["task"]["robot_base_pose"]
    interaction_pose = validation["interaction_pose"]
    approach = validation["door_approach"]
    return {
        "case_id": nav["case_id"],
        "house_index": int(episode["house_index"]),
        "target_name": target["selected_instance"],
        "target_category": target["category"],
        "target_center": np.asarray(target["object_aabb_center"], dtype=float),
        "container_name": target["container_name"],
        "container_category": target["container_category"],
        "container_center": np.asarray(target["container_aabb_center"], dtype=float),
        "channel_object_names": [row["object_name"] for row in channel_interactions],
        "required_door_roots": required_roots,
        "crossed_door_roots": list(validation["all_open_path_crossed_door_roots"]),
        "start_pose": np.asarray(start_pose, dtype=float),
        "start_xy": np.asarray(start_pose[:2], dtype=float),
        "interaction_pose": np.asarray(interaction_pose, dtype=float),
        "interaction_xy": np.asarray(interaction_pose[:2], dtype=float),
        "door_approach_xy": np.asarray(approach["approach_xy"], dtype=float),
        "recorded_gt_path_length_m": float(validation["all_open_path_length_m"]),
        "recorded_approach_path_length_m": float(validation["approach_path_length_m"]),
        "recorded_initial_path_found": bool(validation["initial_state_path_found"]),
        "interactions": interactions,
    }


def select_episode_rows(
    episodes: list[dict[str, Any]],
    *,
    episode_indices: set[int] | None,
    house_indices: set[int] | None,
    case_ids: set[str] | None,
    max_episodes: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, episode in enumerate(episodes):
        if episode_indices is not None and index not in episode_indices:
            continue
        if house_indices is not None and int(episode["house_index"]) not in house_indices:
            continue
        case_id = episode.get("interactive_nav", {}).get("case_id")
        if case_ids is not None and case_id not in case_ids:
            continue
        selected.append((index, episode))
        if max_episodes is not None and len(selected) >= max_episodes:
            break
    return selected


def set_joint_position_by_name(env, joint_name: str, value: float) -> None:
    model = env.current_model
    data = env.current_data
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Episode articulation joint was not found: {joint_name}")
    joint_type = int(model.jnt_type[joint_id])
    if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError(f"Unsupported episode articulation joint type={joint_type}: {joint_name}")
    qpos_addr = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_addr] = float(value)


def apply_episode_initial_state(ctx: probe.LoadedContext, episode: dict[str, Any]) -> dict[str, int]:
    env = ctx.env
    modifications = episode.get("scene_modifications", {})
    applied_poses = 0
    for object_name, pose_7d in modifications.get("object_poses", {}).items():
        pose = pos_quat_to_pose_mat(np.asarray(pose_7d, dtype=float))
        if not probe.set_free_joint_pose(env, object_name, pose):
            raise ValueError(f"Episode free-joint object was not found: {object_name}")
        applied_poses += 1

    applied_joints = 0
    for state in modifications.get("articulation_states", []):
        set_joint_position_by_name(env, state["joint_name"], float(state["position"]))
        applied_joints += 1
    mujoco.mj_forward(env.current_model, env.current_data)
    return {"object_pose_count": applied_poses, "articulation_state_count": applied_joints}


def collect_plot_records(ctx: probe.LoadedContext, doorway_analysis: dict[str, Any] | None):
    records = emi.collect_scene_plot_records(ctx.env)
    records.extend(emi.collect_non_interactive_doorway_object_records(ctx.env, doorway_analysis))
    return emi.dedupe_plot_records(records)


def object_role(name: str, annotations: dict[str, Any]) -> str:
    if name == annotations["target_name"]:
        return "target"
    if name == annotations["container_name"]:
        return "container"
    if name in annotations["required_door_roots"]:
        return "required_channel_root"
    if name in annotations["channel_object_names"]:
        return "required_channel_leaf"
    if name in annotations["crossed_door_roots"]:
        return "crossed_channel_root"
    return "scene_object"


def serialize_box_record(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    if rec is None:
        return None
    return {
        "aabb_center": np.asarray(rec["aabb_center"], dtype=float).tolist(),
        "aabb_size": np.asarray(rec["aabb_size"], dtype=float).tolist(),
        "position": np.asarray(rec["position"], dtype=float).tolist(),
    }


def build_object_catalog(
    initial_records: list[dict[str, Any]],
    open_records: list[dict[str, Any]],
    annotations: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    initial_by_name = {row["name"]: row for row in initial_records}
    open_by_name = {row["name"]: row for row in open_records}
    names = sorted(set(initial_by_name) | set(open_by_name))
    plot_ids = {name: f"O{index:03d}" for index, name in enumerate(names, start=1)}
    catalog = []
    for name in names:
        rec = initial_by_name.get(name) or open_by_name[name]
        catalog.append(
            {
                "plot_id": plot_ids[name],
                "name": name,
                "category": rec.get("category"),
                "role": object_role(name, annotations),
                "is_structural": bool(rec.get("is_structural", False)),
                "is_receptacle": bool(rec.get("is_receptacle", False)),
                "is_pickup_candidate": bool(rec.get("is_pickup_candidate", False)),
                "is_articulable": bool(rec.get("is_articulable", False)),
                "initial_state": serialize_box_record(initial_by_name.get(name)),
                "channel_open_state": serialize_box_record(open_by_name.get(name)),
            }
        )
    return plot_ids, catalog


def role_style(role: str) -> dict[str, Any] | None:
    return {
        "target": {"color": "#f59e0b", "linewidth": 3.0, "alpha": 1.0, "label": "TARGET"},
        "container": {"color": "#9333ea", "linewidth": 2.8, "alpha": 1.0, "label": "CONTAINER"},
        "required_channel_root": {
            "color": "#dc2626",
            "linewidth": 3.2,
            "alpha": 1.0,
            "label": "REQUIRED DOOR",
        },
        "required_channel_leaf": {
            "color": "#ef4444",
            "linewidth": 2.5,
            "alpha": 1.0,
            "label": "DOOR LEAF",
        },
        "crossed_channel_root": {
            "color": "#0ea5e9",
            "linewidth": 2.0,
            "alpha": 0.9,
            "label": "CROSSED DOOR",
        },
    }.get(role)


def draw_object_boxes(
    ax,
    scene_map,
    records: list[dict[str, Any]],
    plot_ids: dict[str, str],
    annotations: dict[str, Any],
    *,
    label_all_objects: bool,
    include_structural: bool,
) -> None:
    from matplotlib.patches import Rectangle

    for rec in records:
        if rec.get("is_structural", False) and not include_structural:
            continue
        box = emi.object_box_to_px(scene_map, rec)
        if box is None:
            continue
        col, row, width, height = box
        role = object_role(rec["name"], annotations)
        highlighted = role_style(role)
        if highlighted is None:
            kind = emi.scene_visual_kind(rec)
            color = emi.scene_plot_color(kind)
            linewidth = 0.45 if kind == "structural" else 0.75
            alpha = 0.22 if kind == "structural" else 0.42
            label = None
        else:
            color = highlighted["color"]
            linewidth = highlighted["linewidth"]
            alpha = highlighted["alpha"]
            label = highlighted["label"]

        rect = Rectangle(
            (col, row),
            max(width, 2.0),
            max(height, 2.0),
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            linestyle="--" if rec.get("is_pickup_candidate", False) and highlighted is None else "-",
            alpha=alpha,
            zorder=5 if highlighted is not None else 2,
        )
        ax.add_patch(rect)

        plot_id = plot_ids[rec["name"]]
        if highlighted is not None:
            category = str(rec.get("category") or "object")
            text = f"{plot_id} {label}\n{category}"
            fontsize = 6.5
            text_color = color
            bbox_alpha = 0.88
            zorder = 8
        elif label_all_objects:
            text = plot_id
            fontsize = 3.8
            text_color = color
            bbox_alpha = 0.42
            zorder = 3
        else:
            continue
        ax.text(
            col + width / 2.0,
            row + height / 2.0,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            zorder=zorder,
            bbox={"facecolor": (1.0, 1.0, 1.0, bbox_alpha), "edgecolor": "none", "pad": 0.5},
        )


def draw_xy_marker(ax, scene_map, xy: np.ndarray, **scatter_kwargs) -> None:
    px = emi.points_xy_to_px(scene_map, np.asarray([xy], dtype=float))
    if px is None:
        return
    ax.scatter(px[:, 1], px[:, 0], **scatter_kwargs)


def draw_path(ax, scene_map, path: np.ndarray | None, **plot_kwargs) -> None:
    px = emi.points_xy_to_px(scene_map, path)
    if px is None:
        return
    ax.plot(px[:, 1], px[:, 0], **plot_kwargs)


def draw_start_heading(ax, scene_map, pose_7d: np.ndarray) -> None:
    start_xy = pose_7d[:2]
    yaw = float(R.from_quat(pose_7d[3:7], scalar_first=True).as_euler("xyz")[2])
    heading_xy = start_xy + 0.45 * np.asarray([math.cos(yaw), math.sin(yaw)])
    pixels = emi.points_xy_to_px(scene_map, np.asarray([start_xy, heading_xy]))
    if pixels is None:
        return
    row0, col0 = pixels[0]
    row1, col1 = pixels[1]
    ax.annotate(
        "",
        xy=(col1, row1),
        xytext=(col0, row0),
        arrowprops={"arrowstyle": "-|>", "color": "#16a34a", "lw": 2.0},
        zorder=10,
    )


def panel_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, loc="left")


def draw_panel_base(
    ax,
    scene_map,
    records: list[dict[str, Any]],
    plot_ids: dict[str, str],
    annotations: dict[str, Any],
    *,
    label_all_objects: bool,
    include_structural: bool,
) -> None:
    background = emi.make_scene_plot_background(scene_map)
    ax.imshow(background, origin="upper")
    draw_object_boxes(
        ax,
        scene_map,
        records,
        plot_ids,
        annotations,
        label_all_objects=label_all_objects,
        include_structural=include_structural,
    )
    draw_start_heading(ax, scene_map, annotations["start_pose"])
    draw_xy_marker(
        ax,
        scene_map,
        annotations["start_xy"],
        marker="o",
        s=62,
        c="#16a34a",
        edgecolors="black",
        linewidths=0.8,
        zorder=11,
    )
    draw_xy_marker(
        ax,
        scene_map,
        annotations["door_approach_xy"],
        marker="D",
        s=52,
        c="#f97316",
        edgecolors="black",
        linewidths=0.7,
        zorder=11,
    )
    draw_xy_marker(
        ax,
        scene_map,
        annotations["interaction_xy"],
        marker="*",
        s=115,
        c="#9333ea",
        edgecolors="black",
        linewidths=0.8,
        zorder=11,
    )
    draw_xy_marker(
        ax,
        scene_map,
        annotations["target_center"][:2],
        marker="X",
        s=74,
        c="#f59e0b",
        edgecolors="black",
        linewidths=0.7,
        zorder=11,
    )
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(background.shape[0], 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("top-down map x (px)")
    ax.set_ylabel("top-down map y (px)")
    ax.grid(False)


def interaction_chain_text(annotations: dict[str, Any]) -> str:
    chain = []
    for index, interaction in enumerate(annotations["interactions"], start=1):
        category = interaction.get("object_category") or interaction["object_name"]
        chain.append(f"{index}. {interaction['type']} ({category}, joint={interaction['joint_index']})")
    return " -> ".join(chain)


def render_episode_figure(
    out_path: Path,
    *,
    initial_map,
    open_map,
    initial_records: list[dict[str, Any]],
    open_records: list[dict[str, Any]],
    plot_ids: dict[str, str],
    annotations: dict[str, Any],
    initial_full_path: np.ndarray | None,
    approach_path: np.ndarray | None,
    gt_path: np.ndarray,
    recomputed_gt_length_m: float,
    label_all_objects: bool,
    include_structural: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(20, 10.8))
    draw_panel_base(
        axes[0],
        initial_map,
        initial_records,
        plot_ids,
        annotations,
        label_all_objects=label_all_objects,
        include_structural=include_structural,
    )
    draw_path(
        axes[0],
        initial_map,
        approach_path,
        color="#f97316",
        linewidth=2.4,
        linestyle="--",
        zorder=9,
    )
    if initial_full_path is not None:
        draw_path(
            axes[0],
            initial_map,
            initial_full_path,
            color="#64748b",
            linewidth=2.0,
            linestyle=":",
            zorder=8,
        )
    panel_title(
        axes[0],
        "A. Initial benchmark state",
        "required channel door CLOSED; path to container interaction pose is BLOCKED",
    )

    draw_panel_base(
        axes[1],
        open_map,
        open_records,
        plot_ids,
        annotations,
        label_all_objects=label_all_objects,
        include_structural=include_structural,
    )
    draw_path(
        axes[1],
        open_map,
        gt_path,
        color="#2563eb",
        linewidth=3.0,
        zorder=9,
    )
    panel_title(
        axes[1],
        "B. Oracle channel-open state",
        "recomputed GT path reaches the measured container interaction pose",
    )

    legend_handles = [
        Patch(facecolor="#2f3437", edgecolor="black", label="occupied"),
        Patch(facecolor="none", edgecolor="#64748b", linewidth=0.8, label="all scene object AABBs (Oxxx)"),
        Patch(facecolor="none", edgecolor="#dc2626", linewidth=2.8, label="required channel door"),
        Patch(facecolor="none", edgecolor="#9333ea", linewidth=2.8, label="container interaction object"),
        Patch(facecolor="none", edgecolor="#f59e0b", linewidth=2.8, label="target object"),
        Line2D([0], [0], color="#f97316", linewidth=2.4, linestyle="--", label="initial approach path"),
        Line2D([0], [0], color="#2563eb", linewidth=3.0, label="GT path after door interaction"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markeredgecolor="black", markersize=8, label="start + heading"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#f97316", markeredgecolor="black", markersize=7, label="door approach pose"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#9333ea", markeredgecolor="black", markersize=11, label="container interaction pose"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#f59e0b", markeredgecolor="black", markersize=8, label="target center"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=7.5,
        framealpha=0.96,
        bbox_to_anchor=(0.5, 0.045),
    )

    case_id = annotations["case_id"]
    fig.suptitle(
        f"Mixed InteractiveNav V3 | house={annotations['house_index']} | target={annotations['target_category']}\n"
        f"{case_id}",
        fontsize=14,
        y=0.985,
    )
    chain = interaction_chain_text(annotations)
    metrics = (
        f"interaction chain: {chain}   |   "
        f"GT length recorded={annotations['recorded_gt_path_length_m']:.3f} m, "
        f"recomputed={recomputed_gt_length_m:.3f} m   |   "
        f"object labels Oxxx are fully resolved in the adjacent JSON sidecar"
    )
    fig.text(
        0.5,
        0.005,
        textwrap.fill(metrics, 190),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.12, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    out_path.chmod(0o644)


def compare_path_length(
    recorded: float,
    recomputed: float,
    tolerance_m: float,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    absolute_error = abs(float(recorded) - float(recomputed))
    relative_error = absolute_error / max(abs(float(recorded)), 1e-9)
    allowed_error = max(float(tolerance_m), abs(float(recorded)) * float(relative_tolerance))
    return {
        "recorded_m": float(recorded),
        "recomputed_m": float(recomputed),
        "absolute_error_m": absolute_error,
        "relative_error": relative_error,
        "absolute_tolerance_m": float(tolerance_m),
        "relative_tolerance": float(relative_tolerance),
        "allowed_error_m": allowed_error,
        "passed": absolute_error <= allowed_error,
    }


def episode_output_stem(episode_index: int, annotations: dict[str, Any]) -> str:
    return (
        f"episode_{episode_index:04d}_h{annotations['house_index']}_"
        f"{safe_slug(annotations['case_id'], 72)}"
    )


def load_existing_result(
    args: argparse.Namespace,
    *,
    episode_index: int,
    episode: dict[str, Any],
) -> dict[str, Any] | None:
    annotations = extract_episode_annotations(episode)
    stem = episode_output_stem(episode_index, annotations)
    image_path = args.output_dir / f"{stem}.png"
    sidecar_path = args.output_dir / f"{stem}.json"
    if not image_path.is_file() or not sidecar_path.is_file():
        return None
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("schema_version") != VISUALIZATION_SCHEMA_VERSION:
        return None
    previous_check = sidecar["path_validation"]["gt_path_length"]
    length_check = compare_path_length(
        annotations["recorded_gt_path_length_m"],
        float(previous_check["recomputed_m"]),
        args.path_length_tolerance_m,
        args.path_length_relative_tolerance,
    )
    if args.strict_path_length and not length_check["passed"]:
        return None
    sidecar["path_validation"]["gt_path_length"] = length_check
    write_json(sidecar_path, sidecar)
    return {
        "benchmark_episode_index": episode_index,
        "case_id": annotations["case_id"],
        "house_index": annotations["house_index"],
        "target_category": annotations["target_category"],
        "container_category": annotations["container_category"],
        "interaction_types": [row["type"] for row in annotations["interactions"]],
        "object_box_count": len(sidecar["objects"]),
        "image": image_path.name,
        "sidecar": sidecar_path.name,
        "path_validation": length_check,
        "reused_existing": True,
    }


def render_one_episode(
    args: argparse.Namespace,
    *,
    episode_index: int,
    episode: dict[str, Any],
    ctx: probe.LoadedContext,
) -> dict[str, Any]:
    annotations = extract_episode_annotations(episode)
    applied = apply_episode_initial_state(ctx, episode)

    initial_map, doorway_analysis = emi.build_live_procthor_map(
        ctx.env.current_model,
        ctx.env.current_data,
        model_path=str(ctx.env.current_model_path),
        px_per_m=args.px_per_m,
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
        open_threshold=args.open_threshold,
        treat_all_non_interactive_doorways_as_open=True,
        return_doorway_analysis=True,
    )
    initial_records = collect_plot_records(ctx, doorway_analysis)
    initial_full_path = emi.compute_path_from_map(
        initial_map,
        annotations["start_xy"],
        annotations["interaction_xy"],
        downscale_factor=1,
    )
    approach_path = emi.compute_path_from_map(
        initial_map,
        annotations["start_xy"],
        annotations["door_approach_xy"],
        downscale_factor=1,
    )
    if annotations["recorded_initial_path_found"] != (initial_full_path is not None):
        raise ValueError(
            "Rebuilt initial-state reachability does not match the V3 navigation validation"
        )
    if approach_path is None:
        raise ValueError("Rebuilt initial state cannot reach the recorded door approach pose")

    door_transitions = []
    for root_name in annotations["required_door_roots"]:
        door_transitions.append(
            emi.set_door_root_state(ctx.env, doorway_analysis, root_name, "open")
        )
    open_map = emi.build_live_procthor_map(
        ctx.env.current_model,
        ctx.env.current_data,
        model_path=str(ctx.env.current_model_path),
        px_per_m=args.px_per_m,
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
        open_threshold=args.open_threshold,
        treat_all_non_interactive_doorways_as_open=True,
    )
    open_records = collect_plot_records(ctx, doorway_analysis)
    gt_path = emi.compute_path_from_map(
        open_map,
        annotations["start_xy"],
        annotations["interaction_xy"],
        downscale_factor=1,
    )
    if gt_path is None:
        raise ValueError("Opening the required channel did not restore the GT path")
    recomputed_gt_length_m = emi.path_length(gt_path)
    assert recomputed_gt_length_m is not None
    length_check = compare_path_length(
        annotations["recorded_gt_path_length_m"],
        recomputed_gt_length_m,
        args.path_length_tolerance_m,
        args.path_length_relative_tolerance,
    )
    if args.strict_path_length and not length_check["passed"]:
        raise ValueError(
            "Recomputed GT path length differs from the V3 record by "
            f"{length_check['absolute_error_m']:.3f} m"
        )

    plot_ids, object_catalog = build_object_catalog(
        initial_records, open_records, annotations
    )
    filename = episode_output_stem(episode_index, annotations)
    image_path = args.output_dir / f"{filename}.png"
    sidecar_path = args.output_dir / f"{filename}.json"
    render_episode_figure(
        image_path,
        initial_map=initial_map,
        open_map=open_map,
        initial_records=initial_records,
        open_records=open_records,
        plot_ids=plot_ids,
        annotations=annotations,
        initial_full_path=initial_full_path,
        approach_path=approach_path,
        gt_path=gt_path,
        recomputed_gt_length_m=recomputed_gt_length_m,
        label_all_objects=args.label_all_objects,
        include_structural=args.include_structural,
    )
    sidecar = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "benchmark_episode_index": episode_index,
        "case_id": annotations["case_id"],
        "house_index": annotations["house_index"],
        "image": image_path.name,
        "scene_state_application": applied,
        "path_validation": {
            "initial_full_path_found": initial_full_path is not None,
            "initial_approach_path_length_m": emi.path_length(approach_path),
            "gt_path_length": length_check,
        },
        "annotations": {
            "start_pose": annotations["start_pose"].tolist(),
            "door_approach_xy": annotations["door_approach_xy"].tolist(),
            "container_interaction_pose": annotations["interaction_pose"].tolist(),
            "target_name": annotations["target_name"],
            "container_name": annotations["container_name"],
            "required_door_roots": annotations["required_door_roots"],
            "crossed_door_roots": annotations["crossed_door_roots"],
            "interaction_chain": annotations["interactions"],
        },
        "door_open_transitions": door_transitions,
        "objects": object_catalog,
    }
    write_json(sidecar_path, sidecar)
    return {
        "benchmark_episode_index": episode_index,
        "case_id": annotations["case_id"],
        "house_index": annotations["house_index"],
        "target_category": annotations["target_category"],
        "container_category": annotations["container_category"],
        "interaction_types": [row["type"] for row in annotations["interactions"]],
        "object_box_count": len(object_catalog),
        "image": image_path.name,
        "sidecar": sidecar_path.name,
        "path_validation": length_check,
        "reused_existing": False,
    }


def make_contact_sheet(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    columns: int,
    max_images: int,
) -> None:
    if not rows or max_images <= 0:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selected = rows[:max_images]
    columns = max(1, min(columns, len(selected)))
    grid_rows = int(math.ceil(len(selected) / columns))
    fig, axes = plt.subplots(grid_rows, columns, figsize=(columns * 8.0, grid_rows * 4.7))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for ax, row in zip(axes_array, selected):
        image = plt.imread(output_path.parent / row["image"])
        stride = max(1, int(math.ceil(max(image.shape[:2]) / 1400)))
        ax.imshow(image[::stride, ::stride])
        ax.set_title(
            f"episode {row['benchmark_episode_index']} | house {row['house_index']} | "
            f"{row['target_category']}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes_array[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Mixed InteractiveNav V3 annotated top-down gallery", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=120, facecolor="white")
    plt.close(fig)
    output_path.chmod(0o644)


def write_html_index(output_path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        interactions = " &rarr; ".join(html.escape(value) for value in row["interaction_types"])
        check = row["path_validation"]
        cards.append(
            f"""
            <article class="card">
              <a href="{html.escape(row['image'])}"><img src="{html.escape(row['image'])}" loading="lazy"></a>
              <h2>Episode {row['benchmark_episode_index']} · House {row['house_index']}</h2>
              <p><code>{html.escape(row['case_id'])}</code></p>
              <p>{html.escape(row['target_category'])} in {html.escape(row['container_category'])}<br>{interactions}</p>
              <p>{row['object_box_count']} object boxes · GT Δ={check['absolute_error_m']:.3f} m · <a href="{html.escape(row['sidecar'])}">annotations JSON</a></p>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mixed InteractiveNav V3 visualizations</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f1f5f9; color: #0f172a; font: 14px/1.45 system-ui, sans-serif; }}
    h1 {{ margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 20px; }}
    .card {{ background: white; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; box-shadow: 0 2px 8px #0f172a14; }}
    .card img {{ width: 100%; height: auto; border: 1px solid #e2e8f0; }}
    .card h2 {{ margin: 10px 0 4px; font-size: 17px; }}
    .card p {{ margin: 5px 0; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Mixed InteractiveNav V3 annotated top-down gallery</h1>
  <p>Each image contrasts the frozen initial state with the oracle channel-open state. Oxxx labels resolve to the per-episode JSON sidecar.</p>
  <main class="grid">{''.join(cards)}</main>
</body>
</html>
"""
    output_path.write_text(document)
    output_path.chmod(0o644)


def run(args: argparse.Namespace) -> int:
    episodes = load_benchmark_episodes(args.benchmark)
    selected = select_episode_rows(
        episodes,
        episode_indices=parse_int_set(args.episode_indices),
        house_indices=parse_int_set(args.house_indices),
        case_ids=set(args.case_id) if args.case_id else None,
        max_episodes=args.max_episodes,
    )
    if not selected:
        raise ValueError("No benchmark episodes matched the requested filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    current_house = None
    ctx = None
    started_at = time.perf_counter()
    try:
        for progress_index, (episode_index, episode) in enumerate(selected, start=1):
            house_index = int(episode["house_index"])
            if args.reuse_existing:
                reused = load_existing_result(
                    args,
                    episode_index=episode_index,
                    episode=episode,
                )
                if reused is not None:
                    results.append(reused)
                    print(
                        f"[{progress_index}/{len(selected)}] reused episode={episode_index} "
                        f"house={house_index} boxes={reused['object_box_count']}",
                        flush=True,
                    )
                    continue
            if current_house != house_index:
                if ctx is not None:
                    probe.close_context(ctx)
                ctx = container_builder.load_episode_context(args, episode)
                current_house = house_index
            episode_started = time.perf_counter()
            try:
                row = render_one_episode(
                    args,
                    episode_index=episode_index,
                    episode=episode,
                    ctx=ctx,
                )
                row["elapsed_sec"] = time.perf_counter() - episode_started
                results.append(row)
                print(
                    f"[{progress_index}/{len(selected)}] rendered episode={episode_index} "
                    f"house={house_index} boxes={row['object_box_count']} "
                    f"gt_error={row['path_validation']['absolute_error_m']:.3f}m "
                    f"elapsed={row['elapsed_sec']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "benchmark_episode_index": episode_index,
                        "house_index": house_index,
                        "case_id": episode.get("interactive_nav", {}).get("case_id"),
                        "error": str(exc),
                    }
                )
                print(
                    f"[{progress_index}/{len(selected)}] failed episode={episode_index} "
                    f"house={house_index}: {exc}",
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
    finally:
        if ctx is not None:
            probe.close_context(ctx)

    contact_sheet = args.output_dir / "contact_sheet.png"
    make_contact_sheet(
        contact_sheet,
        results,
        columns=args.contact_sheet_columns,
        max_images=args.contact_sheet_max_images,
    )
    html_index = args.output_dir / "index.html"
    write_html_index(html_index, results)
    manifest = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "benchmark": str(resolve_benchmark_path(args.benchmark)),
        "requested_episode_count": len(selected),
        "rendered_episode_count": len(results),
        "failed_episode_count": len(errors),
        "px_per_m": args.px_per_m,
        "label_all_objects": args.label_all_objects,
        "include_structural": args.include_structural,
        "contact_sheet": contact_sheet.name if results else None,
        "html_index": html_index.name,
        "elapsed_sec": time.perf_counter() - started_at,
        "episodes": results,
        "errors": errors,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if results and not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render mixed interactive_nav_v3 episodes as annotated initial/GT top-down maps "
            "with all scene object AABBs."
        )
    )
    parser.add_argument("benchmark", type=Path, nargs="?", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode_indices", help="Comma-separated zero-based benchmark indices")
    parser.add_argument("--house_indices", help="Comma-separated house indices")
    parser.add_argument("--case_id", action="append", default=[])
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--px_per_m", type=int, default=100)
    parser.add_argument("--open_threshold", type=float, default=0.67)
    parser.add_argument("--path_length_tolerance_m", type=float, default=0.35)
    parser.add_argument("--path_length_relative_tolerance", type=float, default=0.10)
    parser.add_argument(
        "--strict_path_length", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--label_all_objects", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include_structural", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--continue_on_error", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reuse_existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trust and reuse matching PNG/JSON pairs already present in output_dir.",
    )
    parser.add_argument("--contact_sheet_columns", type=int, default=2)
    parser.add_argument("--contact_sheet_max_images", type=int, default=10)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
