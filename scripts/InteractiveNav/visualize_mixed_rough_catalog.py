from __future__ import annotations

import argparse
import html
import json
import logging
import math
import multiprocessing
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import benchmark_door_state_scan as door_scan
from scripts.InteractiveNav import build_container_interaction_benchmark as container_builder
from scripts.InteractiveNav import collect_mixed_rough_catalog as mixed_rough
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi
from molmo_spaces.utils.pose import pos_quat_to_pose_mat


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("molmo_spaces.env.camera_manager").setLevel(logging.WARNING)

DEFAULT_CATALOG = mixed_rough.DEFAULT_OUTPUT_DIR / "mixed_rough_catalog.json"
DEFAULT_OUTPUT_DIR = mixed_rough.DEFAULT_OUTPUT_DIR / "crossing_only_visualizations"
VISUALIZATION_SCHEMA_VERSION = "mixed_rough_topdown_visualization_v2"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )
    temporary.chmod(0o644)
    temporary.replace(path)


def safe_slug(value: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return (slug or "candidate")[:max_len]


def selected_mixed_door_root(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("selected_mixed_door_root")
        or candidate["selected_crossed_door_root"]
    )


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(token.strip()) for token in value.split(",") if token.strip()}


def load_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "mixed_rough_catalog_v1":
        raise ValueError(f"Expected mixed_rough_catalog_v1: {path}")
    if not isinstance(payload.get("candidates"), list):
        raise ValueError(f"Catalog does not contain a candidate list: {path}")
    return payload


def path_length_bin(length_m: float) -> str:
    if length_m < 5.0:
        return "lt_5m"
    if length_m < 8.0:
        return "5_to_8m"
    if length_m < 12.0:
        return "8_to_12m"
    if length_m < 20.0:
        return "12_to_20m"
    return "ge_20m"


def crossed_door_count_bin(count: int) -> str:
    if count <= 1:
        return "1_door"
    if count == 2:
        return "2_doors"
    return "3plus_doors"


def candidate_features(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(candidate.get("container_category") or "unknown"),
        path_length_bin(float(candidate["all_open_path_length_m"])),
        crossed_door_count_bin(len(candidate.get("crossed_door_roots", []))),
        str(candidate.get("target_category") or "unknown"),
    )


def balanced_select_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_samples: int | None,
    seed: int,
    unique_houses: bool,
) -> list[dict[str, Any]]:
    """Greedily cover rare door counts, path bins, container types, and targets."""
    if max_samples is None or max_samples >= len(candidates):
        return list(candidates)
    if max_samples <= 0:
        return []

    rng = random.Random(seed)
    rows = [dict(row) for row in candidates]
    tie_break = {int(row["_candidate_index"]): rng.random() for row in rows}
    feature_rows = {int(row["_candidate_index"]): candidate_features(row) for row in rows}
    feature_totals = [Counter(values[index] for values in feature_rows.values()) for index in range(4)]
    selected_counts = [Counter() for _ in range(4)]
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    selected_houses: set[int] = set()

    # Door-count rarity is weighted most heavily so that the five 3+-door cases
    # are not hidden by the 585 one-door cases in the default 692-pair pool.
    rarity_weights = (2.0, 4.0, 10.0, 1.5)
    balance_weights = (16.0, 7.0, 12.0, 2.0)

    while len(selected) < max_samples:
        available = [
            row for row in rows if int(row["_candidate_index"]) not in selected_indices
        ]
        if not available:
            break
        if unique_houses:
            unseen_house_rows = [
                row for row in available if int(row["house_index"]) not in selected_houses
            ]
            if unseen_house_rows:
                available = unseen_house_rows

        container_categories = sorted(feature_totals[0])
        minimum_category_count = min(
            selected_counts[0][category] for category in container_categories
        )
        category_balanced_rows = [
            row
            for row in available
            if selected_counts[0][candidate_features(row)[0]] == minimum_category_count
        ]
        if category_balanced_rows:
            available = category_balanced_rows

        best_row = None
        best_key = None
        for row in available:
            candidate_index = int(row["_candidate_index"])
            features = feature_rows[candidate_index]
            score = 0.0
            for feature_index, feature_value in enumerate(features):
                score += rarity_weights[feature_index] / float(
                    feature_totals[feature_index][feature_value]
                )
                score += balance_weights[feature_index] / float(
                    1 + selected_counts[feature_index][feature_value]
                )
            key = (score, tie_break[candidate_index], -candidate_index)
            if best_key is None or key > best_key:
                best_key = key
                best_row = row

        assert best_row is not None
        selected.append(best_row)
        candidate_index = int(best_row["_candidate_index"])
        selected_indices.add(candidate_index)
        selected_houses.add(int(best_row["house_index"]))
        for feature_index, feature_value in enumerate(feature_rows[candidate_index]):
            selected_counts[feature_index][feature_value] += 1

    return selected


def select_candidate_rows(
    catalog: dict[str, Any],
    *,
    candidate_type: str,
    case_ids: set[str] | None,
    house_indices: set[int] | None,
    max_samples: int | None,
    seed: int,
    unique_houses: bool,
) -> list[dict[str, Any]]:
    rows = []
    for candidate_index, candidate in enumerate(catalog["candidates"]):
        if candidate_type != "all" and candidate.get("rough_candidate_type") != candidate_type:
            continue
        if case_ids is not None and candidate.get("case_id") not in case_ids:
            continue
        if house_indices is not None and int(candidate["house_index"]) not in house_indices:
            continue
        rows.append({**candidate, "_candidate_index": candidate_index})
    if not rows:
        raise ValueError("No rough candidates matched the requested filters")

    if case_ids is not None or house_indices is not None:
        rows.sort(key=lambda row: (int(row["house_index"]), int(row["_candidate_index"])))
        return rows if max_samples is None else rows[:max_samples]
    return balanced_select_candidates(
        rows,
        max_samples=max_samples,
        seed=seed,
        unique_houses=unique_houses,
    )


def collect_plot_records(ctx: probe.LoadedContext, doorway_analysis: dict[str, Any] | None):
    records = emi.collect_scene_plot_records(ctx.env)
    records.extend(emi.collect_non_interactive_doorway_object_records(ctx.env, doorway_analysis))
    return emi.dedupe_plot_records(records)


def record_role(name: str, annotations: dict[str, Any]) -> str:
    if name == annotations["target_name"]:
        return "target"
    if name == annotations["container_name"]:
        return "container"
    if name == annotations["selected_door_root"]:
        return "selected_crossed_door"
    if name in annotations["crossed_door_roots"]:
        return "other_crossed_door"
    return "scene_object"


def role_style(role: str) -> dict[str, Any] | None:
    return {
        "target": {
            "color": "#f59e0b",
            "linewidth": 3.0,
            "label": "TARGET OBJECT",
        },
        "container": {
            "color": "#9333ea",
            "linewidth": 3.0,
            "label": "CONTAINER",
        },
        "selected_crossed_door": {
            "color": "#dc2626",
            "linewidth": 3.3,
            "label": "CLOSED IN PANEL B",
        },
        "other_crossed_door": {
            "color": "#0ea5e9",
            "linewidth": 2.5,
            "label": "OTHER CROSSED DOOR",
        },
    }.get(role)


def serialize_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "aabb_center": np.asarray(record["aabb_center"], dtype=float).tolist(),
        "aabb_size": np.asarray(record["aabb_size"], dtype=float).tolist(),
        "position": np.asarray(record["position"], dtype=float).tolist(),
    }


def build_object_catalog(
    open_records: list[dict[str, Any]],
    closed_records: list[dict[str, Any]],
    annotations: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    open_by_name = {row["name"]: row for row in open_records}
    closed_by_name = {row["name"]: row for row in closed_records}
    names = sorted(set(open_by_name) | set(closed_by_name))
    plot_ids = {name: f"O{index:03d}" for index, name in enumerate(names, start=1)}
    catalog = []
    for name in names:
        row = open_by_name.get(name) or closed_by_name[name]
        catalog.append(
            {
                "plot_id": plot_ids[name],
                "name": name,
                "category": row.get("category"),
                "role": record_role(name, annotations),
                "is_structural": bool(row.get("is_structural", False)),
                "is_receptacle": bool(row.get("is_receptacle", False)),
                "is_pickup_candidate": bool(row.get("is_pickup_candidate", False)),
                "is_articulable": bool(row.get("is_articulable", False)),
                "all_open_state": serialize_record(open_by_name.get(name)),
                "selected_door_closed_state": serialize_record(closed_by_name.get(name)),
            }
        )
    return plot_ids, catalog


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

    for record in records:
        if record.get("is_structural", False) and not include_structural:
            continue
        box = emi.object_box_to_px(scene_map, record)
        if box is None:
            continue
        col, row, width, height = box
        role = record_role(record["name"], annotations)
        highlighted = role_style(role)
        if highlighted is None:
            kind = emi.scene_visual_kind(record)
            color = emi.scene_plot_color(kind)
            linewidth = 0.42 if kind == "structural" else 0.72
            alpha = 0.18 if kind == "structural" else 0.38
        else:
            color = highlighted["color"]
            linewidth = highlighted["linewidth"]
            alpha = 1.0

        ax.add_patch(
            Rectangle(
                (col, row),
                max(width, 2.0),
                max(height, 2.0),
                facecolor="none",
                edgecolor=color,
                linewidth=linewidth,
                linestyle=(
                    "--"
                    if record.get("is_pickup_candidate", False) and highlighted is None
                    else "-"
                ),
                alpha=alpha,
                zorder=6 if highlighted is not None else 2,
            )
        )

        plot_id = plot_ids[record["name"]]
        if highlighted is not None:
            category = str(record.get("category") or "object")
            label = f"{plot_id} {highlighted['label']}\n{category}"
            font_size = 6.2
            text_color = color
            box_alpha = 0.90
            zorder = 9
        elif label_all_objects:
            label = plot_id
            font_size = 3.5
            text_color = color
            box_alpha = 0.40
            zorder = 3
        else:
            continue
        ax.text(
            col + width / 2.0,
            row + height / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            color=text_color,
            zorder=zorder,
            bbox={"facecolor": (1.0, 1.0, 1.0, box_alpha), "edgecolor": "none", "pad": 0.5},
        )


def draw_xy_marker(ax, scene_map, xy: np.ndarray | None, **kwargs) -> None:
    if xy is None:
        return
    pixels = emi.points_xy_to_px(scene_map, np.asarray([xy], dtype=float))
    if pixels is not None:
        ax.scatter(pixels[:, 1], pixels[:, 0], **kwargs)


def draw_path(ax, scene_map, path: np.ndarray | None, **kwargs) -> None:
    pixels = emi.points_xy_to_px(scene_map, path)
    if pixels is not None:
        ax.plot(pixels[:, 1], pixels[:, 0], **kwargs)


def draw_start_heading(ax, scene_map, pose_7d: np.ndarray) -> None:
    start_xy = pose_7d[:2]
    w, x, y, z = pose_7d[3:7]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
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
        zorder=12,
    )


def door_facing_robot_pose(
    ctx: probe.LoadedContext,
    approach_xy: np.ndarray,
    door_center_xy: np.ndarray,
) -> np.ndarray:
    direction = np.asarray(door_center_xy, dtype=float) - np.asarray(approach_xy, dtype=float)
    if float(np.linalg.norm(direction)) < 1e-6:
        raise ValueError("Door approach and door center coincide; cannot define camera yaw")
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    current_pose = np.asarray(ctx.env.current_robot.robot_view.base.pose, dtype=float)
    return np.asarray(
        [
            float(approach_xy[0]),
            float(approach_xy[1]),
            float(current_pose[2, 3]),
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ],
        dtype=float,
    )


def render_head_camera_at_pose(
    ctx: probe.LoadedContext,
    robot_pose_7d: np.ndarray,
    *,
    camera_name: str,
) -> np.ndarray:
    env = ctx.env
    robot_pose_before = np.asarray(env.current_robot.robot_view.base.pose, dtype=float).copy()
    try:
        env.current_robot.robot_view.base.pose = pos_quat_to_pose_mat(robot_pose_7d)
        mujoco.mj_forward(env.current_model, env.current_data)
        env.camera_manager.registry.update_all_cameras(env)
        return np.asarray(env.render_rgb_frame(camera_name)).copy()
    finally:
        env.current_robot.robot_view.base.pose = robot_pose_before
        mujoco.mj_forward(env.current_model, env.current_data)
        env.camera_manager.registry.update_all_cameras(env)


def render_first_person_figure(
    output_path: Path,
    *,
    open_rgb: np.ndarray,
    closed_rgb: np.ndarray,
    candidate: dict[str, Any],
    camera_pose_7d: np.ndarray,
    camera_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    for ax, frame, title in (
        (axes[0], open_rgb, "A. Selected doorway open"),
        (axes[1], closed_rgb, "B. Selected doorway closed"),
    ):
        ax.imshow(frame)
        ax.axhline(frame.shape[0] / 2.0, color="white", linewidth=0.7, alpha=0.55)
        ax.axvline(frame.shape[1] / 2.0, color="white", linewidth=0.7, alpha=0.55)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(
        f"Mixed rough door-front first person | house {candidate['house_index']} | "
        f"{candidate['target_category']} in {candidate['container_category']}\n"
        f"camera={camera_name} · door={selected_mixed_door_root(candidate)}",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.015,
        "Camera pose faces the selected door-root AABB center from the catalog's geometric "
        f"path-backoff hint: xy=({camera_pose_7d[0]:.2f}, {camera_pose_7d[1]:.2f}). "
        "This is a diagnostic view, not a manipulation-validated interaction pose.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)
    output_path.chmod(0o644)


def draw_first_person_camera_pose(ax, scene_map, camera_pose_7d: np.ndarray) -> None:
    camera_xy = np.asarray(camera_pose_7d[:2], dtype=float)
    w, x, y, z = np.asarray(camera_pose_7d[3:7], dtype=float)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_xy = camera_xy + 0.85 * np.asarray([math.cos(yaw), math.sin(yaw)])
    left_xy = camera_xy + 0.65 * np.asarray(
        [math.cos(yaw + math.radians(34.0)), math.sin(yaw + math.radians(34.0))]
    )
    right_xy = camera_xy + 0.65 * np.asarray(
        [math.cos(yaw - math.radians(34.0)), math.sin(yaw - math.radians(34.0))]
    )
    pixels = emi.points_xy_to_px(
        scene_map, np.asarray([camera_xy, forward_xy, left_xy, right_xy], dtype=float)
    )
    if pixels is None:
        return
    camera_row, camera_col = pixels[0]
    ax.scatter(
        [camera_col],
        [camera_row],
        marker="^",
        s=115,
        c="#06b6d4",
        edgecolors="black",
        linewidths=1.0,
        zorder=15,
    )
    for endpoint in pixels[1:]:
        ax.plot(
            [camera_col, endpoint[1]],
            [camera_row, endpoint[0]],
            color="#06b6d4",
            linewidth=2.0 if np.array_equal(endpoint, pixels[1]) else 1.1,
            linestyle="-" if np.array_equal(endpoint, pixels[1]) else "--",
            zorder=14,
        )
    ax.annotate(
        "FP CAMERA",
        xy=(camera_col, camera_row),
        xytext=(8, 8),
        textcoords="offset points",
        color="#075985",
        fontsize=8.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#06b6d4", "alpha": 0.9},
        zorder=16,
    )


def render_first_person_topdown_composite(
    output_path: Path,
    *,
    open_map,
    open_records: list[dict[str, Any]],
    plot_ids: dict[str, str],
    annotations: dict[str, Any],
    open_path: np.ndarray,
    open_rgb: np.ndarray,
    closed_rgb: np.ndarray,
    camera_pose_7d: np.ndarray,
    candidate: dict[str, Any],
    label_all_objects: bool,
    include_structural: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(23, 8.2))
    draw_panel_base(
        axes[0],
        open_map,
        open_records,
        plot_ids,
        annotations,
        label_all_objects=label_all_objects,
        include_structural=include_structural,
    )
    draw_path(
        axes[0],
        open_map,
        open_path,
        color="#2563eb",
        linewidth=3.0,
        zorder=10,
    )
    draw_first_person_camera_pose(axes[0], open_map, camera_pose_7d)
    axes[0].set_title(
        "A. Annotated top-down\ncyan triangle/rays = first-person camera pose",
        fontsize=12,
        loc="left",
    )
    for ax, frame, title in (
        (axes[1], open_rgb, "B. First person: selected door open"),
        (axes[2], closed_rgb, "C. First person: selected door closed"),
    ):
        ax.imshow(frame)
        ax.axhline(frame.shape[0] / 2.0, color="white", linewidth=0.7, alpha=0.55)
        ax.axvline(frame.shape[1] / 2.0, color="white", linewidth=0.7, alpha=0.55)
        ax.set_title(title, fontsize=12, loc="left")
        ax.axis("off")
    fig.suptitle(
        f"Mixed occupancy / doorway diagnostic | house {candidate['house_index']} | "
        f"{candidate['target_category']} in {candidate['container_category']}\n"
        f"door={selected_mixed_door_root(candidate)}",
        fontsize=14,
    )
    wall_stats = getattr(open_map, "wall_slice_stats", {})
    fig.text(
        0.5,
        0.012,
        "Top-down occupancy includes robot-height collision-wall slices: "
        f"height={wall_stats.get('height_m', 'n/a')} m, "
        f"segments={wall_stats.get('slice_segment_count', 'n/a')}, "
        f"rasterized pixels={wall_stats.get('rasterized_pixel_count', 'n/a')}.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=165, facecolor="white")
    plt.close(fig)
    output_path.chmod(0o644)


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
        zorder=12,
    )
    draw_xy_marker(
        ax,
        scene_map,
        annotations.get("door_approach_xy"),
        marker="D",
        s=50,
        c="#f97316",
        edgecolors="black",
        linewidths=0.7,
        zorder=12,
    )
    draw_xy_marker(
        ax,
        scene_map,
        annotations["goal_xy"],
        marker="*",
        s=120,
        c="#9333ea",
        edgecolors="black",
        linewidths=0.8,
        zorder=12,
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
        zorder=12,
    )
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(background.shape[0], 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("top-down map x (px)")
    ax.set_ylabel("top-down map y (px)")
    ax.grid(False)


def render_candidate_figure(
    output_path: Path,
    *,
    open_map,
    closed_map,
    open_records: list[dict[str, Any]],
    closed_records: list[dict[str, Any]],
    plot_ids: dict[str, str],
    annotations: dict[str, Any],
    open_path: np.ndarray,
    closed_path: np.ndarray | None,
    open_length_m: float,
    closed_length_m: float | None,
    reachability_matches_catalog: bool,
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
        open_map,
        open_records,
        plot_ids,
        annotations,
        label_all_objects=label_all_objects,
        include_structural=include_structural,
    )
    draw_path(
        axes[0],
        open_map,
        open_path,
        color="#2563eb",
        linewidth=3.0,
        zorder=10,
    )
    axes[0].set_title(
        "A. All interactive doors open\n"
        f"rough GT planner path crosses {len(annotations['crossed_door_roots'])} door root(s); "
        f"length={open_length_m:.2f} m",
        fontsize=11,
        loc="left",
    )

    draw_panel_base(
        axes[1],
        closed_map,
        closed_records,
        plot_ids,
        annotations,
        label_all_objects=label_all_objects,
        include_structural=include_structural,
    )
    draw_path(
        axes[1],
        closed_map,
        open_path,
        color="#64748b",
        linewidth=1.8,
        linestyle=":",
        alpha=0.90,
        zorder=8,
    )
    draw_path(
        axes[1],
        closed_map,
        closed_path,
        color="#db2777",
        linewidth=3.0,
        zorder=10,
    )
    if closed_length_m is None:
        closed_subtitle = "goal is unreachable after closing the selected crossed door"
    else:
        closed_subtitle = (
            f"alternate path still reaches the container; length={closed_length_m:.2f} m "
            f"(geometric length delta {closed_length_m - open_length_m:+.2f} m)"
        )
    axes[1].set_title(
        "B. Selected crossed door closed\n" + closed_subtitle,
        fontsize=11,
        loc="left",
    )

    legend_handles = [
        Patch(facecolor="#2f3437", edgecolor="black", label="occupied"),
        Patch(facecolor="none", edgecolor="#64748b", linewidth=0.8, label="scene object AABBs (Oxxx)"),
        Patch(facecolor="none", edgecolor="#dc2626", linewidth=3.0, label="selected crossed door"),
        Patch(facecolor="none", edgecolor="#0ea5e9", linewidth=2.5, label="other crossed doors"),
        Patch(facecolor="none", edgecolor="#9333ea", linewidth=3.0, label="container"),
        Patch(facecolor="none", edgecolor="#f59e0b", linewidth=3.0, label="target object"),
        Line2D([0], [0], color="#2563eb", linewidth=3.0, label="all-open rough GT path"),
        Line2D([0], [0], color="#db2777", linewidth=3.0, label="closed-door alternate path"),
        Line2D([0], [0], color="#64748b", linewidth=1.8, linestyle=":", label="original route over closed map"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markeredgecolor="black", markersize=8, label="start + heading"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#f97316", markeredgecolor="black", markersize=7, label="rough door approach"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#9333ea", markeredgecolor="black", markersize=11, label="rough container goal"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#f59e0b", markeredgecolor="black", markersize=8, label="target center"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        fontsize=7.3,
        framealpha=0.96,
        bbox_to_anchor=(0.5, 0.047),
    )

    replay_status = "" if reachability_matches_catalog else " | CATALOG/REPLAY MISMATCH"
    candidate_type_label = str(annotations.get("rough_candidate_type", "unknown")).replace(
        "_", " "
    )
    fig.suptitle(
        f"Mixed rough {candidate_type_label}{replay_status} | sample {annotations['sample_rank']:02d} | "
        f"candidate {annotations['candidate_index']} | house {annotations['house_index']}\n"
        f"{annotations['target_category']} in {annotations['container_category']} | "
        f"selected door={annotations['selected_door_root']}",
        fontsize=14,
        y=0.985,
    )
    if reachability_matches_catalog:
        if annotations.get("rough_candidate_type") == "mixed_shortcut_verified":
            caveat = (
                "Why this is a mixed shortcut sample: the all-open GT planner path crosses an interactive door, "
                "and closing that door still leaves a route to the rough container goal, but the route is longer. "
                "This is rough evidence only: the goal is nearest free space to the container AABB, and target "
                "visibility/reveal plus manipulation-valid poses are not yet V3 fine-data truth."
            )
        else:
            caveat = (
                "Why this is a mixed crossing sample: the all-open GT planner path crosses an interactive door. "
                "This is rough evidence only: the goal is nearest free space to the container AABB, and target "
                "visibility/reveal plus manipulation-valid poses are not yet V3 fine-data truth."
            )
    else:
        caveat = (
            "REVALIDATION WARNING: the rough catalog recorded the selected-door closed path as found, "
            "but the current independent replay produced different reachability. Treat this sample as an "
            "unstable rough-map boundary case and rescan it before using the candidate label."
        )
    fig.text(
        0.5,
        0.006,
        textwrap.fill(caveat, 205),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.89, bottom=0.13, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)
    output_path.chmod(0o644)


def compare_path_length(
    recorded: float,
    recomputed: float,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    error = abs(float(recorded) - float(recomputed))
    allowed = max(absolute_tolerance_m, abs(float(recorded)) * relative_tolerance)
    return {
        "recorded_m": float(recorded),
        "recomputed_m": float(recomputed),
        "absolute_error_m": error,
        "allowed_error_m": allowed,
        "passed": error <= allowed,
    }


def candidate_output_stem(sample_rank: int, candidate: dict[str, Any]) -> str:
    return (
        f"sample_{sample_rank:03d}_candidate_{int(candidate['_candidate_index']):04d}_"
        f"h{int(candidate['house_index'])}_{safe_slug(candidate['case_id'], 72)}"
    )


def render_one_candidate(
    args: argparse.Namespace,
    *,
    sample_rank: int,
    candidate: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    ctx = None
    try:
        ctx = container_builder.load_episode_context(args, episode)
        container_builder.open_all_available_doors(ctx)
        _, initial_containers = probe.collect_scene_records(ctx)
        container_builder.close_all_containers(ctx.env, initial_containers)
        records, containers = probe.collect_scene_records(ctx)
        containers_by_name = {row["name"]: row for row in containers}
        objects_by_name = {row["name"]: row for row in records}

        container = containers_by_name.get(candidate["container_name"])
        target = objects_by_name.get(candidate["object_name"])
        if container is None:
            raise ValueError(f"Container missing in live scene: {candidate['container_name']}")
        if target is None:
            raise ValueError(f"Target missing in live scene: {candidate['object_name']}")

        open_map, doorway_analysis = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        open_records = collect_plot_records(ctx, doorway_analysis)
        start_pose = np.asarray(candidate["source_robot_base_pose"], dtype=float)
        start_xy = start_pose[:2]
        goal_xy = np.asarray(candidate["rough_nav_goal_xy"], dtype=float)
        open_path = emi.compute_path_from_map(open_map, start_xy, goal_xy, downscale_factor=1)
        if open_path is None:
            raise ValueError("Recomputed all-open path was not found")
        open_length_m = emi.path_length(open_path)
        assert open_length_m is not None
        length_check = compare_path_length(
            float(candidate["all_open_path_length_m"]),
            open_length_m,
            args.path_length_tolerance_m,
            args.path_length_relative_tolerance,
        )
        if args.strict_validation and not length_check["passed"]:
            raise ValueError(
                "Recomputed open path length differs from rough catalog by "
                f"{length_check['absolute_error_m']:.3f} m"
            )

        recomputed_crossed = door_scan.traversed_interactive_doors_on_path(
            ctx.env,
            doorway_analysis,
            open_path,
            padding_m=args.door_on_path_padding_m,
            sample_step_m=args.path_region_sample_step_m,
        )
        recomputed_crossed_names = [row["name"] for row in recomputed_crossed]
        selected_door_root = selected_mixed_door_root(candidate)
        all_live_door_records = emi.collect_interactive_door_root_object_records(
            ctx.env, doorway_analysis
        )
        selected_door_record = next(
            (row for row in all_live_door_records if row["name"] == selected_door_root),
            None,
        )
        if args.strict_validation and selected_door_root not in recomputed_crossed_names:
            raise ValueError(
                f"Selected door root no longer intersects the open path: {selected_door_root}"
            )

        selected_check = next(
            (
                row
                for row in candidate.get("door_requirement_checks", [])
                if row.get("door_root_name") == selected_door_root
            ),
            None,
        )
        recomputed_approach = (
            None
            if selected_door_record is None
            else mixed_rough.path_door_approach(
                open_path,
                selected_door_record,
                padding_m=args.door_on_path_padding_m,
                sample_step_m=args.path_region_sample_step_m,
                standoff_m=args.door_approach_standoff_m,
            )
        )
        approach = (
            recomputed_approach
            or candidate.get("selected_door_approach")
            or candidate.get("selected_crossed_door_approach")
            or {}
        )
        door_approach_xy = approach.get("approach_xy")
        first_person_pose = None
        first_person_open_rgb = None
        if args.render_first_person:
            if door_approach_xy is None:
                raise ValueError("First-person rendering requires a door approach hint")
            if selected_door_record is None:
                raise ValueError(
                    f"First-person rendering could not resolve door root: {selected_door_root}"
                )
            first_person_pose = door_facing_robot_pose(
                ctx,
                np.asarray(door_approach_xy, dtype=float),
                np.asarray(
                    selected_door_record.get(
                        "portal_center_xy", selected_door_record["aabb_center"]
                    ),
                    dtype=float,
                )[:2],
            )
            first_person_open_rgb = render_head_camera_at_pose(
                ctx,
                first_person_pose,
                camera_name=args.first_person_camera,
            )
        door_transition = emi.set_door_root_state(
            ctx.env, doorway_analysis, selected_door_root, "closed"
        )
        closed_map = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
        )
        closed_records = collect_plot_records(ctx, doorway_analysis)
        closed_path = emi.compute_path_from_map(
            closed_map, start_xy, goal_xy, downscale_factor=1
        )
        closed_length_m = emi.path_length(closed_path)
        first_person_closed_rgb = None
        if args.render_first_person:
            assert first_person_pose is not None
            first_person_closed_rgb = render_head_camera_at_pose(
                ctx,
                first_person_pose,
                camera_name=args.first_person_camera,
            )
        expected_closed_path = None if selected_check is None else selected_check.get("closed_path_found")
        reachability_matches_catalog = (
            expected_closed_path is None
            or bool(expected_closed_path) == (closed_path is not None)
        )
        if (
            args.strict_validation
            and not reachability_matches_catalog
        ):
            raise ValueError(
                "Recomputed selected-door reachability differs from rough catalog: "
                f"recorded={expected_closed_path}, recomputed={closed_path is not None}"
            )

        annotations = {
            "sample_rank": sample_rank,
            "candidate_index": int(candidate["_candidate_index"]),
            "case_id": candidate["case_id"],
            "house_index": int(candidate["house_index"]),
            "target_name": candidate["object_name"],
            "target_category": candidate["target_category"],
            "target_center": np.asarray(target["aabb_center"], dtype=float),
            "container_name": candidate["container_name"],
            "container_category": candidate["container_category"],
            "container_center": np.asarray(container["aabb_center"], dtype=float),
            "start_pose": start_pose,
            "start_xy": start_xy,
            "goal_xy": goal_xy,
            "door_approach_xy": (
                None if door_approach_xy is None else np.asarray(door_approach_xy, dtype=float)
            ),
            "selected_door_root": selected_door_root,
            "crossed_door_roots": recomputed_crossed_names,
            "catalog_crossed_door_roots": list(candidate["crossed_door_roots"]),
            "rough_candidate_type": candidate.get("rough_candidate_type", "unknown"),
        }
        plot_ids, object_catalog = build_object_catalog(
            open_records, closed_records, annotations
        )
        stem = candidate_output_stem(sample_rank, candidate)
        image_path = args.output_dir / f"{stem}.png"
        sidecar_path = args.output_dir / f"{stem}.json"
        first_person_path = (
            args.output_dir / f"{stem}_first_person.png"
            if args.render_first_person
            else None
        )
        first_person_topdown_path = (
            args.output_dir / f"{stem}_first_person_topdown.png"
            if args.render_first_person
            else None
        )
        render_candidate_figure(
            image_path,
            open_map=open_map,
            closed_map=closed_map,
            open_records=open_records,
            closed_records=closed_records,
            plot_ids=plot_ids,
            annotations=annotations,
            open_path=open_path,
            closed_path=closed_path,
            open_length_m=open_length_m,
            closed_length_m=closed_length_m,
            reachability_matches_catalog=reachability_matches_catalog,
            label_all_objects=args.label_all_objects,
            include_structural=args.include_structural,
        )
        if first_person_path is not None:
            assert first_person_open_rgb is not None
            assert first_person_closed_rgb is not None
            assert first_person_pose is not None
            render_first_person_figure(
                first_person_path,
                open_rgb=first_person_open_rgb,
                closed_rgb=first_person_closed_rgb,
                candidate=candidate,
                camera_pose_7d=first_person_pose,
                camera_name=args.first_person_camera,
            )
            assert first_person_topdown_path is not None
            render_first_person_topdown_composite(
                first_person_topdown_path,
                open_map=open_map,
                open_records=open_records,
                plot_ids=plot_ids,
                annotations=annotations,
                open_path=open_path,
                open_rgb=first_person_open_rgb,
                closed_rgb=first_person_closed_rgb,
                camera_pose_7d=first_person_pose,
                candidate=candidate,
                label_all_objects=args.label_all_objects,
                include_structural=args.include_structural,
            )

        sidecar = {
            "schema_version": VISUALIZATION_SCHEMA_VERSION,
            "sample_rank": sample_rank,
            "candidate_index": int(candidate["_candidate_index"]),
            "case_id": candidate["case_id"],
            "house_index": int(candidate["house_index"]),
            "image": image_path.name,
            "first_person": (
                None
                if first_person_path is None
                else {
                    "image": first_person_path.name,
                    "topdown_composite_image": first_person_topdown_path.name,
                    "camera_name": args.first_person_camera,
                    "robot_pose": first_person_pose.tolist(),
                    "pose_semantics": "geometric_path_backoff_hint_facing_door_root_aabb_center",
                }
            ),
            "rough_candidate": {
                key: value for key, value in candidate.items() if not key.startswith("_")
            },
            "recomputed": {
                "all_open_path": np.asarray(open_path, dtype=float).tolist(),
                "all_open_path_length": length_check,
                "all_open_wall_slice_stats": getattr(
                    open_map, "wall_slice_stats", None
                ),
                "all_open_crossed_door_roots": recomputed_crossed_names,
                "selected_door_root": selected_door_root,
                "selected_door_catalog_check": selected_check,
                "selected_door_recomputed_approach": recomputed_approach,
                "selected_door_closed_path_found": closed_path is not None,
                "selected_door_reachability_matches_catalog": reachability_matches_catalog,
                "selected_door_closed_path_length_m": closed_length_m,
                "selected_door_closed_wall_slice_stats": getattr(
                    closed_map, "wall_slice_stats", None
                ),
                "selected_door_closed_path": (
                    None if closed_path is None else np.asarray(closed_path, dtype=float).tolist()
                ),
                "selected_door_transition": door_transition,
            },
            "objects": object_catalog,
        }
        write_json(sidecar_path, sidecar)
        return {
            "sample_rank": sample_rank,
            "candidate_index": int(candidate["_candidate_index"]),
            "case_id": candidate["case_id"],
            "house_index": int(candidate["house_index"]),
            "target_category": candidate["target_category"],
            "container_category": candidate["container_category"],
            "rough_candidate_type": candidate.get("rough_candidate_type", "unknown"),
            "path_length_bin": path_length_bin(open_length_m),
            "crossed_door_count": len(candidate["crossed_door_roots"]),
            "selected_door_root": selected_door_root,
            "open_path_length_m": open_length_m,
            "closed_path_length_m": closed_length_m,
            "detour_length_m": (
                None if closed_length_m is None else closed_length_m - open_length_m
            ),
            "reachability_matches_catalog": reachability_matches_catalog,
            "object_box_count": len(object_catalog),
            "image": image_path.name,
            "first_person_image": (
                None if first_person_path is None else first_person_path.name
            ),
            "first_person_topdown_image": (
                None
                if first_person_topdown_path is None
                else first_person_topdown_path.name
            ),
            "sidecar": sidecar_path.name,
            "path_validation": length_check,
        }
    finally:
        if ctx is not None:
            probe.close_context(ctx)


def render_worker(
    args_payload: dict[str, Any],
    sample_rank: int,
    candidate: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", str(args_payload["mujoco_gl"]))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return render_one_candidate(
        argparse.Namespace(**args_payload),
        sample_rank=sample_rank,
        candidate=candidate,
        episode=episode,
    )


def load_existing_result(
    args: argparse.Namespace,
    *,
    sample_rank: int,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    stem = candidate_output_stem(sample_rank, candidate)
    image_path = args.output_dir / f"{stem}.png"
    sidecar_path = args.output_dir / f"{stem}.json"
    if not image_path.is_file() or not sidecar_path.is_file():
        return None
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("schema_version") != VISUALIZATION_SCHEMA_VERSION:
        return None
    first_person = sidecar.get("first_person")
    first_person_image = None if first_person is None else first_person.get("image")
    first_person_topdown_image = (
        None if first_person is None else first_person.get("topdown_composite_image")
    )
    if args.render_first_person and (
        first_person_image is None
        or not (args.output_dir / first_person_image).is_file()
        or first_person_topdown_image is None
        or not (args.output_dir / first_person_topdown_image).is_file()
    ):
        return None
    recomputed = sidecar["recomputed"]
    length_check = compare_path_length(
        float(candidate["all_open_path_length_m"]),
        float(recomputed["all_open_path_length"]["recomputed_m"]),
        args.path_length_tolerance_m,
        args.path_length_relative_tolerance,
    )
    closed_length_m = recomputed.get("selected_door_closed_path_length_m")
    reachability_matches_catalog = bool(
        recomputed.get("selected_door_reachability_matches_catalog", True)
    )
    return {
        "sample_rank": sample_rank,
        "candidate_index": int(candidate["_candidate_index"]),
        "case_id": candidate["case_id"],
        "house_index": int(candidate["house_index"]),
        "target_category": candidate["target_category"],
        "container_category": candidate["container_category"],
        "path_length_bin": path_length_bin(length_check["recomputed_m"]),
        "crossed_door_count": len(candidate["crossed_door_roots"]),
        "selected_door_root": candidate["selected_crossed_door_root"],
        "open_path_length_m": length_check["recomputed_m"],
        "closed_path_length_m": closed_length_m,
        "detour_length_m": (
            None if closed_length_m is None else closed_length_m - length_check["recomputed_m"]
        ),
        "reachability_matches_catalog": reachability_matches_catalog,
        "object_box_count": len(sidecar["objects"]),
        "image": image_path.name,
        "first_person_image": first_person_image,
        "first_person_topdown_image": first_person_topdown_image,
        "sidecar": sidecar_path.name,
        "path_validation": length_check,
        "reused_existing": True,
    }


def make_contact_sheets(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    columns: int,
    page_size: int,
) -> list[str]:
    if not rows or page_size <= 0:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_names = []
    for page_index, start in enumerate(range(0, len(rows), page_size), start=1):
        selected = rows[start : start + page_size]
        page_columns = max(1, min(columns, len(selected)))
        page_rows = int(math.ceil(len(selected) / page_columns))
        fig, axes = plt.subplots(
            page_rows,
            page_columns,
            figsize=(page_columns * 9.0, page_rows * 5.2),
        )
        axes_array = np.atleast_1d(axes).reshape(-1)
        for ax, row in zip(axes_array, selected):
            image = plt.imread(output_dir / row["image"])
            stride = max(1, int(math.ceil(max(image.shape[:2]) / 1600)))
            ax.imshow(image[::stride, ::stride])
            detour = row["detour_length_m"]
            detour_text = "blocked" if detour is None else f"length delta {detour:+.1f}m"
            if not row.get("reachability_matches_catalog", True):
                detour_text += " · REPLAY MISMATCH"
            ax.set_title(
                f"#{row['sample_rank']:02d} · h{row['house_index']} · "
                f"{row['container_category']}/{row['target_category']} · "
                f"{row['crossed_door_count']} door(s) · {detour_text}",
                fontsize=9,
            )
            ax.axis("off")
        for ax in axes_array[len(selected) :]:
            ax.axis("off")
        candidate_types = sorted(
            {str(row.get("rough_candidate_type", "unknown")).replace("_", " ") for row in selected}
        )
        candidate_type_label = ", ".join(candidate_types)
        fig.suptitle(
            f"Mixed rough {candidate_type_label}: all-open GT route vs closed-door replan",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        output_path = output_dir / f"contact_sheet_{page_index:02d}.png"
        fig.savefig(output_path, dpi=120, facecolor="white")
        plt.close(fig)
        output_path.chmod(0o644)
        output_names.append(output_path.name)
    return output_names


def write_html_index(output_path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        detour = row["detour_length_m"]
        detour_text = "goal blocked" if detour is None else f"length delta {detour:+.2f} m"
        if not row.get("reachability_matches_catalog", True):
            detour_text += " · catalog/replay mismatch"
        cards.append(
            f"""
            <article class="card">
              <a href="{html.escape(row['image'])}"><img src="{html.escape(row['image'])}" loading="lazy"></a>
              <h2>#{row['sample_rank']:02d} · Candidate {row['candidate_index']} · House {row['house_index']}</h2>
              <p><code>{html.escape(row['case_id'])}</code></p>
              <p>{html.escape(row['target_category'])} in {html.escape(row['container_category'])} ·
                 {row['crossed_door_count']} crossed door(s) · {html.escape(detour_text)}</p>
              <p>{row['object_box_count']} object boxes ·
                 <a href="{html.escape(row['sidecar'])}">paths and Oxxx annotations JSON</a></p>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mixed rough top-down gallery</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f1f5f9; color: #0f172a; font: 14px/1.45 system-ui, sans-serif; }}
    h1 {{ margin-top: 0; }}
    .note {{ max-width: 1100px; padding: 12px 14px; background: #fff7ed; border: 1px solid #fdba74; border-radius: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 20px; margin-top: 20px; }}
    .card {{ background: white; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; box-shadow: 0 2px 8px #0f172a14; }}
    .card img {{ width: 100%; height: auto; border: 1px solid #e2e8f0; }}
    .card h2 {{ margin: 10px 0 4px; font-size: 17px; }}
    .card p {{ margin: 5px 0; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Mixed rough candidates</h1>
  <p class="note">These views compare the all-open GT planner path with a replan after closing the selected crossed door. They are rough evidence only: the goal is nearest free space to the container AABB, and target visibility/reveal plus manipulation-valid poses are not yet V3 fine-data truth.</p>
  <main class="grid">{''.join(cards)}</main>
</body>
</html>
"""
    output_path.write_text(document)
    output_path.chmod(0o644)


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "house_count": len({int(row["house_index"]) for row in rows}),
        "container_category_counts": dict(
            Counter(str(row["container_category"]) for row in rows)
        ),
        "path_length_bin_counts": dict(
            Counter(path_length_bin(float(row["all_open_path_length_m"])) for row in rows)
        ),
        "crossed_door_count_bin_counts": dict(
            Counter(
                crossed_door_count_bin(len(row.get("crossed_door_roots", [])))
                for row in rows
            )
        ),
        "target_category_counts": dict(
            Counter(str(row["target_category"]) for row in rows)
        ),
    }


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    catalog = load_catalog(args.catalog)
    selected = select_candidate_rows(
        catalog,
        candidate_type=args.candidate_type,
        case_ids=set(args.case_id) if args.case_id else None,
        house_indices=parse_int_set(args.house_indices),
        max_samples=args.max_samples,
        seed=args.selection_seed,
        unique_houses=args.unique_houses,
    )
    benchmark_dir = args.benchmark_dir or Path(catalog["benchmark_dir"])
    episodes = container_builder.load_benchmark_episodes(benchmark_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_manifest = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "catalog": str(args.catalog),
        "benchmark_dir": str(benchmark_dir),
        "candidate_type": args.candidate_type,
        "selection_seed": args.selection_seed,
        "selection": selection_summary(selected),
        "candidates": [
            {
                "sample_rank": rank,
                "candidate_index": int(row["_candidate_index"]),
                "case_id": row["case_id"],
                "house_index": int(row["house_index"]),
                "container_category": row["container_category"],
                "target_category": row["target_category"],
                "all_open_path_length_m": row["all_open_path_length_m"],
                "crossed_door_roots": row["crossed_door_roots"],
            }
            for rank, row in enumerate(selected, start=1)
        ],
    }
    write_json(args.output_dir / "selected_candidates.json", selected_manifest)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    work = []
    for sample_rank, candidate in enumerate(selected, start=1):
        if args.reuse_existing:
            reused = load_existing_result(
                args, sample_rank=sample_rank, candidate=candidate
            )
            if reused is not None:
                results.append(reused)
                print(
                    f"[{sample_rank}/{len(selected)}] reused candidate={candidate['_candidate_index']} "
                    f"house={candidate['house_index']}",
                    flush=True,
                )
                continue
        source_episode_index = int(candidate["source_episode_index"])
        if not 0 <= source_episode_index < len(episodes):
            errors.append(
                {
                    "sample_rank": sample_rank,
                    "candidate_index": int(candidate["_candidate_index"]),
                    "case_id": candidate["case_id"],
                    "error": f"Source episode index out of range: {source_episode_index}",
                }
            )
            continue
        work.append((sample_rank, candidate, episodes[source_episode_index]))

    started_at = time.perf_counter()
    args_payload = vars(args).copy()
    if args.workers <= 1:
        for progress, (sample_rank, candidate, episode) in enumerate(work, start=1):
            candidate_started = time.perf_counter()
            try:
                row = render_worker(args_payload, sample_rank, candidate, episode)
                row["elapsed_sec"] = time.perf_counter() - candidate_started
                results.append(row)
                print(
                    f"[{progress}/{len(work)}] rendered candidate={row['candidate_index']} "
                    f"house={row['house_index']} doors={row['crossed_door_count']} "
                    f"detour={row['detour_length_m']} elapsed={row['elapsed_sec']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "sample_rank": sample_rank,
                        "candidate_index": int(candidate["_candidate_index"]),
                        "case_id": candidate["case_id"],
                        "house_index": int(candidate["house_index"]),
                        "error": str(exc),
                    }
                )
                print(
                    f"[{progress}/{len(work)}] failed candidate={candidate['_candidate_index']} "
                    f"house={candidate['house_index']}: {exc}",
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
    elif work:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            future_to_item = {
                executor.submit(render_worker, args_payload, sample_rank, candidate, episode): (
                    sample_rank,
                    candidate,
                )
                for sample_rank, candidate, episode in work
            }
            for progress, future in enumerate(as_completed(future_to_item), start=1):
                sample_rank, candidate = future_to_item[future]
                try:
                    row = future.result()
                    results.append(row)
                    print(
                        f"[{progress}/{len(work)}] rendered candidate={row['candidate_index']} "
                        f"house={row['house_index']} doors={row['crossed_door_count']} "
                        f"detour={row['detour_length_m']}",
                        flush=True,
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "sample_rank": sample_rank,
                            "candidate_index": int(candidate["_candidate_index"]),
                            "case_id": candidate["case_id"],
                            "house_index": int(candidate["house_index"]),
                            "error": str(exc),
                        }
                    )
                    print(
                        f"[{progress}/{len(work)}] failed candidate={candidate['_candidate_index']} "
                        f"house={candidate['house_index']}: {exc}",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        raise

    results.sort(key=lambda row: int(row["sample_rank"]))
    contact_sheets = make_contact_sheets(
        args.output_dir,
        results,
        columns=args.contact_sheet_columns,
        page_size=args.contact_sheet_page_size,
    )
    write_html_index(args.output_dir / "index.html", results)
    manifest = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "catalog": str(args.catalog),
        "benchmark_dir": str(benchmark_dir),
        "requested_sample_count": len(selected),
        "rendered_sample_count": len(results),
        "failed_sample_count": len(errors),
        "elapsed_sec": time.perf_counter() - started_at,
        "selection": selection_summary(selected),
        "contact_sheets": contact_sheets,
        "html_index": "index.html",
        "results": results,
        "errors": errors,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rendered": len(results),
                "failed": len(errors),
                "contact_sheets": contact_sheets,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if results and (not errors or args.continue_on_error) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render mixed rough-catalog candidates as paired annotated top-down maps: "
            "all-open GT planner path versus the route after closing a crossed door."
        )
    )
    parser.add_argument("catalog", type=Path, nargs="?", default=DEFAULT_CATALOG)
    parser.add_argument("--benchmark_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate_type",
        choices=[
            "door_crossing_only",
            "mixed_shortcut_verified",
            "mixed_door_set_shortcut_verified",
            "mixed_door_set_required_verified",
            "mixed_required_verified",
            "all",
        ],
        default="door_crossing_only",
    )
    parser.add_argument("--case_id", action="append", default=[])
    parser.add_argument("--house_indices", help="Comma-separated house indices")
    parser.add_argument("--max_samples", type=int, default=24)
    parser.add_argument("--selection_seed", type=int, default=7)
    parser.add_argument(
        "--unique_houses", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--px_per_m", type=float, default=50.0)
    parser.add_argument("--open_threshold", type=float, default=0.67)
    parser.add_argument("--door_on_path_padding_m", type=float, default=0.2)
    parser.add_argument("--path_region_sample_step_m", type=float, default=0.05)
    parser.add_argument("--door_approach_standoff_m", type=float, default=0.9)
    parser.add_argument("--path_length_tolerance_m", type=float, default=0.35)
    parser.add_argument("--path_length_relative_tolerance", type=float, default=0.10)
    parser.add_argument(
        "--strict_validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of rendering when current replay disagrees with rough-catalog evidence.",
    )
    parser.add_argument(
        "--label_all_objects", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include_structural", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--render_first_person",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also render open/closed door-front RGB views from the geometric approach hint.",
    )
    parser.add_argument("--first_person_camera", default="head_camera")
    parser.add_argument(
        "--continue_on_error", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reuse_existing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--contact_sheet_columns", type=int, default=2)
    parser.add_argument("--contact_sheet_page_size", type=int, default=4)
    parser.add_argument("--mujoco_gl", default="egl")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
