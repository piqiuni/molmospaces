#!/usr/bin/env python3
"""Render evaluator-only ObjectNav trajectories with official goal geometry.

The helpers in this file are called only after a policy action has been emitted.
Goal centers, valid view points, simulator poses, and official metrics are never
returned to the policy, Module-1, or Module-2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _trace_events(path: Path, public_episode: dict[str, str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    expected_scene = str(public_episode["scene_id"])
    expected_episode = str(public_episode["episode_id"])
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("scene_id")) == expected_scene and str(row.get("episode_id")) == expected_episode:
            events.append(row)
    return events


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _nearest_row(rows: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(int(row.get("step", 0)) - int(step)))


def _positions_at_steps(rows: list[dict[str, Any]], steps: Iterable[int]) -> list[list[float]]:
    positions: list[list[float]] = []
    for step in steps:
        row = _nearest_row(rows, int(step))
        if row is not None and isinstance(row.get("agent_world_position"), list):
            positions.append([float(value) for value in row["agent_world_position"]])
    return positions


def _goal_geometry(
    episode: Any,
    *,
    floor_y: float,
    floor_tolerance_m: float = 0.75,
) -> tuple[list[list[float]], list[list[float]]]:
    centers: list[list[float]] = []
    view_points: list[list[float]] = []
    for goal in episode.goals:
        same_floor_views = [
            [float(value) for value in view_point.agent_state.position]
            for view_point in goal.view_points or []
            if abs(float(view_point.agent_state.position[1]) - floor_y) <= floor_tolerance_m
        ]
        if same_floor_views:
            centers.append([float(value) for value in goal.position])
            view_points.extend(same_floor_views)
    return centers, view_points


def _grid_points(pathfinder: Any, shape: tuple[int, int], positions: Iterable[list[float]]) -> np.ndarray:
    from habitat.utils.visualizations import maps

    points = []
    for position in positions:
        # Habitat's map utility expects (world z, world x) and returns (row, col).
        row, col = maps.to_grid(float(position[2]), float(position[0]), shape, pathfinder=pathfinder)
        points.append((float(col), float(row)))
    return np.asarray(points, dtype=np.float32).reshape((-1, 2))


def render_posthoc_topdown(
    *,
    output_dir: Path,
    env: Any,
    episode: Any,
    public_episode: dict[str, str],
    posthoc_rows: list[dict[str, Any]],
    trace_path: Path,
    episode_vision: dict[str, int],
    metrics: dict[str, Any],
    map_resolution: int = 1024,
) -> dict[str, Any]:
    """Write one top-down PNG and a machine-readable verdict."""

    from habitat.utils.visualizations import maps

    events = _trace_events(trace_path, public_episode)
    legacy_detection_steps = [
        int(row["step"])
        for row in events
        if row.get("event") == "module1_target_detection" and row.get("selected_target") is not None
    ]
    full_ros_target_offer_steps = [
        int(row["step"])
        for row in events
        if row.get("event") == "replan"
        and any(
            str(candidate.get("behavior_type") or "").upper() == "NAVIGATE"
            and str(candidate.get("candidate_id") or "").startswith("target:")
            for candidate in (row.get("candidates") or [])
            if isinstance(candidate, dict)
        )
    ]
    detection_steps = legacy_detection_steps + full_ros_target_offer_steps
    promoted_steps = [int(row["step"]) for row in events if row.get("event") == "target_track_promoted"]
    target_selection_steps = [
        int(row["step"])
        for row in events
        if row.get("event") == "module2_selection"
        and (
            bool(row.get("target_track_selected"))
            or (
                str(row.get("behavior_type") or "").upper() == "NAVIGATE"
                and str(row.get("candidate_id") or "").startswith("target:")
            )
        )
    ]

    distances = [
        (int(row["step"]), float(row["distance_to_goal"]))
        for row in posthoc_rows
        if _float_or_none(row.get("distance_to_goal")) is not None
    ]
    first_distance = distances[0][1] if distances else None
    final_distance = distances[-1][1] if distances else None
    min_step, min_distance = min(distances, key=lambda item: item[1]) if distances else (None, None)

    selected_distance = None
    min_after_selection = None
    approached_after_selection = False
    if target_selection_steps and distances:
        first_selection = target_selection_steps[0]
        selection_row = _nearest_row(posthoc_rows, first_selection)
        selected_distance = _float_or_none(selection_row.get("distance_to_goal")) if selection_row else None
        later = [distance for step, distance in distances if step >= first_selection]
        min_after_selection = min(later) if later else None
        approached_after_selection = bool(
            selected_distance is not None
            and min_after_selection is not None
            and min_after_selection <= selected_distance - 0.10
        )

    start_position = [float(value) for value in episode.start_position]
    goal_centers, view_points = _goal_geometry(episode, floor_y=float(start_position[1]))
    trajectory = [
        [float(value) for value in row["agent_world_position"]]
        for row in posthoc_rows
        if isinstance(row.get("agent_world_position"), list)
    ]
    topdown = maps.get_topdown_map(
        env.sim.pathfinder,
        height=float(start_position[1]),
        map_resolution=map_resolution,
        draw_border=True,
    )
    colored = maps.colorize_topdown_map(topdown)

    shape = (int(topdown.shape[0]), int(topdown.shape[1]))
    trajectory_px = _grid_points(env.sim.pathfinder, shape, trajectory)
    start_px = _grid_points(env.sim.pathfinder, shape, [start_position])
    centers_px = _grid_points(env.sim.pathfinder, shape, goal_centers)
    views_px = _grid_points(env.sim.pathfinder, shape, view_points)
    detection_px = _grid_points(
        env.sim.pathfinder,
        shape,
        _positions_at_steps(posthoc_rows, detection_steps[:1]),
    )
    selection_px = _grid_points(
        env.sim.pathfinder,
        shape,
        _positions_at_steps(posthoc_rows, target_selection_steps),
    )
    min_px = _grid_points(
        env.sim.pathfinder,
        shape,
        _positions_at_steps(posthoc_rows, [int(min_step)]) if min_step is not None else [],
    )

    fig, (axis, status_axis) = plt.subplots(
        1,
        2,
        figsize=(12.8, 7.2),
        gridspec_kw={"width_ratios": [4.4, 1.6]},
        constrained_layout=True,
    )
    axis.imshow(colored, origin="upper")
    if len(trajectory_px):
        axis.plot(trajectory_px[:, 0], trajectory_px[:, 1], color="#00a6d6", linewidth=2.0, label="Agent trajectory")
        axis.scatter(trajectory_px[-1, 0], trajectory_px[-1, 1], marker="s", s=70, color="#1565c0", label="Final pose", zorder=6)
    if len(start_px):
        axis.scatter(start_px[:, 0], start_px[:, 1], marker="^", s=90, color="#2e7d32", label="Start", zorder=7)
    if len(views_px):
        axis.scatter(
            views_px[:, 0],
            views_px[:, 1],
            marker="o",
            s=20,
            facecolors="none",
            edgecolors="#d32f2f",
            linewidths=0.8,
            label="Valid ViewPoints (GT)",
            zorder=5,
        )
    if len(centers_px):
        axis.scatter(centers_px[:, 0], centers_px[:, 1], marker="X", s=120, color="#b71c1c", label="Target instances (GT)", zorder=8)
    if len(detection_px):
        axis.scatter(detection_px[:, 0], detection_px[:, 1], marker="o", s=95, color="#fb8c00", label="First target detection", zorder=8)
    if len(selection_px):
        axis.scatter(selection_px[:, 0], selection_px[:, 1], marker="D", s=80, color="#8e24aa", label="M2 selected target track", zorder=8)
    if len(min_px):
        axis.scatter(min_px[:, 0], min_px[:, 1], marker="*", s=180, color="#fdd835", edgecolors="#5d4037", label="Minimum official distance", zorder=9)

    visible_arrays = [array for array in (trajectory_px, start_px, centers_px, views_px) if len(array)]
    if visible_arrays:
        all_points = np.concatenate(visible_arrays, axis=0)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        pad = max(28.0, 0.06 * max(x_max - x_min, y_max - y_min, 1.0))
        axis.set_xlim(max(0.0, x_min - pad), min(shape[1] - 1.0, x_max + pad))
        axis.set_ylim(min(shape[0] - 1.0, y_max + pad), max(0.0, y_min - pad))
    scene_name = Path(str(public_episode["scene_id"])).parent.name
    axis.set_title(f"{scene_name} · target={public_episode['object_category']}")
    axis.set_axis_off()
    axis.legend(loc="lower left", fontsize=8, framealpha=0.88)

    detected = bool(detection_steps) or int(episode_vision.get("module1_detector_positive", 0)) > 0
    promoted = int(episode_vision.get("target_track_promoted", 0)) > 0
    selected = bool(target_selection_steps)
    success = float(metrics.get("success", 0.0)) > 0.0
    status_axis.axis("off")
    status_lines = [
        "POSTHOC VERDICT",
        "",
        f"Target perceived: {'YES' if detected else 'NO'}",
        f"  target hits: {int(episode_vision.get('module1_detector_positive', 0))}",
        f"Track promoted: {'YES' if promoted else 'NO'}",
        f"  promotions: {int(episode_vision.get('target_track_promoted', 0))}",
        f"M2 target navigation: {'YES' if selected else 'NO'}",
        f"  selections: {len(target_selection_steps)}",
        f"Approached after M2 select: {'YES' if approached_after_selection else ('NO' if selected else 'N/A')}",
        "",
        f"Official success: {'YES' if success else 'NO'}",
        f"First distance: {first_distance:.3f} m" if first_distance is not None else "First distance: N/A",
        f"Minimum distance: {min_distance:.3f} m @ step {min_step}" if min_distance is not None else "Minimum distance: N/A",
        f"Final distance: {final_distance:.3f} m" if final_distance is not None else "Final distance: N/A",
        "Threshold: ViewPoint <= 0.1 m + STOP",
        "",
        "GT markers and official metrics are",
        "evaluator-only posthoc diagnostics.",
        "They never enter the policy.",
    ]
    status_axis.text(0.02, 0.98, "\n".join(status_lines), va="top", ha="left", fontsize=10, family="monospace")

    output_path = output_dir / "topdown_with_targets.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    verdict = {
        "scene_id": str(public_episode["scene_id"]),
        "episode_id": str(public_episode["episode_id"]),
        "object_category": str(public_episode["object_category"]),
        "target_instance_count": len(goal_centers),
        "valid_viewpoint_count": len(view_points),
        "target_perceived": detected,
        "target_detection_hits": max(
            int(episode_vision.get("module1_detector_positive", 0)),
            len(full_ros_target_offer_steps),
        ),
        "full_ros_target_candidate_offers": len(full_ros_target_offer_steps),
        "target_track_promoted": promoted,
        "target_track_promotions": int(episode_vision.get("target_track_promoted", 0)),
        "module2_target_navigation": selected,
        "module2_target_selection_steps": target_selection_steps,
        "approached_after_module2_selection": approached_after_selection,
        "distance_at_first_target_selection_m": selected_distance,
        "minimum_distance_after_target_selection_m": min_after_selection,
        "official_success": success,
        "first_distance_to_goal_m": first_distance,
        "minimum_distance_to_goal_m": min_distance,
        "minimum_distance_step": min_step,
        "final_distance_to_goal_m": final_distance,
        "success_definition": "distance_to_nearest_valid_VIEW_POINT<=0.1m and explicit velocity_stop",
        "gt_usage": "posthoc visualization only; not supplied to policy, Module-1, or Module-2",
        "image": str(output_path),
    }
    (output_dir / "posthoc_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verdict
