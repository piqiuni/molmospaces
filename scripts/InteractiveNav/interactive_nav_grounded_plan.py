"""Shared grounded-plan helpers for V3 PointGoal and InstructionGoal generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_episodes(path: Path) -> list[dict[str, Any]]:
    benchmark_path = path / "benchmark.json" if path.is_dir() else path
    payload = json.loads(benchmark_path.read_text())
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    if not isinstance(episodes, list):
        raise ValueError(f"Unsupported benchmark payload in {benchmark_path}")
    return episodes


def select_episode(
    episodes: list[dict[str, Any]],
    *,
    episode_index: int | None = None,
    case_id: str | None = None,
) -> tuple[int, dict[str, Any]]:
    if case_id is not None:
        for index, episode in enumerate(episodes):
            if episode.get("interactive_nav", {}).get("case_id") == case_id:
                return index, episode
        raise ValueError(f"V3 case_id not found: {case_id}")
    index = 0 if episode_index is None else int(episode_index)
    if index < 0 or index >= len(episodes):
        raise IndexError(f"Episode index {index} is outside [0, {len(episodes)})")
    return index, episodes[index]


def path_length(path: Iterable[Iterable[float]] | None) -> float | None:
    if path is None:
        return None
    points = np.asarray(list(path), dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def build_grounded_plan(
    episode: dict[str, Any],
    *,
    navigation_paths: dict[int, list[list[float]]] | None = None,
) -> dict[str, Any]:
    """Convert privileged V3 oracle annotations into a stable generation input."""

    interactive = episode.get("interactive_nav", {})
    plan = interactive.get("oracle_plan") or {}
    interactions = interactive.get("interactions") or []
    interaction_by_id = {
        str(row["interaction_id"]): row
        for row in interactions
        if row.get("interaction_id") is not None
    }
    target = interactive.get("target") or {}
    task = episode.get("task") or {}
    language = episode.get("language") or {}
    paths = navigation_paths or {}
    navigation_validation = (
        interactive.get("generation_validation", {}).get("navigation_validation", {})
    )
    gt_path_waypoints = list(navigation_validation.get("gt_path_waypoints") or [])
    grounded_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(plan.get("steps") or []):
        step = dict(raw_step)
        grounded: dict[str, Any] = {
            "step_index": index,
            "type": step.get("type"),
            "reason": step.get("reason"),
        }
        interaction_id = step.get("interaction_id")
        if interaction_id is not None:
            grounded["interaction_id"] = interaction_id
            interaction = interaction_by_id.get(str(interaction_id))
            if interaction is not None:
                grounded["interaction"] = {
                    "type": interaction.get("type"),
                    "object_name": interaction.get("object_name"),
                    "object_category": interaction.get("object_category"),
                    "effect_types": list(interaction.get("effect_types") or []),
                }
        if step.get("type") == "navigate":
            grounded.update(
                {
                    "goal_point": list(step.get("goal_point") or []),
                    "goal_yaw": step.get("goal_yaw"),
                    "position_tolerance_m": step.get("position_tolerance_m"),
                }
            )
            if index in paths:
                grounded["path_waypoints"] = paths[index]
                grounded["path_length_m"] = path_length(paths[index])
        elif step.get("type") == "open_joint":
            grounded.update(
                {
                    "object_name": step.get("object_name"),
                    "joint_name": step.get("joint_name"),
                    "joint_index": step.get("joint_index"),
                    "target_fraction": step.get("target_fraction"),
                }
            )
        elif step.get("type") == "set_view":
            grounded["view_profile"] = step.get("view_profile")
        elif step.get("type") == "observe_target":
            grounded["object_name"] = step.get("object_name")
        grounded_steps.append(grounded)

    target_type = "point" if task.get("task_type") == "nav_to_point" else "object"
    target_summary: dict[str, Any] = {"target_type": target_type}
    if target_type == "point":
        target_summary.update(
            {
                "goal_point": list(task.get("goal_point") or target.get("goal_point") or []),
                "goal_yaw": task.get("goal_yaw"),
            }
        )
    else:
        referrals = language.get("referral_expressions") or {}
        referral_expression = next(
            (
                str(referrals[key])
                for key in ("pickup_name", "object_name", "target_name")
                if referrals.get(key)
            ),
            None,
        )
        target_summary.update(
            {
                "category": target.get("category"),
                "selected_instance": target.get("selected_instance"),
                "container_name": target.get("container_name"),
                "container_category": target.get("container_category"),
                "referral_expression": referral_expression,
            }
        )

    return {
        "schema_version": "interactive_nav_grounded_plan_v1",
        "case_id": interactive.get("case_id"),
        "house_index": episode.get("house_index"),
        "scene_dataset": episode.get("scene_dataset"),
        "data_split": episode.get("data_split"),
        "task_type": task.get("task_type"),
        "robot_base_pose": list(task.get("robot_base_pose") or []),
        "interaction_domains": list(interactive.get("interaction_domains") or []),
        "interaction_requirement": interactive.get("interaction_requirement"),
        "target": target_summary,
        "gt_path_waypoints": gt_path_waypoints,
        "required_interaction_ids": list(plan.get("required_interaction_ids") or []),
        "steps": grounded_steps,
    }


def point_to_segment_distance_xy(
    point: Iterable[float], start: Iterable[float], end: Iterable[float]
) -> float:
    point_xy = np.asarray(list(point), dtype=float)[:2]
    start_xy = np.asarray(list(start), dtype=float)[:2]
    end_xy = np.asarray(list(end), dtype=float)[:2]
    delta = end_xy - start_xy
    denom = float(delta @ delta)
    if denom <= 1e-12:
        return float(np.linalg.norm(point_xy - start_xy))
    fraction = min(1.0, max(0.0, float((point_xy - start_xy) @ delta) / denom))
    return float(np.linalg.norm(point_xy - (start_xy + fraction * delta)))


def point_to_path_distance_xy(point: Iterable[float], path: Iterable[Iterable[float]]) -> float:
    points = [list(row) for row in path]
    if not points:
        return math.inf
    if len(points) == 1:
        return float(np.linalg.norm(np.asarray(list(point))[:2] - np.asarray(points[0])[:2]))
    return min(
        point_to_segment_distance_xy(point, start, end)
        for start, end in zip(points, points[1:])
    )


def build_path_corridor_graph(
    graph: dict[str, Any],
    path: Iterable[Iterable[float]],
    *,
    radius_m: float = 1.0,
    required_entity_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Keep graph nodes grounded near a path plus explicitly required entities."""

    path_points = [list(point) for point in path]
    required = {str(value) for value in required_entity_ids}
    kept_nodes = []
    kept_ids: set[str] = set()
    for node in graph.get("nodes") or []:
        node_id = str(node.get("id") or "")
        names = {
            node_id,
            str(node.get("name") or ""),
            str((node.get("attributes") or {}).get("source_object_name") or ""),
        }
        centroid = node.get("centroid") or node.get("aabb_center") or []
        near_path = len(centroid) >= 2 and point_to_path_distance_xy(
            centroid, path_points
        ) <= radius_m
        if near_path or bool(required & names) or node.get("type") in {"scene"}:
            kept_nodes.append(node)
            if node_id:
                kept_ids.add(node_id)
    kept_edges = [
        edge
        for edge in graph.get("edges") or []
        if str(edge.get("src_id")) in kept_ids and str(edge.get("dst_id")) in kept_ids
    ]
    return {
        "schema_version": "interactive_nav_path_corridor_graph_v1",
        "radius_m": float(radius_m),
        "nodes": kept_nodes,
        "edges": kept_edges,
    }


def select_segment_keyframes(segments: Iterable[str]) -> list[int]:
    """Return the first, middle, and last step of every contiguous segment."""

    names = [str(value) for value in segments]
    selected: list[int] = []
    start = 0
    while start < len(names):
        end = start + 1
        while end < len(names) and names[end] == names[start]:
            end += 1
        for index in (start, (start + end - 1) // 2, end - 1):
            if index not in selected:
                selected.append(index)
        start = end
    return selected
