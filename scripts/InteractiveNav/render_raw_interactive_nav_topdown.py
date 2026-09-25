#!/usr/bin/env python3
"""Render a post-run top-down report for the legacy/raw Evo runner.

The raw ROS runner does not produce the evaluator's ``episode_visualization``
sidecar.  This renderer therefore consumes only artifacts written after (or
during) an episode: the final occupancy map, the recorder trajectory, force
interaction results, and an optional fixed-route/benchmark description.  GT
data is kept in the output metadata and is never sent to the policy process.

The output deliberately has a small, stable interface so both the raw shell
runner and external batch tools can call it::

    python render_raw_interactive_nav_topdown.py --run-dir RUN

It writes ``raw_episode_topdown.png`` and a same-stem JSON metadata file by
default.  Missing optional GT information is not fatal; the image is still
rendered and the metadata explicitly records which layers are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
    raise SystemExit(
        "raw top-down rendering requires numpy, opencv-python and pyyaml"
    ) from exc


SCHEMA_VERSION = "interactive_nav_raw_evo_topdown_v1"


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "succeeded", "success", "ok"}
    return bool(value)


def _as_xy(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        point = np.asarray(value[:2], dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return point


def _as_xyyaw(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    values = [_finite_float(v) for v in value[:3]]
    if values[0] is None or values[1] is None:
        return None
    return [float(values[0]), float(values[1]), float(values[2] or 0.0)]


def _dedupe_points(points: Iterable[Any], tolerance: float = 1e-6) -> list[list[float]]:
    result: list[list[float]] = []
    for value in points:
        point = _as_xy(value)
        if point is None:
            continue
        row = [float(point[0]), float(point[1])]
        if result and math.hypot(row[0] - result[-1][0], row[1] - result[-1][1]) <= tolerance:
            continue
        result.append(row)
    return result


def _resolve_map_paths(run_dir: Path, explicit_yaml: Path | None) -> tuple[Path, Path]:
    candidates = []
    if explicit_yaml is not None:
        candidates.append(explicit_yaml)
    candidates.extend(
        [
            run_dir / "debug" / "final_occ_map.yaml",
            run_dir / "final_occ_map.yaml",
            run_dir / "debug" / "final_raw_occ_map.yaml",
            run_dir / "final_raw_occ_map.yaml",
        ]
    )
    yaml_path = next((path for path in candidates if path.is_file()), None)
    if yaml_path is None:
        raise FileNotFoundError(
            "No final occupancy metadata found; looked in "
            + ", ".join(str(path) for path in candidates)
        )
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid occupancy metadata: {yaml_path}")
    image_value = metadata.get("image")
    image_path = (Path(str(image_value)).expanduser() if image_value else yaml_path.with_suffix(".pgm"))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    return yaml_path, image_path


def load_map(
    run_dir: Path, explicit_yaml: Path | None = None
) -> tuple[np.ndarray, float, np.ndarray, float, Path, Path]:
    """Load the recorder map and its world-coordinate metadata."""

    yaml_path, image_path = _resolve_map_paths(run_dir, explicit_yaml)
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read occupancy image: {image_path}")
    resolution = _finite_float(metadata.get("resolution"))
    if resolution is None or resolution <= 0.0:
        raise ValueError(f"Occupancy resolution must be positive: {yaml_path}")
    origin = metadata.get("origin", [0.0, 0.0, 0.0])
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        raise ValueError(f"Occupancy origin is invalid: {yaml_path}")
    origin_xy = np.asarray(
        [_finite_float(origin[0], 0.0), _finite_float(origin[1], 0.0)], dtype=float
    )
    origin_yaw = _finite_float(origin[2], 0.0) if len(origin) >= 3 else 0.0
    return image, float(resolution), origin_xy, float(origin_yaw or 0.0), yaml_path, image_path


def _resolve_artifact(run_dir: Path, relative_name: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [run_dir / relative_name, run_dir / "debug" / relative_name]
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    points: list[tuple[float, float]] = []
    yaws: list[float] = []
    steps: list[float] = []
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,)), rows
    try:
        stream = path.open(encoding="utf-8", newline="")
    except OSError:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,)), rows
    with stream:
        for row in csv.DictReader(stream):
            x = _finite_float(row.get("x"))
            y = _finite_float(row.get("y"))
            if x is None or y is None:
                continue
            yaw = _finite_float(row.get("yaw"), 0.0) or 0.0
            step = _finite_float(row.get("step_id"), float(len(points)))
            points.append((x, y))
            yaws.append(yaw)
            steps.append(step if step is not None else float(len(points) - 1))
            rows.append(dict(row))
    if not points:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,)), rows
    return np.asarray(points, dtype=float), np.asarray(yaws, dtype=float), np.asarray(steps), rows


def load_force_events(path: Path) -> list[dict[str, Any]]:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        return []
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []
    result: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        row = event.get("result") if isinstance(event.get("result"), dict) else event
        if isinstance(row, dict):
            result.append(row)
    return result


def _route_from_config(path: Path | None, route_id: str | None) -> tuple[dict[str, Any] | None, str]:
    if path is None or not path.is_file():
        return None, "unavailable"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None, "invalid_route_config"
    routes = payload.get("routes", []) if isinstance(payload, dict) else []
    if not isinstance(routes, list):
        return None, "invalid_route_config"
    selected = None
    if route_id:
        selected = next(
            (row for row in routes if isinstance(row, dict) and str(row.get("route_id")) == str(route_id)),
            None,
        )
    if selected is None and len(routes) == 1 and isinstance(routes[0], dict):
        selected = routes[0]
    if selected is None:
        return None, "route_not_found"
    return selected, "route_config"


def _episode_from_benchmark(path: Path | None, episode_index: int, case_id: str | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = _json_load(path)
    if isinstance(payload, dict):
        episodes = payload.get("episodes", payload)
    else:
        episodes = payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    if not isinstance(episodes, list):
        return None
    if 0 <= episode_index < len(episodes) and isinstance(episodes[episode_index], dict):
        candidate = episodes[episode_index]
        candidate_case = candidate.get("interactive_nav", {}).get("case_id")
        if case_id is None or str(candidate_case) == str(case_id):
            return candidate
    if case_id is not None:
        for candidate in episodes:
            if isinstance(candidate, dict) and str(candidate.get("interactive_nav", {}).get("case_id")) == str(case_id):
                return candidate
    return None


def _benchmark_split(episode: dict[str, Any] | None) -> str | None:
    if not isinstance(episode, dict):
        return None
    value = episode.get("data_split")
    return None if value is None else str(value).strip().lower()


def _benchmark_house_index(episode: dict[str, Any] | None) -> int | None:
    """Return the frozen benchmark house id when it is explicitly present."""

    if not isinstance(episode, dict):
        return None
    value = episode.get("house_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_house_index(run_dir: Path) -> int | None:
    """Read the raw runner's house id without relying on graph node names.

    ``graph_latest.json`` intentionally uses a generic scene id, while object
    instance ids are reused across many ProcTHOR houses.  Prefer the live
    ``HOUSE_IND`` environment (the shell invokes this before writing its final
    result), then fall back to ``semantic_exploration_result.json``.
    """

    value = os.environ.get("HOUSE_IND", "").strip()
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    payload = _json_load(run_dir / "semantic_exploration_result.json")
    if not isinstance(payload, dict):
        return None
    value = payload.get("house_ind")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _benchmark_geometry(episode: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(episode, dict):
        return {}
    robot = episode.get("robot") if isinstance(episode.get("robot"), dict) else {}
    init_qpos = robot.get("init_qpos") if isinstance(robot.get("init_qpos"), dict) else {}
    start = _as_xyyaw(init_qpos.get("base"))
    task = episode.get("task") if isinstance(episode.get("task"), dict) else {}
    if start is None:
        task_pose = task.get("robot_base_pose")
        if isinstance(task_pose, (list, tuple)) and len(task_pose) >= 7:
            try:
                start = [
                    float(task_pose[0]),
                    float(task_pose[1]),
                    float(2.0 * math.atan2(float(task_pose[6]), float(task_pose[3]))),
                ]
            except (TypeError, ValueError):
                start = None
        else:
            start = _as_xyyaw(task_pose)
    nav = episode.get("interactive_nav") if isinstance(episode.get("interactive_nav"), dict) else {}
    target = nav.get("target") if isinstance(nav.get("target"), dict) else {}
    target_xy = _as_xy(target.get("object_aabb_center"))
    target_source = "benchmark_object_aabb_center" if target_xy is not None else None
    if target_xy is None:
        selected_instance = target.get("selected_instance") or task.get("pickup_obj_name")
        scene_modifications = episode.get("scene_modifications")
        object_poses = (
            scene_modifications.get("object_poses")
            if isinstance(scene_modifications, dict)
            else None
        )
        pose = object_poses.get(str(selected_instance)) if isinstance(object_poses, dict) else None
        target_xy = _as_xy(pose)
        if target_xy is not None:
            target_source = "benchmark_object_pose"
    plans = nav.get("oracle_plans") or ([] if not isinstance(nav.get("oracle_plan"), dict) else [nav["oracle_plan"]])
    goal = None
    route: list[list[float]] = []
    if isinstance(plans, list):
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            for step in plan.get("steps", []):
                if not isinstance(step, dict) or step.get("type") != "navigate":
                    continue
                point = _as_xy(step.get("goal_point"))
                if point is not None:
                    route.append([float(point[0]), float(point[1])])
                    goal = [float(point[0]), float(point[1])]
    return {
        "start_xyyaw": start,
        "goal_xyyaw": ([goal[0], goal[1], 0.0] if goal is not None else None),
        "target_object_xy": target_xy.tolist() if target_xy is not None else None,
        "target_object_source": target_source,
        "route_xy": _dedupe_points(([start[:2]] if start else []) + route),
    }


def _runtime_target_geometry(path: Path | None) -> dict[str, Any]:
    """Best-effort extraction from run-time target-selection sidecars."""

    payload = _json_load(path) if path is not None and path.is_file() else None
    if not isinstance(payload, dict):
        return {}
    # Different revisions wrapped the selection under one of these keys.
    candidates: list[dict[str, Any]] = [payload]
    for key in ("selection", "target_selection", "target", "selected_target", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        for key in ("goal_xyyaw", "target_xyyaw", "far_goal_xyyaw", "goal", "target_center", "target_xy"):
            value = candidate.get(key)
            xy = _as_xy(value)
            if xy is not None:
                yaw = _finite_float(value[2], 0.0) if isinstance(value, (list, tuple)) and len(value) >= 3 else 0.0
                return {
                    "goal_xyyaw": [float(xy[0]), float(xy[1]), float(yaw or 0.0)],
                    "route_xy": [],
                    "source": f"target_selection.{key}",
                }
    return {}


def _graph_target_geometry(run_dir: Path, target_id: str | None) -> dict[str, Any]:
    """Find an observed target AABB in the recorder's latest semantic graph."""

    if not target_id:
        return {}
    candidates = [
        run_dir / "debug" / "graph" / "graph_latest.json",
        run_dir / "debug" / "graph_latest.json",
    ]
    for path in candidates:
        payload = _json_load(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
            continue
        for node in payload["nodes"]:
            if not isinstance(node, dict):
                continue
            attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
            identifiers = {
                str(node.get("id") or ""),
                str(node.get("name") or ""),
                str(node.get("object_id") or ""),
                str(attrs.get("instance_id") or ""),
                str(attrs.get("object_id") or ""),
            }
            if str(target_id) not in identifiers:
                continue
            center = _as_xy(node.get("aabb_center"))
            if center is None:
                center = _as_xy(node.get("viz_aabb_center"))
            if center is None:
                center = _as_xy(attrs.get("aabb_center"))
            if center is None:
                continue
            return {
                "target_object_xy": center.tolist(),
                "target_object_source": "debug_semantic_graph",
                "graph_target_id": str(target_id),
            }
    return {}


def world_to_pixels(
    points_xy: np.ndarray,
    image_shape: tuple[int, int],
    resolution: float,
    origin_xy: np.ndarray,
    origin_yaw: float,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = points.reshape((-1, 2))
    delta = points - origin_xy[None, :]
    cosine, sine = math.cos(origin_yaw), math.sin(origin_yaw)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    cols = local_x / resolution
    rows = image_shape[0] - 1 - local_y / resolution
    return np.column_stack((rows, cols))


def _valid_pixel(point: np.ndarray, shape: tuple[int, int]) -> bool:
    return bool(
        len(point) >= 2
        and np.isfinite(point[:2]).all()
        and 0 <= point[0] < shape[0]
        and 0 <= point[1] < shape[1]
    )


def _draw_polyline(image: np.ndarray, points_rc: np.ndarray, color: tuple[int, int, int], thickness: int, dashed: bool = False) -> None:
    if len(points_rc) < 2:
        return
    shape = image.shape[:2]
    for index in range(len(points_rc) - 1):
        first, second = points_rc[index], points_rc[index + 1]
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            continue
        if dashed:
            # Draw short pieces along each segment so the route remains visible
            # over both free and occupied map cells.
            length = float(np.linalg.norm(second - first))
            pieces = max(1, int(length / max(8, thickness * 4)))
            for part in range(pieces):
                if part % 2:
                    continue
                start = first + (second - first) * (part / pieces)
                end = first + (second - first) * (min(part + 1, pieces) / pieces)
                cv2.line(image, tuple(np.round(start[::-1]).astype(int)), tuple(np.round(end[::-1]).astype(int)), color, thickness, cv2.LINE_AA)
        else:
            cv2.line(image, tuple(np.round(first[::-1]).astype(int)), tuple(np.round(second[::-1]).astype(int)), color, thickness, cv2.LINE_AA)


def _draw_start_marker(image: np.ndarray, point_rc: np.ndarray, yaw: float, color: tuple[int, int, int], radius: int) -> None:
    if not _valid_pixel(point_rc, image.shape[:2]):
        return
    center = np.round(point_rc[::-1]).astype(int)
    # A triangle points in the robot's world yaw direction.  The image has a
    # downward row axis, hence the minus sign on the local y component.
    direction = np.asarray([math.cos(yaw), -math.sin(yaw)], dtype=float)
    side = np.asarray([-direction[1], direction[0]], dtype=float)
    vertices = np.vstack(
        [center + direction * (radius * 1.35), center - direction * radius + side * radius * 0.8, center - direction * radius - side * radius * 0.8]
    ).round().astype(np.int32)
    cv2.fillConvexPoly(image, vertices, color, cv2.LINE_AA)
    cv2.polylines(image, [vertices], True, (20, 20, 20), max(1, radius // 5), cv2.LINE_AA)


def _draw_label(image: np.ndarray, text: str, point_rc: np.ndarray, color: tuple[int, int, int]) -> None:
    if not _valid_pixel(point_rc, image.shape[:2]):
        return
    x, y = np.round(point_rc[::-1]).astype(int)
    cv2.putText(image, text, (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(image, text, (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_legend(image: np.ndarray, entries: list[tuple[str, tuple[int, int, int]]]) -> None:
    if not entries:
        return
    line_height = 19
    width = min(image.shape[1] - 8, 260)
    height = line_height * len(entries) + 12
    overlay = image.copy()
    cv2.rectangle(overlay, (4, 4), (4 + width, 4 + height), (245, 245, 245), -1)
    image[:] = cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0)
    for index, (label, color) in enumerate(entries):
        y = 19 + index * line_height
        cv2.line(image, (11, y - 5), (32, y - 5), color, 3, cv2.LINE_AA)
        cv2.putText(image, label, (39, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)


def _event_geometry(
    event: dict[str, Any],
    trajectory: np.ndarray,
    trajectory_steps: np.ndarray,
) -> dict[str, Any]:
    validation = event.get("interaction_pose_validation")
    if not isinstance(validation, dict):
        validation = {}
    approach = _as_xyyaw(event.get("approach_goal_xyyaw"))
    actual = _as_xyyaw(validation.get("actual_pose_xyyaw"))
    target = _as_xy(validation.get("target_center_xy"))
    if target is None:
        target = _as_xy(event.get("target_center_xy"))
    if target is None:
        target = _as_xy(event.get("portal_center_xy"))
    step = _finite_float(event.get("force_applied_step"), None)
    if step is None:
        step = _finite_float(event.get("step"), None)
    trajectory_pose = None
    if len(trajectory) and step is not None and len(trajectory_steps):
        trajectory_pose = trajectory[int(np.argmin(np.abs(trajectory_steps - step)))]
    if actual is None and trajectory_pose is not None:
        actual = [float(trajectory_pose[0]), float(trajectory_pose[1]), 0.0]
    if actual is None and approach is not None and len(trajectory):
        # A few older drawer/door records omit both validation and a force
        # step.  Use the nearest recorded pose as the executed endpoint so the
        # approach segment remains visible instead of silently disappearing.
        nearest = int(np.argmin(np.linalg.norm(trajectory - np.asarray(approach[:2]), axis=1)))
        actual = [float(trajectory[nearest, 0]), float(trajectory[nearest, 1]), 0.0]
    return {
        "event_id": str(event.get("event_id") or ""),
        "command_id": str(event.get("command_id") or ""),
        "node_id": str(event.get("node_id") or event.get("object_id") or ""),
        "action": str(event.get("action") or ""),
        "success": _as_bool(
            event.get("success", str(event.get("status") or "").casefold() in {"succeeded", "success"})
        ),
        "status": str(event.get("status") or ""),
        "step": int(step) if step is not None else None,
        "approach_goal_xyyaw": approach,
        "actual_pose_xyyaw": actual,
        "target_center_xy": target.tolist() if target is not None else None,
        "trajectory_pose_xyyaw": trajectory_pose.tolist() + [0.0] if trajectory_pose is not None else None,
        "failure_reason": str(event.get("failure_reason") or ""),
    }


def _route_geometry(route: dict[str, Any] | None) -> dict[str, Any]:
    if not route:
        return {}
    start = _as_xyyaw(route.get("start_xyyaw"))
    if start is None:
        # ``robot_base_pose`` is [x,y,z,qw,qx,qy,qz] in the route file.
        pose = route.get("robot_base_pose")
        if isinstance(pose, (list, tuple)) and len(pose) >= 7:
            yaw = 2.0 * math.atan2(float(pose[6]), float(pose[3]))
            start = [float(pose[0]), float(pose[1]), float(yaw)]
    approach = _as_xyyaw(route.get("door_approach_xyyaw"))
    portal = _as_xy(route.get("portal_center_xy"))
    goal = _as_xyyaw(route.get("far_goal_xyyaw"))
    points: list[Any] = [start[:2] if start else None, approach[:2] if approach else None, portal]
    post = route.get("post_interaction_path_xy")
    if isinstance(post, list):
        points.extend(post)
    elif goal is not None:
        points.append(goal[:2])
    route_xy = _dedupe_points(points)
    return {
        "start_xyyaw": start,
        "approach_xyyaw": approach,
        "portal_center_xy": portal.tolist() if portal is not None else None,
        "goal_xyyaw": goal,
        "route_xy": route_xy,
    }


def render(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    image, resolution, origin_xy, origin_yaw, map_yaml, map_image = load_map(run_dir, args.map_yaml)
    trajectory_path = _resolve_artifact(run_dir, "trajectory.csv", args.trajectory)
    events_path = _resolve_artifact(run_dir, "force_interaction_events.json", args.events)
    trajectory, yaws, trajectory_steps, _ = load_trajectory(trajectory_path)
    events = load_force_events(events_path)

    route, route_source = _route_from_config(args.route_config, args.route_id)
    route_geometry = _route_geometry(route)
    benchmark_episode = _episode_from_benchmark(args.benchmark, args.episode_index, args.case_id)
    warnings: list[str] = []
    run_house_index = _run_house_index(run_dir)
    benchmark_house_index = _benchmark_house_index(benchmark_episode)
    raw_split = str(os.environ.get("RAW_DATA_SPLIT", "train")).strip().lower()
    benchmark_split = _benchmark_split(benchmark_episode)
    if (
        benchmark_episode is not None
        and benchmark_split is not None
        and benchmark_split != raw_split
        and not args.allow_benchmark_mismatch
    ):
        # A val benchmark episode with the same integer house index is not a
        # valid GT for the legacy train launch.  Do not draw a misleading star
        # or route; the metadata still records why GT was withheld.
        warnings.append(
            f"benchmark split {benchmark_split!r} differs from raw split {raw_split!r}; "
            "benchmark GT suppressed (pass --allow-benchmark-mismatch to override)"
        )
        benchmark_episode = None
    if (
        benchmark_episode is not None
        and run_house_index is not None
        and benchmark_house_index is not None
        and run_house_index != benchmark_house_index
        and not args.allow_benchmark_mismatch
    ):
        # Object instance ids are not globally unique across ProcTHOR houses;
        # suppress all benchmark-derived target/route geometry when the raw
        # launch explicitly identifies a different house.  Fixed-route geometry
        # (when supplied separately) remains available below.
        warnings.append(
            f"benchmark house_index {benchmark_house_index} differs from raw house_ind "
            f"{run_house_index}; benchmark GT suppressed "
            "(pass --allow-benchmark-mismatch to override)"
        )
        benchmark_episode = None
    benchmark_geometry = _benchmark_geometry(benchmark_episode)
    target_geometry = _runtime_target_geometry(args.target_selection)
    benchmark_target_id = None
    if isinstance(benchmark_episode, dict):
        nav = benchmark_episode.get("interactive_nav")
        if isinstance(nav, dict) and isinstance(nav.get("target"), dict):
            benchmark_target_id = nav["target"].get("selected_instance")
    graph_geometry = _graph_target_geometry(run_dir, benchmark_target_id)

    # Fixed-route geometry has precedence because it is in the same train scene
    # and coordinate frame as the raw run.  Benchmark/target sidecars fill gaps
    # for runtime object-goal runs and are explicitly labelled in metadata.
    geometry = dict(benchmark_geometry)
    geometry.update({key: value for key, value in target_geometry.items() if value not in (None, [], {})})
    geometry.update({key: value for key, value in graph_geometry.items() if value not in (None, [], {})})
    geometry.update({key: value for key, value in route_geometry.items() if value not in (None, [], {})})
    gt_start_offset_m = None
    gt_start_candidate = _as_xy(geometry.get("start_xyyaw"))
    if gt_start_candidate is not None and len(trajectory):
        gt_start_offset_m = float(np.linalg.norm(np.asarray(trajectory[0]) - gt_start_candidate))
    try:
        suppress_route_offset_m = float(os.environ.get("RAW_TOPDOWN_GT_START_SUPPRESS_M", "2.0"))
    except ValueError:
        suppress_route_offset_m = 2.0
    if not geometry.get("route_xy") and geometry.get("start_xyyaw") and geometry.get("goal_xyyaw"):
        geometry["route_xy"] = _dedupe_points([geometry["start_xyyaw"][:2], geometry["goal_xyyaw"][:2]])
    if route is not None:
        geometry_source = route_source
    elif benchmark_episode is not None:
        geometry_source = "benchmark"
    elif target_geometry:
        geometry_source = str(target_geometry.get("source", "target_selection"))
    elif graph_geometry:
        geometry_source = "debug_semantic_graph"
    else:
        geometry_source = "unavailable"

    if (
        gt_start_offset_m is not None
        and gt_start_offset_m > max(0.0, suppress_route_offset_m)
        and geometry_source in {"route_config", "benchmark"}
        and geometry.get("route_xy")
    ):
        # Keep a valid GT target/goal marker but avoid suggesting that a route
        # from a different initial placement was the executed GT path.
        geometry["route_xy"] = []
        warnings.append(
            f"GT route suppressed because actual/GT start offset is {gt_start_offset_m:.3f} m"
        )

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Give unknown cells a neutral gray while retaining the recorder's occupied
    # (black) and free (white) semantics.
    canvas[image == 205] = (180, 180, 180)
    thickness = max(2, int(round(min(canvas.shape[:2]) / 260.0)))
    actual_color = (255, 220, 0)  # cyan in BGR
    gt_color = (190, 0, 190)  # magenta
    interaction_color = (0, 145, 255)  # orange
    initial_color = (255, 80, 0)  # blue
    goal_color = (0, 215, 255)  # yellow

    if len(trajectory):
        actual_pixels = world_to_pixels(trajectory, canvas.shape[:2], resolution, origin_xy, origin_yaw)
        _draw_polyline(canvas, actual_pixels, actual_color, thickness + 1)
    else:
        actual_pixels = np.empty((0, 2))

    route_xy = np.asarray(geometry.get("route_xy") or [], dtype=float)
    route_pixels = world_to_pixels(route_xy, canvas.shape[:2], resolution, origin_xy, origin_yaw) if len(route_xy) else np.empty((0, 2))
    if len(route_pixels) >= 2:
        _draw_polyline(canvas, route_pixels, gt_color, thickness, dashed=True)

    # The first recorder pose is the actual initial position.  A benchmark or
    # fixed-route sidecar may also contain a frozen GT start; keep it separate
    # so a random raw launch cannot accidentally label GT as the executed pose.
    gt_start_pose = _as_xyyaw(geometry.get("start_xyyaw"))
    trajectory_start_pose = (
        [float(trajectory[0, 0]), float(trajectory[0, 1]), float(yaws[0] if len(yaws) else 0.0)]
        if len(trajectory)
        else None
    )
    start_pose = trajectory_start_pose or gt_start_pose
    if trajectory_start_pose is not None:
        start_source = "trajectory_first_pose"
    else:
        start_source = geometry_source if start_pose is not None else "unavailable"
    if start_pose is not None:
        start_pixel = world_to_pixels(np.asarray([start_pose[:2]]), canvas.shape[:2], resolution, origin_xy, origin_yaw)[0]
        _draw_start_marker(canvas, start_pixel, float(start_pose[2]), initial_color, max(6, 4 * thickness))
        _draw_label(canvas, "initial", start_pixel, initial_color)
    else:
        start_pixel = np.empty((0,))

    gt_start_pixel = np.empty((0,))
    gt_start_separation_m = None
    if gt_start_pose is not None:
        gt_start_pixel = world_to_pixels(np.asarray([gt_start_pose[:2]]), canvas.shape[:2], resolution, origin_xy, origin_yaw)[0]
        if trajectory_start_pose is not None:
            gt_start_separation_m = float(
                math.hypot(
                    gt_start_pose[0] - trajectory_start_pose[0],
                    gt_start_pose[1] - trajectory_start_pose[1],
                )
            )
        # Avoid hiding the actual marker when the two poses coincide.  When
        # they differ, the square makes a benchmark-start mismatch explicit.
        if gt_start_separation_m is None or gt_start_separation_m > 0.20:
            if _valid_pixel(gt_start_pixel, canvas.shape[:2]):
                center = tuple(np.round(gt_start_pixel[::-1]).astype(int))
                side = max(8, 5 * thickness)
                cv2.rectangle(canvas, (center[0] - side, center[1] - side), (center[0] + side, center[1] + side), gt_color, max(1, thickness), cv2.LINE_AA)
                _draw_label(canvas, "GT start", gt_start_pixel, gt_color)

    goal_pose = _as_xyyaw(geometry.get("goal_xyyaw"))
    if goal_pose is not None:
        goal_pixel = world_to_pixels(np.asarray([goal_pose[:2]]), canvas.shape[:2], resolution, origin_xy, origin_yaw)[0]
        if _valid_pixel(goal_pixel, canvas.shape[:2]):
            center = tuple(np.round(goal_pixel[::-1]).astype(int))
            cv2.drawMarker(canvas, center, goal_color, cv2.MARKER_STAR, max(12, 7 * thickness), thickness, cv2.LINE_AA)
            _draw_label(canvas, "GT goal", goal_pixel, goal_color)
    else:
        goal_pixel = np.empty((0,))

    target_object_xy = _as_xy(geometry.get("target_object_xy"))
    if target_object_xy is not None:
        target_object_pixel = world_to_pixels(
            np.asarray([target_object_xy]), canvas.shape[:2], resolution, origin_xy, origin_yaw
        )[0]
        if _valid_pixel(target_object_pixel, canvas.shape[:2]):
            center = tuple(np.round(target_object_pixel[::-1]).astype(int))
            cv2.drawMarker(
                canvas,
                center,
                goal_color,
                cv2.MARKER_STAR,
                max(14, 8 * thickness),
                thickness,
                cv2.LINE_AA,
            )
            _draw_label(canvas, "GT target object", target_object_pixel, goal_color)
    else:
        target_object_pixel = np.empty((0,))

    # If the benchmark start differs from the first recorder pose, retain both
    # instead of silently presenting a benchmark pose as the executed start.
    actual_start_pixel = np.empty((0,))
    if len(trajectory):
        actual_start = world_to_pixels(
            np.asarray([trajectory[0]]), canvas.shape[:2], resolution, origin_xy, origin_yaw
        )[0]
        if start_pose is None or np.linalg.norm(np.asarray(trajectory[0]) - np.asarray(start_pose[:2])) > 1e-3:
            actual_start_pixel = actual_start
            if _valid_pixel(actual_start_pixel, canvas.shape[:2]):
                cv2.circle(
                    canvas,
                    tuple(np.round(actual_start_pixel[::-1]).astype(int)),
                    max(6, 3 * thickness),
                    (255, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                _draw_label(canvas, "actual start", actual_start_pixel, (255, 0, 255))

    interactions: list[dict[str, Any]] = []
    for event in events:
        row = _event_geometry(event, trajectory, trajectory_steps)
        interactions.append(row)
        actual_pose = _as_xyyaw(row.get("actual_pose_xyyaw"))
        approach_pose = _as_xyyaw(row.get("approach_goal_xyyaw"))
        target_xy = _as_xy(row.get("target_center_xy"))
        connector = []
        trajectory_pose = _as_xyyaw(row.get("trajectory_pose_xyyaw"))
        if trajectory_pose is not None:
            connector.append(trajectory_pose[:2])
        if actual_pose is not None:
            connector.append(actual_pose[:2])
        if approach_pose is not None:
            connector.append(approach_pose[:2])
        if target_xy is not None:
            connector.append(target_xy)
        if len(connector) >= 2:
            pixels = world_to_pixels(np.asarray(connector), canvas.shape[:2], resolution, origin_xy, origin_yaw)
            _draw_polyline(canvas, pixels, interaction_color, max(1, thickness), dashed=False)
        if approach_pose is not None:
            approach_pixel = world_to_pixels(np.asarray([approach_pose[:2]]), canvas.shape[:2], resolution, origin_xy, origin_yaw)[0]
            if _valid_pixel(approach_pixel, canvas.shape[:2]):
                cv2.circle(canvas, tuple(np.round(approach_pixel[::-1]).astype(int)), max(5, 3 * thickness), interaction_color, 2, cv2.LINE_AA)
        if target_xy is not None:
            target_pixel = world_to_pixels(np.asarray([target_xy]), canvas.shape[:2], resolution, origin_xy, origin_yaw)[0]
            marker_color = (0, 190, 0) if row["success"] else (0, 0, 255)
            if _valid_pixel(target_pixel, canvas.shape[:2]):
                cv2.drawMarker(
                    canvas,
                    tuple(np.round(target_pixel[::-1]).astype(int)),
                    marker_color,
                    cv2.MARKER_DIAMOND,
                    max(11, 6 * thickness),
                    thickness,
                    cv2.LINE_AA,
                )
                _draw_label(canvas, "interaction", target_pixel, marker_color)

    # The route's approach-to-portal segment is the GT interaction path.  Keep
    # it visually distinct from the full GT navigation route.
    gt_interaction_points = []
    if route_geometry.get("approach_xyyaw") is not None:
        gt_interaction_points.append(route_geometry["approach_xyyaw"][:2])
    if route_geometry.get("portal_center_xy") is not None:
        gt_interaction_points.append(route_geometry["portal_center_xy"])
    if len(gt_interaction_points) >= 2:
        pixels = world_to_pixels(np.asarray(gt_interaction_points), canvas.shape[:2], resolution, origin_xy, origin_yaw)
        _draw_polyline(canvas, pixels, (255, 0, 255), max(1, thickness), dashed=True)

    legend_entries = [
        ("actual walking path", actual_color),
        ("GT route", gt_color),
        ("interaction path", interaction_color),
        ("initial pose", initial_color),
        ("GT goal", goal_color),
        ("GT target object", goal_color),
        ("actual start (if offset)", (255, 0, 255)),
    ]
    if gt_start_pose is not None and (gt_start_separation_m is None or gt_start_separation_m > 0.20):
        legend_entries.append(("GT start", gt_color))
    _draw_legend(canvas, legend_entries)
    cv2.putText(canvas, "Raw Evo top-down", (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Raw Evo top-down", (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)

    output_png = args.output or (run_dir / "raw_episode_topdown.png")
    output_json = args.metadata or output_png.with_suffix(".json")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_png), canvas):
        raise OSError(f"Could not write top-down image: {output_png}")

    path_distance = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()) if len(trajectory) >= 2 else 0.0
    if not len(trajectory):
        warnings.append("trajectory.csv missing or contained no valid poses")
    if route is None and benchmark_episode is None and not target_geometry and not graph_geometry:
        warnings.append("GT geometry unavailable; initial pose falls back to first trajectory pose")
    if route is None and benchmark_episode is not None and benchmark_split == raw_split:
        warnings.append(f"benchmark geometry accepted for matching split {raw_split!r}")
    if gt_start_separation_m is not None and gt_start_separation_m > 0.20:
        warnings.append(
            f"actual initial pose differs from GT start by {gt_start_separation_m:.3f} m; both markers are shown"
        )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "renderer": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "output_png": str(output_png),
        "output_metadata": str(output_json),
        "map": {
            "yaml": str(map_yaml),
            "image": str(map_image),
            "resolution_m": resolution,
            "origin_xy": origin_xy.tolist(),
            "origin_yaw_rad": origin_yaw,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        },
        "trajectory": {
            "csv": str(trajectory_path),
            "pose_count": int(len(trajectory)),
            "distance_m": path_distance,
            "start_xyyaw": start_pose,
            "final_xyyaw": ([float(trajectory[-1, 0]), float(trajectory[-1, 1]), float(yaws[-1])] if len(trajectory) else None),
        },
        "initial_pose": {
            "xyyaw": start_pose,
            "source": start_source,
            "trajectory_first_xyyaw": trajectory_start_pose,
            "gt_start_xyyaw": gt_start_pose,
            "gt_start_separation_m": gt_start_separation_m,
        },
        "gt": {
            "source": geometry_source,
            "available": bool(goal_pose is not None or target_object_xy is not None or len(route_xy)),
            "start_xyyaw": gt_start_pose,
            "goal_xyyaw": goal_pose,
            "target_object_xy": target_object_xy.tolist() if target_object_xy is not None else None,
            "target_object_source": geometry.get("target_object_source"),
            "graph_target_id": geometry.get("graph_target_id"),
            "route_xy": route_xy.tolist() if len(route_xy) else [],
            "interaction_path_xy": gt_interaction_points,
            "route_id": args.route_id or "",
            "raw_split": raw_split,
            "benchmark_split": benchmark_split,
            "raw_house_index": run_house_index,
            "benchmark_house_index": benchmark_house_index,
            "house_match": (
                None
                if run_house_index is None or benchmark_house_index is None
                else bool(run_house_index == benchmark_house_index)
            ),
        },
        "interactions": interactions,
        "events_json": str(events_path),
        "event_count": len(interactions),
        "warnings": warnings,
    }
    _atomic_json(output_json, metadata)
    return metadata


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--map-yaml", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--route-config", type=Path)
    parser.add_argument("--route-id", default=os.environ.get("ROUTE_ID", ""))
    parser.add_argument(
        "--disable-route-config",
        action="store_true",
        help="Do not read ROUTE_CONFIG from the environment (for random raw runs).",
    )
    parser.add_argument("--target-selection", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=int(
            os.environ.get(
                "INTERACTIVE_NAV_TOPDOWN_EPISODE_INDEX",
                os.environ.get("EPISODE_INDEX", "0"),
            )
            or 0
        ),
    )
    parser.add_argument("--case-id", default=os.environ.get("INTERACTIVE_NAV_TOPDOWN_CASE_ID", os.environ.get("CASE_ID", "")) or None)
    parser.add_argument(
        "--allow-benchmark-mismatch",
        action="store_true",
        help="Draw benchmark GT even when its data split differs from RAW_DATA_SPLIT.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.map_yaml = args.map_yaml or _env_path("INTERACTIVE_NAV_TOPDOWN_MAP_YAML", "TOPDOWN_MAP_YAML")
    args.trajectory = args.trajectory or _env_path("INTERACTIVE_NAV_TOPDOWN_TRAJECTORY", "TOPDOWN_TRAJECTORY")
    args.events = args.events or _env_path("INTERACTIVE_NAV_TOPDOWN_EVENTS", "TOPDOWN_EVENTS")
    if not args.disable_route_config:
        args.route_config = args.route_config or _env_path("INTERACTIVE_NAV_TOPDOWN_ROUTE_CONFIG", "ROUTE_CONFIG")
    else:
        args.route_config = None
    args.target_selection = args.target_selection or _env_path("INTERACTIVE_NAV_TOPDOWN_TARGET_SELECTION", "RUNTIME_TARGET_SELECTION_INPUT_PATH")
    args.benchmark = args.benchmark or _env_path("INTERACTIVE_NAV_TOPDOWN_BENCHMARK", "TOPDOWN_BENCHMARK")
    try:
        metadata = render(args)
    except Exception as exc:  # shell wrapper records the traceback in topdown.log
        print(f"raw top-down rendering failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_png": metadata["output_png"],
                "output_metadata": metadata["output_metadata"],
                "event_count": metadata["event_count"],
                "gt_available": metadata["gt"]["available"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
