from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.utils.pose import pose_mat_to_7d, pos_quat_to_pose_mat
from scripts.InteractiveNav import benchmark_longest_nav_paths as nav_paths
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav import interactive_nav_v3 as v3
from scripts.InteractiveNav.select_container_interaction_candidates import (
    build_dynamic_collection_plan,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("molmo_spaces.env.camera_manager").setLevel(logging.WARNING)
logging.getLogger("molmo_spaces.env.env").setLevel(logging.INFO)

DEFAULT_BENCHMARK_DIR = Path(
    "/home/user/ldl/molmospaces/assets/benchmarks/molmospaces-bench-v2/"
    "procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/output/"
    "container_interaction_benchmark_preview_first10"
)
VIEW_PROFILES = ("default", "drawer_low_view")
SMALL_TARGET_CATEGORIES = {"pen", "pencil"}


def load_benchmark_episodes(benchmark_dir: Path) -> list[dict[str, Any]]:
    benchmark_path = benchmark_dir / "benchmark.json" if benchmark_dir.is_dir() else benchmark_dir
    with open(benchmark_path) as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    return list(episodes)


def build_episode_args(args: argparse.Namespace, episode: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        scene_dataset=episode["scene_dataset"],
        data_split=episode["data_split"],
        robot=episode.get("robot", {}).get("robot_name", args.robot),
        variant=args.variant,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def load_episode_context(args: argparse.Namespace, episode: dict[str, Any]) -> probe.LoadedContext:
    episode_args = build_episode_args(args, episode)
    return probe.load_scene_context(episode_args, int(episode["house_index"]))


def close_all_containers(env, containers: list[dict[str, Any]]) -> None:
    for container in containers:
        probe.set_all_articulation_joints_closed(env, container, container["joints"])


def open_all_available_doors(ctx: probe.LoadedContext) -> list[dict[str, Any]]:
    """Open every directly addressable door without failing on fixed doorway roots."""
    transitions: list[dict[str, Any]] = []
    try:
        transitions.extend(emi.open_all_doors(ctx.env))
    except Exception as exc:
        log.warning("Root-level door opening was partial: %s", exc)
    for door in probe.collect_door_records(ctx):
        try:
            probe.set_articulation_state_by_record(
                ctx.env,
                door,
                int(door["hinge_joint_index"]),
                float(door["open_value"]),
            )
            transitions.append(
                {
                    "door_name": door["name"],
                    "joint_name": door["hinge_joint_name"],
                    "target_value": float(door["open_value"]),
                    "actual_value": probe.joint_value_by_name(
                        ctx.env, door["hinge_joint_name"]
                    ),
                }
            )
        except Exception as exc:
            log.warning("Could not open door %s: %s", door["name"], exc)
    return transitions


def semantic_open_fraction(value: float, closed_value: float, open_value: float) -> float:
    span = float(open_value) - float(closed_value)
    if abs(span) < 1e-9:
        raise ValueError("Cannot compute open fraction for a zero-range joint")
    return float(np.clip((float(value) - float(closed_value)) / span, 0.0, 1.0))


def validate_all_doors_open(
    ctx: probe.LoadedContext,
    *,
    threshold: float = 0.99,
) -> dict[str, Any]:
    rows = []
    for door in probe.collect_door_records(ctx):
        value = probe.joint_value_by_name(ctx.env, door["hinge_joint_name"])
        fraction = semantic_open_fraction(
            value, door["closed_value"], door["open_value"]
        )
        rows.append(
            {
                "door_name": door["name"],
                "joint_name": door["hinge_joint_name"],
                "joint_value": value,
                "closed_value": float(door["closed_value"]),
                "open_value": float(door["open_value"]),
                "open_fraction": fraction,
                "passed": fraction >= threshold,
            }
        )
    return {
        "threshold": threshold,
        "door_count": len(rows),
        "all_open": all(row["passed"] for row in rows),
        "doors": rows,
    }


def validate_all_containers_closed(
    ctx: probe.LoadedContext,
    containers: list[dict[str, Any]],
    *,
    threshold: float = 0.01,
) -> dict[str, Any]:
    rows = []
    for container in containers:
        for joint in container["joints"]:
            if probe.joint_mujoco_type_name(ctx.env, joint) not in {"hinge", "slide"}:
                continue
            value = probe.joint_value_by_name(ctx.env, joint["joint_name"])
            fraction = semantic_open_fraction(
                value, joint["closed_value"], joint["open_value"]
            )
            rows.append(
                {
                    "container_name": container["name"],
                    "joint_name": joint["joint_name"],
                    "joint_value": value,
                    "closed_value": float(joint["closed_value"]),
                    "open_value": float(joint["open_value"]),
                    "open_fraction": fraction,
                    "passed": fraction <= threshold,
                }
            )
    return {
        "threshold": threshold,
        "joint_count": len(rows),
        "all_closed": all(row["passed"] for row in rows),
        "joints": rows,
    }


def articulation_initial_states(
    ctx: probe.LoadedContext,
    containers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    env = ctx.env
    states: list[dict[str, Any]] = []
    seen: set[str] = set()
    for door in probe.collect_door_records(ctx):
        joint_name = door["hinge_joint_name"]
        if joint_name in seen:
            continue
        seen.add(joint_name)
        value = probe.joint_value_by_name(env, joint_name)
        states.append(
            {
                "object_name": door["name"],
                "joint_name": joint_name,
                "joint_index": int(door["hinge_joint_index"]),
                "position": value,
                "open_fraction": semantic_open_fraction(
                    value, door["closed_value"], door["open_value"]
                ),
            }
        )
    for container in containers:
        for joint in container["joints"]:
            if probe.joint_mujoco_type_name(env, joint) not in {"hinge", "slide"}:
                continue
            joint_name = joint["joint_name"]
            if joint_name in seen:
                continue
            seen.add(joint_name)
            value = probe.joint_value_by_name(env, joint_name)
            states.append(
                {
                    "object_name": container["name"],
                    "joint_name": joint_name,
                    "joint_index": int(joint["joint_index"]),
                    "position": value,
                    "open_fraction": semantic_open_fraction(
                        value, joint["closed_value"], joint["open_value"]
                    ),
                }
            )
    return states


def visibility_trace_reveals_on_final_step(
    trace: list[dict[str, Any]],
    threshold: float,
    min_visible_pixels: int = 0,
) -> tuple[bool, str | None]:
    def visible(row: dict[str, Any]) -> bool:
        fraction = float(row["visibility_fraction"])
        fraction_visible = fraction > 0.0 if threshold <= 0.0 else fraction >= threshold
        return fraction_visible or int(row.get("visible_pixels", 0)) >= min_visible_pixels > 0

    if len(trace) < 2:
        return False, "visibility_trace_too_short"
    if visible(trace[0]):
        return False, "target_visible_before_interaction"
    if any(visible(row) for row in trace[1:-1]):
        return False, "target_visible_before_controlling_joint"
    if not visible(trace[-1]):
        return False, "target_not_visible_after_interaction"
    return True, None


def slide_trace_has_consistent_partial_motion(
    trace: list[dict[str, Any]],
    *,
    min_joint_motion_m: float = 0.05,
    min_motion_ratio: float = 0.5,
) -> tuple[bool, dict[str, float]]:
    """Accept a physically useful partial pull when the target follows the drawer."""
    if len(trace) < 2:
        return False, {"joint_motion_m": 0.0, "object_motion_m": 0.0, "motion_ratio": 0.0}
    first = trace[0]
    final = trace[-1]
    joint_delta = np.asarray(final["target_joint_aabb"]["center"], dtype=float) - np.asarray(
        first["target_joint_aabb"]["center"], dtype=float
    )
    object_delta = np.asarray(final["object_position"], dtype=float) - np.asarray(
        first["object_position"], dtype=float
    )
    joint_motion = float(np.linalg.norm(joint_delta))
    object_motion = float(np.linalg.norm(object_delta))
    if joint_motion <= 1e-9:
        motion_ratio = 0.0
    else:
        motion_ratio = float(np.dot(object_delta, joint_delta / joint_motion) / joint_motion)
    metrics = {
        "joint_motion_m": joint_motion,
        "object_motion_m": object_motion,
        "motion_ratio": motion_ratio,
    }
    return joint_motion >= min_joint_motion_m and motion_ratio >= min_motion_ratio, metrics


def robot_pose_from_episode(episode: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(episode["task"]["robot_base_pose"], dtype=float)
    return pos_quat_to_pose_mat(pose[:3], pose[3:7])


def goal_from_robot_pose(robot_pose: np.ndarray) -> tuple[list[float], float]:
    point = np.asarray(robot_pose[:3, 3], dtype=float).copy()
    point[2] = 0.0
    yaw = float(np.arctan2(robot_pose[1, 0], robot_pose[0, 0]))
    return point.tolist(), yaw


def target_category(record: dict[str, Any]) -> str:
    category = str(record.get("category") or record["name"].split("_", 1)[0])
    return category.replace("_", " ").lower()


def rough_target_record(
    record: dict[str, Any],
    container: dict[str, Any],
    indexed_episodes: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    object_center = np.asarray(record["aabb_center"], dtype=float)
    container_center = np.asarray(container["aabb_center"], dtype=float)
    starts = []
    for episode_index, episode in indexed_episodes:
        pose = np.asarray(episode["task"]["robot_base_pose"], dtype=float)
        start_xyz = pose[:3]
        starts.append(
            {
                "episode_index": episode_index,
                "robot_base_pose": pose.tolist(),
                "distance_to_object_m": float(np.linalg.norm(start_xyz - object_center)),
                "planar_distance_to_object_m": float(
                    np.linalg.norm(start_xyz[:2] - object_center[:2])
                ),
                "distance_to_container_m": float(
                    np.linalg.norm(start_xyz - container_center)
                ),
                "planar_distance_to_container_m": float(
                    np.linalg.norm(start_xyz[:2] - container_center[:2])
                ),
            }
        )
    return {
        "name": record["name"],
        "category": target_category(record),
        "aabb_center": object_center.tolist(),
        "aabb_size": np.asarray(record["aabb_size"], dtype=float).tolist(),
        "source_starts": starts,
    }


def load_candidate_manifest(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    with open(path) as handle:
        payload = json.load(handle)
    return {int(row["house_index"]): row for row in payload.get("houses", [])}


def build_scene_map(args: argparse.Namespace, ctx: probe.LoadedContext):
    return emi.build_live_procthor_map(
        ctx.env.current_model,
        ctx.env.current_data,
        model_path=str(ctx.env.current_model_path),
        px_per_m=args.px_per_m,
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
        open_threshold=args.open_threshold,
        treat_all_non_interactive_doorways_as_open=True,
    )


def source_start_validation(
    ctx: probe.LoadedContext,
    container: dict[str, Any],
    object_name: str,
    source_episode: dict[str, Any],
    scene_map,
    goal_pose: np.ndarray,
    visibility_threshold: float,
) -> dict[str, Any]:
    start_pose = robot_pose_from_episode(source_episode)
    start_trace = probe.container_visibility_trace(
        ctx,
        container,
        object_name,
        [],
        start_pose,
        view_profile="default",
    )["trace"][0]
    start_xy = np.asarray(start_pose[:2, 3], dtype=float)
    goal_xy = np.asarray(goal_pose[:2, 3], dtype=float)
    path = emi.compute_path_from_map(scene_map, start_xy, goal_xy, downscale_factor=1)
    return {
        "valid": bool(
            float(start_trace["visibility_fraction"]) <= 0.0
            and int(start_trace["visible_pixels"]) == 0
            and path is not None
        ),
        "start_pose": pose_mat_to_7d(start_pose).tolist(),
        "start_visibility_fraction": float(start_trace["visibility_fraction"]),
        "start_visible_pixels": int(start_trace["visible_pixels"]),
        "path_found": path is not None,
        "path_length_m": emi.path_length(path),
    }


def analyze_object_pair(
    args: argparse.Namespace,
    ctx: probe.LoadedContext,
    container: dict[str, Any],
    object_record: dict[str, Any],
    dependency_rows: list[dict[str, Any]],
    case_id: str,
    candidate_acceptor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    valid_joint_candidates: list[dict[str, Any]] = []
    joint_failures: list[dict[str, Any]] = []
    dependencies_by_index = {
        int(row["joint_index"]): row for row in dependency_rows
    }
    for joint in container["joints"]:
        joint_index = int(joint["joint_index"])
        joint_type = probe.joint_mujoco_type_name(ctx.env, joint)
        if joint_type not in {"hinge", "slide"}:
            continue
        try:
            joint_sequence = probe.articulation_dependency_order(joint_index, dependency_rows)
        except ValueError as exc:
            joint_failures.append(
                {"joint_index": joint_index, "reason": "cyclic_dependency", "error": str(exc)}
            )
            continue

        binding = {
            "applicable": False,
            "consistent": True,
            "reason": "target_joint_not_slide",
        }
        containment = None
        if joint_type == "slide":
            containment = probe.object_in_closed_joint_box(
                ctx,
                container,
                joint,
                object_record,
                padding=args.drawer_box_padding,
            )
            if not containment["contained"]:
                joint_failures.append(
                    {
                        "joint_index": joint_index,
                        "reason": "object_not_in_joint_compartment_box",
                        "containment": containment,
                    }
                )
                continue
            try:
                binding = probe.slide_compartment_object_binding(
                    ctx,
                    container,
                    object_record["name"],
                    joint_index,
                    joint_sequence[:-1],
                )
            except Exception as exc:
                joint_failures.append(
                    {
                        "joint_index": joint_index,
                        "reason": "object_binding_check_failed",
                        "error": str(exc),
                        "containment": containment,
                    }
                )
                continue
            dependency = dependencies_by_index[joint_index]
            if args.save_plots and all(
                key in binding
                for key in (
                    "closed_joint_box",
                    "open_joint_box",
                    "closed_object_box",
                    "open_object_box",
                )
            ):
                plot_path = (
                    args.output_dir
                    / "diagnostics"
                    / "slide_force_boxes"
                    / case_id
                    / f"joint_{joint_index}.png"
                )
                probe.save_slide_force_transition_plot(
                    plot_path,
                    container,
                    joint_index,
                    binding,
                    np.asarray(dependency["front_axis_xy"], dtype=float),
                )
                binding["plot_path"] = str(plot_path)
            if not binding.get("applicable", False):
                joint_failures.append(
                    {
                        "joint_index": joint_index,
                        "reason": binding.get("reason", "slide_binding_not_applicable"),
                        "binding": binding,
                        "containment": containment,
                    }
                )
                continue
            if not binding.get("drive", {}).get("reached", False):
                joint_failures.append(
                    {
                        "joint_index": joint_index,
                        "reason": "force_drive_not_reached_during_binding",
                        "binding": binding,
                        "containment": containment,
                    }
                )
                continue
            if not binding["consistent"]:
                joint_failures.append(
                    {
                        "joint_index": joint_index,
                        "reason": "object_not_bound_to_moving_compartment",
                        "binding": binding,
                        "containment": containment,
                    }
                )
                continue

        poses = probe.valid_robot_poses_for_joint_sequence(
            ctx,
            container,
            joint_sequence,
            desired_distance=args.interaction_distance,
            max_poses=args.max_poses_per_joint,
            front_axis_xy=probe.container_approach_axis(ctx.env, container),
        )
        if not poses:
            joint_failures.append(
                {"joint_index": joint_index, "reason": "no_collision_free_interaction_pose"}
            )
            continue

        selected = None
        trace_failures = []
        view_profiles = (
            ("drawer_low_view", "default") if joint_type == "slide" else VIEW_PROFILES
        )
        min_visible_pixels = 1 if (
            str(container.get("category", "")).lower() == "fridge"
            or target_category(object_record) in SMALL_TARGET_CATEGORIES
        ) else 0
        for robot_pose, pose_meta in poses:
            for view_profile in view_profiles:
                diagnostic_image_dir = None
                if joint_type == "slide" and args.save_images:
                    pose_tag = probe.sanitize_name(pose_meta["candidate_label"])
                    diagnostic_image_dir = (
                        args.output_dir
                        / "diagnostics"
                        / "slide_headcam"
                        / case_id
                        / f"joint_{joint_index}__{pose_tag}__{view_profile}"
                    )
                trace_result = probe.container_visibility_trace(
                    ctx,
                    container,
                    object_record["name"],
                    joint_sequence,
                    robot_pose,
                    view_profile=view_profile,
                    force_slide_joints=joint_type == "slide",
                    output_dir=diagnostic_image_dir,
                )
                valid, reason = visibility_trace_reveals_on_final_step(
                    trace_result["trace"],
                    0.0,
                    min_visible_pixels=min_visible_pixels,
                )
                drive_reached = bool(
                    (trace_result["trace"][-1].get("drive") or {}).get(
                        "reached", False
                    )
                )
                partial_motion_ok = False
                partial_motion = None
                if joint_type == "slide" and not drive_reached:
                    partial_motion_ok, partial_motion = slide_trace_has_consistent_partial_motion(
                        trace_result["trace"]
                    )
                if joint_type == "slide" and not drive_reached:
                    trace_failures.append(
                        {
                            "pose_label": pose_meta["candidate_label"],
                            "pose_meta": pose_meta,
                            "robot_pose": robot_pose.tolist(),
                            "view_profile": view_profile,
                            "view_state": trace_result["view_state"],
                            "reason": "force_drive_not_reached",
                            "partial_motion": partial_motion,
                            "trace": trace_result["trace"],
                        }
                    )
                    continue
                if valid:
                    candidate = {
                        "joint": joint,
                        "joint_sequence": joint_sequence,
                        "robot_pose": robot_pose,
                        "pose_meta": pose_meta,
                        "view_profile": view_profile,
                        "view_state": trace_result["view_state"],
                        "visibility_trace": trace_result["trace"],
                        "joint_type": joint_type,
                        "force_slide_joints": joint_type == "slide",
                        "binding": binding,
                        "containment": containment,
                        "minimum_visible_pixels": min_visible_pixels,
                        "force_drive_reached": drive_reached,
                        "accepted_partial_motion": partial_motion
                        if not drive_reached
                        else None,
                    }
                    if candidate_acceptor is not None:
                        acceptance = candidate_acceptor(candidate)
                        if not acceptance.get("accepted", False):
                            trace_failures.append(
                                {
                                    "pose_label": pose_meta["candidate_label"],
                                    "pose_meta": pose_meta,
                                    "robot_pose": robot_pose.tolist(),
                                    "view_profile": view_profile,
                                    "view_state": trace_result["view_state"],
                                    "reason": acceptance.get(
                                        "reason", "candidate_rejected_by_acceptor"
                                    ),
                                    "trace": trace_result["trace"],
                                }
                            )
                            continue
                        candidate.update(acceptance.get("metadata", {}))
                    selected = candidate
                    break
                trace_failures.append(
                    {
                        "pose_label": pose_meta["candidate_label"],
                        "pose_meta": pose_meta,
                        "robot_pose": robot_pose.tolist(),
                        "view_profile": view_profile,
                        "view_state": trace_result["view_state"],
                        "reason": reason,
                        "trace": trace_result["trace"],
                    }
                )
            if selected is not None:
                break
        if selected is None:
            joint_failures.append(
                {
                    "joint_index": joint_index,
                    "reason": "no_force_visibility_unlock"
                    if joint_type == "slide"
                    else "no_visibility_unlock",
                    "attempts": trace_failures,
                    "joint_sequence": joint_sequence,
                    "binding": binding,
                    "containment": containment,
                }
            )
            continue
        valid_joint_candidates.append(selected)

    if not valid_joint_candidates:
        return {
            "valid": False,
            "reason": "no_controlling_joint",
            "joint_failures": joint_failures,
        }
    valid_joint_candidates.sort(key=lambda candidate: int(candidate["joint"]["joint_index"]))
    selected = valid_joint_candidates[0]
    return {
        "valid": True,
        "selected": selected,
        "candidate_joint_indices": [
            int(candidate["joint"]["joint_index"]) for candidate in valid_joint_candidates
        ],
        "candidate_joint_results": valid_joint_candidates,
        "binding": selected["binding"],
        "multi_oracle": len(valid_joint_candidates) > 1,
        "joint_failures": joint_failures,
    }


def save_best_rejected_evidence(
    args: argparse.Namespace,
    ctx: probe.LoadedContext,
    container: dict[str, Any],
    object_record: dict[str, Any],
    case_id: str,
    analysis: dict[str, Any],
) -> dict[str, Any] | None:
    """Render one representative closed/final comparison for a rejected pair."""
    candidates = []
    for selected in analysis.get("candidate_joint_results", []):
        trace = selected.get("visibility_trace") or []
        if not trace:
            continue
        final = trace[-1]
        drive = final.get("drive") or {}
        container_fraction = float(final.get("container_visibility_fraction", 0.0))
        candidates.append(
            (
                int(bool(drive.get("reached", True))),
                int(final.get("visible_pixels", 0)),
                int(selected.get("view_profile") == "default"),
                -abs(container_fraction - 0.10),
                float(final.get("visibility_fraction", 0.0)),
                -abs(float(drive.get("final_error", 0.0))),
                {"joint_sequence": selected["joint_sequence"]},
                {
                    "pose_meta": selected.get("pose_meta"),
                    "robot_pose": probe.to_jsonable(selected["robot_pose"]),
                    "view_profile": selected["view_profile"],
                    "reason": "ambiguous_controlling_joint",
                    "trace": trace,
                },
            )
        )
    for failure in analysis.get("joint_failures", []):
        joint_sequence = failure.get("joint_sequence")
        if not joint_sequence:
            continue
        for attempt in failure.get("attempts", []):
            trace = attempt.get("trace") or []
            if not trace:
                continue
            final = trace[-1]
            drive = final.get("drive") or {}
            container_fraction = float(final.get("container_visibility_fraction", 0.0))
            framing_score = -abs(container_fraction - 0.10)
            candidates.append(
                (
                    int(bool(drive.get("reached", False))),
                    int(final.get("visible_pixels", 0)),
                    int(attempt.get("view_profile") == "default"),
                    framing_score,
                    float(final.get("visibility_fraction", 0.0)),
                    -abs(float(drive.get("final_error", 0.0))),
                    failure,
                    attempt,
                )
            )
    if not candidates:
        return None
    _, _, _, _, _, _, failure, attempt = max(candidates, key=lambda item: item[:6])
    joint_sequence = [int(index) for index in failure["joint_sequence"]]
    joints_by_index = {int(joint["joint_index"]): joint for joint in container["joints"]}
    target_joint = joints_by_index[joint_sequence[-1]]
    target_joint_type = probe.joint_mujoco_type_name(ctx.env, target_joint)
    domain = "fridge" if str(container.get("category", "")).lower() == "fridge" else "drawer"
    evidence_dir = args.output_dir / "rejected_evidence" / domain / case_id
    result = probe.container_visibility_trace(
        ctx,
        container,
        object_record["name"],
        joint_sequence,
        np.asarray(attempt["robot_pose"], dtype=float),
        view_profile=attempt["view_profile"],
        force_slide_joints=target_joint_type == "slide",
        output_dir=evidence_dir,
    )
    payload = {
        "case_id": case_id,
        "container_name": container["name"],
        "object_name": object_record["name"],
        "selected_failed_joint_index": int(target_joint["joint_index"]),
        "joint_sequence": joint_sequence,
        "joint_type": target_joint_type,
        "pose_meta": attempt.get("pose_meta"),
        "robot_pose": attempt["robot_pose"],
        "view_profile": attempt["view_profile"],
        "original_failure_reason": attempt.get("reason"),
        "visibility_trace": result["trace"],
    }
    ambiguous_evidence = []
    for selected in analysis.get("candidate_joint_results", []):
        candidate_joint_index = int(selected["joint"]["joint_index"])
        candidate_result = probe.container_visibility_trace(
            ctx,
            container,
            object_record["name"],
            [int(index) for index in selected["joint_sequence"]],
            np.asarray(selected["robot_pose"], dtype=float),
            view_profile=selected["view_profile"],
            force_slide_joints=selected.get("joint_type") == "slide",
            output_dir=evidence_dir / f"joint_{candidate_joint_index}",
        )
        ambiguous_evidence.append(
            {
                "joint_index": candidate_joint_index,
                "joint_sequence": selected["joint_sequence"],
                "view_profile": selected["view_profile"],
                "pose_meta": selected.get("pose_meta"),
                "visibility_trace": candidate_result["trace"],
            }
        )
    if ambiguous_evidence:
        payload["ambiguous_candidate_evidence"] = ambiguous_evidence
    probe.write_json(evidence_dir / "evidence.json", payload)
    return payload


def build_oracle_plan(
    container: dict[str, Any],
    selected: dict[str, Any],
    object_name: str,
    visibility_threshold: float,
    interaction_id_by_joint_index: dict[int, str] | None = None,
    plan_id: str = "oracle_0",
) -> dict[str, Any]:
    goal_point, goal_yaw = goal_from_robot_pose(selected["robot_pose"])
    joints_by_index = {int(joint["joint_index"]): joint for joint in container["joints"]}
    controlling_joint_index = int(
        selected.get("joint", {}).get("joint_index", selected["joint_sequence"][-1])
    )
    interaction_id_by_joint_index = interaction_id_by_joint_index or {
        int(index): v3.build_interaction_id(container["name"], int(index))
        for index in selected["joint_sequence"]
    }
    required_interaction_ids = [
        interaction_id_by_joint_index[int(index)] for index in selected["joint_sequence"]
    ]
    steps: list[dict[str, Any]] = [
        {
            "type": "navigate",
            "interaction_id": required_interaction_ids[0],
            "goal_point": goal_point,
            "goal_yaw": goal_yaw,
            "position_tolerance_m": 0.25,
            "yaw_tolerance_rad": 0.35,
            "reason": "approach_container_interaction",
        }
    ]
    if selected["view_profile"] != "default":
        steps.append(
            {
                "type": "set_view",
                "view_profile": selected["view_profile"],
                "head_qpos": selected["view_state"]["head_qpos"],
                "torso_qpos": selected["view_state"]["torso_qpos"],
                "reason": "improve_target_visibility",
            }
        )
    for sequence_index, joint_index in enumerate(selected["joint_sequence"]):
        joint = joints_by_index[joint_index]
        steps.append(
            {
                "type": "open_joint",
                "interaction_id": interaction_id_by_joint_index[joint_index],
                "object_name": container["name"],
                "joint_name": joint["joint_name"],
                "joint_index": joint_index,
                "target_fraction": 1.0,
                "control_mode": "force"
                if joint_index == controlling_joint_index
                and selected.get("joint_type") == "slide"
                else "direct",
                "reason": "reveal_target_object"
                if sequence_index == len(selected["joint_sequence"]) - 1
                else "prerequisite_for_interaction",
            }
        )
    steps.append(
        {
            "type": "observe_target",
            "object_name": object_name,
            "camera_name": "head_camera",
            "visibility_threshold": 0.0,
            "reason": "verify_target_visible",
        }
    )
    return {
        "plan_id": plan_id,
        "required_interaction_ids": required_interaction_ids,
        "steps": steps,
    }


def generated_episode(
    template_episode: dict[str, Any],
    source_episode: dict[str, Any],
    source_episode_index: int,
    case_id: str,
    container: dict[str, Any],
    object_record: dict[str, Any],
    selected: dict[str, Any],
    oracle_candidates: list[dict[str, Any]],
    scene_object_poses: dict[str, list[float]],
    articulation_states: list[dict[str, Any]],
    start_validation: dict[str, Any],
    binding: dict[str, Any],
    visibility_threshold: float,
    matching_instance_count: int,
    door_state_validation: dict[str, Any],
    container_state_validation: dict[str, Any],
) -> dict[str, Any]:
    episode = copy.deepcopy(template_episode)
    episode["task"]["robot_base_pose"] = copy.deepcopy(
        source_episode["task"]["robot_base_pose"]
    )
    episode["task"]["pickup_obj_name"] = object_record["name"]
    episode["task"]["pickup_obj_candidates"] = [object_record["name"]]
    episode["task"]["selection_mode"] = "specific_instance"
    episode["task"]["task_cls"] = "molmo_spaces.tasks.nav_task.NavToObjTask"
    episode["task"]["task_type"] = "nav_to_obj"
    episode["task"]["succ_pos_threshold"] = float(
        source_episode["task"].get("succ_pos_threshold", 1.5)
    )
    episode["task_relevant_objects"] = [container["name"], object_record["name"]]
    category = target_category(object_record)
    episode["language"] = {
        "task_description": f"find the {category}.",
        "instruction_type": "object_goal",
        "locale": "en",
        "interaction_disclosure": "hidden",
        "referral_expressions": {"object_name": category},
        "referral_expressions_priority": {},
    }
    episode["scene_modifications"] = {
        "added_objects": {},
        "object_poses": scene_object_poses,
        "removed_objects": [],
        "articulation_states": articulation_states,
    }
    if "left_arm" in episode["robot"]["init_qpos"]:
        episode["robot"]["init_qpos"]["left_arm"] = probe.DEFAULT_LEFT_ARM_QPOS.tolist()
    if "right_arm" in episode["robot"]["init_qpos"]:
        episode["robot"]["init_qpos"]["right_arm"] = probe.DEFAULT_RIGHT_ARM_QPOS.tolist()

    final_trace = selected["visibility_trace"][-1]
    goal_point, goal_yaw = goal_from_robot_pose(selected["robot_pose"])
    interactions, interaction_id_by_joint_index = v3.build_container_interactions(
        container=container,
        oracle_candidates=oracle_candidates,
        articulation_states=articulation_states,
    )
    oracle_plans = [
        build_oracle_plan(
            container,
            candidate,
            object_record["name"],
            0.0,
            interaction_id_by_joint_index,
            plan_id=f"oracle_{plan_index}",
        )
        for plan_index, candidate in enumerate(oracle_candidates)
    ]
    oracle_validations = []
    for candidate in oracle_candidates:
        candidate_trace = candidate["visibility_trace"]
        candidate_goal_point, candidate_goal_yaw = goal_from_robot_pose(candidate["robot_pose"])
        oracle_validations.append(
            {
                "controlling_joint_index": int(candidate["joint"]["joint_index"]),
                "controlling_joint_name": candidate["joint"]["joint_name"],
                "joint_sequence": candidate["joint_sequence"],
                "navigate_goal_point": candidate_goal_point,
                "navigate_goal_yaw": candidate_goal_yaw,
                "interaction_pose": pose_mat_to_7d(candidate["robot_pose"]).tolist(),
                "interaction_pose_meta": candidate["pose_meta"],
                "view_profile": candidate["view_profile"],
                "visibility_trace": candidate_trace,
                "reveal_mode": candidate.get(
                    "reveal_mode",
                    "force_slide_visibility"
                    if candidate.get("joint_type") == "slide"
                    else "hinge_pixel_reveal",
                ),
                "visual_observation_succeeded": candidate.get(
                    "visual_observation_succeeded", True
                ),
                "visual_validation_reason": candidate.get("visual_validation_reason"),
                "final_visibility_fraction": float(
                    candidate_trace[-1]["visibility_fraction"]
                ),
                "final_visible_pixels": int(candidate_trace[-1]["visible_pixels"]),
                "start_validation": candidate["start_validation"],
                "object_binding": candidate["binding"],
            }
        )
    target_position = np.asarray(
        final_trace.get("object_position", object_record["aabb_center"]), dtype=float
    )
    robot_position = np.asarray(selected["robot_pose"], dtype=float)[:3, 3]
    planar_distance_m = float(np.linalg.norm(robot_position[:2] - target_position[:2]))
    distance_threshold_m = float(episode["task"]["succ_pos_threshold"])
    distance_passed = planar_distance_m < distance_threshold_m
    visibility_fraction = float(final_trace["visibility_fraction"])
    visible_pixels = int(final_trace["visible_pixels"])
    visibility_passed = visibility_fraction > 0.0
    expected_task_success = distance_passed and visibility_passed
    if not expected_task_success:
        raise ValueError(
            "terminal_nav_to_obj_success_failed: "
            f"distance={planar_distance_m:.4f}/{distance_threshold_m:.4f}, "
            f"visibility={visibility_fraction:.8f}, pixels={visible_pixels}"
        )

    oracle_prefixes = []
    for plan, candidate in zip(oracle_plans, oracle_candidates, strict=True):
        oracle_prefixes.extend(
            v3.build_oracle_prefixes(
                plan=plan,
                visibility_trace=candidate["visibility_trace"],
                distance_passed=distance_passed,
                reachable=bool(candidate["start_validation"].get("path_found", False)),
            )
        )

    episode["interactive_nav"] = {
        "schema_version": "interactive_nav_v3",
        "case_id": case_id,
        "parent_benchmark_episode_index": source_episode_index,
        "interaction_domains": ["container"],
        "interaction_requirement": "required",
        "target": v3.build_container_target(
            object_record=object_record,
            category=category,
            container=container,
            matching_instance_count=matching_instance_count,
        ),
        "success_criteria": v3.build_nav_to_obj_success_criteria(
            distance_threshold_m
        ),
        "initial_state": v3.build_initial_state(
            interactions,
            all_doors_open=bool(door_state_validation["all_open"]),
            container_joints_closed=bool(container_state_validation["all_closed"]),
            target_visible=False,
        ),
        "interactions": interactions,
        "oracle_plan": oracle_plans[0],
        "oracle_plans": oracle_plans,
        "generation_validation": {
            "navigation_validation": {
                **start_validation,
                "navigate_goal_point": goal_point,
                "navigate_goal_yaw": goal_yaw,
                "interaction_pose": pose_mat_to_7d(selected["robot_pose"]).tolist(),
                "interaction_pose_meta": selected["pose_meta"],
                "interaction_pose_collision_free": True,
            },
            "interaction_validations": oracle_validations,
            "oracle_prefixes": oracle_prefixes,
            "compartment_evidence": binding if binding.get("applicable") else None,
            "success_evidence": {
                "status": "passed",
                "validation_mode": "simulated_terminal_state",
                "target_object_name": object_record["name"],
                "planar_distance_m": planar_distance_m,
                "distance_threshold_m": distance_threshold_m,
                "camera_name": "head_camera",
                "visibility_fraction": visibility_fraction,
                "visible_pixels": visible_pixels,
                "distance_passed": distance_passed,
                "visibility_passed": visibility_passed,
                "expected_task_success": expected_task_success,
            },
            "minimal_plan_verified": None,
            "minimal_plan_validation": {
                "status": "not_executed",
                "reason": "prerequisite_leave_one_out_not_run",
            },
            "door_state_validation": door_state_validation,
            "container_state_validation": container_state_validation,
            "generation_quality_visibility_threshold": visibility_threshold,
            "view_profile": selected["view_profile"],
            "reveal_mode": selected.get(
                "reveal_mode",
                "force_slide_visibility"
                if selected.get("joint_type") == "slide"
                else "hinge_pixel_reveal",
            ),
            "visual_observation_succeeded": selected.get(
                "visual_observation_succeeded", True
            ),
            "visual_validation_reason": selected.get("visual_validation_reason"),
            "controlling_joint_type": selected.get("joint_type"),
            "first_visible_after_joint_index": int(selected["joint"]["joint_index"]),
            "joint_assignment_ambiguous": len(oracle_candidates) > 1,
        },
    }
    for validation in oracle_validations:
        if validation.get("visual_validation_reason") is None:
            validation.pop("visual_validation_reason", None)
    if (
        episode["interactive_nav"]["generation_validation"].get(
            "visual_validation_reason"
        )
        is None
    ):
        episode["interactive_nav"]["generation_validation"].pop(
            "visual_validation_reason", None
        )
    return v3.validate_container_v3_episode(episode)


def safe_case_id(house_index: int, container_name: str, object_name: str) -> str:
    return (
        f"house_{house_index}__{probe.sanitize_name(container_name)}__"
        f"{probe.sanitize_name(object_name)}"
    )


def write_summary_markdown(
    path: Path,
    valid_pairs: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    house_catalog: list[dict[str, Any]],
) -> None:
    lines = [
        "# Container Interaction Benchmark Preview",
        "",
        f"- Valid pairs: {len(valid_pairs)}",
        f"- Rejected pairs: {len(rejected)}",
        "",
        "## Houses",
        "",
        "| House | Source episodes | Containers | Strict pairs | Valid | Rejected | Time (s) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for house in house_catalog:
        valid_count = int(house["valid_pair_count"])
        rejected_count = int(house["rejected_pair_count"])
        lines.append(
            f"| {house['house_index']} | `{house['source_episode_indices']}` | "
            f"{house['num_containers']} | {valid_count + rejected_count} | {valid_count} | "
            f"{rejected_count} | {house['elapsed_sec']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Valid Pairs",
            "",
            "| Episode | House | Container | Object | Oracles | GT joint sequences | View | Visibility trace | Navigate goal |",
            "|---:|---:|---|---|---:|---|---|---|---|",
        ]
    )
    for row in valid_pairs:
        validation = row["generation_validation"]
        target = row["target"]
        plans = row.get("oracle_plans") or [row["oracle_plan"]]
        joint_sequences = [
            [step["joint_index"] for step in plan["steps"] if step["type"] == "open_joint"]
            for plan in plans
        ]
        goal = validation["navigation_validation"]["navigate_goal_point"]
        visibility_trace = [
            round(float(trace_row["target_visibility_fraction"]), 7)
            for trace_row in validation["oracle_prefixes"]
            if trace_row["plan_id"] == plans[0]["plan_id"]
        ]
        lines.append(
            f"| {row['parent_benchmark_episode_index']} | {row['house_index']} | "
            f"`{target['container_name']}` | `{target['selected_instance']}` | {len(plans)} | "
            f"`{joint_sequences}` | "
            f"{validation['view_profile']} | `{visibility_trace}` | "
            f"({goal[0]:.2f}, {goal[1]:.2f}) |"
        )
    reason_counts = Counter(row["reason"] for row in rejected)
    lines.extend(["", "## Rejection Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{reason}` | {count} |")
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    episodes = load_benchmark_episodes(args.benchmark_dir)
    if args.candidate_file is not None and args.rough_catalog is not None:
        raise ValueError("Use either --candidate_file or --rough_catalog, not both")
    collection_plan = None
    if args.rough_catalog is not None:
        explicit_houses = (
            [int(value) for value in args.house_indices.split(",")]
            if args.house_indices
            else None
        )
        collection_plan = build_dynamic_collection_plan(
            args.rough_catalog,
            max_samples=args.max_samples,
            samples_per_house=args.samples_per_house,
            target_house_count=(
                args.target_house_count
                if args.target_house_count is not None
                else args.max_houses
            ),
            house_indices=explicit_houses,
            seed=args.seed,
        )
        candidate_manifest = {
            int(row["house_index"]): row for row in collection_plan["houses"]
        }
    else:
        candidate_manifest = load_candidate_manifest(args.candidate_file)
    requested_slot_count = sum(
        int(row.get("target_sample_count", len(row.get("slots", []))))
        for row in candidate_manifest.values()
    )
    fixed_collection_mode = collection_plan is not None or any(
        "target_sample_count" in row for row in candidate_manifest.values()
    )
    completed_candidate_slots: set[tuple[int, str]] = set()
    indexed = list(enumerate(episodes[args.start_idx :], start=args.start_idx))
    if candidate_manifest:
        requested_houses = set(candidate_manifest)
        selected = [
            row for row in indexed if int(row[1]["house_index"]) in requested_houses
        ]
    elif args.house_indices:
        requested_houses = {int(value) for value in args.house_indices.split(",")}
        selected = [
            row for row in indexed if int(row[1]["house_index"]) in requested_houses
        ]
    elif args.max_houses is not None:
        selected_house_ids: list[int] = []
        selected_house_set: set[int] = set()
        for _, episode in indexed:
            house_index = int(episode["house_index"])
            if house_index not in selected_house_set:
                selected_house_ids.append(house_index)
                selected_house_set.add(house_index)
                if len(selected_house_ids) >= args.max_houses:
                    break
        selected = [
            row for row in indexed if int(row[1]["house_index"]) in selected_house_set
        ]
    else:
        selected = indexed[: args.max_episodes]
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for episode_index, episode in selected:
        grouped.setdefault(int(episode["house_index"]), []).append((episode_index, episode))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if collection_plan is not None:
        probe.write_json(args.output_dir / "collection_plan.json", collection_plan)
    benchmark_episodes: list[dict[str, Any]] = []
    valid_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    fridge_slide_candidates: list[dict[str, Any]] = []
    house_catalog: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    complete_collection_house_count = 0

    for house_index, indexed_episodes in grouped.items():
        started_at = time.perf_counter()
        should_stop = False
        template_episode_index, template_episode = indexed_episodes[0]
        log.info("Scanning house %s from template episode %s", house_index, template_episode_index)
        ctx = None
        try:
            ctx = load_episode_context(args, template_episode)
            door_transitions = [] if args.catalog_only else open_all_available_doors(ctx)
            initial_records, initial_containers = probe.collect_scene_records(ctx)
            close_all_containers(ctx.env, initial_containers)
            door_state_validation = (
                {"all_open": None, "door_count": 0, "doors": []}
                if args.catalog_only
                else validate_all_doors_open(ctx)
            )
            if not args.catalog_only and not door_state_validation["all_open"]:
                failed_doors = [
                    row["joint_name"]
                    for row in door_state_validation["doors"]
                    if not row["passed"]
                ]
                raise ValueError(f"not_all_channel_doors_open: {failed_doors}")
            records, containers = probe.collect_scene_records(ctx)
            container_state_validation = (
                {"all_closed": None, "joint_count": 0, "joints": []}
                if args.catalog_only
                else validate_all_containers_closed(ctx, containers)
            )
            if not args.catalog_only and not container_state_validation["all_closed"]:
                failed_joints = [
                    row["joint_name"]
                    for row in container_state_validation["joints"]
                    if not row["passed"]
                ]
                raise ValueError(f"not_all_container_joints_closed: {failed_joints}")
            scene_object_poses = {
                record["name"]: (
                    np.asarray(record["position"], dtype=float).tolist()
                    + np.asarray(record["quat"], dtype=float).tolist()
                )
                for record in records
                if record.get("has_free_joint")
            }
            articulation_states = (
                [] if args.catalog_only else articulation_initial_states(ctx, containers)
            )
            scene_map = None if args.catalog_only else build_scene_map(args, ctx)
            target_records = [record for record in records if probe.is_target_like(record)]
            target_category_counts = Counter(
                target_category(record) for record in target_records
            )
            contained_by_container = {
                container["name"]: [
                    object_record
                    for object_record in target_records
                    if probe.compute_relation(container, object_record)["inside_aabb"]
                ]
                for container in containers
            }
            strict_pair_count = sum(map(len, contained_by_container.values()))
            log.info(
                "House %s has %s containers, %s target-like objects, and %s strict inside pairs",
                house_index,
                len(containers),
                len(target_records),
                strict_pair_count,
            )
            dependencies_by_container = (
                {
                    container["name"]: probe.infer_joint_open_dependencies(
                        ctx.env,
                        container,
                        method="front_occlusion",
                    )
                    for container in containers
                }
                if not args.catalog_only
                else {container["name"]: [] for container in containers}
            )
            for container in containers:
                if str(container.get("category", "")).lower() != "fridge":
                    continue
                dependencies_by_index = {
                    int(row["joint_index"]): row
                    for row in dependencies_by_container[container["name"]]
                }
                for joint in container["joints"]:
                    if probe.joint_mujoco_type_name(ctx.env, joint) != "slide":
                        continue
                    contained = probe.drawer_joint_contained_objects(
                        ctx,
                        container,
                        joint,
                        target_records,
                        padding=args.drawer_box_padding,
                    )
                    dependency = dependencies_by_index.get(int(joint["joint_index"]), {})
                    fridge_slide_candidates.append(
                        {
                            "house_index": house_index,
                            "container_name": container["name"],
                            "joint_index": int(joint["joint_index"]),
                            "joint_name": joint["joint_name"],
                            "prerequisite_joint_indices": dependency.get(
                                "prerequisite_joint_indices", []
                            ),
                            "front_axis_xy": dependency.get("front_axis_xy"),
                            "contained_objects": [
                                {
                                    "name": record["name"],
                                    "category": target_category(record),
                                    "aabb_center": np.asarray(
                                        record["aabb_center"], dtype=float
                                    ).tolist(),
                                    "aabb_size": np.asarray(
                                        record["aabb_size"], dtype=float
                                    ).tolist(),
                                }
                                for record in contained
                            ],
                            "multi_interaction_candidate": bool(
                                contained
                                and dependency.get("prerequisite_joint_indices", [])
                            ),
                        }
                    )
            house_valid_before = len(valid_pairs)
            house_rejected_before = len(rejected_pairs)
            house_benchmark_before = len(benchmark_episodes)
            pair_index = 0
            containers_by_name = {container["name"]: container for container in containers}
            objects_by_name = {record["name"]: record for record in target_records}
            house_manifest = candidate_manifest.get(house_index)
            dynamic_house_plan = bool(
                house_manifest and "target_sample_count" in house_manifest
            )
            requested_house_samples = (
                int(house_manifest["target_sample_count"])
                if dynamic_house_plan
                else len(house_manifest.get("slots", [])) if house_manifest else 0
            )
            if args.catalog_only:
                pair_inputs = []
            elif dynamic_house_plan:
                pair_inputs = []
                for candidate in house_manifest.get("candidates", []):
                    container = containers_by_name.get(candidate["container_name"])
                    object_record = objects_by_name.get(candidate["object_name"])
                    if container is None or object_record is None:
                        continue
                    pair_inputs.append((None, candidate, container, object_record))
            elif house_manifest:
                pair_inputs = []
                for slot in house_manifest.get("slots", []):
                    for candidate in slot.get("candidates", []):
                        container = containers_by_name.get(candidate["container_name"])
                        object_record = objects_by_name.get(candidate["object_name"])
                        if container is None or object_record is None:
                            continue
                        pair_inputs.append(
                            (str(slot["slot_id"]), candidate, container, object_record)
                        )
            else:
                pair_inputs = [
                    (None, None, container, object_record)
                    for container in containers
                    for object_record in target_records
                ]
            completed_slots: set[str] = set()
            for slot_id, manifest_candidate, container, object_record in pair_inputs:
                    if (
                        dynamic_house_plan
                        and len(valid_pairs) - house_valid_before >= requested_house_samples
                    ):
                        break
                    if slot_id is not None and slot_id in completed_slots:
                        continue
                    dependency_rows = dependencies_by_container[container["name"]]
                    relation = probe.compute_relation(container, object_record)
                    if not relation["inside_aabb"]:
                        continue
                    case_id = safe_case_id(house_index, container["name"], object_record["name"])
                    analysis = analyze_object_pair(
                        args,
                        ctx,
                        container,
                        object_record,
                        dependency_rows,
                        case_id,
                    )
                    if not analysis["valid"]:
                        diagnostic_paths = {}
                        if args.save_plots and str(container.get("category", "")).lower() == "fridge":
                            plot_path = (
                                args.output_dir
                                / "diagnostics"
                                / "rejected_fridge_joint_boxes"
                                / f"{case_id}.png"
                            )
                            probe.save_joint_dependency_plot(
                                plot_path,
                                container,
                                dependency_rows,
                                object_record,
                            )
                            diagnostic_paths["joint_object_box_plot"] = str(plot_path)
                        rejected_evidence = None
                        if args.save_images:
                            rejected_evidence = save_best_rejected_evidence(
                                args,
                                ctx,
                                container,
                                object_record,
                                case_id,
                                analysis,
                            )
                        rejected_pairs.append(
                            {
                                "case_id": case_id,
                                "house_index": house_index,
                                "container_name": container["name"],
                                "object_name": object_record["name"],
                                "reason": analysis["reason"],
                                "relation": relation,
                                "diagnostics": analysis,
                                "diagnostic_paths": diagnostic_paths,
                                "rejected_evidence": rejected_evidence,
                            }
                        )
                        continue
                    interaction_candidates = analysis["candidate_joint_results"]
                    selected_interaction = analysis["selected"]
                    selected_start = None
                    selected_candidates: list[dict[str, Any]] = []
                    source_episode_index = None
                    source_episode = None
                    source_options = indexed_episodes
                    if manifest_candidate:
                        preferred = {
                            int(index): rank
                            for rank, index in enumerate(
                                manifest_candidate.get(
                                    "preferred_source_episode_indices", []
                                )
                            )
                        }
                        source_options = sorted(
                            indexed_episodes,
                            key=lambda row: preferred.get(row[0], len(preferred)),
                        )
                    elif indexed_episodes:
                        offset = pair_index % len(indexed_episodes)
                        source_options = (
                            indexed_episodes[offset:] + indexed_episodes[:offset]
                        )
                    for candidate_episode_index, candidate_episode in source_options:
                        reachable_candidates = []
                        for interaction_candidate in interaction_candidates:
                            start_validation = source_start_validation(
                                ctx,
                                container,
                                object_record["name"],
                                candidate_episode,
                                scene_map,
                                interaction_candidate["robot_pose"],
                                args.visibility_threshold,
                            )
                            if start_validation["valid"]:
                                reachable_candidates.append(
                                    {
                                        **interaction_candidate,
                                        "start_validation": start_validation,
                                    }
                                )
                        if reachable_candidates:
                            selected_candidates = reachable_candidates
                            selected_interaction = reachable_candidates[0]
                            selected_start = reachable_candidates[0]["start_validation"]
                            source_episode_index = candidate_episode_index
                            source_episode = candidate_episode
                            break
                    pair_index += 1
                    if selected_start is None or source_episode is None or source_episode_index is None:
                        rejected_pairs.append(
                            {
                                "case_id": case_id,
                                "house_index": house_index,
                                "container_name": container["name"],
                                "object_name": object_record["name"],
                                "reason": "no_valid_source_start",
                                "relation": relation,
                            }
                        )
                        continue

                    evidence_dir = args.output_dir / "evidence" / case_id
                    rendered_candidates = []
                    for candidate in selected_candidates:
                        candidate_joint_index = int(candidate["joint"]["joint_index"])
                        trace_with_images = probe.container_visibility_trace(
                            ctx,
                            container,
                            object_record["name"],
                            candidate["joint_sequence"],
                            candidate["robot_pose"],
                            view_profile=candidate["view_profile"],
                            force_slide_joints=candidate.get("force_slide_joints", False),
                            output_dir=(
                                evidence_dir / f"oracle_joint_{candidate_joint_index}"
                                if args.save_images
                                else None
                            ),
                        )
                        rendered_candidates.append(
                            {**candidate, "visibility_trace": trace_with_images["trace"]}
                        )
                    selected_candidates = rendered_candidates
                    selected_interaction = selected_candidates[0]
                    try:
                        episode = generated_episode(
                            template_episode,
                            source_episode,
                            source_episode_index,
                            case_id,
                            container,
                            object_record,
                            selected_interaction,
                            selected_candidates,
                            scene_object_poses,
                            articulation_states,
                            selected_start,
                            selected_interaction["binding"],
                            args.visibility_threshold,
                            target_category_counts[target_category(object_record)],
                            door_state_validation,
                            container_state_validation,
                        )
                    except ValueError as exc:
                        rejected_pairs.append(
                            {
                                "case_id": case_id,
                                "house_index": house_index,
                                "container_name": container["name"],
                                "object_name": object_record["name"],
                                "reason": "v3_episode_validation_failed",
                                "error": str(exc),
                                "relation": relation,
                            }
                        )
                        continue
                    if manifest_candidate is not None:
                        selection_slot_id = slot_id
                        if dynamic_house_plan:
                            selection_slot_id = (
                                f"house_{house_index}_sample_"
                                f"{len(valid_pairs) - house_valid_before}"
                            )
                        episode["interactive_nav"]["generation_validation"][
                            "candidate_selection"
                        ] = {
                            "slot_id": selection_slot_id,
                            "candidate_rank": manifest_candidate.get("candidate_rank", 0),
                            "preferred_source_episode_indices": manifest_candidate.get(
                                "preferred_source_episode_indices", []
                            ),
                            "selected_start_distance_m": manifest_candidate.get(
                                "selected_start_distance_m"
                            ),
                            "selected_start_distance_bin": manifest_candidate.get(
                                "selected_start_distance_bin"
                            ),
                        }
                    benchmark_episodes.append(episode)
                    interactive_nav = episode["interactive_nav"]
                    valid_pairs.append(
                        {
                            "case_id": case_id,
                            "house_index": house_index,
                            **interactive_nav,
                        }
                    )
                    if slot_id is not None:
                        completed_slots.add(slot_id)
                        completed_candidate_slots.add((house_index, slot_id))
                    elif dynamic_house_plan:
                        completed_candidate_slots.add(
                            (
                                house_index,
                                f"house_{house_index}_sample_"
                                f"{len(valid_pairs) - house_valid_before - 1}",
                            )
                        )
            completed_house_samples = (
                len(valid_pairs) - house_valid_before
                if dynamic_house_plan
                else len(completed_slots)
            )
            requested_house_slots = requested_house_samples
            collection_house_complete = (
                not house_manifest or completed_house_samples >= requested_house_slots
            )
            if (
                house_manifest
                and not dynamic_house_plan
                and args.require_complete_house
                and not collection_house_complete
            ):
                del benchmark_episodes[house_benchmark_before:]
                del valid_pairs[house_valid_before:]
                completed_candidate_slots.difference_update(
                    (house_index, slot_id) for slot_id in completed_slots
                )
                rejected_pairs.append(
                    {
                        "case_id": f"house_{house_index}__incomplete_house_quota",
                        "house_index": house_index,
                        "reason": "incomplete_house_quota",
                        "requested_slot_count": requested_house_slots,
                        "completed_slot_count": completed_house_samples,
                    }
                )
                completed_slots.clear()
                collection_house_complete = False
            elif house_manifest and collection_house_complete:
                complete_collection_house_count += 1

            probe.write_json(args.output_dir / "benchmark.partial.json", benchmark_episodes)
            probe.write_json(args.output_dir / "valid_pairs.partial.json", valid_pairs)
            probe.write_json(args.output_dir / "rejected_pairs.partial.json", rejected_pairs)

            house_catalog.append(
                {
                    "house_index": house_index,
                    "template_episode_index": template_episode_index,
                    "source_episode_indices": [index for index, _ in indexed_episodes],
                    "num_objects": len(records),
                    "num_target_objects": len(target_records),
                    "num_containers": len(containers),
                    "strict_pair_count": strict_pair_count,
                    "requested_sample_count": requested_house_slots,
                    "collected_sample_count": len(valid_pairs) - house_valid_before,
                    "collection_quota_complete": collection_house_complete,
                    "num_fridges": sum(
                        str(container.get("category", "")).lower() == "fridge"
                        for container in containers
                    ),
                    "num_dressers": sum(
                        str(container.get("category", "")).lower() == "dresser"
                        for container in containers
                    ),
                    "num_fridges_with_objects": sum(
                        str(container.get("category", "")).lower() == "fridge"
                        and bool(contained_by_container[container["name"]])
                        for container in containers
                    ),
                    "num_dressers_with_objects": sum(
                        str(container.get("category", "")).lower() == "dresser"
                        and bool(contained_by_container[container["name"]])
                        for container in containers
                    ),
                    "door_open_transition_count": len(door_transitions),
                    "valid_pair_count": len(valid_pairs) - house_valid_before,
                    "rejected_pair_count": len(rejected_pairs) - house_rejected_before,
                    "requested_collection_slots": requested_house_slots,
                    "completed_collection_slots": len(completed_slots),
                    "collection_house_complete": collection_house_complete,
                    "containers": [
                        {
                            "name": container["name"],
                            "category": container.get("category"),
                            "asset_id": container.get("asset_id"),
                            "aabb_center": np.asarray(
                                container["aabb_center"], dtype=float
                            ).tolist(),
                            "aabb_size": np.asarray(
                                container["aabb_size"], dtype=float
                            ).tolist(),
                            "joints": container["joints"],
                            "dependencies": dependencies_by_container[container["name"]],
                            "strict_contained_objects": [
                                rough_target_record(record, container, indexed_episodes)
                                for record in contained_by_container[container["name"]]
                            ],
                        }
                        for container in containers
                    ],
                    "elapsed_sec": time.perf_counter() - started_at,
                }
            )
            should_stop = bool(
                args.stop_after_complete_houses is not None
                and complete_collection_house_count >= args.stop_after_complete_houses
            )
        except Exception as exc:
            log.exception("House %s failed", house_index)
            failures.append(
                {
                    "house_index": house_index,
                    "template_episode_index": template_episode_index,
                    "error": str(exc),
                }
            )
        finally:
            if ctx is not None:
                probe.close_context(ctx)
        probe.write_json(args.output_dir / "house_catalog.partial.json", house_catalog)
        probe.write_json(
            args.output_dir / "fridge_slide_compartment_candidates.partial.json",
            fridge_slide_candidates,
        )
        probe.write_json(args.output_dir / "failures.partial.json", failures)
        if should_stop:
            log.info(
                "Reached %s complete collection houses; stopping early",
                complete_collection_house_count,
            )
            break

    summary = {
        "schema_version": "container_interaction_benchmark_summary_v1",
        "benchmark_source": str(args.benchmark_dir),
        "start_idx": args.start_idx,
        "max_episodes": args.max_episodes,
        "max_houses": args.max_houses,
        "house_indices": args.house_indices,
        "candidate_file": None if args.candidate_file is None else str(args.candidate_file),
        "rough_catalog": None if args.rough_catalog is None else str(args.rough_catalog),
        "collection_mode": "fixed" if fixed_collection_mode else "legacy",
        "samples_per_house": args.samples_per_house,
        "target_house_count": args.target_house_count,
        "require_complete_house": args.require_complete_house,
        "stop_after_complete_houses": args.stop_after_complete_houses,
        "catalog_only": args.catalog_only,
        "selected_episode_indices": [index for index, _ in selected],
        "selected_house_indices": list(grouped),
        "valid_pair_count": len(valid_pairs),
        "rejected_pair_count": len(rejected_pairs),
        "generated_episode_count": len(benchmark_episodes),
        "requested_candidate_slot_count": requested_slot_count,
        "completed_candidate_slot_count": len(completed_candidate_slots),
        "complete_collection_house_count": complete_collection_house_count,
        "collection_house_count": len(
            {int(row["house_index"]) for row in valid_pairs}
        ),
        "partial_collection_house_count": sum(
            0 < int(row.get("collected_sample_count", 0))
            < int(row.get("requested_sample_count", 0))
            for row in house_catalog
        ),
        "zero_sample_house_count": sum(
            int(row.get("requested_sample_count", 0)) > 0
            and int(row.get("collected_sample_count", 0)) == 0
            for row in house_catalog
        ),
        "multi_oracle_episode_count": sum(
            len(episode["interactive_nav"].get("oracle_plans", [])) > 1
            for episode in benchmark_episodes
        ),
        "fridge_slide_joint_count": len(fridge_slide_candidates),
        "fridge_slide_object_count": sum(
            len(row["contained_objects"]) for row in fridge_slide_candidates
        ),
        "fridge_slide_joints_with_objects": sum(
            bool(row["contained_objects"]) for row in fridge_slide_candidates
        ),
        "total_fridges": sum(row["num_fridges"] for row in house_catalog),
        "total_dressers": sum(row["num_dressers"] for row in house_catalog),
        "fridges_with_strict_objects": sum(
            row["num_fridges_with_objects"] for row in house_catalog
        ),
        "dressers_with_strict_objects": sum(
            row["num_dressers_with_objects"] for row in house_catalog
        ),
        "strict_pair_count": sum(row["strict_pair_count"] for row in house_catalog),
        "fridge_multi_interaction_candidate_count": sum(
            int(row["multi_interaction_candidate"])
            for row in fridge_slide_candidates
        ),
        "failure_count": len(failures),
        "rejection_reason_counts": dict(Counter(row["reason"] for row in rejected_pairs)),
    }
    probe.write_json(args.output_dir / "benchmark.json", benchmark_episodes)
    probe.write_json(args.output_dir / "house_catalog.json", house_catalog)
    probe.write_json(args.output_dir / "valid_pairs.json", valid_pairs)
    probe.write_json(args.output_dir / "rejected_pairs.json", rejected_pairs)
    probe.write_json(
        args.output_dir / "fridge_slide_compartment_candidates.json",
        fridge_slide_candidates,
    )
    probe.write_json(args.output_dir / "failures.json", failures)
    probe.write_json(args.output_dir / "summary.json", summary)
    write_summary_markdown(
        args.output_dir / "summary.md",
        valid_pairs,
        rejected_pairs,
        house_catalog,
    )
    log.info("Wrote container interaction benchmark preview to %s", args.output_dir)
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a GT container-interaction nav benchmark.")
    parser.add_argument("--benchmark_dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--max_houses", type=int)
    parser.add_argument(
        "--house_indices",
        help="Comma-separated house indices; overrides --max_houses.",
    )
    parser.add_argument(
        "--candidate_file",
        type=Path,
        help="Fine candidate manifest with per-house slots and redundant candidates.",
    )
    parser.add_argument(
        "--rough_catalog",
        type=Path,
        help="Rough container-object catalog; builds the fixed plan in-process.",
    )
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--samples_per_house", type=int, default=2)
    parser.add_argument("--target_house_count", type=int)
    parser.add_argument(
        "--require_complete_house",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Commit a candidate house only when all requested slots succeed.",
    )
    parser.add_argument(
        "--stop_after_complete_houses",
        type=int,
        help="Stop after this many candidate houses satisfy all requested slots.",
    )
    parser.add_argument(
        "--catalog_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--robot", default="rby1", choices=["rby1", "droid", "rum"])
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility_threshold", type=float, default=1e-4)
    parser.add_argument("--interaction_distance", type=float, default=0.8)
    parser.add_argument("--max_poses_per_joint", type=int, default=4)
    parser.add_argument("--drawer_box_padding", type=float, default=0.05)
    parser.add_argument("--px_per_m", type=float, default=100.0)
    parser.add_argument("--open_threshold", type=float, default=0.67)
    parser.add_argument(
        "--save_images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
