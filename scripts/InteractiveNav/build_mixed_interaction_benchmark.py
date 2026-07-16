from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.utils.pose import pose_mat_to_7d
from scripts.InteractiveNav import benchmark_door_state_scan as door_scan
from scripts.InteractiveNav import build_container_interaction_benchmark as container_builder
from scripts.InteractiveNav import build_door_interaction_benchmark as door_builder
from scripts.InteractiveNav import collect_mixed_rough_catalog as mixed_rough
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav import interactive_nav_v3 as v3


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_ROUGH_CATALOG = mixed_rough.DEFAULT_OUTPUT_DIR / "mixed_rough_catalog.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts/InteractiveNav/output/mixed_interaction_v3_smoke10"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")
    temporary.chmod(0o644)
    temporary.replace(path)


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return slug[:max_len]


def rejection_reason(exc: Exception) -> str:
    prefix = str(exc).split(":", 1)[0]
    reason = safe_slug(prefix, 120).replace("-", "_")
    return reason or type(exc).__name__.lower()


def channel_interaction_id(case_id: str, root_name: str, leaf: dict[str, Any]) -> str:
    return (
        f"channel::{safe_slug(case_id, 40)}::{safe_slug(root_name, 40)}::"
        f"{safe_slug(leaf['object_name'], 40)}::{int(leaf['joint_index'])}"
    )


def flatten_single_leaf_transition(
    root_transition: dict[str, Any],
) -> dict[str, Any]:
    leaves = door_builder.flatten_root_transitions([root_transition])
    if len(leaves) != 1:
        raise ValueError(
            "Mixed fine collection currently requires a single-leaf door root; "
            f"got {len(leaves)} leaves for {root_transition.get('door_root_name')}"
        )
    return leaves[0]


def build_channel_interaction(
    *,
    case_id: str,
    root_name: str,
    closed_leaf: dict[str, Any],
    opened_leaf: dict[str, Any],
    interaction_requirement: str = "required",
) -> dict[str, Any]:
    if (
        closed_leaf["object_name"] != opened_leaf["object_name"]
        or closed_leaf["joint_name"] != opened_leaf["joint_name"]
        or int(closed_leaf["joint_index"]) != int(opened_leaf["joint_index"])
    ):
        raise ValueError("Closed/open door readbacks refer to different leaf joints")
    if interaction_requirement not in {"required", "beneficial"}:
        raise ValueError(
            f"Unsupported mixed channel requirement: {interaction_requirement}"
        )
    if float(closed_leaf["open_fraction"]) > 0.01:
        raise ValueError("Mixed channel leaf is not measured closed")
    if float(opened_leaf["open_fraction"]) < 0.99:
        raise ValueError("Mixed channel leaf is not measured open after oracle transition")
    return {
        "interaction_id": channel_interaction_id(case_id, root_name, closed_leaf),
        "type": "channel_hinged_door",
        "object_name": closed_leaf["object_name"],
        "object_category": "Door",
        "joint_name": closed_leaf["joint_name"],
        "joint_index": int(closed_leaf["joint_index"]),
        "effect_types": [
            "restore_reachability"
            if interaction_requirement == "required"
            else "reduce_navigation_cost"
        ],
        "prerequisites": [],
        "initial_state": {
            "joint_fraction": float(closed_leaf["open_fraction"]),
            "semantic_state": "closed",
        },
        "target_state": {
            "joint_fraction": float(opened_leaf["open_fraction"]),
            "semantic_state": "open",
        },
        "door_root_name": root_name,
        "initial_joint_position": float(closed_leaf["joint_position"]),
        "target_joint_position": float(opened_leaf["joint_position"]),
    }


def attach_channel_prerequisite(
    container_interactions: list[dict[str, Any]],
    channel_interaction_id_value: str,
) -> None:
    if not container_interactions:
        raise ValueError("Mixed episode has no container interaction")
    first = container_interactions[0]
    first.setdefault("prerequisites", []).insert(
        0,
        {
            "interaction_id": channel_interaction_id_value,
            "type": "reachability",
        },
    )


def build_mixed_oracle_plan(
    *,
    channel_interaction: dict[str, Any],
    container: dict[str, Any],
    selected: dict[str, Any],
    container_interaction_ids: dict[int, str],
    target_object_name: str,
    approach_xy: np.ndarray,
    door_center_xy: np.ndarray,
    interaction_requirement: str = "required",
) -> dict[str, Any]:
    channel_id = channel_interaction["interaction_id"]
    goal_point, goal_yaw = container_builder.goal_from_robot_pose(selected["robot_pose"])
    steps: list[dict[str, Any]] = [
        {
            "type": "navigate",
            "interaction_id": channel_id,
            "goal_point": [float(approach_xy[0]), float(approach_xy[1]), 0.0],
            "goal_yaw": door_builder.yaw_towards(approach_xy, door_center_xy),
            "position_tolerance_m": 0.25,
            "yaw_tolerance_rad": 0.35,
            "reason": "approach_channel_interaction",
        },
        {
            "type": "open_joint",
            "interaction_id": channel_id,
            "object_name": channel_interaction["object_name"],
            "joint_name": channel_interaction["joint_name"],
            "joint_index": channel_interaction["joint_index"],
            "target_fraction": channel_interaction["target_state"]["joint_fraction"],
            "control_mode": "direct",
            "reason": (
                "restore_reachability"
                if interaction_requirement == "required"
                else "reduce_navigation_cost"
            ),
        },
        {
            "type": "navigate",
            "interaction_id": container_interaction_ids[int(selected["joint_sequence"][0])],
            "goal_point": goal_point,
            "goal_yaw": goal_yaw,
            "position_tolerance_m": 0.25,
            "yaw_tolerance_rad": 0.35,
            "reason": "approach_container_interaction",
        },
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
    joints_by_index = {int(row["joint_index"]): row for row in container["joints"]}
    controlling_joint_index = int(selected["joint"]["joint_index"])
    for sequence_index, joint_index in enumerate(selected["joint_sequence"]):
        joint = joints_by_index[int(joint_index)]
        steps.append(
            {
                "type": "open_joint",
                "interaction_id": container_interaction_ids[int(joint_index)],
                "object_name": container["name"],
                "joint_name": joint["joint_name"],
                "joint_index": int(joint_index),
                "target_fraction": 1.0,
                "control_mode": (
                    "force"
                    if int(joint_index) == controlling_joint_index
                    and selected.get("joint_type") == "slide"
                    else "direct"
                ),
                "reason": (
                    "reveal_target_object"
                    if sequence_index == len(selected["joint_sequence"]) - 1
                    else "prerequisite_for_interaction"
                ),
            }
        )
    steps.append(
        {
            "type": "observe_target",
            "object_name": target_object_name,
            "camera_name": "head_camera",
            "visibility_threshold": 0.0,
            "reason": "verify_target_visible",
        }
    )
    required_ids = [channel_id] + [
        container_interaction_ids[int(index)] for index in selected["joint_sequence"]
    ]
    return {"plan_id": "oracle_0", "required_interaction_ids": required_ids, "steps": steps}


def build_mixed_prefixes(
    *,
    plan: dict[str, Any],
    channel_interaction_id_value: str,
    selected: dict[str, Any],
    container_interaction_ids: dict[int, str],
    distance_passed: bool,
    interaction_requirement: str = "required",
) -> list[dict[str, Any]]:
    door_open_step_index = next(
        index
        for index, step in enumerate(plan["steps"])
        if step.get("type") == "open_joint"
        and step.get("interaction_id") == channel_interaction_id_value
    )
    container_open_indices = [
        index
        for index, step in enumerate(plan["steps"])
        if step.get("type") == "open_joint"
        and step.get("interaction_id") != channel_interaction_id_value
    ]
    prefixes = [
        {
            "plan_id": plan["plan_id"],
            "completed_step_count": 0,
            "robot_reachable_to_next_goal": True,
            "target_distance_passed": False,
            "target_visibility_fraction": 0.0,
            "target_visible_pixels": 0,
            "task_success": False,
            "opened_interaction_ids": [],
            "state_label": (
                "initial_required_door_and_container_closed"
                if interaction_requirement == "required"
                else "initial_beneficial_door_and_container_closed"
            ),
        },
        {
            "plan_id": plan["plan_id"],
            "completed_step_count": door_open_step_index + 1,
            "robot_reachable_to_next_goal": True,
            "target_distance_passed": False,
            "target_visibility_fraction": 0.0,
            "target_visible_pixels": 0,
            "task_success": False,
            "opened_interaction_ids": [channel_interaction_id_value],
            "state_label": (
                "required_door_open_container_closed"
                if interaction_requirement == "required"
                else "beneficial_door_open_container_closed"
            ),
        },
    ]
    opened = [channel_interaction_id_value]
    trace = selected["visibility_trace"]
    for sequence_index, joint_index in enumerate(selected["joint_sequence"]):
        interaction_id = container_interaction_ids[int(joint_index)]
        opened.append(interaction_id)
        row = trace[sequence_index + 1]
        visibility = float(row["visibility_fraction"])
        pixels = int(row["visible_pixels"])
        prefixes.append(
            {
                "plan_id": plan["plan_id"],
                "completed_step_count": container_open_indices[sequence_index] + 1,
                "robot_reachable_to_next_goal": True,
                "target_distance_passed": bool(distance_passed),
                "target_visibility_fraction": visibility,
                "target_visible_pixels": pixels,
                "task_success": bool(distance_passed and visibility > 0.0),
                "opened_interaction_ids": list(opened),
                "state_label": f"container_prefix_{sequence_index + 1}",
            }
        )
    return prefixes


def build_minimal_plan_validation(
    *,
    channel_interaction: dict[str, Any],
    container_interactions: list[dict[str, Any]],
    selected: dict[str, Any],
    initial_path_found: bool,
    interaction_requirement: str = "required",
    path_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    channel_required = interaction_requirement == "required"
    channel_beneficial = bool(
        interaction_requirement == "beneficial"
        and path_evidence is not None
        and path_evidence.get("mixed_shortcut_verified") is True
    )
    omission_results = [
        {
            "omitted_interaction_id": channel_interaction["interaction_id"],
            "interaction_domain": "channel",
            "required": channel_required and initial_path_found is False,
            "beneficial": channel_beneficial,
            "measured_path_found": bool(initial_path_found),
            "measured_path_length_delta_m": (
                None
                if path_evidence is None
                else path_evidence.get("path_length_delta_m")
            ),
            "failure_mode": (
                "container_interaction_pose_unreachable"
                if channel_required
                else "longer_alternate_navigation_path"
            ),
        }
    ]
    interaction_by_joint = {
        int(row["joint_index"]): row for row in container_interactions
    }
    for sequence_index, joint_index in enumerate(selected["joint_sequence"]):
        trace_before = selected["visibility_trace"][sequence_index]
        visibility = float(trace_before["visibility_fraction"])
        pixels = int(trace_before["visible_pixels"])
        interaction = interaction_by_joint[int(joint_index)]
        omission_results.append(
            {
                "omitted_interaction_id": interaction["interaction_id"],
                "interaction_domain": "container",
                "required": visibility == 0.0 and pixels == 0,
                "measured_visibility_fraction_before_interaction": visibility,
                "measured_visible_pixels_before_interaction": pixels,
                "failure_mode": "target_remains_hidden",
            }
        )
    container_rows = [
        row for row in omission_results if row["interaction_domain"] == "container"
    ]
    passed = bool(
        all(row["required"] is True for row in container_rows)
        and (
            (channel_required and omission_results[0]["required"] is True)
            or (not channel_required and omission_results[0]["beneficial"] is True)
        )
    )
    return {
        "status": "passed" if passed else "failed",
        "method": "measured_leave_one_interaction_out",
        "omission_results": omission_results,
    }


def candidate_sources(
    rough_candidate: dict[str, Any],
    episodes_by_index: dict[int, dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    preferred_source = rough_candidate.get("_preferred_source_episode_index")
    if preferred_source is not None:
        preferred_source = int(preferred_source)
        if preferred_source not in episodes_by_index:
            return []
        return [(preferred_source, episodes_by_index[preferred_source])]
    ordered = []
    seen = set()
    for option in rough_candidate.get("path_options", []):
        index = int(option["source_episode_index"])
        if index in episodes_by_index and index not in seen:
            seen.add(index)
            ordered.append((index, episodes_by_index[index]))
    for index in rough_candidate.get("source_episode_indices", []):
        index = int(index)
        if index in episodes_by_index and index not in seen:
            seen.add(index)
            ordered.append((index, episodes_by_index[index]))
    return ordered


def measured_mixed_path_candidate(
    args: argparse.Namespace,
    *,
    ctx: probe.LoadedContext,
    container: dict[str, Any],
    object_record: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    source_options: list[tuple[int, dict[str, Any]]],
    open_map,
    doorway_analysis: dict[str, Any],
    interaction_requirement: str = "required",
    closed_maps: dict[str, Any] | None = None,
    doors_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if interaction_requirement not in {"required", "beneficial"}:
        raise ValueError(
            f"Unsupported measured mixed requirement: {interaction_requirement}"
        )
    if closed_maps is None:
        closed_maps = {}
    if doors_by_name is None:
        door_records = emi.collect_interactive_door_root_object_records(
            ctx.env, doorway_analysis
        )
        doors_by_name = {row["name"]: row for row in door_records}
    for source_index, source_episode in source_options:
        start_xy = np.asarray(source_episode["task"]["robot_base_pose"][:2], dtype=float)
        for selected in selected_candidates:
            start_validation = container_builder.source_start_validation(
                ctx,
                container,
                object_record["name"],
                source_episode,
                open_map,
                selected["robot_pose"],
                args.visibility_threshold,
            )
            if not start_validation["valid"]:
                continue
            goal_xy = np.asarray(selected["robot_pose"][:2, 3], dtype=float)
            open_path = emi.compute_path_from_map(
                open_map, start_xy, goal_xy, downscale_factor=1
            )
            crossed = door_scan.traversed_interactive_doors_on_path(
                ctx.env,
                doorway_analysis,
                open_path,
                padding_m=args.door_on_path_padding_m,
                sample_step_m=args.path_region_sample_step_m,
            )
            crossed = door_builder.sort_door_records_by_path_entry(crossed, open_path, args)
            for door_record in crossed:
                root_name = door_record["name"]
                approach = mixed_rough.path_door_approach(
                    open_path,
                    door_record,
                    padding_m=args.door_on_path_padding_m,
                    sample_step_m=args.path_region_sample_step_m,
                    standoff_m=args.door_approach_standoff_m,
                )
                if approach is None:
                    continue
                try:
                    if root_name not in closed_maps:
                        container_builder.open_all_available_doors(ctx)
                        close_transition = emi.set_door_root_state(
                            ctx.env, doorway_analysis, root_name, "closed"
                        )
                        flatten_single_leaf_transition(close_transition)
                        closed_maps[root_name] = emi.build_live_procthor_map(
                            ctx.env.current_model,
                            ctx.env.current_data,
                            model_path=str(ctx.env.current_model_path),
                            px_per_m=args.px_per_m,
                            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
                            open_threshold=args.open_threshold,
                            treat_all_non_interactive_doorways_as_open=True,
                        )
                        emi.set_door_root_state(
                            ctx.env, doorway_analysis, root_name, "open"
                        )
                    initial_map = closed_maps[root_name]
                    initial_path = emi.compute_path_from_map(
                        initial_map, start_xy, goal_xy, downscale_factor=1
                    )
                    approach_path = emi.compute_path_from_map(
                        initial_map,
                        start_xy,
                        np.asarray(approach["approach_xy"], dtype=float),
                        downscale_factor=1,
                    )
                except Exception:
                    continue
                path_evidence = mixed_rough.closed_path_evidence(
                    open_path,
                    initial_path,
                    min_shortcut_delta_m=args.min_shortcut_delta_m,
                    min_shortcut_ratio=args.min_shortcut_ratio,
                )
                requirement_passed = bool(
                    (
                        interaction_requirement == "required"
                        and initial_path is None
                    )
                    or (
                        interaction_requirement == "beneficial"
                        and initial_path is not None
                        and path_evidence["mixed_shortcut_verified"]
                    )
                )
                if requirement_passed and approach_path is not None:
                    return {
                        **selected,
                        "source_episode_index": source_index,
                        "source_episode": source_episode,
                        "start_validation": start_validation,
                        "start_xy": start_xy,
                        "goal_xy": goal_xy,
                        "open_path": open_path,
                        "initial_path": initial_path,
                        "interaction_requirement": interaction_requirement,
                        "path_evidence": path_evidence,
                        "crossed_door_roots": [row["name"] for row in crossed],
                        "door_record": doors_by_name[root_name],
                        "door_root_name": root_name,
                        "approach": approach,
                        "approach_path": approach_path,
                    }
    return None


def build_episode(
    args: argparse.Namespace,
    *,
    case_id: str,
    source_episode_index: int,
    source_episode: dict[str, Any],
    container: dict[str, Any],
    object_record: dict[str, Any],
    selected: dict[str, Any],
    channel_interaction: dict[str, Any],
    container_interactions: list[dict[str, Any]],
    container_interaction_ids: dict[int, str],
    articulation_states: list[dict[str, Any]],
    scene_object_poses: dict[str, list[float]],
    door_state_validation: dict[str, Any],
    container_state_validation: dict[str, Any],
    open_transition: dict[str, Any],
    initial_path,
    restored_path,
    initial_start_visibility: dict[str, Any],
    matching_instance_count: int,
) -> dict[str, Any]:
    interaction_requirement = str(selected["interaction_requirement"])
    path_evidence = dict(selected["path_evidence"])
    episode = copy.deepcopy(source_episode)
    category = container_builder.target_category(object_record)
    episode["task"]["task_cls"] = "molmo_spaces.tasks.nav_task.NavToObjTask"
    episode["task"]["task_type"] = "nav_to_obj"
    episode["task"]["selection_mode"] = "specific_instance"
    episode["task"]["pickup_obj_name"] = object_record["name"]
    episode["task"]["pickup_obj_candidates"] = [object_record["name"]]
    episode["task_relevant_objects"] = [
        object_record["name"],
        container["name"],
        channel_interaction["object_name"],
        selected["door_root_name"],
    ]
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

    interactions = [channel_interaction] + container_interactions
    plan = build_mixed_oracle_plan(
        channel_interaction=channel_interaction,
        container=container,
        selected=selected,
        container_interaction_ids=container_interaction_ids,
        target_object_name=object_record["name"],
        approach_xy=np.asarray(selected["approach"]["approach_xy"], dtype=float),
        door_center_xy=np.asarray(
            selected["door_record"].get(
                "portal_center_xy", selected["door_record"]["aabb_center"]
            ),
            dtype=float,
        )[:2],
        interaction_requirement=interaction_requirement,
    )
    final_trace = selected["visibility_trace"][-1]
    target_position = np.asarray(
        final_trace.get("object_position", object_record["aabb_center"]), dtype=float
    )
    robot_position = np.asarray(selected["robot_pose"], dtype=float)[:3, 3]
    planar_distance = float(np.linalg.norm(robot_position[:2] - target_position[:2]))
    distance_threshold = float(episode["task"]["succ_pos_threshold"])
    distance_passed = planar_distance < distance_threshold
    visibility_fraction = float(final_trace["visibility_fraction"])
    visible_pixels = int(final_trace["visible_pixels"])
    visibility_passed = visibility_fraction > 0.0 and visible_pixels > 0
    if not (distance_passed and visibility_passed):
        raise ValueError("Measured mixed terminal state does not satisfy NavToObj success")
    prefixes = build_mixed_prefixes(
        plan=plan,
        channel_interaction_id_value=channel_interaction["interaction_id"],
        selected=selected,
        container_interaction_ids=container_interaction_ids,
        distance_passed=distance_passed,
        interaction_requirement=interaction_requirement,
    )
    minimal = build_minimal_plan_validation(
        channel_interaction=channel_interaction,
        container_interactions=container_interactions,
        selected=selected,
        initial_path_found=initial_path is not None,
        interaction_requirement=interaction_requirement,
        path_evidence=path_evidence,
    )
    if minimal["status"] != "passed":
        raise ValueError("Measured leave-one-interaction-out validation failed")

    interaction_validations = [
        {
            "interaction_id": channel_interaction["interaction_id"],
            "interaction_domain": "channel",
            "door_root_name": selected["door_root_name"],
            "closed_joint_position": channel_interaction["initial_joint_position"],
            "closed_open_fraction": channel_interaction["initial_state"]["joint_fraction"],
            "open_joint_position": channel_interaction["target_joint_position"],
            "open_fraction": channel_interaction["target_state"]["joint_fraction"],
            "path_found_before": initial_path is not None,
            "path_found_after": restored_path is not None,
            "interaction_requirement": interaction_requirement,
            "path_length_before_m": emi.path_length(initial_path),
            "path_length_after_m": emi.path_length(restored_path),
            "path_length_delta_m": path_evidence.get("path_length_delta_m"),
            "path_length_ratio_delta": path_evidence.get(
                "path_length_ratio_delta"
            ),
            "validated_by": "mujoco_joint_readback_and_live_occupancy_map",
        }
    ]
    container_by_joint = {int(row["joint_index"]): row for row in container_interactions}
    for sequence_index, joint_index in enumerate(selected["joint_sequence"]):
        interaction = container_by_joint[int(joint_index)]
        before = selected["visibility_trace"][sequence_index]
        after = selected["visibility_trace"][sequence_index + 1]
        interaction_validations.append(
            {
                "interaction_id": interaction["interaction_id"],
                "interaction_domain": "container",
                "joint_index": int(joint_index),
                "visibility_fraction_before": float(before["visibility_fraction"]),
                "visible_pixels_before": int(before["visible_pixels"]),
                "visibility_fraction_after": float(after["visibility_fraction"]),
                "visible_pixels_after": int(after["visible_pixels"]),
                "validated_by": "mujoco_articulation_transition_and_segmentation_readback",
            }
        )

    episode["interactive_nav"] = {
        "schema_version": "interactive_nav_v3",
        "case_id": case_id,
        "parent_benchmark_episode_index": source_episode_index,
        "interaction_domains": ["channel", "container"],
        "interaction_requirement": interaction_requirement,
        "target": v3.build_container_target(
            object_record=object_record,
            category=category,
            container=container,
            matching_instance_count=matching_instance_count,
        ),
        "success_criteria": v3.build_nav_to_obj_success_criteria(distance_threshold),
        "initial_state": {
            "interaction_states": [
                {
                    "interaction_id": row["interaction_id"],
                    **row["initial_state"],
                }
                for row in interactions
            ],
            "all_doors_open": False,
            "required_door_roots_closed": (
                [selected["door_root_name"]]
                if interaction_requirement == "required"
                else []
            ),
            "beneficial_door_roots_closed": (
                [selected["door_root_name"]]
                if interaction_requirement == "beneficial"
                else []
            ),
            "container_joints_closed": bool(container_state_validation["all_closed"]),
            "target_visible": False,
        },
        "interactions": interactions,
        "oracle_plan": plan,
        "oracle_plans": [copy.deepcopy(plan)],
        "generation_validation": {
            "navigation_validation": {
                "validation_mode": "live_occupancy_map_and_simulated_terminal_state",
                "all_open_path_found": selected["open_path"] is not None,
                "all_open_path_length_m": emi.path_length(selected["open_path"]),
                "all_open_path_crossed_door_roots": selected["crossed_door_roots"],
                "initial_state_path_found": initial_path is not None,
                "initial_state_path_length_m": emi.path_length(initial_path),
                "approach_path_found": selected["approach_path"] is not None,
                "approach_path_length_m": emi.path_length(selected["approach_path"]),
                "oracle_restored_path_found": restored_path is not None,
                "oracle_restored_path_length_m": emi.path_length(restored_path),
                "path_length_delta_m": path_evidence.get("path_length_delta_m"),
                "path_length_ratio_delta": path_evidence.get(
                    "path_length_ratio_delta"
                ),
                "shortcut_thresholds": path_evidence.get("shortcut_thresholds"),
                "shortcut_verified": path_evidence.get(
                    "mixed_shortcut_verified", False
                ),
                "start_visibility_fraction": float(
                    initial_start_visibility["visibility_fraction"]
                ),
                "start_visible_pixels": int(initial_start_visibility["visible_pixels"]),
                "interaction_pose": pose_mat_to_7d(selected["robot_pose"]).tolist(),
                "interaction_pose_meta": selected["pose_meta"],
                "interaction_pose_collision_free": True,
                "door_approach": selected["approach"],
            },
            "interaction_validations": interaction_validations,
            "oracle_prefixes": prefixes,
            "compartment_evidence": (
                selected["binding"] if selected["binding"].get("applicable") else None
            ),
            "success_evidence": {
                "status": "passed",
                "validation_mode": "simulated_terminal_state",
                "target_object_name": object_record["name"],
                "planar_distance_m": planar_distance,
                "distance_threshold_m": distance_threshold,
                "camera_name": "head_camera",
                "visibility_fraction": visibility_fraction,
                "visible_pixels": visible_pixels,
                "distance_passed": distance_passed,
                "visibility_passed": visibility_passed,
                "expected_task_success": True,
            },
            "minimal_plan_verified": True,
            "minimal_plan_validation": minimal,
            "door_state_validation": door_state_validation,
            "container_state_validation": container_state_validation,
            "door_open_transition": open_transition,
            "controlling_joint_type": selected.get("joint_type"),
            "first_visible_after_joint_index": int(selected["joint"]["joint_index"]),
        },
    }
    return v3.validate_mixed_v3_episode(episode)


def collect_candidate(
    args: argparse.Namespace,
    *,
    ctx: probe.LoadedContext,
    rough_candidate: dict[str, Any],
    episodes_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    container_builder.open_all_available_doors(ctx)
    _, initial_containers = probe.collect_scene_records(ctx)
    container_builder.close_all_containers(ctx.env, initial_containers)
    records, containers = probe.collect_scene_records(ctx)
    containers_by_name = {row["name"]: row for row in containers}
    objects_by_name = {row["name"]: row for row in records if probe.is_target_like(row)}
    container = containers_by_name.get(rough_candidate["container_name"])
    object_record = objects_by_name.get(rough_candidate["object_name"])
    if container is None or object_record is None:
        raise ValueError("Rough container-object pair is absent from the live scene")
    if not probe.compute_relation(container, object_record)["inside_aabb"]:
        raise ValueError("Rough container-object pair no longer satisfies live containment")
    dependencies = probe.infer_joint_open_dependencies(
        ctx.env, container, method="front_occlusion"
    )
    container_builder.open_all_available_doors(ctx)
    container_builder.close_all_containers(ctx.env, containers)
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
    source_options = candidate_sources(rough_candidate, episodes_by_index)
    doors_by_name = {
        row["name"]: row
        for row in emi.collect_interactive_door_root_object_records(
            ctx.env, doorway_analysis
        )
    }
    closed_maps: dict[str, Any] = {}
    rough_candidate_type = str(rough_candidate.get("rough_candidate_type"))
    if rough_candidate_type == "mixed_required_verified":
        requested_requirement = "required"
    elif rough_candidate_type == "mixed_shortcut_verified":
        requested_requirement = "beneficial"
    else:
        raise ValueError(
            f"Unsupported fine mixed rough type: {rough_candidate_type}"
        )

    def accept_mixed_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        measured = measured_mixed_path_candidate(
            args,
            ctx=ctx,
            container=container,
            object_record=object_record,
            selected_candidates=[candidate],
            source_options=source_options,
            open_map=open_map,
            doorway_analysis=doorway_analysis,
            interaction_requirement=requested_requirement,
            closed_maps=closed_maps,
            doors_by_name=doors_by_name,
        )
        if measured is None:
            return {
                "accepted": False,
                "reason": f"fine_pose_not_mixed_{requested_requirement}",
            }
        metadata = {
            key: value
            for key, value in measured.items()
            if key not in candidate
        }
        return {"accepted": True, "metadata": metadata}

    analysis = container_builder.analyze_object_pair(
        args,
        ctx,
        container,
        object_record,
        dependencies,
        rough_candidate["case_id"],
        candidate_acceptor=accept_mixed_candidate,
    )
    if not analysis["valid"]:
        raise ValueError(
            f"No measured mixed container visibility unlock: {analysis['reason']}"
        )
    selected = analysis["selected"]

    container_builder.open_all_available_doors(ctx)
    container_builder.close_all_containers(ctx.env, containers)
    close_transition = emi.set_door_root_state(
        ctx.env, doorway_analysis, selected["door_root_name"], "closed"
    )
    closed_leaf = flatten_single_leaf_transition(close_transition)
    container_builder.close_all_containers(ctx.env, containers)
    initial_map = emi.build_live_procthor_map(
        ctx.env.current_model,
        ctx.env.current_data,
        model_path=str(ctx.env.current_model_path),
        px_per_m=args.px_per_m,
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
        open_threshold=args.open_threshold,
        treat_all_non_interactive_doorways_as_open=True,
    )
    initial_path = emi.compute_path_from_map(
        initial_map, selected["start_xy"], selected["goal_xy"], downscale_factor=1
    )
    approach_path = emi.compute_path_from_map(
        initial_map,
        selected["start_xy"],
        np.asarray(selected["approach"]["approach_xy"], dtype=float),
        downscale_factor=1,
    )
    if approach_path is None:
        raise ValueError("Initial closed-door state cannot reach the door approach")
    if requested_requirement == "required" and initial_path is not None:
        raise ValueError("Initial closed-door state failed mixed_required path proof")
    if requested_requirement == "beneficial" and initial_path is None:
        raise ValueError("Initial closed-door state lacks a shortcut comparison path")
    selected["approach_path"] = approach_path
    source_pose = container_builder.robot_pose_from_episode(selected["source_episode"])
    initial_start_trace = probe.container_visibility_trace(
        ctx,
        container,
        object_record["name"],
        [],
        source_pose,
        view_profile="default",
    )["trace"][0]
    if (
        float(initial_start_trace["visibility_fraction"]) > 0.0
        or int(initial_start_trace["visible_pixels"]) > 0
    ):
        raise ValueError("Target is visible in the measured mixed initial state")
    articulation_states = container_builder.articulation_initial_states(ctx, containers)
    container_validation = container_builder.validate_all_containers_closed(ctx, containers)
    if not container_validation["all_closed"]:
        raise ValueError("Container joints are not all measured closed")
    scene_object_poses = {
        record["name"]: (
            np.asarray(record["position"], dtype=float).tolist()
            + np.asarray(record["quat"], dtype=float).tolist()
        )
        for record in records
        if record.get("has_free_joint")
    }

    open_transition = emi.set_door_root_state(
        ctx.env, doorway_analysis, selected["door_root_name"], "open"
    )
    opened_leaf = flatten_single_leaf_transition(open_transition)
    channel_interaction = build_channel_interaction(
        case_id=rough_candidate["case_id"],
        root_name=selected["door_root_name"],
        closed_leaf=closed_leaf,
        opened_leaf=opened_leaf,
        interaction_requirement=requested_requirement,
    )
    restored_map = emi.build_live_procthor_map(
        ctx.env.current_model,
        ctx.env.current_data,
        model_path=str(ctx.env.current_model_path),
        px_per_m=args.px_per_m,
        agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
        open_threshold=args.open_threshold,
        treat_all_non_interactive_doorways_as_open=True,
    )
    restored_path = emi.compute_path_from_map(
        restored_map, selected["start_xy"], selected["goal_xy"], downscale_factor=1
    )
    if restored_path is None:
        raise ValueError("Opening mixed door did not produce a measured path")
    selected["path_evidence"] = mixed_rough.closed_path_evidence(
        restored_path,
        initial_path,
        min_shortcut_delta_m=args.min_shortcut_delta_m,
        min_shortcut_ratio=args.min_shortcut_ratio,
    )
    selected["interaction_requirement"] = requested_requirement
    if (
        requested_requirement == "beneficial"
        and not selected["path_evidence"]["mixed_shortcut_verified"]
    ):
        raise ValueError(
            "Opening beneficial door did not pass the measured shortcut thresholds"
        )
    trace_result = probe.container_visibility_trace(
        ctx,
        container,
        object_record["name"],
        selected["joint_sequence"],
        selected["robot_pose"],
        view_profile=selected["view_profile"],
        force_slide_joints=selected.get("force_slide_joints", False),
        output_dir=(
            args.output_dir / "evidence" / rough_candidate["case_id"]
            if args.save_images
            else None
        ),
    )
    selected["visibility_trace"] = trace_result["trace"]
    selected["view_state"] = trace_result["view_state"]
    container_interactions, container_ids = v3.build_container_interactions(
        container=container,
        oracle_candidates=[selected],
        articulation_states=articulation_states,
    )
    if requested_requirement == "required":
        attach_channel_prerequisite(
            container_interactions, channel_interaction["interaction_id"]
        )
    all_closed = float(closed_leaf["open_fraction"]) <= 0.01
    door_validation = {
        "threshold": 0.01,
        "door_count": 1,
        "closed_root_names": [selected["door_root_name"]],
        "required_closed_root_names": (
            [selected["door_root_name"]]
            if requested_requirement == "required"
            else []
        ),
        "beneficial_closed_root_names": (
            [selected["door_root_name"]]
            if requested_requirement == "beneficial"
            else []
        ),
        "all_closed": all_closed,
        "all_required_closed": all_closed,
        "doors": [
            {
                "interaction_id": channel_interaction["interaction_id"],
                "door_root_name": selected["door_root_name"],
                "object_name": closed_leaf["object_name"],
                "joint_name": closed_leaf["joint_name"],
                "joint_index": int(closed_leaf["joint_index"]),
                "joint_value": float(closed_leaf["joint_position"]),
                "open_fraction": float(closed_leaf["open_fraction"]),
                "passed_closed": float(closed_leaf["open_fraction"]) <= 0.01,
            }
        ],
    }
    matching_count = sum(
        container_builder.target_category(record)
        == container_builder.target_category(object_record)
        for record in records
        if probe.is_target_like(record)
    )
    case_id = f"{rough_candidate['case_id']}__src{selected['source_episode_index']}"
    return build_episode(
        args,
        case_id=case_id,
        source_episode_index=selected["source_episode_index"],
        source_episode=selected["source_episode"],
        container=container,
        object_record=object_record,
        selected=selected,
        channel_interaction=channel_interaction,
        container_interactions=container_interactions,
        container_interaction_ids=container_ids,
        articulation_states=articulation_states,
        scene_object_poses=scene_object_poses,
        door_state_validation=door_validation,
        container_state_validation=container_validation,
        open_transition=open_transition,
        initial_path=initial_path,
        restored_path=restored_path,
        initial_start_visibility=initial_start_trace,
        matching_instance_count=int(matching_count),
    )


def run(args: argparse.Namespace) -> int:
    rough_paths = [args.mixed_rough_catalog] + list(
        args.additional_mixed_rough_catalog or []
    )
    candidates_by_case: dict[str, dict[str, Any]] = {}
    for rough_path in rough_paths:
        rough_payload = json.loads(rough_path.read_text())
        if rough_payload.get("schema_version") != "mixed_rough_catalog_v1":
            raise ValueError("Fine mixed production only accepts mixed_rough_catalog_v1")
        for candidate in rough_payload.get("candidates", []):
            candidates_by_case.setdefault(candidate["case_id"], candidate)
    episodes = container_builder.load_benchmark_episodes(args.benchmark_dir)
    episodes_by_index = dict(enumerate(episodes))
    candidates = list(candidates_by_case.values())
    allowed_rough_types = {
        value.strip()
        for value in args.rough_candidate_types.split(",")
        if value.strip()
    }
    supported_rough_types = {
        "mixed_required_verified",
        "mixed_shortcut_verified",
    }
    unsupported = allowed_rough_types - supported_rough_types
    if unsupported:
        raise ValueError(f"Unsupported fine mixed rough types: {sorted(unsupported)}")
    candidates = [
        row
        for row in candidates
        if row.get("rough_candidate_type") in allowed_rough_types
    ]
    if args.source_variants_per_pair > 1:
        expanded = []
        for candidate in candidates:
            options = []
            seen_sources = set()
            for option in candidate.get("path_options", []):
                source_index = int(option["source_episode_index"])
                if source_index in seen_sources:
                    continue
                seen_sources.add(source_index)
                options.append(option)
                if len(options) >= args.source_variants_per_pair:
                    break
            if not options:
                expanded.append(candidate)
                continue
            for variant_index, option in enumerate(options):
                expanded.append(
                    {
                        **candidate,
                        "_preferred_source_episode_index": int(
                            option["source_episode_index"]
                        ),
                        "_source_variant_index": variant_index,
                        "_preferred_path_length_m": float(
                            option["all_open_path_length_m"]
                        ),
                    }
                )
        candidates = expanded
    if args.house_indices:
        requested = {int(value) for value in args.house_indices.split(",")}
        candidates = [row for row in candidates if int(row["house_index"]) in requested]
    candidates.sort(
        key=lambda row: (
            float(
                row.get(
                    "_preferred_path_length_m", row["all_open_path_length_m"]
                )
            ),
            int(row["estimated_total_interaction_count"]),
            row["case_id"],
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = []
    valid = []
    rejected = []
    collected_houses: set[int] = set()
    started_at = time.perf_counter()
    current_house = None
    ctx = None
    try:
        for candidate_index, candidate in enumerate(candidates, start=1):
            if len(benchmark) >= args.max_samples:
                break
            house_index = int(candidate["house_index"])
            if args.max_samples_per_house is not None and sum(
                row["house_index"] == house_index for row in valid
            ) >= args.max_samples_per_house:
                continue
            if current_house != house_index:
                if ctx is not None:
                    probe.close_context(ctx)
                source_indices = candidate.get("source_episode_indices", [])
                if not source_indices:
                    rejected.append({**candidate, "reason": "no_source_episode_indices"})
                    continue
                template_episode = episodes_by_index[int(source_indices[0])]
                ctx = container_builder.load_episode_context(args, template_episode)
                current_house = house_index
            candidate_started = time.perf_counter()
            try:
                episode = collect_candidate(
                    args,
                    ctx=ctx,
                    rough_candidate=candidate,
                    episodes_by_index=episodes_by_index,
                )
                benchmark.append(episode)
                collected_houses.add(house_index)
                valid.append(
                    {
                        "case_id": episode["interactive_nav"]["case_id"],
                        "house_index": house_index,
                        "source_episode_index": episode["interactive_nav"][
                            "parent_benchmark_episode_index"
                        ],
                        "container_name": candidate["container_name"],
                        "object_name": candidate["object_name"],
                        "rough_candidate_type": candidate["rough_candidate_type"],
                        "interaction_requirement": episode["interactive_nav"][
                            "interaction_requirement"
                        ],
                        "path_length_m": episode["interactive_nav"][
                            "generation_validation"
                        ]["navigation_validation"]["all_open_path_length_m"],
                        "interaction_count": len(
                            episode["interactive_nav"]["interactions"]
                        ),
                        "elapsed_sec": time.perf_counter() - candidate_started,
                    }
                )
                print(
                    f"[{candidate_index}/{len(candidates)}] collected={len(benchmark)}/"
                    f"{args.max_samples} house={house_index} case={candidate['case_id']} "
                    f"elapsed={valid[-1]['elapsed_sec']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                reason = rejection_reason(exc)
                rejected.append(
                    {
                        "case_id": candidate["case_id"],
                        "house_index": house_index,
                        "container_name": candidate["container_name"],
                        "object_name": candidate["object_name"],
                        "reason": reason,
                        "error": str(exc),
                        "elapsed_sec": time.perf_counter() - candidate_started,
                    }
                )
                print(
                    f"[{candidate_index}/{len(candidates)}] rejected house={house_index} "
                    f"case={candidate['case_id']} error={exc}",
                    flush=True,
                )
            write_json(args.output_dir / "benchmark.partial.json", benchmark)
            write_json(args.output_dir / "valid.partial.json", valid)
            write_json(args.output_dir / "rejected.partial.json", rejected)
    finally:
        if ctx is not None:
            probe.close_context(ctx)

    summary = {
        "schema_version": "mixed_interaction_v3_collection_summary_v1",
        "mixed_rough_catalogs": [str(path) for path in rough_paths],
        "benchmark_dir": str(args.benchmark_dir),
        "requested_sample_count": args.max_samples,
        "generated_episode_count": len(benchmark),
        "collection_house_count": len(collected_houses),
        "rough_candidate_types": sorted(allowed_rough_types),
        "interaction_requirement_counts": dict(
            Counter(row["interaction_requirement"] for row in valid)
        ),
        "rejected_candidate_count": len(rejected),
        "rejection_reason_counts": dict(Counter(row["reason"] for row in rejected)),
        "interaction_count_distribution": dict(
            Counter(str(row["interaction_count"]) for row in valid)
        ),
        "path_length_m": mixed_rough.numeric_summary(
            [float(row["path_length_m"]) for row in valid]
        ),
        "elapsed_sec": time.perf_counter() - started_at,
    }
    write_json(args.output_dir / "benchmark.json", benchmark)
    write_json(args.output_dir / "valid.json", valid)
    write_json(args.output_dir / "rejected.json", rejected)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if len(benchmark) >= args.max_samples else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Directly collect production-valid interactive_nav_v3 mixed episodes with "
            "required or measured cost-beneficial channel interactions."
        )
    )
    parser.add_argument("--mixed_rough_catalog", type=Path, default=DEFAULT_ROUGH_CATALOG)
    parser.add_argument(
        "--additional_mixed_rough_catalog",
        type=Path,
        action="append",
        default=[],
        help="Additional mixed_rough_catalog_v1 files to merge by case_id before collection.",
    )
    parser.add_argument("--benchmark_dir", type=Path, default=container_builder.DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument(
        "--rough_candidate_types",
        default="mixed_required_verified,mixed_shortcut_verified",
        help=(
            "Comma-separated rough types to materialize. Supported: "
            "mixed_required_verified,mixed_shortcut_verified."
        ),
    )
    parser.add_argument("--max_samples_per_house", type=int, default=1)
    parser.add_argument(
        "--source_variants_per_pair",
        type=int,
        default=1,
        help="Collect up to this many measured source-start variants per rough pair.",
    )
    parser.add_argument("--house_indices")
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility_threshold", type=float, default=1e-4)
    parser.add_argument("--interaction_distance", type=float, default=0.8)
    parser.add_argument("--max_poses_per_joint", type=int, default=4)
    parser.add_argument("--drawer_box_padding", type=float, default=0.05)
    parser.add_argument("--px_per_m", type=float, default=100.0)
    parser.add_argument("--open_threshold", type=float, default=0.67)
    parser.add_argument("--door_on_path_padding_m", type=float, default=0.2)
    parser.add_argument("--path_region_sample_step_m", type=float, default=0.05)
    parser.add_argument("--door_approach_standoff_m", type=float, default=0.65)
    parser.add_argument("--min_shortcut_delta_m", type=float, default=0.25)
    parser.add_argument("--min_shortcut_ratio", type=float, default=0.02)
    parser.add_argument(
        "--save_images", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save_plots", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
