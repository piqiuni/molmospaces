from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec


SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "dataset_definition/v3/interactive_nav_episode.schema.json"
)


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def build_interaction_id(container_name: str, joint_index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", container_name).strip("-").lower()
    return f"container::{slug}::{int(joint_index)}"


def build_container_target(
    *,
    object_record: dict[str, Any],
    category: str,
    container: dict[str, Any],
    matching_instance_count: int,
) -> dict[str, Any]:
    def json_value(value: Any) -> Any:
        return value.tolist() if hasattr(value, "tolist") else value

    object_name = str(object_record["name"])
    if matching_instance_count < 1:
        raise ValueError("Target grounding must match at least one scene instance")
    container_category = container.get("category")
    if not container_category:
        raise ValueError(f"Missing category for container {container.get('name')!r}")
    return {
        "selection_mode": "specific_instance",
        "category": category,
        "selected_instance": object_name,
        "instruction_consistent_candidates": [object_name],
        "container_name": container["name"],
        "container_category": str(container_category),
        "grounding": {
            "unique": matching_instance_count == 1,
            "matching_instance_count": int(matching_instance_count),
            "description": category,
            "attributes": {},
        },
        "object_aabb_center": json_value(object_record.get("aabb_center")),
        "object_aabb_size": json_value(object_record.get("aabb_size")),
        "container_aabb_center": json_value(container.get("aabb_center")),
        "container_aabb_size": json_value(container.get("aabb_size")),
    }


def build_nav_to_obj_success_criteria(succ_pos_threshold: float) -> dict[str, Any]:
    return {
        "type": "nav_to_obj",
        "target_selection": "specific_instance",
        "distance": {
            "metric": "planar_robot_base_to_object",
            "threshold_m": float(succ_pos_threshold),
            "comparison": "strictly_less",
        },
        "visibility": {
            "camera_name": "head_camera",
            "metric": "visibility_fraction",
            "threshold": 0.0,
            "comparison": "strictly_greater",
        },
        "combination": "all",
    }


def _joint_type(joint: dict[str, Any]) -> str:
    value = str(joint.get("mujoco_joint_type", joint.get("joint_type", ""))).lower()
    normalized = value.strip().strip("[]")
    if "hinge" in value or normalized == "3":
        return "container_hinged_door"
    if "slide" in value or normalized == "2":
        return "container_sliding_drawer"
    raise ValueError(f"Unsupported container joint type for v3: {value!r}")


def _initial_fraction_by_joint(
    articulation_states: list[dict[str, Any]],
) -> dict[str, float]:
    fractions: dict[str, float] = {}
    for state in articulation_states:
        fraction = state.get("open_fraction")
        if fraction is None:
            raise ValueError(
                f"Missing open_fraction for initial joint {state.get('joint_name')!r}"
            )
        fractions[str(state["joint_name"])] = float(fraction)
    return fractions


def build_container_interactions(
    *,
    container: dict[str, Any],
    oracle_candidates: list[dict[str, Any]],
    articulation_states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    container_category = container.get("category")
    if not container_category:
        raise ValueError(f"Missing category for container {container.get('name')!r}")
    joints_by_index = {int(joint["joint_index"]): joint for joint in container["joints"]}
    initial_fractions = _initial_fraction_by_joint(articulation_states)
    interaction_ids: dict[int, str] = {}
    effects: dict[int, set[str]] = {}
    prerequisites: dict[int, set[int]] = {}
    order: list[int] = []

    for candidate in oracle_candidates:
        sequence = [int(index) for index in candidate["joint_sequence"]]
        for sequence_index, joint_index in enumerate(sequence):
            if joint_index not in order:
                order.append(joint_index)
            interaction_ids.setdefault(
                joint_index, build_interaction_id(container["name"], joint_index)
            )
            effects.setdefault(joint_index, set()).add(
                "reveal_target_object"
                if sequence_index == len(sequence) - 1
                else "enable_interaction"
            )
            prerequisites.setdefault(joint_index, set()).update(sequence[:sequence_index])

    interactions = []
    for joint_index in order:
        joint = joints_by_index[joint_index]
        joint_name = str(joint["joint_name"])
        if joint_name not in initial_fractions:
            raise ValueError(f"No articulation initial state for joint {joint_name!r}")
        initial_fraction = initial_fractions[joint_name]
        if abs(initial_fraction) > 1e-6:
            raise ValueError(
                f"Container joint {joint_name!r} is not initially closed: {initial_fraction}"
            )
        interactions.append(
            {
                "interaction_id": interaction_ids[joint_index],
                "type": _joint_type(joint),
                "object_name": container["name"],
                "object_category": str(container_category),
                "joint_name": joint_name,
                "joint_index": joint_index,
                "effect_types": sorted(effects[joint_index]),
                "prerequisites": [
                    {
                        "interaction_id": interaction_ids[required_index],
                        "type": "mechanical",
                    }
                    for required_index in sorted(
                        prerequisites[joint_index], key=order.index
                    )
                ],
                "initial_state": {
                    "joint_fraction": initial_fraction,
                    "semantic_state": "closed",
                },
                "target_state": {
                    "joint_fraction": 1.0,
                    "semantic_state": "open",
                },
            }
        )
    return interactions, interaction_ids


def build_initial_state(
    interactions: list[dict[str, Any]],
    *,
    all_doors_open: bool,
    container_joints_closed: bool,
    target_visible: bool,
) -> dict[str, Any]:
    return {
        "interaction_states": [
            {
                "interaction_id": interaction["interaction_id"],
                **interaction["initial_state"],
            }
            for interaction in interactions
        ],
        "all_doors_open": bool(all_doors_open),
        "container_joints_closed": bool(container_joints_closed),
        "target_visible": bool(target_visible),
    }


def build_oracle_prefixes(
    *,
    plan: dict[str, Any],
    visibility_trace: list[dict[str, Any]],
    distance_passed: bool,
    reachable: bool,
) -> list[dict[str, Any]]:
    open_steps = [step for step in plan["steps"] if step["type"] == "open_joint"]
    first_open_step = next(
        (index for index, step in enumerate(plan["steps"]) if step["type"] == "open_joint"),
        len(plan["steps"]),
    )
    prefixes = []
    for trace_index, row in enumerate(visibility_trace):
        completed_open_steps = min(trace_index, len(open_steps))
        visibility = float(row.get("visibility_fraction", 0.0))
        pixels = int(row.get("visible_pixels", 0))
        prefixes.append(
            {
                "plan_id": plan["plan_id"],
                "completed_step_count": first_open_step + completed_open_steps,
                "robot_reachable_to_next_goal": bool(reachable),
                "target_distance_passed": bool(distance_passed),
                "target_visibility_fraction": visibility,
                "target_visible_pixels": pixels,
                "task_success": bool(distance_passed and visibility > 0.0),
                "opened_interaction_ids": [
                    step["interaction_id"]
                    for step in open_steps[:completed_open_steps]
                ],
            }
        )
    return prefixes


def validate_container_v3_episode(episode: dict[str, Any]) -> dict[str, Any]:
    payload = episode.get("interactive_nav", {})
    if payload.get("interaction_domains") != ["container"]:
        raise ValueError("Container v3 episodes must use interaction_domains=['container']")
    if payload.get("interaction_requirement") != "required":
        raise ValueError("Generated container episodes must require interaction")

    task = episode.get("task", {})
    target = payload.get("target", {})
    selected_instance = target.get("selected_instance")
    if task.get("selection_mode") != "specific_instance":
        raise ValueError("Generated container episodes must use specific_instance")
    if task.get("pickup_obj_name") != selected_instance:
        raise ValueError("task.pickup_obj_name does not match target.selected_instance")
    if task.get("pickup_obj_candidates") != [selected_instance]:
        raise ValueError("specific_instance candidates must contain only the selected target")
    grounding = target.get("grounding", {})
    matching_count = grounding.get("matching_instance_count")
    if not isinstance(matching_count, int) or matching_count < 1:
        raise ValueError("target grounding count must be a positive measured integer")
    if grounding.get("unique") != (matching_count == 1):
        raise ValueError("target grounding unique flag does not match the measured count")
    relevant_objects = episode.get("task_relevant_objects", [])
    for required_object in (selected_instance, target.get("container_name")):
        if required_object not in relevant_objects:
            raise ValueError(f"task_relevant_objects is missing {required_object!r}")

    interactions = payload.get("interactions", [])
    interaction_ids = {
        interaction["interaction_id"] for interaction in payload.get("interactions", [])
    }
    if len(interaction_ids) != len(interactions):
        raise ValueError("Duplicate interaction IDs")
    for interaction in interactions:
        if interaction.get("object_category") in {None, "", "unknown"}:
            raise ValueError("Interaction object_category must be measured, not a placeholder")
        for prerequisite in interaction.get("prerequisites", []):
            if prerequisite["interaction_id"] not in interaction_ids:
                raise ValueError("Interaction prerequisite references an unknown interaction")

    interaction_by_id = {
        interaction["interaction_id"]: interaction for interaction in interactions
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(interaction_id: str) -> None:
        if interaction_id in visiting:
            raise ValueError("Interaction prerequisites contain a cycle")
        if interaction_id in visited:
            return
        visiting.add(interaction_id)
        for prerequisite in interaction_by_id[interaction_id].get("prerequisites", []):
            visit(prerequisite["interaction_id"])
        visiting.remove(interaction_id)
        visited.add(interaction_id)

    for interaction_id in interaction_ids:
        visit(interaction_id)

    initial_by_id = {
        state["interaction_id"]: state
        for state in payload.get("initial_state", {}).get("interaction_states", [])
    }
    articulation_by_joint = {
        state["joint_name"]: state
        for state in episode.get("scene_modifications", {}).get("articulation_states", [])
    }
    for interaction in interactions:
        interaction_id = interaction["interaction_id"]
        if initial_by_id.get(interaction_id) != {
            "interaction_id": interaction_id,
            **interaction["initial_state"],
        }:
            raise ValueError("interactive_nav initial state does not mirror interactions")
        articulation = articulation_by_joint.get(interaction["joint_name"])
        if articulation is None:
            raise ValueError("Interaction joint is missing from scene articulation state")
        if abs(
            float(articulation.get("open_fraction", -1.0))
            - float(interaction["initial_state"]["joint_fraction"])
        ) > 1e-6:
            raise ValueError("Interaction and scene articulation initial fractions disagree")

    generation_validation = payload.get("generation_validation", {})
    door_validation = generation_validation.get("door_state_validation", {})
    door_rows = door_validation.get("doors", [])
    if door_validation.get("door_count") != len(door_rows):
        raise ValueError("door_state_validation count does not match its measured rows")
    if door_validation.get("all_open") != all(row.get("passed") is True for row in door_rows):
        raise ValueError("door_state_validation all_open does not match measured door rows")
    for row in door_rows:
        articulation = articulation_by_joint.get(row["joint_name"])
        if articulation is None:
            raise ValueError("Validated channel door is missing from articulation states")
        if abs(float(articulation.get("position")) - float(row["joint_value"])) > 1e-6:
            raise ValueError("Door articulation position does not match measured validation")
        if abs(
            float(articulation.get("open_fraction")) - float(row["open_fraction"])
        ) > 1e-6:
            raise ValueError("Door articulation fraction does not match measured validation")
    if payload.get("initial_state", {}).get("all_doors_open") is True and not door_validation.get(
        "all_open", False
    ):
        raise ValueError("all_doors_open is not supported by measured door state validation")
    container_validation = generation_validation.get("container_state_validation", {})
    container_rows = container_validation.get("joints", [])
    if container_validation.get("joint_count") != len(container_rows):
        raise ValueError("container_state_validation count does not match measured rows")
    if container_validation.get("all_closed") != all(
        row.get("passed") is True for row in container_rows
    ):
        raise ValueError("container_state_validation all_closed disagrees with measured rows")
    for row in container_rows:
        articulation = articulation_by_joint.get(row["joint_name"])
        if articulation is None:
            raise ValueError("Validated container joint is missing from articulation states")
        if abs(float(articulation.get("position")) - float(row["joint_value"])) > 1e-6:
            raise ValueError("Container articulation position disagrees with measured validation")
        if abs(
            float(articulation.get("open_fraction")) - float(row["open_fraction"])
        ) > 1e-6:
            raise ValueError("Container articulation fraction disagrees with measured validation")
    if payload.get("initial_state", {}).get("container_joints_closed") is True and not (
        container_validation.get("all_closed", False)
    ):
        raise ValueError(
            "container_joints_closed is not supported by measured container state validation"
        )
    if payload.get("initial_state", {}).get("target_visible") is not False:
        raise ValueError("Required container interactions must start with a hidden target")
    navigation_validation = generation_validation.get("navigation_validation", {})
    if not navigation_validation.get("path_found", False):
        raise ValueError("Generated episode has no measured path to its interaction pose")
    if float(navigation_validation.get("start_visibility_fraction", 0.0)) > 0.0 or int(
        navigation_validation.get("start_visible_pixels", 0)
    ) > 0:
        raise ValueError("Generated episode target is visible at the source start")
    success_evidence = generation_validation.get("success_evidence", {})
    if not (
        success_evidence.get("status") == "passed"
        and success_evidence.get("distance_passed") is True
        and success_evidence.get("visibility_passed") is True
        and success_evidence.get("expected_task_success") is True
    ):
        raise ValueError("Generated episode lacks passing measured NavToObj success evidence")
    success_criteria = payload.get("success_criteria", {})
    task_threshold = float(task.get("succ_pos_threshold", -1.0))
    if float(success_criteria.get("distance", {}).get("threshold_m", -2.0)) != task_threshold:
        raise ValueError("success_criteria distance threshold does not mirror the task")
    if float(success_evidence.get("distance_threshold_m", -2.0)) != task_threshold:
        raise ValueError("success_evidence distance threshold does not mirror the task")
    if success_evidence.get("target_object_name") != selected_instance:
        raise ValueError("success_evidence target does not match the selected instance")

    for plan in payload.get("oracle_plans", []):
        completed: list[str] = []
        for step in plan.get("steps", []):
            if step.get("type") != "open_joint":
                continue
            interaction = interaction_by_id[step["interaction_id"]]
            if (
                step.get("object_name") != interaction["object_name"]
                or step.get("joint_name") != interaction["joint_name"]
                or step.get("joint_index") != interaction["joint_index"]
            ):
                raise ValueError("Oracle open_joint step disagrees with its interaction")
            prerequisites = [
                prerequisite["interaction_id"]
                for prerequisite in interaction.get("prerequisites", [])
            ]
            if any(prerequisite not in completed for prerequisite in prerequisites):
                raise ValueError("Oracle plan opens an interaction before its prerequisite")
            completed.append(step["interaction_id"])
        observe_steps = [
            step for step in plan.get("steps", []) if step.get("type") == "observe_target"
        ]
        if len(observe_steps) != 1 or observe_steps[0].get("object_name") != selected_instance:
            raise ValueError("Oracle plan must observe exactly the selected target")

    prefixes_by_plan: dict[str, list[dict[str, Any]]] = {}
    for prefix in generation_validation.get("oracle_prefixes", []):
        prefixes_by_plan.setdefault(prefix["plan_id"], []).append(prefix)
    for plan in payload.get("oracle_plans", []):
        prefixes = prefixes_by_plan.get(plan["plan_id"], [])
        if not prefixes:
            raise ValueError("Oracle plan is missing prefix validation")
        if prefixes[0].get("task_success") is not False:
            raise ValueError("Oracle prefix validation does not prove initial failure")
        if prefixes[-1].get("task_success") is not True:
            raise ValueError("Oracle prefix validation does not prove terminal success")

    minimal_verified = generation_validation.get("minimal_plan_verified")
    minimal_validation = generation_validation.get("minimal_plan_validation", {})
    if minimal_verified is None and minimal_validation.get("status") != "not_executed":
        raise ValueError("Unverified minimal plan must explicitly record not_executed")
    for plan in payload.get("oracle_plans", []):
        required_ids = plan.get("required_interaction_ids", [])
        if any(interaction_id not in interaction_ids for interaction_id in required_ids):
            raise ValueError(f"Plan {plan.get('plan_id')} references an unknown interaction")
        step_ids = [
            step["interaction_id"]
            for step in plan.get("steps", [])
            if step.get("type") == "open_joint"
        ]
        if step_ids != required_ids:
            raise ValueError(
                f"Plan {plan.get('plan_id')} required_interaction_ids do not match steps"
            )
    if payload.get("oracle_plans", [None])[0] != payload.get("oracle_plan"):
        raise ValueError("oracle_plan must equal oracle_plans[0]")

    cleaned_episode = _without_none(episode)
    cleaned_episode["interactive_nav"]["generation_validation"][
        "minimal_plan_verified"
    ] = None
    validated_spec = EpisodeSpec.model_validate(cleaned_episode)
    # JSON Schema validators treat tuples differently from JSON arrays.
    validated = json.loads(validated_spec.model_dump_json(exclude_none=True))
    validated["interactive_nav"]["generation_validation"][
        "minimal_plan_verified"
    ] = None
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return validated
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator(schema).validate(validated)
    return validated
