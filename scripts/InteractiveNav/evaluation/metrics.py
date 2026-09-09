"""V3-aware terminal checks and aggregate metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import mujoco
import numpy as np

from scripts.InteractiveNav import container_scene_probe as probe


def joint_open_fraction(env: Any, interaction: dict[str, Any]) -> float:
    """Read a joint in its semantic closed→open orientation."""

    joint_name = str(interaction["joint_name"])
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Interaction joint not found: {joint_name}")
    value = probe.joint_value_by_name(env, joint_name)
    lower, upper = [float(x) for x in env.current_model.jnt_range[joint_id]]
    closed, opened = probe.joint_closed_open_values([lower, upper])
    return float(np.clip(probe.semantic_open_fraction(value, closed, opened), 0.0, 1.0))


def target_metrics(task: Any, episode: dict[str, Any]) -> tuple[bool, float, float]:
    """Evaluate the V3 selected instance, not a category-level substitute."""

    target_name = str(episode["interactive_nav"]["target"]["selected_instance"])
    om = task.env.object_managers[task.env.current_batch_index]
    target = om.get_object_by_name(target_name)
    robot_xy = task.env.current_robot.robot_view.base.pose[:2, 3]
    distance = float(np.linalg.norm(np.asarray(target.position[:2]) - robot_xy))
    visibility = float(task.env.check_visibility("head_camera", target_name))
    threshold = float(episode["interactive_nav"]["success_criteria"]["distance"]["threshold_m"])
    return bool(distance < threshold and visibility > 0.0), distance, visibility


def interaction_terminal_metrics(
    env: Any, episode: dict[str, Any], executed_ids: list[str]
) -> tuple[bool, bool, bool, dict[str, float]]:
    """Return required, sequence and unnecessary-case metrics.

    Sequence validity is evaluated from the recorded order and the prerequisite
    graph.  It does not grant credit merely because a joint was externally
    found open at terminal state.
    """

    nav = episode["interactive_nav"]
    requirement = str(nav["interaction_requirement"])
    interactions = list(nav.get("interactions", []))
    by_id = {str(row["interaction_id"]): row for row in interactions}
    fractions = {key: joint_open_fraction(env, row) for key, row in by_id.items()}
    plans = nav.get("oracle_plans", [])
    # A benchmark may carry alternative valid plans.  Do not union their
    # requirements: that would incorrectly require mutually exclusive actions.
    plan_ids = [
        {str(item) for item in plan.get("required_interaction_ids", [])}
        for plan in plans
    ]
    if plans:
        required_ok = any(
            all(fractions.get(item, 0.0) >= 0.8 for item in ids) for ids in plan_ids
        )
    else:
        required_ok = requirement == "unnecessary"

    completed: set[str] = set()
    sequence_ok = True
    for interaction_id in executed_ids:
        row = by_id.get(interaction_id)
        if row is None:
            continue
        prereq = {str(p["interaction_id"]) for p in row.get("prerequisites", [])}
        if not prereq.issubset(completed):
            sequence_ok = False
        completed.add(interaction_id)

    # Required interactions that were not explicitly actioned are never a
    # sequence success, even if a simulator artefact moved the joint.  As with
    # terminal fractions, completing one *entire* alternative plan is enough.
    if requirement != "unnecessary" and plan_ids and not any(ids.issubset(completed) for ids in plan_ids):
        sequence_ok = False

    non_interaction_ok = requirement != "unnecessary" or not executed_ids
    return bool(required_ok), bool(sequence_ok), bool(non_interaction_ok), fractions


def summarise_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce stable JSON-friendly overall and subgroup aggregates."""

    def rate(values: list[bool | None]) -> float | None:
        usable = [bool(v) for v in values if v is not None]
        return None if not usable else float(sum(usable) / len(usable))

    groups: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for domain in ("channel", "container", "mixed"):
        groups[f"domain/{domain}"] = [
            row for row in rows if set(row.get("domains", [])) == ({"channel", "container"} if domain == "mixed" else {domain})
        ]
    for requirement in ("required", "unnecessary", "beneficial", "unknown"):
        groups[f"requirement/{requirement}"] = [
            row for row in rows if row.get("interaction_requirement") == requirement
        ]

    result: dict[str, Any] = {}
    for name, values in groups.items():
        if not values:
            continue
        result[name] = {
            "episode_count": len(values),
            "success_rate": rate([row.get("success") for row in values]),
            "nav_success_rate": rate([row.get("nav_success") for row in values]),
            "required_interaction_success_rate": rate([row.get("required_interaction_success") for row in values]),
            "sequence_success_rate": rate([row.get("sequence_success") for row in values]),
            "non_interaction_success_rate": rate([row.get("non_interaction_success") for row in values]),
            "mean_step_count": float(np.mean([row.get("step_count", 0) for row in values])),
            "mean_navigation_path_length_m": float(np.mean([row.get("navigation_path_length_m", 0.0) for row in values])),
            "mean_wrong_interaction_count": float(np.mean([row.get("wrong_interaction_count", 0) for row in values])),
            "terminal_reason_counts": dict(Counter(str(row.get("terminal_reason")) for row in values)),
        }
    return {"schema_version": "interactive_nav_v3_eval_summary_v1", "groups": result}
