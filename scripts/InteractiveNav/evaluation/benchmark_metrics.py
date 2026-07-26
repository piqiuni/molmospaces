"""V3 benchmark terminal checks and aggregate reporting metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from scripts.InteractiveNav import container_scene_probe as probe


SUCCESS_OPEN_FRACTION = 0.8
DEFAULT_PATH_BINS_M = (0.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def joint_open_fraction(env: Any, interaction: dict[str, Any]) -> float:
    """Read a joint in the benchmark's semantic closed-to-open direction."""

    joint_name = str(interaction["joint_name"])
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Interaction joint not found: {joint_name}")
    value = probe.joint_value_by_name(env, joint_name)
    lower, upper = [float(item) for item in env.current_model.jnt_range[joint_id]]
    closed, opened = probe.joint_closed_open_values([lower, upper])
    return float(np.clip(probe.semantic_open_fraction(value, closed, opened), 0.0, 1.0))


def target_metrics(task: Any, episode: dict[str, Any]) -> tuple[bool, float, float]:
    """Evaluate the frozen V3 selected instance, never a category substitute."""

    target_name = str(episode["interactive_nav"]["target"]["selected_instance"])
    objects = task.env.object_managers[task.env.current_batch_index]
    target = objects.get_object_by_name(target_name)
    robot_xy = np.asarray(task.env.current_robot.robot_view.base.pose[:2, 3], dtype=float)
    distance = float(np.linalg.norm(np.asarray(target.position[:2], dtype=float) - robot_xy))
    criteria = episode["interactive_nav"]["success_criteria"]
    camera_name = str(criteria["visibility"].get("camera_name", "head_camera"))
    visibility = float(task.env.check_visibility(camera_name, target_name))
    threshold = float(criteria["distance"]["threshold_m"])
    return bool(distance < threshold and visibility > 0.0), distance, visibility


def oracle_terminal_goal_consistency(task: Any, episode: dict[str, Any]) -> dict[str, Any]:
    """Check that frozen terminal waypoints still describe the live target.

    V3 stores a selected *instance* and a frozen oracle plan separately.  A
    benchmark assembled from a stale source can accidentally pair the selected
    instance with a navigation goal for a different same-category object.  The
    evaluator must surface that before treating a policy score as a formal
    benchmark result.

    The check is intentionally conservative: it only rejects an episode when a
    plan explicitly labels a navigation step as ``satisfy_nav_to_obj_success``
    and none of those terminal goals could meet the recorded distance criterion
    under the live replayed scene.  Episodes without such a step remain
    ``checked=False`` rather than being guessed incompatible.
    """

    nav = episode["interactive_nav"]
    target_name = str(nav["target"]["selected_instance"])
    objects = task.env.object_managers[task.env.current_batch_index]
    target = objects.get_object_by_name(target_name)
    target_xy = np.asarray(target.position[:2], dtype=float)
    threshold = float(nav["success_criteria"]["distance"]["threshold_m"])
    candidates: list[dict[str, Any]] = []
    for plan in _oracle_plans(nav):
        plan_id = str(plan.get("plan_id", "oracle"))
        for step in plan.get("steps", []):
            if step.get("type") != "navigate" or step.get("reason") != "satisfy_nav_to_obj_success":
                continue
            goal = np.asarray(step.get("goal_point", [])[:2], dtype=float)
            if goal.shape != (2,):
                continue
            tolerance = float(step.get("position_tolerance_m", 0.0))
            distance = float(np.linalg.norm(goal - target_xy))
            allowed = threshold + max(0.0, tolerance)
            candidates.append(
                {
                    "plan_id": plan_id,
                    "goal_point_xy": goal.tolist(),
                    "position_tolerance_m": tolerance,
                    "distance_to_live_target_m": distance,
                    "allowed_distance_m": allowed,
                    "consistent": bool(distance < allowed),
                }
            )
    return {
        "checked": bool(candidates),
        "consistent": True if not candidates else any(bool(item["consistent"]) for item in candidates),
        "target_name": target_name,
        "target_position_xy": target_xy.tolist(),
        "success_distance_threshold_m": threshold,
        "terminal_goal_candidates": candidates,
    }


@dataclass(frozen=True)
class InteractionTerminalScore:
    required_interaction_success: bool
    sequence_success: bool
    non_interaction_success: bool | None
    interaction_fractions: dict[str, float]
    valid_plan_id: str | None
    correct_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _oracle_plans(nav: dict[str, Any]) -> list[dict[str, Any]]:
    plans = list(nav.get("oracle_plans") or [])
    if not plans and nav.get("oracle_plan"):
        plans = [dict(nav["oracle_plan"])]
    return plans


def score_interactions(
    env: Any,
    episode: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> InteractionTerminalScore:
    """Score V3 interactions including failed/extra attempts.

    In particular, no-interaction episodes use *attempt count*, not successful
    interaction ids.  This prevents a policy from receiving credit after an
    invalid or failed door-open request.
    """

    nav = episode["interactive_nav"]
    requirement = str(nav["interaction_requirement"])
    interactions = list(nav.get("interactions", []))
    by_id = {str(row["interaction_id"]): row for row in interactions}
    fractions = {key: joint_open_fraction(env, row) for key, row in by_id.items()}
    successful_required = [
        str(row["resolved_interaction_id"])
        for row in attempts
        if row.get("classification") == "required_valid"
        and bool(row.get("success"))
        and row.get("resolved_interaction_id") is not None
    ]
    completed: set[str] = set()
    sequence_success = True
    correct_action_count = 0
    for interaction_id in successful_required:
        row = by_id.get(interaction_id)
        if row is None:
            sequence_success = False
            continue
        prereq = {str(item["interaction_id"]) for item in row.get("prerequisites", [])}
        order_ok = prereq.issubset(completed)
        if not order_ok:
            sequence_success = False
        elif interaction_id not in completed:
            correct_action_count += 1
        completed.add(interaction_id)

    plans = _oracle_plans(nav)
    plan_rows: list[tuple[str, set[str]]] = [
        (str(plan.get("plan_id", "oracle")), {str(value) for value in plan.get("required_interaction_ids", [])})
        for plan in plans
    ]
    if requirement == "unnecessary":
        required_ok = True
        valid_plan_id = None
        sequence_success = True
        non_interaction_success: bool | None = len(attempts) == 0
    else:
        valid_plan_id = next(
            (
                plan_id
                for plan_id, required_ids in plan_rows
                if required_ids.issubset(completed)
                and all(fractions.get(item, 0.0) >= SUCCESS_OPEN_FRACTION for item in required_ids)
            ),
            None,
        )
        required_ok = valid_plan_id is not None
        if plan_rows and not required_ok:
            sequence_success = False
        non_interaction_success = None
    return InteractionTerminalScore(
        required_interaction_success=bool(required_ok),
        sequence_success=bool(sequence_success),
        non_interaction_success=non_interaction_success,
        interaction_fractions=fractions,
        valid_plan_id=valid_plan_id,
        correct_action_count=int(correct_action_count),
    )


def reference_path_length_m(episode: dict[str, Any]) -> float | None:
    """Return the best frozen V3 path reference usable for SPL.

    Channel/mixed episodes normally carry ``oracle_restored_path_length_m``;
    container episodes store the validated start-to-interaction path as
    ``path_length_m``.  This ordered fallback covers all currently frozen V3
    episodes while retaining the provenance in the result trace.
    """

    nav = episode["interactive_nav"]
    validation = dict(nav.get("generation_validation", {}).get("navigation_validation", {}))
    if str(nav.get("interaction_requirement")) == "unnecessary":
        keys = ("initial_state_path_length_m", "path_length_m", "all_open_path_length_m")
    else:
        keys = ("oracle_restored_path_length_m", "path_length_m", "all_open_path_length_m")
    for key in keys:
        value = validation.get(key)
        if value is not None:
            return float(value)
    return None


def path_length_bin(length_m: float | None, bins: tuple[float, ...] = DEFAULT_PATH_BINS_M) -> str | None:
    if length_m is None:
        return None
    length = float(length_m)
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        if lower <= length < upper:
            return f"[{lower:g},{upper:g})"
    return f"[{bins[-1]:g},inf)"


def spl(success: bool, reference_length_m: float | None, actual_length_m: float) -> float | None:
    if reference_length_m is None:
        return None
    reference = max(0.0, float(reference_length_m))
    actual = max(0.0, float(actual_length_m))
    if not success:
        return 0.0
    return float(reference / max(reference, actual, 1e-9))


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(bool(value) for value in values) / len(values))


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    interaction_attempts = sum(int(row.get("interaction_action_count", 0)) for row in rows)
    correct_actions = sum(int(row.get("correct_interaction_action_count", 0)) for row in rows)
    return {
        "episode_count": len(rows),
        "success_rate": _rate(rows, "success"),
        "nav_success_rate": _rate(rows, "nav_success"),
        "required_interaction_success_rate": _rate(rows, "required_interaction_success"),
        "sequence_success_rate": _rate(rows, "sequence_success"),
        "non_interaction_success_rate": _rate(rows, "non_interaction_success"),
        "interaction_precision": None if interaction_attempts == 0 else correct_actions / interaction_attempts,
        "mean_step_count": _mean(rows, "step_count"),
        "mean_navigation_path_length_m": _mean(rows, "navigation_path_length_m"),
        "mean_reference_path_length_m": _mean(rows, "reference_path_length_m"),
        "mean_spl": _mean(rows, "spl"),
        "spl_eligible_episode_count": sum(row.get("spl") is not None for row in rows),
        "mean_total_simulated_seconds": _mean(rows, "total_simulated_seconds"),
        "mean_interaction_action_count": _mean(rows, "interaction_action_count"),
        "mean_extra_interaction_action_count": _mean(rows, "extra_interaction_action_count"),
        "mean_invalid_interaction_action_count": _mean(rows, "invalid_interaction_action_count"),
        "terminal_reason_counts": dict(Counter(str(row.get("terminal_reason")) for row in rows)),
    }


def summarise_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate overall, task-family and distribution-stratified metrics."""

    eligible_rows = [row for row in rows if bool(row.get("scoring_eligible", True))]
    groups: dict[str, list[dict[str, Any]]] = {"overall": list(eligible_rows)}
    for domain in ("channel", "container", "mixed"):
        expected = {"channel", "container"} if domain == "mixed" else {domain}
        groups[f"domain/{domain}"] = [row for row in eligible_rows if set(row.get("domains", [])) == expected]
    for requirement in ("required", "unnecessary", "beneficial"):
        groups[f"requirement/{requirement}"] = [
            row for row in eligible_rows if row.get("interaction_requirement") == requirement
        ]
    for recipe in sorted({str(row["recipe"]) for row in eligible_rows if row.get("recipe")}):
        groups[f"recipe/{recipe}"] = [row for row in eligible_rows if row.get("recipe") == recipe]
    for interaction_type in sorted({item for row in eligible_rows for item in row.get("interaction_types", [])}):
        groups[f"interaction_type/{interaction_type}"] = [
            row for row in eligible_rows if interaction_type in row.get("interaction_types", [])
        ]
    for label in sorted({str(row["path_length_bin"]) for row in eligible_rows if row.get("path_length_bin")}):
        groups[f"path_length/{label}"] = [row for row in eligible_rows if row.get("path_length_bin") == label]
    return {
        "schema_version": "interactive_nav_v3_benchmark_eval_summary_v2",
        "total_episode_count": len(rows),
        "scoring_eligible_episode_count": len(eligible_rows),
        "runtime_ineligible_episode_count": len(rows) - len(eligible_rows),
        "groups": {name: _group_summary(values) for name, values in groups.items() if values},
    }
