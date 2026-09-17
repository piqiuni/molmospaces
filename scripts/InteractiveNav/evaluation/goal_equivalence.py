"""Private, post-hoc target equivalence; never an input to the policy.

This is a new scoring protocol, not a correction to the frozen instance score.
An equivalent object still needs the existing public-evidence/distance verifier.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


PROTOCOL = "interactive_nav_goal_equivalence_v1"
CONTAINER_CATEGORIES = frozenset({
    "fridge", "refrigerator", "cabinet", "drawer", "dresser", "chestofdrawers",
    "microwave", "oven", "dishwasher", "box", "safe",
})


def label(value: Any) -> str:
    return str(value or "").strip().casefold().replace("_", "").replace(" ", "")


def ancestors(name: str, objects: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    current = objects.get(name, {}).get("parent")
    while current and current not in result:
        result.append(current)
        current = objects.get(current, {}).get("parent")
    return result


def equivalent_target(
    *, target: Mapping[str, Any], candidate_name: str,
    objects: Mapping[str, Any], positions: Mapping[str, Any],
    near_radius_m: float = 0.30,
    same_room: bool = False, door_requirements_equal: bool | None = None,
) -> dict[str, Any]:
    """Classify same-category objects using frozen geometry, not policy labels.

    The near rule uses planar centre distance (the benchmark's distance metric).
    It additionally requires the same known room, and either the same immediate
    support/parent or two non-container objects. Same-room expansion is opt-in
    and requires a separate topology audit; equal room IDs alone are not proof.
    """
    if not math.isfinite(near_radius_m) or near_radius_m <= 0:
        raise ValueError("near_radius_m must be finite and positive")
    selected = str(target.get("selected_instance") or "")
    original, candidate = objects.get(selected, {}), objects.get(candidate_name, {})
    decision = {"accepted": False, "reason": "missing_metadata", "candidate_name": candidate_name}
    if not original or not candidate:
        return decision
    if not label(original.get("category")) or label(original.get("category")) != label(candidate.get("category")):
        return dict(decision, reason="different_category")
    if selected == candidate_name:
        return dict(decision, reason="same_selected_instance_not_a_relaxation")
    grounding = target.get("grounding") or {}
    if grounding.get("unique") is True or grounding.get("attributes"):
        return dict(decision, reason="explicit_instance_grounding_requires_review")
    original_room, candidate_room = original.get("room_id"), candidate.get("room_id")
    equal_room = original_room is not None and candidate_room is not None and str(original_room) == str(candidate_room)
    original_ancestors, candidate_ancestors = ancestors(selected, objects), ancestors(candidate_name, objects)
    container = target.get("container_name")
    candidate_containers = [n for n in candidate_ancestors if label(objects.get(n, {}).get("category")) in CONTAINER_CATEGORIES]
    original_containers = [n for n in original_ancestors if label(objects.get(n, {}).get("category")) in CONTAINER_CATEGORIES]
    point, other = positions.get(selected), positions.get(candidate_name)
    distance = None
    if point is not None and other is not None:
        try:
            distance = math.hypot(float(point[0]) - float(other[0]), float(point[1]) - float(other[1]))
            if not math.isfinite(distance):
                distance = None
        except (TypeError, ValueError, IndexError):
            pass
    decision.update(distance_xy_m=distance, target_room_id=original_room, candidate_room_id=candidate_room,
                    target_container=container, candidate_parent=candidate.get("parent"))

    # Parent means support as well as containment in ProcTHOR. Require a point
    # inside the frozen closed-container volume too, excluding its top surface.
    center, size = target.get("container_aabb_center"), target.get("container_aabb_size")
    inside = False
    if container and container in candidate_ancestors and center and size and other:
        try:
            inside = all(abs(float(other[i]) - float(center[i])) < float(size[i]) / 2 for i in range(3))
            inside = inside and float(other[2]) < float(center[2]) + float(size[2]) / 2 - 0.01
        except (TypeError, ValueError, IndexError):
            inside = False
    if inside and equal_room:
        return dict(decision, accepted=True, reason="same_container")
    compatible_support = bool(original.get("parent")) and original.get("parent") == candidate.get("parent")
    ancestry_known = all(n in objects for n in original_ancestors + candidate_ancestors)
    both_outside = ancestry_known and not container and not original_containers and not candidate_containers
    if equal_room and distance is not None and distance <= near_radius_m and (compatible_support or both_outside):
        return dict(decision, accepted=True, reason="near_same_category")
    if equal_room and both_outside:
        if same_room and door_requirements_equal is True:
            return dict(decision, accepted=True, reason="same_room_same_doors")
        return dict(decision, reason="same_room_requires_door_topology_audit")
    return dict(decision, reason="not_equivalent")


def rescore_result(result: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    """Keep interaction facts/eligibility unchanged; update dependent metrics."""
    revised = dict(result)
    eligible = result.get("scoring_eligible") is True and result.get("status") == "complete"
    verified = (result.get("goal_definition_relaxed_success") is True
                and result.get("goal_definition_relaxed_reason") == "verified"
                and bool(result.get("goal_definition_relaxed_instance_id")))
    promoted = bool(eligible and verified and decision.get("accepted") and not result.get("nav_success"))
    revised["goal_equivalence"] = dict(decision, protocol=PROTOCOL, promoted=promoted,
                                      public_claim_verified=verified, scoring_eligible=eligible)
    if not promoted:
        return revised
    revised.update(nav_success=True, task_success=True)
    complete_interactions = bool(result.get("required_interaction_success") and result.get("sequence_success"))
    if result.get("interaction_requirement") == "unnecessary":
        complete_interactions = complete_interactions and result.get("non_interaction_success") is True
    revised.update(success=complete_interactions, interaction_conditioned_success=complete_interactions)
    # Original shortest path is NOT recomputed for the expanded goal set.
    reference, path = result.get("reference_path_length_m"), result.get("navigation_path_length_m")
    revised["spl"] = None
    if reference is not None and path is not None and float(reference) >= 0 and float(path) >= 0:
        denominator = max(float(reference), float(path))
        revised["spl"] = float(reference) / denominator if denominator else 1.0
    breakdown = dict(result.get("episode_total_cost_breakdown") or {})
    if breakdown and result.get("episode_total_cost") is not None:
        revised["episode_total_cost"] = float(result["episode_total_cost"]) - float(breakdown["failure_penalty"])
        breakdown.update(failure_penalty=0.0, nav_success_indicator=1, total_cost=revised["episode_total_cost"])
        revised["episode_total_cost_breakdown"] = breakdown
    return revised
