"""Private, post-hoc target identity and interaction-contract scoring.

The policy never receives the metadata consumed here.  The protocol deliberately
keeps exact-instance, category-goal, interaction-contract, and complete
interactive-episode success separate.  It never rewrites the frozen V3 score.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


PROTOCOL = "interactive_nav_goal_equivalence_v2"
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


def _declared_contract_candidates(target: Mapping[str, Any]) -> set[str] | None:
    """Return the pre-frozen candidate contract, when a dataset declares one."""

    identity = target.get("identity_contract")
    identity = identity if isinstance(identity, Mapping) else {}
    raw = identity.get("contract_candidates")
    if raw is None:
        raw = target.get("contract_candidates")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {str(value) for value in raw if str(value)}


def _accept(
    decision: Mapping[str, Any],
    *,
    reason: str,
    contract_accepted: bool,
    contract_reason: str,
) -> dict[str, Any]:
    return {
        **decision,
        "accepted": True,  # compatibility alias: category-goal acceptance
        "reason": reason,
        "category_goal_accepted": True,
        "category_reason": reason,
        "interaction_contract_accepted": bool(contract_accepted),
        "interaction_contract_reason": contract_reason,
    }


def equivalent_target(
    *, target: Mapping[str, Any], candidate_name: str,
    objects: Mapping[str, Any], positions: Mapping[str, Any],
    near_radius_m: float = 0.30,
    same_room: bool = False,
    door_requirements_equal: bool | None = None,
    interaction_requirements_equal: bool | None = None,
) -> dict[str, Any]:
    """Classify one verified candidate without conflating target and task success.

    Proximity and equal room/channel requirements may establish a category-goal
    match.  Neither proves the container/full interaction contract.  Contract
    equivalence needs a frozen declaration or candidate-specific proof that the
    complete necessary interaction requirements are equal.
    """

    if not math.isfinite(near_radius_m) or near_radius_m <= 0:
        raise ValueError("near_radius_m must be finite and positive")
    selected = str(target.get("selected_instance") or "")
    original, candidate = objects.get(selected, {}), objects.get(candidate_name, {})
    decision: dict[str, Any] = {
        "accepted": False,
        "reason": "missing_metadata",
        "category_goal_accepted": False,
        "category_reason": "missing_metadata",
        "interaction_contract_accepted": False,
        "interaction_contract_reason": "category_goal_not_accepted",
        "candidate_name": candidate_name,
    }
    if not original or not candidate:
        return decision
    if not label(original.get("category")) or label(original.get("category")) != label(candidate.get("category")):
        return dict(decision, reason="different_category", category_reason="different_category")
    if selected == candidate_name:
        return dict(decision, reason="same_selected_instance_not_a_relaxation",
                    category_reason="same_selected_instance_not_a_relaxation")
    grounding = target.get("grounding") or {}
    if grounding.get("unique") is True or grounding.get("attributes"):
        return dict(decision, reason="explicit_instance_grounding_requires_review",
                    category_reason="explicit_instance_grounding_requires_review")

    original_room, candidate_room = original.get("room_id"), candidate.get("room_id")
    equal_room = original_room is not None and candidate_room is not None and str(original_room) == str(candidate_room)
    original_ancestors, candidate_ancestors = ancestors(selected, objects), ancestors(candidate_name, objects)
    container = target.get("container_name")
    candidate_containers = [n for n in candidate_ancestors if label(objects.get(n, {}).get("category")) in CONTAINER_CATEGORIES]
    original_containers = [n for n in original_ancestors if label(objects.get(n, {}).get("category")) in CONTAINER_CATEGORIES]
    declared_container_record = objects.get(container, {}) if container else {}
    target_in_declared_container = bool(container and container in original_ancestors)
    candidate_in_declared_container = bool(container and container in candidate_ancestors)
    invalid_declared_target_container = bool(
        container and not target_in_declared_container
    )
    ancestry_known = all(
        n in objects for n in original_ancestors + candidate_ancestors
    )
    # A shared declared container is stronger than a missing room annotation:
    # when both objects resolve to that container, their room relationship is
    # still known.  If only one room annotation is present, it must agree with
    # the container's own room when available.
    declared_container_room = declared_container_record.get("room_id")
    container_room_consistent = bool(
        target_in_declared_container
        and candidate_in_declared_container
        and (
            equal_room
            or (
                (original_room is None or candidate_room is None)
                and (
                    declared_container_room is None
                    or all(
                        room is None
                        or str(room) == str(declared_container_room)
                        for room in (original_room, candidate_room)
                    )
                )
            )
        )
    )
    point, other = positions.get(selected), positions.get(candidate_name)
    distance = None
    if point is not None and other is not None:
        try:
            distance = math.hypot(float(point[0]) - float(other[0]), float(point[1]) - float(other[1]))
            if not math.isfinite(distance):
                distance = None
        except (TypeError, ValueError, IndexError):
            pass
    declared_contract = _declared_contract_candidates(target)
    declared_match = None if declared_contract is None else candidate_name in declared_contract
    decision.update(distance_xy_m=distance, target_room_id=original_room,
                    candidate_room_id=candidate_room, target_container=container,
                    candidate_parent=candidate.get("parent"),
                    declared_contract_candidate=declared_match)

    # ProcTHOR's parent can mean support as well as containment.  Require the
    # point to lie inside the frozen closed-container volume, excluding its top.
    center, size = target.get("container_aabb_center"), target.get("container_aabb_size")
    inside = False
    if container and container in candidate_ancestors and center and size and other:
        try:
            inside = all(abs(float(other[i]) - float(center[i])) < float(size[i]) / 2 for i in range(3))
            inside = inside and float(other[2]) < float(center[2]) + float(size[2]) / 2 - 0.01
        except (TypeError, ValueError, IndexError):
            inside = False
    if (
        inside
        and (equal_room or container_room_consistent)
        and target_in_declared_container
        and candidate_in_declared_container
    ):
        contract_ok = declared_match is True or interaction_requirements_equal is True
        return _accept(
            decision,
            reason="same_container",
            contract_accepted=contract_ok,
            contract_reason=("declared_candidate_contract" if declared_match is True
                             else "excluded_by_declared_candidate_contract" if declared_match is False
                             else "candidate_specific_requirements_equal" if interaction_requirements_equal is True
                             else "same_container_without_candidate_contract_proof"),
        )

    # A candidate explicitly attached to the target container but outside its
    # frozen volume is inconsistent metadata, not a valid same-room fallback.
    # Keep this distinction so a malformed height/pose cannot be promoted by
    # the broad Category rule below.
    invalid_declared_container_geometry = bool(
        container
        and target_in_declared_container
        and candidate_in_declared_container
        and not inside
    )
    compatible_support = bool(original.get("parent")) and original.get("parent") == candidate.get("parent")
    both_outside = ancestry_known and not container and not original_containers and not candidate_containers
    if (
        not invalid_declared_target_container
        and equal_room
        and distance is not None
        and distance <= near_radius_m
        and (compatible_support or both_outside)
        and not invalid_declared_container_geometry
    ):
        contract_ok = declared_match is True or interaction_requirements_equal is True
        return _accept(
            decision,
            reason="near_same_category",
            contract_accepted=contract_ok,
            contract_reason=("declared_candidate_contract" if declared_match is True
                             else "candidate_specific_requirements_equal" if interaction_requirements_equal is True
                             else "proximity_does_not_prove_interaction_contract"),
        )
    # A category-level object goal is allowed to match another same-category
    # object in the same observed room, including the common case where the
    # selected target is in a container but the found object is on a surface.
    # This is intentionally only the Category layer: a different container (or
    # an unresolvable parent chain) remains a distinct interaction contract.
    different_known_containers = bool(
        ancestry_known
        and original_containers
        and candidate_containers
        and set(original_containers).isdisjoint(candidate_containers)
    )
    if (
        equal_room
        and ancestry_known
        and not different_known_containers
        and not invalid_declared_container_geometry
        and not invalid_declared_target_container
    ):
        contract_ok = declared_match is True or interaction_requirements_equal is True
        same_door_shape = bool(
            both_outside and same_room and door_requirements_equal is True
        )
        return _accept(
            decision,
            reason="same_room_same_doors" if same_door_shape else "same_room_category",
            contract_accepted=contract_ok,
            contract_reason=(
                "declared_candidate_contract"
                if declared_match is True
                else "candidate_specific_requirements_equal"
                if interaction_requirements_equal is True
                else "same_room_category_does_not_prove_interaction_contract"
                if not same_door_shape
                else "shared_channel_does_not_prove_full_contract"
            ),
        )
    return dict(decision, reason="not_equivalent", category_reason="not_equivalent")


def classify_category_candidates(
    *,
    target: Mapping[str, Any],
    candidate_names: list[str] | tuple[str, ...] | set[str],
    objects: Mapping[str, Any],
    positions: Mapping[str, Any],
    near_radius_m: float = 0.30,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Apply the same evaluator-private relation gate to category candidates.

    The live ROS evaluator and the post-hoc scorer both call this helper.  The
    selected instance is retained as the strict endpoint; every other name must
    pass :func:`equivalent_target` before it can enter the public-evidence
    ledger.  Returning the per-candidate decisions is useful for private
    diagnostics, while callers remain responsible for keeping the metadata out
    of the policy-facing trace.
    """

    selected = str(target.get("selected_instance") or "").strip()
    ordered = list(dict.fromkeys(
        str(name).strip() for name in candidate_names if str(name).strip()
    ))
    if selected and selected not in ordered:
        ordered.insert(0, selected)
    accepted: list[str] = []
    decisions: dict[str, dict[str, Any]] = {}
    for candidate_name in ordered:
        if candidate_name == selected:
            decision = {
                "accepted": True,
                "reason": "strict_selected_instance",
                "category_goal_accepted": True,
                "category_reason": "strict_selected_instance",
                "interaction_contract_accepted": True,
                "interaction_contract_reason": "strict_selected_instance",
                "candidate_name": candidate_name,
            }
        else:
            decision = equivalent_target(
                target=target,
                candidate_name=candidate_name,
                objects=objects,
                positions=positions,
                near_radius_m=near_radius_m,
            )
        decisions[candidate_name] = decision
        if decision.get("category_goal_accepted") is True:
            accepted.append(candidate_name)
    return accepted, decisions


def rescore_result(result: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    """Append layered scores while preserving every frozen V3 score field."""

    revised = dict(result)
    eligible = result.get("scoring_eligible") is True and result.get("status") == "complete"
    verified = (
        result.get("goal_definition_relaxed_success") is True
        and result.get("goal_definition_relaxed_reason") in {
            "verified", "verified_open_container_anchor"
        }
        and bool(result.get("goal_definition_relaxed_instance_id"))
    )
    exact = bool(eligible and result.get("nav_success") is True)
    category_promoted = bool(
        eligible and verified
        and decision.get("category_goal_accepted", decision.get("accepted")) is True
        and not exact
    )
    category = bool(exact or category_promoted)
    contract_promoted = bool(
        category_promoted and decision.get("interaction_contract_accepted") is True
    )
    contract = bool(exact or contract_promoted)

    interaction_complete = bool(
        result.get("required_interaction_success") is True
        and result.get("sequence_success") is True
    )
    if result.get("interaction_requirement") == "unnecessary":
        interaction_complete = bool(
            interaction_complete and result.get("non_interaction_success") is True
        )
    interactive = bool(contract and interaction_complete)
    original_interactive = bool(
        eligible and result.get("interaction_conditioned_success", result.get("success")) is True
    )

    revised.update(
        exact_instance_success=exact,
        category_goal_success=category,
        interaction_contract_goal_success=contract,
        interactive_episode_success=interactive,
        category_goal_instance_id=(
            result.get("goal_definition_relaxed_instance_id") if category_promoted else None
        ),
    )
    revised["goal_success_layers"] = {
        "protocol": PROTOCOL,
        "exact_instance_success": exact,
        "category_goal_success": category,
        "interaction_contract_goal_success": contract,
        "interactive_episode_success": interactive,
        "interaction_plan_complete": interaction_complete,
        "strict_metrics_preserved": True,
    }
    revised["goal_equivalence"] = {
        **decision,
        "protocol": PROTOCOL,
        "promoted": category_promoted,
        "category_promoted": category_promoted,
        "interaction_contract_promoted": contract_promoted,
        "interactive_episode_promoted": bool(interactive and not original_interactive),
        "public_claim_verified": verified,
        "scoring_eligible": eligible,
        "strict_metrics_preserved": True,
        "category_spl": None,
        "category_spl_reason": "candidate_specific_reference_path_not_available",
    }
    return revised
