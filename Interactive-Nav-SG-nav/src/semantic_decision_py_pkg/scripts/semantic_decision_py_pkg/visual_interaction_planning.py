from __future__ import annotations


_DRAWER_NAME_TOKENS = (
    "drawer",
    "dresser",
    "chestofdrawers",
    "chest_of_drawers",
    "chest of drawers",
)


def infer_visual_interaction_target_type(candidate: dict, node: dict) -> str:
    metadata = candidate.get("metadata") or {}
    attributes = node.get("attributes") or {}
    tokens = " ".join(
        str(value or "")
        for value in (
            candidate.get("interaction_type"),
            metadata.get("node_type"),
            node.get("type"),
            node.get("name"),
            node.get("label"),
            attributes.get("category"),
            attributes.get("semantic_name"),
            candidate.get("target_name"),
        )
    ).casefold()
    if "portal" in tokens or "door" in tokens or "gate" in tokens:
        return "door"
    if any(token in tokens for token in _DRAWER_NAME_TOKENS):
        return "drawer_container"
    if any(token in tokens for token in ("container", "cabinet", "fridge", "refrigerator")):
        return "other_container"
    return "unknown"


def candidate_with_visual_operation_plan(candidate: dict, plan: dict) -> dict:
    planned = dict(candidate)
    interaction = dict(planned.get("interaction_command") or {})
    interaction["visual_operation_plan"] = dict(plan)
    interaction["operation_method"] = str(plan.get("operation_method") or "unknown")
    interaction["open_regions"] = list(plan.get("open_regions") or [])
    for simulator_only_key in (
        "interaction_group_id",
        "interaction_groups",
        "joint_names",
        "close_other_joint_names",
        "close_other_joints",
        "part_ids",
        "open_fraction_threshold",
    ):
        interaction.pop(simulator_only_key, None)
    if str(plan.get("target_type") or "") == "drawer_container":
        interaction.update(
            {
                "action": "scan",
                "interaction_mode": "drawer_scan",
                "sequence_type": "drawer_scan",
            }
        )
    else:
        interaction["action"] = str(
            plan.get("action") or interaction.get("action") or "open"
        )
    planned["interaction_command"] = interaction
    return planned


def action_for_opaque_open_contract(action: object, *, enabled: bool) -> str:
    """Project an action onto an evaluator's opaque-object ``open`` API.

    V3 keeps part and joint resolution private, so it accepts only
    ``open(opaque_object_id)``.  Normal ROS simulation retains its richer action
    vocabulary unless this explicit adapter is enabled.
    """

    normalized = str(action or "open").casefold()
    return "open" if enabled else normalized
