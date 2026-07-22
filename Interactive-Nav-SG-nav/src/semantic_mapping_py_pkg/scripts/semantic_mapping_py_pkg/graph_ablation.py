from __future__ import annotations

import copy
from typing import Any


def apply_module1_ablation(graph: dict[str, Any], mode: str) -> dict[str, Any]:
    normalized = str(mode or "dynamic_rule").casefold()
    if normalized != "static_semantic":
        result = copy.deepcopy(graph)
        result["module1_mode"] = normalized
        return result

    result = copy.deepcopy(graph)
    result["source_mode"] = "static_semantic_ablation"
    result["module1_mode"] = normalized
    for node in result.get("nodes") or []:
        node_type = str(node.get("type") or "")
        attributes = node.get("attributes") or {}
        for key in (
            "interaction_state_override",
            "observation_evidence",
            "visible_pixels",
            "visible_fraction",
            "consecutive_observations",
            "frame_index",
        ):
            attributes.pop(key, None)
        node["attributes"] = attributes
        node["is_currently_visible"] = False
        node["state_age_sec"] = 0.0
        if node_type not in {"portal", "container"}:
            node["interaction"] = {}
            continue
        previous = node.get("interaction") or {}
        is_portal = node_type == "portal"
        node["interaction"] = {
            "is_interactable": True,
            "interaction_mode": str(previous.get("interaction_mode") or "open_close"),
            "state": "unknown",
            "state_source": "static_semantic_ablation",
            "state_confidence": float(node.get("confidence", 0.0) or 0.0),
            "requires_interaction": True,
            "traversable": None,
            "interaction_cost": float(previous.get("interaction_cost", 1.0) or 1.0),
            "expected_effect": "unlock_connectivity" if is_portal else "reveal_contents",
            "operation_history": [],
        }
    return result
