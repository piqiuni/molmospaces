from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError("model response must be a JSON object")
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response does not contain a JSON object")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def _confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalized_token(value: Any) -> str:
    """Normalize short categorical MLLM fields before graph routing."""

    return re.sub(r"[\s-]+", "_", str(value or "").strip().casefold())


def _interaction_class(value: Any) -> str:
    token = _normalized_token(value)
    return {
        "door": "portal",
        "doorway": "portal",
        "gate": "portal",
        "barrier": "portal",
        "portal": "portal",
        "fridge": "container",
        "refrigerator": "container",
        "cabinet": "container",
        "drawer": "container",
        "container": "container",
        "none": "none",
        "non_interactable": "none",
        "not_interactable": "none",
        "unknown": "unknown",
    }.get(token, "unknown")


def _coarse_interaction_state(value: Any) -> str:
    """Accept ordinary model casing/phrasing without widening planner states."""

    token = _normalized_token(value)
    if token in {"open", "opened", "fully_open", "wide_open", "open_state"}:
        return "open"
    if token in {"closed", "close", "shut", "fully_closed", "closed_state"}:
        return "closed"
    if token in {"ajar", "partially_open", "half_open", "part_open"}:
        return "ajar"
    if token in {"static_open", "fixed_open", "always_open"}:
        return "static_open"
    return "unknown"


def _portal_morphology(value: Any) -> dict[str, Any]:
    """Normalize a purely visual door-leaf observation.

    This is intentionally morphology only: an RGB model may identify whether
    a leaf is visible, but it must not claim world connectivity or simulator
    articulation.  The latter is supplied independently by occupancy/room
    observations before a fixed passage is accepted.
    """

    raw = value if isinstance(value, dict) else {}
    leaf = _normalized_token(raw.get("door_leaf") or raw.get("leaf") or "unknown")
    if leaf in {"absent", "none", "missing", "no_leaf", "no_door"}:
        leaf = "absent"
    elif leaf in {"present", "leaf", "door", "door_leaf", "visible"}:
        leaf = "present"
    else:
        leaf = "unknown"
    return {
        "door_leaf": leaf,
        "confidence": _confidence(raw.get("confidence"), 0.0),
    }


def _portal_aperture_evidence(value: Any) -> dict[str, Any]:
    """Normalize visual evidence that an opening is actually visible.

    This remains an image-only statement.  It cannot establish map
    connectivity, joint capability, or a hidden door state; graph policy uses
    it only to decide whether an MLLM ``open``/``ajar`` claim is admissible.
    """

    raw = value if isinstance(value, dict) else {}
    aperture = _normalized_token(
        raw.get("open_aperture")
        or raw.get("aperture")
        or raw.get("opening")
        or "unknown"
    )
    if raw.get("opening_visible") is True or raw.get("aperture_open") is True:
        aperture = "visible"
    elif raw.get("opening_visible") is False or raw.get("aperture_open") is False:
        aperture = "not_visible"
    if aperture in {"visible", "open", "opening_visible", "clear_gap", "gap"}:
        aperture = "visible"
    elif aperture in {
        "not_visible",
        "closed",
        "occluded",
        "no_gap",
        "not_open",
    }:
        aperture = "not_visible"
    else:
        aperture = "unknown"
    return {
        "open_aperture": aperture,
        "confidence": _confidence(raw.get("confidence"), 0.0),
    }


def build_attribute_patch_response_schema(
    object_id: str, *, expected_node_type: str | None = None
) -> dict[str, Any]:
    """Return the strict wire schema for one Module-1 object observation.

    The portal-only fields remain semantically optional by being nullable.  A
    strict OpenAI-compatible schema requires each top-level property to be
    present, so non-portal observations must emit ``null`` for those fields
    instead of inventing door evidence.  This keeps the payload bounded while
    avoiding conditional/``oneOf`` schemas that are less portable across local
    OpenAI-compatible servers.
    """

    normalized_object_id = str(object_id or "").strip()
    if not normalized_object_id:
        raise ValueError("attribute response schema requires an object_id")
    normalized_expected_type = str(expected_node_type or "").strip().casefold()
    if normalized_expected_type not in {"", "container", "portal"}:
        raise ValueError(
            "attribute response schema expected_node_type must be container or portal"
        )
    container_only = normalized_expected_type == "container"
    confidence = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    nullable_portal_morphology = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "door_leaf": {
                "type": "string",
                "enum": ["absent", "present", "unknown"],
            },
            "confidence": confidence,
        },
        "required": ["door_leaf", "confidence"],
    }
    nullable_portal_aperture_evidence = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "open_aperture": {
                "type": "string",
                "enum": ["visible", "not_visible", "unknown"],
            },
            "confidence": confidence,
        },
        "required": ["open_aperture", "confidence"],
    }
    return {
        "name": "attribute_inference",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "object_id": {
                    "type": "string",
                    "enum": [normalized_object_id],
                },
                "interactable": {"type": "boolean"},
                "interaction_class": {
                    "type": "string",
                    "enum": (
                        ["container"]
                        if container_only
                        else ["portal", "container", "none", "unknown"]
                    ),
                },
                "coarse_state": {
                    "type": "string",
                    "enum": (
                        ["open", "closed", "ajar", "unknown"]
                        if container_only
                        else ["open", "closed", "ajar", "static_open", "unknown"]
                    ),
                },
                "portal_morphology": (
                    {"type": "null"}
                    if container_only
                    else nullable_portal_morphology
                ),
                "portal_aperture_evidence": (
                    {"type": "null"}
                    if container_only
                    else nullable_portal_aperture_evidence
                ),
                # These fields belong to Module 1's pre-interaction visual
                # observation.  They deliberately describe only what is
                # visible from the current camera pose; no world-space axis or
                # simulator geometry is requested from the model.
                "view_state": {
                    "type": "string",
                    "enum": ["front", "oblique", "side_or_back", "occluded", "unknown"],
                },
                "view_state_confidence": confidence,
                "front_surface_visible": {"type": "boolean"},
                "front_surface_confidence": confidence,
                "approach_ready": {"type": "boolean"},
                "needs_reobserve": {"type": "boolean"},
                # These are visual points on the padded target-crop inset in
                # the single M1 composite image.  They are deliberately
                # image-relative: downstream code may use them to ground a
                # visible drawer scan, but never to infer simulator joints or
                # hidden drawer locations.
                "action_regions": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "center": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "confidence": confidence,
                        },
                        "required": ["center", "confidence"],
                    },
                },
                "interaction_parts": {
                    "type": "array",
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "part_id": {"type": "string", "maxLength": 32},
                            "type": {"type": "string", "maxLength": 32},
                            "state": {"type": "string", "maxLength": 32},
                            "handle_visible": {"type": "boolean"},
                            "confidence": confidence,
                        },
                        "required": [
                            "part_id",
                            "type",
                            "state",
                            "handle_visible",
                            "confidence",
                        ],
                    },
                },
                "confidence": confidence,
            },
            "required": [
                "object_id",
                "interactable",
                "interaction_class",
                "coarse_state",
                "portal_morphology",
                "portal_aperture_evidence",
                "view_state",
                "view_state_confidence",
                "front_surface_visible",
                "front_surface_confidence",
                "approach_ready",
                "needs_reobserve",
                "action_regions",
                "interaction_parts",
                "confidence",
            ],
        },
    }


def validate_attribute_patch(value: Any) -> dict[str, Any]:
    result = parse_json_object(value)
    if not str(result.get("object_id") or ""):
        raise ValueError("attribute patch requires object_id")
    result["selected_view_id"] = str(
        result.get("selected_view_id") or "view_1"
    ).strip() or "view_1"
    result["interactable"] = bool(result.get("interactable", False))
    result["interaction_class"] = _interaction_class(result.get("interaction_class"))
    result["coarse_state"] = _coarse_interaction_state(result.get("coarse_state"))
    if result["interaction_class"] == "portal":
        result["portal_morphology"] = _portal_morphology(
            result.get("portal_morphology")
        )
        result["portal_aperture_evidence"] = _portal_aperture_evidence(
            result.get("portal_aperture_evidence")
        )
    else:
        result.pop("portal_morphology", None)
        result.pop("portal_aperture_evidence", None)
    raw_action_regions = result.get("action_regions") or []
    if not isinstance(raw_action_regions, list):
        raise ValueError("attribute patch action_regions must be a list")
    action_regions: list[dict[str, Any]] = []
    for raw_region in raw_action_regions:
        center = _normalized_center(raw_region)
        if center is None:
            continue
        if any(
            (center[0] - item["center"][0]) ** 2
            + (center[1] - item["center"][1]) ** 2
            < 0.0016
            for item in action_regions
        ):
            continue
        confidence = (
            _confidence(raw_region.get("confidence"), _confidence(result.get("confidence")))
            if isinstance(raw_region, dict)
            else _confidence(result.get("confidence"))
        )
        action_regions.append({"center": center, "confidence": confidence})
        if len(action_regions) >= 8:
            break
    action_regions.sort(key=lambda item: (item["center"][1], item["center"][0]))

    parts = result.get("interaction_parts") or []
    if not isinstance(parts, list):
        raise ValueError("interaction_parts must be a list")
    normalized_parts = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        normalized_parts.append(
            {
                "part_id": str(part.get("part_id") or f"part_{index}"),
                "type": str(part.get("type") or "unknown"),
                "state": str(part.get("state") or "unknown"),
                "handle_visible": bool(part.get("handle_visible", False)),
                "confidence": _confidence(part.get("confidence"), 0.0),
            }
        )
    result["interaction_parts"] = normalized_parts
    part_confidence = max(
        (float(part.get("confidence", 0.0) or 0.0) for part in normalized_parts),
        default=0.0,
    )
    result["confidence"] = _confidence(
        result.get("confidence"),
        part_confidence,
    )
    view_state = _view_state(result.get("view_state"))
    view_state_confidence = _confidence(
        result.get("view_state_confidence"), result["confidence"]
    )
    front_surface_visible = _boolean(
        result.get("front_surface_visible"), default=False
    )
    front_surface_confidence = _confidence(
        result.get("front_surface_confidence"), view_state_confidence
    )
    approach_ready = _boolean(result.get("approach_ready"), default=False)
    needs_reobserve = _boolean(
        result.get("needs_reobserve"), default=not approach_ready
    )
    # A front-facing interaction pose is a safety-critical visual claim.  Be
    # conservative if the model reports a side/occluded/unknown view, omits a
    # visible front surface, or asks for another observation.  The planner may
    # derive a world-space approach pose later from this image observation and
    # calibrated robot geometry; Module 1 never authors that pose directly.
    if (
        view_state in {"side_or_back", "occluded", "unknown"}
        or not front_surface_visible
        or needs_reobserve
    ):
        approach_ready = False
        needs_reobserve = True
        action_regions = []
    result.update(
        {
            "view_state": view_state,
            "view_state_confidence": view_state_confidence,
            "front_surface_visible": front_surface_visible,
            "front_surface_confidence": front_surface_confidence,
            "approach_ready": approach_ready,
            "needs_reobserve": needs_reobserve,
            "action_regions": action_regions,
        }
    )
    result["evidence_frame_ids"] = [str(item) for item in result.get("evidence_frame_ids") or []]
    return result


def validate_room_attribute_patch(value: Any) -> dict[str, Any]:
    """Normalize a room-level Module-1 response.

    Room inference deliberately receives only a room identifier and its visible
    object evidence.  Keep the result separate from an object attribute patch
    so a delayed room result can never be routed onto an interaction node.
    """

    result = parse_json_object(value)
    room_id = result.get("room_id")
    if room_id is None or str(room_id).strip() == "":
        raise ValueError("room attribute patch requires room_id")
    try:
        result["room_id"] = int(room_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("room attribute patch room_id must be an integer") from exc
    room_attribute = str(
        result.get("room_attribute") or result.get("scene_attribute") or "unknown"
    ).strip()
    result["room_attribute"] = room_attribute.casefold() if room_attribute else "unknown"
    result["confidence"] = _confidence(result.get("confidence"), 0.0)
    evidence_object_ids = result.get("evidence_object_ids") or []
    if not isinstance(evidence_object_ids, list):
        raise ValueError("evidence_object_ids must be a list")
    result["evidence_object_ids"] = [str(item) for item in evidence_object_ids if str(item)]
    return result


def validate_subgoal_selection(value: Any, candidate_ids: set[str]) -> dict[str, Any]:
    result = parse_json_object(value)
    ranked_ids = result.get("ranked_ids") or []
    if not isinstance(ranked_ids, list):
        raise ValueError("ranked_ids must be a list")
    if not ranked_ids and result.get("candidate_id"):
        ranked_ids = [result.get("candidate_id")]
    normalized_ranked_ids = []
    for value in ranked_ids:
        candidate_id = str(value or "")
        if candidate_id not in candidate_ids:
            raise ValueError(f"model selected unknown candidate: {candidate_id}")
        if candidate_id not in normalized_ranked_ids:
            normalized_ranked_ids.append(candidate_id)
    if not normalized_ranked_ids:
        raise ValueError("model response requires at least one ranked candidate")
    candidate_id = normalized_ranked_ids[0]
    scores = result.get("scores") or {}
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object")
    result["candidate_id"] = candidate_id
    result["ranked_ids"] = normalized_ranked_ids[:3]
    result["scores"] = {str(key): float(score) for key, score in scores.items()}
    result["reason"] = str(result.get("reason") or "NO_SEMANTIC_PREFERENCE").upper()
    confidence = str(result.get("confidence") or "medium").casefold()
    result["confidence"] = confidence if confidence in {"low", "medium", "high"} else "medium"
    return result


def validate_skill_plan(value: Any, object_id: str) -> dict[str, Any]:
    result = parse_json_object(value)
    if str(result.get("object_id") or object_id) != object_id:
        raise ValueError("skill plan object_id does not match selected object")
    subactions = result.get("subactions") or []
    if not isinstance(subactions, list) or not subactions:
        raise ValueError("skill plan requires at least one subaction")
    normalized = []
    for action in subactions:
        if not isinstance(action, dict) or not str(action.get("skill") or ""):
            raise ValueError("each subaction requires a skill")
        normalized.append(
            {
                "skill": str(action["skill"]),
                "part_id": str(action.get("part_id") or ""),
                "desired_state": str(action.get("desired_state") or ""),
                "view_profile": str(action.get("view_profile") or "default"),
            }
        )
    result["object_id"] = object_id
    result["subactions"] = normalized
    result["max_retries"] = max(0, min(3, int(result.get("max_retries", 1))))
    return result


def validate_skill_action(
    value: Any, allowed_part_ids: set[str], requested_action: str = "open"
) -> dict[str, str]:
    """Validate the single atomic action supported by the current interaction backend."""
    result = parse_json_object(value)
    action = str(result.get("action") or requested_action).casefold()
    if action not in {"open", "close"}:
        raise ValueError("skill action must be open or close")
    part_id = str(result.get("part_id") or "")
    if part_id and part_id not in allowed_part_ids:
        raise ValueError("skill action selected an unknown part_id")
    return {"action": action, "part_id": part_id}


_TARGET_TYPE_ALIASES = {
    "door": "door",
    "portal": "door",
    "drawer": "drawer_container",
    "drawers": "drawer_container",
    "dresser": "drawer_container",
    "chest_of_drawers": "drawer_container",
    "drawer_container": "drawer_container",
    "cabinet": "other_container",
    "fridge": "other_container",
    "refrigerator": "other_container",
    "container": "other_container",
    "other_container": "other_container",
    "unknown": "unknown",
}

_OPERATION_METHOD_ALIASES = {
    "push": "hinged_push",
    "hinged_push": "hinged_push",
    "pull": "pull",
    "hinged_pull": "hinged_pull",
    "hinged": "hinged_unknown",
    "hinged_unknown": "hinged_unknown",
    "hinged_unknown_direction": "hinged_unknown",
    "double": "double_hinged",
    "double_door": "double_hinged",
    "double_hinged": "double_hinged",
    "slide_left": "slide_left",
    "sliding_left": "slide_left",
    "slide_right": "slide_right",
    "sliding_right": "slide_right",
    "sliding": "unknown",
    "unknown": "unknown",
}

_VIEW_STATE_ALIASES = {
    "front": "front",
    "front_facing": "front",
    "frontal": "front",
    "oblique": "oblique",
    "angled": "oblique",
    "side": "side_or_back",
    "side_on": "side_or_back",
    "side_or_back": "side_or_back",
    "rear": "side_or_back",
    "back": "side_or_back",
    "occluded": "occluded",
    "blocked": "occluded",
    "unknown": "unknown",
}


def _view_state(value: Any) -> str:
    return _VIEW_STATE_ALIASES.get(
        str(value or "unknown").strip().casefold(), "unknown"
    )


def _boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = _normalized_token(value)
    if token in {"true", "yes", "ready", "required"}:
        return True
    if token in {"false", "no", "not_ready", "none", ""}:
        return False
    return bool(default)


def _normalized_center(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if "center" in value:
            value = value["center"]
        elif "point" in value:
            value = value["point"]
        elif "x" in value and "y" in value:
            value = [value["x"], value["y"]]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return [round(x, 4), round(y, 4)]


def validate_visual_interaction_plan(
    value: Any,
    *,
    expected_target_type: str = "unknown",
    requested_action: str = "open",
    max_regions: int = 12,
) -> dict[str, Any]:
    """Normalize a post-arrival visual operation plan without exposing simulator joints."""
    result = parse_json_object(value)
    raw_target_type = str(result.get("target_type") or expected_target_type or "unknown")
    target_type = _TARGET_TYPE_ALIASES.get(raw_target_type.strip().casefold(), "unknown")
    expected = _TARGET_TYPE_ALIASES.get(
        str(expected_target_type or "unknown").strip().casefold(), "unknown"
    )
    if expected == "door" and target_type not in {"door", "unknown"}:
        raise ValueError("visual interaction plan target_type conflicts with door target")
    if expected == "drawer_container" and target_type not in {
        "drawer_container",
        "unknown",
    }:
        raise ValueError("visual interaction plan target_type conflicts with drawer target")
    if expected in {"drawer_container", "other_container"} and target_type == "door":
        raise ValueError("visual interaction plan target_type conflicts with container target")
    if target_type == "unknown" and expected != "unknown":
        target_type = expected

    action = str(result.get("action") or requested_action or "open").strip().casefold()
    if action not in {"open", "scan"}:
        raise ValueError("visual interaction plan action must be open or scan")
    if target_type == "drawer_container":
        action = "scan"

    raw_method = str(result.get("operation_method") or "unknown").strip().casefold()
    operation_method = _OPERATION_METHOD_ALIASES.get(raw_method, "unknown")
    if target_type == "drawer_container":
        operation_method = "pull"
    elif target_type == "door" and operation_method == "pull":
        operation_method = "hinged_pull"

    raw_view_state = str(result.get("view_state") or "unknown").strip().casefold()
    view_state = _VIEW_STATE_ALIASES.get(raw_view_state, "unknown")
    # A container must explicitly establish a usable front-facing view before
    # the backend receives an open command.  This makes older/partial M3
    # replies conservative rather than silently retaining the radial-side
    # behavior that caused side-view refrigerator interactions.
    container_target = expected == "other_container" or target_type == "other_container"
    approach_ready = _boolean(
        result.get("approach_ready"),
        default=not container_target,
    )
    reposition_required = _boolean(
        result.get("reposition_required"),
        default=container_target and not approach_ready,
    )
    if container_target and (
        not approach_ready
        or reposition_required
        or view_state in {"side_or_back", "occluded", "unknown"}
    ):
        approach_ready = False
        reposition_required = True
        operation_method = "unknown"

    raw_regions = (
        result.get("open_regions")
        or result.get("interaction_points")
        or result.get("centers")
        or []
    )
    if not isinstance(raw_regions, list):
        raise ValueError("visual interaction plan open_regions must be a list")
    normalized_regions: list[dict[str, Any]] = []
    for raw_region in raw_regions:
        center = _normalized_center(raw_region)
        if center is None:
            continue
        if any(
            (center[0] - item["center"][0]) ** 2
            + (center[1] - item["center"][1]) ** 2
            < 0.0016
            for item in normalized_regions
        ):
            continue
        confidence = (
            _confidence(raw_region.get("confidence"), _confidence(result.get("confidence")))
            if isinstance(raw_region, dict)
            else _confidence(result.get("confidence"))
        )
        normalized_regions.append({"center": center, "confidence": confidence})
        if len(normalized_regions) >= max(1, int(max_regions)):
            break
    normalized_regions.sort(key=lambda item: (item["center"][1], item["center"][0]))
    if container_target and not approach_ready:
        normalized_regions = []

    return {
        "target_type": target_type,
        "action": action,
        "operation_method": operation_method,
        "open_regions": normalized_regions,
        "confidence": _confidence(result.get("confidence"), 0.0),
        "reason": str(result.get("reason") or "")[:160],
        "view_state": view_state,
        "approach_ready": approach_ready,
        "reposition_required": reposition_required,
        "coordinate_frame": "normalized_target_crop",
    }


def build_visual_interaction_plan_response_schema() -> dict[str, Any]:
    """Return the strict wire schema shared by the post-arrival M3 call.

    It purposefully contains no simulator-specific fields.  The executor still
    validates semantic consistency against the selected graph node after the
    response is decoded.
    """

    return {
        "name": "visual_interaction_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_type": {
                    "type": "string",
                    "enum": ["door", "drawer_container", "other_container", "unknown"],
                },
                "action": {"type": "string", "enum": ["open", "scan"]},
                "operation_method": {
                    "type": "string",
                    "enum": [
                        "hinged_push",
                        "hinged_pull",
                        "hinged_unknown",
                        "double_hinged",
                        "slide_left",
                        "slide_right",
                        "pull",
                        "unknown",
                    ],
                },
                "view_state": {
                    "type": "string",
                    "enum": ["front", "oblique", "side_or_back", "occluded", "unknown"],
                },
                "approach_ready": {"type": "boolean"},
                "reposition_required": {"type": "boolean"},
                "open_regions": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "center": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["center", "confidence"],
                    },
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string", "maxLength": 160},
            },
            "required": [
                "target_type",
                "action",
                "operation_method",
                "view_state",
                "approach_ready",
                "reposition_required",
                "open_regions",
                "confidence",
                "reason",
            ],
        },
    }


def build_visual_verification_response_schema() -> dict[str, Any]:
    """Return the compact strict wire schema for post-action visual feedback.

    Verification is consumed only as a success/retry signal, so its evidence
    must stay bounded.  In particular, do not leave ``observed_states`` as an
    open-ended model-authored object: a verbose explanation previously consumed
    the small completion budget and could truncate the enclosing JSON object.
    """

    confidence = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    return {
        "name": "visual_verification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "success": {"type": "boolean"},
                "confidence": confidence,
                "reason": {"type": "string", "maxLength": 96},
                "observed_states": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_state": {
                            "type": "string",
                            "enum": ["open", "closed", "ajar", "unchanged", "unknown"],
                        },
                        "visible_change": {
                            "type": "string",
                            "enum": ["yes", "no", "unknown"],
                        },
                    },
                    "required": ["target_state", "visible_change"],
                },
                "new_contents_visible": {"type": "boolean"},
                "retry_action": {
                    "type": "string",
                    "enum": ["none", "retry", "reposition", "rescan"],
                },
            },
            "required": [
                "success",
                "confidence",
                "reason",
                "observed_states",
                "new_contents_visible",
                "retry_action",
            ],
        },
    }


def validate_visual_verification(value: Any) -> dict[str, Any]:
    result = parse_json_object(value)
    result["success"] = bool(result.get("success", False))
    result["confidence"] = _confidence(result.get("confidence"), 0.0)
    observed_states = result.get("observed_states") or {}
    if isinstance(observed_states, dict):
        result["observed_states"] = dict(observed_states)
    elif isinstance(observed_states, list):
        result["observed_states"] = {
            f"observation_{index}": item for index, item in enumerate(observed_states)
        }
    else:
        result["observed_states"] = {"summary": str(observed_states)}
    result["new_contents_visible"] = bool(result.get("new_contents_visible", False))
    result["retry_action"] = str(result.get("retry_action") or "none")
    result["reason"] = str(result.get("reason") or "")
    return result
