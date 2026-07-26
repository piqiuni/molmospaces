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


def validate_attribute_patch(value: Any) -> dict[str, Any]:
    result = parse_json_object(value)
    if not str(result.get("object_id") or ""):
        raise ValueError("attribute patch requires object_id")
    result["interactable"] = bool(result.get("interactable", False))
    result["interaction_class"] = str(result.get("interaction_class") or "unknown")
    result["coarse_state"] = str(result.get("coarse_state") or "unknown")
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
    result["confidence"] = _confidence(result.get("confidence"), 0.0)
    result["evidence_frame_ids"] = [str(item) for item in result.get("evidence_frame_ids") or []]
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
