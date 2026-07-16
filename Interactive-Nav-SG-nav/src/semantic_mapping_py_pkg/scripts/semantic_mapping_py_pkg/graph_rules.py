from __future__ import annotations

import math
from typing import Any

from .geometry_utils import normalize_label

PORTAL_LABELS = {
    "door",
    "doorway",
    "gate",
    "entrance",
}
SUPPORT_LABELS = {
    "table",
    "desk",
    "bed",
    "sofa",
    "couch",
    "nightstand",
    "countertop",
    "counter",
    "shelf",
    "rack",
    "bench",
    "cabinet_top",
}
CONTAINER_LABELS = {
    "fridge",
    "refrigerator",
    "cabinet",
    "drawer",
    "wardrobe",
    "closet",
    "cupboard",
    "dresser",
    "chest_of_drawers",
    "microwave",
    "dishwasher",
    "box",
    "storage_bin",
}

HINGE_NAMES = {"hinge", "mjjnthinge"}
SLIDE_NAMES = {"slide", "mjJNT_SLIDE", "mjjntslide"}


def sanitize_token(value: str) -> str:
    text = normalize_label(value)
    return text.replace("/", "_").replace("|", "_").replace(":", "_")


def point3(values=None):
    if isinstance(values, dict):
        return [
            float(values.get("x", 0.0)),
            float(values.get("y", 0.0)),
            float(values.get("z", 0.0)),
        ]
    vals = list(values or [])
    if len(vals) < 3:
        vals.extend([0.0] * (3 - len(vals)))
    return [float(vals[0]), float(vals[1]), float(vals[2])]


def normalize_joint_type(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    if not text:
        return "none"
    lowered = text.lower()
    if lowered in HINGE_NAMES or "hinge" in lowered:
        return "hinge"
    if lowered in {name.lower() for name in SLIDE_NAMES} or "slide" in lowered:
        return "slide"
    return "none"


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    semantic_name = normalize_label(
        observation.get("semantic_name") or observation.get("semantic_class") or observation.get("class")
    )
    category = str(observation.get("category") or semantic_name or "object")
    position = point3(observation.get("position") or observation.get("coord") or observation.get("centroid"))
    aabb_center = point3(observation.get("aabb_center") or observation.get("box3d_center") or position)
    aabb_size = point3(observation.get("aabb_size") or observation.get("size") or observation.get("box3d_size"))
    connected_room_ids = observation.get("connected_room_ids") or []
    room_id = observation.get("room_id")
    if room_id is not None:
        try:
            room_id = int(room_id)
        except (TypeError, ValueError):
            room_id = None
    viz_aabb_center = observation.get("viz_aabb_center") or observation.get("world_box3d_center") or observation.get("aabb_center")
    viz_aabb_size = observation.get("viz_aabb_size") or observation.get("world_box3d_size") or observation.get("aabb_size") or observation.get("size")
    return {
        "observation_id": str(observation.get("observation_id") or ""),
        "instance_id": str(observation.get("instance_id") or ""),
        "semantic_name": semantic_name or "object",
        "category": category,
        "candidate_labels": list(observation.get("candidate_labels") or []),
        "label_votes": dict(observation.get("label_votes") or {}),
        "confidence": float(observation.get("confidence", observation.get("conf", 0.0)) or 0.0),
        "position": position,
        "aabb_center": aabb_center,
        "aabb_size": aabb_size,
        "room_id": room_id,
        "connected_room_ids": [int(room) for room in connected_room_ids if room is not None],
        "parent": observation.get("parent"),
        "children": list(observation.get("children") or []),
        "is_receptacle": bool(observation.get("is_receptacle", False)),
        "is_pickup_candidate": bool(observation.get("is_pickup_candidate", False)),
        "is_articulable": bool(observation.get("is_articulable", False)),
        "is_door": bool(observation.get("is_door", False)),
        "is_movable_door": bool(observation.get("is_movable_door", False)),
        "joint_type": normalize_joint_type(observation.get("joint_type")),
        "joint_range": point_range(observation.get("joint_range")),
        "joint_value": float(observation["joint_value"]) if observation.get("joint_value") is not None else None,
        "joint_infos": list(observation.get("joint_infos") or []),
        "primary_joint_name": str(observation.get("primary_joint_name") or ""),
        "orientation": list(observation.get("orientation") or [0.0, 0.0, 0.0, 1.0]),
        "source_object_name": str(observation.get("source_object_name") or ""),
        "visible_pixels": int(observation.get("visible_pixels", 0) or 0),
        "camera_name": str(observation.get("camera_name") or ""),
        "frame_index": int(observation.get("frame_index", 0) or 0),
        "episode_id": str(observation.get("episode_id") or ""),
        "source": str(observation.get("source") or "detector"),
        "name": str(observation.get("name") or observation.get("object_name") or semantic_name or "object"),
        "asset_id": observation.get("asset_id"),
        "object_id": observation.get("object_id"),
        "viz_aabb_center": point3(viz_aabb_center),
        "viz_aabb_size": point3(viz_aabb_size),
    }


def point_range(values):
    vals = list(values or [])
    if len(vals) < 2:
        vals.extend([0.0] * (2 - len(vals)))
    return [float(vals[0]), float(vals[1])]


def infer_node_type(observation: dict[str, Any]) -> str:
    label = normalize_label(observation.get("semantic_name"))
    if observation.get("is_door") or label in PORTAL_LABELS:
        return "portal"
    if observation.get("is_receptacle"):
        if label in SUPPORT_LABELS:
            return "support"
        if label in CONTAINER_LABELS:
            return "container"
        if observation.get("is_articulable"):
            return "container"
        return "support"
    return "object"


def default_interaction_payload(node_type: str, observation: dict[str, Any]) -> dict[str, Any]:
    joint_type = normalize_joint_type(observation.get("joint_type"))
    joint_range = point_range(observation.get("joint_range"))
    joint_value = observation.get("joint_value")
    interaction_mode = "none"
    if node_type == "portal":
        interaction_mode = "slide" if joint_type == "slide" else "open_close"
    elif node_type == "container":
        if observation.get("is_articulable"):
            interaction_mode = "slide" if joint_type == "slide" else "open_close"
        else:
            interaction_mode = "place_in"
    elif node_type == "support":
        interaction_mode = "place_on"
    elif observation.get("is_pickup_candidate"):
        interaction_mode = "pickup"
    state = infer_interaction_state(node_type, joint_type, joint_range, joint_value, observation)
    is_interactable = interaction_mode != "none"
    if node_type == "portal":
        requires_interaction = bool(is_interactable and state not in {"open", "static_open"})
        traversable = state in {"open", "static_open"}
    else:
        requires_interaction = bool(is_interactable and state in {"closed", "unknown"})
        traversable = True if state in {"open", "ajar", "static_open"} else False if state == "closed" else None
    return {
        "is_interactable": is_interactable,
        "interaction_mode": interaction_mode,
        "state": state,
        "cost": 1.0,
        "confidence": float(observation.get("confidence", 0.0) or 0.0),
        "state_source": str(observation.get("source") or "detector_rule"),
        "state_confidence": float(observation.get("confidence", 0.0) or 0.0),
        "interaction_cost": 1.0,
        "requires_interaction": requires_interaction,
        "traversable": traversable,
        "expected_effect": "unlock_connectivity" if node_type == "portal" else "reveal_contents" if node_type == "container" else "none",
        "operation_history": [],
    }


def infer_interaction_state(node_type: str, joint_type: str, joint_range: list[float], joint_value: float | None, observation: dict[str, Any]) -> str:
    if joint_type == "none" or joint_value is None:
        if node_type == "portal" and not observation.get("is_movable_door", True):
            return "static_open"
        return "unknown"
    joint_min, joint_max = float(joint_range[0]), float(joint_range[1])
    span = abs(joint_max - joint_min)
    if span <= 1e-6:
        return "unknown"
    value = float(joint_value)
    lower_dist = abs(value - joint_min)
    upper_dist = abs(value - joint_max)
    tolerance = max(0.05 * span, 0.02)
    if lower_dist <= tolerance:
        return "closed"
    if upper_dist <= tolerance:
        return "open"
    return "ajar"


def observation_from_detection(detection: dict[str, Any], observation_id: str, source: str = "detector") -> dict[str, Any]:
    world_position = detection.get("world_position") or detection.get("position") or {}
    world_box_center = detection.get("world_box3d_center") or detection.get("aabb_center") or detection.get("box3d_center") or world_position
    size = detection.get("world_box3d_size") or detection.get("aabb_size") or detection.get("box3d_size") or detection.get("size") or {}
    viz_box_center = detection.get("viz_aabb_center") or world_box_center
    viz_box_size = detection.get("viz_aabb_size") or size
    if isinstance(world_position, dict):
        position = [
            world_position.get("x", 0.0),
            world_position.get("y", 0.0),
            world_position.get("z", 0.0),
        ]
    else:
        position = point3(world_position)
    if isinstance(world_box_center, dict):
        aabb_center = [
            world_box_center.get("x", 0.0),
            world_box_center.get("y", 0.0),
            world_box_center.get("z", 0.0),
        ]
    else:
        aabb_center = point3(world_box_center)
    if isinstance(size, dict):
        aabb_size = [
            size.get("x", 0.0),
            size.get("y", 0.0),
            size.get("z", 0.0),
        ]
    else:
        aabb_size = point3(size)
    viz_aabb_center = point3(viz_box_center)
    viz_aabb_size = point3(viz_box_size)
    semantic_name = detection.get("semantic_class") or detection.get("semantic_name") or detection.get("class") or "object"
    return normalize_observation(
        {
            "observation_id": observation_id,
            "instance_id": detection.get("instance_id") or "",
            "semantic_name": semantic_name,
            "category": detection.get("category") or detection.get("semantic_class") or detection.get("class") or semantic_name,
            "candidate_labels": list(detection.get("candidate_labels") or []),
            "label_votes": dict(detection.get("label_votes") or {}),
            "confidence": detection.get("confidence", detection.get("conf", 0.0)),
            "position": position,
            "aabb_center": aabb_center,
            "aabb_size": aabb_size,
            "room_id": detection.get("room_id"),
            "connected_room_ids": detection.get("connected_room_ids") or [],
            "parent": detection.get("parent"),
            "children": list(detection.get("children") or []),
            "is_receptacle": detection.get("is_receptacle", False),
            "is_pickup_candidate": detection.get("is_pickup_candidate", False),
            "is_articulable": detection.get("is_articulable", False),
            "is_door": detection.get("is_door", False),
            "is_movable_door": detection.get("is_movable_door", False),
            "joint_type": detection.get("joint_type"),
            "joint_range": detection.get("joint_range") or [0.0, 0.0],
            "joint_value": detection.get("joint_value"),
            "name": detection.get("name") or detection.get("object_name") or detection.get("instance_id") or semantic_name,
            "asset_id": detection.get("asset_id"),
            "object_id": detection.get("object_id"),
            "source": source,
            "viz_aabb_center": viz_aabb_center,
            "viz_aabb_size": viz_aabb_size,
        }
    )


def room_node_label(room_id_to_name: dict[int, str], room_id: int | None) -> str:
    if room_id is None:
        return "unknown"
    return normalize_label(room_id_to_name.get(int(room_id), f"room_{room_id}"))


def distance_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
