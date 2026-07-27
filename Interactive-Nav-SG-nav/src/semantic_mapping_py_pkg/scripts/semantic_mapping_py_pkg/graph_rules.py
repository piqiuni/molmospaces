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
OPENABLE_CONTAINER_LABELS = CONTAINER_LABELS - {"box", "storage_bin"}

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


def segmentation_pixel_count(segmentation: Any) -> int:
    if isinstance(segmentation, dict):
        rows = list(segmentation.get("rows") or [])
        cols = list(segmentation.get("cols") or [])
        return min(len(rows), len(cols))
    if isinstance(segmentation, list):
        return sum(
            1
            for row in segmentation
            if isinstance(row, list)
            for value in row
            if bool(value)
        )
    return 0


def bbox_area(bbox: Any) -> float:
    values = list(bbox or [])
    if len(values) < 4:
        return 0.0
    return max(0.0, float(values[2]) - float(values[0]) + 1.0) * max(
        0.0, float(values[3]) - float(values[1]) + 1.0
    )


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
    minimal_gt = bool(
        observation.get("id") is not None and observation.get("box_3d") is not None
    )
    box_3d = observation.get("box_3d") or {}
    if not isinstance(box_3d, dict):
        box_3d = {}
    if minimal_gt:
        semantic_name = normalize_label(observation.get("name"))
        category = semantic_name or "object"
        position = point3(box_3d.get("center"))
        aabb_center = point3(box_3d.get("center"))
        aabb_size = point3(box_3d.get("size"))
        bbox_2d = list(observation.get("bbox_2d") or [])
        segmentation = observation.get("segmentation")
        visible_pixels = segmentation_pixel_count(segmentation)
        area = bbox_area(bbox_2d)
        visible_fraction = (
            min(1.0, float(visible_pixels) / area) if area > 0.0 else 0.0
        )
    else:
        semantic_name = normalize_label(
            observation.get("semantic_name")
            or observation.get("semantic_class")
            or observation.get("class")
            or observation.get("name")
        )
        category = str(observation.get("category") or semantic_name or "object")
        position = point3(
            observation.get("position")
            or observation.get("coord")
            or observation.get("centroid")
            or box_3d.get("center")
        )
        aabb_center = point3(
            observation.get("aabb_center")
            or observation.get("box3d_center")
            or box_3d.get("center")
            or position
        )
        aabb_size = point3(
            observation.get("aabb_size")
            or observation.get("size")
            or observation.get("box3d_size")
            or box_3d.get("size")
        )
        bbox_2d = list(
            observation.get("bbox_2d") or observation.get("bbox") or []
        )
        segmentation = observation.get("segmentation")
        if segmentation is None:
            segmentation = observation.get("mask")
        visible_pixels = int(
            observation.get("visible_pixels", segmentation_pixel_count(segmentation))
            or 0
        )
        visible_fraction = observation.get("visible_fraction")
        if visible_fraction is None:
            area = bbox_area(bbox_2d)
            visible_fraction = (
                min(1.0, float(visible_pixels) / area) if area > 0.0 else 0.0
            )
    connected_room_ids = [] if minimal_gt else observation.get("connected_room_ids") or []
    room_id = None if minimal_gt else observation.get("room_id")
    if room_id is not None:
        try:
            room_id = int(room_id)
        except (TypeError, ValueError):
            room_id = None
    viz_aabb_center = (
        box_3d.get("center")
        if minimal_gt
        else observation.get("viz_aabb_center")
        or observation.get("world_box3d_center")
        or observation.get("aabb_center")
    )
    viz_aabb_size = (
        box_3d.get("size")
        if minimal_gt
        else observation.get("viz_aabb_size")
        or observation.get("world_box3d_size")
        or observation.get("aabb_size")
        or observation.get("size")
    )
    return {
        "minimal_gt_observation": minimal_gt,
        "observation_id": str(observation.get("id") if minimal_gt else observation.get("observation_id") or observation.get("id") or ""),
        "instance_id": str(observation.get("id") if minimal_gt else observation.get("instance_id") or observation.get("id") or ""),
        "semantic_name": semantic_name or "object",
        "category": category,
        "candidate_labels": [] if minimal_gt else list(observation.get("candidate_labels") or []),
        "label_votes": {} if minimal_gt else dict(observation.get("label_votes") or {}),
        "confidence": 1.0 if minimal_gt else float(
            observation.get("confidence", observation.get("conf", 0.0)) or 0.0
        ),
        "position": position,
        "aabb_center": aabb_center,
        "aabb_size": aabb_size,
        "room_id": room_id,
        "connected_room_ids": [int(room) for room in connected_room_ids if room is not None],
        "parent": None if minimal_gt else observation.get("parent"),
        "children": [] if minimal_gt else list(observation.get("children") or []),
        "is_receptacle": False if minimal_gt else bool(observation.get("is_receptacle", False)),
        "is_pickup_candidate": False if minimal_gt else bool(observation.get("is_pickup_candidate", False)),
        "is_articulable": False if minimal_gt else bool(observation.get("is_articulable", False)),
        "is_door": semantic_name in PORTAL_LABELS if minimal_gt else bool(observation.get("is_door", semantic_name in PORTAL_LABELS)),
        "is_movable_door": False if minimal_gt else bool(observation.get("is_movable_door", False)),
        "joint_type": "none" if minimal_gt else normalize_joint_type(observation.get("joint_type")),
        "joint_range": [0.0, 0.0] if minimal_gt else point_range(observation.get("joint_range")),
        "joint_value": None if minimal_gt else float(observation["joint_value"]) if observation.get("joint_value") is not None else None,
        "joint_infos": [] if minimal_gt else list(observation.get("joint_infos") or []),
        "primary_joint_name": "" if minimal_gt else str(observation.get("primary_joint_name") or ""),
        "orientation": [0.0, 0.0, 0.0, 1.0] if minimal_gt else list(observation.get("orientation") or [0.0, 0.0, 0.0, 1.0]),
        "interaction_approach_axis_xy": [] if minimal_gt else list(observation.get("interaction_approach_axis_xy") or []),
        "source_object_name": str(
            observation.get("source_object_name") or observation.get("id") or ""
        ),
        "visible_pixels": visible_pixels,
        "visible_fraction": float(visible_fraction or 0.0),
        "bbox_2d": bbox_2d,
        "segmentation": segmentation,
        "box_3d_frame_id": str(box_3d.get("frame_id") or ""),
        "projected_bbox_2d": [] if minimal_gt else list(observation.get("projected_bbox_2d") or []),
        "consecutive_observations": int(
            0 if minimal_gt else observation.get("consecutive_observations", 0) or 0
        ),
        "camera_name": "" if minimal_gt else str(observation.get("camera_name") or ""),
        "frame_index": 0 if minimal_gt else int(observation.get("frame_index", 0) or 0),
        "episode_id": "" if minimal_gt else str(observation.get("episode_id") or ""),
        "source": "realtime_gt_observation" if minimal_gt else str(observation.get("source") or "detector"),
        "name": str(
            observation.get("name")
            if minimal_gt
            else observation.get("name")
            or observation.get("object_name")
            or semantic_name
            or "object"
        ),
        "asset_id": None if minimal_gt else observation.get("asset_id"),
        "object_id": None if minimal_gt else observation.get("object_id"),
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
    if label in PORTAL_LABELS:
        return "portal"
    if label in SUPPORT_LABELS:
        return "support"
    if label in CONTAINER_LABELS:
        return "container"
    return "object"


def default_interaction_payload(node_type: str, observation: dict[str, Any]) -> dict[str, Any]:
    label = normalize_label(observation.get("semantic_name"))
    interaction_mode = "none"
    if node_type == "portal":
        interaction_mode = "open_close"
    elif node_type == "container":
        if label in {"box", "storage_bin"}:
            interaction_mode = "none"
        elif label in {"drawer", "dresser", "chest_of_drawers"}:
            interaction_mode = "slide"
        elif label in OPENABLE_CONTAINER_LABELS:
            interaction_mode = "open_close"
    elif node_type == "support":
        interaction_mode = "place_on"
    state = "unknown"
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
        "state_source": "semantic_graph_default",
        "state_confidence": float(observation.get("confidence", 0.0) or 0.0),
        "interaction_cost": 1.0,
        "requires_interaction": requires_interaction,
        "traversable": traversable,
        "expected_effect": "unlock_connectivity" if node_type == "portal" else "reveal_contents" if node_type == "container" else "none",
        "operation_history": [],
        "completed_interaction_groups": [],
        "failed_interaction_groups": [],
    }
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
