from __future__ import annotations

import random
from typing import Any

import numpy as np

from molmo_spaces.utils.mj_model_and_data_utils import body_aabb


CONTAINER_TOKENS = {
    "fridge",
    "refrigerator",
    "cabinet",
    "drawer",
    "dresser",
    "wardrobe",
    "closet",
    "cupboard",
    "chest",
    "box",
    "microwave",
    "dishwasher",
    "storage",
}


def _normalized_tokens(value: Any) -> set[str]:
    return {
        token
        for token in str(value or "").casefold().replace("_", " ").split()
        if token
    }


def _category(object_manager, name: str, metadata: dict[str, Any]) -> str:
    try:
        return str(metadata.get("category") or object_manager.get_annotation_category(name) or name)
    except Exception:
        return str(metadata.get("category") or name)


def _is_container(object_manager, name: str, metadata: dict[str, Any], door_names: set[str]) -> bool:
    if name in door_names:
        return False
    category = _category(object_manager, name, metadata)
    tokens = _normalized_tokens(f"{name} {category}")
    try:
        has_receptacle = bool(object_manager.has_receptacle_site(name))
        articulable = bool(object_manager.is_object_articulable(name))
    except Exception:
        return False
    return articulable and has_receptacle and bool(tokens & CONTAINER_TOKENS)


def _is_inside(container_center, container_size, object_center, object_size, margin: float = 0.05) -> tuple[bool, float]:
    container_center = np.asarray(container_center, dtype=float)
    container_size = np.maximum(np.asarray(container_size, dtype=float), 1e-4)
    object_center = np.asarray(object_center, dtype=float)
    object_size = np.maximum(np.asarray(object_size, dtype=float), 1e-4)
    container_min = container_center - container_size * 0.5
    container_max = container_center + container_size * 0.5
    object_min = object_center - object_size * 0.5
    object_max = object_center + object_size * 0.5
    overlap = np.maximum(0.0, np.minimum(container_max, object_max) - np.maximum(container_min, object_min))
    overlap_ratio = float(np.prod(overlap) / max(float(np.prod(object_size)), 1e-6))
    center_inside = bool(np.all(object_center >= container_min - margin) and np.all(object_center <= container_max + margin))
    bounds_inside = bool(np.all(object_min >= container_min - margin) and np.all(object_max <= container_max + margin))
    return bounds_inside or (center_inside and overlap_ratio >= 0.55), overlap_ratio


def _parent_chain(object_manager, name: str) -> list[str]:
    try:
        return [str(value) for value in object_manager.get_parent_chain_names(name)]
    except Exception:
        return []


def _robot_xy(task) -> list[float]:
    pose = np.asarray(task.env.current_robot.robot_view.base.pose, dtype=float)
    return [float(pose[0, 3]), float(pose[1, 3])]


def select_far_container_target(
    task,
    *,
    selection_seed: int,
    top_k: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = task.env
    object_manager = env.object_managers[env.current_batch_index]
    metadata_by_name = dict((env.current_scene_metadata or {}).get("objects", {}) or {})
    model = env.current_model
    data = env.current_data
    try:
        door_names = {str(name) for name in object_manager.find_door_names()}
    except Exception:
        door_names = set()

    records: dict[str, dict[str, Any]] = {}
    try:
        objects = object_manager.list_top_level_objects()
    except Exception:
        objects = []
    for obj in objects:
        name = str(getattr(obj, "name", "") or "")
        if not name or name not in metadata_by_name or name in door_names:
            continue
        metadata = dict(metadata_by_name.get(name) or {})
        try:
            body_id = int(model.body(name).id)
            center, size = body_aabb(model, data, body_id, visual_only=True)
            structural = bool(object_manager.is_structural(name))
        except Exception:
            continue
        category = _category(object_manager, name, metadata)
        records[name] = {
            "name": name,
            "category": category,
            "metadata": metadata,
            "center": np.asarray(center, dtype=float),
            "size": np.asarray(size, dtype=float),
            "structural": structural,
            "parent": metadata.get("parent"),
            "parent_chain": _parent_chain(object_manager, name),
        }

    containers = {
        name: record
        for name, record in records.items()
        if not record["structural"] and _is_container(object_manager, name, record["metadata"], door_names)
    }
    targets = {
        name: record
        for name, record in records.items()
        if not record["structural"]
        and name not in containers
        and not bool(record["metadata"].get("is_structural", False))
    }

    candidate_rows: list[dict[str, Any]] = []
    robot_xy = np.asarray(_robot_xy(task), dtype=float)
    for target_name, target in targets.items():
        matches = []
        for container_name, container in containers.items():
            inside, overlap_ratio = _is_inside(
                container["center"],
                container["size"],
                target["center"],
                target["size"],
            )
            parent_hit = bool(
                target.get("parent") == container_name
                or container_name in set(target.get("parent_chain") or [])
            )
            if not inside and not parent_hit and overlap_ratio < 0.80:
                continue
            relation_score = (4.0 if inside else 0.0) + (2.0 if parent_hit else 0.0) + overlap_ratio
            matches.append((relation_score, float(np.prod(container["size"])), container_name, overlap_ratio, inside, parent_hit))
        if not matches:
            continue
        _, _, container_name, overlap_ratio, inside, parent_hit = max(
            matches,
            key=lambda row: (row[0], -row[1]),
        )
        distance_m = float(np.linalg.norm(target["center"][:2] - robot_xy))
        candidate_rows.append(
            {
                "target_name": target_name,
                "target_category": target["category"],
                "target_center": target["center"].tolist(),
                "target_size": target["size"].tolist(),
                "container_name": container_name,
                "container_category": containers[container_name]["category"],
                "container_center": containers[container_name]["center"].tolist(),
                "container_size": containers[container_name]["size"].tolist(),
                "distance_m": distance_m,
                "overlap_ratio": float(overlap_ratio),
                "inside_aabb": bool(inside),
                "parent_relation": bool(parent_hit),
            }
        )

    if not candidate_rows:
        context = {
            "enabled": False,
            "selection_mode": "random_far_container_object",
            "selection_status": "no_container_object_candidate",
        }
        return context, {"target_context": context, "candidate_count": 0, "candidates": []}

    candidate_rows.sort(key=lambda row: (-float(row["distance_m"]), str(row["target_name"])))
    eligible_count = min(max(1, int(top_k)), len(candidate_rows))
    rng = random.Random(int(selection_seed))
    selected = dict(rng.choice(candidate_rows[:eligible_count]))
    target_category = str(selected["target_category"])
    container_category = str(selected["container_category"])
    target_labels = sorted({target_category, selected["target_name"]})
    container_labels = sorted({container_category, selected["container_name"]})
    context = {
        "enabled": True,
        "selection_mode": "random_far_container_object",
        "selection_seed": int(selection_seed),
        "selection_top_k": int(eligible_count),
        "target_name": target_category,
        "object_labels": target_labels,
        "target_source_object_name": selected["target_name"],
        "target_container_name": container_category,
        "target_container_labels": container_labels,
        "target_container_source_object_name": selected["container_name"],
        "target_instance_id": "",
        "target_container_instance_id": "",
        "standoff_m": 1.0,
        "require_interaction": True,
        "target_container_requires_interaction": True,
        "target_object_requires_interaction": False,
        "completion_requires_visibility": True,
        "require_current_visibility": False,
        "target_min_visible_pixels": 16,
        "target_require_same_room": False,
        "target_allow_connected_room": True,
    }
    selected["target_context"] = context
    selected["robot_xy"] = robot_xy.tolist()
    selected["candidate_count"] = len(candidate_rows)
    selected["candidate_rows"] = candidate_rows
    return context, selected
