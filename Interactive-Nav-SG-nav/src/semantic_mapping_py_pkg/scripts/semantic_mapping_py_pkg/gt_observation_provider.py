from __future__ import annotations

import math
import random
from typing import Any

from .graph_rules import normalize_joint_type, normalize_observation


def observation_from_gt_record(record: dict[str, Any], observation_id: str, source: str = "gt_replay") -> dict[str, Any]:
    joint_infos = list(record.get("joint_infos") or [])
    primary_joint = joint_infos[0] if joint_infos else {}
    connected_room_ids = record.get("connected_room_ids") or []
    return normalize_observation(
        {
            "observation_id": observation_id,
            "instance_id": record.get("object_id") or record.get("name") or "",
            "semantic_name": record.get("category") or record.get("name") or "object",
            "category": record.get("category") or record.get("name") or "object",
            "confidence": 1.0,
            "position": list(record.get("position") or [0.0, 0.0, 0.0]),
            "aabb_center": list(record.get("aabb_center") or record.get("position") or [0.0, 0.0, 0.0]),
            "aabb_size": list(record.get("aabb_size") or [0.0, 0.0, 0.0]),
            "room_id": record.get("room_id"),
            "connected_room_ids": connected_room_ids,
            "parent": record.get("parent"),
            "children": list(record.get("children") or []),
            "is_receptacle": bool(record.get("is_receptacle", False)),
            "is_pickup_candidate": bool(record.get("is_pickup_candidate", False)),
            "is_articulable": bool(record.get("is_articulable", False)),
            "is_door": bool(record.get("is_door", False)),
            "is_movable_door": bool(record.get("is_movable_door", False)),
            "joint_type": normalize_joint_type(primary_joint.get("joint_type")),
            "joint_range": list(primary_joint.get("joint_range") or [0.0, 0.0]),
            "joint_value": primary_joint.get("joint_value"),
            "joint_infos": joint_infos,
            "primary_joint_name": primary_joint.get("joint_name", ""),
            "source": source,
            "name": record.get("name") or record.get("object_id") or record.get("category") or "object",
            "asset_id": record.get("asset_id"),
            "object_id": record.get("object_id"),
        }
    )


def build_gt_observation_batches(records: list[dict[str, Any]], num_batches=1, shuffle=False, seed=0):
    normalized = [
        observation_from_gt_record(record, observation_id=f"gt_obs_{index:04d}")
        for index, record in enumerate(records, start=1)
    ]
    return split_observations_into_batches(normalized, num_batches=num_batches, shuffle=shuffle, seed=seed)


def split_observations_into_batches(observations, num_batches=1, shuffle=False, seed=0, batch_size=None):
    normalized = list(observations or [])
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(normalized)
    if batch_size is None:
        num_batches = max(1, int(num_batches))
        batch_size = int(math.ceil(len(normalized) / float(num_batches))) if normalized else 1
    else:
        batch_size = max(1, int(batch_size))
    batches = []
    for start in range(0, len(normalized), batch_size):
        batches.append(normalized[start : start + batch_size])
    return batches or [[]]


def build_gt_observation_stream(scene_id: str, records: list[dict[str, Any]], num_batches=1, shuffle=False, seed=0):
    return {
        "scene_id": str(scene_id),
        "source_mode": "gt_replay",
        "batches": build_gt_observation_batches(records, num_batches=num_batches, shuffle=shuffle, seed=seed),
    }
