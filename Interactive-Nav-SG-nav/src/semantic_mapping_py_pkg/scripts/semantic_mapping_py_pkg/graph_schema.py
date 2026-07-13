from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_float_list(values, size=3):
    vals = list(values or [])
    if len(vals) < size:
        vals.extend([0.0] * (size - len(vals)))
    return [float(vals[i]) for i in range(size)]


@dataclass
class SceneGraphNode:
    id: str
    type: str
    label: str
    name: str
    centroid: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    aabb_center: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    aabb_size: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    parent_id: str | None = None
    room_id: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    interaction: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    observation_count: int = 0
    last_seen: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "name": self.name,
            "centroid": _as_float_list(self.centroid),
            "aabb_center": _as_float_list(self.aabb_center),
            "aabb_size": _as_float_list(self.aabb_size),
            "parent_id": self.parent_id,
            "room_id": None if self.room_id is None else int(self.room_id),
            "attributes": dict(self.attributes),
            "interaction": dict(self.interaction),
            "confidence": float(self.confidence),
            "observation_count": int(self.observation_count),
            "last_seen": None if self.last_seen is None else float(self.last_seen),
        }


@dataclass
class SceneGraphEdge:
    id: str
    src_id: str
    relation: str
    dst_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    last_seen: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "src_id": self.src_id,
            "relation": self.relation,
            "dst_id": self.dst_id,
            "attributes": dict(self.attributes),
            "confidence": float(self.confidence),
            "last_seen": None if self.last_seen is None else float(self.last_seen),
        }


@dataclass
class NavigationHint:
    hint_id: str
    type: str
    node_id: str
    position: list[float]
    room_id: int | None
    priority: float
    confidence: float
    requires_interaction: bool
    interaction_node_id: str | None
    interaction_mode: str
    state: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "type": self.type,
            "node_id": self.node_id,
            "position": _as_float_list(self.position),
            "room_id": None if self.room_id is None else int(self.room_id),
            "priority": float(self.priority),
            "confidence": float(self.confidence),
            "requires_interaction": bool(self.requires_interaction),
            "interaction_node_id": self.interaction_node_id,
            "interaction_mode": self.interaction_mode,
            "state": self.state,
            "reason": self.reason,
        }


@dataclass
class SceneGraphBundle:
    scene_id: str
    source_mode: str
    timestamp: float
    nodes: list[SceneGraphNode]
    edges: list[SceneGraphEdge]
    semantic_node_ids: list[str]
    semantic_edge_ids: list[str]
    interaction_node_ids: list[str]
    interaction_edge_ids: list[str]
    navigation_node_ids: list[str]
    navigation_edge_ids: list[str]
    navigation_hints: list[NavigationHint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "source_mode": self.source_mode,
            "timestamp": float(self.timestamp),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "views": {
                "semantic_view": {
                    "node_ids": list(self.semantic_node_ids),
                    "edge_ids": list(self.semantic_edge_ids),
                },
                "interaction_view": {
                    "node_ids": list(self.interaction_node_ids),
                    "edge_ids": list(self.interaction_edge_ids),
                },
                "navigation_view": {
                    "node_ids": list(self.navigation_node_ids),
                    "edge_ids": list(self.navigation_edge_ids),
                    "hints": [hint.to_dict() for hint in self.navigation_hints],
                },
            },
        }
