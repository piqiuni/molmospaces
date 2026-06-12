from __future__ import annotations

import time
from collections import defaultdict

from .geometry_utils import grid_index, normalize_label, world_to_grid
from .graph_rules import (
    default_interaction_payload,
    distance_xy,
    infer_node_type,
    normalize_observation,
    room_node_label,
    sanitize_token,
)
from .graph_schema import NavigationHint, SceneGraphBundle, SceneGraphEdge, SceneGraphNode


class InteractionGraphStore:
    def __init__(self, scene_id="scene", match_distance=0.5, room_id_to_name=None, room_box_height=0.2):
        self.scene_id = str(scene_id or "scene")
        self.match_distance = float(match_distance)
        self.room_id_to_name = dict(room_id_to_name or {})
        self.room_box_height = float(room_box_height)
        self.room_geometries = {}
        self.nodes = {}
        self.edges = {}
        self.next_node_index = 1
        self.edge_counter = 1
        self.room_grid = None
        self.source_mode = "detector_online"

    def update_room_grid(self, grid_info, scene_data, confidence_data=None, room_id_to_name=None):
        if room_id_to_name:
            self.room_id_to_name.update({int(k): str(v) for k, v in room_id_to_name.items()})
        self.room_grid = {
            "info": grid_info,
            "scene_data": list(scene_data or []),
            "confidence_data": list(confidence_data or []),
        }
        self._refresh_room_nodes_from_grid()
        self._rebuild_relations()

    def set_room_geometries(self, rooms):
        self.room_geometries = {}
        for room in rooms or []:
            room_id = room.get("room_id")
            if room_id is None:
                continue
            room_id = int(room_id)
            self.room_geometries[room_id] = {
                "center": self._grounded_center(room.get("center") or room.get("aabb_center") or [float(room_id), 0.0, 0.0]),
                "aabb_center": self._grounded_center(room.get("aabb_center") or room.get("center") or [float(room_id), 0.0, 0.0]),
                "aabb_size": self._room_box_size(room.get("aabb_size") or [0.5, 0.5, self.room_box_height]),
                "name": str(room.get("name") or room_node_label(self.room_id_to_name, room_id)),
                "cell_count": int(room.get("cell_count", 0)),
            }
            self.room_id_to_name.setdefault(room_id, self.room_geometries[room_id]["name"])
            node = self._ensure_room_node(room_id)
            geom = self.room_geometries[room_id]
            node.name = geom["name"]
            node.centroid = list(geom["center"])
            node.aabb_center = list(geom["aabb_center"])
            node.aabb_size = list(geom["aabb_size"])
            node.attributes["cell_count"] = geom["cell_count"]

    def update_observations(self, observations, stamp=None, source_mode=None):
        now = float(stamp if stamp is not None else time.time())
        if source_mode:
            self.source_mode = str(source_mode)
        for raw_observation in observations:
            observation = normalize_observation(raw_observation)
            node = self._find_or_create_node(observation)
            self._apply_observation(node, observation, now)
        self._refresh_room_nodes_from_grid()
        self._rebuild_relations(now=now)
        self._refresh_missing_room_nodes_from_observations()

    def as_graph_bundle(self, stamp=None):
        now = float(stamp if stamp is not None else time.time())
        nodes = sorted(self.nodes.values(), key=lambda item: item.id)
        edges = sorted(self.edges.values(), key=lambda item: item.id)
        semantic_node_ids = [node.id for node in nodes]
        semantic_edge_ids = [edge.id for edge in edges]

        interaction_core = {node.id for node in nodes if node.type in {"portal", "support", "container"}}
        interaction_edge_ids = []
        for edge in edges:
            if edge.src_id in interaction_core or edge.dst_id in interaction_core:
                interaction_core.add(edge.src_id)
                interaction_core.add(edge.dst_id)
                interaction_edge_ids.append(edge.id)

        navigation_hints = self._build_navigation_hints()
        navigation_node_ids = sorted({hint.node_id for hint in navigation_hints})
        navigation_edge_ids = [
            edge.id
            for edge in edges
            if edge.src_id in navigation_node_ids or edge.dst_id in navigation_node_ids
        ]

        return SceneGraphBundle(
            scene_id=self.scene_id,
            source_mode=self.source_mode,
            timestamp=now,
            nodes=nodes,
            edges=edges,
            semantic_node_ids=semantic_node_ids,
            semantic_edge_ids=semantic_edge_ids,
            interaction_node_ids=sorted(interaction_core),
            interaction_edge_ids=sorted(set(interaction_edge_ids)),
            navigation_node_ids=navigation_node_ids,
            navigation_edge_ids=navigation_edge_ids,
            navigation_hints=navigation_hints,
        )

    def as_graph_dict(self, stamp=None):
        return self.as_graph_bundle(stamp=stamp).to_dict()

    def as_navigation_hints(self):
        return [hint.to_dict() for hint in self._build_navigation_hints()]

    def prune_stale_nodes(self, stale_after_sec, now=None):
        stale_after_sec = float(stale_after_sec)
        if stale_after_sec <= 0.0:
            return
        now = float(now if now is not None else time.time())
        stale_ids = [
            node_id
            for node_id, node in self.nodes.items()
            if node.type != "room" and node.last_seen is not None and now - float(node.last_seen) > stale_after_sec
        ]
        if not stale_ids:
            return
        stale_id_set = set(stale_ids)
        for node_id in stale_ids:
            self.nodes.pop(node_id, None)
        self.edges = {
            edge_id: edge
            for edge_id, edge in self.edges.items()
            if edge.src_id not in stale_id_set and edge.dst_id not in stale_id_set
        }
        self._rebuild_relations(now=now)

    def _find_or_create_node(self, observation):
        instance_id = observation.get("instance_id")
        if instance_id:
            for node in self.nodes.values():
                if node.attributes.get("instance_id") == instance_id:
                    return node

        node_type = infer_node_type(observation)
        label = normalize_label(observation.get("semantic_name"))
        best = None
        best_dist = None
        for node in self.nodes.values():
            if node.type == "room":
                continue
            if node.type != node_type or node.label != label:
                continue
            dist = distance_xy(node.centroid, observation["position"])
            if dist <= self.match_distance and (best is None or dist < best_dist):
                best = node
                best_dist = dist
        if best is not None:
            return best

        node_id = self._make_node_id(node_type, observation)
        node = SceneGraphNode(
            id=node_id,
            type=node_type,
            label=label or node_type,
            name=str(observation.get("name") or label or node_type),
        )
        self.nodes[node_id] = node
        return node

    def _make_node_id(self, node_type, observation):
        instance_id = sanitize_token(observation.get("instance_id") or "")
        if instance_id:
            node_id = f"{node_type}_{instance_id}"
            if node_id not in self.nodes:
                return node_id
        label = sanitize_token(observation.get("semantic_name") or node_type)
        while True:
            node_id = f"{node_type}_{label}_{self.next_node_index}"
            self.next_node_index += 1
            if node_id not in self.nodes:
                return node_id

    def _apply_observation(self, node, observation, now):
        node.type = infer_node_type(observation)
        node.label = normalize_label(observation.get("semantic_name")) or node.type
        node.name = str(observation.get("name") or node.label or node.type)
        node.centroid = self._ground_non_room_centroid(observation["position"], observation["aabb_size"])
        node.aabb_center = self._ground_non_room_centroid(observation["aabb_center"], observation["aabb_size"])
        node.aabb_size = list(observation["aabb_size"])
        node.room_id = observation.get("room_id") if observation.get("room_id") is not None else node.room_id
        node.confidence = max(float(node.confidence), float(observation.get("confidence", 0.0)))
        node.observation_count += 1
        node.last_seen = now
        node.attributes.update(
            {
                "instance_id": observation.get("instance_id") or node.attributes.get("instance_id") or "",
                "category": observation.get("category"),
                "candidate_labels": list(observation.get("candidate_labels") or []),
                "label_votes": dict(observation.get("label_votes") or {}),
                "parent": observation.get("parent"),
                "children": list(observation.get("children") or []),
                "is_receptacle": bool(observation.get("is_receptacle", False)),
                "is_pickup_candidate": bool(observation.get("is_pickup_candidate", False)),
                "is_articulable": bool(observation.get("is_articulable", False)),
                "is_door": bool(observation.get("is_door", False)),
                "is_movable_door": bool(observation.get("is_movable_door", False)),
                "connected_room_ids": list(observation.get("connected_room_ids") or []),
                "asset_id": observation.get("asset_id"),
                "object_id": observation.get("object_id"),
                "source": observation.get("source"),
                "viz_aabb_center": list(observation.get("viz_aabb_center") or observation["aabb_center"]),
                "viz_aabb_size": list(observation.get("viz_aabb_size") or observation["aabb_size"]),
            }
        )
        node.interaction = default_interaction_payload(node.type, observation)

    def _refresh_room_nodes_from_grid(self):
        if not self.room_grid:
            return
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        confidence_data = self.room_grid["confidence_data"]
        if grid_info is None or not scene_data:
            return
        room_points = defaultdict(list)
        room_conf = defaultdict(list)
        width = int(grid_info.width)
        for idx, scene_id in enumerate(scene_data):
            scene_id = int(scene_id)
            if scene_id < 0:
                continue
            mx = idx % width
            my = idx // width
            wx = float(grid_info.origin.position.x) + (mx + 0.5) * float(grid_info.resolution)
            wy = float(grid_info.origin.position.y) + (my + 0.5) * float(grid_info.resolution)
            room_points[scene_id].append((wx, wy))
            if idx < len(confidence_data):
                room_conf[scene_id].append(float(confidence_data[idx]))
        for room_id, points in room_points.items():
            node = self._ensure_room_node(room_id)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            center = [sum(xs) / len(xs), sum(ys) / len(ys), 0.5 * self.room_box_height]
            size = [
                (max(xs) - min(xs)) if len(xs) > 1 else float(grid_info.resolution),
                (max(ys) - min(ys)) if len(ys) > 1 else float(grid_info.resolution),
                self.room_box_height,
            ]
            node.centroid = center
            node.aabb_center = center
            node.aabb_size = size
            node.confidence = max(node.confidence, sum(room_conf[room_id]) / max(len(room_conf[room_id]), 1) / 100.0)
            node.attributes["cell_count"] = len(points)

    def _refresh_missing_room_nodes_from_observations(self):
        room_to_nodes = defaultdict(list)
        for node in self.nodes.values():
            if node.type == "room" or node.room_id is None:
                continue
            if int(node.room_id) in self.room_geometries:
                continue
            room_to_nodes[int(node.room_id)].append(node)
        for room_id, child_nodes in room_to_nodes.items():
            if not child_nodes:
                continue
            room_node = self._ensure_room_node(room_id)
            mins = []
            maxs = []
            for child in child_nodes:
                center = [float(v) for v in child.aabb_center]
                size = [float(v) for v in child.aabb_size]
                mins.append([center[i] - size[i] * 0.5 for i in range(3)])
                maxs.append([center[i] + size[i] * 0.5 for i in range(3)])
            min_corner = [min(point[i] for point in mins) for i in range(3)]
            max_corner = [max(point[i] for point in maxs) for i in range(3)]
            center = [(min_corner[i] + max_corner[i]) * 0.5 for i in range(3)]
            center[2] = 0.5 * self.room_box_height
            size = [max(max_corner[i] - min_corner[i], 0.1) for i in range(3)]
            size[2] = self.room_box_height
            room_node.centroid = center
            room_node.aabb_center = center
            room_node.aabb_size = size
            room_node.attributes["estimated_from_observations"] = True
            room_node.attributes["cell_count"] = len(child_nodes)

    def _ensure_room_node(self, room_id):
        room_id = int(room_id)
        node_id = f"room_{room_id}"
        node = self.nodes.get(node_id)
        if node is None:
            label = room_node_label(self.room_id_to_name, room_id)
            geometry = self.room_geometries.get(room_id, {})
            center = self._grounded_center(geometry.get("center") or self._default_room_center(room_id))
            aabb_center = self._grounded_center(geometry.get("aabb_center") or center)
            size = self._room_box_size(geometry.get("aabb_size") or self._default_room_size(room_id))
            node = SceneGraphNode(
                id=node_id,
                type="room",
                label=label,
                name=str(geometry.get("name") or f"{label}_{room_id}"),
                centroid=center,
                aabb_center=aabb_center,
                aabb_size=size,
                room_id=room_id,
            )
            if "cell_count" in geometry:
                node.attributes["cell_count"] = int(geometry["cell_count"])
            self.nodes[node_id] = node
        return node

    def _default_room_center(self, room_id):
        if not self.room_grid:
            return [float(room_id), 0.0, 0.0]
        grid_info = self.room_grid.get("info")
        scene_data = list(self.room_grid.get("scene_data") or [])
        if grid_info is None or not scene_data:
            return [float(room_id), 0.0, 0.0]
        width = int(grid_info.width)
        xs = []
        ys = []
        for idx, scene_id in enumerate(scene_data):
            if int(scene_id) != int(room_id):
                continue
            mx = idx % width
            my = idx // width
            xs.append(float(grid_info.origin.position.x) + (mx + 0.5) * float(grid_info.resolution))
            ys.append(float(grid_info.origin.position.y) + (my + 0.5) * float(grid_info.resolution))
        if xs and ys:
            return [sum(xs) / len(xs), sum(ys) / len(ys), 0.5 * self.room_box_height]
        return [float(room_id), 0.0, 0.0]

    def _default_room_size(self, room_id):
        if not self.room_grid:
            return [0.5, 0.5, 0.1]
        grid_info = self.room_grid.get("info")
        scene_data = list(self.room_grid.get("scene_data") or [])
        if grid_info is None or not scene_data:
            return [0.5, 0.5, 0.1]
        width = int(grid_info.width)
        xs = []
        ys = []
        for idx, scene_id in enumerate(scene_data):
            if int(scene_id) != int(room_id):
                continue
            mx = idx % width
            my = idx // width
            xs.append(float(grid_info.origin.position.x) + (mx + 0.5) * float(grid_info.resolution))
            ys.append(float(grid_info.origin.position.y) + (my + 0.5) * float(grid_info.resolution))
        if len(xs) > 1 and len(ys) > 1:
            return [max(max(xs) - min(xs), float(grid_info.resolution)), max(max(ys) - min(ys), float(grid_info.resolution)), self.room_box_height]
        resolution = float(grid_info.resolution) if grid_info is not None else 0.5
        return [resolution, resolution, self.room_box_height]

    def _ground_non_room_centroid(self, center, size):
        grounded = [float(center[0]), float(center[1]), float(center[2])]
        half_height = max(float(size[2]) * 0.5, 0.01)
        grounded[2] = max(grounded[2], half_height)
        return grounded

    def _grounded_center(self, center):
        grounded = list(center)
        if len(grounded) < 3:
            grounded.extend([0.0] * (3 - len(grounded)))
        grounded = [float(v) for v in grounded[:3]]
        grounded[2] = 0.5 * self.room_box_height
        return grounded

    def _room_box_size(self, size):
        size = list(size or [])
        if len(size) < 3:
            size.extend([0.0] * (3 - len(size)))
        room_size = [max(float(size[0]), 0.1), max(float(size[1]), 0.1), self.room_box_height]
        return room_size

    def _rebuild_relations(self, now=None):
        now = float(now if now is not None else time.time())
        self.edges = {}
        rooms = {node.id: node for node in self.nodes.values() if node.type == "room"}
        non_rooms = [node for node in self.nodes.values() if node.type != "room"]

        for node in non_rooms:
            room_id = node.room_id
            if room_id is None:
                room_id = self._infer_room_id_from_node(node)
                node.room_id = room_id
            if room_id is not None:
                room_node = self._ensure_room_node(room_id)
                self._upsert_edge(node.id, "in_room", room_node.id, now=now)
                self._upsert_edge(room_node.id, "has_child", node.id, now=now)

        for node in non_rooms:
            if node.type == "portal":
                connected_room_ids = list(node.attributes.get("connected_room_ids") or [])
                if not connected_room_ids and node.room_id is not None:
                    connected_room_ids = [node.room_id]
                for room_id in sorted(set(int(room) for room in connected_room_ids if room is not None)):
                    room_node = self._ensure_room_node(room_id)
                    self._upsert_edge(node.id, "connects", room_node.id, now=now)
                    self._upsert_edge(room_node.id, "adjacent_via", node.id, now=now)

        support_nodes = [node for node in non_rooms if node.type == "support"]
        container_nodes = [node for node in non_rooms if node.type == "container"]
        object_nodes = [node for node in non_rooms if node.type == "object"]
        id_lookup = {node.id: node for node in non_rooms}
        name_lookup = {node.name: node for node in non_rooms}
        instance_lookup = {
            node.attributes.get("instance_id"): node
            for node in non_rooms
            if node.attributes.get("instance_id")
        }

        for obj in object_nodes:
            obj.parent_id = None
            parent = self._find_parent_node(obj, support_nodes, container_nodes, id_lookup, name_lookup, instance_lookup)
            if parent is None:
                continue
            obj.parent_id = parent.id
            if parent.type == "support":
                self._upsert_edge(parent.id, "supports", obj.id, now=now)
            elif parent.type == "container":
                self._upsert_edge(parent.id, "contains", obj.id, now=now)

    def _find_parent_node(self, obj, support_nodes, container_nodes, id_lookup, name_lookup, instance_lookup):
        parent_name = obj.attributes.get("parent")
        for lookup in (instance_lookup, name_lookup, id_lookup):
            if parent_name and parent_name in lookup:
                candidate = lookup[parent_name]
                if candidate.type in {"support", "container"}:
                    return candidate

        same_room_containers = [node for node in container_nodes if node.room_id == obj.room_id]
        containing = [node for node in same_room_containers if self._is_inside_volume(obj, node)]
        if containing:
            return sorted(containing, key=lambda node: volume(node.aabb_size))[0]

        same_room_supports = [node for node in support_nodes if node.room_id == obj.room_id]
        supporting = [node for node in same_room_supports if self._is_on_support(obj, node)]
        if supporting:
            return sorted(supporting, key=lambda node: abs(top_surface_z(node) - obj.centroid[2]))[0]
        return None

    def _upsert_edge(self, src_id, relation, dst_id, attributes=None, confidence=1.0, now=None):
        edge_id = f"edge_{relation}_{sanitize_token(src_id)}_{sanitize_token(dst_id)}"
        self.edges[edge_id] = SceneGraphEdge(
            id=edge_id,
            src_id=src_id,
            relation=relation,
            dst_id=dst_id,
            attributes=dict(attributes or {}),
            confidence=float(confidence),
            last_seen=now,
        )

    def _infer_room_id_from_node(self, node):
        if not self.room_grid:
            return None
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        if grid_info is None or not scene_data:
            return None
        candidates = defaultdict(int)
        center = node.aabb_center
        size = node.aabb_size
        half_x = max(float(size[0]) * 0.45, 0.02)
        half_y = max(float(size[1]) * 0.45, 0.02)
        sample_points = [
            (float(center[0]), float(center[1])),
            (float(center[0]) - half_x, float(center[1]) - half_y),
            (float(center[0]) - half_x, float(center[1]) + half_y),
            (float(center[0]) + half_x, float(center[1]) - half_y),
            (float(center[0]) + half_x, float(center[1]) + half_y),
        ]
        for px, py in sample_points:
            coords = world_to_grid(px, py, grid_info)
            if coords is None:
                continue
            idx = grid_index(coords[0], coords[1], grid_info.width)
            if idx < 0 or idx >= len(scene_data):
                continue
            room_id = int(scene_data[idx])
            if room_id >= 0:
                candidates[room_id] += 1
        if candidates:
            return max(sorted(candidates.keys()), key=lambda room_id: candidates[room_id])
        return None

    def _build_navigation_hints(self):
        hints = []
        counter = 1
        for node in sorted(self.nodes.values(), key=lambda item: item.id):
            if node.type == "room":
                hints.append(
                    self._make_hint(counter, "room_center", node, False, node.id, "none", "known_room_center")
                )
                counter += 1
            elif node.type == "object":
                hints.append(
                    self._make_hint(counter, "target_object", node, False, None, "none", "detected_object")
                )
                counter += 1
            elif node.type == "portal":
                hints.append(
                    self._make_hint(
                        counter,
                        "interactive_portal",
                        node,
                        True,
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "door_may_unlock_room",
                    )
                )
                counter += 1
            elif node.type == "container":
                hints.append(
                    self._make_hint(
                        counter,
                        "interactive_container",
                        node,
                        bool(node.interaction.get("is_interactable")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "container_may_reveal_object",
                    )
                )
                counter += 1
            elif node.type == "support":
                hints.append(
                    self._make_hint(
                        counter,
                        "support_surface",
                        node,
                        bool(node.interaction.get("is_interactable")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "support_surface_context",
                    )
                )
                counter += 1
        return hints

    def _make_hint(self, index, hint_type, node, requires_interaction, interaction_node_id, interaction_mode, reason):
        return NavigationHint(
            hint_id=f"hint_{index:04d}",
            type=hint_type,
            node_id=node.id,
            position=list(node.centroid),
            room_id=node.room_id,
            priority=1.0,
            confidence=float(node.confidence),
            requires_interaction=bool(requires_interaction),
            interaction_node_id=interaction_node_id,
            interaction_mode=interaction_mode,
            state=node.interaction.get("state", "unknown"),
            reason=reason,
        )


def volume(size):
    return max(float(size[0]), 0.0) * max(float(size[1]), 0.0) * max(float(size[2]), 0.0)


def top_surface_z(node):
    return float(node.aabb_center[2]) + float(node.aabb_size[2]) * 0.5


def point_inside_2d(point, center, size, margin=0.05):
    half_x = float(size[0]) * 0.5 + margin
    half_y = float(size[1]) * 0.5 + margin
    return (
        abs(float(point[0]) - float(center[0])) <= half_x
        and abs(float(point[1]) - float(center[1])) <= half_y
    )


def point_inside_3d(point, center, size, margin=0.05):
    return (
        abs(float(point[0]) - float(center[0])) <= float(size[0]) * 0.5 + margin
        and abs(float(point[1]) - float(center[1])) <= float(size[1]) * 0.5 + margin
        and abs(float(point[2]) - float(center[2])) <= float(size[2]) * 0.5 + margin
    )


def _vertical_gap(obj, support):
    return abs(float(obj.centroid[2]) - top_surface_z(support))


def _support_height_limit(support):
    return max(0.25, float(support.aabb_size[2]) + 0.6)


def _same_room_or_unknown(obj, candidate):
    return candidate.room_id is None or obj.room_id is None or candidate.room_id == obj.room_id


def _support_xy_match(obj, support):
    return point_inside_2d(obj.centroid, support.aabb_center, support.aabb_size, margin=0.08)


def _object_above_support(obj, support):
    return float(obj.centroid[2]) >= top_surface_z(support) - 0.12


def _object_not_too_high(obj, support):
    return _vertical_gap(obj, support) <= _support_height_limit(support)


def _container_contains(obj, container):
    return point_inside_3d(obj.centroid, container.aabb_center, container.aabb_size, margin=0.08)


InteractionGraphStore._is_inside_volume = staticmethod(lambda obj, container: _same_room_or_unknown(obj, container) and _container_contains(obj, container))
InteractionGraphStore._is_on_support = staticmethod(
    lambda obj, support: _same_room_or_unknown(obj, support)
    and _support_xy_match(obj, support)
    and _object_above_support(obj, support)
    and _object_not_too_high(obj, support)
)
