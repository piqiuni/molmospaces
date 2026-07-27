from __future__ import annotations

from typing import Any


CANDIDATE_COLORS = {
    "EXPLORE": (230, 40, 40),
    "INTERACT": (255, 140, 0),
    "NAVIGATE": (230, 40, 40),
}


def candidate_color(behavior_type: str) -> tuple[int, int, int]:
    return CANDIDATE_COLORS.get(str(behavior_type).upper(), (100, 100, 100))


def candidate_overlays(
    candidates_payload: dict[str, Any] | None,
    proposals_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates_payload = candidates_payload or {}
    candidates = list(candidates_payload.get("candidates") or [])
    if candidates:
        overlays = []
        for candidate in candidates:
            goal = list(candidate.get("goal_xyyaw") or [])
            if len(goal) < 2:
                continue
            behavior_type = str(candidate.get("behavior_type") or "").upper()
            metadata = candidate.get("metadata") or {}
            overlays.append(
                {
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "behavior_type": behavior_type,
                    "target_id": str(candidate.get("target_id") or ""),
                    "target_name": str(
                        candidate.get("target_name")
                        or candidate.get("target_id")
                        or candidate.get("candidate_id")
                        or behavior_type
                    ),
                    "goal_xyyaw": [
                        float(goal[0]),
                        float(goal[1]),
                        float(goal[2]) if len(goal) > 2 else 0.0,
                    ],
                    "frame_id": str(
                        candidate.get("frame_id")
                        or metadata.get("frame_id")
                        or candidates_payload.get("frame_id")
                        or ""
                    ),
                    "color": candidate_color(behavior_type),
                    "source": str(candidate.get("source") or "semantic_decision"),
                }
            )
        return overlays

    proposals_payload = proposals_payload or {}
    overlays = []
    for proposal in proposals_payload.get("proposals") or []:
        goal = list(proposal.get("goal_xyyaw") or [])
        if len(goal) < 2:
            continue
        proposal_id = str(
            proposal.get("proposal_id") or proposal.get("cluster_id") or ""
        )
        overlays.append(
            {
                "candidate_id": f"frontier:{proposal_id}",
                "behavior_type": "EXPLORE",
                "target_id": proposal_id,
                "target_name": proposal_id,
                "goal_xyyaw": [
                    float(goal[0]),
                    float(goal[1]),
                    float(goal[2]) if len(goal) > 2 else 0.0,
                ],
                "frame_id": str(
                    proposal.get("frame_id")
                    or proposals_payload.get("frame_id")
                    or ""
                ),
                "color": candidate_color("EXPLORE"),
                "source": str(proposal.get("source") or "explore_py"),
            }
        )
    return overlays


def topology_edge_style(
    relation: str,
    src_type: str = "",
    dst_type: str = "",
) -> dict[str, Any] | None:
    relation = str(relation)
    src_type = str(src_type)
    dst_type = str(dst_type)
    if relation == "connects":
        return {"color": (40, 135, 235), "thickness": 3, "label": "portal-room"}
    if relation == "contains":
        return {"color": (190, 65, 205), "thickness": 3, "label": "contains"}
    if relation == "supports":
        return {"color": (220, 120, 40), "thickness": 3, "label": "supports"}
    if relation == "has_child":
        if dst_type == "container":
            return {"color": (65, 170, 75), "thickness": 2, "label": "room-container"}
        if dst_type == "support":
            return {"color": (155, 145, 45), "thickness": 2, "label": "room-support"}
        if src_type == "room":
            return {"color": (145, 145, 145), "thickness": 2, "label": "room-object"}
    return None


def topology_edge_visible(
    edge: dict[str, Any],
    node_lookup: dict[str, dict[str, Any]],
) -> bool:
    relation = str(edge.get("relation") or "")
    src_type = str((node_lookup.get(str(edge.get("src_id") or "")) or {}).get("type") or "")
    dst_type = str((node_lookup.get(str(edge.get("dst_id") or "")) or {}).get("type") or "")
    endpoint_types = {src_type, dst_type}
    if relation == "connects":
        return endpoint_types == {"room", "portal"}
    if relation == "has_child":
        return src_type == "room" and dst_type == "container"
    if relation == "contains":
        return src_type == "container" and dst_type in {"container", "object"}
    return False


def portal_room_node_ids(
    portal: dict[str, Any],
    edges: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return the room node IDs connected to a portal without relying on GT parents."""
    room_ids: set[str] = set()
    for room_id in (portal.get("attributes") or {}).get("connected_room_ids") or []:
        text = str(room_id)
        room_ids.add(text if text.startswith("room_") else f"room_{text}")

    portal_id = str(portal.get("id") or "")
    for edge in edges or []:
        if str(edge.get("relation") or "") not in {"connects", "adjacent_via"}:
            continue
        src_id = str(edge.get("src_id") or "")
        dst_id = str(edge.get("dst_id") or "")
        if src_id == portal_id and dst_id.startswith("room_"):
            room_ids.add(dst_id)
        elif dst_id == portal_id and src_id.startswith("room_"):
            room_ids.add(src_id)
    return sorted(room_ids)


def topology_order_rooms(
    rooms: list[dict[str, Any]],
    portals: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Order rooms horizontally with a well-connected room in the center.

    World-space x/y remains the stable fallback and tie-breaker. When the
    portal graph has a room connected to at least two other rooms, that room
    is inserted into the middle column so a circulation hub is not rendered
    at an outer edge of the topology figure.
    """

    def world_key(room: dict[str, Any]) -> tuple[int, float, float, str]:
        center = room.get("aabb_center") or room.get("centroid") or []
        if len(center) < 2:
            return (1, 0.0, 0.0, str(room.get("id") or ""))
        return (0, float(center[0]), float(center[1]), str(room.get("id") or ""))

    ordered = sorted(rooms, key=world_key)
    if len(ordered) < 3:
        return ordered

    room_by_id = {str(room.get("id") or ""): room for room in ordered}
    adjacency = {room_id: set() for room_id in room_by_id}
    for portal in portals:
        connected = [
            room_id
            for room_id in portal_room_node_ids(portal, edges)
            if room_id in room_by_id
        ]
        for index, first in enumerate(connected):
            for second in connected[index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)

    max_degree = max((len(neighbors) for neighbors in adjacency.values()), default=0)
    if max_degree < 2:
        return ordered

    spatial_index = {str(room.get("id") or ""): index for index, room in enumerate(ordered)}
    middle_index = len(ordered) // 2
    center_id = min(
        (room_id for room_id, neighbors in adjacency.items() if len(neighbors) == max_degree),
        key=lambda room_id: (
            abs(spatial_index[room_id] - middle_index),
            spatial_index[room_id],
            room_id,
        ),
    )
    center_room = room_by_id[center_id]
    outer_rooms = [room for room in ordered if str(room.get("id") or "") != center_id]
    outer_rooms.insert(middle_index, center_room)
    return outer_rooms


def portal_positions_between_rooms(
    portals: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None,
    room_positions: dict[str, tuple[int, int]],
    spacing: float = 24.0,
    vertical_offset_y: float = 0.0,
) -> dict[str, tuple[int, int]]:
    """Place connected portals between rooms, optionally half a level lower."""
    portals_by_room_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for portal in portals:
        connected_rooms = [
            room_id
            for room_id in portal_room_node_ids(portal, edges)
            if room_id in room_positions
        ]
        if len(connected_rooms) < 2:
            continue
        pair = tuple(sorted(connected_rooms[:2]))
        portals_by_room_pair.setdefault(pair, []).append(portal)

    result = {}
    for room_pair, paired_portals in portals_by_room_pair.items():
        first = room_positions[room_pair[0]]
        second = room_positions[room_pair[1]]
        midpoint_x = (first[0] + second[0]) * 0.5
        midpoint_y = (first[1] + second[1]) * 0.5 + float(vertical_offset_y)
        delta_x = float(second[0] - first[0])
        delta_y = float(second[1] - first[1])
        length = max(1.0, (delta_x * delta_x + delta_y * delta_y) ** 0.5)
        perpendicular_x = -delta_y / length
        perpendicular_y = delta_x / length
        ordered_portals = sorted(paired_portals, key=lambda item: str(item.get("id")))
        for index, portal in enumerate(ordered_portals):
            offset = (index - (len(ordered_portals) - 1) * 0.5) * float(spacing)
            result[str(portal.get("id"))] = (
                int(midpoint_x + perpendicular_x * offset),
                int(midpoint_y + perpendicular_y * offset),
            )
    return result


def topology_hierarchy_layout(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None,
    panel_width: int,
    panel_height: int,
) -> dict[str, Any]:
    """Lay out the interaction hierarchy inside a compact video panel."""
    width = max(1, int(panel_width))
    height = max(1, int(panel_height))
    left_gutter = min(max(48, int(width * 0.12)), max(48, width // 3))
    x_min = min(width - 2, left_gutter + 4)
    x_max = max(x_min, width - 8)
    graph_top = min(height - 8, max(42, int(height * 0.17)))
    graph_bottom = max(graph_top, height - max(24, int(height * 0.08)))
    row_span = max(0, graph_bottom - graph_top)
    row_y = {
        "room": int(graph_top + row_span * 0.08),
        "portal": int(graph_top + row_span * 0.36),
        "container": int(graph_top + row_span * 0.64),
        "object": int(graph_top + row_span * 0.90),
    }
    by_type = {
        node_type: sorted(
            [node for node in nodes if str(node.get("type") or "") == node_type],
            key=lambda item: str(item.get("id") or ""),
        )
        for node_type in ("room", "portal", "container", "object")
    }
    positions: dict[str, tuple[int, int]] = {}

    def place_row(
        node_type: str,
        preferred_x: dict[str, float] | None = None,
    ) -> None:
        row_nodes = by_type[node_type]
        preferred_x = preferred_x or {}
        ordered = sorted(
            row_nodes,
            key=lambda node: (
                preferred_x.get(str(node.get("id") or ""), float("inf")),
                str(node.get("id") or ""),
            ),
        )
        for index, node in enumerate(ordered):
            x = int(x_min + (index + 1) * (x_max - x_min) / (len(ordered) + 1))
            positions[str(node.get("id") or "")] = (x, row_y[node_type])

    place_row("room")
    room_positions = {
        node_id: position
        for node_id, position in positions.items()
        if node_id.startswith("room_")
    }
    portal_preferred = {}
    for portal in by_type["portal"]:
        room_ids = [
            room_id
            for room_id in portal_room_node_ids(portal, edges)
            if room_id in room_positions
        ]
        if room_ids:
            portal_preferred[str(portal.get("id") or "")] = sum(
                room_positions[room_id][0] for room_id in room_ids
            ) / len(room_ids)
    place_row("portal", portal_preferred)

    container_preferred = {}
    for container in by_type["container"]:
        room_id = container.get("room_id")
        parent_id = (
            f"room_{int(room_id)}"
            if room_id is not None
            else str(container.get("parent_id") or "")
        )
        if parent_id in room_positions:
            container_preferred[str(container.get("id") or "")] = room_positions[
                parent_id
            ][0]
    place_row("container", container_preferred)

    object_parent = {
        str(edge.get("dst_id") or ""): str(edge.get("src_id") or "")
        for edge in edges or []
        if str(edge.get("relation") or "") == "contains"
    }
    object_preferred = {}
    for node in by_type["object"]:
        parent_position = positions.get(object_parent.get(str(node.get("id") or ""), ""))
        if parent_position is not None:
            object_preferred[str(node.get("id") or "")] = parent_position[0]
    place_row("object", object_preferred)

    nominal_sizes = {
        "room": (104, 34),
        "portal": (58, 28),
        "container": (88, 28),
        "object": (62, 24),
    }
    minimum_widths = {"room": 44, "portal": 34, "container": 44, "object": 34}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for node_type, row_nodes in by_type.items():
        available_per_node = (x_max - x_min) / max(1, len(row_nodes))
        node_width = int(
            min(
                nominal_sizes[node_type][0],
                max(minimum_widths[node_type], available_per_node - 5),
            )
        )
        node_height = nominal_sizes[node_type][1]
        for node in row_nodes:
            node_id = str(node.get("id") or "")
            center = positions.get(node_id)
            if center is None:
                continue
            x1 = max(x_min, center[0] - node_width // 2)
            x2 = min(width - 3, center[0] + node_width // 2)
            y1 = max(graph_top, center[1] - node_height // 2)
            y2 = min(graph_bottom, center[1] + node_height // 2)
            boxes[node_id] = (x1, y1, x2, y2)
            positions[node_id] = ((x1 + x2) // 2, (y1 + y2) // 2)
    return {
        "positions": positions,
        "boxes": boxes,
        "row_y": row_y,
        "left_gutter": left_gutter,
        "graph_top": graph_top,
        "graph_bottom": graph_bottom,
    }


def topology_node_style(node: dict[str, Any]) -> dict[str, Any]:
    node_type = str(node.get("type") or "object")
    state = str((node.get("interaction") or {}).get("state") or "unknown").casefold()
    if node_type == "room":
        return {"fill": (235, 238, 248), "border": (45, 45, 45), "dashed": False}
    if node_type == "portal":
        if state in {"open", "ajar", "static_open"}:
            return {"fill": (236, 250, 236), "border": (45, 175, 70), "dashed": False}
        if state in {"closed", "static_closed"}:
            return {"fill": (244, 244, 252), "border": (35, 35, 210), "dashed": True}
        return {"fill": (245, 245, 245), "border": (130, 130, 130), "dashed": True}
    if node_type == "container":
        return {
            "fill": (242, 246, 252),
            "border": (35, 135, 238),
            "dashed": state not in {"open", "ajar", "static_open"},
        }
    return {"fill": (238, 242, 250), "border": (60, 60, 60), "dashed": False}


def latest_state_change(events: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    for event in reversed(events or []):
        if str(event.get("event") or "") == "STATE_CHANGED":
            return event
    return None


def interaction_state_color(state: str) -> tuple[int, int, int]:
    state = str(state).lower()
    if state in {"open", "ajar", "static_open"}:
        return (55, 185, 70)
    if state == "closed":
        return (55, 70, 225)
    return (45, 180, 235)


def room_style_by_id(graph: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    result = {}
    for node in (graph or {}).get("nodes") or []:
        if str(node.get("type") or "") != "room":
            continue
        room_id = node.get("room_id")
        if room_id is None:
            node_id = str(node.get("id") or "")
            if node_id.startswith("room_"):
                try:
                    room_id = int(node_id.split("_", 1)[1])
                except ValueError:
                    continue
        try:
            room_id = int(room_id)
        except (TypeError, ValueError):
            continue
        attributes = node.get("attributes") or {}
        style = str(
            attributes.get("room_attribute")
            or node.get("label")
            or node.get("name")
            or "unknown"
        )
        confidence = float(attributes.get("room_attribute_confidence", 0.0) or 0.0)
        result[room_id] = {
            "style": style,
            "confidence": confidence,
            "label": f"Room {room_id} | {style} | {confidence:.2f}",
        }
    return result
