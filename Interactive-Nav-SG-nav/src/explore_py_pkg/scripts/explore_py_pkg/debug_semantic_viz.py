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


def portal_positions_between_rooms(
    portals: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None,
    room_positions: dict[str, tuple[int, int]],
    spacing: float = 24.0,
) -> dict[str, tuple[int, int]]:
    """Place connected portals on the line between their two room nodes."""
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
        midpoint_y = (first[1] + second[1]) * 0.5
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
