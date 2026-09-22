from __future__ import annotations

from typing import Any


def room_context_for_xy(
    graph: dict[str, Any], xy: tuple[float, float] | list[float] | None
) -> dict[str, Any]:
    """Return an occupancy/graph room label for a physical XY position.

    Potential rooms are useful door-side labels, but they must not steal a
    normal segmented room simply because their synthetic AABB overlaps the
    doorway.  Prefer an observed room when both contain the point; use a
    potential room only when it is the only available physical label.
    """

    values = list(xy or [])
    if len(values) < 2:
        return {}
    matches: list[tuple[int, float, str, Any, dict[str, Any]]] = []
    for node in graph.get("nodes") or []:
        if str(node.get("type") or "").casefold() != "room":
            continue
        attributes = node.get("attributes") or {}
        if not bool(attributes.get("active", True)):
            continue
        center = list(node.get("aabb_center") or node.get("centroid") or [])
        size = list(node.get("aabb_size") or [])
        room_id = node.get("room_id")
        if room_id is None:
            room_id = node.get("id")
        if len(center) < 2 or len(size) < 2 or room_id in (None, ""):
            continue
        half_x = 0.5 * abs(float(size[0]))
        half_y = 0.5 * abs(float(size[1]))
        if (
            abs(float(values[0]) - float(center[0])) > half_x + 1e-6
            or abs(float(values[1]) - float(center[1])) > half_y + 1e-6
        ):
            continue
        is_potential = bool(attributes.get("is_potential_room"))
        matches.append(
            (
                1 if is_potential else 0,
                max(half_x * half_y, 1e-6),
                str(room_id),
                room_id,
                node,
            )
        )
    if not matches:
        return {}
    _potential_rank, _area, _sortable_room_id, room_id, node = min(matches)
    attributes = node.get("attributes") or {}
    result: dict[str, Any] = {
        "room_id": room_id,
        "potential_room": bool(attributes.get("is_potential_room")),
    }
    source_portal_id = str(attributes.get("source_portal_id") or "")
    if source_portal_id:
        result["source_portal_id"] = source_portal_id
    room_attribute = str(attributes.get("room_attribute") or "").strip()
    if room_attribute and room_attribute.casefold() != "unknown":
        result["room_attribute"] = room_attribute
        result["room_attribute_confidence"] = max(
            0.0,
            min(1.0, float(attributes.get("room_attribute_confidence", 0.0) or 0.0)),
        )
        scores = dict(attributes.get("room_attribute_scores") or {})
        if scores:
            result["room_attribute_scores"] = scores
    return result
