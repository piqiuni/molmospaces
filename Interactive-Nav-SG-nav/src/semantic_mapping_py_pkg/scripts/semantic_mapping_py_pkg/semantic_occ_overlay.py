from __future__ import annotations

import math
from typing import Any

import numpy as np


class OverlayUpdateRegionTracker:
    """Build map diffs plus persistent updates for the current semantic overlay."""

    def __init__(self, retired_bounds_hold_builds=20):
        self.previous_bounds: dict[str, int] | None = None
        self.previous_data: np.ndarray | None = None
        self.geometry_key: Any = None
        self.retired_bounds: dict[str, int] | None = None
        self.retired_bounds_hold_builds = max(0, int(retired_bounds_hold_builds))
        self.retired_bounds_remaining = 0

    def reset(self) -> None:
        self.previous_bounds = None
        self.previous_data = None
        self.geometry_key = None
        self.retired_bounds = None
        self.retired_bounds_remaining = 0

    def build(self, width, height, planning_data, current_bounds, geometry_key=None):
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0 or len(planning_data) != width * height:
            return None
        current_data = np.asarray(planning_data, dtype=np.int8)
        geometry_changed = geometry_key != self.geometry_key
        if geometry_changed:
            self.previous_bounds = None
            self.previous_data = None
            self.retired_bounds = None
            self.retired_bounds_remaining = 0
            self.geometry_key = geometry_key

        current = self._normalize_bounds(current_bounds, width, height)
        if self.previous_bounds is not None and self.previous_bounds != current:
            self.retired_bounds = self._union_bounds(self.retired_bounds, self.previous_bounds)
            self.retired_bounds_remaining = self.retired_bounds_hold_builds
        changed_bounds = self._changed_bounds(current_data, self.previous_data, width)
        update_bounds = self._union_bounds(self.previous_bounds, current)
        update_bounds = self._union_bounds(update_bounds, self.retired_bounds)
        update_bounds = self._union_bounds(update_bounds, changed_bounds)
        self.previous_bounds = current
        self.previous_data = current_data.copy()
        if self.retired_bounds is not None:
            self.retired_bounds_remaining -= 1
            if self.retired_bounds_remaining <= 0:
                self.retired_bounds = None
                self.retired_bounds_remaining = 0
        if update_bounds is None:
            return None

        x = update_bounds["x"]
        y = update_bounds["y"]
        update_width = update_bounds["width"]
        update_height = update_bounds["height"]
        data = (
            current_data.reshape(height, width)[y : y + update_height, x : x + update_width]
            .reshape(-1)
            .astype(np.int16)
            .tolist()
        )
        return {**update_bounds, "data": data}

    @staticmethod
    def _changed_bounds(current_data, previous_data, width):
        if previous_data is None or previous_data.shape != current_data.shape:
            return None
        changed = np.flatnonzero(current_data != previous_data)
        if changed.size == 0:
            return None
        rows = changed // int(width)
        cols = changed % int(width)
        x = int(cols.min())
        y = int(rows.min())
        return {
            "x": x,
            "y": y,
            "width": int(cols.max()) - x + 1,
            "height": int(rows.max()) - y + 1,
        }

    @staticmethod
    def _normalize_bounds(bounds, width, height):
        if not bounds:
            return None
        x = max(0, int(bounds.get("x", 0)))
        y = max(0, int(bounds.get("y", 0)))
        x_end = min(width, x + max(0, int(bounds.get("width", 0))))
        y_end = min(height, y + max(0, int(bounds.get("height", 0))))
        if x >= x_end or y >= y_end:
            return None
        return {"x": x, "y": y, "width": x_end - x, "height": y_end - y}

    @staticmethod
    def _union_bounds(first, second):
        if first is None:
            return dict(second) if second is not None else None
        if second is None:
            return dict(first)
        x = min(first["x"], second["x"])
        y = min(first["y"], second["y"])
        x_end = max(first["x"] + first["width"], second["x"] + second["width"])
        y_end = max(first["y"] + first["height"], second["y"] + second["height"])
        return {"x": x, "y": y, "width": x_end - x, "height": y_end - y}


class SemanticOccupancyOverlay:
    """Persistently clear open portal AABBs from a raw occupancy grid."""

    def __init__(self, enabled=True, clear_padding_m=0.10, open_states=None):
        self.enabled = bool(enabled)
        self.clear_padding_m = max(0.0, float(clear_padding_m))
        self.open_states = set(open_states or ["open"])
        self.reference_aabbs: dict[str, tuple[list[float], list[float]]] = {}
        self.active_portal_ids: set[str] = set()
        self.pending_portal_ids: set[str] = set()

    def reset(self) -> None:
        self.reference_aabbs.clear()
        self.active_portal_ids.clear()
        self.pending_portal_ids.clear()

    def set_interaction_pending(self, node_id: str, pending: bool) -> bool:
        node_id = str(node_id or "")
        if not node_id:
            return False
        before = set(self.pending_portal_ids)
        if pending:
            self.pending_portal_ids.add(node_id)
        else:
            self.pending_portal_ids.discard(node_id)
        return before != self.pending_portal_ids

    def update_graph(self, graph_payload: dict[str, Any]) -> None:
        active = set()
        for node in graph_payload.get("nodes") or []:
            if node.get("type") != "portal":
                continue
            node_id = str(node.get("id") or "")
            center = self._point3(node.get("aabb_center"))
            size = self._point3(node.get("aabb_size"))
            if not node_id or center is None or size is None or size[0] <= 0.0 or size[1] <= 0.0:
                continue
            state = str((node.get("interaction") or {}).get("state") or "unknown")
            if node_id not in self.reference_aabbs or state == "closed":
                self.reference_aabbs[node_id] = (center, size)
            if state in self.open_states and node_id in self.reference_aabbs:
                active.add(node_id)
        self.active_portal_ids = active | self.pending_portal_ids

    def apply(self, grid_info: Any, raw_data: list[int]) -> tuple[list[int], list[int], dict[str, Any]]:
        width = int(grid_info.width)
        height = int(grid_info.height)
        cell_count = width * height
        result = [int(value) for value in raw_data]
        mask = [0] * cell_count
        if len(result) != cell_count:
            return result, mask, {
                "active_portal_ids": [],
                "cleared_cells": 0,
                "update_bounds": None,
                "valid": False,
            }
        if not self.enabled or not self.active_portal_ids:
            return result, mask, {
                "active_portal_ids": [],
                "cleared_cells": 0,
                "update_bounds": None,
                "valid": True,
            }

        resolution = float(grid_info.resolution)
        if resolution <= 0.0:
            return result, mask, {
                "active_portal_ids": [],
                "cleared_cells": 0,
                "update_bounds": None,
                "valid": False,
            }
        origin = grid_info.origin
        origin_x = float(origin.position.x)
        origin_y = float(origin.position.y)
        origin_yaw = self._quaternion_yaw(origin.orientation)
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)

        cleared_cells = 0
        applied_ids = []
        bounds = None
        for node_id in sorted(self.active_portal_ids):
            reference = self.reference_aabbs.get(node_id)
            if reference is None:
                continue
            center, size = reference
            half_x = 0.5 * float(size[0]) + self.clear_padding_m
            half_y = 0.5 * float(size[1]) + self.clear_padding_m
            local_corners = []
            for wx in (float(center[0]) - half_x, float(center[0]) + half_x):
                for wy in (float(center[1]) - half_y, float(center[1]) + half_y):
                    dx = wx - origin_x
                    dy = wy - origin_y
                    local_x = cos_yaw * dx + sin_yaw * dy
                    local_y = -sin_yaw * dx + cos_yaw * dy
                    local_corners.append((local_x, local_y))
            min_x = min(point[0] for point in local_corners)
            max_x = max(point[0] for point in local_corners)
            min_y = min(point[1] for point in local_corners)
            max_y = max(point[1] for point in local_corners)
            col_min = max(0, int(math.floor(min_x / resolution)))
            col_max = min(width - 1, int(math.ceil(max_x / resolution) - 1))
            row_min = max(0, int(math.floor(min_y / resolution)))
            row_max = min(height - 1, int(math.ceil(max_y / resolution) - 1))
            if col_min > col_max or row_min > row_max:
                continue
            applied_ids.append(node_id)
            portal_bounds = {
                "x": col_min,
                "y": row_min,
                "width": col_max - col_min + 1,
                "height": row_max - row_min + 1,
            }
            bounds = OverlayUpdateRegionTracker._union_bounds(bounds, portal_bounds)
            for row in range(row_min, row_max + 1):
                offset = row * width
                for col in range(col_min, col_max + 1):
                    index = offset + col
                    if result[index] != 0:
                        cleared_cells += 1
                    result[index] = 0
                    mask[index] = 100
        return result, mask, {
            "active_portal_ids": applied_ids,
            "cleared_cells": cleared_cells,
            "update_bounds": bounds,
            "valid": True,
        }

    @staticmethod
    def _point3(values: Any) -> list[float] | None:
        vals = list(values or [])
        if len(vals) < 3:
            return None
        try:
            return [float(vals[0]), float(vals[1]), float(vals[2])]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _quaternion_yaw(quaternion: Any) -> float:
        x = float(getattr(quaternion, "x", 0.0))
        y = float(getattr(quaternion, "y", 0.0))
        z = float(getattr(quaternion, "z", 0.0))
        w = float(getattr(quaternion, "w", 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
