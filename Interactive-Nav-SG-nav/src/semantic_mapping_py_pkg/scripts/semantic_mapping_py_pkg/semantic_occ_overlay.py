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
    """Persistently clear narrow, verified portal apertures for planning.

    The raw occupancy map is never changed.  This overlay is deliberately more
    conservative than the old ``open portal AABB -> free`` rule: an open door
    contributes only an inset doorway slab.  In particular, the broad AABB of
    a rotating door leaf must not erase either of the walls next to the door.
    """

    def __init__(
        self,
        enabled=True,
        clear_padding_m=-0.05,
        open_states=None,
        max_aperture_thickness_m=0.25,
        raw_free_confirmations=3,
    ):
        self.enabled = bool(enabled)
        # This is a *signed* inset/outset applied to the immutable closed-door
        # reference.  The safe default is an inset of 5 cm on every side.  A
        # positive value remains supported for legacy configs, but the slab
        # thickness cap below still prevents it from clearing a whole wall.
        self.clear_padding_m = float(clear_padding_m)
        self.max_aperture_thickness_m = max(
            0.0, float(max_aperture_thickness_m)
        )
        self.open_states = set(open_states or ["open"])
        self.raw_free_confirmations = max(1, int(raw_free_confirmations))
        self.reference_aabbs: dict[str, tuple[list[float], list[float]]] = {}
        # A pending interaction command can arrive before its result, but it
        # must never be enough to turn an arbitrary openable object into a map
        # opening.  Keep the current graph's topology-qualified portal IDs
        # separately from the mutable presentation ``node.type`` field.
        self.known_portal_ids: set[str] = set()
        self.active_portal_ids: set[str] = set()
        self.pending_portal_ids: set[str] = set()
        # A successful open is a map-state transition, not a traversal lease.
        # Keep its aperture alive even if the graph/traversal snapshot briefly
        # disappears.  Raw-free streaks are diagnostic only: retiring the
        # overlay after a few free frames allowed a later noisy/depth return to
        # write the opened leaf back into planning OCC permanently.
        self.confirmed_portal_ids: set[str] = set()
        self.raw_free_portal_ids: set[str] = set()
        self.raw_free_streaks: dict[str, int] = {}
        self.graph_portal_states: dict[str, str] = {}

    def reset(self) -> None:
        self.reference_aabbs.clear()
        self.known_portal_ids.clear()
        self.active_portal_ids.clear()
        self.pending_portal_ids.clear()
        self.confirmed_portal_ids.clear()
        self.raw_free_portal_ids.clear()
        self.raw_free_streaks.clear()
        self.graph_portal_states.clear()

    def set_interaction_pending(
        self, node_id: str, pending: bool, *, node_type: str | None = None
    ) -> bool:
        """Record an in-flight *portal* open, never a generic container open.

        ``node_type`` is optional because the normal graph refresh has already
        established ``known_portal_ids``.  Callers that have a freshly checked
        portal command may pass it while the next graph snapshot is still in
        flight.  A non-portal hint is deliberately fail-closed.
        """

        node_id = str(node_id or "")
        if not node_id:
            return False
        before = set(self.pending_portal_ids)
        if pending:
            normalized_type = str(node_type or "").strip().casefold()
            if normalized_type and normalized_type != "portal":
                return False
            if not normalized_type and node_id not in self.known_portal_ids:
                return False
            self.pending_portal_ids.add(node_id)
        else:
            self.pending_portal_ids.discard(node_id)
        return before != self.pending_portal_ids

    def update_graph(self, graph_payload: dict[str, Any]) -> None:
        known = set()
        present = set()
        for node in graph_payload.get("nodes") or []:
            node_id = str(node.get("id") or "")
            if node_id:
                present.add(node_id)
            if not self._is_topology_portal(node):
                continue
            if node_id:
                known.add(node_id)
            attributes = node.get("attributes") or {}
            # The visual AABB can follow a rotating door leaf.  A semantic
            # portal is anchored to the immutable pre-open doorway geometry
            # whenever that reference is available.
            center = self._point3(
                attributes.get("interaction_reference_aabb_center")
                or node.get("aabb_center")
            )
            size = self._point3(
                attributes.get("interaction_reference_aabb_size")
                or node.get("aabb_size")
            )
            if not node_id or center is None or size is None or size[0] <= 0.0 or size[1] <= 0.0:
                continue
            state = str((node.get("interaction") or {}).get("state") or "unknown")
            if node_id not in self.reference_aabbs or state == "closed":
                self.reference_aabbs[node_id] = (center, size)
            previous_state = self.graph_portal_states.get(node_id)
            if state == "closed":
                self.confirmed_portal_ids.discard(node_id)
                self.raw_free_portal_ids.discard(node_id)
                self.raw_free_streaks.pop(node_id, None)
            elif (
                state in self.open_states
                and previous_state not in self.open_states
                and node_id not in self.raw_free_portal_ids
                and node_id in self.reference_aabbs
            ):
                self.confirmed_portal_ids.add(node_id)
                self.raw_free_streaks[node_id] = 0
            self.graph_portal_states[node_id] = state
        reclassified = self.known_portal_ids.intersection(present).difference(known)
        for node_id in reclassified:
            self.confirmed_portal_ids.discard(node_id)
            self.raw_free_portal_ids.discard(node_id)
            self.raw_free_streaks.pop(node_id, None)
            self.graph_portal_states.pop(node_id, None)
        self.known_portal_ids = known
        # A node that was reclassified from a transient MLLM portal proposal
        # to a source-observed container must immediately lose both its cached
        # doorway AABB and optimistic pending clear.
        self.reference_aabbs = {
            node_id: reference
            for node_id, reference in self.reference_aabbs.items()
            if node_id in known or node_id in self.confirmed_portal_ids
        }
        self.pending_portal_ids.intersection_update(known)
        self.active_portal_ids = self.confirmed_portal_ids | self.pending_portal_ids

    @staticmethod
    def _is_topology_portal(node: dict[str, Any]) -> bool:
        """Return whether a graph node is eligible to clear occupancy.

        Module 1 may suggest a semantic class while looking at a partial box.
        It must not promote a source-observed container into a topological
        portal for planning.  Newer graph payloads carry the immutable
        ``topology_type`` / ``observation_node_type`` provenance; legacy
        payloads without either key retain their historical portal behavior.
        """

        if str(node.get("type") or "").casefold() != "portal":
            return False
        attributes = node.get("attributes") or {}
        source_type = str(
            attributes.get("topology_type")
            or attributes.get("observation_node_type")
            or ""
        ).strip().casefold()
        return source_type in {"", "portal"}

    def has_active_portals(self, *, include_pending: bool = True) -> bool:
        """Whether this consumer needs a materialized overlay right now.

        Room topology deliberately excludes optimistic, in-flight interaction
        clears.  Keeping that distinction here lets its hot path use the raw
        occupancy message without allocating a full data copy and zero mask.
        """

        if not self.enabled:
            return False
        if include_pending:
            return bool(self.active_portal_ids)
        return bool(self.active_portal_ids.difference(self.pending_portal_ids))

    def apply(
        self,
        grid_info: Any,
        raw_data: list[int],
        *,
        include_pending: bool = True,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        """Apply portal clear regions to an occupancy grid.

        Planning may optimistically clear a portal while an object-skill is in
        flight so that the controller can keep its local map current.  Room
        topology must be stricter: it may only use a clear region after the
        successful interaction has updated the graph state.  ``include_pending``
        keeps those two consumers on the same implementation without letting a
        failed open create an observed room.
        """
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
        active_portal_ids = set(self.active_portal_ids)
        if not include_pending:
            active_portal_ids.difference_update(self.pending_portal_ids)
        if not self.enabled or not active_portal_ids:
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
        for node_id in sorted(active_portal_ids):
            reference = self.reference_aabbs.get(node_id)
            if reference is None:
                continue
            center, size = reference
            half_x, half_y = self._aperture_half_extents(
                float(size[0]),
                float(size[1]),
                resolution,
            )
            if half_x <= 0.0 or half_y <= 0.0:
                continue
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
            selected_cols = []
            selected_rows = []
            selected_indices = []
            for row in range(row_min, row_max + 1):
                offset = row * width
                for col in range(col_min, col_max + 1):
                    # Bounds above are deliberately conservative so that a
                    # rotated occupancy-grid origin cannot clip the aperture.
                    # Clear only cells whose *centres* lie in the narrow world
                    # doorway slab.  This avoids freeing neighbouring wall
                    # cells merely because their square overlaps a thin door.
                    local_x = (float(col) + 0.5) * resolution
                    local_y = (float(row) + 0.5) * resolution
                    world_x = origin_x + cos_yaw * local_x - sin_yaw * local_y
                    world_y = origin_y + sin_yaw * local_x + cos_yaw * local_y
                    if (
                        abs(world_x - float(center[0])) > half_x + 1e-9
                        or abs(world_y - float(center[1])) > half_y + 1e-9
                    ):
                        continue
                    index = offset + col
                    selected_indices.append(index)
                    selected_cols.append(col)
                    selected_rows.append(row)
            if not selected_cols:
                continue
            if node_id in self.confirmed_portal_ids:
                raw_is_free = all(result[index] == 0 for index in selected_indices)
                streak = self.raw_free_streaks.get(node_id, 0) + 1 if raw_is_free else 0
                self.raw_free_streaks[node_id] = streak
                if streak >= self.raw_free_confirmations:
                    self.raw_free_portal_ids.add(node_id)
                    # Do not retire ``confirmed_portal_ids``.  The graph's open
                    # state is the authority until an explicit closed update;
                    # raw OCC can flicker back to occupied after this point.
            for index in selected_indices:
                if result[index] != 0:
                    cleared_cells += 1
                result[index] = 0
                mask[index] = 100
            applied_ids.append(node_id)
            portal_bounds = {
                "x": min(selected_cols),
                "y": min(selected_rows),
                "width": max(selected_cols) - min(selected_cols) + 1,
                "height": max(selected_rows) - min(selected_rows) + 1,
            }
            bounds = OverlayUpdateRegionTracker._union_bounds(bounds, portal_bounds)
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

    def _aperture_half_extents(
        self,
        size_x: float,
        size_y: float,
        resolution: float,
    ) -> tuple[float, float]:
        """Return an inset, thin doorway slab for an axis-aligned reference.

        The immutable reference AABB stores the closed doorway.  Its longer
        horizontal axis is the opening width; its shorter axis is the wall/door
        thickness.  A 5 cm inset is applied to both, but a rasterized aperture
        must retain at least one map cell or it could disappear entirely for a
        thin reference.  The thickness is also capped, which is the fail-safe
        for coarse or nearly square source AABBs.
        """

        extent_x = self._inset_extent(size_x, resolution)
        extent_y = self._inset_extent(size_y, resolution)
        if extent_x <= 0.0 or extent_y <= 0.0:
            return 0.0, 0.0

        # Use the smaller source axis as the normal/thickness direction.  Keep
        # the lateral inset on the long opening axis, but do not contract the
        # normal axis below the closed-door reference thickness.  At least two
        # raster cells are needed here: a one-cell slab can select only one row
        # when the reference centre is not grid-aligned, leaving the adjacent
        # occupied door row as a complete barrier in the planning map.
        if size_x >= size_y:
            extent_y = self._normal_aperture_extent(size_y, resolution)
        else:
            extent_x = self._normal_aperture_extent(size_x, resolution)
        return 0.5 * extent_x, 0.5 * extent_y

    def _inset_extent(self, source_extent: float, resolution: float) -> float:
        source_extent = max(0.0, float(source_extent))
        if source_extent <= 0.0:
            return 0.0
        inset_extent = source_extent + 2.0 * self.clear_padding_m
        # Apply the configured inset to every portal length, including narrow
        # doors.  Restoring the full measured width for ~1 m doors erased cells
        # occupied by an opened leaf near the jamb, allowing the global plan to
        # cut through geometry the local planner still observed.
        # A negative padding must never erase the portal just because the
        # closed leaf is thinner than twice the requested 5 cm inset.  One map
        # cell is the smallest meaningful clearance in the planning grid.
        return max(float(resolution), inset_extent)

    def _thickness_limit(self, resolution: float) -> float:
        return max(2.0 * float(resolution), self.max_aperture_thickness_m)

    def _normal_aperture_extent(
        self,
        source_extent: float,
        resolution: float,
    ) -> float:
        """Raster-safe doorway thickness without widening the lateral opening."""

        source_extent = max(0.0, float(source_extent))
        if source_extent <= 0.0:
            return 0.0
        # Clear the full bounded normal-axis allowance, not merely the source
        # leaf thickness.  A thin, off-grid leaf can occupy two adjacent rows;
        # using only ``max(source, 2*resolution)`` may select the wrong pair and
        # leave the actual second row sealed.  The configured cap keeps this
        # expansion local to the wall-normal direction.
        return self._thickness_limit(resolution)

    @staticmethod
    def _quaternion_yaw(quaternion: Any) -> float:
        x = float(getattr(quaternion, "x", 0.0))
        y = float(getattr(quaternion, "y", 0.0))
        z = float(getattr(quaternion, "z", 0.0))
        w = float(getattr(quaternion, "w", 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
