"""Small, dependency-free occupancy checks for candidate preflight.

The decision node still treats ``move_base/make_plan`` as the authoritative
planner check.  These helpers are only a cheap first pass over the latest
``OccupancyGrid``: an explicitly occupied cell or a point outside the map is
unsafe, while an unknown cell is retained and left for the planner/costmap to
resolve.  Keeping the implementation independent of ROS makes it usable by
offline replay and unit tests as well as the ROS candidate node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


STATUS_FREE = "free"
STATUS_UNKNOWN = "unknown"
STATUS_OCCUPIED = "occupied"
STATUS_OUT_OF_BOUNDS = "out_of_bounds"
STATUS_UNAVAILABLE = "unavailable"
STATUS_NO_PATH = "no_path"
STATUS_SEARCH_LIMIT = "search_limit"


@dataclass
class _PathSearch:
    heap: list = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)
    settled: set = field(default_factory=set)
    expanded: int = 0
    unknown_seen: bool = False


class OccupancyTraversal:
    """Share footprint classification across a batch of anchor searches.

    Own one immutable ROS grid receipt for the duration of a candidate batch.
    Construct a new instance for the next batch so map/parameter changes cannot
    reuse stale collision results. Shared searches use Dijkstra's ordering so
    settled costs are valid for every goal, independent of query order.
    """

    def __init__(self, grid: Any, *, robot_radius_m: float = 0.0,
                 safety_margin_m: float = 0.0, occupied_threshold: int = 50):
        self.grid = grid
        self.context = _grid_context(grid)
        self.radius = max(0.0, _finite(robot_radius_m) + _finite(safety_margin_m))
        self.threshold = int(occupied_threshold)
        self.states: dict[tuple[int, int], str] = {}
        self.cache_hits = 0
        self.point_states: dict[tuple[float, float], str] = {}
        self.point_cache_hits = 0
        self.searches: dict[tuple, _PathSearch] = {}
        resolution = self.context[0][2] if self.context else 1.0
        cells = int(math.ceil(self.radius / resolution + 0.5)) if self.radius else 0
        self.kernel = [(dx, dy) for dy in range(-cells, cells + 1)
                       for dx in range(-cells, cells + 1)
                       if ((max(abs(dx) - 0.5, 0.0) * resolution) ** 2
                           + (max(abs(dy) - 0.5, 0.0) * resolution) ** 2
                           <= self.radius ** 2 + 1e-12)]

    def cell_state(self, cell: tuple[int, int], *, start: bool = False) -> str:
        if self.context is None:
            return STATUS_UNAVAILABLE
        if not start and cell in self.states:
            self.cache_hits += 1
            return self.states[cell]
        geometry, data = self.context
        width, height = geometry[:2]
        unknown = False
        status = STATUS_FREE
        for dx, dy in self.kernel:
            x, y = cell[0] + dx, cell[1] + dy
            if not (0 <= x < width and 0 <= y < height):
                status = STATUS_OUT_OF_BOUNDS
                break
            try:
                value = int(data[y * width + x])
            except (IndexError, TypeError, ValueError):
                status = STATUS_UNAVAILABLE
                break
            if value < 0:
                unknown = True
            elif value >= self.threshold and not (start and dx == 0 and dy == 0):
                status = STATUS_OCCUPIED
                break
        else:
            if unknown:
                status = STATUS_UNKNOWN
        # The start exception must never leak into an ordinary footprint test.
        if not start:
            self.states[cell] = status
        return status

    def point_state(self, x: float, y: float) -> str:
        if self.context is None:
            return STATUS_UNAVAILABLE
        try:
            point = (float(x), float(y))
        except (TypeError, ValueError, OverflowError):
            return STATUS_UNAVAILABLE
        if not all(math.isfinite(value) for value in point):
            return STATUS_UNAVAILABLE
        if point in self.point_states:
            self.point_cache_hits += 1
            return self.point_states[point]
        result = _footprint_status_context(self.context, *point, self.radius, self.threshold)
        self.point_states[point] = result
        return result


def _grid_context(
    grid: Any,
) -> tuple[tuple[int, int, float, float, float, float], Any] | None:
    """Validate and snapshot grid geometry/data once for a hot-path query.

    Candidate generation can evaluate many options and the bounded A* fallback
    can inspect thousands of cells.  Re-reading ROS message attributes and
    recomputing the origin yaw for every sample is unnecessary overhead.
    """

    geometry = _grid_geometry(grid)
    if geometry is None:
        return None
    width, height, _resolution, _origin_x, _origin_y, _origin_yaw = geometry
    try:
        data = getattr(grid, "data")
        if len(data) < width * height:
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    return geometry, data


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _grid_geometry(grid: Any) -> tuple[int, int, float, float, float, float] | None:
    """Return ``width,height,resolution,origin_x,origin_y,origin_yaw``."""

    if grid is None:
        return None
    info = getattr(grid, "info", None)
    try:
        width = int(getattr(info, "width"))
        height = int(getattr(info, "height"))
        resolution = _finite(getattr(info, "resolution"), 0.0)
        origin = getattr(info, "origin")
        position = getattr(origin, "position")
        orientation = getattr(origin, "orientation")
        origin_x = _finite(getattr(position, "x"))
        origin_y = _finite(getattr(position, "y"))
        qx = _finite(getattr(orientation, "x"))
        qy = _finite(getattr(orientation, "y"))
        qz = _finite(getattr(orientation, "z"))
        qw = _finite(getattr(orientation, "w"), 1.0)
    except (AttributeError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return None
    origin_yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return width, height, resolution, origin_x, origin_y, origin_yaw


def _world_to_cell(grid: Any, x: float, y: float) -> tuple[int, int] | None:
    geometry = _grid_geometry(grid)
    if geometry is None:
        return None
    _width, _height, resolution, origin_x, origin_y, origin_yaw = geometry
    dx = float(x) - origin_x
    dy = float(y) - origin_y
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    return int(math.floor(local_x / resolution)), int(math.floor(local_y / resolution))


def _world_to_cell_geometry(
    geometry: tuple[int, int, float, float, float, float],
    x: float,
    y: float,
    *,
    cos_yaw: float | None = None,
    sin_yaw: float | None = None,
) -> tuple[int, int]:
    """Convert a world point using already-decoded grid geometry."""

    _width, _height, resolution, origin_x, origin_y, origin_yaw = geometry
    dx = float(x) - origin_x
    dy = float(y) - origin_y
    if cos_yaw is None:
        cos_yaw = math.cos(origin_yaw)
    if sin_yaw is None:
        sin_yaw = math.sin(origin_yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    return int(math.floor(local_x / resolution)), int(math.floor(local_y / resolution))


def _cell_status(grid: Any, grid_x: int, grid_y: int, occupied_threshold: int) -> str:
    geometry = _grid_geometry(grid)
    if geometry is None:
        return STATUS_UNAVAILABLE
    width, height, _resolution, _origin_x, _origin_y, _origin_yaw = geometry
    if not (0 <= int(grid_x) < width and 0 <= int(grid_y) < height):
        return STATUS_OUT_OF_BOUNDS
    try:
        data = getattr(grid, "data")
        value = int(data[int(grid_y) * width + int(grid_x)])
    except (AttributeError, IndexError, TypeError, ValueError):
        return STATUS_UNAVAILABLE
    if value < 0:
        return STATUS_UNKNOWN
    return STATUS_OCCUPIED if value >= int(occupied_threshold) else STATUS_FREE


def _cell_status_context(
    context: tuple[tuple[int, int, float, float, float, float], Any] | None,
    grid_x: int,
    grid_y: int,
    occupied_threshold: int,
) -> str:
    """Classify a cell without reparsing the ROS grid message."""

    if context is None:
        return STATUS_UNAVAILABLE
    geometry, data = context
    width, height, _resolution, _origin_x, _origin_y, _origin_yaw = geometry
    if not (0 <= int(grid_x) < width and 0 <= int(grid_y) < height):
        return STATUS_OUT_OF_BOUNDS
    try:
        value = int(data[int(grid_y) * width + int(grid_x)])
    except (IndexError, TypeError, ValueError):
        return STATUS_UNAVAILABLE
    if value < 0:
        return STATUS_UNKNOWN
    return STATUS_OCCUPIED if value >= int(occupied_threshold) else STATUS_FREE


def point_occupancy_status(
    grid: Any,
    x: float,
    y: float,
    *,
    occupied_threshold: int = 50,
) -> str:
    """Classify one world point without treating unknown as occupied."""

    context = _grid_context(grid)
    if context is None:
        return STATUS_UNAVAILABLE
    geometry = context[0]
    cell = _world_to_cell_geometry(
        geometry,
        x,
        y,
        cos_yaw=math.cos(geometry[5]),
        sin_yaw=math.sin(geometry[5]),
    )
    return _cell_status_context(context, cell[0], cell[1], occupied_threshold)


def footprint_occupancy_status(
    grid: Any,
    x: float,
    y: float,
    *,
    radius_m: float = 0.0,
    occupied_threshold: int = 50,
) -> str:
    """Classify a circular endpoint footprint.

    ``occupied`` and ``out_of_bounds`` dominate ``unknown``.  A map with no
    data returns ``unavailable`` so callers can safely retain the candidate
    and defer to the normal planner instead of deleting the whole pool.
    """

    context = _grid_context(grid)
    if context is None:
        return STATUS_UNAVAILABLE
    return _footprint_status_context(context, x, y, radius_m, occupied_threshold)


def _footprint_status_context(context, x, y, radius_m, occupied_threshold):
    """Intersect the circle with cell squares, retaining subcell position."""
    geometry = context[0]
    _width, _height, resolution, _origin_x, _origin_y, _origin_yaw = geometry
    try:
        x, y, radius = float(x), float(y), float(radius_m)
    except (TypeError, ValueError, OverflowError):
        return STATUS_UNAVAILABLE
    if not all(math.isfinite(value) for value in (x, y, radius)) or radius < 0:
        return STATUS_UNAVAILABLE
    cos_yaw, sin_yaw = math.cos(geometry[5]), math.sin(geometry[5])
    dx, dy = x - _origin_x, y - _origin_y
    grid_x = (cos_yaw * dx + sin_yaw * dy) / resolution
    grid_y = (-sin_yaw * dx + cos_yaw * dy) / resolution
    center = (math.floor(grid_x), math.floor(grid_y))
    if radius == 0.0:
        return _cell_status_context(context, *center, occupied_threshold)
    phase_x, phase_y = grid_x - center[0], grid_y - center[1]
    radius_cells = int(math.ceil(radius / resolution)) + 1
    saw_unknown = False
    for offset_y in range(-radius_cells, radius_cells + 1):
        distance_y = max(offset_y - phase_y, phase_y - offset_y - 1.0, 0.0) * resolution
        for offset_x in range(-radius_cells, radius_cells + 1):
            distance_x = max(offset_x - phase_x, phase_x - offset_x - 1.0, 0.0) * resolution
            if distance_x * distance_x + distance_y * distance_y > radius * radius + 1e-12:
                continue
            status = _cell_status_context(
                context,
                center[0] + offset_x,
                center[1] + offset_y,
                occupied_threshold,
            )
            if status in {STATUS_OCCUPIED, STATUS_OUT_OF_BOUNDS}:
                return status
            if status == STATUS_UNKNOWN:
                saw_unknown = True
            elif status == STATUS_UNAVAILABLE:
                return STATUS_UNAVAILABLE
    return STATUS_UNKNOWN if saw_unknown else STATUS_FREE


def segment_occupancy_status(
    grid: Any,
    start_xy: tuple[float, float] | list[float] | None,
    goal_xy: tuple[float, float] | list[float] | None,
    *,
    sample_step_m: float | None = None,
    start_ignore_m: float = 0.0,
    occupied_threshold: int = 50,
) -> dict[str, Any]:
    """Sample a straight candidate segment and return a compact trace.

    The result deliberately distinguishes ``unknown`` from a hard collision.
    This avoids turning an incompletely explored map into an artificial empty
    candidate list, while an explicitly occupied or out-of-bounds sample can
    be rejected before an expensive ``make_plan`` RPC.
    """

    context = _grid_context(grid)
    geometry = context[0] if context is not None else None
    if geometry is None:
        return {
            "status": STATUS_UNAVAILABLE,
            "occupied": False,
            "unknown": False,
            "out_of_bounds": False,
            "sample_count": 0,
        }
    if not isinstance(start_xy, (list, tuple)) or not isinstance(goal_xy, (list, tuple)):
        return {
            "status": STATUS_UNAVAILABLE,
            "occupied": False,
            "unknown": False,
            "out_of_bounds": False,
            "sample_count": 0,
        }
    if len(start_xy) < 2 or len(goal_xy) < 2:
        return {
            "status": STATUS_UNAVAILABLE,
            "occupied": False,
            "unknown": False,
            "out_of_bounds": False,
            "sample_count": 0,
        }
    try:
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
        goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
    except (TypeError, ValueError):
        return {
            "status": STATUS_UNAVAILABLE,
            "occupied": False,
            "unknown": False,
            "out_of_bounds": False,
            "sample_count": 0,
        }
    if not all(math.isfinite(value) for value in (start_x, start_y, goal_x, goal_y)):
        return {
            "status": STATUS_UNAVAILABLE,
            "occupied": False,
            "unknown": False,
            "out_of_bounds": False,
            "sample_count": 0,
        }
    _width, _height, resolution, _origin_x, _origin_y, _origin_yaw = geometry
    cos_yaw, sin_yaw = math.cos(_origin_yaw), math.sin(_origin_yaw)
    distance = math.hypot(goal_x - start_x, goal_y - start_y)
    step = max(0.02, _finite(sample_step_m, resolution))
    sample_count = max(1, int(math.ceil(distance / step)))
    ignore = max(0.0, _finite(start_ignore_m, 0.0))
    saw_unknown = False
    first_occupied_distance: float | None = None
    for index in range(sample_count + 1):
        travelled = distance * float(index) / float(sample_count)
        # The current robot footprint is often marked occupied/inflated in the
        # map.  Skip only the explicitly configured initial interval; all
        # later cells, including the endpoint, remain fail-closed.
        if travelled + 1e-9 < ignore:
            continue
        fraction = float(index) / float(sample_count)
        x = start_x + (goal_x - start_x) * fraction
        y = start_y + (goal_y - start_y) * fraction
        cell = _world_to_cell_geometry(
            geometry, x, y, cos_yaw=cos_yaw, sin_yaw=sin_yaw
        )
        status = _cell_status_context(
            context, cell[0], cell[1], occupied_threshold
        )
        if status == STATUS_OCCUPIED:
            first_occupied_distance = travelled
            return {
                "status": STATUS_OCCUPIED,
                "occupied": True,
                "unknown": saw_unknown,
                "out_of_bounds": False,
                "sample_count": index + 1,
                "first_occupied_distance_m": first_occupied_distance,
            }
        if status == STATUS_OUT_OF_BOUNDS:
            return {
                "status": STATUS_OUT_OF_BOUNDS,
                "occupied": False,
                "unknown": saw_unknown,
                "out_of_bounds": True,
                "sample_count": index + 1,
            }
        if status == STATUS_UNKNOWN:
            saw_unknown = True
        elif status == STATUS_UNAVAILABLE:
            return {
                "status": STATUS_UNAVAILABLE,
                "occupied": False,
                "unknown": saw_unknown,
                "out_of_bounds": False,
                "sample_count": index + 1,
            }
    return {
        "status": STATUS_UNKNOWN if saw_unknown else STATUS_FREE,
        "occupied": False,
        "unknown": saw_unknown,
        "out_of_bounds": False,
        "sample_count": sample_count + 1,
    }


def grid_path_status(
    grid: Any,
    start_xy: tuple[float, float] | list[float] | None,
    goal_xy: tuple[float, float] | list[float] | None,
    *,
    robot_radius_m: float = 0.0,
    safety_margin_m: float = 0.0,
    occupied_threshold: int = 50,
    unknown_is_blocked: bool = False,
    max_expansions: int = 20000,
    allow_diagonal: bool = True,
    traversal: OccupancyTraversal | None = None,
) -> dict[str, Any]:
    """Find a short collision-free route on the latest occupancy grid.

    ``segment_occupancy_status`` is intentionally only a *hard-obstacle
    preflight*: a straight ray may cross a wall even though a valid route
    exists around it. This bounded search uses A* for standalone queries and
    resumes one Dijkstra wavefront when a traversal is shared across goals.
    The expansion limit applies to the whole shared search, not per anchor.
    It is not a replacement for the controller's
    final trajectory check, but it prevents the candidate layer from deleting
    every safe subgoal merely because its centre-line is blocked.

    Unknown cells are traversable by default and are reported separately.  A
    physical deployment may set ``unknown_is_blocked`` when it wants a strict
    known-free route.  The start cell is admitted even if the map marks it as
    occupied/inflated (the robot footprint commonly causes that condition).
    The returned trace is JSON-friendly and bounded by ``max_expansions``.
    """

    shared_search = traversal is not None
    if traversal is None:
        traversal = OccupancyTraversal(
            grid, robot_radius_m=robot_radius_m, safety_margin_m=safety_margin_m,
            occupied_threshold=occupied_threshold,
        )
    elif (traversal.grid is not grid
          or traversal.threshold != int(occupied_threshold)
          or traversal.radius != max(0.0, _finite(robot_radius_m) + _finite(safety_margin_m))):
        raise ValueError("Traversal must match the grid and footprint parameters")
    context = traversal.context
    geometry = context[0] if context is not None else None
    result_base = {
        "status": STATUS_UNAVAILABLE,
        "reachable": False,
        "unknown": False,
        "out_of_bounds": False,
        "occupied": False,
        "expanded_count": 0,
        "path_cell_count": 0,
        "path_cost_m": 0.0,
    }
    if geometry is None:
        return result_base
    if not isinstance(start_xy, (list, tuple)) or not isinstance(goal_xy, (list, tuple)):
        return result_base
    if len(start_xy) < 2 or len(goal_xy) < 2:
        return result_base
    try:
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
        goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
    except (TypeError, ValueError):
        return result_base
    if not all(math.isfinite(value) for value in (start_x, start_y, goal_x, goal_y)):
        return result_base

    width, height, resolution, origin_x, origin_y, origin_yaw = geometry
    cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)
    start_cell = _world_to_cell_geometry(
        geometry, start_x, start_y, cos_yaw=cos_yaw, sin_yaw=sin_yaw
    )
    goal_cell = _world_to_cell_geometry(
        geometry, goal_x, goal_y, cos_yaw=cos_yaw, sin_yaw=sin_yaw
    )
    if not (
        0 <= start_cell[0] < width
        and 0 <= start_cell[1] < height
        and 0 <= goal_cell[0] < width
        and 0 <= goal_cell[1] < height
    ):
        return {
            **result_base,
            "status": STATUS_OUT_OF_BOUNDS,
            "out_of_bounds": True,
        }

    cell_state = traversal.cell_state

    start_state = cell_state(start_cell, start=True)
    goal_state = traversal.point_state(goal_x, goal_y)
    if STATUS_UNAVAILABLE in {start_state, goal_state}:
        return result_base
    if goal_state == STATUS_OUT_OF_BOUNDS:
        return {**result_base, "status": STATUS_OUT_OF_BOUNDS, "out_of_bounds": True}
    if goal_state == STATUS_OCCUPIED:
        return {**result_base, "status": STATUS_OCCUPIED, "occupied": True}
    if start_state == STATUS_OUT_OF_BOUNDS:
        return {**result_base, "status": STATUS_OUT_OF_BOUNDS, "out_of_bounds": True}
    if start_state == STATUS_OCCUPIED:
        return {**result_base, "status": STATUS_OCCUPIED, "occupied": True}

    # Keep unknown cells in the search unless explicitly forbidden.  A route
    # through unknown space is still useful evidence for the caller; the
    # caller can require a strict known-free route by setting the flag.
    if unknown_is_blocked and (start_state == STATUS_UNKNOWN or goal_state == STATUS_UNKNOWN):
        return {
            **result_base,
            "status": STATUS_UNKNOWN,
            "unknown": True,
        }

    if start_cell == goal_cell:
        return {
            **result_base,
            "status": STATUS_UNKNOWN if start_state == STATUS_UNKNOWN else STATUS_FREE,
            "reachable": True,
            "unknown": start_state == STATUS_UNKNOWN,
            "path_cell_count": 1,
        }

    if allow_diagonal:
        moves = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
    else:
        moves = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))

    import heapq

    def heuristic(cell: tuple[int, int]) -> float:
        if shared_search:
            return 0.0
        dx = abs(cell[0] - goal_cell[0])
        dy = abs(cell[1] - goal_cell[1])
        return (max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)) * resolution

    search_key = (start_cell, bool(unknown_is_blocked), bool(allow_diagonal))
    search = traversal.searches.get(search_key) if shared_search else None
    if search is None:
        search = _PathSearch(
            heap=[(heuristic(start_cell), 0.0, start_cell)],
            costs={start_cell: 0.0}, unknown_seen=start_state == STATUS_UNKNOWN,
        )
        if shared_search:
            traversal.searches[search_key] = search
    open_heap, best_cost, parent = search.heap, search.costs, search.parents
    max_nodes = max(1, int(max_expansions))

    def reached_result() -> dict[str, Any]:
        path_length = 1
        probe = goal_cell
        path_unknown = start_state == STATUS_UNKNOWN
        while probe in parent:
            path_unknown |= cell_state(probe) == STATUS_UNKNOWN
            previous = parent[probe]
            if previous[0] != probe[0] and previous[1] != probe[1]:
                path_unknown |= cell_state((previous[0], probe[1])) == STATUS_UNKNOWN
                path_unknown |= cell_state((probe[0], previous[1])) == STATUS_UNKNOWN
            probe = previous
            path_length += 1
        return {
            **result_base,
            "status": STATUS_UNKNOWN if path_unknown else STATUS_FREE,
            "reachable": True,
            "unknown": bool(path_unknown),
            "expanded_count": search.expanded,
            "path_cell_count": path_length,
            "path_cost_m": float(best_cost[goal_cell]),
        }

    if goal_cell in search.settled:
        return reached_result()
    while open_heap and search.expanded < max_nodes:
        _priority, cost, current = heapq.heappop(open_heap)
        if current in search.settled or cost > best_cost.get(current, math.inf) + 1e-9:
            continue
        search.expanded += 1
        search.settled.add(current)
        for dx, dy, step_cost in moves:
            neighbour = (current[0] + dx, current[1] + dy)
            new_cost = cost + float(step_cost) * resolution
            # A non-improving edge cannot change the route. Its destination
            # was already classified when best_cost was assigned; avoid
            # repeating footprint and diagonal-corner checks for that edge.
            if new_cost + 1e-9 >= best_cost.get(neighbour, math.inf):
                continue
            state = cell_state(neighbour)
            if state in {STATUS_OUT_OF_BOUNDS, STATUS_OCCUPIED, STATUS_UNAVAILABLE}:
                continue
            if state == STATUS_UNKNOWN:
                if unknown_is_blocked:
                    continue
                search.unknown_seen = True
            # Do not cut diagonally through the corner of two blocked cells.
            if dx and dy:
                side_a = cell_state((current[0] + dx, current[1]))
                side_b = cell_state((current[0], current[1] + dy))
                if side_a in {STATUS_OUT_OF_BOUNDS, STATUS_OCCUPIED} or side_b in {
                    STATUS_OUT_OF_BOUNDS,
                    STATUS_OCCUPIED,
                }:
                    continue
                if STATUS_UNAVAILABLE in {side_a, side_b}:
                    continue
                if unknown_is_blocked and STATUS_UNKNOWN in {side_a, side_b}:
                    continue
            best_cost[neighbour] = new_cost
            parent[neighbour] = current
            heapq.heappush(open_heap, (new_cost + heuristic(neighbour), new_cost, neighbour))
        # Expand the reached goal before returning: the next anchor may need
        # to pass through this cell when resuming the shared wavefront.
        if current == goal_cell:
            return reached_result()

    return {
        **result_base,
        "status": STATUS_SEARCH_LIMIT if open_heap else STATUS_NO_PATH,
        "unknown": bool(search.unknown_seen),
        "expanded_count": search.expanded,
    }
