from __future__ import annotations

from dataclasses import dataclass, field
import math
from math import atan2, ceil, hypot
from typing import Iterable


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str = "map"

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        return int((wx - self.origin_x) / self.resolution), int((wy - self.origin_y) / self.resolution)

    def grid_to_world(self, x: int, y: int) -> tuple[float, float]:
        return (
            self.origin_x + (float(x) + 0.5) * self.resolution,
            self.origin_y + (float(y) + 0.5) * self.resolution,
        )


@dataclass
class OccupancyGridData:
    spec: GridSpec
    data: list[int]

    def cell(self, x: int, y: int) -> int | None:
        if not self.spec.in_bounds(x, y):
            return None
        return int(self.data[self.spec.index(x, y)])


@dataclass
class FrontierCluster:
    cluster_id: str
    cells: list[tuple[int, int]]
    centroid_cell: tuple[float, float]
    centroid_world: tuple[float, float]
    subgoal_cell: tuple[int, int]
    subgoal_world: tuple[float, float]
    subgoal_yaw: float
    information_gain: float
    distance_to_robot: float
    unknown_component_area_m2: float = 0.0
    frontier_length_m: float = 0.0
    expected_visible_unknown_area_m2: float = 0.0
    score: float = 0.0
    score_terms: dict[str, float] = field(default_factory=dict)


@dataclass
class FrontierConfig:
    free_max: int = 20
    occupied_min: int = 50
    hard_min_cluster_cells: int = 3
    min_cluster_cells: int = 3
    connect_8: bool = True
    candidate_top_k: int = 12
    sensor_range_m: float = 5.0
    unknown_component_radius_m: float = 5.0
    subgoal_search_radius_cells: int = 8
    min_subgoal_distance_m: float = 0.75
    hard_min_subgoal_distance_m: float = 0.50
    target_frontier_offset_m: float = 0.35
    use_voronoi_viewpoints: bool = True
    min_viewpoint_frontier_distance_m: float = 0.65
    max_viewpoint_frontier_distance_m: float = 2.8
    clearance_weight: float = 0.8
    frontier_offset_weight: float = 0.45
    local_horizon_m: float = 3.0
    local_horizon_penalty: float = 0.35
    far_cluster_penalty: float = 0.45
    far_cluster_penalty_saturation_m: float = 4.0
    min_obstacle_clearance_m: float = 0.25
    max_clearance_check_m: float = 0.8
    robot_radius_m: float = 0.35
    footprint_safety_margin_m: float = 0.10
    require_footprint_free: bool = True
    footprint_unknown_is_free: bool = False
    turning_safety_margin_m: float = 0.25
    require_turning_clearance: bool = True
    information_weight: float = 1.0
    distance_weight: float = 0.55
    semantic_weight: float = 0.35
    llm_weight: float = 0.8
    revisit_penalty: float = 0.6
    failure_penalty: float = 1.0
    receding_distance_weight: float = 0.15
    previous_subgoal_weight: float = 0.35
    continuity_cost_weight: float = 0.25
    continuity_cost_saturation_m: float = 4.0
    near_frontier_relax_distance_m: float = 1.5
    relaxed_min_viewpoint_frontier_distance_m: float = 0.35
    initial_local_radius_m: float = 2.2
    initial_backward_weight: float = 0.35


class FrontierExplorerCore:
    """ROS-free frontier extraction and receding-horizon candidate ranking."""

    def __init__(self, config: FrontierConfig | None = None):
        self.config = config or FrontierConfig()
        self.last_debug_stats: dict[str, int] = {}

    def extract_frontier_clusters(
        self,
        grid: OccupancyGridData,
        robot_xy: tuple[float, float],
        value_provider=None,
        state=None,
    ) -> list[FrontierCluster]:
        frontier_cells = self._find_frontier_cells(grid)
        raw_clusters = self._cluster_cells(frontier_cells)
        stats = {
            "frontier_cells": len(frontier_cells),
            "raw_clusters": len(raw_clusters),
            "dropped_tiny": 0,
            "dropped_no_viewpoint": 0,
            "dropped_state": 0,
            "kept_clusters": 0,
        }
        clusters: list[FrontierCluster] = []
        for cells in raw_clusters:
            if len(cells) < max(1, self.config.hard_min_cluster_cells):
                stats["dropped_tiny"] += 1
                continue
            cluster = self._build_cluster(grid, cells, robot_xy, state=state)
            if cluster is None:
                stats["dropped_no_viewpoint"] += 1
                continue
            if state is not None and not state.is_cluster_available(cluster):
                stats["dropped_state"] += 1
                continue
            self._score_cluster(cluster, grid, robot_xy, value_provider, state)
            clusters.append(cluster)
        stats["kept_clusters"] = len(clusters)
        self.last_debug_stats = stats
        clusters.sort(key=lambda item: item.score, reverse=True)
        return clusters

    def select_next_cluster(
        self,
        grid: OccupancyGridData,
        robot_xy: tuple[float, float],
        value_provider=None,
        state=None,
    ) -> FrontierCluster | None:
        clusters = self.extract_frontier_clusters(grid, robot_xy, value_provider=value_provider, state=state)
        ranked = self.rank_clusters(clusters, robot_xy, state=state)
        return ranked[0] if ranked else None

    def rank_clusters(
        self,
        clusters: list[FrontierCluster],
        robot_xy: tuple[float, float],
        state=None,
    ) -> list[FrontierCluster]:
        if not clusters:
            return []
        cursor_xy = robot_xy
        if state is not None and getattr(state, "last_subgoal_world", None) is not None:
            cursor_xy = state.last_subgoal_world
        return self._receding_horizon_rank(clusters[: max(1, self.config.candidate_top_k)], cursor_xy)

    def select_initial_local_cluster(
        self,
        clusters: list[FrontierCluster],
        robot_xy: tuple[float, float],
        robot_yaw: float | None = None,
    ) -> FrontierCluster | None:
        if not clusters:
            return None
        radius = max(0.0, self.config.initial_local_radius_m)
        local = [cluster for cluster in clusters if cluster.distance_to_robot <= radius]
        candidates = local if local else list(clusters)

        def key(cluster: FrontierCluster) -> tuple[float, float, float]:
            dx = cluster.subgoal_world[0] - robot_xy[0]
            dy = cluster.subgoal_world[1] - robot_xy[1]
            behind_bonus = 0.0
            if robot_yaw is not None:
                heading_x = math.cos(robot_yaw)
                heading_y = math.sin(robot_yaw)
                dist = max(cluster.distance_to_robot, 1e-6)
                # Positive when the candidate is behind the robot.
                behind_bonus = max(0.0, -(dx * heading_x + dy * heading_y) / dist)
            info_tie_break = self._normalize_information(cluster.information_gain)
            return (
                cluster.distance_to_robot - self.config.initial_backward_weight * behind_bonus,
                -info_tie_break,
                -cluster.score,
            )

        return min(candidates, key=key)

    def has_frontier_near(
        self,
        grid: OccupancyGridData,
        world_xy: tuple[float, float],
        radius_m: float,
        min_cells: int = 1,
    ) -> bool:
        mx, my = grid.spec.world_to_grid(world_xy[0], world_xy[1])
        radius_cells = max(1, int(radius_m / max(grid.spec.resolution, 1e-6)))
        count = 0
        for y in range(my - radius_cells, my + radius_cells + 1):
            for x in range(mx - radius_cells, mx + radius_cells + 1):
                if not grid.spec.in_bounds(x, y):
                    continue
                if hypot(x - mx, y - my) > radius_cells:
                    continue
                if self._is_frontier_cell(grid, x, y):
                    count += 1
                    if count >= min_cells:
                        return True
        return False

    def is_free_world(self, grid: OccupancyGridData, world_xy: tuple[float, float]) -> bool:
        mx, my = grid.spec.world_to_grid(world_xy[0], world_xy[1])
        return self._is_free(grid.cell(mx, my))

    def _find_frontier_cells(self, grid: OccupancyGridData) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for y in range(grid.spec.height):
            for x in range(grid.spec.width):
                if self._is_frontier_cell(grid, x, y):
                    cells.add((x, y))
        return cells

    def _is_frontier_cell(self, grid: OccupancyGridData, x: int, y: int) -> bool:
        if not self._is_free(grid.cell(x, y)):
            return False
        return any(self._is_unknown(grid.cell(nx, ny)) for nx, ny in self._neighbors4(x, y))

    def _cluster_cells(self, cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        clusters: list[list[tuple[int, int]]] = []
        remaining = set(cells)
        neighbor_fn = self._neighbors8 if self.config.connect_8 else self._neighbors4
        while remaining:
            start = remaining.pop()
            cluster = [start]
            queue = [start]
            while queue:
                cx, cy = queue.pop()
                for neighbor in neighbor_fn(cx, cy):
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    cluster.append(neighbor)
            clusters.append(cluster)
        return clusters

    def _build_cluster(
        self,
        grid: OccupancyGridData,
        cells: list[tuple[int, int]],
        robot_xy: tuple[float, float],
        state=None,
    ) -> FrontierCluster | None:
        cx = sum(cell[0] for cell in cells) / float(len(cells))
        cy = sum(cell[1] for cell in cells) / float(len(cells))
        centroid_world = grid.spec.grid_to_world(round(cx), round(cy))
        subgoal_cell = self._choose_subgoal_cell(grid, cells, robot_xy, state=state)
        if subgoal_cell is None:
            return None
        subgoal_world = grid.spec.grid_to_world(subgoal_cell[0], subgoal_cell[1])
        subgoal_yaw = atan2(centroid_world[1] - subgoal_world[1], centroid_world[0] - subgoal_world[0])
        dist = hypot(subgoal_world[0] - robot_xy[0], subgoal_world[1] - robot_xy[1])
        if dist + 1e-6 < self.config.hard_min_subgoal_distance_m:
            return None
        cluster_id = self._cluster_id(grid, centroid_world)
        unknown_component_area_m2 = self._unknown_component_area_m2(
            grid, cells, centroid_cell=(cx, cy)
        )
        frontier_length_m = float(len(cells)) * max(
            float(grid.spec.resolution), 0.0
        )
        expected_visible_unknown_area_m2 = min(
            unknown_component_area_m2,
            frontier_length_m * max(float(self.config.sensor_range_m), 0.0),
        )
        return FrontierCluster(
            cluster_id=cluster_id,
            cells=list(cells),
            centroid_cell=(cx, cy),
            centroid_world=centroid_world,
            subgoal_cell=subgoal_cell,
            subgoal_world=subgoal_world,
            subgoal_yaw=subgoal_yaw,
            information_gain=float(len(cells)),
            distance_to_robot=dist,
            unknown_component_area_m2=unknown_component_area_m2,
            frontier_length_m=frontier_length_m,
            expected_visible_unknown_area_m2=expected_visible_unknown_area_m2,
        )

    def _unknown_component_area_m2(
        self,
        grid: OccupancyGridData,
        frontier_cells: list[tuple[int, int]],
        centroid_cell: tuple[float, float],
    ) -> float:
        """Measure bounded connected unknown space touching a frontier cluster."""
        resolution = max(float(grid.spec.resolution), 1e-6)
        radius_cells = max(
            1,
            int(ceil(float(self.config.unknown_component_radius_m) / resolution)),
        )
        center_x, center_y = centroid_cell

        def inside_window(cell: tuple[int, int]) -> bool:
            return hypot(cell[0] - center_x, cell[1] - center_y) <= radius_cells

        seeds = {
            neighbor
            for cell in frontier_cells
            for neighbor in self._neighbors4(*cell)
            if grid.spec.in_bounds(*neighbor)
            and inside_window(neighbor)
            and self._is_unknown(grid.cell(*neighbor))
        }
        visited: set[tuple[int, int]] = set()
        queue = list(seeds)
        while queue:
            cell = queue.pop()
            if cell in visited:
                continue
            visited.add(cell)
            for neighbor in self._neighbors4(*cell):
                if (
                    neighbor not in visited
                    and grid.spec.in_bounds(*neighbor)
                    and inside_window(neighbor)
                    and self._is_unknown(grid.cell(*neighbor))
                ):
                    queue.append(neighbor)
        return float(len(visited)) * resolution * resolution

    def _choose_subgoal_cell(
        self,
        grid: OccupancyGridData,
        cells: list[tuple[int, int]],
        robot_xy: tuple[float, float],
        state=None,
    ) -> tuple[int, int] | None:
        robot_cell = grid.spec.world_to_grid(robot_xy[0], robot_xy[1])
        candidates: set[tuple[int, int]] = set()
        radius = max(1, self.config.subgoal_search_radius_cells)
        min_view_frontier_cells = self.config.min_viewpoint_frontier_distance_m / max(grid.spec.resolution, 1e-6)
        max_view_frontier_cells = self.config.max_viewpoint_frontier_distance_m / max(grid.spec.resolution, 1e-6)
        for fx, fy in cells:
            for y in range(fy - radius, fy + radius + 1):
                for x in range(fx - radius, fx + radius + 1):
                    if not grid.spec.in_bounds(x, y):
                        continue
                    if not self._is_free(grid.cell(x, y)):
                        continue
                    if state is not None:
                        wx, wy = grid.spec.grid_to_world(x, y)
                        if state.is_goal_point_blocked((wx, wy)):
                            continue
                    candidates.add((x, y))
        if not candidates:
            return None

        resolution = max(grid.spec.resolution, 1e-6)
        min_clearance_cells = max(0, int(round(self.config.min_obstacle_clearance_m / resolution)))
        max_clearance_cells = max(min_clearance_cells, int(round(self.config.max_clearance_check_m / resolution)))
        footprint_radius_cells = max(
            0,
            int(ceil((self.config.robot_radius_m + self.config.footprint_safety_margin_m) / resolution)),
        )
        turning_radius_cells = max(
            footprint_radius_cells,
            int(
                ceil(
                    (
                        self.config.robot_radius_m
                        + self.config.footprint_safety_margin_m
                        + self.config.turning_safety_margin_m
                    )
                    / resolution
                )
            ),
        )
        if self.config.require_footprint_free and footprint_radius_cells > 0:
            candidates = {
                cell
                for cell in candidates
                if self._footprint_is_free(grid, cell, footprint_radius_cells)
            }
            if not candidates:
                return None
        if self.config.require_turning_clearance and turning_radius_cells > footprint_radius_cells:
            turning_candidates = {
                cell
                for cell in candidates
                if self._footprint_is_free(grid, cell, turning_radius_cells)
            }
            if not turning_candidates:
                return None
            candidates = turning_candidates
        clearance_by_cell = {
            cell: self._occupied_clearance_cells(grid, cell, max_clearance_cells)
            for cell in candidates
        }
        nearest_frontier_by_cell = {
            cell: min(hypot(cell[0] - fx, cell[1] - fy) for fx, fy in cells)
            for cell in candidates
        }

        if self.config.use_voronoi_viewpoints:
            relax_dist_cells = self.config.near_frontier_relax_distance_m / max(grid.spec.resolution, 1e-6)
            relaxed_min_view_frontier_cells = (
                self.config.relaxed_min_viewpoint_frontier_distance_m / max(grid.spec.resolution, 1e-6)
            )
            viewpoint_candidates = [
                cell
                for cell in candidates
                if nearest_frontier_by_cell[cell] >= (
                    relaxed_min_view_frontier_cells
                    if hypot(cell[0] - robot_cell[0], cell[1] - robot_cell[1]) <= relax_dist_cells
                    else min_view_frontier_cells
                )
                and nearest_frontier_by_cell[cell] <= max_view_frontier_cells
                and clearance_by_cell[cell] >= min_clearance_cells
            ]
            if not viewpoint_candidates:
                viewpoint_candidates = [
                    cell
                    for cell in candidates
                    if nearest_frontier_by_cell[cell] >= min_view_frontier_cells * 0.5
                    and clearance_by_cell[cell] >= min_clearance_cells
                ]
            if not viewpoint_candidates:
                return None
        else:
            viewpoint_candidates = list(candidates)

        def robot_world_distance(cell: tuple[int, int]) -> float:
            wx, wy = grid.spec.grid_to_world(cell[0], cell[1])
            return hypot(wx - robot_xy[0], wy - robot_xy[1])

        hard_distance_candidates = [
            cell
            for cell in viewpoint_candidates
            if robot_world_distance(cell) >= self.config.hard_min_subgoal_distance_m
        ]
        if not hard_distance_candidates:
            return None
        distant_candidates = [
            cell
            for cell in hard_distance_candidates
            if robot_world_distance(cell) >= self.config.min_subgoal_distance_m
        ]
        if not distant_candidates:
            distant_candidates = hard_distance_candidates

        clear_candidates = [
            cell
            for cell in distant_candidates
            if clearance_by_cell[cell] >= min_clearance_cells
        ]
        if not clear_candidates:
            return None

        frontier_cx = sum(fx for fx, _ in cells) / float(len(cells))
        frontier_cy = sum(fy for _, fy in cells) / float(len(cells))
        frontier_anchor = min(cells, key=lambda item: hypot(item[0] - frontier_cx, item[1] - frontier_cy))
        target_offset_cells = max(1.0, self.config.target_frontier_offset_m / resolution)
        if self.config.use_voronoi_viewpoints:
            target_offset_cells = max(target_offset_cells, min_view_frontier_cells)
        local_horizon_cells = max(1.0, self.config.local_horizon_m / resolution)

        def key(cell: tuple[int, int]) -> tuple[float, float, float, float]:
            center_dist = hypot(cell[0] - frontier_cx, cell[1] - frontier_cy)
            anchor_dist = hypot(cell[0] - frontier_anchor[0], cell[1] - frontier_anchor[1])
            near_frontier = nearest_frontier_by_cell[cell]
            desired_offset_error = abs(near_frontier - target_offset_cells)
            clearance = clearance_by_cell[cell]
            robot_dist = hypot(cell[0] - robot_cell[0], cell[1] - robot_cell[1])
            horizon_penalty = max(0.0, robot_dist - local_horizon_cells)
            ridge_bonus = self._clearance_ridge_bonus(grid, cell, clearance, clearance_by_cell, max_clearance_cells)
            # Frontier picks the information target; this key picks a safer stand-off viewpoint.
            return (
                center_dist + 0.25 * anchor_dist,
                self.config.frontier_offset_weight * desired_offset_error,
                self.config.local_horizon_penalty * horizon_penalty,
                -self.config.clearance_weight * clearance - ridge_bonus,
                robot_dist,
            )

        return min(clear_candidates, key=key)

    def _footprint_is_free(
        self,
        grid: OccupancyGridData,
        cell: tuple[int, int],
        radius_cells: int,
    ) -> bool:
        cx, cy = cell
        radius_sq = radius_cells * radius_cells
        for y in range(cy - radius_cells, cy + radius_cells + 1):
            for x in range(cx - radius_cells, cx + radius_cells + 1):
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) > radius_sq:
                    continue
                if not grid.spec.in_bounds(x, y):
                    return False
                value = grid.cell(x, y)
                if self.config.footprint_unknown_is_free and self._is_unknown(value):
                    continue
                if not self._is_free(value):
                    return False
        return True

    def _score_cluster(self, cluster: FrontierCluster, grid, robot_xy, value_provider, state) -> None:
        info = (
            self._normalize_visible_area(cluster.expected_visible_unknown_area_m2)
            if cluster.expected_visible_unknown_area_m2 > 0.0
            else self._normalize_information(cluster.information_gain)
        )
        distance_score = 1.0 / (1.0 + cluster.distance_to_robot)
        far_overrun_m = max(0.0, cluster.distance_to_robot - self.config.local_horizon_m)
        far_denominator = max(self.config.far_cluster_penalty_saturation_m, 1e-6)
        far_penalty = min(1.0, far_overrun_m / far_denominator)
        semantic = float(value_provider.semantic_value(cluster, grid) if value_provider is not None else 0.0)
        llm = float(value_provider.llm_value(cluster, grid) if value_provider is not None else 0.0)
        revisit = float(state.revisit_penalty(cluster) if state is not None else 0.0)
        failure = float(state.failure_penalty(cluster) if state is not None else 0.0)
        previous_subgoal_score = 0.0
        continuity_cost = 0.0
        cursor_xy = robot_xy
        if state is not None and getattr(state, "last_subgoal_world", None) is not None:
            previous = state.last_subgoal_world
            previous_subgoal_score = 1.0 / (
                1.0 + hypot(cluster.subgoal_world[0] - previous[0], cluster.subgoal_world[1] - previous[1])
            )
            cursor_xy = previous
        continuity_dist = hypot(cluster.subgoal_world[0] - cursor_xy[0], cluster.subgoal_world[1] - cursor_xy[1])
        continuity_cost = min(1.0, continuity_dist / max(self.config.continuity_cost_saturation_m, 1e-6))
        score = (
            self.config.information_weight * info
            + self.config.distance_weight * distance_score
            + self.config.previous_subgoal_weight * previous_subgoal_score
            + self.config.semantic_weight * semantic
            + self.config.llm_weight * llm
            - self.config.far_cluster_penalty * far_penalty
            - self.config.continuity_cost_weight * continuity_cost
            - self.config.revisit_penalty * revisit
            - self.config.failure_penalty * failure
        )
        cluster.score = score
        cluster.score_terms = {
            "information": info,
            "expected_visible_unknown_area_m2": float(
                cluster.expected_visible_unknown_area_m2
            ),
            "distance": distance_score,
            "previous_subgoal": previous_subgoal_score,
            "continuity_cost": continuity_cost,
            "far_cluster_penalty": far_penalty,
            "semantic": semantic,
            "llm": llm,
            "revisit_penalty": revisit,
            "failure_penalty": failure,
        }

    def _receding_horizon_rank(
        self,
        clusters: list[FrontierCluster],
        robot_xy: tuple[float, float],
    ) -> list[FrontierCluster]:
        remaining = list(clusters)
        ordered: list[FrontierCluster] = []
        cursor = robot_xy
        while remaining:
            next_cluster = max(
                remaining,
                key=lambda item: item.score
                - self.config.receding_distance_weight
                * hypot(item.subgoal_world[0] - cursor[0], item.subgoal_world[1] - cursor[1]),
            )
            ordered.append(next_cluster)
            remaining.remove(next_cluster)
            cursor = next_cluster.subgoal_world
        return ordered

    def _normalize_information(self, value: float) -> float:
        return min(1.0, max(0.0, value / max(float(self.config.min_cluster_cells * 10), 1.0)))

    def _normalize_visible_area(self, value: float) -> float:
        reference_area_m2 = max(
            1.0,
            2.0 * float(self.config.sensor_range_m) ** 2,
        )
        return min(1.0, max(0.0, float(value) / reference_area_m2))

    def _cluster_id(self, grid: OccupancyGridData, world_xy: tuple[float, float]) -> str:
        bucket = max(grid.spec.resolution * 4.0, 0.25)
        return f"{round(world_xy[0] / bucket):d}:{round(world_xy[1] / bucket):d}"

    def _is_free(self, value: int | None) -> bool:
        return value is not None and 0 <= int(value) <= self.config.free_max

    def _occupied_clearance_cells(
        self,
        grid: OccupancyGridData,
        cell: tuple[int, int],
        max_radius_cells: int,
    ) -> float:
        max_radius_cells = max(0, max_radius_cells)
        if max_radius_cells == 0:
            return float("inf")
        cx, cy = cell
        best: float | None = None
        for y in range(cy - max_radius_cells, cy + max_radius_cells + 1):
            for x in range(cx - max_radius_cells, cx + max_radius_cells + 1):
                if not grid.spec.in_bounds(x, y):
                    continue
                value = grid.cell(x, y)
                if value is None or int(value) < self.config.occupied_min:
                    continue
                dist = hypot(x - cx, y - cy)
                if dist <= max_radius_cells and (best is None or dist < best):
                    best = dist
        return float(max_radius_cells + 1 if best is None else best)

    def _clearance_ridge_bonus(
        self,
        grid: OccupancyGridData,
        cell: tuple[int, int],
        clearance: float,
        cached_clearance: dict[tuple[int, int], float],
        max_radius_cells: int,
    ) -> float:
        if not self.config.use_voronoi_viewpoints:
            return 0.0
        lower_or_equal = 0
        checked = 0
        for nx, ny in self._neighbors8(cell[0], cell[1]):
            if not grid.spec.in_bounds(nx, ny) or not self._is_free(grid.cell(nx, ny)):
                continue
            checked += 1
            neighbor = cached_clearance.get((nx, ny))
            if neighbor is None:
                neighbor = self._occupied_clearance_cells(grid, (nx, ny), max_radius_cells)
            if clearance >= neighbor:
                lower_or_equal += 1
        if checked == 0:
            return 0.0
        return 0.15 * float(lower_or_equal) / float(checked)

    @staticmethod
    def _is_unknown(value: int | None) -> bool:
        return value is not None and int(value) < 0

    @staticmethod
    def _neighbors4(x: int, y: int) -> Iterable[tuple[int, int]]:
        yield x - 1, y
        yield x + 1, y
        yield x, y - 1
        yield x, y + 1

    @staticmethod
    def _neighbors8(x: int, y: int) -> Iterable[tuple[int, int]]:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yield x + dx, y + dy
