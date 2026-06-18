from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
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
    information_gain: float
    distance_to_robot: float
    score: float = 0.0
    score_terms: dict[str, float] = field(default_factory=dict)


@dataclass
class FrontierConfig:
    free_max: int = 20
    occupied_min: int = 50
    min_cluster_cells: int = 3
    connect_8: bool = True
    candidate_top_k: int = 12
    sensor_range_m: float = 5.0
    subgoal_search_radius_cells: int = 8
    information_weight: float = 1.0
    distance_weight: float = 0.55
    semantic_weight: float = 0.35
    llm_weight: float = 0.8
    revisit_penalty: float = 0.6
    failure_penalty: float = 1.0


class FrontierExplorerCore:
    """ROS-free frontier extraction and receding-horizon candidate ranking."""

    def __init__(self, config: FrontierConfig | None = None):
        self.config = config or FrontierConfig()

    def extract_frontier_clusters(
        self,
        grid: OccupancyGridData,
        robot_xy: tuple[float, float],
        value_provider=None,
        state=None,
    ) -> list[FrontierCluster]:
        frontier_cells = self._find_frontier_cells(grid)
        raw_clusters = self._cluster_cells(frontier_cells)
        clusters: list[FrontierCluster] = []
        for cells in raw_clusters:
            if len(cells) < self.config.min_cluster_cells:
                continue
            cluster = self._build_cluster(grid, cells, robot_xy)
            if cluster is None:
                continue
            if state is not None and not state.is_cluster_available(cluster):
                continue
            self._score_cluster(cluster, grid, robot_xy, value_provider, state)
            clusters.append(cluster)
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
        if not clusters:
            return None
        ranked = self._receding_horizon_rank(clusters[: max(1, self.config.candidate_top_k)], robot_xy)
        return ranked[0] if ranked else None

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
    ) -> FrontierCluster | None:
        cx = sum(cell[0] for cell in cells) / float(len(cells))
        cy = sum(cell[1] for cell in cells) / float(len(cells))
        centroid_world = grid.spec.grid_to_world(round(cx), round(cy))
        subgoal_cell = self._choose_subgoal_cell(grid, cells, robot_xy)
        if subgoal_cell is None:
            return None
        subgoal_world = grid.spec.grid_to_world(subgoal_cell[0], subgoal_cell[1])
        dist = hypot(subgoal_world[0] - robot_xy[0], subgoal_world[1] - robot_xy[1])
        cluster_id = self._cluster_id(grid, centroid_world)
        return FrontierCluster(
            cluster_id=cluster_id,
            cells=list(cells),
            centroid_cell=(cx, cy),
            centroid_world=centroid_world,
            subgoal_cell=subgoal_cell,
            subgoal_world=subgoal_world,
            information_gain=float(len(cells)),
            distance_to_robot=dist,
        )

    def _choose_subgoal_cell(
        self,
        grid: OccupancyGridData,
        cells: list[tuple[int, int]],
        robot_xy: tuple[float, float],
    ) -> tuple[int, int] | None:
        robot_cell = grid.spec.world_to_grid(robot_xy[0], robot_xy[1])
        candidates: set[tuple[int, int]] = set()
        radius = max(1, self.config.subgoal_search_radius_cells)
        for fx, fy in cells:
            for y in range(fy - radius, fy + radius + 1):
                for x in range(fx - radius, fx + radius + 1):
                    if not grid.spec.in_bounds(x, y):
                        continue
                    if not self._is_free(grid.cell(x, y)):
                        continue
                    candidates.add((x, y))
        if not candidates:
            return None

        def key(cell: tuple[int, int]) -> tuple[float, float]:
            near_frontier = min(hypot(cell[0] - fx, cell[1] - fy) for fx, fy in cells)
            robot_dist = hypot(cell[0] - robot_cell[0], cell[1] - robot_cell[1])
            return near_frontier, robot_dist

        return min(candidates, key=key)

    def _score_cluster(self, cluster: FrontierCluster, grid, robot_xy, value_provider, state) -> None:
        info = self._normalize_information(cluster.information_gain)
        distance_score = 1.0 / (1.0 + cluster.distance_to_robot)
        semantic = float(value_provider.semantic_value(cluster, grid) if value_provider is not None else 0.0)
        llm = float(value_provider.llm_value(cluster, grid) if value_provider is not None else 0.0)
        revisit = float(state.revisit_penalty(cluster) if state is not None else 0.0)
        failure = float(state.failure_penalty(cluster) if state is not None else 0.0)
        score = (
            self.config.information_weight * info
            + self.config.distance_weight * distance_score
            + self.config.semantic_weight * semantic
            + self.config.llm_weight * llm
            - self.config.revisit_penalty * revisit
            - self.config.failure_penalty * failure
        )
        cluster.score = score
        cluster.score_terms = {
            "information": info,
            "distance": distance_score,
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
                key=lambda item: item.score - 0.05 * hypot(item.subgoal_world[0] - cursor[0], item.subgoal_world[1] - cursor[1]),
            )
            ordered.append(next_cluster)
            remaining.remove(next_cluster)
            cursor = next_cluster.subgoal_world
        return ordered

    def _normalize_information(self, value: float) -> float:
        return min(1.0, max(0.0, value / max(float(self.config.min_cluster_cells * 10), 1.0)))

    def _cluster_id(self, grid: OccupancyGridData, world_xy: tuple[float, float]) -> str:
        bucket = max(grid.spec.resolution * 4.0, 0.25)
        return f"{round(world_xy[0] / bucket):d}:{round(world_xy[1] / bucket):d}"

    def _is_free(self, value: int | None) -> bool:
        return value is not None and 0 <= int(value) <= self.config.free_max

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
