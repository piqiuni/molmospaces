from __future__ import annotations

import json
from math import hypot


class ValueMapFusion:
    """Combines optional semantic and LLM value layers without owning navigation validity."""

    def __init__(self):
        self.scene_id_grid = None
        self.scene_confidence_grid = None
        self.llm_value_grid = None
        self.object_map = []
        self.navigation_hints = []
        self.unified_graph = {}
        self.strategy_bias = {}

    def set_scene_id_grid(self, grid) -> None:
        self.scene_id_grid = grid

    def set_scene_confidence_grid(self, grid) -> None:
        self.scene_confidence_grid = grid

    def set_llm_value_grid(self, grid) -> None:
        self.llm_value_grid = grid

    def set_object_map_json(self, data: str) -> None:
        value = self._loads(data, [])
        self.object_map = value if isinstance(value, list) else []

    def set_navigation_hints_json(self, data: str) -> None:
        value = self._loads(data, [])
        self.navigation_hints = value if isinstance(value, list) else []

    def set_unified_graph_json(self, data: str) -> None:
        value = self._loads(data, {})
        self.unified_graph = value if isinstance(value, dict) else {}

    def set_strategy_bias_json(self, data: str) -> None:
        value = self._loads(data, {})
        self.strategy_bias = value if isinstance(value, dict) else {}

    def semantic_value(self, cluster, grid) -> float:
        object_score = self._object_interest(cluster.subgoal_world)
        room_score = self._room_interest(cluster, grid)
        hint_score = self._navigation_hint_interest(cluster.subgoal_world)
        return max(0.0, min(1.0, object_score + room_score + hint_score))

    def llm_value(self, cluster, grid) -> float:
        return self._cluster_grid_value(self.llm_value_grid, cluster, grid)

    def _object_interest(self, point: tuple[float, float]) -> float:
        target = str(self.strategy_bias.get("target_object", "") or "").lower().strip()
        if not target:
            return 0.0
        best = 0.0
        for obj in self.object_map:
            name = str(obj.get("semantic_name", "") or "").lower()
            labels = [str(label).lower() for label in obj.get("candidate_labels", []) or []]
            if target not in [name, *labels]:
                continue
            coord = obj.get("coord") or [0.0, 0.0, 0.0]
            try:
                dist = hypot(float(coord[0]) - point[0], float(coord[1]) - point[1])
            except (TypeError, ValueError, IndexError):
                continue
            confidence = float(obj.get("conf", 0.5) or 0.5)
            best = max(best, confidence / (1.0 + dist))
        return min(1.0, best)

    def _room_interest(self, cluster, grid) -> float:
        wanted_room = self.strategy_bias.get("room_type")
        if wanted_room is None:
            return 0.0
        wanted_ids = self.strategy_bias.get("room_ids")
        if not isinstance(wanted_ids, list):
            return 0.0
        scene_value = self._grid_raw_value(self.scene_id_grid, cluster.subgoal_world)
        if scene_value is None:
            return 0.0
        if int(scene_value) not in {int(value) for value in wanted_ids if self._is_intlike(value)}:
            return 0.0
        confidence = self._grid_value(self.scene_confidence_grid, cluster.subgoal_world)
        return max(0.2, confidence)

    def _navigation_hint_interest(self, point: tuple[float, float]) -> float:
        target_hint = str(self.strategy_bias.get("hint_type", "") or "").lower().strip()
        if not target_hint:
            return 0.0
        best = 0.0
        for hint in self.navigation_hints:
            hint_type = str(hint.get("hint_type", hint.get("type", "")) or "").lower()
            if target_hint and target_hint != hint_type:
                continue
            pos = hint.get("position") or hint.get("world_position") or {}
            try:
                hx = float(pos["x"] if isinstance(pos, dict) else pos[0])
                hy = float(pos["y"] if isinstance(pos, dict) else pos[1])
            except (TypeError, ValueError, KeyError, IndexError):
                continue
            best = max(best, 1.0 / (1.0 + hypot(hx - point[0], hy - point[1])))
        return min(1.0, best)

    def _grid_value(self, grid, point: tuple[float, float]) -> float:
        value = self._grid_raw_value(grid, point)
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value) / 100.0))

    def _cluster_grid_value(self, value_grid, cluster, grid) -> float:
        values = [
            self._grid_value(value_grid, cluster.subgoal_world),
            self._grid_value(value_grid, cluster.centroid_world),
        ]
        for cell in getattr(cluster, "cells", [])[:100]:
            values.append(self._grid_value(value_grid, grid.spec.grid_to_world(cell[0], cell[1])))
        return max(values) if values else 0.0

    @staticmethod
    def _grid_raw_value(grid, point: tuple[float, float]):
        if grid is None:
            return None
        info = getattr(grid, "info", None)
        data = getattr(grid, "data", None)
        if info is None or data is None:
            return None
        resolution = float(info.resolution)
        if resolution <= 0.0:
            return None
        x = int((point[0] - float(info.origin.position.x)) / resolution)
        y = int((point[1] - float(info.origin.position.y)) / resolution)
        if x < 0 or y < 0 or x >= int(info.width) or y >= int(info.height):
            return None
        index = y * int(info.width) + x
        if index < 0 or index >= len(data):
            return None
        return int(data[index])

    @staticmethod
    def _loads(data: str, default):
        if not data:
            return default
        try:
            return json.loads(data)
        except ValueError:
            return default

    @staticmethod
    def _is_intlike(value) -> bool:
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            return False
