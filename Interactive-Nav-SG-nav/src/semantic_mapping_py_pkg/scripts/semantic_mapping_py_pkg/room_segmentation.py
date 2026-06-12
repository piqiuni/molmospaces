from __future__ import annotations

from collections import deque

import numpy as np


class RoomSegmentationState:
    def __init__(self):
        self.prev_room_grid_signature = None
        self.prev_room_ids = None
        self.next_room_segment_id = 1


class RoomSegmenter:
    def __init__(
        self,
        room_free_threshold=20,
        room_unknown_id=-1,
        room_min_component_cells=25,
        room_boundary_margin_cells=1,
        room_core_min_component_cells=60,
        room_core_clearance_cells=5,
        room_small_obstacle_max_cells=0,
        state=None,
    ):
        self.room_free_threshold = int(room_free_threshold)
        self.room_unknown_id = int(room_unknown_id)
        self.room_min_component_cells = int(room_min_component_cells)
        self.room_boundary_margin_cells = max(0, int(room_boundary_margin_cells))
        self.room_core_min_component_cells = max(
            self.room_min_component_cells,
            int(room_core_min_component_cells),
        )
        self.room_core_clearance_cells = max(1, int(room_core_clearance_cells))
        self.room_small_obstacle_max_cells = max(0, int(room_small_obstacle_max_cells))
        self.state = state if state is not None else RoomSegmentationState()

    def segment(self, occ_grid):
        try:
            import cv2
        except Exception:
            cv2 = None
        width = int(occ_grid.info.width)
        height = int(occ_grid.info.height)
        size = width * height
        room_ids = [self.room_unknown_id] * size
        room_conf = [-1] * size
        if size <= 0 or len(occ_grid.data) != size:
            return room_ids, room_conf
        values = np.asarray(occ_grid.data, dtype=np.int16).reshape(height, width)
        free_mask = ((values >= 0) & (values <= self.room_free_threshold)).astype(np.uint8)
        occupied_mask = (values > self.room_free_threshold).astype(np.uint8)
        segmentation_free = free_mask.copy()

        if cv2 is not None and self.room_small_obstacle_max_cells > 0 and np.any(occupied_mask):
            component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(occupied_mask, 8)
            for component_id in range(1, component_count):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area > self.room_small_obstacle_max_cells:
                    continue
                left = int(stats[component_id, cv2.CC_STAT_LEFT])
                top = int(stats[component_id, cv2.CC_STAT_TOP])
                comp_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
                comp_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
                touches_border = (
                    left <= 0
                    or top <= 0
                    or (left + comp_width) >= width
                    or (top + comp_height) >= height
                )
                if touches_border:
                    continue
                density = float(area) / max(float(comp_width * comp_height), 1.0)
                aspect = max(float(comp_width), float(comp_height)) / max(min(float(comp_width), float(comp_height)), 1.0)
                span = max(int(comp_width), int(comp_height))
                if density < 0.55 or aspect > 1.8 or span > 18:
                    continue
                segmentation_free[labels == component_id] = 1

        if cv2 is not None:
            distance = cv2.distanceTransform((segmentation_free * 255).astype(np.uint8), cv2.DIST_L2, 5)
            core_mask = (
                (segmentation_free > 0)
                & (distance >= float(self.room_core_clearance_cells))
            ).astype(np.uint8)
            component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(core_mask, 8)
            component_cells = {}
            next_temp_id = 1
            for component_id in range(1, component_count):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area < self.room_core_min_component_cells:
                    continue
                ys, xs = np.where(labels == component_id)
                component_cells[next_temp_id] = [int(y) * width + int(x) for y, x in zip(ys.tolist(), xs.tolist())]
                next_temp_id += 1
        else:
            component_cells = self._fallback_component_cells(segmentation_free, width, height)

        if not component_cells:
            fallback_component = np.flatnonzero(segmentation_free > 0).tolist()
            if len(fallback_component) >= self.room_min_component_cells:
                component_cells[1] = [int(index) for index in fallback_component]

        remapped_ids = self._remap_room_component_ids(component_cells, occ_grid.info)
        for temp_room_id, component in component_cells.items():
            stable_room_id = remapped_ids.get(temp_room_id, temp_room_id)
            for comp_idx in component:
                room_ids[comp_idx] = stable_room_id
                room_conf[comp_idx] = 100

        queue = deque(idx for idx, room_id in enumerate(room_ids) if room_id >= 0)
        while queue:
            current = queue.popleft()
            x = current % width
            y = current // width
            current_room_id = room_ids[current]
            current_conf = room_conf[current]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                nidx = ny * width + nx
                if room_ids[nidx] != self.room_unknown_id:
                    continue
                if segmentation_free[ny, nx] <= 0:
                    continue
                room_ids[nidx] = current_room_id
                room_conf[nidx] = max(current_conf - 5, 60)
                queue.append(nidx)

        signature = self._grid_signature(occ_grid.info)
        self.state.prev_room_grid_signature = signature
        self.state.prev_room_ids = list(room_ids)
        return room_ids, room_conf

    def _fallback_component_cells(self, segmentation_free, width, height):
        flat = segmentation_free.reshape(height * width)
        visited = np.zeros(height * width, dtype=bool)
        components = {}
        next_temp_id = 1
        for index in range(height * width):
            if visited[index] or flat[index] <= 0:
                continue
            visited[index] = True
            queue = deque([index])
            component = [index]
            while queue:
                current = queue.popleft()
                x = current % width
                y = current // width
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    nidx = ny * width + nx
                    if visited[nidx] or flat[nidx] <= 0:
                        continue
                    visited[nidx] = True
                    component.append(nidx)
                    queue.append(nidx)
            if len(component) < self.room_core_min_component_cells:
                continue
            components[next_temp_id] = component
            next_temp_id += 1
        return components

    def _remap_room_component_ids(self, component_cells, grid_info):
        remapped = {}
        used_previous = set()
        previous_ids = None
        signature = self._grid_signature(grid_info)
        if self.state.prev_room_grid_signature == signature and self.state.prev_room_ids:
            previous_ids = self.state.prev_room_ids

        for temp_room_id, component in sorted(component_cells.items(), key=lambda item: -len(item[1])):
            best_prev_room_id = None
            best_overlap = 0
            if previous_ids is not None:
                overlap_counts = {}
                for idx in component:
                    prev_room_id = int(previous_ids[idx])
                    if prev_room_id < 0 or prev_room_id in used_previous:
                        continue
                    overlap_counts[prev_room_id] = overlap_counts.get(prev_room_id, 0) + 1
                if overlap_counts:
                    best_prev_room_id, best_overlap = max(
                        sorted(overlap_counts.items()),
                        key=lambda item: item[1],
                    )
            if best_prev_room_id is not None and best_overlap > 0:
                remapped[temp_room_id] = best_prev_room_id
                used_previous.add(best_prev_room_id)
            else:
                remapped[temp_room_id] = self.state.next_room_segment_id
                self.state.next_room_segment_id += 1
        return remapped

    @staticmethod
    def _grid_signature(grid_info):
        return (
            int(grid_info.width),
            int(grid_info.height),
            float(grid_info.resolution),
            float(grid_info.origin.position.x),
            float(grid_info.origin.position.y),
        )
