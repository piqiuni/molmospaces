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
        room_core_min_component_cells=40,
        room_core_clearance_cells=7,
        room_small_obstacle_max_cells=0,
        room_remove_enclosed_occupied=True,
        room_enclosed_occupied_max_cells=700,
        room_enclosed_occupied_max_aspect=2.5,
        room_enclosed_occupied_known_ring_ratio=0.95,
        room_enclosed_occupied_free_ring_ratio=0.45,
        room_fill_enclosed_obstacles=False,
        room_enclosed_obstacle_min_cells=120,
        room_enclosed_obstacle_max_cells=700,
        room_enclosed_obstacle_dominance_ratio=0.82,
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
        self.room_remove_enclosed_occupied = bool(room_remove_enclosed_occupied)
        self.room_enclosed_occupied_max_cells = max(0, int(room_enclosed_occupied_max_cells))
        self.room_enclosed_occupied_max_aspect = float(room_enclosed_occupied_max_aspect)
        self.room_enclosed_occupied_known_ring_ratio = float(room_enclosed_occupied_known_ring_ratio)
        self.room_enclosed_occupied_free_ring_ratio = float(room_enclosed_occupied_free_ring_ratio)
        self.room_fill_enclosed_obstacles = bool(room_fill_enclosed_obstacles)
        self.room_enclosed_obstacle_min_cells = max(0, int(room_enclosed_obstacle_min_cells))
        self.room_enclosed_obstacle_max_cells = max(
            self.room_enclosed_obstacle_min_cells,
            int(room_enclosed_obstacle_max_cells),
        )
        self.room_enclosed_obstacle_dominance_ratio = float(room_enclosed_obstacle_dominance_ratio)
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

        if cv2 is not None and self.room_remove_enclosed_occupied and np.any(occupied_mask):
            known_mask = (values >= 0).astype(np.uint8)
            known_ys, known_xs = np.where(known_mask > 0)
            if known_ys.size > 0 and known_xs.size > 0:
                row_min = int(np.min(known_ys))
                row_max = int(np.max(known_ys)) + 1
                col_min = int(np.min(known_xs))
                col_max = int(np.max(known_xs)) + 1
                component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(occupied_mask, 8)
                for component_id in range(1, component_count):
                    area = int(stats[component_id, cv2.CC_STAT_AREA])
                    if area <= 0 or area > self.room_enclosed_occupied_max_cells:
                        continue
                    left = int(stats[component_id, cv2.CC_STAT_LEFT])
                    top = int(stats[component_id, cv2.CC_STAT_TOP])
                    comp_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
                    comp_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
                    if (
                        left <= col_min
                        or top <= row_min
                        or (left + comp_width) >= col_max
                        or (top + comp_height) >= row_max
                    ):
                        continue
                    aspect = max(float(comp_width), float(comp_height)) / max(
                        min(float(comp_width), float(comp_height)),
                        1.0,
                    )
                    if aspect > self.room_enclosed_occupied_max_aspect:
                        continue
                    ring_row_min = max(top - 1, 0)
                    ring_row_max = min(top + comp_height + 1, height)
                    ring_col_min = max(left - 1, 0)
                    ring_col_max = min(left + comp_width + 1, width)
                    ring_mask = np.ones(
                        (ring_row_max - ring_row_min, ring_col_max - ring_col_min),
                        dtype=bool,
                    )
                    if ring_mask.shape[0] > 2 and ring_mask.shape[1] > 2:
                        ring_mask[1:-1, 1:-1] = False
                    ring_known = known_mask[ring_row_min:ring_row_max, ring_col_min:ring_col_max][ring_mask]
                    ring_free = free_mask[ring_row_min:ring_row_max, ring_col_min:ring_col_max][ring_mask]
                    if ring_known.size == 0:
                        continue
                    if float(np.mean(ring_known)) < self.room_enclosed_occupied_known_ring_ratio:
                        continue
                    if float(np.mean(ring_free)) < self.room_enclosed_occupied_free_ring_ratio:
                        continue
                    segmentation_free[labels == component_id] = 1

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

        if cv2 is not None and self.room_fill_enclosed_obstacles:
            room_ids, room_conf = self._fill_enclosed_obstacles(
                values,
                room_ids,
                room_conf,
                width,
                height,
                cv2,
            )

        signature = self._grid_signature(occ_grid.info)
        self.state.prev_room_grid_signature = signature
        self.state.prev_room_ids = list(room_ids)
        return room_ids, room_conf

    def _fill_enclosed_obstacles(self, values, room_ids, room_conf, width, height, cv2):
        room_grid = np.asarray(room_ids, dtype=np.int32).reshape(height, width)
        conf_grid = np.asarray(room_conf, dtype=np.int32).reshape(height, width)
        occupied_mask = (values > self.room_free_threshold).astype(np.uint8)
        known_mask = (values >= 0).astype(np.uint8)
        if not np.any(occupied_mask) or not np.any(known_mask):
            return room_ids, room_conf
        ys, xs = np.where(known_mask > 0)
        row_min, row_max = int(np.min(ys)), int(np.max(ys)) + 1
        col_min, col_max = int(np.min(xs)), int(np.max(xs)) + 1
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(occupied_mask, 8)
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < self.room_enclosed_obstacle_min_cells or area > self.room_enclosed_obstacle_max_cells:
                continue
            left = int(stats[component_id, cv2.CC_STAT_LEFT])
            top = int(stats[component_id, cv2.CC_STAT_TOP])
            comp_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            comp_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if (
                left <= col_min
                or top <= row_min
                or (left + comp_width) >= col_max
                or (top + comp_height) >= row_max
            ):
                continue
            ys_comp, xs_comp = np.where(labels == component_id)
            neighbor_rooms = []
            for y, x in zip(ys_comp.tolist(), xs_comp.tolist()):
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    room_id = int(room_grid[ny, nx])
                    if room_id >= 0:
                        neighbor_rooms.append(room_id)
            if not neighbor_rooms:
                continue
            room_counts = {}
            for room_id in neighbor_rooms:
                room_counts[room_id] = room_counts.get(room_id, 0) + 1
            dominant_room_id = max(sorted(room_counts.keys()), key=lambda room_id: room_counts[room_id])
            dominant_ratio = float(room_counts[dominant_room_id]) / float(sum(room_counts.values()))
            if dominant_ratio < self.room_enclosed_obstacle_dominance_ratio:
                continue
            mask = labels == component_id
            room_grid[mask] = int(dominant_room_id)
            conf_grid[mask] = 55
        return room_grid.reshape(height * width).tolist(), conf_grid.reshape(height * width).tolist()

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
