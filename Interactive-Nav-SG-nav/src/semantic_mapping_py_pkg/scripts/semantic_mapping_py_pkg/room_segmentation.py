from __future__ import annotations

from collections import deque

import numpy as np

from .geometry_utils import grid_origin_yaw, world_to_grid


class RoomSegmentationState:
    def __init__(self):
        self.prev_room_grid_signature = None
        self.prev_room_ids = None
        self.stable_room_grid_signature = None
        self.stable_room_ids = None
        self.stable_room_conf = None
        self.candidate_room_ids = None
        self.candidate_room_conf = None
        self.candidate_room_count = 0
        self.next_room_segment_id = 1
        self.portal_hints = {}
        self.pending_merges = {}
        self.last_confirmed_merges = {}


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
        room_enclosed_occupied_max_aspect=1.8,
        room_enclosed_occupied_known_ring_ratio=0.95,
        room_enclosed_occupied_free_ring_ratio=0.45,
        room_fill_enclosed_unknown=False,
        room_enclosed_unknown_max_cells=250,
        room_enclosed_unknown_known_ring_ratio=0.90,
        room_enclosed_unknown_free_ring_ratio=0.75,
        room_fill_enclosed_obstacles=False,
        room_enclosed_obstacle_min_cells=120,
        room_enclosed_obstacle_max_cells=700,
        room_enclosed_obstacle_dominance_ratio=0.82,
        room_portal_cut_enabled=True,
        room_portal_cut_margin_m=0.15,
        room_portal_cut_thickness_cells=2,
        room_portal_detector_min_confirmations=3,
        room_portal_detector_max_center_jump_m=0.4,
        room_portal_hint_merge_distance_m=0.6,
        room_portal_min_width_m=0.5,
        room_portal_max_width_m=2.5,
        room_id_overlap_ratio=0.25,
        room_merge_confirmations=3,
        room_grid_stability_frames=1,
        room_portal_small_component_confidence=70,
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
        # RGB-D OCC contains small, fully enclosed unknown holes when a wall
        # edge or a depth return is missing.  Treating every unknown cell as a
        # wall fragments an otherwise connected room.  The fill is deliberately
        # opt-in and only accepts compact components whose one-cell ring is
        # overwhelmingly known/free; open unknown frontiers remain blocked.
        self.room_fill_enclosed_unknown = bool(room_fill_enclosed_unknown)
        self.room_enclosed_unknown_max_cells = max(
            0, int(room_enclosed_unknown_max_cells)
        )
        self.room_enclosed_unknown_known_ring_ratio = float(
            room_enclosed_unknown_known_ring_ratio
        )
        self.room_enclosed_unknown_free_ring_ratio = float(
            room_enclosed_unknown_free_ring_ratio
        )
        self.room_fill_enclosed_obstacles = bool(room_fill_enclosed_obstacles)
        self.room_enclosed_obstacle_min_cells = max(0, int(room_enclosed_obstacle_min_cells))
        self.room_enclosed_obstacle_max_cells = max(
            self.room_enclosed_obstacle_min_cells,
            int(room_enclosed_obstacle_max_cells),
        )
        self.room_enclosed_obstacle_dominance_ratio = float(room_enclosed_obstacle_dominance_ratio)
        self.room_portal_cut_enabled = bool(room_portal_cut_enabled)
        self.room_portal_cut_margin_m = max(0.0, float(room_portal_cut_margin_m))
        self.room_portal_cut_thickness_cells = max(1, int(room_portal_cut_thickness_cells))
        self.room_portal_detector_min_confirmations = max(1, int(room_portal_detector_min_confirmations))
        self.room_portal_detector_max_center_jump_m = max(
            0.0,
            float(room_portal_detector_max_center_jump_m),
        )
        self.room_portal_hint_merge_distance_m = max(0.0, float(room_portal_hint_merge_distance_m))
        self.room_portal_min_width_m = max(0.0, float(room_portal_min_width_m))
        self.room_portal_max_width_m = max(
            self.room_portal_min_width_m,
            float(room_portal_max_width_m),
        )
        self.room_id_overlap_ratio = min(1.0, max(0.0, float(room_id_overlap_ratio)))
        self.room_merge_confirmations = max(1, int(room_merge_confirmations))
        self.room_grid_stability_frames = max(1, int(room_grid_stability_frames))
        self.room_portal_small_component_confidence = min(
            99,
            max(1, int(room_portal_small_component_confidence)),
        )
        self.state = state if state is not None else RoomSegmentationState()

    @staticmethod
    def _connected_components_with_stats(mask, cv2, *, connectivity=4):
        """Label components without connecting rooms through diagonal pinholes.

        Free-space/core components use four-connectivity by default: two rooms
        touching only at one diagonal pixel are not traversably connected.
        Occupied-object callers explicitly request eight-connectivity so a
        diagonally sampled wall remains one conservative obstacle component.
        """
        connectivity = 8 if int(connectivity) == 8 else 4
        algorithm = getattr(cv2, "CCL_BBDT", None)
        accelerated = getattr(cv2, "connectedComponentsWithStatsWithAlgorithm", None)
        if algorithm is not None and accelerated is not None:
            return accelerated(mask, connectivity, cv2.CV_32S, algorithm)
        return cv2.connectedComponentsWithStats(mask, connectivity)

    def update_portal_hints(
        self,
        observations,
        source_mode="detector_online",
        *,
        refresh_active=False,
    ):
        if not self.room_portal_cut_enabled:
            return False
        changed = False
        is_gt = str(source_mode) == "realtime_gt_observation"
        for observation in observations or []:
            if not self._is_portal_observation(observation):
                continue
            box_3d = observation.get("box_3d") or {}
            if not isinstance(box_3d, dict):
                box_3d = {}
            center = self._point3(
                observation.get("aabb_center")
                or observation.get("position")
                or box_3d.get("center")
            )
            size = self._point3(
                observation.get("aabb_size") or box_3d.get("size")
            )
            if center is None or size is None:
                continue
            span = max(float(size[0]), float(size[1]))
            if span <= 0.0:
                continue
            key = self._portal_hint_key(observation, center)
            hint = self.state.portal_hints.get(key)
            if hint is None:
                hint = {
                    "center": center,
                    "size": size,
                    "candidate_center": center,
                    "candidate_size": size,
                    "confirmations": 0,
                    "active": False,
                    "source_mode": str(source_mode),
                }
                self.state.portal_hints[key] = hint
            if hint["active"]:
                # A successful interaction supplies the pre-open doorway
                # reference geometry.  Preserve it as the virtual cut anchor:
                # subsequent GT observations can describe the rotated door
                # leaf instead of the doorway plane.  Ordinary detector
                # updates intentionally retain the existing frozen behavior.
                if refresh_active:
                    geometry_changed = (
                        self._distance_xy(hint["center"], center) > 1e-6
                        or any(
                            abs(float(hint["size"][axis]) - float(size[axis]))
                            > 1e-6
                            for axis in range(3)
                        )
                    )
                    hint["center"] = list(center)
                    hint["size"] = list(size)
                    hint["candidate_center"] = list(center)
                    hint["candidate_size"] = list(size)
                    hint["confirmations"] = max(
                        int(hint.get("confirmations", 0)), 1
                    )
                    changed = changed or geometry_changed
                continue
            jump = self._distance_xy(hint["candidate_center"], center)
            if jump > self.room_portal_detector_max_center_jump_m:
                hint["candidate_center"] = center
                hint["candidate_size"] = size
                hint["confirmations"] = 1
            else:
                count = int(hint["confirmations"])
                blend = 1.0 / float(count + 1)
                hint["candidate_center"] = [
                    (1.0 - blend) * float(hint["candidate_center"][axis]) + blend * float(center[axis])
                    for axis in range(3)
                ]
                hint["candidate_size"] = [
                    (1.0 - blend) * float(hint["candidate_size"][axis]) + blend * float(size[axis])
                    for axis in range(3)
                ]
                hint["confirmations"] = count + 1
            required = 1 if is_gt else self.room_portal_detector_min_confirmations
            if int(hint["confirmations"]) >= required:
                hint["center"] = list(hint["candidate_center"])
                hint["size"] = list(hint["candidate_size"])
                hint["active"] = True
                changed = True
        return changed

    def segment(self, occ_grid, *, force_stable=False):
        try:
            import cv2
        except Exception:
            cv2 = None
        width = int(occ_grid.info.width)
        height = int(occ_grid.info.height)
        size = width * height
        room_ids = np.full(size, self.room_unknown_id, dtype=np.int32)
        room_conf = np.full(size, -1, dtype=np.int16)
        if size <= 0 or len(occ_grid.data) != size:
            return room_ids.tolist(), room_conf.tolist()
        values = np.asarray(occ_grid.data, dtype=np.int16).reshape(height, width)
        free_mask = ((values >= 0) & (values <= self.room_free_threshold)).astype(np.uint8)
        occupied_mask = (values > self.room_free_threshold).astype(np.uint8)
        segmentation_free = free_mask.copy()
        occupied_components = None
        if (cv2 is not None and np.any(occupied_mask)
            and (self.room_remove_enclosed_occupied or self.room_small_obstacle_max_cells > 0)):
            occupied_components = self._connected_components_with_stats(
                occupied_mask, cv2, connectivity=8
            )

        if cv2 is not None and self.room_fill_enclosed_unknown:
            self._fill_enclosed_unknown_cells(
                values,
                free_mask,
                segmentation_free,
                cv2,
            )

        if cv2 is not None and self.room_remove_enclosed_occupied and np.any(occupied_mask):
            known_mask = (values >= 0).astype(np.uint8)
            known_ys, known_xs = np.where(known_mask > 0)
            if known_ys.size > 0 and known_xs.size > 0:
                row_min = int(np.min(known_ys))
                row_max = int(np.max(known_ys)) + 1
                col_min = int(np.min(known_xs))
                col_max = int(np.max(known_xs)) + 1
                component_count, labels, stats, _centroids = occupied_components
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
                    region = np.s_[top:top + comp_height, left:left + comp_width]
                    segmentation_free[region][labels[region] == component_id] = 1

        if cv2 is not None and self.room_small_obstacle_max_cells > 0 and np.any(occupied_mask):
            component_count, labels, stats, _centroids = occupied_components
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
                region = np.s_[top:top + comp_height, left:left + comp_width]
                segmentation_free[region][labels[region] == component_id] = 1

        portal_cut_mask = np.zeros_like(segmentation_free, dtype=np.uint8)
        pre_portal_cut_free = None
        if cv2 is not None and self.room_portal_cut_enabled:
            pre_portal_cut_free = segmentation_free.copy()
            portal_cut_mask = self._apply_portal_cuts(
                segmentation_free,
                occ_grid.info,
                cv2,
            )

        component_confidence = {}
        if cv2 is not None:
            distance = cv2.distanceTransform((segmentation_free * 255).astype(np.uint8), cv2.DIST_L2, 5)
            core_mask = (
                (segmentation_free > 0)
                & (distance >= float(self.room_core_clearance_cells))
            ).astype(np.uint8)
            component_count, labels, stats, _centroids = self._connected_components_with_stats(core_mask, cv2)
            component_cells = {}
            next_temp_id = 1
            for component_id in range(1, component_count):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area < self.room_core_min_component_cells:
                    continue
                # Keep the native index array until raster assignment.  The
                # previous list conversion was immediately converted back to
                # NumPy by the propagation/remap paths and cost a full-grid
                # Python allocation on every room refresh.
                component_cells[next_temp_id] = self._component_flat_indices(
                    labels, stats[component_id], component_id, cv2
                )
                component_confidence[next_temp_id] = 100
                next_temp_id += 1

            if pre_portal_cut_free is not None and np.any(portal_cut_mask):
                for component in self._portal_separated_small_components(
                    segmentation_free,
                    pre_portal_cut_free,
                    portal_cut_mask,
                    component_cells,
                    width,
                    height,
                    cv2=cv2,
                ):
                    component_cells[next_temp_id] = component
                    component_confidence[next_temp_id] = (
                        self.room_portal_small_component_confidence
                    )
                    next_temp_id += 1
        else:
            component_cells = self._fallback_component_cells(segmentation_free, width, height)
            component_confidence = {
                temp_room_id: 100 for temp_room_id in component_cells
            }

        if not component_cells:
            # Narrow observed regions may have no distance-transform core.
            # Preserve their connectivity instead of assigning every free
            # cell in the entire map to one artificial room.
            component_cells = self._fallback_component_cells(
                segmentation_free, width, height,
                minimum_cells=self.room_min_component_cells, cv2=cv2,
            )
            component_confidence = {room_id: 100 for room_id in component_cells}

        remapped_ids = self._remap_room_component_ids(component_cells, occ_grid.info)
        for temp_room_id, component in component_cells.items():
            stable_room_id = remapped_ids.get(temp_room_id, temp_room_id)
            indices = np.asarray(component, dtype=np.int64)
            room_ids[indices] = int(stable_room_id)
            room_conf[indices] = int(
                component_confidence.get(temp_room_id, 100)
            )

        # In the usual map each traversable connected component contains one
        # distance-transform core. Connected-components propagation is then
        # equivalent to the old Python deque flood-fill, but runs in OpenCV
        # and NumPy. Keep the exact BFS for multi-seed transition frames.
        propagated = False
        if cv2 is not None:
            propagated = self._propagate_single_seed_components(
                segmentation_free, room_ids, room_conf, width, height, cv2
            )
        if not propagated:
            # Door-opening frames often connect several seeded rooms.  A
            # Python deque over a 640k-cell map made this transition take
            # roughly a second and blocked the room worker.  OpenCV's
            # marker watershed performs the same topology-constrained growth
            # in C++; only its thin tie boundaries need a tiny vectorised
            # cleanup.  Keep the deque as a dependency-free safety fallback.
            propagated = False
            if cv2 is not None:
                propagated = self._propagate_watershed(
                    segmentation_free, room_ids, room_conf, width, height, cv2
                )
            if not propagated:
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
        # Most segmentation paths keep NumPy arrays, while the optional
        # enclosed-obstacle fill returns already-flattened Python lists.  Do
        # not call ``.tolist()`` unconditionally: enabling that cleanup path
        # used to raise ``AttributeError`` and stop room publication.
        room_ids_list = room_ids.tolist() if hasattr(room_ids, "tolist") else list(room_ids)
        room_conf_list = room_conf.tolist() if hasattr(room_conf, "tolist") else list(room_conf)
        self.state.prev_room_grid_signature = signature
        self.state.prev_room_ids = room_ids_list
        return self._stabilize_room_grid(
            signature,
            room_ids_list,
            room_conf_list,
            force=force_stable,
        )

    def _fill_enclosed_unknown_cells(
        self,
        values,
        free_mask,
        segmentation_free,
        cv2,
    ):
        """Fill compact unknown holes surrounded by known free OCC cells.

        Unknown space at the sensor frontier is intentionally not traversable:
        only components strictly inside the bounding box of known cells are
        considered.  Eight-connectivity also treats a diagonal pinhole as an
        open leak, preventing accidental room connections through a one-cell
        corner.  The operation mutates ``segmentation_free`` in place and
        returns the number of filled cells for diagnostics/tests.
        """

        if self.room_enclosed_unknown_max_cells <= 0:
            return 0
        unknown_mask = (values < 0).astype(np.uint8)
        known_mask = (values >= 0).astype(np.uint8)
        if not np.any(unknown_mask) or not np.any(known_mask):
            return 0
        known_ys, known_xs = np.where(known_mask > 0)
        if known_ys.size == 0 or known_xs.size == 0:
            return 0
        row_min = int(np.min(known_ys))
        row_max = int(np.max(known_ys)) + 1
        col_min = int(np.min(known_xs))
        col_max = int(np.max(known_xs)) + 1
        component_count, labels, stats, _centroids = (
            self._connected_components_with_stats(
                unknown_mask,
                cv2,
                connectivity=8,
            )
        )
        filled = 0
        height, width = values.shape
        for component_id in range(1, int(component_count)):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area <= 0 or area > self.room_enclosed_unknown_max_cells:
                continue
            left = int(stats[component_id, cv2.CC_STAT_LEFT])
            top = int(stats[component_id, cv2.CC_STAT_TOP])
            comp_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            comp_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            # A component touching the known-data envelope is part of the
            # unknown frontier, not a hole.  Keep it conservative even when
            # its local ring happens to contain many free cells.
            if (
                left <= col_min
                or top <= row_min
                or (left + comp_width) >= col_max
                or (top + comp_height) >= row_max
            ):
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
            ring_known = known_mask[
                ring_row_min:ring_row_max,
                ring_col_min:ring_col_max,
            ][ring_mask]
            ring_free = free_mask[
                ring_row_min:ring_row_max,
                ring_col_min:ring_col_max,
            ][ring_mask]
            if ring_known.size == 0:
                continue
            if float(np.mean(ring_known)) < self.room_enclosed_unknown_known_ring_ratio:
                continue
            if float(np.mean(ring_free)) < self.room_enclosed_unknown_free_ring_ratio:
                continue
            region = np.s_[top:top + comp_height, left:left + comp_width]
            segmentation_free[region][labels[region] == component_id] = 1
            filled += area
        return filled

    def _propagate_single_seed_components(
        self, segmentation_free, room_ids, room_conf, width, height, cv2
    ):
        """Propagate seed IDs with C++ distance transforms, component-wise.

        Running one transform per free connected component prevents a seed in
        another disconnected room from winning across a wall. The result is a
        nearest-seed partition rather than the old FIFO tie-break, which is
        stable for the large maps used on the physical robot and avoids a
        Python operation for every free cell.
        """
        mask = (segmentation_free > 0).astype(np.uint8)
        count, component_labels, stats, _centroids = (
            self._connected_components_with_stats(mask, cv2, connectivity=4)
        )
        room_array = np.asarray(room_ids, dtype=np.int32).reshape(height, width)
        conf_array = np.asarray(room_conf, dtype=np.int16).reshape(height, width)

        # A connected free component can temporarily contain multiple room
        # seeds after a confirmed door opens.  The fast distance transform is
        # Euclidean and would be allowed to cut across an occupied wall in
        # that case.  Detect this before mutating the arrays and let the exact
        # 4-neighbour BFS below preserve the topology for that transition.
        for component_id in range(1, int(count)):
            top = int(stats[component_id, cv2.CC_STAT_TOP])
            left = int(stats[component_id, cv2.CC_STAT_LEFT])
            comp_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            comp_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if comp_width <= 0 or comp_height <= 0:
                continue
            component = (
                component_labels[top : top + comp_height, left : left + comp_width]
                == component_id
            )
            local_rooms = room_array[
                top : top + comp_height, left : left + comp_width
            ]
            seed_rooms = local_rooms[component & (local_rooms >= 0)]
            if seed_rooms.size and np.unique(seed_rooms).size > 1:
                return False

        for component_id in range(1, int(count)):
            top = int(stats[component_id, cv2.CC_STAT_TOP])
            left = int(stats[component_id, cv2.CC_STAT_LEFT])
            comp_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            comp_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if comp_width <= 0 or comp_height <= 0:
                continue
            component = (
                component_labels[top : top + comp_height, left : left + comp_width]
                == component_id
            )
            local_rooms = room_array[
                top : top + comp_height, left : left + comp_width
            ]
            local_conf = conf_array[
                top : top + comp_height, left : left + comp_width
            ]
            seed_mask = component & (local_rooms >= 0)
            if not np.any(seed_mask):
                continue
            target = component & ~seed_mask
            if not np.any(target):
                continue
            seed_confs = local_conf[seed_mask]
            if np.all(seed_confs == seed_confs[0]):
                # The validation pass established one room ID per component.
                # Uniform seed confidence makes nearest-seed lookup redundant:
                # every target gets exactly the same ID and confidence.
                local_rooms[target] = local_rooms[seed_mask][0]
                local_conf[target] = max(int(seed_confs[0]) - 5, 60)
                continue
            source = np.ones(component.shape, dtype=np.uint8)
            source[seed_mask] = 0
            _distance, nearest = cv2.distanceTransformWithLabels(
                source, cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL
            )
            seed_labels = nearest[seed_mask]
            seed_rooms = local_rooms[seed_mask]
            seed_confs = local_conf[seed_mask]
            max_label = int(np.max(nearest))
            lookup_room = np.full(max_label + 1, self.room_unknown_id, dtype=np.int32)
            lookup_conf = np.zeros(max_label + 1, dtype=np.int16)
            unique_labels, first_indices = np.unique(
                seed_labels, return_index=True
            )
            lookup_room[unique_labels] = seed_rooms[first_indices]
            np.maximum.at(lookup_conf, seed_labels, seed_confs)
            mapped_rooms = lookup_room[nearest]
            target = component & (mapped_rooms >= 0)
            local_rooms[target] = mapped_rooms[target]
            local_conf[target] = np.maximum(
                lookup_conf[nearest[target]].astype(np.int16) - 5, 60
            )
            # Preserve explicit low-confidence scores for portal-created
            # pocket components that were seeded before propagation.
            local_rooms[seed_mask] = seed_rooms
            local_conf[seed_mask] = seed_confs
        if isinstance(room_ids, np.ndarray):
            room_ids[:] = room_array.reshape(-1)
            room_conf[:] = conf_array.reshape(-1)
        else:
            room_ids[:] = room_array.reshape(-1).tolist()
            room_conf[:] = conf_array.reshape(-1).tolist()
        return True

    def _propagate_watershed(
        self, segmentation_free, room_ids, room_conf, width, height, cv2
    ):
        """Grow multiple room seeds through free cells without crossing walls.

        ``cv2.watershed`` is used only for transition maps with multiple room
        seeds in one connected component.  Occupied/unknown cells are marked
        as watershed boundaries, so propagation remains topological rather
        than taking a straight-line shortcut through a wall.  Watershed can
        leave a one-cell tie boundary; the short vectorised pass below fills
        it from an adjacent label.
        """
        free = np.asarray(segmentation_free, dtype=np.uint8).reshape(height, width) > 0
        rooms = np.asarray(room_ids, dtype=np.int32).reshape(height, width)
        confidences = np.asarray(room_conf, dtype=np.int16).reshape(height, width)
        seeds = free & (rooms >= 0)
        if not np.any(seeds):
            return True

        markers = np.zeros((height, width), dtype=np.int32)
        markers[seeds] = rooms[seeds]
        markers[~free] = -1
        image = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            cv2.watershed(image, markers)
        except Exception:
            return False

        labels = np.where(free & (markers >= 0), markers, self.room_unknown_id).astype(
            np.int32, copy=False
        )
        # Assign the small watershed tie boundary from an adjacent room.  Do
        # not iterate over all labelled cells; each pass touches only unknown
        # free cells and normally converges in one or two rounds.
        for _ in range(8):
            unknown = free & (labels < 0)
            if not np.any(unknown):
                break
            up = np.zeros_like(labels)
            up[1:] = labels[:-1]
            down = np.zeros_like(labels)
            down[:-1] = labels[1:]
            left = np.zeros_like(labels)
            left[:, 1:] = labels[:, :-1]
            right = np.zeros_like(labels)
            right[:, :-1] = labels[:, 1:]
            replacement = np.where(
                up >= 0,
                up,
                np.where(left >= 0, left, np.where(right >= 0, right, down)),
            )
            take = unknown & (replacement >= 0)
            if not np.any(take):
                break
            labels[take] = replacement[take]

        # Preserve the explicit confidence of core/portal seeds and use the
        # same conservative propagated confidence as the former BFS.
        propagated_conf = np.where(labels >= 0, 60, -1).astype(np.int16)
        propagated_conf[seeds] = confidences[seeds]
        if isinstance(room_ids, np.ndarray):
            room_ids[:] = labels.reshape(-1)
            room_conf[:] = propagated_conf.reshape(-1)
        else:
            room_ids[:] = labels.reshape(-1).tolist()
            room_conf[:] = propagated_conf.reshape(-1).tolist()
        return True
    def _stabilize_room_grid(self, signature, room_ids, room_conf, *, force=False):
        # ``segment`` has already finished mutating the current raster.  Keep
        # that owned list as the state snapshot instead of copying the entire
        # map once for stable IDs, once for candidates, and once for the
        # return value.  The next segment allocates a new raster, so these
        # aliases cannot be mutated by a later call.  On NumPy-based custom
        # callers convert only once at this boundary.
        current_ids = room_ids if isinstance(room_ids, list) else room_ids.tolist()
        current_conf = room_conf if isinstance(room_conf, list) else room_conf.tolist()
        if self.state.stable_room_grid_signature != signature or self.state.stable_room_ids is None:
            self.state.stable_room_grid_signature = signature
            self.state.stable_room_ids = current_ids
            self.state.stable_room_conf = current_conf
            self.state.candidate_room_ids = current_ids
            self.state.candidate_room_conf = current_conf
            self.state.candidate_room_count = self.room_grid_stability_frames
            self._commit_room_merges()
            return current_ids, current_conf

        if force:
            self.state.stable_room_ids = current_ids
            self.state.stable_room_conf = current_conf
            self.state.candidate_room_ids = current_ids
            self.state.candidate_room_conf = current_conf
            self.state.candidate_room_count = self.room_grid_stability_frames
            self._commit_room_merges()
            return current_ids, current_conf

        candidate_ids = self.state.candidate_room_ids
        if candidate_ids is not None and self._room_grids_compatible(candidate_ids, current_ids):
            self.state.candidate_room_count += 1
        else:
            self.state.candidate_room_count = 1
        self.state.candidate_room_ids = current_ids
        self.state.candidate_room_conf = current_conf
        if (
            self.state.candidate_room_count >= self.room_grid_stability_frames
            and all(
                count >= self.room_merge_confirmations
                for count in self.state.pending_merges.values()
            )
        ):
            self.state.stable_room_ids = current_ids
            self.state.stable_room_conf = current_conf
            self._commit_room_merges()
        return self.state.stable_room_ids, self.state.stable_room_conf or current_conf

    @staticmethod
    def _room_grids_compatible(previous_ids, current_ids):
        if len(previous_ids) != len(current_ids):
            return False
        previous = np.asarray(previous_ids, dtype=np.int32)
        current = np.asarray(current_ids, dtype=np.int32)
        if not np.array_equal(
            np.unique(previous[previous >= 0]),
            np.unique(current[current >= 0]),
        ):
            return False
        common = (previous >= 0) & (current >= 0)
        if not np.any(common):
            return False
        return float(np.mean(previous[common] == current[common])) >= 0.97

    def consume_confirmed_merges(self):
        merges = dict(self.state.last_confirmed_merges)
        self.state.last_confirmed_merges.clear()
        return merges

    def _apply_portal_cuts(self, segmentation_free, grid_info, cv2):
        resolution = float(grid_info.resolution)
        cut_mask = np.zeros_like(segmentation_free, dtype=np.uint8)
        if resolution <= 0.0:
            return cut_mask
        height, width = segmentation_free.shape
        for hint in self.state.portal_hints.values():
            if not hint.get("active"):
                continue
            center = hint["center"]
            size = hint["size"]
            span_axis = 0 if float(size[0]) >= float(size[1]) else 1
            span = min(
                max(max(float(size[0]), float(size[1])), self.room_portal_min_width_m),
                self.room_portal_max_width_m,
            ) + 2.0 * self.room_portal_cut_margin_m
            start = list(center)
            end = list(center)
            start[span_axis] -= 0.5 * span
            end[span_axis] += 0.5 * span
            start_cell = world_to_grid(
                start[0],
                start[1],
                grid_info,
                check_bounds=False,
            )
            end_cell = world_to_grid(
                end[0],
                end[1],
                grid_info,
                check_bounds=False,
            )
            if start_cell is None or end_cell is None:
                continue
            if not self._line_may_intersect_grid(start_cell, end_cell, width, height):
                continue
            cv2.line(
                cut_mask,
                start_cell,
                end_cell,
                1,
                thickness=self.room_portal_cut_thickness_cells,
                lineType=cv2.LINE_8,
            )
        # Keep the complete rasterised portal line in the mask, including
        # cells currently labelled occupied/unknown.  A detected doorway is
        # often sampled exactly on the occupied jamb or an unknown ray edge;
        # restricting the mask to ``segmentation_free`` made the cut vanish
        # in that common case and left both sides as one room.  Removing the
        # line from free space is still the only topology change, so a line
        # that does not intersect known free cells remains a no-op.
        cut_mask = (cut_mask > 0).astype(np.uint8)
        segmentation_free[cut_mask > 0] = 0
        return cut_mask

    def _portal_separated_small_components(
        self,
        segmentation_free,
        pre_portal_cut_free,
        portal_cut_mask,
        core_component_cells,
        width,
        height,
        *,
        cv2=None,
    ):
        """Keep room-sized pockets that an active virtual portal cut isolated.

        Core-based room seeds intentionally discard narrow spaces.  Once a
        portal cut separates such a pocket from a core room, however, leaving
        it unknown loses the topology change caused by the portal.  Only
        retain components that were connected before the cut, are now split
        by it, and meet the ordinary minimum room area.
        """

        if cv2 is None:
            import cv2 as cv2_module

            cv2 = cv2_module

        # Portal pocket preservation is the only path that needs *four*
        # connectivity.  The ordinary room seed path intentionally remains
        # eight-connected.  This used to call the Python deque flood-fill
        # twice over the complete grid; OpenCV labels the same components in
        # native code, while the vectorized bookkeeping below preserves the
        # original selection rules.
        post_count, post_labels, post_stats, _centroids = (
            cv2.connectedComponentsWithStats(
                (segmentation_free > 0).astype(np.uint8),
                connectivity=4,
                ltype=cv2.CV_32S,
            )
        )
        if int(post_count) - 1 < 2:
            return []
        pre_count, pre_labels, _pre_stats, _pre_centroids = (
            cv2.connectedComponentsWithStats(
                (pre_portal_cut_free > 0).astype(np.uint8),
                connectivity=4,
                ltype=cv2.CV_32S,
            )
        )

        # OpenCV's numerical labels are implementation details.  Derive each
        # component's first raster cell so the retained-component order is the
        # same as the previous scan-order flood-fill, which keeps downstream
        # temporary IDs deterministic.
        flat_post_labels = np.asarray(post_labels, dtype=np.int32).reshape(-1)
        flat_pre_labels = np.asarray(pre_labels, dtype=np.int32).reshape(-1)
        component_ids = np.arange(1, int(post_count), dtype=np.int32)
        free_indices = np.flatnonzero(flat_post_labels > 0)
        first_indices = np.full(
            int(post_count),
            flat_post_labels.size,
            dtype=np.intp,
        )
        np.minimum.at(
            first_indices,
            flat_post_labels[free_indices],
            free_indices,
        )
        pre_component_ids = flat_pre_labels[first_indices[component_ids]]

        # A post-cut component is eligible only if its pre-cut component was
        # split into at least two post-cut pieces.  ``0`` is the OpenCV
        # background label; a post-cut free cell must never map to it, but the
        # explicit check preserves the old ``pre_label < 0`` rejection.
        pre_split_counts = np.bincount(
            pre_component_ids,
            minlength=int(pre_count),
        )
        split_by_cut = (
            (pre_component_ids > 0)
            & (pre_split_counts[pre_component_ids] >= 2)
        )

        # The old helper checked each component against the eight neighbours
        # of the portal cut.  Dilation provides exactly that relation here;
        # including the centre has no effect because cut cells were removed
        # from ``segmentation_free`` before components were labelled.
        portal_neighbourhood = cv2.dilate(
            (portal_cut_mask > 0).astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).reshape(-1) > 0
        touches_portal = np.zeros(int(post_count), dtype=bool)
        touches_portal[np.unique(flat_post_labels[portal_neighbourhood])] = True
        touches_portal[0] = False

        seeded_cells = np.zeros(flat_post_labels.size, dtype=bool)
        for component in core_component_cells.values():
            indices = np.asarray(component, dtype=np.intp)
            if indices.size:
                seeded_cells[indices] = True
        contains_seed = np.zeros(int(post_count), dtype=bool)
        contains_seed[np.unique(flat_post_labels[seeded_cells])] = True
        contains_seed[0] = False

        areas = post_stats[component_ids, cv2.CC_STAT_AREA]
        keep = (
            (areas >= self.room_min_component_cells)
            & split_by_cut
            & ~contains_seed[component_ids]
            & touches_portal[component_ids]
        )
        kept_component_ids = component_ids[keep]
        if kept_component_ids.size == 0:
            return []
        kept_component_ids = kept_component_ids[
            np.argsort(first_indices[kept_component_ids], kind="stable")
        ]
        return [
            np.flatnonzero(flat_post_labels == component_id).astype(np.intp).tolist()
            for component_id in kept_component_ids
        ]

    def _portal_hint_key(self, observation, center):
        explicit = (
            observation.get("id")
            or observation.get("instance_id")
            or observation.get("source_object_name")
            or observation.get("object_id")
            or observation.get("name")
        )
        if explicit not in (None, ""):
            return str(explicit)
        for key, hint in self.state.portal_hints.items():
            if self._distance_xy(hint["candidate_center"], center) <= self.room_portal_hint_merge_distance_m:
                return key
        return "portal_{:.2f}_{:.2f}".format(float(center[0]), float(center[1]))

    @staticmethod
    def _is_portal_observation(observation):
        if bool(observation.get("is_door")):
            return True
        label = str(
            observation.get("semantic_name")
            or observation.get("category")
            or observation.get("name")
            or ""
        ).lower()
        return "door" in label or "portal" in label or "gate" in label

    @staticmethod
    def _point3(value):
        if isinstance(value, dict):
            value = [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)]
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        padded = list(value[:3]) + [0.0] * max(0, 3 - len(value))
        try:
            return [float(padded[0]), float(padded[1]), float(padded[2])]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _distance_xy(first, second):
        dx = float(first[0]) - float(second[0])
        dy = float(first[1]) - float(second[1])
        return float(np.hypot(dx, dy))

    @staticmethod
    def _line_may_intersect_grid(start, end, width, height):
        return not (
            max(start[0], end[0]) < 0
            or min(start[0], end[0]) >= width
            or max(start[1], end[1]) < 0
            or min(start[1], end[1]) >= height
        )

    @staticmethod
    def _component_touches_mask(component, mask, width, height):
        for index in component:
            x = int(index) % width
            y = int(index) // width
            for dx, dy in (
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] > 0:
                    return True
        return False

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
        component_count, labels, stats, _centroids = self._connected_components_with_stats(
            occupied_mask, cv2, connectivity=8
        )
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

    @staticmethod
    def _free_components_with_labels(segmentation_free, width, height):
        flat = segmentation_free.reshape(height * width)
        visited = np.zeros(height * width, dtype=bool)
        labels = np.full(height * width, -1, dtype=np.int32)
        components = []
        for index in range(height * width):
            if visited[index] or flat[index] <= 0:
                continue
            visited[index] = True
            queue = deque([index])
            component = [index]
            component_id = len(components)
            labels[index] = component_id
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
                    labels[nidx] = component_id
                    component.append(nidx)
                    queue.append(nidx)
            components.append(component)
        return components, labels

    @staticmethod
    def _component_flat_indices(labels, stat, component_id, cv2):
        left = int(stat[cv2.CC_STAT_LEFT])
        top = int(stat[cv2.CC_STAT_TOP])
        component_width = int(stat[cv2.CC_STAT_WIDTH])
        component_height = int(stat[cv2.CC_STAT_HEIGHT])
        rows, cols = np.nonzero(
            labels[top:top + component_height, left:left + component_width] == component_id
        )
        return (rows + top) * labels.shape[1] + cols + left

    def _fallback_component_cells(self, segmentation_free, width, height,
                                  *, minimum_cells=None, cv2=None):
        minimum = self.room_core_min_component_cells if minimum_cells is None else minimum_cells
        if cv2 is not None:
            count, labels, stats, _ = self._connected_components_with_stats(
                segmentation_free, cv2, connectivity=4
            )
            accepted = {}
            for component_id in range(1, count):
                if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum:
                    accepted[len(accepted) + 1] = self._component_flat_indices(
                        labels, stats[component_id], component_id, cv2
                    )
            return accepted
        components, _ = self._free_components_with_labels(
            segmentation_free,
            width,
            height,
        )
        accepted = {}
        next_temp_id = 1
        for component in components:
            if len(component) < minimum:
                continue
            accepted[next_temp_id] = component
            next_temp_id += 1
        return accepted

    def _remap_room_component_ids(self, component_cells, grid_info):
        remapped = {}
        used_previous = set()
        merge_candidates = []
        stable_ids = None
        signature = self._grid_signature(grid_info)
        if (
            self.state.stable_room_grid_signature == signature
            and self.state.stable_room_ids
        ):
            stable_ids = self.state.stable_room_ids
        previous_ids = stable_ids
        # Pending splits must inherit candidate IDs until their grid is accepted.
        # Keep the published grid as a fallback when a transient merge re-splits.
        if self.state.prev_room_grid_signature == signature and self.state.prev_room_ids:
            previous_ids = self.state.prev_room_ids

        # Materialize the previous raster once. Converting the complete room
        # grid inside every component loop becomes expensive on large OCC
        # maps with several seeded rooms.
        previous_array = (
            np.asarray(previous_ids, dtype=np.int32)
            if previous_ids is not None
            else None
        )
        stable_array = (previous_array if stable_ids is previous_ids else
                        np.asarray(stable_ids, dtype=np.int32) if stable_ids is not None else None)

        for temp_room_id, component in sorted(component_cells.items(), key=lambda item: -len(item[1])):
            previous_overlaps = self._room_component_overlaps(component, previous_array)
            stable_overlaps = (
                previous_overlaps
                if stable_ids is previous_ids
                else self._room_component_overlaps(component, stable_array)
            )
            room_id = None
            for overlaps in (previous_overlaps, stable_overlaps):
                matches = [
                    (overlap, previous_id)
                    for previous_id, overlap in overlaps.items()
                    if previous_id not in used_previous
                    and overlap / max(len(component), 1) >= self.room_id_overlap_ratio
                ]
                if matches:
                    _overlap, room_id = min(matches, key=lambda item: (-item[0], item[1]))
                    break
            if room_id is None:
                room_id = self.state.next_room_segment_id
                self.state.next_room_segment_id += 1
            remapped[temp_room_id] = room_id
            used_previous.add(room_id)

            # Merge evidence stays relative to the published grid, even after
            # the previous candidate already contains the proposed merge.
            for stable_id, overlap in stable_overlaps.items():
                if overlap / max(len(component), 1) >= self.room_id_overlap_ratio:
                    merge_candidates.append((overlap, stable_id, room_id))
        observed_merges = {}
        for _overlap, secondary, primary in sorted(merge_candidates, reverse=True):
            if secondary not in used_previous:
                observed_merges.setdefault(secondary, primary)
        self._update_merge_confirmations(observed_merges)
        return remapped

    @staticmethod
    def _room_component_overlaps(component, room_ids):
        if room_ids is None:
            return {}
        values = np.asarray(room_ids, dtype=np.int32)[np.asarray(component, dtype=np.int64)]
        values, counts = np.unique(values[values >= 0], return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))

    def _update_merge_confirmations(self, observed_merges):
        next_pending = {}
        for secondary, primary in observed_merges.items():
            key = (int(secondary), int(primary))
            count = int(self.state.pending_merges.get(key, 0)) + 1
            next_pending[key] = min(count, self.room_merge_confirmations)
        self.state.pending_merges = next_pending

    def _commit_room_merges(self):
        # Consumers must never redirect rooms while the published grid still
        # contains the pre-merge labels or after an unconfirmed merge reverses.
        for (secondary, primary), count in self.state.pending_merges.items():
            if count >= self.room_merge_confirmations:
                self.state.last_confirmed_merges[secondary] = primary
        self.state.pending_merges.clear()

    @staticmethod
    def _grid_signature(grid_info):
        return (
            int(grid_info.width),
            int(grid_info.height),
            float(grid_info.resolution),
            float(grid_info.origin.position.x),
            float(grid_info.origin.position.y),
            round(grid_origin_yaw(grid_info), 6),
        )
