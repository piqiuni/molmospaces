import math
import time
from statistics import median

from .geometry_utils import euclidean_2d, grid_index, normalize_label, point_dict, world_to_grid


class ObjectMapStore:
    STABLE_BOX_MIN_OVERLAP = 0.5
    STABLE_BOX_MIN_SIZE_RATIO = 0.4
    STABLE_BOX_UPDATE_ALPHA = 0.45
    STABLE_BOX_MAX_STEP_RATIO = 1.35
    STABLE_BOX_MIN_ABS_STEP = 0.08

    def __init__(
        self,
        match_distance=0.5,
        stale_after_sec=0.0,
        min_confirmations=2,
        size_match_ratio=0.7,
        stable_history_size=5,
    ):
        self.match_distance = float(match_distance)
        self.stale_after_sec = float(stale_after_sec)
        self.min_confirmations = max(1, int(min_confirmations))
        self.size_match_ratio = max(0.05, float(size_match_ratio))
        self.stable_history_size = max(1, int(stable_history_size))
        self.objects = []
        self.next_id = 1

    def update(self, detections, stamp):
        now = float(stamp if stamp is not None else time.time())
        matched_ids = set()
        for det in detections:
            label = normalize_label(det.get("semantic_class") or det.get("class") or det.get("semantic_name"))
            if not label:
                continue
            pos = self._point_from_detection(det, "world_position", "position")
            confidence = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
            instance_id = str(det.get("instance_id", ""))
            size = self._point_from_detection(det, "world_box3d_size", "box3d_size", "size")
            center = self._point_from_detection(det, "world_box3d_center", "box3d_center", "world_position", "position")
            viz_center = self._point_from_detection(
                det,
                "world_box3d_center",
                "aabb_center",
                "box3d_center",
                "world_position",
                "position",
            )
            viz_size = self._point_from_detection(
                det,
                "world_box3d_size",
                "aabb_size",
                "box3d_size",
                "size",
            )

            match = self._find_match(label, pos, size, instance_id)
            if match is None:
                match = {
                    "object_id": self.next_id,
                    "track_id": f"track_{self.next_id:04d}",
                    "semantic_name": label,
                    "label_votes": {},
                    "conf": confidence,
                    "coord": [pos["x"], pos["y"], pos["z"]],
                    "aabb_center": [center["x"], center["y"], center["z"]],
                    "aabb_size": [size["x"], size["y"], size["z"]],
                    "viz_aabb_center": [viz_center["x"], viz_center["y"], viz_center["z"]],
                    "viz_aabb_size": [viz_size["x"], viz_size["y"], viz_size["z"]],
                    "coord_history": [],
                    "observation_count": 0,
                    "hit_streak": 0,
                    "miss_streak": 0,
                    "is_confirmed": False,
                    "instance_id": instance_id,
                    "last_seen": now,
                }
                self.next_id += 1
                self.objects.append(match)

            self._append_history(
                match,
                "coord_history",
                [pos["x"], pos["y"], pos["z"]],
            )
            match["coord"] = self._history_median(match.get("coord_history"))
            if match["observation_count"] <= 0 or self._should_update_stable_box(match, center, size):
                blended_center, blended_size = self._blend_stable_box(
                    match.get("aabb_center", [center["x"], center["y"], center["z"]]),
                    match.get("aabb_size", [size["x"], size["y"], size["z"]]),
                    [center["x"], center["y"], center["z"]],
                    [size["x"], size["y"], size["z"]],
                    is_first=match["observation_count"] <= 0,
                )
                match["aabb_center"] = blended_center
                match["aabb_size"] = blended_size
            match["viz_aabb_center"] = [viz_center["x"], viz_center["y"], viz_center["z"]]
            match["viz_aabb_size"] = [viz_size["x"], viz_size["y"], viz_size["z"]]
            match["conf"] = min(1.0, max(float(match["conf"]), confidence) + 0.05 * confidence)
            label_votes = dict(match.get("label_votes") or {})
            label_votes[label] = float(label_votes.get(label, 0.0)) + max(confidence, 0.05)
            match["label_votes"] = label_votes
            match["semantic_name"] = max(
                sorted(label_votes.keys()),
                key=lambda key: (float(label_votes[key]), key == match.get("semantic_name")),
            )
            match["observation_count"] += 1
            match["hit_streak"] = int(match.get("hit_streak", 0)) + 1
            match["miss_streak"] = 0
            match["is_confirmed"] = bool(match["observation_count"] >= self.min_confirmations)
            match["last_seen"] = now
            matched_ids.add(int(match["object_id"]))

        for obj in self.objects:
            if int(obj["object_id"]) in matched_ids:
                continue
            obj["hit_streak"] = 0
            obj["miss_streak"] = int(obj.get("miss_streak", 0)) + 1

        self._purge_stale(now)

    def as_obj_map(self):
        return [
            {
                "semantic_name": obj["semantic_name"],
                "candidate_labels": self._candidate_labels(obj),
                "label_votes": {str(k): float(v) for k, v in (obj.get("label_votes") or {}).items()},
                "conf": float(obj["conf"]),
                "coord": [float(v) for v in obj["coord"]],
                "object_id": int(obj["object_id"]),
                "track_id": str(obj["track_id"]),
                "observation_count": int(obj["observation_count"]),
                "aabb_center": [float(v) for v in obj["aabb_center"]],
                "aabb_size": [float(v) for v in obj["aabb_size"]],
            }
            for obj in self.objects
            if obj.get("is_confirmed")
        ]

    def as_tracked_detections(self, min_observations=None, confirmed_only=True):
        if min_observations is None:
            min_observations = self.min_confirmations if confirmed_only else 1
        min_observations = max(1, int(min_observations))
        detections = []
        for obj in self.objects:
            if confirmed_only and not obj.get("is_confirmed"):
                continue
            if int(obj.get("observation_count", 0)) < min_observations:
                continue
            detections.append(
                {
                    "semantic_class": obj["semantic_name"],
                    "candidate_labels": self._candidate_labels(obj),
                    "label_votes": {str(k): float(v) for k, v in (obj.get("label_votes") or {}).items()},
                    "confidence": float(obj["conf"]),
                    "instance_id": str(obj.get("instance_id") or obj["track_id"]),
                    "object_id": int(obj["object_id"]),
                    "track_id": str(obj["track_id"]),
                    "world_position": point_dict(obj["coord"][0], obj["coord"][1], obj["coord"][2]),
                    "world_box3d_center": point_dict(
                        obj["aabb_center"][0], obj["aabb_center"][1], obj["aabb_center"][2]
                    ),
                    "world_box3d_size": point_dict(
                        obj["aabb_size"][0], obj["aabb_size"][1], obj["aabb_size"][2]
                    ),
                    "observation_count": int(obj["observation_count"]),
                    "source": "tracked_object_store",
                    "viz_aabb_center": point_dict(
                        obj["aabb_center"][0], obj["aabb_center"][1], obj["aabb_center"][2]
                    ),
                    "viz_aabb_size": point_dict(
                        obj["aabb_size"][0], obj["aabb_size"][1], obj["aabb_size"][2]
                    ),
                    "latest_box3d_center": point_dict(
                        obj["viz_aabb_center"][0], obj["viz_aabb_center"][1], obj["viz_aabb_center"][2]
                    ),
                    "latest_box3d_size": point_dict(
                        obj["viz_aabb_size"][0], obj["viz_aabb_size"][1], obj["viz_aabb_size"][2]
                    ),
                }
            )
        return detections

    def _append_history(self, obj, key, value):
        history = list(obj.get(key) or [])
        history.append([float(value[0]), float(value[1]), float(value[2])])
        if len(history) > self.stable_history_size:
            history = history[-self.stable_history_size :]
        obj[key] = history

    def _history_median(self, history):
        history = list(history or [])
        if not history:
            return [0.0, 0.0, 0.0]
        dims = list(zip(*history))
        return [float(median(axis_values)) for axis_values in dims]

    def _blend_stable_box(self, old_center, old_size, new_center, new_size, is_first=False):
        if is_first:
            return [float(v) for v in new_center], [float(v) for v in new_size]
        alpha = self.STABLE_BOX_UPDATE_ALPHA
        blended_center = [
            float(old_center[axis]) * (1.0 - alpha) + float(new_center[axis]) * alpha
            for axis in range(3)
        ]
        min_step_ratio = 1.0 / self.STABLE_BOX_MAX_STEP_RATIO
        blended_size = []
        for axis in range(3):
            old_axis = max(float(old_size[axis]), 0.02)
            new_axis = max(float(new_size[axis]), 0.02)
            target = old_axis * (1.0 - alpha) + new_axis * alpha
            lower = max(0.02, min(old_axis * min_step_ratio, old_axis - self.STABLE_BOX_MIN_ABS_STEP))
            upper = max(old_axis * self.STABLE_BOX_MAX_STEP_RATIO, old_axis + self.STABLE_BOX_MIN_ABS_STEP)
            blended_size.append(min(max(target, lower), upper))
        return blended_center, blended_size

    def _find_match(self, label, pos, size, instance_id):
        best = None
        best_dist = math.inf
        for obj in self.objects:
            if instance_id and obj.get("instance_id") == instance_id:
                return obj
            obj_pos = point_dict(obj["coord"][0], obj["coord"][1], obj["coord"][2])
            dist = euclidean_2d(pos, obj_pos)
            if dist >= self.match_distance or dist >= best_dist:
                continue
            if obj["semantic_name"] != label:
                if not self._should_merge_cross_label(obj, pos, size):
                    continue
            elif not self._size_compatible(obj.get("aabb_size", [0.0, 0.0, 0.0]), [size["x"], size["y"], size["z"]]):
                old_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
                old_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
                overlap = self._aabb_overlap_ratio(
                    old_center,
                    old_size,
                    [pos["x"], pos["y"], pos["z"]],
                    [size["x"], size["y"], size["z"]],
                )
                if overlap < 0.15:
                    continue
            if dist < best_dist:
                best = obj
                best_dist = dist
        return best

    def _point_from_detection(self, det, *keys):
        for key in keys:
            value = det.get(key)
            if not isinstance(value, dict):
                continue
            return point_dict(value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0))
        return point_dict()

    def _size_compatible(self, old_size, new_size):
        return self._size_ratio(old_size, new_size) >= self.size_match_ratio

    def _size_ratio(self, old_size, new_size):
        old_norm = max(sum(abs(float(v)) for v in old_size), 1e-3)
        new_norm = max(sum(abs(float(v)) for v in new_size), 1e-3)
        return min(old_norm, new_norm) / max(old_norm, new_norm)

    def _should_update_stable_box(self, obj, center, size):
        old_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
        old_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
        new_center = [center["x"], center["y"], center["z"]]
        new_size = [size["x"], size["y"], size["z"]]
        if self._size_compatible(old_size, new_size):
            return True
        overlap = self._aabb_overlap_ratio(old_center, old_size, new_center, new_size)
        size_ratio = self._size_ratio(old_size, new_size)
        return overlap >= self.STABLE_BOX_MIN_OVERLAP and size_ratio >= self.STABLE_BOX_MIN_SIZE_RATIO

    def _candidate_labels(self, obj):
        label_votes = obj.get("label_votes") or {}
        return [
            str(label)
            for label, _score in sorted(
                label_votes.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )
        ]

    def _should_merge_cross_label(self, obj, pos, size):
        obj_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
        obj_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
        overlap = self._aabb_overlap_ratio(
            obj_center,
            obj_size,
            [pos["x"], pos["y"], pos["z"]],
            [size["x"], size["y"], size["z"]],
        )
        return overlap >= 0.6

    def _aabb_overlap_ratio(self, center_a, size_a, center_b, size_b):
        mins_a = [float(center_a[i]) - 0.5 * max(float(size_a[i]), 0.02) for i in range(3)]
        maxs_a = [float(center_a[i]) + 0.5 * max(float(size_a[i]), 0.02) for i in range(3)]
        mins_b = [float(center_b[i]) - 0.5 * max(float(size_b[i]), 0.02) for i in range(3)]
        maxs_b = [float(center_b[i]) + 0.5 * max(float(size_b[i]), 0.02) for i in range(3)]
        inter = 1.0
        vol_a = 1.0
        vol_b = 1.0
        for axis in range(3):
            inter_axis = max(0.0, min(maxs_a[axis], maxs_b[axis]) - max(mins_a[axis], mins_b[axis]))
            inter *= inter_axis
            vol_a *= max(0.02, maxs_a[axis] - mins_a[axis])
            vol_b *= max(0.02, maxs_b[axis] - mins_b[axis])
        union = max(vol_a + vol_b - inter, 1e-6)
        iou = inter / union
        contained = inter / max(min(vol_a, vol_b), 1e-6)
        return max(iou, contained)

    def _purge_stale(self, now):
        if self.stale_after_sec <= 0.0:
            return
        self.objects = [obj for obj in self.objects if now - obj.get("last_seen", now) <= self.stale_after_sec]


class SceneGridStore:
    def __init__(self, unknown_id=-1, confidence_step=5):
        self.unknown_id = int(unknown_id)
        self.confidence_step = int(confidence_step)
        self.info = None
        self.scene_data = []
        self.confidence_data = []

    def initialize_from_occupancy_grid(self, occ_grid):
        self.info = occ_grid.info
        size = int(self.info.width * self.info.height)
        if len(self.scene_data) != size:
            self.scene_data = [self.unknown_id] * size
            self.confidence_data = [-1] * size

    def update_cells(self, world_points, scene_id):
        if self.info is None or scene_id is None or scene_id < 0:
            return
        for x, y in world_points:
            coords = world_to_grid(x, y, self.info)
            if coords is None:
                continue
            idx = grid_index(coords[0], coords[1], self.info.width)
            if self.scene_data[idx] == scene_id:
                old = 0 if self.confidence_data[idx] < 0 else self.confidence_data[idx]
                self.confidence_data[idx] = min(100, old + self.confidence_step)
            else:
                self.scene_data[idx] = int(scene_id)
                self.confidence_data[idx] = max(self.confidence_step, 1)
