import math
import time

from .geometry_utils import euclidean_2d, grid_index, normalize_label, point_dict, world_to_grid


class ObjectMapStore:
    def __init__(
        self,
        match_distance=0.5,
        stale_after_sec=0.0,
        min_confirmations=2,
        size_match_ratio=0.7,
    ):
        self.match_distance = float(match_distance)
        self.stale_after_sec = float(stale_after_sec)
        self.min_confirmations = max(1, int(min_confirmations))
        self.size_match_ratio = max(0.05, float(size_match_ratio))
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
                    "label_votes": {label: float(confidence)},
                    "conf": confidence,
                    "coord": [pos["x"], pos["y"], pos["z"]],
                    "aabb_center": [center["x"], center["y"], center["z"]],
                    "aabb_size": [size["x"], size["y"], size["z"]],
                    "viz_aabb_center": [viz_center["x"], viz_center["y"], viz_center["z"]],
                    "viz_aabb_size": [viz_size["x"], viz_size["y"], viz_size["z"]],
                    "observation_count": 0,
                    "hit_streak": 0,
                    "miss_streak": 0,
                    "is_confirmed": False,
                    "instance_id": instance_id,
                    "last_seen": now,
                }
                self.next_id += 1
                self.objects.append(match)

            count = float(match["observation_count"])
            weight_old = count / (count + 1.0) if count > 0 else 0.0
            weight_new = 1.0 / (count + 1.0)
            match["coord"] = [
                weight_old * match["coord"][0] + weight_new * pos["x"],
                weight_old * match["coord"][1] + weight_new * pos["y"],
                weight_old * match["coord"][2] + weight_new * pos["z"],
            ]
            match["aabb_center"] = [
                weight_old * match["aabb_center"][0] + weight_new * center["x"],
                weight_old * match["aabb_center"][1] + weight_new * center["y"],
                weight_old * match["aabb_center"][2] + weight_new * center["z"],
            ]
            match["aabb_size"] = [
                weight_old * match["aabb_size"][0] + weight_new * size["x"],
                weight_old * match["aabb_size"][1] + weight_new * size["y"],
                weight_old * match["aabb_size"][2] + weight_new * size["z"],
            ]
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

    def as_tracked_detections(self):
        detections = []
        for obj in self.objects:
            if not obj.get("is_confirmed"):
                continue
            detections.append(
                {
                    "semantic_class": obj["semantic_name"],
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
                        obj["viz_aabb_center"][0], obj["viz_aabb_center"][1], obj["viz_aabb_center"][2]
                    ),
                    "viz_aabb_size": point_dict(
                        obj["viz_aabb_size"][0], obj["viz_aabb_size"][1], obj["viz_aabb_size"][2]
                    ),
                }
            )
        return detections

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
            if not self._size_compatible(obj.get("aabb_size", [0.0, 0.0, 0.0]), [size["x"], size["y"], size["z"]]):
                continue
            if obj["semantic_name"] != label and not self._should_merge_cross_label(obj, pos, size):
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
        old_norm = max(sum(abs(float(v)) for v in old_size), 1e-3)
        new_norm = max(sum(abs(float(v)) for v in new_size), 1e-3)
        ratio = min(old_norm, new_norm) / max(old_norm, new_norm)
        return ratio >= self.size_match_ratio

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
