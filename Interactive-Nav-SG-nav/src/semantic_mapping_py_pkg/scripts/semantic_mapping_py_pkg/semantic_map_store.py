import math
import time

from .geometry_utils import euclidean_2d, grid_index, normalize_label, point_dict, world_to_grid


class ObjectMapStore:
    def __init__(self, match_distance=0.5, stale_after_sec=0.0):
        self.match_distance = float(match_distance)
        self.stale_after_sec = float(stale_after_sec)
        self.objects = []
        self.next_id = 1

    def update(self, detections, stamp):
        now = time.time()
        for det in detections:
            label = normalize_label(det.get("semantic_class") or det.get("class") or det.get("semantic_name"))
            if not label:
                continue
            pos = det.get("world_position") or det.get("position") or {}
            pos = point_dict(pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0))
            confidence = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
            instance_id = str(det.get("instance_id", ""))

            match = self._find_match(label, pos, instance_id)
            if match is None:
                match = {
                    "object_id": self.next_id,
                    "semantic_name": label,
                    "conf": confidence,
                    "coord": [pos["x"], pos["y"], pos["z"]],
                    "observation_count": 0,
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
            match["conf"] = min(1.0, max(float(match["conf"]), confidence) + 0.05 * confidence)
            match["observation_count"] += 1
            match["last_seen"] = now

        self._purge_stale(now)

    def as_obj_map(self):
        return [
            {
                "semantic_name": obj["semantic_name"],
                "conf": float(obj["conf"]),
                "coord": [float(v) for v in obj["coord"]],
                "object_id": int(obj["object_id"]),
                "observation_count": int(obj["observation_count"]),
            }
            for obj in self.objects
        ]

    def _find_match(self, label, pos, instance_id):
        best = None
        best_dist = math.inf
        for obj in self.objects:
            if instance_id and obj.get("instance_id") == instance_id:
                return obj
            if obj["semantic_name"] != label:
                continue
            obj_pos = point_dict(obj["coord"][0], obj["coord"][1], obj["coord"][2])
            dist = euclidean_2d(pos, obj_pos)
            if dist < self.match_distance and dist < best_dist:
                best = obj
                best_dist = dist
        return best

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
