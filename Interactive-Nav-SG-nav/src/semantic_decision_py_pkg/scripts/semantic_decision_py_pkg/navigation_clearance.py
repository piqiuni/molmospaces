"""Public occupancy clearance for the robot footprint at a navigation pose."""

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, label


@dataclass(frozen=True)
class ArrivalClearanceGrid:
    values: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    frame_id: str
    occupied_threshold: int = 50
    inscribed_threshold: int | None = None

    def cell(self, xy):
        dx, dy = xy[0] - self.origin_x, xy[1] - self.origin_y
        x = math.cos(self.origin_yaw)*dx + math.sin(self.origin_yaw)*dy
        y = -math.sin(self.origin_yaw)*dx + math.cos(self.origin_yaw)*dy
        col, row = math.floor(x/self.resolution), math.floor(y/self.resolution)
        return (row, col) if 0 <= row < self.values.shape[0] and 0 <= col < self.values.shape[1] else None

    @cached_property
    def components(self):
        # Four-connected free space cannot cross a diagonal obstacle corner.
        result = label((self.values >= 0) & (self.values < (self.inscribed_threshold or self.occupied_threshold)))[0]
        result.flags.writeable = False
        return result

    def _reuse_preprocessing(self, previous):
        if previous is None or self.values.flags.writeable or previous.values.flags.writeable:
            return self
        geometry = ("resolution", "origin_x", "origin_y", "origin_yaw", "frame_id",
                    "occupied_threshold", "inscribed_threshold")
        if (self.values.shape != previous.values.shape
                or any(getattr(self, key) != getattr(previous, key) for key in geometry)):
            return self
        cached = {key: previous.__dict__[key] for key in ("components", "obstacle_distance")
                  if key in previous.__dict__}
        if not cached:
            return self
        equal_values = np.array_equal(self.values, previous.values)
        for key, result in cached.items():
            threshold = ((self.inscribed_threshold or self.occupied_threshold)
                         if key == "components" else self.occupied_threshold)
            if equal_values or np.array_equal(
                (self.values >= 0) & (self.values < threshold),
                (previous.values >= 0) & (previous.values < threshold),
            ):
                # A new snapshot still owns current costs; only immutable mask
                # products are shared, never the mutable ROS message or its age.
                self.__dict__[key] = result
        return self

    def reachable(self, start, goal):
        a, b = self.cell(start), self.cell(goal)
        if a is None or b is None:
            return {"clear": False, "reason": "path_outside_map"}
        start_id, goal_id = int(self.components[a]), int(self.components[b])
        return {"clear": bool(start_id and start_id == goal_id),
                "reason": "path_connected" if start_id and start_id == goal_id else "path_disconnected",
                "start_component": start_id, "goal_component": goal_id,
                "start_cost": int(self.values[a]), "goal_cost": int(self.values[b])}

    def updated(self, message):
        x, y, w, h = int(message.x), int(message.y), int(message.width), int(message.height)
        if (x < 0 or y < 0 or w < 0 or h < 0 or x+w > self.values.shape[1]
                or y+h > self.values.shape[0] or len(message.data) != w*h):
            raise ValueError("invalid costmap update")
        values = self.values.copy()
        values[y:y+h, x:x+w] = np.asarray(message.data).reshape(h, w)
        values.flags.writeable = False
        return ArrivalClearanceGrid(values, self.resolution, self.origin_x, self.origin_y,
                                    self.origin_yaw, self.frame_id, self.occupied_threshold,
                                    self.inscribed_threshold)._reuse_preprocessing(self)

    @cached_property
    def obstacle_distance(self):
        free = (self.values >= 0) & (self.values < self.occupied_threshold)
        result = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * self.resolution
        result.flags.writeable = False
        return result

    def visible(self, start, end):
        steps = max(1, int(math.ceil(math.dist(start[:2], end[:2]) / (self.resolution * .5))))
        for t in np.linspace(0.0, 1.0, steps+1):
            cell = self.cell((start[0]+t*(end[0]-start[0]), start[1]+t*(end[1]-start[1])))
            if cell is None or not 0 <= self.values[cell] < self.occupied_threshold:
                return False
        return True

    def recovery_goals(self, goal, tolerance, targets, *, radius_m=.8, robot_radius_m=.25, safety_margin_m=.05):
        """Nearby footprint-safe points that still see the same frontier."""
        center = self.cell(goal)
        if center is None:
            return []
        row, col = center
        extent = int(math.ceil(radius_m/self.resolution))
        required = robot_radius_m + safety_margin_m + self.resolution/math.sqrt(2)
        choices = []
        for y in range(max(0, row-extent), min(self.values.shape[0], row+extent+1)):
            for x in range(max(0, col-extent), min(self.values.shape[1], col+extent+1)):
                if self.obstacle_distance[y, x] <= required:
                    continue
                lx, ly = (x+.5)*self.resolution, (y+.5)*self.resolution
                xy = [self.origin_x+math.cos(self.origin_yaw)*lx-math.sin(self.origin_yaw)*ly,
                      self.origin_y+math.sin(self.origin_yaw)*lx+math.cos(self.origin_yaw)*ly]
                distance = math.dist(xy, goal[:2])
                if distance <= radius_m:
                    choices.append((distance, xy))
        goals = []
        for _, xy in sorted(choices):
            target = next((p for p in sorted(targets, key=lambda p: math.dist(xy, p[:2])) if self.visible(xy, p)), None)
            if target is not None:
                goals.append([*xy, math.atan2(target[1]-xy[1], target[0]-xy[0])])
                if len(goals) >= 64:
                    break
        return goals

    @classmethod
    def from_message(cls, message, *, costmap=False, previous=None):
        info = message.info
        if (int(info.width) <= 0 or int(info.height) <= 0
                or not math.isfinite(float(info.resolution)) or float(info.resolution) <= 0
                or not str(message.header.frame_id or "")):
            raise ValueError("invalid occupancy geometry")
        values = np.asarray(message.data, dtype=np.int16).reshape(int(info.height), int(info.width)).copy()
        values.flags.writeable = False
        q = info.origin.orientation
        yaw = math.atan2(2.0 * (q.w*q.z + q.x*q.y), 1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        return cls(values, float(info.resolution), float(info.origin.position.x),
                   float(info.origin.position.y), yaw, str(message.header.frame_id),
                   100 if costmap else 50, 99 if costmap else None)._reuse_preprocessing(previous)

    def check(self, goal_xy, arrival_tolerance_m, *, robot_radius_m=0.25, safety_margin_m=0.05):
        # Arrival tolerance controls stopping, not obstacle clearance.
        radius = robot_radius_m + safety_margin_m + self.resolution/math.sqrt(2.0)
        detail = {"clearance_radius_m": radius, "arrival_region_radius_m": 0.0,
                  "costmap_resolution_m": self.resolution, "costmap_frame": self.frame_id}
        if not all(math.isfinite(float(v)) for v in (*goal_xy[:2], radius, self.resolution,
                                                     self.origin_x, self.origin_y, self.origin_yaw)) or radius < 0:
            return {**detail, "clear": False, "reason": "invalid_geometry"}
        dx, dy = goal_xy[0]-self.origin_x, goal_xy[1]-self.origin_y
        gx = math.cos(self.origin_yaw)*dx + math.sin(self.origin_yaw)*dy
        gy = -math.sin(self.origin_yaw)*dx + math.cos(self.origin_yaw)*dy
        height, width = self.values.shape
        if not (radius <= gx < width*self.resolution-radius and radius <= gy < height*self.resolution-radius):
            return {**detail, "clear": False, "reason": "outside_map_window"}
        col, row = int(gx/self.resolution), int(gy/self.resolution)
        center = int(self.values[row, col])
        detail["center_cost"] = center
        if center < 0 or center >= (self.inscribed_threshold or self.occupied_threshold):
            return {**detail, "clear": False, "reason": "center_blocked"}
        extent = int(math.ceil(max(radius, 0.8)/self.resolution)) + 1
        x0, x1 = max(0, col-extent), min(width, col+extent+1)
        y0, y1 = max(0, row-extent), min(height, row+extent+1)
        patch = self.values[y0:y1, x0:x1]
        # Inscribed cost already includes the robot footprint. Sweep only
        # lethal/unknown cells, otherwise the footprint is inflated twice.
        yy, xx = np.where((patch < 0) | (patch >= self.occupied_threshold))
        distances = np.hypot((xx+x0+.5)*self.resolution-gx, (yy+y0+.5)*self.resolution-gy)
        nearest = float(distances.min()) if distances.size else max(radius, 0.8)
        clear = not distances.size or nearest > radius
        return {**detail, "obstacle_clearance_m": nearest, "clear": clear,
                "reason": "clearance_confirmed" if clear else "arrival_region_blocked"}
