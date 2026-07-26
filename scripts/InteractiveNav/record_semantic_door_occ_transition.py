#!/usr/bin/env python3
from __future__ import annotations

import os

# This recorder runs beside MuJoCo, mapping, and move_base. Avoid importing
# NumPy/OpenCV with a machine-sized BLAS thread pool, which can starve the ROS
# callbacks on high-core-count hosts.
for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import threading
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEMANTIC_SCRIPTS = _REPO_ROOT / "Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts"
if str(_SEMANTIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_SCRIPTS))

import numpy as np
import rospy
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser(description="Record closed/open raw and semantic OCC around one door root.")
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=150.0)
    parser.add_argument("--closed-min-raw-messages", type=int, default=5)
    parser.add_argument("--phase-stable-publishes", type=int, default=3)
    parser.add_argument("--open-settle-raw-messages", type=int, default=3)
    parser.add_argument("--open-settle-global-messages", type=int, default=5)
    parser.add_argument("--clear-padding-m", type=float, default=0.10)
    parser.add_argument("--raw-topic", default="/struct_mapping/occ_map")
    parser.add_argument("--planning-topic", default="/semantic_mapping/planning_occ_map")
    parser.add_argument("--mask-topic", default="/semantic_mapping/door_clear_mask")
    parser.add_argument("--graph-topic", default="/semantic_mapping/unified_graph")
    parser.add_argument("--global-costmap-topic", default="/move_base/global_costmap/costmap")
    parser.add_argument(
        "--global-costmap-updates-topic",
        default="/move_base/global_costmap/costmap_updates",
    )
    parser.add_argument("--phase-file", type=Path, default=None)
    parser.add_argument("--expected-phases", type=int, default=0)
    parser.add_argument("--phase-settle-raw-messages", type=int, default=3)
    parser.add_argument("--phase-settle-planning-messages", type=int, default=2)
    parser.add_argument("--phase-settle-global-messages", type=int, default=3)
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--pose-tolerance-m", type=float, default=0.08)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.12)
    return parser.parse_args()


def grid_snapshot(message: OccupancyGrid) -> dict:
    return {
        "data": np.asarray(message.data, dtype=np.int16).reshape(
            int(message.info.height), int(message.info.width)
        ),
        "width": int(message.info.width),
        "height": int(message.info.height),
        "resolution": float(message.info.resolution),
        "origin": {
            "position": [
                float(message.info.origin.position.x),
                float(message.info.origin.position.y),
                float(message.info.origin.position.z),
            ],
            "orientation": [
                float(message.info.origin.orientation.x),
                float(message.info.origin.orientation.y),
                float(message.info.origin.orientation.z),
                float(message.info.origin.orientation.w),
            ],
        },
        "frame_id": str(message.header.frame_id),
        "stamp_sec": float(message.header.stamp.to_sec()),
    }


def grid_key(grid: dict | None):
    if grid is None:
        return None
    return (
        grid["width"],
        grid["height"],
        round(grid["resolution"], 9),
        tuple(round(value, 6) for value in grid["origin"]["position"]),
        tuple(round(value, 6) for value in grid["origin"]["orientation"]),
        grid["frame_id"],
    )


def compatible(*grids) -> bool:
    keys = [grid_key(grid) for grid in grids]
    return bool(keys and keys[0] is not None and all(key == keys[0] for key in keys[1:]))


def same_source_stamp(*grids, tolerance_s=1e-6) -> bool:
    stamps = [float(grid.get("stamp_sec", 0.0)) for grid in grids if grid is not None]
    return bool(stamps and max(stamps) - min(stamps) <= tolerance_s)


def portal_for_root(graph: dict | None, target_root: str):
    if not graph:
        return None
    for node in graph.get("nodes") or []:
        if node.get("type") != "portal":
            continue
        attrs = node.get("attributes") or {}
        if attrs.get("source_object_name") == target_root or node.get("name") == target_root:
            return node
    return None


def copy_phase(raw, planning, mask, graph, node, global_costmap):
    return {
        "raw": {**raw, "data": raw["data"].copy()},
        "planning": {**planning, "data": planning["data"].copy()},
        "mask": {**mask, "data": mask["data"].copy()},
        "global_costmap": (
            {**global_costmap, "data": global_costmap["data"].copy()}
            if global_costmap is not None
            else None
        ),
        "graph": json.loads(json.dumps(graph)),
        "portal": json.loads(json.dumps(node)),
        "capture_wall_time": time.time(),
    }


def origin_yaw(grid: dict) -> float:
    x, y, z, w = grid["origin"]["orientation"]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def aabb_mask(grid: dict, center, size, padding: float) -> np.ndarray:
    result = np.zeros((grid["height"], grid["width"]), dtype=bool)
    half_x = 0.5 * float(size[0]) + float(padding)
    half_y = 0.5 * float(size[1]) + float(padding)
    origin_x, origin_y = grid["origin"]["position"][:2]
    yaw = origin_yaw(grid)
    c = math.cos(yaw)
    s = math.sin(yaw)
    corners = []
    for wx in (float(center[0]) - half_x, float(center[0]) + half_x):
        for wy in (float(center[1]) - half_y, float(center[1]) + half_y):
            dx = wx - origin_x
            dy = wy - origin_y
            corners.append((c * dx + s * dy, -s * dx + c * dy))
    resolution = grid["resolution"]
    col_min = max(0, int(math.floor(min(point[0] for point in corners) / resolution)))
    col_max = min(grid["width"] - 1, int(math.ceil(max(point[0] for point in corners) / resolution) - 1))
    row_min = max(0, int(math.floor(min(point[1] for point in corners) / resolution)))
    row_max = min(grid["height"] - 1, int(math.ceil(max(point[1] for point in corners) / resolution) - 1))
    if col_min <= col_max and row_min <= row_max:
        result[row_min : row_max + 1, col_min : col_max + 1] = True
    return result


def occupancy_rgb(data: np.ndarray) -> np.ndarray:
    rgb = np.full((*data.shape, 3), 180, dtype=np.uint8)
    rgb[data < 0] = (128, 128, 128)
    rgb[data == 0] = (255, 255, 255)
    rgb[data >= 50] = (0, 0, 0)
    mid = (data > 0) & (data < 50)
    rgb[mid] = (220, 170, 80)
    return rgb


def mask_rgb(data: np.ndarray) -> np.ndarray:
    rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
    rgb[data > 0] = (40, 40, 230)
    return rgb


def labeled(image: np.ndarray, title: str, fixed_width: int | None = None) -> np.ndarray:
    import cv2

    image = cv2.resize(image, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_NEAREST)
    text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0]
    width = max(300, image.shape[1], text_width + 16, int(fixed_width or 0))
    canvas = np.full((image.shape[0] + 38, width, 3), 245, dtype=np.uint8)
    image_x = (width - image.shape[1]) // 2
    canvas[38:, image_x : image_x + image.shape[1]] = image
    cv2.putText(canvas, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def crop_bounds(mask: np.ndarray, margin=20):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0, mask.shape[0], 0, mask.shape[1]
    return (
        max(0, int(rows.min()) - margin),
        min(mask.shape[0], int(rows.max()) + margin + 1),
        max(0, int(cols.min()) - margin),
        min(mask.shape[1], int(cols.max()) + margin + 1),
    )


def phase_metadata(phase: dict) -> dict:
    metadata = {
        "raw": {key: value for key, value in phase["raw"].items() if key != "data"},
        "planning": {key: value for key, value in phase["planning"].items() if key != "data"},
        "mask": {key: value for key, value in phase["mask"].items() if key != "data"},
        "portal": phase["portal"],
        "capture_wall_time": phase["capture_wall_time"],
    }
    if phase["global_costmap"] is not None:
        metadata["global_costmap"] = {
            key: value for key, value in phase["global_costmap"].items() if key != "data"
        }
    return metadata


def save_phase_checkpoint(output_dir: Path, name: str, phase: dict) -> None:
    arrays = {
        "raw": phase["raw"]["data"],
        "planning": phase["planning"]["data"],
        "mask": phase["mask"]["data"],
    }
    if phase["global_costmap"] is not None:
        arrays["global_costmap"] = phase["global_costmap"]["data"]
    np.savez_compressed(output_dir / f"{name}_checkpoint_maps.npz", **arrays)
    (output_dir / f"{name}_checkpoint.json").write_text(
        json.dumps(phase_metadata(phase), ensure_ascii=False, indent=2)
    )


def save_results(output_dir: Path, closed: dict, opened: dict, args) -> dict:
    import cv2

    cv2.setNumThreads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, phase in (("closed", closed), ("open", opened)):
        arrays = {
            "raw": phase["raw"]["data"],
            "planning": phase["planning"]["data"],
            "mask": phase["mask"]["data"],
        }
        if phase["global_costmap"] is not None:
            arrays["global_costmap"] = phase["global_costmap"]["data"]
        np.savez_compressed(output_dir / f"{name}_maps.npz", **arrays)
        (output_dir / f"{name}_graph.json").write_text(
            json.dumps(phase["graph"], ensure_ascii=False, indent=2)
        )

    closed_node = closed["portal"]
    closed_region = aabb_mask(
        closed["raw"],
        closed_node["aabb_center"],
        closed_node["aabb_size"],
        args.clear_padding_m,
    )
    open_mask = opened["mask"]["data"] > 0
    closed_raw = closed["raw"]["data"]
    closed_planning = closed["planning"]["data"]
    open_raw = opened["raw"]["data"]
    open_planning = opened["planning"]["data"]
    closed_global = (
        closed["global_costmap"]["data"] if closed["global_costmap"] is not None else None
    )
    open_global = (
        opened["global_costmap"]["data"] if opened["global_costmap"] is not None else None
    )
    same_geometry = compatible(closed["raw"], opened["raw"])

    metrics = {
        "closed_raw_planning_diff_total": int(np.count_nonzero(closed_raw != closed_planning)),
        "closed_raw_planning_diff_in_door_aabb": int(
            np.count_nonzero((closed_raw != closed_planning) & closed_region)
        ),
        "closed_door_aabb_cells": int(np.count_nonzero(closed_region)),
        "open_clear_mask_cells": int(np.count_nonzero(open_mask)),
        "open_raw_planning_diff_total": int(np.count_nonzero(open_raw != open_planning)),
        "open_raw_planning_diff_in_clear_mask": int(
            np.count_nonzero((open_raw != open_planning) & open_mask)
        ),
        "open_raw_occupied_cells_in_clear_mask": int(np.count_nonzero((open_raw >= 50) & open_mask)),
        "open_raw_unknown_cells_in_clear_mask": int(np.count_nonzero((open_raw < 0) & open_mask)),
        "open_raw_free_cells_in_clear_mask": int(np.count_nonzero((open_raw == 0) & open_mask)),
        "open_planning_nonfree_cells_in_clear_mask": int(
            np.count_nonzero((open_planning != 0) & open_mask)
        ),
        "closed_open_same_geometry": bool(same_geometry),
    }
    global_compatible = (
        closed_global is not None
        and open_global is not None
        and compatible(
            closed["global_costmap"],
            opened["global_costmap"],
            opened["planning"],
        )
    )
    metrics["global_costmap_compatible"] = bool(global_compatible)
    if global_compatible:
        metrics.update(
            {
                "global_closed_to_open_changed_total": int(
                    np.count_nonzero(closed_global != open_global)
                ),
                "global_closed_to_open_changed_in_open_mask": int(
                    np.count_nonzero((closed_global != open_global) & open_mask)
                ),
                "open_global_free_cells_in_clear_mask": int(
                    np.count_nonzero((open_global == 0) & open_mask)
                ),
                "open_global_lethal_cells_in_clear_mask": int(
                    np.count_nonzero((open_global >= 100) & open_mask)
                ),
                "open_global_inscribed_cells_in_clear_mask": int(
                    np.count_nonzero((open_global > 0) & (open_global < 100) & open_mask)
                ),
            }
        )
    if same_geometry:
        metrics.update(
            {
                "raw_closed_to_open_changed_total": int(np.count_nonzero(closed_raw != open_raw)),
                "raw_closed_to_open_changed_in_open_mask": int(
                    np.count_nonzero((closed_raw != open_raw) & open_mask)
                ),
                "planning_closed_to_open_changed_total": int(
                    np.count_nonzero(closed_planning != open_planning)
                ),
                "planning_closed_to_open_changed_in_open_mask": int(
                    np.count_nonzero((closed_planning != open_planning) & open_mask)
                ),
            }
        )

    success_checks = {
        "closed_door_region_unchanged": metrics["closed_raw_planning_diff_in_door_aabb"] == 0,
        "open_mask_published": metrics["open_clear_mask_cells"] > 0,
        "open_mask_forced_free": metrics["open_planning_nonfree_cells_in_clear_mask"] == 0,
        "raw_map_still_has_nonfree_door_cells": (
            metrics["open_raw_occupied_cells_in_clear_mask"]
            + metrics["open_raw_unknown_cells_in_clear_mask"]
        )
        > 0,
        "semantic_map_differs_from_raw_at_door": metrics["open_raw_planning_diff_in_clear_mask"] > 0,
        "global_costmap_consumed_door_update": bool(
            global_compatible
            and metrics["global_closed_to_open_changed_in_open_mask"] > 0
            and metrics["open_global_free_cells_in_clear_mask"] > 0
        ),
    }
    summary = {
        "success": all(success_checks.values()),
        "target_root": args.target_root,
        "success_checks": success_checks,
        "metrics": metrics,
        "closed": phase_metadata(closed),
        "open": phase_metadata(opened),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    crop_mask = open_mask if np.any(open_mask) else closed_region
    r0, r1, c0, c1 = crop_bounds(crop_mask)
    closed_global_image = (
        occupancy_rgb(closed_global) if closed_global is not None else np.full_like(occupancy_rgb(closed_raw), 255)
    )
    open_global_image = (
        occupancy_rgb(open_global) if open_global is not None else np.full_like(occupancy_rgb(open_raw), 255)
    )
    panels = [
        labeled(np.flipud(occupancy_rgb(closed_raw)[r0:r1, c0:c1]), "raw OCC - closed"),
        labeled(np.flipud(occupancy_rgb(closed_planning)[r0:r1, c0:c1]), "semantic OCC - closed"),
        labeled(np.flipud(closed_global_image[r0:r1, c0:c1]), "global costmap - closed"),
        labeled(np.flipud(mask_rgb(closed["mask"]["data"])[r0:r1, c0:c1]), "clear mask - closed"),
        labeled(np.flipud(occupancy_rgb(open_raw)[r0:r1, c0:c1]), "raw OCC - open"),
        labeled(np.flipud(occupancy_rgb(open_planning)[r0:r1, c0:c1]), "semantic OCC - open"),
        labeled(np.flipud(open_global_image[r0:r1, c0:c1]), "global costmap - open"),
        labeled(np.flipud(mask_rgb(opened["mask"]["data"])[r0:r1, c0:c1]), "clear mask - open"),
    ]
    comparison = np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:])))
    cv2.imwrite(str(output_dir / "door_occ_comparison.png"), comparison)
    return summary


class Recorder:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.raw = None
        self.raw_history = deque(maxlen=40)
        self.planning = None
        self.mask = None
        self.global_costmap = None
        self.graph = None
        self.raw_count = 0
        self.planning_count = 0
        self.global_count = 0
        self.closed = None
        self.opened = None
        self.closed_ready_count = 0
        self.open_ready_count = 0
        self.open_detect_raw_count = None
        self.open_detect_planning_count = None
        self.open_detect_global_count = None
        self.last_update_reason = "waiting_for_messages"
        rospy.Subscriber(args.raw_topic, OccupancyGrid, self._raw_callback, queue_size=1)
        rospy.Subscriber(args.planning_topic, OccupancyGrid, self._planning_callback, queue_size=1)
        rospy.Subscriber(args.mask_topic, OccupancyGrid, self._mask_callback, queue_size=1)
        rospy.Subscriber(args.graph_topic, String, self._graph_callback, queue_size=2)
        rospy.Subscriber(args.global_costmap_topic, OccupancyGrid, self._global_callback, queue_size=1)
        rospy.Subscriber(
            args.global_costmap_updates_topic,
            OccupancyGridUpdate,
            self._global_update_callback,
            queue_size=10,
        )

    def _raw_callback(self, message):
        with self.lock:
            self.raw = grid_snapshot(message)
            self.raw_history.append(self.raw)
            self.raw_count += 1

    def _planning_callback(self, message):
        with self.lock:
            self.planning = grid_snapshot(message)
            self.planning_count += 1

    def _mask_callback(self, message):
        with self.lock:
            self.mask = grid_snapshot(message)

    def _global_callback(self, message):
        with self.lock:
            self.global_costmap = grid_snapshot(message)
            self.global_count += 1

    def _global_update_callback(self, message):
        with self.lock:
            if self.global_costmap is None:
                return
            x = int(message.x)
            y = int(message.y)
            width = int(message.width)
            height = int(message.height)
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > self.global_costmap["width"]
                or y + height > self.global_costmap["height"]
                or len(message.data) != width * height
            ):
                return
            update = np.asarray(message.data, dtype=np.int16).reshape(height, width)
            self.global_costmap["data"][y : y + height, x : x + width] = update
            self.global_costmap["stamp_sec"] = float(message.header.stamp.to_sec())
            self.global_count += 1

    def _graph_callback(self, message):
        try:
            graph = json.loads(message.data)
        except Exception:
            return
        with self.lock:
            self.graph = graph

    def _matching_raw(self):
        if self.planning is None:
            return None
        planning_key = grid_key(self.planning)
        candidates = [grid for grid in self.raw_history if grid_key(grid) == planning_key]
        if not candidates:
            return None
        exact = [grid for grid in candidates if same_source_stamp(grid, self.planning)]
        return exact[-1] if exact else candidates[-1]

    def update(self):
        with self.lock:
            node = portal_for_root(self.graph, self.args.target_root)
            if node is None:
                self.last_update_reason = "target_portal_missing"
                self.closed_ready_count = 0
                self.open_ready_count = 0
                return
            if not compatible(self.planning, self.mask) or not same_source_stamp(
                self.planning, self.mask
            ):
                self.last_update_reason = "planning_mask_not_paired"
                self.closed_ready_count = 0
                self.open_ready_count = 0
                return
            matching_raw = self._matching_raw()
            if matching_raw is None:
                self.last_update_reason = "matching_raw_missing"
                self.closed_ready_count = 0
                self.open_ready_count = 0
                return
            state = str((node.get("interaction") or {}).get("state") or "unknown")
            mask_cells = int(np.count_nonzero(self.mask["data"] > 0))
            self.last_update_reason = f"ready_state_{state}_mask_{mask_cells}"
            if self.closed is None and state == "closed" and mask_cells == 0:
                if self.raw_count >= self.args.closed_min_raw_messages:
                    self.closed_ready_count += 1
                if self.closed_ready_count >= self.args.phase_stable_publishes:
                    self.closed = copy_phase(
                        matching_raw,
                        self.planning,
                        self.mask,
                        self.graph,
                        node,
                        self.global_costmap,
                    )
                    save_phase_checkpoint(self.args.output_dir, "closed", self.closed)
                    rospy.loginfo("Captured closed OCC phase after %d raw map messages", self.raw_count)
                return
            if self.closed is None:
                return
            if state == "open" and mask_cells > 0:
                if self.open_detect_raw_count is None:
                    self.open_detect_raw_count = self.raw_count
                    self.open_detect_planning_count = self.planning_count
                    self.open_detect_global_count = self.global_count
                raw_settled = self.raw_count >= (
                    self.open_detect_raw_count + self.args.open_settle_raw_messages
                )
                planning_settled = self.planning_count >= (
                    self.open_detect_planning_count + self.args.phase_stable_publishes
                )
                global_settled = (
                    self.global_costmap is not None
                    and compatible(self.global_costmap, self.planning)
                    and self.global_count
                    >= self.open_detect_global_count + self.args.open_settle_global_messages
                )
                if raw_settled and planning_settled and global_settled:
                    self.open_ready_count += 1
                if self.open_ready_count >= self.args.phase_stable_publishes:
                    self.opened = copy_phase(
                        matching_raw,
                        self.planning,
                        self.mask,
                        self.graph,
                        node,
                        self.global_costmap,
                    )
                    save_phase_checkpoint(self.args.output_dir, "open", self.opened)
                    rospy.loginfo("Captured open OCC phase after %d raw map messages", self.raw_count)

    def diagnostics(self):
        with self.lock:
            portals = []
            if self.graph:
                for node in self.graph.get("nodes") or []:
                    if node.get("type") == "portal":
                        portals.append(
                            {
                                "name": node.get("name"),
                                "source_object_name": (node.get("attributes") or {}).get(
                                    "source_object_name"
                                ),
                                "state": (node.get("interaction") or {}).get("state"),
                            }
                        )
            return {
                "raw_count": self.raw_count,
                "planning_count": self.planning_count,
                "global_count": self.global_count,
                "raw_history_size": len(self.raw_history),
                "has_raw": self.raw is not None,
                "has_planning": self.planning is not None,
                "has_mask": self.mask is not None,
                "raw_key": grid_key(self.raw),
                "planning_key": grid_key(self.planning),
                "mask_key": grid_key(self.mask),
                "raw_stamp_sec": self.raw.get("stamp_sec") if self.raw else None,
                "planning_stamp_sec": self.planning.get("stamp_sec") if self.planning else None,
                "mask_stamp_sec": self.mask.get("stamp_sec") if self.mask else None,
                "mask_cells": (
                    int(np.count_nonzero(self.mask["data"] > 0)) if self.mask is not None else None
                ),
                "closed_captured": self.closed is not None,
                "open_captured": self.opened is not None,
                "last_update_reason": self.last_update_reason,
                "portals": portals,
            }


def odom_xyyaw(message: Odometry) -> list[float]:
    pose = message.pose.pose
    quaternion = pose.orientation
    yaw = math.atan2(
        2.0 * (float(quaternion.w) * float(quaternion.z) + float(quaternion.x) * float(quaternion.y)),
        1.0 - 2.0 * (float(quaternion.y) ** 2 + float(quaternion.z) ** 2),
    )
    return [float(pose.position.x), float(pose.position.y), float(yaw)]


def angle_distance(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(float(first) - float(second)), math.cos(float(first) - float(second))))


def safe_phase_label(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in str(value))
    return normalized.strip("_") or "phase"


def world_to_grid_cell(grid: dict, x: float, y: float) -> tuple[int, int]:
    origin_x, origin_y = grid["origin"]["position"][:2]
    yaw = origin_yaw(grid)
    dx = float(x) - float(origin_x)
    dy = float(y) - float(origin_y)
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return (
        int(math.floor(local_y / grid["resolution"])),
        int(math.floor(local_x / grid["resolution"])),
    )


def pose_marked_crop(image, grid, robot_xyyaw, bounds):
    import cv2

    r0, r1, c0, c1 = bounds
    cropped = np.flipud(image[r0:r1, c0:c1]).copy()
    row, col = world_to_grid_cell(grid, robot_xyyaw[0], robot_xyyaw[1])
    marker_x = int(col - c0)
    marker_y = int(r1 - 1 - row)
    if 0 <= marker_x < cropped.shape[1] and 0 <= marker_y < cropped.shape[0]:
        cv2.circle(cropped, (marker_x, marker_y), 2, (0, 215, 255), -1, cv2.LINE_AA)
        length = 5
        end_x = int(round(marker_x + length * math.cos(float(robot_xyyaw[2]))))
        end_y = int(round(marker_y - length * math.sin(float(robot_xyyaw[2]))))
        cv2.arrowedLine(
            cropped,
            (marker_x, marker_y),
            (end_x, end_y),
            (20, 20, 20),
            1,
            cv2.LINE_AA,
            tipLength=0.35,
        )
    return cropped


def pose_sequence_crop_bounds(phases: list[dict], margin_cells=8):
    grid = phases[0]["planning"]
    focus = np.zeros((grid["height"], grid["width"]), dtype=bool)
    reference_portal = phases[0]["portal"]
    focus |= aabb_mask(
        grid,
        reference_portal["aabb_center"],
        reference_portal["aabb_size"],
        padding=0.35,
    )
    for phase in phases:
        focus |= phase["mask"]["data"] > 0
        row, col = world_to_grid_cell(
            grid,
            phase["transition"]["robot_xyyaw"][0],
            phase["transition"]["robot_xyyaw"][1],
        )
        if 0 <= row < grid["height"] and 0 <= col < grid["width"]:
            focus[
                max(0, row - 2) : min(grid["height"], row + 3),
                max(0, col - 2) : min(grid["width"], col + 3),
            ] = True
    return crop_bounds(focus, margin=margin_cells)


def save_pose_sequence_results(output_dir: Path, phases: list[dict], args) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_mask = np.logical_or.reduce(
        [phase["mask"]["data"] > 0 for phase in phases]
    )
    phase_summaries = []
    for phase in phases:
        transition = phase["transition"]
        phase_index = int(transition["phase_index"])
        label = safe_phase_label(transition["label"])
        prefix = f"phase_{phase_index:02d}_{label}"
        arrays = {
            "raw": phase["raw"]["data"],
            "planning": phase["planning"]["data"],
            "mask": phase["mask"]["data"],
        }
        if phase["global_costmap"] is not None:
            arrays["global_costmap"] = phase["global_costmap"]["data"]
        np.savez_compressed(output_dir / f"{prefix}_maps.npz", **arrays)
        (output_dir / f"{prefix}_graph.json").write_text(
            json.dumps(phase["graph"], ensure_ascii=False, indent=2)
        )

        mask = phase["mask"]["data"] > 0
        raw = phase["raw"]["data"]
        planning = phase["planning"]["data"]
        global_map = phase["global_costmap"]["data"] if phase["global_costmap"] else None
        metrics = {
            "mask_cells": int(np.count_nonzero(mask)),
            "reference_mask_cells": int(np.count_nonzero(reference_mask)),
            "raw_planning_diff_total": int(np.count_nonzero(raw != planning)),
            "raw_planning_diff_in_mask": int(np.count_nonzero((raw != planning) & mask)),
            "planning_nonfree_in_mask": int(np.count_nonzero((planning != 0) & mask)),
            "planning_nonfree_in_reference_mask": int(
                np.count_nonzero((planning != 0) & reference_mask)
            ),
        }
        if global_map is not None and compatible(phase["global_costmap"], phase["planning"]):
            metrics.update(
                {
                    "global_free_in_mask": int(np.count_nonzero((global_map == 0) & mask)),
                    "global_inscribed_in_mask": int(
                        np.count_nonzero((global_map > 0) & (global_map < 100) & mask)
                    ),
                    "global_lethal_in_mask": int(np.count_nonzero((global_map >= 100) & mask)),
                    "global_unknown_in_mask": int(np.count_nonzero((global_map < 0) & mask)),
                    "global_lethal_in_reference_mask": int(
                        np.count_nonzero((global_map >= 100) & reference_mask)
                    ),
                    "global_unknown_in_reference_mask": int(
                        np.count_nonzero((global_map < 0) & reference_mask)
                    ),
                }
            )
        phase_summaries.append(
            {
                "phase_index": phase_index,
                "label": transition["label"],
                "step": transition["step"],
                "state": transition["state"],
                "robot_xyyaw": transition["robot_xyyaw"],
                "observed_xyyaw": phase["odom_xyyaw"],
                "portal_state": (phase["portal"].get("interaction") or {}).get("state"),
                "metrics": metrics,
                "capture_wall_time": phase["capture_wall_time"],
            }
        )

    changes = []
    for previous, current in zip(phases, phases[1:]):
        change = {
            "from_phase": int(previous["transition"]["phase_index"]),
            "to_phase": int(current["transition"]["phase_index"]),
        }
        for key in ("raw", "planning", "global_costmap"):
            if previous.get(key) is None or current.get(key) is None or not compatible(
                previous[key], current[key]
            ):
                continue
            change[f"{key}_changed_cells"] = int(
                np.count_nonzero(previous[key]["data"] != current[key]["data"])
            )
        changes.append(change)

    closed_after_open_checks = []
    seen_open = False
    for phase_summary in phase_summaries:
        if phase_summary["state"] == "open":
            seen_open = True
        elif phase_summary["state"] == "closed" and seen_open:
            metrics = phase_summary["metrics"]
            closed_after_open_checks.append(
                metrics["planning_nonfree_in_reference_mask"] > 0
                and metrics.get("global_lethal_in_reference_mask", 0) > 0
            )

    checks = {
        "captured_all_phases": len(phases) == int(args.expected_phases),
        "portal_states_match_schedule": all(
            phase["transition"]["state"]
            == str((phase["portal"].get("interaction") or {}).get("state"))
            for phase in phases
        ),
        "open_phases_clear_planning_occ": all(
            phase_summary["state"] != "open"
            or (
                phase_summary["metrics"]["mask_cells"] > 0
                and phase_summary["metrics"]["planning_nonfree_in_mask"] == 0
            )
            for phase_summary in phase_summaries
        ),
        "open_phases_clear_global_costmap": all(
            phase_summary["state"] != "open"
            or (
                phase_summary["metrics"].get("global_lethal_in_mask", 1) == 0
                and phase_summary["metrics"].get("global_unknown_in_mask", 1) == 0
            )
            for phase_summary in phase_summaries
        ),
        "closed_phases_disable_clear_mask": all(
            phase_summary["state"] != "closed" or phase_summary["metrics"]["mask_cells"] == 0
            for phase_summary in phase_summaries
        ),
        "closed_after_open_restores_global_costmap": all(closed_after_open_checks),
    }
    summary = {
        "success": all(checks.values()),
        "target_root": args.target_root,
        "checks": checks,
        "phases": phase_summaries,
        "changes": changes,
    }
    (output_dir / "pose_sequence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    r0, r1, c0, c1 = pose_sequence_crop_bounds(phases)
    panel_rows = []
    for phase in phases:
        transition = phase["transition"]
        pose = transition["robot_xyyaw"]
        global_data = (
            phase["global_costmap"]["data"]
            if phase["global_costmap"] is not None
            else np.full_like(phase["planning"]["data"], -1)
        )
        phase_tag = f"P{transition['phase_index']}"
        row_title = f"{phase_tag} {transition['label']} {transition['state']}"
        panels = [
            labeled(
                pose_marked_crop(occupancy_rgb(phase["raw"]["data"]), phase["raw"], pose, (r0, r1, c0, c1)),
                row_title + " | raw",
                fixed_width=360,
            ),
            labeled(
                pose_marked_crop(
                    occupancy_rgb(phase["planning"]["data"]),
                    phase["planning"],
                    pose,
                    (r0, r1, c0, c1),
                ),
                phase_tag + " | semantic OCC",
                fixed_width=360,
            ),
            labeled(
                pose_marked_crop(
                    occupancy_rgb(global_data),
                    phase["planning"],
                    pose,
                    (r0, r1, c0, c1),
                ),
                phase_tag + " | global costmap",
                fixed_width=360,
            ),
            labeled(
                pose_marked_crop(
                    mask_rgb(phase["mask"]["data"]),
                    phase["mask"],
                    pose,
                    (r0, r1, c0, c1),
                ),
                phase_tag + " | clear mask",
                fixed_width=360,
            ),
        ]
        panel_rows.append(np.hstack(panels))
    comparison = np.vstack(panel_rows)
    import cv2

    cv2.imwrite(str(output_dir / "pose_sequence_occ_comparison.png"), comparison)
    return summary


class PoseSequenceRecorder:
    def __init__(self, args):
        self.args = args
        self.base = Recorder(args)
        self.odom = None
        self.active_phase = None
        self.captures = []
        self.reference_open_mask = None
        self.last_reason = "waiting_for_phase_file"
        rospy.Subscriber(args.odom_topic, Odometry, self._odom_callback, queue_size=2)

    def _odom_callback(self, message):
        with self.base.lock:
            self.odom = odom_xyyaw(message)

    def _runtime_phase(self, phase_index):
        try:
            payload = json.loads(self.args.phase_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        phases = {
            int(item["phase_index"]): item
            for item in payload.get("transitions") or []
            if item.get("phase_index") is not None
        }
        return phases.get(int(phase_index))

    def update(self):
        next_index = len(self.captures)
        if next_index >= int(self.args.expected_phases):
            return
        transition = self._runtime_phase(next_index)
        if transition is None:
            self.last_reason = f"waiting_for_runtime_phase_{next_index}"
            return
        if self.active_phase is None:
            with self.base.lock:
                self.active_phase = {
                    "transition": transition,
                    "settle_counts": None,
                }

        with self.base.lock:
            node = portal_for_root(self.base.graph, self.args.target_root)
            if node is None:
                self.last_reason = "target_portal_missing"
                return
            if self.odom is None:
                self.last_reason = "odom_missing"
                return
            if not compatible(self.base.planning, self.base.mask) or not same_source_stamp(
                self.base.planning, self.base.mask
            ):
                self.last_reason = "planning_mask_not_paired"
                return
            matching_raw = self.base._matching_raw()
            if matching_raw is None:
                self.last_reason = "matching_raw_missing"
                return
            if self.base.global_costmap is None or not compatible(
                self.base.global_costmap, self.base.planning
            ):
                self.last_reason = "global_costmap_not_compatible"
                return

            expected_state = str(transition["state"])
            observed_state = str((node.get("interaction") or {}).get("state") or "unknown")
            if observed_state != expected_state:
                self.active_phase["settle_counts"] = None
                self.last_reason = f"portal_state_{observed_state}_expected_{expected_state}"
                return
            mask_cells = int(np.count_nonzero(self.base.mask["data"] > 0))
            if (expected_state == "open" and mask_cells == 0) or (
                expected_state == "closed" and mask_cells != 0
            ):
                self.active_phase["settle_counts"] = None
                self.last_reason = f"mask_{mask_cells}_does_not_match_{expected_state}"
                return

            expected_pose = transition["robot_xyyaw"]
            pose_error = math.hypot(
                float(self.odom[0]) - float(expected_pose[0]),
                float(self.odom[1]) - float(expected_pose[1]),
            )
            yaw_error = angle_distance(self.odom[2], expected_pose[2])
            if pose_error > self.args.pose_tolerance_m or yaw_error > self.args.yaw_tolerance_rad:
                self.active_phase["settle_counts"] = None
                self.last_reason = f"pose_not_settled_position_{pose_error:.3f}_yaw_{yaw_error:.3f}"
                return

            active_mask = self.base.mask["data"] > 0
            global_map = self.base.global_costmap["data"]
            if expected_state == "open":
                global_lethal = int(np.count_nonzero((global_map >= 100) & active_mask))
                global_unknown = int(np.count_nonzero((global_map < 0) & active_mask))
                if global_lethal or global_unknown:
                    self.active_phase["settle_counts"] = None
                    self.last_reason = (
                        f"waiting_for_global_open_clear_lethal_{global_lethal}_unknown_{global_unknown}"
                    )
                    return
            elif self.reference_open_mask is not None:
                planning_nonfree = int(
                    np.count_nonzero((self.base.planning["data"] != 0) & self.reference_open_mask)
                )
                global_lethal = int(
                    np.count_nonzero((global_map >= 100) & self.reference_open_mask)
                )
                if planning_nonfree == 0 or global_lethal == 0:
                    self.active_phase["settle_counts"] = None
                    self.last_reason = (
                        "waiting_for_global_close_restore_"
                        f"planning_nonfree_{planning_nonfree}_global_lethal_{global_lethal}"
                    )
                    return

            if self.active_phase["settle_counts"] is None:
                self.active_phase["settle_counts"] = {
                    "raw_count": self.base.raw_count,
                    "planning_count": self.base.planning_count,
                    "global_count": self.base.global_count,
                }
                self.last_reason = "waiting_for_postcondition_settle"
                return
            settle_counts = self.active_phase["settle_counts"]

            if self.base.raw_count < settle_counts["raw_count"] + self.args.phase_settle_raw_messages:
                self.last_reason = "waiting_for_raw_settle"
                return
            if (
                self.base.planning_count
                < settle_counts["planning_count"] + self.args.phase_settle_planning_messages
            ):
                self.last_reason = "waiting_for_planning_settle"
                return
            if (
                self.base.global_count
                < settle_counts["global_count"] + self.args.phase_settle_global_messages
            ):
                self.last_reason = "waiting_for_global_settle"
                return

            phase = copy_phase(
                matching_raw,
                self.base.planning,
                self.base.mask,
                self.base.graph,
                node,
                self.base.global_costmap,
            )
            phase["transition"] = json.loads(json.dumps(transition))
            phase["odom_xyyaw"] = list(self.odom)
            self.captures.append(phase)
            if expected_state == "open":
                if self.reference_open_mask is None:
                    self.reference_open_mask = active_mask.copy()
                else:
                    self.reference_open_mask |= active_mask
            self.active_phase = None
            self.last_reason = f"captured_phase_{next_index}"
            rospy.loginfo(
                "Captured door OCC pose phase %d/%d: %s",
                next_index + 1,
                int(self.args.expected_phases),
                transition["label"],
            )

    def diagnostics(self):
        diagnostics = self.base.diagnostics()
        diagnostics.update(
            {
                "captured_phase_count": len(self.captures),
                "expected_phases": int(self.args.expected_phases),
                "active_phase": self.active_phase,
                "odom_xyyaw": self.odom,
                "sequence_last_reason": self.last_reason,
            }
        )
        return diagnostics


def run_pose_sequence(args):
    if int(args.expected_phases) <= 0:
        raise SystemExit("--expected-phases must be positive with --phase-file")
    recorder = PoseSequenceRecorder(args)
    started = time.monotonic()
    last_diagnostic_write = 0.0
    while not rospy.is_shutdown() and time.monotonic() - started < args.timeout_s:
        recorder.update()
        if len(recorder.captures) == int(args.expected_phases):
            summary = save_pose_sequence_results(args.output_dir, recorder.captures, args)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            raise SystemExit(0 if summary["success"] else 2)
        now = time.monotonic()
        if now - last_diagnostic_write >= 1.0:
            (args.output_dir / "pose_sequence_live_diagnostics.json").write_text(
                json.dumps(recorder.diagnostics(), ensure_ascii=False, indent=2)
            )
            last_diagnostic_write = now
        time.sleep(0.05)
    diagnostics = recorder.diagnostics()
    (args.output_dir / "pose_sequence_timeout_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2)
    )
    raise SystemExit(f"Timed out waiting for OCC pose sequence: {diagnostics}")


def main():
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase_file is not None:
        args.phase_file = args.phase_file.expanduser().resolve()
    patch_roslogging_findcaller_for_py311()
    rospy.init_node("semantic_door_occ_transition_recorder", anonymous=True)
    if args.phase_file is not None:
        run_pose_sequence(args)
        return
    recorder = Recorder(args)
    started = time.monotonic()
    last_diagnostic_write = 0.0
    while not rospy.is_shutdown() and time.monotonic() - started < args.timeout_s:
        recorder.update()
        if recorder.opened is not None:
            summary = save_results(args.output_dir, recorder.closed, recorder.opened, args)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            raise SystemExit(0 if summary["success"] else 2)
        now = time.monotonic()
        if now - last_diagnostic_write >= 1.0:
            (args.output_dir / "live_diagnostics.json").write_text(
                json.dumps(recorder.diagnostics(), ensure_ascii=False, indent=2)
            )
            last_diagnostic_write = now
        # Use wall time here. rospy.Rate follows /clock and can block forever
        # when the simulator stalls or exits before the recorder times out.
        time.sleep(0.05)
    diagnostics = recorder.diagnostics()
    (args.output_dir / "timeout_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2)
    )
    raise SystemExit(f"Timed out waiting for closed/open OCC phases: {diagnostics}")


if __name__ == "__main__":
    main()
