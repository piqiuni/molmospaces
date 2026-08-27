#!/usr/bin/env python3
"""Local/LAN six-panel viewer and read-only sensor WebSocket gateway."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from physical_protocol import validate_packet
from runtime_state import RuntimeState
from safety_gate import ReadOnlySafetyGate
from qwen_client import QwenClient
try:
    from offline_semantic_renderer import (
        OfflineSixPanelRenderer,
        RawGrid,
        TransformResolver,
        draw_camera_title,
        draw_task_subgoal_header,
        known_world_bounds,
    )
except ModuleNotFoundError:  # imported from physical_nav/tests
    _interactive_nav_dir = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    if _interactive_nav_dir not in sys.path:
        sys.path.insert(0, _interactive_nav_dir)
    from offline_semantic_renderer import (
        OfflineSixPanelRenderer,
        RawGrid,
        TransformResolver,
        draw_camera_title,
        draw_task_subgoal_header,
        known_world_bounds,
    )

try:
    import cv2
    import numpy as np
    # The viewer is a diagnostic stream; OpenCV's default worker pool can
    # create one thread per CPU and starve the HTTP/WebSocket event loops.
    cv2.setNumThreads(1)
except ImportError:  # pragma: no cover - useful for protocol-only testing
    cv2 = None
    np = None


class _ReusableHTTPServer(ThreadingHTTPServer):
    # Allow an immediate service restart after a browser/MJPEG disconnect.
    allow_reuse_address = True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np is None or np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _point_xy(value: Any) -> tuple[float, float] | None:
    """Read either graph point dictionaries or the mapper's XYZ arrays."""
    if isinstance(value, dict):
        return _safe_float(value.get("x")), _safe_float(value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _safe_float(value[0]), _safe_float(value[1])
    return None


def _decode(value: str, encoding: str) -> Any:
    raw = base64.b64decode(value)
    if cv2 is None:
        return raw
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode {encoding}")
    return image


class SixPanelRenderer:
    def __init__(self, state: RuntimeState, width: int = 640, height: int = 360) -> None:
        self.state, self.width, self.height = state, width, height
        # Use the exact six-panel renderer shared by build_raw_overview.  The
        # live physical adapter supplies in-memory RawGrid/step receipts in
        # the same shape as record_explore_debug.py's offline artifacts.
        self.panel_size = (480, 270)
        self._canonical = OfflineSixPanelRenderer(
            transforms=TransformResolver([], map_frame="tf_frame_map", odom_frame="tf_frame_map")
        )

    @staticmethod
    def _raw_grid(payload: Any, default_frame: str = "tf_frame_map") -> RawGrid | None:
        if not isinstance(payload, dict) or not payload.get("data"):
            return None
        try:
            width, height = int(payload.get("width", 0)), int(payload.get("height", 0))
            values = np.asarray(payload.get("data", []), dtype=np.int32)
            if width <= 0 or height <= 0 or values.size < width * height:
                return None
            values = values[: width * height].reshape((height, width))
            origin = payload.get("origin") if isinstance(payload.get("origin"), dict) else {}
            qz, qw = float(origin.get("qz", 0.0) or 0.0), float(origin.get("qw", 1.0) or 1.0)
            origin_yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
            return RawGrid(
                values=values,
                width=width,
                height=height,
                resolution=float(payload.get("resolution", 0.0) or 0.0),
                frame_id=str(payload.get("frame_id") or default_frame).lstrip("/"),
                origin_x=float(origin.get("x", 0.0) or 0.0),
                origin_y=float(origin.get("y", 0.0) or 0.0),
                origin_yaw=origin_yaw,
            )
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _draw_live_detections(panel: Any, detections: list[dict[str, Any]], source_shape: tuple[int, int] | None) -> None:
        """Overlay the live YOLOE boxes on canonical panel 1.

        The shared offline renderer intentionally draws only recorder/GT
        overlays.  Physical detections arrive asynchronously, so they are
        added by this adapter after the canonical camera title is rendered.
        """
        if source_shape is None:
            return
        src_h, src_w = source_shape
        if src_w <= 0 or src_h <= 0:
            return
        sx, sy = panel.shape[1] / float(src_w), panel.shape[0] / float(src_h)
        for det in detections:
            box = det.get("bbox") or det.get("bbox_2d")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in box]
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((max(0.0, min(src_w - 1.0, x1)), max(0.0, min(src_w - 1.0, x2))))
            y1, y2 = sorted((max(0.0, min(src_h - 1.0, y1)), max(0.0, min(src_h - 1.0, y2))))
            confidence = _safe_float(det.get("confidence"), 0.0)
            color = (0, 220, 0) if confidence >= 0.5 else (0, 165, 255)
            p1, p2 = (int(round(x1 * sx)), int(round(y1 * sy))), (int(round(x2 * sx)), int(round(y2 * sy)))
            cv2.rectangle(panel, p1, p2, color, 2, cv2.LINE_AA)
            label = f"{det.get('semantic_class', det.get('class', '?'))} {confidence:.2f}"
            cv2.putText(panel, label[:34], (p1[0], max(48, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    @staticmethod
    def _physical_step(snapshot: dict[str, Any]) -> dict[str, Any]:
        telemetry = snapshot.get("telemetry") if isinstance(snapshot.get("telemetry"), dict) else {}
        position = telemetry.get("map_position") or telemetry.get("position") or [0.0, 0.0, 0.0]
        try:
            pose = [float(position[0]), float(position[1]), float(telemetry.get("map_yaw", telemetry.get("yaw", 0.0)) or 0.0)]
        except (IndexError, TypeError, ValueError):
            pose = [0.0, 0.0, 0.0]
        graph = snapshot.get("graph") if isinstance(snapshot.get("graph"), dict) else {}
        navigation = snapshot.get("navigation") if isinstance(snapshot.get("navigation"), dict) else {}
        current = navigation.get("current_subgoal") if isinstance(navigation.get("current_subgoal"), dict) else {}
        selected = navigation.get("selection") if isinstance(navigation.get("selection"), dict) else {}
        candidates = navigation.get("candidates") if isinstance(navigation.get("candidates"), dict) else {}
        goal = current.get("point") or current.get("position") or current.get("goal") or []
        if not isinstance(goal, (list, tuple)) or len(goal) < 2:
            goal = []
        if goal:
            selected = dict(selected)
            selected.setdefault(
                "goal_xyyaw",
                [
                    float(goal[0]),
                    float(goal[1]),
                    _safe_float(current.get("yaw", current.get("theta", 0.0))),
                ],
            )
            selected.setdefault("active", True)
        observed = []
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            observed.extend(
                str(value)
                for value in (
                    node.get("id"), node.get("name"), node.get("object_id"),
                    (node.get("attributes") or {}).get("instance_id"),
                )
                if value not in {None, ""}
            )
        return {
            "step_index": int(snapshot.get("frame_seq", 0) or 0),
            "stamp_sec": float(snapshot.get("frame_stamp", 0.0) or 0.0),
            "pose": pose,
            "pose_frame_id": "tf_frame_map",
            "active_goal": list(goal[:2]) if goal else [],
            "active_goal_yaw": _safe_float(current.get("yaw", current.get("theta", 0.0))),
            "distance_m": 0.0,
            "trajectory": [pose + [float(snapshot.get("frame_stamp", 0.0) or 0.0)]],
            "global_plan": {}, "local_global_plan": {}, "local_plan": {},
            "unified_graph": graph,
            "observed_instance_ids": sorted(set(observed)),
            "semantic_candidates": candidates,
            "semantic_selection": selected or {"active": False},
            "semantic_execution_state": navigation.get("execution_state") or {},
            "semantic_behavior_feedback": navigation.get("behavior_feedback") or {},
            "semantic_decision_trace": navigation.get("decision_trace") or {},
        }

    def _placeholder(self, title: str) -> Any:
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel[:, :, :] = (32, 32, 32)
        cv2.putText(panel, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        return panel

    def _image_panel(self, image: Any, title: str) -> Any:
        if image is None:
            return self._placeholder(title)
        if image.ndim == 2:
            valid = image > 0
            if np.any(valid):
                norm = np.zeros_like(image, dtype=np.uint8)
                clipped = np.clip(image.astype(np.float32), 0, np.percentile(image[valid], 98))
                norm = (clipped / max(1.0, float(clipped.max())) * 255).astype(np.uint8)
                image = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            else:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        panel = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        return panel

    def _detection_panel(self, rgb: Any, detections: list[dict[str, Any]]) -> Any:
        panel = self._image_panel(rgb, f"3 Original M1 / YOLOE detections | n={len(detections)}")
        if rgb is None:
            return panel
        sx, sy = self.width / rgb.shape[1], self.height / rgb.shape[0]
        for det in detections:
            box = det.get("bbox") or det.get("bbox_2d")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box)]
            color = (0, 255, 0) if float(det.get("confidence", 0)) >= .5 else (0, 165, 255)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
            text = f"{det.get('semantic_class', det.get('class', '?'))} {float(det.get('confidence', 0)):.2f}"
            cv2.putText(panel, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
        return panel

    def _depth_panel(self, depth: Any, depth_scale: float) -> Any:
        if depth is None:
            return self._placeholder("2 Public depth | clearance=N/A")
        valid = np.asarray(depth) > 0
        clearance = 0.0
        if np.any(valid):
            center = np.asarray(depth)[..., max(0, np.asarray(depth).shape[1] // 2 - 8): np.asarray(depth).shape[1] // 2 + 8]
            center_valid = center > 0
            if np.any(center_valid):
                clearance = float(np.median(center[center_valid])) * float(depth_scale)
        return self._image_panel(depth, f"2 Public depth | clearance={clearance:.2f}m")

    def _json_panel(self, title: str, value: Any) -> Any:
        panel = self._placeholder(title)
        lines = json.dumps(value, ensure_ascii=False, indent=2, default=str).splitlines()[:16]
        for i, line in enumerate(lines):
            cv2.putText(panel, line[:78], (12, 62 + i * 19), cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 220, 220), 1)
        return panel

    def _map_panel(self, occupancy: Any, telemetry: dict[str, Any]) -> Any:
        if not isinstance(occupancy, dict) or not occupancy.get("data"):
            return self._json_panel("4 Public RGB-D occupancy / route", telemetry)
        panel = np.full((self.height, self.width, 3), 127, dtype=np.uint8)
        width, height = int(occupancy.get("width", 0)), int(occupancy.get("height", 0)); values = np.asarray(occupancy.get("data", []), dtype=np.int16)
        if width > 0 and height > 0 and values.size >= width * height:
            grid = values[:width * height].reshape(height, width); image = np.full((height, width, 3), 127, dtype=np.uint8)
            image[grid == 0] = (235, 235, 235); image[grid > 50] = (30, 30, 30)
            panel = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel, "4 Public RGB-D occupancy / route | no-control", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        pose = telemetry.get("position", [0, 0, 0]); cv2.putText(panel, f"pose={pose[:2]} yaw={telemetry.get('yaw', '?')}", (12, self.height - 15), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 100, 255), 1)
        return panel

    def _graph_panel(self, graph: dict[str, Any]) -> Any:
        panel = np.full((self.height, self.width, 3), 35, dtype=np.uint8); nodes = graph.get("nodes", []) if isinstance(graph, dict) else []; edges = graph.get("edges", []) if isinstance(graph, dict) else []
        positions = {}
        for index, node in enumerate(nodes):
            value = node.get("world_position") or node.get("position") or node.get("centroid") or node.get("aabb_center") or node.get("center") or {}
            point = _point_xy(value)
            x, y = point if point is not None else (float(index % 5), float(index // 5))
            positions[str(node.get("id", node.get("node_id", index)))] = (x, y)
        if positions:
            xs, ys = [p[0] for p in positions.values()], [p[1] for p in positions.values()]; minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            def point(x: float, y: float) -> tuple[int, int]: return (int(40 + (x - minx) / max(1e-6, maxx - minx) * (self.width - 80)), int(self.height - 40 - (y - miny) / max(1e-6, maxy - miny) * (self.height - 80)))
            for edge in edges:
                a, b = positions.get(str(edge.get("source", edge.get("from", edge.get("src_id", ""))))), positions.get(str(edge.get("target", edge.get("to", edge.get("dst_id", "")))))
                if a and b: cv2.line(panel, point(*a), point(*b), (110, 110, 110), 1)
            for node in nodes:
                key = str(node.get("id", node.get("node_id", nodes.index(node)))); xy = positions.get(key); 
                if not xy: continue
                kind = str(node.get("type", node.get("node_type", "object"))); color = {"room": (255, 160, 30), "portal": (30, 220, 255), "container": (180, 80, 220), "support": (200, 200, 60)}.get(kind, (60, 220, 80)); px = point(*xy); cv2.circle(panel, px, 7, color, -1); cv2.putText(panel, str(node.get("label", key))[:18], (px[0] + 8, px[1]), cv2.FONT_HERSHEY_PLAIN, .9, color, 1)
        cv2.putText(panel, f"5 Original semantic graph / room context | nodes={len(nodes)} edges={len(edges)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 2)
        return panel

    def _topdown_panel(self, occupancy: Any, graph: dict[str, Any], telemetry: dict[str, Any], consistency: dict[str, Any]) -> Any:
        """Physical replacement for the canonical six-panel topdown/GT view.

        The simulator's panel 6 is explicitly posthoc GT.  A real Go2 has no
        GT target stream, so this panel keeps the same slot and layout while
        showing the public map-frame audit: occupancy, robot pose and mapped
        semantic nodes. Detailed consistency metrics remain in the side panel.
        """
        if not isinstance(occupancy, dict) or not occupancy.get("data"):
            return self._json_panel("6 Physical topdown / map audit", consistency)
        width, height = int(occupancy.get("width", 0)), int(occupancy.get("height", 0))
        values = np.asarray(occupancy.get("data", []), dtype=np.int16)
        if width <= 0 or height <= 0 or values.size < width * height:
            return self._json_panel("6 Physical topdown / map audit", consistency)
        grid = values[: width * height].reshape(height, width)
        canvas = np.full((height, width, 3), 127, dtype=np.uint8)
        canvas[grid == 0] = (235, 235, 235)
        canvas[grid > 50] = (25, 25, 25)
        origin = occupancy.get("origin") if isinstance(occupancy.get("origin"), dict) else {}
        resolution = float(occupancy.get("resolution", 0.05) or 0.05)
        ox, oy = float(origin.get("x", 0.0) or 0.0), float(origin.get("y", 0.0) or 0.0)

        def world_pixel(value: Any) -> tuple[int, int] | None:
            point = _point_xy(value)
            if point is None or resolution <= 0:
                return None
            px = int(round((point[0] - ox) / resolution))
            py = int(round(height - 1 - (point[1] - oy) / resolution))
            return (px, py) if 0 <= px < width and 0 <= py < height else None

        robot = world_pixel(telemetry.get("position"))
        if robot is not None:
            cv2.circle(canvas, robot, max(3, min(width, height) // 90), (0, 0, 255), -1)
            yaw = _safe_float(telemetry.get("yaw"))
            tip = (int(robot[0] + 18 * math.cos(yaw)), int(robot[1] - 18 * math.sin(yaw)))
            cv2.arrowedLine(canvas, robot, tip, (255, 80, 0), 2, tipLength=0.3)
        for node in graph.get("nodes", []) if isinstance(graph, dict) else []:
            if not isinstance(node, dict):
                continue
            point = world_pixel(node.get("centroid") or node.get("aabb_center") or node.get("world_position") or node.get("position"))
            if point is None:
                continue
            kind = str(node.get("type", node.get("node_type", "object")))
            color = {"room": (255, 160, 30), "portal": (30, 220, 255), "container": (180, 80, 220)}.get(kind, (60, 190, 80))
            cv2.circle(canvas, point, max(2, min(width, height) // 140), color, -1)
        panel = cv2.resize(canvas, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        status = str(consistency.get("status", "waiting")) if isinstance(consistency, dict) else "waiting"
        counts = consistency.get("counts", {}) if isinstance(consistency, dict) else {}
        cv2.putText(panel, f"6 Physical topdown / map audit | consistency={status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 255, 255), 2)
        cv2.putText(panel, f"pass={counts.get('pass', 0)} warn={counts.get('warn', 0)} fail={counts.get('fail', 0)}", (12, self.height - 15), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 180, 255), 1)
        return panel

    def render(self) -> bytes:
        if cv2 is None:
            raise RuntimeError("physical viewer requires opencv-python and numpy")
        with self.state._lock:
            snapshot = {
                "frame_seq": self.state.frame_seq,
                "frame_stamp": self.state.frame_stamp,
                "telemetry": dict(self.state.telemetry),
                "graph": dict(self.state.graph),
                "consistency": dict(self.state.consistency),
                "navigation": dict(self.state.navigation),
                "detections": [dict(item) for item in self.state.detections if isinstance(item, dict)],
            }
            rgb = None if self.state.rgb is None else self.state.rgb.copy()
            planning = self._raw_grid(self.state.occupancy)
            room = self._raw_grid(self.state.room_grid)
            global_grid = self._raw_grid(self.state.global_costmap)
            local_grid = self._raw_grid(self.state.local_costmap)
        step = self._physical_step(snapshot)
        if planning is None:
            global_grid = global_grid or planning
            local_grid = local_grid or planning
        else:
            global_grid = global_grid or planning
            local_grid = local_grid or planning
        world_bounds = known_world_bounds(planning, margin_m=2.5) if planning is not None else None
        width, height = self.panel_size
        if rgb is None:
            camera = np.full((height, width, 3), 235, dtype=np.uint8)
            cv2.putText(camera, "NO RGB YET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .9, (80, 80, 80), 2, cv2.LINE_AA)
        else:
            camera = cv2.resize(rgb, self.panel_size, interpolation=cv2.INTER_AREA)
        draw_camera_title(camera, step, int(step["step_index"]))
        self._draw_live_detections(camera, snapshot["detections"], None if rgb is None else (rgb.shape[0], rgb.shape[1]))
        occ = self._canonical.render_map_panel(
            planning, self.panel_size, step, int(step["step_index"]),
            title="OCC", kind="occupancy", world_bounds=world_bounds,
            draw_global_plan=False, draw_local_plan=False, draw_frontiers=True,
            draw_semantic_candidates=True, draw_route_plan=False,
        )
        draw_task_subgoal_header(occ, step, box_width_px=width // 2 - 10, background_alpha=.55)
        room_panel = self._canonical.render_room_panel(
            planning, room, self.panel_size, step, int(step["step_index"]), world_bounds,
            view_scale=1.5,
        )
        global_width = width // 2
        global_panel = self._canonical.render_map_panel(
            global_grid, (global_width, height), step, int(step["step_index"]),
            title="GLOBAL COSTMAP", kind="costmap", world_bounds=world_bounds,
            draw_global_plan=False, draw_local_plan=False, draw_frontiers=True,
            draw_semantic_candidates=True,
        )
        local_panel = self._canonical.render_map_panel(
            local_grid, (width - global_width, height), step, int(step["step_index"]),
            title="LOCAL COSTMAP", kind="costmap", draw_global_plan=False,
            draw_local_global_plan=False, draw_local_plan=False, draw_frontiers=True,
            draw_semantic_candidates=True,
        )
        costmaps = np.concatenate([global_panel, local_panel], axis=1)
        spatial = self._canonical.render_semantic_xy(
            planning, self.panel_size, step, int(step["step_index"]), world_bounds,
            view_scale=1.8, label_mode="all", draw_overview_inset=False,
        )
        topology_step = dict(step)
        topology_candidates = dict(step.get("semantic_candidates") or {})
        raw_topology_candidates = list(topology_candidates.get("candidates") or [])
        allowed_topology_families = ("door", "portal", "gate", "fridge", "refrigerator", "drawer", "cabinet", "dresser")
        topology_candidates["candidates"] = [
            candidate
            for candidate in raw_topology_candidates
            if str(candidate.get("behavior_type") or candidate.get("type") or "").upper() != "INTERACT"
            or any(
                marker in " ".join(
                    str(value or "").casefold()
                    for value in (
                        candidate.get("target_name"),
                        (candidate.get("metadata") or {}).get("semantic_name"),
                        (candidate.get("metadata") or {}).get("node_type"),
                    )
                )
                for marker in allowed_topology_families
            )
        ]
        topology_step["semantic_candidates"] = topology_candidates
        # Keep panel 6's interaction layer aligned with the physical policy:
        # generic detector boxes are useful evidence in panels 1/5, but they
        # are not interaction targets.  Preserve rooms/ordinary objects and
        # retain only door/portal, fridge and drawer/cabinet containers in the
        # topology's interaction container layer.
        topology_graph = dict(step.get("unified_graph") or {})
        if isinstance(topology_graph, dict):
            filtered_nodes = []
            for node in list(topology_graph.get("nodes") or []):
                if not isinstance(node, dict):
                    continue
                if str(node.get("type") or node.get("node_type") or "").casefold() != "container":
                    filtered_nodes.append(node)
                    continue
                node_text = " ".join(
                    str(value or "").casefold()
                    for value in (
                        node.get("label"), node.get("name"),
                        (node.get("attributes") or {}).get("semantic_name"),
                        (node.get("attributes") or {}).get("category"),
                    )
                )
                if any(marker in node_text for marker in ("fridge", "refrigerator", "drawer", "cabinet", "dresser")):
                    filtered_nodes.append(node)
            topology_graph["nodes"] = filtered_nodes
            topology_step["unified_graph"] = topology_graph
        topology = self._canonical.render_topology(self.panel_size, topology_step, int(step["step_index"]))
        canvas = np.vstack([
            np.concatenate([camera, occ, room_panel], axis=1),
            np.concatenate([costmaps, spatial, topology], axis=1),
        ])
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("six-panel JPEG encoding failed")
        return bytes(encoded)


class _WebHandler(BaseHTTPRequestHandler):
    state: RuntimeState
    renderer: SixPanelRenderer
    gate: ReadOnlySafetyGate
    qwen_submit = None
    frame_lock = threading.Lock()
    latest_jpeg: bytes = b""

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, value: Any, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False, default=str).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    @staticmethod
    def _state_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return a browser-sized view of state (full masks stay on /api/state)."""
        def detection_view(item: Any) -> dict[str, Any]:
            if not isinstance(item, dict):
                return {"value": item}
            return {
                "semantic_class": item.get("semantic_class", item.get("raw_class", "")),
                "confidence": item.get("confidence"),
                "depth_median_m": item.get("depth_median_m"),
                "position": item.get("world_position", item.get("position")),
                "capture_seq": item.get("capture_seq"),
            }

        graph = snapshot.get("graph")
        graph_summary = graph
        if isinstance(graph, dict):
            graph_summary = {
                "scene_id": graph.get("scene_id"),
                "graph_revision": graph.get("graph_revision"),
                "capture_step": graph.get("capture_step"),
                "node_count": len(graph.get("nodes") or []),
                "edge_count": len(graph.get("edges") or []),
            }
        mllm = [item for item in (snapshot.get("mllm_events") or []) if isinstance(item, dict)]
        stages = {stage: [item for item in mllm if item.get("stage") == stage][-20:] for stage in ("M1", "M2")}
        navigation = snapshot.get("navigation") or {}
        return {
            "frame_seq": snapshot.get("frame_seq"),
            "frame_stamp": snapshot.get("frame_stamp"),
            "generated_at": snapshot.get("generated_at"),
            "read_only": snapshot.get("read_only", True),
            "status": snapshot.get("status"),
            "link": snapshot.get("link"),
            "counters": snapshot.get("counters"),
            "telemetry": snapshot.get("telemetry"),
            "detections": [detection_view(item) for item in (snapshot.get("detections") or [])],
            "mapped_detections": [detection_view(item) for item in (snapshot.get("mapped_detections") or [])],
            "graph": graph_summary,
            "consistency": snapshot.get("consistency"),
            "safety": snapshot.get("safety"),
            "qwen": snapshot.get("qwen"),
            "mllm": stages,
            "m3": {
                "stage": "M3",
                "model_call": False,
                "interaction_result": navigation.get("interaction_result", {}),
                "behavior_feedback": navigation.get("behavior_feedback", {}),
                "execution_state": navigation.get("execution_state", {}),
            },
            "navigation": navigation,
            "last_error": snapshot.get("last_error", ""),
        }

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/state":
            self._json({**self.state.snapshot(), "safety": self.gate.snapshot()}); return
        if path == "/api/state-summary":
            self._json(self._state_summary({**self.state.snapshot(), "safety": self.gate.snapshot()})); return
        if path == "/api/raw-frame":
            self._json(self.state.raw_frame()); return
        if path == "/api/health":
            snapshot = self.state.snapshot(); self._json({"ok": bool(snapshot["frame_seq"] >= 0), "read_only": True, "frame_seq": snapshot["frame_seq"], "generated_at": snapshot["generated_at"]}); return
        if path == "/snapshot.jpg":
            # Short-lived JPEG requests are more reliable than a long-lived
            # multipart stream through some LAN proxies/browser setups.
            with self.frame_lock:
                frame = self.latest_jpeg
            if not frame:
                self._json({"ok": False, "error": "frame not ready"}, 503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(frame))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(frame); return
        if path == "/stream.mjpg":
            # Browser refreshes can leave an old multipart request half-open.
            # A write timeout ensures those abandoned stream threads are
            # reclaimed instead of accumulating until the viewer stalls.
            self.connection.settimeout(2.0)
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            while True:
                with self.frame_lock: frame = self.latest_jpeg
                if frame:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError): return
                # A physical audit view does not need camera-rate streaming;
                # limiting this to 5 Hz reduces network pressure for remote
                # browsers while still showing continuous state changes.
                time.sleep(.2)
        elif path in {"/panel1.mjpg", "/panel5.mjpg"}:
            panel_index = 0 if path.startswith("/panel1") else 4
            self.connection.settimeout(2.0)
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Connection", "close"); self.end_headers()
            while True:
                with self.frame_lock: frame = self.latest_jpeg
                if frame and cv2 is not None:
                    try:
                        image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is not None:
                            x = (panel_index % 3) * 480; y = (panel_index // 3) * 270
                            crop = image[y:y + 270, x:x + 480]
                            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                            if ok:
                                data = bytes(encoded)
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError): return
                time.sleep(.2)
        else:
            html = _HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(html))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(html)

    def do_POST(self) -> None:
        if self.path == "/api/ros-state":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            name, value = str(payload.get("name", "")), payload.get("value")
            if name not in {"detections", "mapped_detections", "graph", "consistency", "occupancy", "room_grid", "global_costmap", "local_costmap", "telemetry", "explore_status", "current_subgoal", "candidates", "selection", "execution_state", "behavior_feedback", "interaction_result", "decision_trace"}:
                self._json({"accepted": False, "error": "unsupported ROS state"}, 400); return
            if name in {"explore_status", "current_subgoal", "candidates", "selection", "execution_state", "behavior_feedback", "interaction_result", "decision_trace"}:
                self.state.navigation[name] = value
            else:
                self.state.update_topic(name, value)
            self._json({"accepted": True}); return
        if self.path == "/api/mllm-event":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            self.state.add_mllm_event(payload); self._json({"accepted": True}); return
        if self.path == "/api/qwen":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            if self.qwen_submit is None: self._json({"error": "Qwen client is disabled"}, 503); return
            self._json(self.qwen_submit(payload), 202); return
        if self.path != "/api/teleop-intent": self._json({"error": "not found"}, 404); return
        length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
        self._json(self.gate.handle_intent(payload), 202)


_HTML = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Go2 Physical Interactive Navigation</title>
<style>
:root{color-scheme:dark;--bg:#0b0d11;--card:#141821;--line:#2e3748;--blue:#67a7ff;--green:#54d68b;--amber:#ffca58;--red:#ff6c67;--muted:#9ba8ba}*{box-sizing:border-box}body{font-family:Inter,"Noto Sans SC",system-ui,sans-serif;background:var(--bg);color:#edf2fa;margin:0;padding:14px}.title{display:flex;align-items:center;gap:12px;margin:0 0 10px;font-size:23px}.readonly{font-size:13px;color:#101418;background:var(--amber);padding:4px 9px;border-radius:99px}.dashboard{display:grid;grid-template-columns:minmax(640px,1fr) 430px;gap:12px;align-items:start}.left{min-width:0}.overview{display:block;width:100%;border:1px solid var(--line);border-radius:8px;background:#111}.ratebar{display:flex;gap:14px;flex-wrap:wrap;color:var(--green);font-size:13px;padding:7px 2px}.mllm-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.right{display:grid;gap:10px}.card{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:10px;min-width:0;box-shadow:0 4px 14px #0004}.card h3{display:flex;align-items:center;justify-content:space-between;margin:0 0 8px;color:var(--blue);font-size:15px}.hint{color:var(--muted);font-size:11px;font-weight:400}.visual{width:100%;display:block;border-radius:6px;border:1px solid #343c49;background:#0d0f13}.events{max-height:390px;overflow:auto;display:grid;gap:8px}.event{display:grid;grid-template-columns:minmax(76px,.8fr) minmax(105px,1.25fr) minmax(90px,1fr);gap:7px;padding:7px;border:1px solid #313947;border-radius:8px;background:#0e1218;font-size:11px}.event-col{min-width:0;overflow:hidden}.label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}.thumb{width:100%;height:70px;object-fit:cover;border-radius:4px;border:1px solid #354053}.chips{display:flex;gap:4px;flex-wrap:wrap}.chip{display:inline-block;max-width:100%;padding:2px 5px;border-radius:5px;background:#263248;color:#cfe1ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.prompt,.result{line-height:1.35;word-break:break-word}.result{color:#d9f7e6}.meta{grid-column:1/-1;color:#7f8da1;font-size:10px}.empty{height:110px;display:grid;place-items:center;color:#768397;border:1px dashed #354052;border-radius:8px}.state-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.metric{padding:9px 7px;border:1px solid #303a4a;border-radius:8px;background:#0e1218;text-align:center}.metric .icon{font-size:19px}.metric .value{font-size:16px;font-weight:700;margin-top:2px}.metric .name{font-size:10px;color:var(--muted)}.statusline{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.badge{padding:3px 7px;border-radius:99px;font-size:11px;background:#263248;color:#cfe1ff}.badge.good{background:#173b2b;color:#7ff0ae}.badge.warn{background:#493b14;color:#ffd46c}.badge.bad{background:#4b2021;color:#ff9692}.m3box{display:grid;gap:8px}.m3head{display:flex;align-items:center;gap:8px}.m3status{font-size:22px;font-weight:800}.m3details{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}.mini{background:#0e1218;border:1px solid #303a4a;border-radius:7px;padding:7px}.mini b{display:block;color:#eaf1fc;font-size:12px}.mini span{font-size:10px;color:var(--muted)}@media(max-width:1150px){.dashboard{grid-template-columns:1fr}.right{grid-template-columns:repeat(3,minmax(0,1fr));grid-row:2}.mllm-grid{grid-row:3}}@media(max-width:800px){body{padding:8px}.dashboard{display:block}.right,.mllm-grid{grid-template-columns:1fr;margin-top:10px}.event{grid-template-columns:1fr 1fr}.meta{grid-column:1/-1}}
</style>
<h1 class='title'>Go2 Physical Interactive Navigation <span class='readonly'>READ ONLY · 动作阻断</span></h1>
<main class='dashboard'><div class='left'><img class='overview' src='/stream.mjpg' alt='实时六面板'><div class='ratebar'><span>● 导航 5 Hz</span><span>● YOLOE / 建图输入 10 Hz</span><span>● 网页 5 Hz</span><span>● Go2 人工遥控</span></div><div class='mllm-grid'><section class='card'><h3>1 · M1 VLM 感知 <span class='hint'>图片 → 简化问题 → 结果</span></h3><div id='m1' class='events'></div></section><section class='card'><h3>2 · M2 LLM 子目标 <span class='hint'>历史 + 候选 + 目标</span></h3><div id='m2' class='events'></div></section><section class='card'><h3>3 · M3 交互评价 <span class='hint'>规则验证，无模型调用</span></h3><div id='m3' class='m3box'></div></section></div></div><aside class='right'><section class='card'><h3>4 · Go2 当前状态 <span id='stamp' class='hint'>连接中</span></h3><div id='go2'></div></section><section class='card'><h3>5 · 图 1 放大 <span class='hint'>RGB + YOLOE box / seg</span></h3><img class='visual' src='/panel1.mjpg' alt='图1放大'></section><section class='card'><h3>6 · 图 5 放大 <span class='hint'>语义地图 + 3D box</span></h3><img class='visual' src='/panel5.mjpg' alt='图5放大'></section></aside></main>
<script>
const q=s=>document.querySelector(s), text=v=>String(v??'').replace(/\\s+/g,' ').trim(), clip=(v,n=150)=>{v=text(v);return v.length>n?v.slice(0,n)+'…':v};
function make(tag,cls,value){const e=document.createElement(tag);if(cls)e.className=cls;if(value!==undefined)e.textContent=value;return e}
function contextCandidates(e){const c=e?.context?.candidates||e?.payload?.candidates||[];return Array.isArray(c)?c:[]}
function outputSummary(e){if(e?.error)return '调用失败：'+clip(e.error,120);let raw=e?.raw_text??e?.response?.raw_text??e?.payload?.result??'';if(typeof raw==='object')raw=JSON.stringify(raw);try{const o=JSON.parse(raw);const ids=o.ranked_ids||o.candidate_id||o.label||o.state||'';return clip([Array.isArray(ids)?ids.join(' → '):ids,o.reason,o.confidence].filter(Boolean).join(' · '),150)}catch(_){return clip(raw||'已完成（无文本结果）',150)}}
function addChips(parent,items,max=5){const wrap=make('div','chips');items.slice(0,max).forEach(v=>wrap.append(make('span','chip',clip(v,28))));if(items.length>max)wrap.append(make('span','chip','+'+(items.length-max)));parent.append(wrap)}
function renderEvents(id,events,stage){const box=q(id);box.replaceChildren();if(!events?.length){box.append(make('div','empty','等待 '+stage+' 调用'));return}events.slice(-8).reverse().forEach((e,index)=>{const row=make('article','event'),left=make('div','event-col'),mid=make('div','event-col'),right=make('div','event-col');left.append(make('div','label',stage==='M1'?'输入图像':'候选 subgoal'));if(stage==='M1'){const im=make('img','thumb');im.src='/panel1.mjpg?event='+index;im.alt='M1图像输入';left.append(im)}else{const cs=contextCandidates(e);addChips(left,cs.map(c=>c.id||c.candidate_id||c.target_name||'candidate'))}mid.append(make('div','label',stage==='M2'?'历史 / 目标 / 简化请求':'简化问题'));if(stage==='M2'){const hist=e?.context?.recent_decisions||e?.payload?.recent_decisions||[];const mission=e?.context?.mission||e?.payload?.mission||{};addChips(mid,['历史 '+(Array.isArray(hist)?hist.length:0),'候选 '+contextCandidates(e).length,'目标 '+(mission.target_name||mission.target||mission.mode||'探索')]);mid.append(make('div','prompt','从当前候选中选择下一语义子目标'))}else mid.append(make('div','prompt',clip(e.instruction||'判断当前交互对象属性与状态')));right.append(make('div','label','MLLM 输出'));right.append(make('div','result',outputSummary(e)));const meta=make('div','meta',`${e.timestamp?new Date(e.timestamp*1000).toLocaleTimeString():'--:--:--'} · ${e.latency_s!=null?Number(e.latency_s).toFixed(2)+' s':'等待耗时'} · ${e.model||e.role||''}`);row.append(left,mid,right,meta);box.append(row)})}
function firstObj(...values){return values.find(v=>v&&typeof v==='object')||{}}
function renderM3(m3){const box=q('#m3');box.replaceChildren();const fb=firstObj(m3?.behavior_feedback,m3?.interaction_result),detail=firstObj(fb.detail,fb.result,fb);let status=text(fb.status||detail.status||'WAITING').toUpperCase(),success=fb.success??detail.success;let kind=success===true||/PASS|SUCCESS|COMPLETE/.test(status)?'good':success===false||/FAIL|ERROR|BLOCK/.test(status)?'bad':'warn';const head=make('div','m3head');head.append(make('div','m3status',status),make('span','badge '+kind,success===true?'验证通过':success===false?'验证未通过':'等待结果'));box.append(head);const grid=make('div','m3details');[['评价对象',fb.target_name||fb.target_id||fb.candidate_id||detail.target||'暂无'],['结果原因',detail.reason||fb.reason||detail.failure_stage||'等待交互反馈'],['状态变化',detail.pre_state&&detail.post_state?detail.pre_state+' → '+detail.post_state:'未产生'],['验证方式','状态/图更新一致性']].forEach(([n,v])=>{const d=make('div','mini');d.append(make('span','',n),make('b','',clip(v,60)));grid.append(d)});box.append(grid)}
function num(v,d=1){const n=Number(v);return Number.isFinite(n)?n.toFixed(d):'--'}
function speed(t){const v=t?.velocity||t?.linear_velocity||[];if(Array.isArray(v))return Math.hypot(...v.slice(0,3).map(Number));if(v&&typeof v==='object')return Math.hypot(Number(v.x||0),Number(v.y||0),Number(v.z||0));return Number(t?.speed||0)}
function renderGo2(s){const t=s.telemetry||{},b=t.battery||{},link=s.link||{},grid=make('div','state-grid');const metrics=[['🔋',num(b.soc??t.battery_soc,0)+' %','电量'],['↗',num(speed(t),2)+' m/s','速度'],['⟳',num((Number(t.yaw||0)*180/Math.PI),1)+'°','航向'],['◉',text(t.mode||'站立'),'动作模式'],['↕',num(t.body_height,2)+' m','机身高度'],['⚠',String(t.error_code??0),'错误码']];metrics.forEach(([i,v,n])=>{const d=make('div','metric');d.append(make('div','icon',i),make('div','value',v),make('div','name',n));grid.append(d)});const line=make('div','statusline');line.append(make('span','badge '+(link.connected===false?'bad':'good'),link.connected===false?'相机断开':'相机在线'),make('span','badge good','控制输出阻断'),make('span','badge','图节点 '+(s.graph?.node_count??0)),make('span','badge','候选 '+((s.navigation?.candidates?.candidates||[]).length)));const box=q('#go2');box.replaceChildren(grid,line);q('#stamp').textContent='帧 '+(s.frame_seq??'--')+' · '+new Date().toLocaleTimeString()}
async function refresh(){try{const r=await fetch('/api/state-summary?ts='+Date.now(),{cache:'no-store'}),s=await r.json();renderEvents('#m1',s.mllm?.M1,'M1');renderEvents('#m2',s.mllm?.M2,'M2');renderM3(s.m3||{});renderGo2(s)}catch(e){q('#stamp').textContent='刷新失败';q('#go2').replaceChildren(make('div','empty',clip(e,100)))}}
setInterval(refresh,1000);refresh();
</script></html>"""


def _compact_qwen_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep sparse masks out of the text prompt sent to the remote Qwen."""
    compact_detections: list[dict[str, Any]] = []
    detections = [item for item in snapshot.get("detections", []) if isinstance(item, dict)]
    detections.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    for detection in detections[:12]:
        item = {
            key: detection.get(key)
            for key in (
                "semantic_class", "confidence", "bbox", "mask_area", "depth_median_m",
                "world_position", "aabb_center", "aabb_size", "map_transform_status",
            )
            if key in detection
        }
        mask = detection.get("mask")
        if isinstance(mask, dict):
            item["mask_summary"] = {
                "area": mask.get("area", mask.get("mask_area", 0)),
                "rows": len(mask.get("rows", [])),
                "cols": len(mask.get("cols", [])),
            }
        compact_detections.append(item)

    graph = snapshot.get("graph")
    compact_graph: dict[str, Any] = {}
    if isinstance(graph, dict):
        compact_graph = {
            key: graph.get(key)
            for key in ("scene_id", "episode_id", "source_mode", "graph_revision", "timestamp", "module1_mode")
            if key in graph
        }
        compact_graph["nodes"] = []
        nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
        nodes.sort(key=lambda item: (bool(item.get("is_currently_visible", True)), float(item.get("last_seen") or 0.0)), reverse=True)
        for node in nodes[:16]:
            item = {
                key: node.get(key)
                for key in ("id", "type", "label", "centroid", "aabb_center", "aabb_size", "room_id")
                if key in node
            }
            interaction = node.get("interaction")
            if isinstance(interaction, dict):
                item["interaction"] = {
                    key: interaction.get(key)
                    for key in ("is_interactable", "interaction_mode", "capability", "state", "cost", "requires_interaction", "traversable")
                    if key in interaction
                }
            compact_graph["nodes"].append(item)
        compact_graph["edges"] = graph.get("edges", [])

    return {
        "frame_seq": snapshot.get("frame_seq", -1),
        "telemetry": _compact_telemetry(snapshot.get("telemetry", {})),
        "detections": compact_detections,
        "graph": compact_graph,
        "consistency": _compact_consistency(snapshot.get("consistency", {})),
    }


def _compact_consistency(consistency: Any) -> dict[str, Any]:
    if not isinstance(consistency, dict):
        return {}
    result = {
        key: consistency.get(key)
        for key in ("status", "counts", "detection_count", "graph_revision", "projection")
        if key in consistency
    }
    result["detections"] = []
    for item in consistency.get("detections", [])[:12]:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        result["detections"].append({
            "object_id": item.get("object_id", ""),
            "status": item.get("status", ""),
            "metrics": {
                key: metrics.get(key)
                for key in ("bbox_iou", "bbox_center_px", "map_projected_depth_abs_m", "map_distance_m", "map_z_abs_m", "rgbd_depth_lift_abs_m")
                if key in metrics
            },
            "reasons": item.get("reasons", []),
        })
    return result


def _compact_telemetry(telemetry: Any) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        return {}
    result = {
        key: telemetry.get(key)
        for key in ("position", "velocity", "yaw", "yaw_speed", "mode", "error_code", "body_height")
        if key in telemetry
    }
    battery = telemetry.get("battery")
    if isinstance(battery, dict):
        result["battery"] = {key: battery.get(key) for key in ("soc", "voltage", "current", "power") if key in battery}
    return result


class PhysicalGateway:
    def __init__(self, host: str, port: int, qwen_url: str = "", qwen_model: str = "qwen3.6-35b-a3b-fp8", camera_parent: str = "tf_frame_base_link", camera_x: float = 0.03, camera_y: float = 0.0, camera_z: float = 0.75, camera_roll: float = 0.0, camera_pitch: float = 0.0, camera_yaw: float = 0.0, qwen_auto_interval: float = 0.0) -> None:
        self.host, self.port = host, port
        self.state, self.gate = RuntimeState(), ReadOnlySafetyGate()
        self.renderer = SixPanelRenderer(self.state)
        self.qwen = QwenClient(qwen_url, qwen_model) if qwen_url else None
        self.camera_parent, self.camera_translation, self.camera_rpy = camera_parent, (camera_x, camera_y, camera_z), (camera_roll, camera_pitch, camera_yaw)
        self.state.set_calibration(
            parent_frame=camera_parent,
            translation_m=[camera_x, camera_y, camera_z],
            rpy_rad=[camera_roll, camera_pitch, camera_yaw],
            source="PHYSICAL_NAV_CAMERA_X/Y/Z/ROLL/PITCH/YAW",
            calibrated=any(abs(value) > 1e-12 for value in (camera_x, camera_y, camera_z, camera_roll, camera_pitch, camera_yaw)),
        )
        self.qwen_auto_interval = qwen_auto_interval
        self.http: ThreadingHTTPServer | None = None

    def receive(self, packet: dict[str, Any]) -> dict[str, Any]:
        validate_packet(packet)
        self.state.link_packet(str(packet.get("type", "")))
        if packet.get("type") == "hello":
            self.state.link_connected(packet)
            return {"type": "ack", "v": 1, "accepted": True, "read_only": True, "capabilities": ["rgb", "depth", "camera_info", "pose", "telemetry"]}
        if packet.get("type") in {"control", "cmd", "lidar", "posture", "speak", "teleop_intent"}:
            # Defensive boundary: even if an old policy client connects to this
            # port, no command is forwarded to Go2 or any local actuator.
            return self.gate.handle_intent(packet)
        if packet["type"] == "sensor_frame":
            rgb = _decode(packet["rgb"]["data"], "jpeg"); depth = _decode(packet["depth"]["data"], "png16")
            self.state.update_frame(rgb=rgb, depth=depth, rgb_b64=packet["rgb"]["data"], depth_b64=packet["depth"]["data"], depth_scale=float(packet.get("depth_scale", .001)), intrinsics=packet.get("intrinsics", {}), camera_frame=packet.get("camera_frame", ""), frame_seq=int(packet["seq"]), frame_stamp=float(packet["stamp"]), sync_ms=packet.get("color_depth_sync_ms"))
            # ROS publication is handled by physical_ros_gateway.py using the
            # /api/raw-frame endpoint. Keeping this process ROS-free avoids a
            # Python 3.13/ROS Noetic runtime conflict.
        elif packet["type"] == "telemetry":
            self.state.update_topic("telemetry", packet.get("telemetry", {}))
        return {"type": "ack", "v": 1, "seq": packet.get("seq", -1), "accepted": True, "read_only": True}

    def start_http(self) -> None:
        handler = type("PhysicalWebHandler", (_WebHandler,), {})
        handler.state, handler.renderer, handler.gate = self.state, self.renderer, self.gate
        def submit_qwen(payload: dict[str, Any]) -> dict[str, Any]:
            if self.qwen is None: return {"accepted": False, "error": "Qwen client is disabled"}
            user_prompt = str(payload.get("prompt", "请分析当前全局语义图与感知一致性"))
            snapshot = self.state.snapshot()
            context = _compact_qwen_context(snapshot)
            prompt = user_prompt + "\n\n当前实物平台状态(JSON)：\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            request = {"prompt": user_prompt, "context": context, "requested_at": time.time(), "model": self.qwen.model}
            self.state.add_qwen(request)
            def worker() -> None:
                result = self.qwen.chat(prompt, max_tokens=int(payload.get("max_tokens", 256)))
                self.state.add_qwen({}, {"completed_at": time.time(), "prompt": user_prompt, "result": result})
            threading.Thread(target=worker, daemon=True).start()
            return {"accepted": True, "queued_at": request["requested_at"]}
        # Store as a static callback; otherwise BaseHTTPRequestHandler binds
        # this closure as an instance method and adds an unwanted ``self``.
        handler.qwen_submit = staticmethod(submit_qwen)
        self.http = _ReusableHTTPServer((self.host, self.port), handler)
        def render_loop() -> None:
            while True:
                try:
                    frame = self.renderer.render()
                    with handler.frame_lock: handler.latest_jpeg = frame
                except Exception as exc:
                    self.state.last_error = str(exc)
                time.sleep(.2)
        threading.Thread(target=render_loop, daemon=True).start()
        if self.qwen is not None and self.qwen_auto_interval > 0:
            def qwen_loop() -> None:
                while True:
                    time.sleep(self.qwen_auto_interval)
                    snapshot = self.state.snapshot()
                    compact = _compact_qwen_context(snapshot)
                    prompt = "请根据当前全局语义图、检测结果和一致性诊断，简要报告空间关系异常和需要人工确认的对象。\n" + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
                    request = {"prompt": prompt, "requested_at": time.time(), "model": self.qwen.model, "source": "auto_graph_review"}
                    self.state.add_qwen(request)
                    result = self.qwen.chat(prompt, max_tokens=256)
                    self.state.add_qwen({}, {"completed_at": time.time(), "source": "auto_graph_review", "result": result})
            threading.Thread(target=qwen_loop, daemon=True).start()
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        print(f"physical six-panel web: http://{self.host}:{self.port}/", flush=True)


async def run_gateway(args: argparse.Namespace) -> None:
    import websockets
    gateway = PhysicalGateway(args.http_host, args.http_port, args.qwen_url, args.qwen_model, args.camera_parent, args.camera_x, args.camera_y, args.camera_z, args.camera_roll, args.camera_pitch, args.camera_yaw, args.qwen_auto_interval)
    gateway.start_http()
    async def handler(websocket: Any) -> None:
        try:
            async for raw in websocket:
                try:
                    packet = json.loads(raw); reply = gateway.receive(packet)
                except Exception as exc:
                    reply = {"type": "error", "accepted": False, "error": str(exc)}
                await websocket.send(json.dumps(reply, separators=(",", ":")))
        except websockets.exceptions.ConnectionClosed:
            # A browser/Go2 reconnect or a normal process shutdown can close
            # without a WebSocket close frame; it is not a sensor error.
            return
        finally:
            gateway.state.link_disconnected()
    async with websockets.serve(handler, args.ws_host, args.ws_port, max_size=args.max_message_mb * 1024 * 1024):
        print(f"physical sensor WebSocket: ws://{args.ws_host}:{args.ws_port}", flush=True)
        await __import__("asyncio").Future()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ws-host", default="0.0.0.0"); p.add_argument("--ws-port", type=int, default=12334)
    p.add_argument("--http-host", default="0.0.0.0"); p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--max-message-mb", type=int, default=16)
    p.add_argument("--qwen-url", default="", help="e.g. http://127.0.0.1:18080/v1 after SSH forwarding")
    p.add_argument("--qwen-model", default="qwen3.6-35b-a3b-fp8")
    p.add_argument("--qwen-auto-interval", type=float, default=0.0, help="seconds; 0 disables periodic graph review")
    p.add_argument("--camera-parent", default="tf_frame_base_link")
    p.add_argument("--camera-x", type=float, default=0.03); p.add_argument("--camera-y", type=float, default=0.0); p.add_argument("--camera-z", type=float, default=0.75); p.add_argument("--camera-roll", type=float, default=0.0); p.add_argument("--camera-pitch", type=float, default=0.0); p.add_argument("--camera-yaw", type=float, default=0.0)
    args = p.parse_args()
    try:
        import asyncio; asyncio.run(run_gateway(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__": main()
