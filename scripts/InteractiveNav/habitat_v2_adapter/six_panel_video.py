#!/usr/bin/env python3
"""Evaluator-only six-panel MP4 recorder for focused Habitat diagnostics."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


_PANEL_W = 640
_PANEL_H = 360


def _fit(image: np.ndarray, *, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[..., :3]
    height, width = array.shape[:2]
    scale = min(_PANEL_W / max(1, width), _PANEL_H / max(1, height))
    resized = cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=interpolation)
    canvas = np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)
    y = (_PANEL_H - resized.shape[0]) // 2
    x = (_PANEL_W - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _title(panel: np.ndarray, text: str, *, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.rectangle(panel, (0, 0), (_PANEL_W, 30), (18, 18, 18), -1)
    cv2.putText(panel, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _lines(panel: np.ndarray, rows: list[str], x: int = 10, y: int = 50) -> None:
    for row in rows:
        cv2.putText(panel, row, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        y += 20


def _action_text(action: dict[str, Any]) -> str:
    if action.get("action") == "velocity_stop":
        return "STOP"
    args = action.get("action_args") or {}
    return f"v={float(args.get('linear_velocity', -1.0)):+.2f} w={float(args.get('angular_velocity', 0.0)):+.2f}"


class SixPanelVideoRecorder:
    """Hide all rendering/encoding behind one append-frame interface."""

    def __init__(self, *, path: Path, env: Any, episode: Any, fps: float = 10.0, frame_stride: int = 1) -> None:
        from habitat.utils.visualizations import maps

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_stride = max(1, int(frame_stride))
        # OpenCV's broadly available ``mp4v`` writer produces MPEG-4 Part 2,
        # which the Codex app/browser video element cannot reliably decode.
        # Feed raw BGR frames to the bundled ffmpeg instead and require H.264
        # with yuv420p + faststart for browser-compatible MP4 playback.
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self._writer = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s:v", f"{_PANEL_W * 3}x{_PANEL_H * 2}",
                "-r", str(float(fps)),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
        )
        self._pathfinder = env.sim.pathfinder
        self._floor_y = float(episode.start_position[1])
        topdown = maps.get_topdown_map(self._pathfinder, height=self._floor_y, map_resolution=768, draw_border=True)
        self._topdown = maps.colorize_topdown_map(topdown)[..., ::-1].copy()
        self._topdown_shape = topdown.shape[:2]
        self._start_position = np.asarray(episode.start_position, dtype=np.float64)
        rotation = np.asarray(episode.start_rotation, dtype=np.float64).reshape(4)
        self._start_rotation_xyzw = rotation
        self._trajectory_world: list[list[float]] = []
        self._goal_centers: list[list[float]] = []
        self._view_points: list[list[float]] = []
        for goal in episode.goals:
            same_floor = [
                [float(v) for v in view.agent_state.position]
                for view in (goal.view_points or [])
                if abs(float(view.agent_state.position[1]) - self._floor_y) <= 0.75
            ]
            if same_floor:
                self._goal_centers.append([float(v) for v in goal.position])
                self._view_points.extend(same_floor)
        self._frames = 0

    def _world_pixel(self, position: list[float]) -> tuple[int, int]:
        from habitat.utils.visualizations import maps

        row, col = maps.to_grid(
            float(position[2]), float(position[0]), self._topdown_shape, pathfinder=self._pathfinder
        )
        return int(col), int(row)

    def _gps_world(self, xy: np.ndarray) -> list[float]:
        """Map public episode-relative GPS coordinates into HM3D world XYZ."""

        x, y, z, w = (float(value) for value in self._start_rotation_xyzw)

        def rotate(vector: np.ndarray) -> np.ndarray:
            qvec = np.asarray([x, y, z], dtype=np.float64)
            return vector + 2.0 * np.cross(qvec, np.cross(qvec, vector) + w * vector)

        forward = rotate(np.asarray([0.0, 0.0, -1.0], dtype=np.float64))
        right = rotate(np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        world = self._start_position + float(xy[0]) * forward + float(xy[1]) * right
        return [float(value) for value in world]

    def _overlay_public_occupancy(self, canvas: np.ndarray, snapshot: dict[str, Any]) -> None:
        grid = np.asarray(snapshot.get("grid"), dtype=np.uint8)
        if grid.ndim != 2:
            return
        origin = np.asarray(snapshot.get("map_origin_cell_xy"), dtype=np.float64).reshape(2)
        resolution = float(snapshot.get("map_resolution_m", 0.0) or 0.0)
        if resolution <= 0.0:
            return
        # Plot one point per 0.2 m at most; this preserves the observed extent
        # without making the official navmesh unreadable.
        stride = max(1, int(round(0.20 / resolution)))
        for value, color in ((1, (90, 180, 90)), (2, (25, 25, 25))):
            ys, xs = np.nonzero(grid[::stride, ::stride] == value)
            xs = xs * stride
            ys = ys * stride
            for gx, gy in zip(xs.tolist(), ys.tolist()):
                gps = (np.asarray([gx, gy], dtype=np.float64) - origin) * resolution
                px = self._world_pixel(self._gps_world(gps))
                if 0 <= px[0] < canvas.shape[1] and 0 <= px[1] < canvas.shape[0]:
                    if value == 1:
                        canvas[px[1], px[0]] = (
                            0.55 * canvas[px[1], px[0]] + 0.45 * np.asarray(color)
                        ).astype(np.uint8)
                    else:
                        cv2.circle(canvas, px, 1, color, -1)

        active = snapshot.get("active_goal") or {}
        route = active.get("route_cells_xy") or []
        route_pixels = []
        for cell in route:
            gps = (np.asarray(cell[:2], dtype=np.float64) - origin) * resolution
            route_pixels.append(self._world_pixel(self._gps_world(gps)))
        if len(route_pixels) >= 2:
            cv2.polylines(canvas, [np.asarray(route_pixels, dtype=np.int32)], False, (255, 120, 20), 2)

    @staticmethod
    def _detections(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        payload = snapshot.get("detections") or {}
        rows = payload.get("detections") or payload.get("detections_2d") or []
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def _rgb_panel(self, observations: dict[str, Any], snapshot: dict[str, Any], action: dict[str, Any]) -> np.ndarray:
        rgb = np.asarray(observations["rgb"], dtype=np.uint8)[..., :3]
        panel = _fit(rgb[..., ::-1])
        goal = str((snapshot.get("episode") or {}).get("object_category") or "object")
        _title(panel, f"1 RGB | target={goal} | step={snapshot.get('step', 0)}")
        _lines(panel, [_action_text(action)], y=52)
        return panel

    def _depth_panel(self, observations: dict[str, Any], snapshot: dict[str, Any]) -> np.ndarray:
        depth = np.asarray(observations["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        valid = np.isfinite(depth)
        lo, hi = (0.0, 1.0) if valid.any() and float(np.nanmax(depth)) <= 1.01 else (0.5, 5.0)
        normalized = np.clip((np.nan_to_num(depth, nan=hi) - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        colored = cv2.applyColorMap(np.asarray(normalized * 255.0, dtype=np.uint8), cv2.COLORMAP_TURBO)
        panel = _fit(colored)
        _title(panel, f"2 Public depth | clearance={snapshot.get('forward_clearance_m', 0.0):.2f}m")
        return panel

    def _detection_panel(self, observations: dict[str, Any], snapshot: dict[str, Any]) -> np.ndarray:
        rgb = np.asarray(observations["rgb"], dtype=np.uint8)[..., :3]
        height, width = rgb.shape[:2]
        canvas = rgb[..., ::-1].copy()
        detections = self._detections(snapshot)
        goal = str((snapshot.get("episode") or {}).get("object_category") or "").casefold().replace(" ", "_")
        aliases = {
            "couch": {"couch", "sofa"}, "sofa": {"couch", "sofa"},
            "television": {"tv", "television", "tv_monitor"}, "potted_plant": {"plant", "potted_plant"},
        }.get(goal, {goal})
        for row in detections:
            box = row.get("bbox_xyxy_normalized") or row.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            values = [float(v) for v in box]
            if max(values) <= 1.01:
                x1, y1, x2, y2 = int(values[0] * width), int(values[1] * height), int(values[2] * width), int(values[3] * height)
            else:
                x1, y1, x2, y2 = (int(v) for v in values)
            label = str(row.get("semantic_class") or row.get("label") or row.get("semantic_class_raw") or "object")
            raw = label.casefold().replace(" ", "_")
            color = (0, 255, 80) if raw in aliases else (0, 180, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{label} {float(row.get('confidence', 0.0)):.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        panel = _fit(canvas)
        _title(panel, f"3 Original M1 / YOLOE detections | n={len(detections)}")
        return panel

    def _occupancy_panel(self, snapshot: dict[str, Any]) -> np.ndarray:
        grid = np.asarray(snapshot["grid"], dtype=np.uint8)
        canvas = np.full((*grid.shape, 3), 45, dtype=np.uint8)
        canvas[grid == 1] = (220, 220, 220)
        canvas[grid == 2] = (20, 20, 20)
        active = snapshot.get("active_goal")
        if active:
            route = active.get("route_cells_xy") or []
            if route:
                pts = np.asarray(route, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], False, (255, 120, 20), 2)
            xy = np.asarray(active["xy"], dtype=np.float32)
            origin = np.asarray(snapshot["map_origin_cell_xy"], dtype=np.float32)
            resolution = float(snapshot["map_resolution_m"])
            cell = np.rint(xy / resolution + origin).astype(np.int32)
            cv2.circle(canvas, tuple(cell), 6, (255, 0, 255), -1)
        pose = snapshot.get("pose_xy")
        if pose is not None:
            origin = np.asarray(snapshot["map_origin_cell_xy"], dtype=np.float32)
            cell = np.rint(np.asarray(pose) / float(snapshot["map_resolution_m"]) + origin).astype(np.int32)
            cv2.circle(canvas, tuple(cell), 6, (0, 0, 255), -1)
        track = snapshot.get("target_track")
        if track:
            origin = np.asarray(snapshot["map_origin_cell_xy"], dtype=np.float32)
            cell = np.rint(np.asarray(track["surface_xy"]) / float(snapshot["map_resolution_m"]) + origin).astype(np.int32)
            cv2.drawMarker(canvas, tuple(cell), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        panel = _fit(canvas, interpolation=cv2.INTER_NEAREST)
        candidate = str(active.get("candidate_id")) if active else "none"
        _title(panel, f"4 Public RGB-D occupancy / route | {candidate}")
        return panel

    def _semantic_panel(self, snapshot: dict[str, Any]) -> np.ndarray:
        panel = np.full((_PANEL_H, _PANEL_W, 3), 26, dtype=np.uint8)
        graph = snapshot.get("semantic_graph") or {}
        nodes = [row for row in (graph.get("nodes") or []) if isinstance(row, dict)]
        edges = [row for row in (graph.get("edges") or []) if isinstance(row, dict)]
        candidates = (snapshot.get("candidate_payload") or {}).get("candidates") or []
        _title(panel, f"5 Original semantic graph / room context | nodes={len(nodes)} edges={len(edges)}")
        labels = []
        for node in nodes[:10]:
            label = str(node.get("semantic_class") or node.get("label") or node.get("name") or node.get("id") or "node")
            attrs = node.get("attributes") or node.get("room_attributes") or {}
            suffix = f" {attrs}" if attrs else ""
            labels.append(f"- {label}{suffix}"[:92])
        if not labels:
            labels = ["No semantic nodes received yet"]
        nav = [row for row in candidates if isinstance(row, dict) and str(row.get("behavior_type")).upper() == "NAVIGATE"]
        labels.extend(["", f"Candidates={len(candidates)}  NAVIGATE={len(nav)}", f"Last decision: {(snapshot.get('last_decision') or {}).get('candidate_id', 'none')}"])
        _lines(panel, labels[:14], y=52)
        return panel

    def _topdown_panel(self, world_position: list[float], metrics: dict[str, Any], snapshot: dict[str, Any]) -> np.ndarray:
        canvas = self._topdown.copy()
        if not self._trajectory_world or not np.allclose(self._trajectory_world[-1], world_position, atol=1e-6):
            self._trajectory_world.append(list(world_position))
        self._overlay_public_occupancy(canvas, snapshot)
        trajectory = [self._world_pixel(position) for position in self._trajectory_world]
        if len(trajectory) >= 2:
            cv2.polylines(canvas, [np.asarray(trajectory, dtype=np.int32)], False, (214, 166, 0), 3)
        for position in self._view_points:
            cv2.circle(canvas, self._world_pixel(position), 3, (0, 0, 220), 1)
        for position in self._goal_centers:
            cv2.drawMarker(canvas, self._world_pixel(position), (0, 0, 150), cv2.MARKER_TILTED_CROSS, 14, 2)
        robot = self._world_pixel(world_position)
        cv2.circle(canvas, robot, 5, (255, 80, 0), -1)
        pose_xy = snapshot.get("pose_xy")
        heading = snapshot.get("heading")
        if pose_xy is not None and heading is not None:
            pose_xy = np.asarray(pose_xy, dtype=np.float64)
            arrow_gps = pose_xy + 0.55 * np.asarray(
                [np.cos(float(heading)), -np.sin(float(heading))], dtype=np.float64
            )
            cv2.arrowedLine(
                canvas,
                robot,
                self._world_pixel(self._gps_world(arrow_gps)),
                (255, 80, 0),
                3,
                tipLength=0.35,
            )
        panel = _fit(canvas)
        distance = metrics.get("distance_to_goal")
        distance_text = f"{float(distance):.3f}m" if distance is not None else "N/A"
        success = float(metrics.get("success", 0.0)) > 0.0
        _title(panel, f"6 POSTHOC ONLY: GT targets/ViewPoints | d={distance_text} success={success}", color=(80, 220, 255))
        cv2.putText(panel, "NOT PROVIDED TO POLICY / M1 / M2", (10, _PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1, cv2.LINE_AA)
        return panel

    def render_recorder_topdown(
        self,
        *,
        policy: Any,
        metrics: dict[str, Any],
        agent_world_position: list[float],
    ) -> np.ndarray:
        """Return the evaluator-only panel destined only for ROS recorder panel 6."""

        return self._topdown_panel(agent_world_position, metrics, policy.diagnostic_snapshot())

    def append(
        self,
        *,
        observations: dict[str, Any],
        action: dict[str, Any],
        policy: Any,
        metrics: dict[str, Any],
        agent_world_position: list[float],
    ) -> None:
        self._frames += 1
        if (self._frames - 1) % self.frame_stride:
            return
        snapshot = policy.diagnostic_snapshot()
        panels = [
            self._rgb_panel(observations, snapshot, action),
            self._depth_panel(observations, snapshot),
            self._detection_panel(observations, snapshot),
            self._occupancy_panel(snapshot),
            self._semantic_panel(snapshot),
            self._topdown_panel(agent_world_position, metrics, snapshot),
        ]
        frame = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
        if self._writer.stdin is None:
            raise RuntimeError("six-panel H.264 encoder stdin is unavailable")
        self._writer.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if getattr(self, "_writer", None) is not None:
            if self._writer.stdin is not None:
                self._writer.stdin.close()
            return_code = self._writer.wait()
            if return_code != 0:
                raise RuntimeError(f"six-panel H.264 encoder exited with status {return_code}")
            self._writer = None

    def __enter__(self) -> "SixPanelVideoRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
