#!/usr/bin/env python3
"""Thread-safe state shared by the physical gateway and web renderer."""

from __future__ import annotations

import copy
import base64
import threading
import time
from typing import Any


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.rgb = None
        self.depth = None
        self.rgb_b64 = ""
        self.depth_b64 = ""
        self.depth_scale = 0.001
        self.intrinsics: dict[str, Any] = {}
        self.rgb_intrinsics: dict[str, Any] = {}
        self.depth_intrinsics: dict[str, Any] = {}
        self.depth_to_color_extrinsics: dict[str, Any] = {}
        self.camera_frame = ""
        self.depth_frame = ""
        self.camera_imu: dict[str, Any] = {}
        self.calibration: dict[str, Any] = {}
        self.frame_seq = -1
        # Local 5 Hz navigation/viewer step. Unlike the Go2 sensor sequence,
        # this starts from zero whenever the policy-side navigation restarts.
        self.navigation_step = -1
        self.frame_stamp = 0.0
        self.sync_ms = None
        self.telemetry: dict[str, Any] = {}
        self.detections: list[dict[str, Any]] = []
        self.detection_meta: dict[str, Any] = {}
        self.mapped_detections: list[dict[str, Any]] = []
        self.mapped_detection_meta: dict[str, Any] = {}
        self.graph: dict[str, Any] = {}
        self.occupancy = None
        # Additional raw grids mirror the recorder's map stages.  They remain
        # in-memory for the live renderer; /api/state exposes metadata only.
        self.room_grid = None
        self.global_costmap = None
        self.local_costmap = None
        self.consistency: dict[str, Any] = {}
        self.qwen: dict[str, Any] = {"requests": [], "results": []}
        # Native M1/M2 model traces and M3 interaction-result evaluations are
        # retained as a small bounded history for the live dashboard.
        self.mllm_events: list[dict[str, Any]] = []
        self.m1_input_images: dict[str, bytes] = {}
        self.navigation: dict[str, Any] = {
            "explore_status": {}, "current_subgoal": {}, "candidates": {},
            "selection": {}, "execution_state": {}, "behavior_feedback": {},
            "interaction_result": {}, "decision_trace": {},
        }
        self.link: dict[str, Any] = {
            "connected": False,
            "connections": 0,
            "last_hello": {},
            "last_packet_at": 0.0,
            "last_disconnect_at": 0.0,
        }
        self.counters = {"frames": 0, "detections": 0, "graph_updates": 0, "dropped": 0}
        # Monotonic revisions let renderers reuse expensive map/graph products
        # when only the camera frame changed.  Keeping these counters here is
        # cheaper than hashing megabyte-sized OccupancyGrid payloads on every
        # six-panel tick.
        self.revision = 0
        self.map_revision = 0
        self.graph_revision = 0
        self.navigation_revision = 0
        self.last_error = ""

    def update_frame(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.counters["frames"] += 1
            self.revision += 1

    def advance_navigation_step(self) -> int:
        with self._lock:
            self.navigation_step += 1
            return self.navigation_step

    def set_calibration(self, **value: Any) -> None:
        with self._lock:
            self.calibration = copy.deepcopy(value)

    def update_topic(self, name: str, value: Any) -> None:
        if name == "mllm_events":
            if isinstance(value, list):
                for event in value:
                    self.add_mllm_event(event)
            else:
                self.add_mllm_event(value)
            return
        with self._lock:
            if name in {"detections", "mapped_detections"} and isinstance(value, dict):
                meta_attr = "detection_meta" if name == "detections" else "mapped_detection_meta"
                setattr(self, meta_attr, {k: v for k, v in value.items() if k not in {"detections", "objects"}})
                value = value.get("detections", value.get("objects", []))
            setattr(self, name, value)
            self.revision += 1
            if name in {"occupancy", "room_grid", "global_costmap", "local_costmap"}:
                self.map_revision += 1
            elif name in {"graph", "mapped_detections", "consistency"}:
                self.graph_revision += 1
            elif name == "navigation":
                self.navigation_revision += 1
            if name == "detections":
                self.counters["detections"] = len(value) if isinstance(value, list) else 0
            elif name == "graph":
                self.counters["graph_updates"] += 1

    def link_connected(self, hello: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.link["connections"] = int(self.link.get("connections", 0)) + 1
            self.link["connected"] = True
            if hello:
                self.link["last_hello"] = copy.deepcopy(hello)
            self.link["last_packet_at"] = time.time()

    def link_packet(self, packet_type: str) -> None:
        with self._lock:
            self.link["last_packet_at"] = time.time()
            self.link["last_packet_type"] = str(packet_type)

    def link_disconnected(self) -> None:
        with self._lock:
            self.link["connections"] = max(0, int(self.link.get("connections", 1)) - 1)
            self.link["connected"] = bool(self.link["connections"])
            self.link["last_disconnect_at"] = time.time()

    def add_qwen(self, request: dict[str, Any], result: dict[str, Any] | None = None) -> None:
        with self._lock:
            if request:
                self.qwen.setdefault("requests", []).append(request)
                self.qwen["requests"] = self.qwen["requests"][-100:]
            if result is not None:
                self.qwen.setdefault("results", []).append(result)
                self.qwen["results"] = self.qwen["results"][-100:]

    def add_mllm_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            if not isinstance(event, dict):
                return
            stored = copy.deepcopy(event)
            data_url = str(stored.pop("m1_input_image_data_url", "") or "")
            if data_url.startswith("data:image/") and "," in data_url:
                key = "{}:{}:{}".format(
                    stored.get("episode_id", "default"),
                    stored.get("request_sequence", len(self.mllm_events)),
                    int(float(stored.get("timestamp", time.time())) * 1000),
                )
                try:
                    self.m1_input_images[key] = base64.b64decode(
                        data_url.split(",", 1)[1], validate=True
                    )
                    stored["m1_input_image_key"] = key
                except (ValueError, TypeError):
                    pass
            self.mllm_events.append(stored)
            self.mllm_events = self.mllm_events[-100:]
            active_keys = {
                str(item.get("m1_input_image_key"))
                for item in self.mllm_events
                if item.get("m1_input_image_key")
            }
            self.m1_input_images = {
                key: value
                for key, value in self.m1_input_images.items()
                if key in active_keys
            }

    def m1_input_image(self, key: str) -> bytes | None:
        with self._lock:
            value = self.m1_input_images.get(str(key))
            return bytes(value) if value is not None else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            graph = copy.deepcopy(self.graph)
            return {
                "frame_seq": self.frame_seq,
                "navigation_step": self.navigation_step,
                "frame_stamp": self.frame_stamp,
                "camera_frame": self.camera_frame,
                "depth_frame": self.depth_frame or self.camera_frame,
                "calibration": copy.deepcopy(self.calibration),
                "sync_ms": self.sync_ms,
                "depth_scale": self.depth_scale,
                "intrinsics": copy.deepcopy(self.intrinsics),
                "rgb_intrinsics": copy.deepcopy(self.rgb_intrinsics or self.intrinsics),
                "depth_intrinsics": copy.deepcopy(self.depth_intrinsics or self.intrinsics),
                "depth_to_color_extrinsics": copy.deepcopy(self.depth_to_color_extrinsics),
                "camera_imu": copy.deepcopy(self.camera_imu),
                "raw_frame_available": bool(self.rgb_b64 and self.depth_b64),
                "telemetry": copy.deepcopy(self.telemetry),
                "detections": copy.deepcopy(self.detections),
                "detection_meta": copy.deepcopy(self.detection_meta),
                "mapped_detections": copy.deepcopy(self.mapped_detections),
                "mapped_detection_meta": copy.deepcopy(self.mapped_detection_meta),
                "graph": graph,
                "occupancy": ({k: self.occupancy.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.occupancy, dict) else None),
                "room_grid": ({k: self.room_grid.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.room_grid, dict) else None),
                "global_costmap": ({k: self.global_costmap.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.global_costmap, dict) else None),
                "local_costmap": ({k: self.local_costmap.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.local_costmap, dict) else None),
                "consistency": copy.deepcopy(self.consistency),
                "qwen": copy.deepcopy(self.qwen),
                "mllm_events": copy.deepcopy(self.mllm_events),
                "navigation": copy.deepcopy(self.navigation),
                "link": copy.deepcopy(self.link),
                "counters": dict(self.counters),
                "last_error": self.last_error,
                "revision": self.revision,
                "map_revision": self.map_revision,
                "graph_revision": self.graph_revision,
                "navigation_revision": self.navigation_revision,
                "read_only": True,
                "status": "READ_ONLY_BLOCKED",
                "generated_at": time.time(),
            }

    def summary_snapshot(self) -> dict[str, Any]:
        """Return only fields consumed by the live dashboard.

        The full snapshot deep-copies the accumulated semantic graph and
        MLLM trace.  Doing that once per browser poll caused visible pauses as
        the graph grew.  The summary endpoint only needs graph counts and
        compact recent events, so avoid copying map payloads and graph nodes.
        """
        with self._lock:
            graph = self.graph if isinstance(self.graph, dict) else {}
            graph_summary = {
                "scene_id": graph.get("scene_id"),
                "graph_revision": graph.get("graph_revision"),
                "capture_step": graph.get("capture_step"),
                "node_count": len(graph.get("nodes") or []),
                "edge_count": len(graph.get("edges") or []),
            }
            return {
                "frame_seq": self.frame_seq,
                "navigation_step": self.navigation_step,
                "frame_stamp": self.frame_stamp,
                "telemetry": copy.deepcopy(self.telemetry),
                "detections": copy.deepcopy(self.detections),
                "detection_meta": dict(self.detection_meta),
                "mapped_detections": copy.deepcopy(self.mapped_detections),
                "graph": graph_summary,
                "consistency": copy.deepcopy(self.consistency),
                "mllm_events": copy.deepcopy(self.mllm_events[-40:]),
                "navigation": copy.deepcopy(self.navigation),
                "link": dict(self.link),
                "counters": dict(self.counters),
                "last_error": self.last_error,
                "read_only": True,
                "status": "READ_ONLY_BLOCKED",
                "generated_at": time.time(),
            }

    def health_snapshot(self) -> dict[str, Any]:
        """Return a constant-size liveness view without copying maps/traces."""
        with self._lock:
            perception_seq = self.detection_meta.get("seq", -1)
            perception_stamp = self.detection_meta.get("stamp", 0.0)
            return {
                "ok": bool(self.frame_seq >= 0),
                "read_only": True,
                "frame_seq": self.frame_seq,
                "frame_stamp": self.frame_stamp,
                "perception_seq": perception_seq,
                "perception_stamp": perception_stamp,
                "navigation_step": self.navigation_step,
                "link_connected": bool(self.link.get("connected")),
                "last_packet_at": self.link.get("last_packet_at", 0.0),
                "generated_at": time.time(),
                "last_error": self.last_error,
            }

    def visualization_snapshot(self) -> dict[str, Any]:
        """Return raw map/graph receipts for presentation-only redrawers.

        Keep this endpoint limited to data consumed by the presentation
        redrawers.  Room/global/local costmaps duplicate OCC at hundreds of
        kilobytes per poll and are available through their dedicated ROS/UI
        paths; including them here made the browser download megabyte-sized
        payloads and serialized the runtime lock for too long.
        """
        with self._lock:
            graph = copy.deepcopy(self.graph)
            return {
                "frame_seq": self.frame_seq,
                "navigation_step": self.navigation_step,
                "telemetry": copy.deepcopy(self.telemetry),
                "occupancy": copy.deepcopy(self.occupancy),
                "graph": graph,
                "mapped_detections": copy.deepcopy(self.mapped_detections),
                "navigation": copy.deepcopy(self.navigation),
            }

    def raw_frame(self) -> dict[str, Any]:
        """Return the latest encoded frame for the system-Python ROS bridge."""
        with self._lock:
            return {
                "seq": self.frame_seq,
                "stamp": self.frame_stamp,
                "rgb": self.rgb_b64,
                "depth": self.depth_b64,
                "width": int(self.intrinsics.get("width", self.rgb.shape[1] if self.rgb is not None else 0)),
                "height": int(self.intrinsics.get("height", self.rgb.shape[0] if self.rgb is not None else 0)),
                "camera_frame": self.camera_frame,
                "depth_frame": self.depth_frame or self.camera_frame,
                "depth_scale": self.depth_scale,
                "intrinsics": copy.deepcopy(self.intrinsics),
                "rgb_intrinsics": copy.deepcopy(self.rgb_intrinsics or self.intrinsics),
                "depth_intrinsics": copy.deepcopy(self.depth_intrinsics or self.intrinsics),
                "depth_to_color_extrinsics": copy.deepcopy(self.depth_to_color_extrinsics),
                "camera_imu": copy.deepcopy(self.camera_imu),
                "intrinsics": copy.deepcopy(self.intrinsics),
                "color_depth_sync_ms": self.sync_ms,
                "telemetry": copy.deepcopy(self.telemetry),
            }

    def occupancy_snapshot(self) -> dict[str, Any]:
        """Return only the latest occupancy grid for the web map panel."""
        with self._lock:
            value = self.occupancy
            return copy.deepcopy(value) if isinstance(value, dict) else {}
