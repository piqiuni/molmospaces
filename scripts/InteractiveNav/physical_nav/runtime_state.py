#!/usr/bin/env python3
"""Thread-safe state shared by the physical gateway and web renderer."""

from __future__ import annotations

import copy
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
        self.camera_frame = ""
        self.calibration: dict[str, Any] = {}
        self.frame_seq = -1
        self.frame_stamp = 0.0
        self.sync_ms = None
        self.telemetry: dict[str, Any] = {}
        self.detections: list[dict[str, Any]] = []
        self.detection_meta: dict[str, Any] = {}
        self.mapped_detections: list[dict[str, Any]] = []
        self.mapped_detection_meta: dict[str, Any] = {}
        self.graph: dict[str, Any] = {}
        self.occupancy = None
        self.consistency: dict[str, Any] = {}
        self.qwen: dict[str, Any] = {"requests": [], "results": []}
        self.link: dict[str, Any] = {
            "connected": False,
            "connections": 0,
            "last_hello": {},
            "last_packet_at": 0.0,
            "last_disconnect_at": 0.0,
        }
        self.counters = {"frames": 0, "detections": 0, "graph_updates": 0, "dropped": 0}
        self.last_error = ""

    def update_frame(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.counters["frames"] += 1

    def set_calibration(self, **value: Any) -> None:
        with self._lock:
            self.calibration = copy.deepcopy(value)

    def update_topic(self, name: str, value: Any) -> None:
        with self._lock:
            if name in {"detections", "mapped_detections"} and isinstance(value, dict):
                meta_attr = "detection_meta" if name == "detections" else "mapped_detection_meta"
                setattr(self, meta_attr, {k: v for k, v in value.items() if k not in {"detections", "objects"}})
                value = value.get("detections", value.get("objects", []))
            setattr(self, name, value)
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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frame_seq": self.frame_seq,
                "frame_stamp": self.frame_stamp,
                "camera_frame": self.camera_frame,
                "calibration": copy.deepcopy(self.calibration),
                "sync_ms": self.sync_ms,
                "depth_scale": self.depth_scale,
                "intrinsics": copy.deepcopy(self.intrinsics),
                "raw_frame_available": bool(self.rgb_b64 and self.depth_b64),
                "telemetry": copy.deepcopy(self.telemetry),
                "detections": copy.deepcopy(self.detections),
                "detection_meta": copy.deepcopy(self.detection_meta),
                "mapped_detections": copy.deepcopy(self.mapped_detections),
                "mapped_detection_meta": copy.deepcopy(self.mapped_detection_meta),
                "graph": copy.deepcopy(self.graph),
                "occupancy": ({k: self.occupancy.get(k) for k in ("width", "height", "resolution", "origin")} if isinstance(self.occupancy, dict) else None),
                "consistency": copy.deepcopy(self.consistency),
                "qwen": copy.deepcopy(self.qwen),
                "link": copy.deepcopy(self.link),
                "counters": dict(self.counters),
                "last_error": self.last_error,
                "read_only": True,
                "status": "READ_ONLY_BLOCKED",
                "generated_at": time.time(),
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
                "depth_scale": self.depth_scale,
                "intrinsics": copy.deepcopy(self.intrinsics),
                "color_depth_sync_ms": self.sync_ms,
                "telemetry": copy.deepcopy(self.telemetry),
            }
