#!/usr/bin/env python3
"""Asynchronous, replayable recording for the physical Go2 gateway.

The live gateway intentionally keeps only the newest RGB-D frame and a bounded
MLLM history.  This module is the durable side-channel used by experiments:
callbacks enqueue immutable receipts quickly and a background writer persists
them under one session directory.  It deliberately has no control/actuation
code and can therefore be enabled while the physical platform remains
read-only.

The on-disk layout follows ``scripts/InteractiveNav/raw_recording_format.md``
where possible.  Raw sensor bytes and event payloads are authoritative;
presentation JPEGs are optional derived streams used for quick inspection.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import time
from typing import Any
import wave

try:  # Optional here so protocol-only tests do not need OpenCV.
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - exercised on minimal installations
    cv2 = None
    np = None


DEFAULT_RECORD_DIR = "/home/user/ldl/recordings/go2_physical"
RECORDING_SCHEMA = "physical_raw_v1"


def _json_default(value: Any) -> Any:
    """Convert common ROS/numpy values without failing a recording job."""

    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_sec"):
        try:
            return float(value.to_sec())
        except Exception:
            pass
    return str(value)


def _safe_component(value: Any, fallback: str = "item") -> str:
    text = str(value or fallback)
    text = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return text.strip("._") or fallback


_DROP_LIVE_BINARY_KEYS = {
    "mask",
    "mask_polygon",
    "mask_polygons",
    "camera_segment_points_f32",
    "world_segment_points_f32",
    "point_cloud",
    "pointcloud",
    "overlay_jpeg",
    "rgb_b64",
    "depth_b64",
    # M1 receipts may carry the same camera image as a data URL.  The
    # recorder extracts that image into ``semantic/mllm_inputs`` from the
    # dedicated MLLM event stream; retaining it in every right-panel
    # snapshot would multiply session size and can make a long run appear to
    # stall while JSON is serialized.
    "m1_input_image_data_url",
    "image_data_url",
}


def _compact_live_value(value: Any, *, depth: int = 0) -> Any:
    """Remove binary/point-cloud fields from right-rail JSON receipts.

    The authoritative RGB-D and map rasters are stored in their own files.
    Right-side replay needs semantic boxes, graph state and text decisions;
    retaining a second copy of sparse masks/clouds would multiply session size
    and can stall the writer on long runs.
    """

    if depth > 8:
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _compact_live_value(item, depth=depth + 1)
            for key, item in value.items()
            if str(key) not in _DROP_LIVE_BINARY_KEYS
        }
    if isinstance(value, (list, tuple)):
        # Keep ordinary semantic lists intact; cap pathological debug arrays.
        if len(value) > 2048:
            return [_compact_live_value(item, depth=depth + 1) for item in value[:2048]] + ["…<truncated>"]
        return [_compact_live_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str) and len(value) > 2_000_000:
        return value[:2_000_000] + "…<truncated>"
    return copy.deepcopy(value)


def _now_iso(timestamp: float | None = None) -> str:
    value = _datetime.datetime.fromtimestamp(
        float(timestamp if timestamp is not None else time.time()),
        tz=_datetime.timezone.utc,
    )
    return value.isoformat().replace("+00:00", "Z")


def _encode_grid_values(values: Any, stage: str) -> tuple[Any, int, int, int, int] | None:
    """Return a lossless PNG raster plus offset/shape metadata.

    Occupancy grids normally fit in uint8; room labels and unusual costmaps
    use uint16.  The +1 offset preserves ROS's -1 unknown value exactly.
    """

    if np is None or cv2 is None:
        return None
    try:
        array = np.asarray(values, dtype=np.int32)
    except Exception:
        return None
    if array.ndim != 2 or array.size == 0:
        return None
    minimum = int(array.min())
    maximum = int(array.max())
    offset = 1
    shifted_minimum = minimum + offset
    shifted_maximum = maximum + offset
    if shifted_minimum < 0 or shifted_maximum > 65535:
        return None
    if minimum >= -1 and maximum <= 100 and str(stage) not in {"room_segment", "room_segmentation"}:
        encoded = (array + offset).astype(np.uint8, copy=True)
        bit_depth = 8
    else:
        encoded = (array + offset).astype(np.uint16, copy=True)
        bit_depth = 16
    return encoded, offset, bit_depth, int(array.shape[1]), int(array.shape[0])


@dataclass
class _Job:
    kind: str
    stage: str
    payload: Any
    metadata: dict[str, Any]
    critical: bool = False


class PhysicalRawRecorder:
    """Durable session writer with non-blocking normal ingress.

    ``critical`` jobs (JSON state/MLLM receipts and step boundaries) wait for a
    short period if the queue is full and then mark the session degraded.  Raw
    camera/presentation frames are best-effort in the bounded queue; every
    drop is exposed in ``session.json`` rather than being silent.
    """

    MODES = {"raw_replay", "raw_plus_panels", "page_capture"}
    PANEL_NAMES = {
        1: "panel1_perception",
        2: "panel2_occ",
        3: "panel3_spatial",
        4: "panel4_costmap",
        5: "panel5_semantic_xy",
        6: "panel6_graph",
    }
    SHOWCASE_SECTIONS = {
        1: "perception",
        2: "occ",
        3: "spatial",
        4: "costmap",
        5: "semantic_xy",
        6: "graph",
    }

    def __init__(
        self,
        root_dir: str | os.PathLike[str] = DEFAULT_RECORD_DIR,
        *,
        default_mode: str = "raw_plus_panels",
        queue_size: int = 4096,
        panel_fps: float = 5.0,
    ) -> None:
        mode = str(default_mode or "raw_plus_panels")
        if mode not in self.MODES:
            raise ValueError(f"unsupported recording mode: {mode}")
        self.root_dir = Path(root_dir).expanduser()
        self.default_mode = mode
        self.queue_size = max(64, int(queue_size))
        self.panel_fps = float(panel_fps)
        self._lock = threading.RLock()
        self._queue: queue.Queue[_Job | None] | None = None
        self._worker: threading.Thread | None = None
        self._accepting = False
        self._closed = True
        self._session_dir: Path | None = None
        self._session_id = ""
        self._token = ""
        self._mode = mode
        self._label = ""
        self._session_metadata: dict[str, Any] = {}
        self._started_wall = 0.0
        self._started_mono = 0.0
        self._ended_wall = 0.0
        self._handles: dict[str, Any] = {}
        self._audio_rate = 48_000
        self._audio_bytes = 0
        self._latest_receipts: dict[str, str] = {}
        self._receipt_counters: dict[str, int] = {}
        self._stats: dict[str, Any] = {
            "source_received": {},
            "accepted": {},
            "persisted": {},
            "queue_dropped": {},
            "write_failed": {},
            "queue_peak": 0,
            "enqueue_wait_ms_total": 0.0,
            "enqueue_wait_ms_max": 0.0,
            "errors": [],
        }
        self._degraded = False

    # ------------------------------------------------------------------
    # Session lifecycle and status
    # ------------------------------------------------------------------
    @staticmethod
    def _increment(mapping: dict[str, int], key: str, amount: int = 1) -> None:
        mapping[key] = int(mapping.get(key, 0)) + int(amount)

    def _note(self, bucket: str, stage: str, amount: int = 1) -> None:
        self._increment(self._stats.setdefault(bucket, {}), str(stage), amount)

    def _new_session_id(self) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return f"go2-{stamp}-{secrets.token_hex(4)}"

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._accepting and self._session_dir is not None)

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    def _directories(self, session_dir: Path) -> None:
        paths = [
            session_dir / "raw" / "camera" / "rgb",
            session_dir / "raw" / "camera" / "depth",
            session_dir / "raw" / "panels",
            session_dir / "raw" / "maps",
            session_dir / "raw" / "navigation",
            session_dir / "raw" / "semantic" / "mllm_inputs",
            session_dir / "raw" / "phone" / "frames",
            session_dir / "derived",
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def _write_session_json(self, payload: dict[str, Any]) -> None:
        path = (self._session_dir or self.root_dir) / "session.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        temporary.replace(path)

    def start(
        self,
        *,
        mode: str | None = None,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new session, or return the currently active one."""

        selected_mode = str(mode or self.default_mode)
        if selected_mode not in self.MODES:
            raise ValueError(f"unsupported recording mode: {selected_mode}")
        with self._lock:
            if self._accepting and self._session_dir is not None:
                return self.status()
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._session_id = self._new_session_id()
            self._token = secrets.token_urlsafe(24)
            self._session_dir = self.root_dir / self._session_id
            self._directories(self._session_dir)
            self._mode = selected_mode
            self._label = str(label or "")
            self._session_metadata = copy.deepcopy(metadata or {})
            self._started_wall = time.time()
            self._started_mono = time.monotonic()
            self._ended_wall = 0.0
            self._accepting = True
            self._closed = False
            self._handles = {}
            self._audio_rate = 48_000
            self._audio_bytes = 0
            self._latest_receipts = {}
            self._receipt_counters = {}
            self._stats = {
                "source_received": {},
                "accepted": {},
                "persisted": {},
                "queue_dropped": {},
                "write_failed": {},
                "queue_peak": 0,
                "enqueue_wait_ms_total": 0.0,
                "enqueue_wait_ms_max": 0.0,
                "errors": [],
            }
            self._degraded = False
            self._queue = queue.Queue(maxsize=self.queue_size)
            self._worker = threading.Thread(
                target=self._run,
                name="physical-raw-recording-writer",
                daemon=True,
            )
            self._worker.start()
            session_payload = {
                "schema": RECORDING_SCHEMA,
                "session_id": self._session_id,
                "mode": self._mode,
                "label": self._label,
                "started_at": _now_iso(self._started_wall),
                "started_wall": self._started_wall,
                "started_monotonic": self._started_mono,
                "root_dir": str(self.root_dir),
                "panel_fps": self.panel_fps,
                "metadata": copy.deepcopy(self._session_metadata),
                "read_only": True,
                "status": "recording",
            }
            self._write_session_json(session_payload)
        self.record_event(
            "recording",
            {"event": "started", "session_id": self._session_id, "mode": self._mode},
            critical=True,
        )
        return self.status()

    def stop(self, *, reason: str = "user", timeout: float = 30.0) -> dict[str, Any]:
        """Drain the writer and close the session; safe to call repeatedly."""

        with self._lock:
            if not self._accepting or self._session_dir is None:
                return self.status()
            self._accepting = False
            worker = self._worker
            jobs = self._queue
            session_dir = self._session_dir
            self._ended_wall = time.time()
        if jobs is not None:
            # A sentinel is queued after all accepted work, so the worker drains
            # every receipt before exiting.  This put is allowed to wait longer
            # than normal ingress because stopping is an explicit operation.
            try:
                jobs.put(None, timeout=max(1.0, float(timeout)))
            except queue.Full:
                self._degraded = True
                self._record_error("failed to enqueue recorder stop sentinel")
        if worker is not None:
            worker.join(timeout=max(1.0, float(timeout)))
            if worker.is_alive():
                self._degraded = True
                self._record_error("recorder worker did not drain before timeout")
        with self._lock:
            self._close_handles()
            self._closed = True
            self._write_session_json(
                {
                    "schema": RECORDING_SCHEMA,
                    "session_id": self._session_id,
                    "mode": self._mode,
                    "label": self._label,
                    "started_at": _now_iso(self._started_wall),
                    "ended_at": _now_iso(self._ended_wall),
                    "started_wall": self._started_wall,
                    "ended_wall": self._ended_wall,
                    "duration_s": max(0.0, self._ended_wall - self._started_wall),
                    "root_dir": str(self.root_dir),
                    "metadata": copy.deepcopy(self._session_metadata),
                    "status": "degraded" if self._degraded else "complete",
                    "stop_reason": str(reason),
                    "read_only": True,
                    "stats": self._stats_payload(),
                }
            )
            self._worker = None
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = self._queue
            queue_size = jobs.qsize() if jobs is not None else 0
            return {
                "active": bool(self._accepting and self._session_dir is not None),
                "session_id": self._session_id,
                "mode": self._mode,
                "record_dir": str(self._session_dir) if self._session_dir else "",
                "token_required": bool(self._token),
                "started_at": _now_iso(self._started_wall) if self._started_wall else "",
                "duration_s": max(
                    0.0,
                    (time.time() if self._accepting else self._ended_wall or time.time())
                    - self._started_wall,
                )
                if self._started_wall
                else 0.0,
                "queue_size": queue_size,
                "queue_capacity": self.queue_size,
                "degraded": self._degraded,
                "stats": self._stats_payload(),
            }

    def token(self) -> str:
        with self._lock:
            return self._token

    def authorize(self, session_id: str = "", token: str = "") -> bool:
        with self._lock:
            if not self._accepting:
                return False
            if session_id and session_id != self._session_id:
                return False
            # Local desktop calls may omit a token.  Remote phone calls should
            # provide the token returned by start/join.
            return not token or secrets.compare_digest(str(token), self._token)

    def _stats_payload(self) -> dict[str, Any]:
        accepted_total = sum(self._stats.get("accepted", {}).values())
        jobs = self._queue
        return {
            "source_received": dict(self._stats.get("source_received", {})),
            "accepted": dict(self._stats.get("accepted", {})),
            "persisted": dict(self._stats.get("persisted", {})),
            "queue_dropped": dict(self._stats.get("queue_dropped", {})),
            "write_failed": dict(self._stats.get("write_failed", {})),
            "latest_receipts": dict(self._latest_receipts),
            "queue_size": jobs.qsize() if jobs is not None else 0,
            "queue_capacity": self.queue_size,
            "queue_peak": int(self._stats.get("queue_peak", 0)),
            "enqueue_wait_ms_avg": float(self._stats.get("enqueue_wait_ms_total", 0.0))
            / max(1, accepted_total),
            "enqueue_wait_ms_max": float(self._stats.get("enqueue_wait_ms_max", 0.0)),
            "errors": list(self._stats.get("errors", [])[-20:]),
            "degraded": self._degraded,
            "closed": self._closed,
        }

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._stats.setdefault("errors", []).append(str(message))
            self._stats["errors"] = self._stats["errors"][-50:]

    # ------------------------------------------------------------------
    # Ingress methods
    # ------------------------------------------------------------------
    def _enqueue(
        self,
        kind: str,
        stage: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
        *,
        critical: bool = False,
    ) -> bool:
        stage = str(stage)
        with self._lock:
            if not self._accepting or self._queue is None:
                self._note("queue_dropped", stage)
                return False
            jobs = self._queue
        job = _Job(kind, stage, payload, dict(metadata or {}), critical)
        started = time.perf_counter()
        try:
            if critical:
                jobs.put(job, timeout=1.0)
            else:
                jobs.put_nowait(job)
        except queue.Full:
            with self._lock:
                self._note("queue_dropped", stage)
                self._degraded = True if critical else self._degraded
            return False
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._note("accepted", stage)
            self._stats["queue_peak"] = max(int(self._stats.get("queue_peak", 0)), jobs.qsize())
            self._stats["enqueue_wait_ms_total"] = float(
                self._stats.get("enqueue_wait_ms_total", 0.0)
            ) + elapsed_ms
            self._stats["enqueue_wait_ms_max"] = max(
                float(self._stats.get("enqueue_wait_ms_max", 0.0)), elapsed_ms
            )
        return True

    def _next_receipt(self, stage: str, prefix: str | None = None) -> str:
        with self._lock:
            count = int(self._receipt_counters.get(stage, 0))
            self._receipt_counters[stage] = count + 1
        return f"{_safe_component(prefix or stage)}:{count:08d}"

    def _accepted_receipt(self, stage: str, receipt_id: str) -> None:
        with self._lock:
            self._latest_receipts[str(stage)] = str(receipt_id)

    def record_sensor_packet(self, packet: dict[str, Any]) -> bool:
        """Capture an incoming encoded RGB-D packet before latest-frame drop."""

        if not self.is_active():
            return False
        if not isinstance(packet, dict):
            return False
        # ``page_capture`` is an intentionally small presentation-only mode:
        # retain panels/events but do not duplicate the 10 Hz RGB-D stream.
        if self._mode == "page_capture":
            return False
        stage = "camera_frame"
        self._note("source_received", stage)
        rgb = packet.get("rgb") if isinstance(packet.get("rgb"), dict) else {}
        depth = packet.get("depth") if isinstance(packet.get("depth"), dict) else {}
        rgb_data = str(rgb.get("data") or "")
        depth_data = str(depth.get("data") or "")
        if not rgb_data and not depth_data:
            self._note("write_failed", stage)
            return False
        seq = packet.get("seq", -1)
        stamp = packet.get("stamp", 0.0)
        receipt = self._next_receipt(stage, "camera")
        metadata = {
            "receipt_id": receipt,
            "seq": seq,
            "stamp": stamp,
            "received_wall": time.time(),
            "received_monotonic": time.monotonic(),
            "width": packet.get("width"),
            "height": packet.get("height"),
            "rgb_encoding": rgb.get("encoding", "jpeg"),
            "depth_encoding": depth.get("encoding", "png16"),
            "camera_frame": packet.get("camera_frame", ""),
            "depth_scale": packet.get("depth_scale", 0.001),
            "intrinsics": copy.deepcopy(packet.get("intrinsics", {})),
            "color_depth_sync_ms": packet.get("color_depth_sync_ms"),
            "capture_timing": copy.deepcopy(packet.get("capture_timing", {})),
        }
        accepted = self._enqueue(
            "sensor",
            stage,
            {"rgb": rgb_data, "depth": depth_data},
            metadata,
            # RGB-D is the highest-volume stream.  Never make the Go2
            # websocket wait for a slow disk or a busy encoder; a bounded
            # queue may drop an occasional frame and the drop is exposed in
            # session stats, while critical semantic/state receipts continue
            # to drain reliably.
            critical=False,
        )
        if accepted:
            self._accepted_receipt(stage, receipt)
        return accepted

    def record_telemetry(self, value: Any, *, source: str = "websocket", packet: Any = None) -> bool:
        if not self.is_active():
            return False
        stage = "telemetry"
        payload = {"value": _compact_live_value(value), "source": str(source)}
        if packet is not None:
            # Do not duplicate encoded RGB-D payloads in telemetry JSON.
            if isinstance(packet, dict):
                payload["packet"] = {
                    key: copy.deepcopy(packet.get(key))
                    for key in ("type", "v", "seq", "stamp", "read_only")
                    if key in packet
                }
        return self.record_event(stage, payload, critical=True)

    def record_ros_state(self, name: str, value: Any, payload: Any = None) -> bool:
        # The web mirror calls this for every ROS state receipt even when no
        # recording session is active.  In particular, map values can contain
        # millions of cells; reject them before compaction or deepcopy rather
        # than paying that cost only for ``_enqueue`` to discard the job.
        if not self.is_active():
            return False
        name = str(name)
        map_stages = {
            "occupancy": "planning_occ",
            "room_grid": "room_segment",
            "global_costmap": "global_costmap",
            "local_costmap": "local_costmap",
        }
        if name in map_stages and isinstance(value, dict) and value.get("data") is not None:
            if self._mode == "page_capture":
                return False
            # A map's cell array is encoded once as a lossless PNG below.  Do
            # not embed the same potentially megabyte-sized flat list in the
            # manifest's diagnostic metadata.
            source_payload = {
                "name": name,
                "source": "ros_state",
                "received_wall": time.time(),
            }
            return self._enqueue_map(map_stages[name], value, source_payload=source_payload)
        event = {
            "name": name,
            "value": _compact_live_value(value),
            "payload": _compact_live_value(payload) if payload is not None else {"name": name, "value": _compact_live_value(value)},
            "source": "ros_state",
        }
        return self.record_event(f"ros_state:{name}", event, critical=True)

    def record_mllm_event(self, event: dict[str, Any], *, source: str = "mllm") -> bool:
        if not self.is_active():
            return False
        if not isinstance(event, dict):
            return False
        payload = {"source": str(source), "event": copy.deepcopy(event)}
        return self.record_event("mllm_event", payload, critical=True)

    def record_qwen_event(self, event: dict[str, Any], *, source: str = "qwen") -> bool:
        if not self.is_active():
            return False
        payload = {"source": str(source), "event": _compact_live_value(event)}
        return self.record_event("qwen_event", payload, critical=True)

    def record_phone_frame(
        self,
        data: bytes,
        *,
        sequence: int = 0,
        client_timestamp: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if not self.is_active():
            return False
        if not data:
            return False
        self._note("source_received", "phone_frame")
        return self._enqueue(
            "phone_frame",
            "phone_frame",
            bytes(data),
            {
                "sequence": int(sequence or 0),
                "client_timestamp": client_timestamp,
                "received_wall": time.time(),
                "received_monotonic": time.monotonic(),
                **dict(metadata or {}),
            },
            critical=False,
        )

    def record_phone_audio(
        self,
        data: bytes,
        *,
        sequence: int = 0,
        sample_rate: int = 48_000,
        client_timestamp: Any = None,
    ) -> bool:
        if not self.is_active():
            return False
        if not data:
            return False
        self._note("source_received", "phone_audio")
        return self._enqueue(
            "phone_audio",
            "phone_audio",
            bytes(data),
            {
                "sequence": int(sequence or 0),
                "sample_rate": int(sample_rate or 48_000),
                "client_timestamp": client_timestamp,
                "received_wall": time.time(),
                "received_monotonic": time.monotonic(),
            },
            critical=True,
        )

    def record_panel(self, index: int, data: bytes, *, stamp: Any = None, frame_seq: Any = None) -> bool:
        if not self.is_active():
            return False
        if self._mode == "raw_replay" or not data:
            return False
        index = int(index)
        stage = self.PANEL_NAMES.get(index, f"panel{index}")
        self._note("source_received", stage)
        return self._enqueue(
            "panel",
            stage,
            bytes(data),
            {
                "panel_index": index,
                "showcase_section": self.SHOWCASE_SECTIONS.get(index, f"panel{index}"),
                "stamp": stamp,
                "frame_seq": frame_seq,
                "received_wall": time.time(),
                "received_monotonic": time.monotonic(),
            },
            critical=False,
        )

    def record_state_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Persist the right-rail state at a render boundary."""

        if not self.is_active():
            return False
        if not isinstance(snapshot, dict):
            return False
        # Keep full right-side semantics, but intentionally omit raw map cell
        # arrays and duplicate RGB bytes.  Those have their own receipts.
        selected = {
            key: copy.deepcopy(snapshot.get(key))
            for key in (
                "frame_seq", "navigation_step", "frame_stamp", "telemetry",
                "camera_frame", "depth_scale", "sync_ms", "raw_frame_available",
                "detection_meta", "mapped_detection_meta",
                "detections", "mapped_detections", "graph", "consistency",
                "navigation", "mllm_events", "qwen", "link", "counters",
                "safety", "calibration", "status", "read_only", "generated_at",
                "last_error",
            )
            if key in snapshot
        }
        selected = _compact_live_value(selected)
        selected["showcase_section"] = "right_rail"
        return self.record_event("right_panel_snapshot", selected, critical=True)

    def record_step_boundary(
        self,
        *,
        step_index: Any,
        stamp: Any,
        frame_seq: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        if not self.is_active():
            return False
        with self._lock:
            receipts = dict(self._latest_receipts)
        record = {
            "step_index": int(step_index or 0),
            "stamp_sec": float(stamp or 0.0),
            "frame_seq": frame_seq,
            "recorded_at": time.time(),
            "receipts": receipts,
            **dict(extra or {}),
        }
        return self.record_event("step_boundary", record, critical=True)

    def record_event(self, stage: str, value: Any, *, critical: bool = True) -> bool:
        # Keep the common JSON/event path cheap while recording is disabled.
        # Callers may hand us a large graph or state snapshot, and function
        # arguments are evaluated before ``_enqueue`` can observe inactivity.
        if not self.is_active():
            return False
        self._note("source_received", str(stage))
        return self._enqueue(
            "json",
            str(stage),
            copy.deepcopy(value),
            {"received_wall": time.time(), "received_monotonic": time.monotonic()},
            critical=critical,
        )

    def _enqueue_map(self, stage: str, value: dict[str, Any], source_payload: Any = None) -> bool:
        # Recheck at the expensive private boundary in case a session stopped
        # after the public ingress check.  This avoids the normal inactive case
        # and narrows the stop race before the full-grid deepcopy below.
        if not self.is_active():
            return False
        self._note("source_received", stage)
        receipt = self._next_receipt(stage, "map")
        metadata = {
            "receipt_id": receipt,
            "source_payload": copy.deepcopy(source_payload) if source_payload is not None else None,
            "received_wall": time.time(),
            "received_monotonic": time.monotonic(),
            "stamp": value.get("stamp", value.get("header_stamp", 0.0)),
            "header_seq": value.get("header_seq", value.get("seq", 0)),
            "frame_id": value.get("frame_id", ""),
            "width": value.get("width", 0),
            "height": value.get("height", 0),
            "resolution": value.get("resolution", 0.0),
            "origin": copy.deepcopy(value.get("origin", {})),
        }
        # Maps are another high-volume stream.  A slow disk must not make the
        # ROS callback wait (which would in turn delay navigation); missing
        # map epochs are explicitly counted in ``queue_dropped`` and the
        # nearest causal receipt remains available for replay.
        accepted = self._enqueue("map", stage, copy.deepcopy(value), metadata, critical=False)
        if accepted:
            self._accepted_receipt(stage, receipt)
        return accepted

    # ------------------------------------------------------------------
    # Writer implementation
    # ------------------------------------------------------------------
    def _handle(self, relative: str, mode: str = "a"):
        handle = self._handles.get(relative)
        if handle is None:
            path = (self._session_dir or self.root_dir) / "raw" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open(mode, encoding="utf-8")
            self._handles[relative] = handle
        return handle

    def _append_json(self, relative: str, value: Any, *, stage: str) -> None:
        handle = self._handle(relative)
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")
        handle.flush()
        with self._lock:
            self._note("persisted", stage)

    def _write_bytes(self, relative: str, data: bytes) -> Path:
        path = (self._session_dir or self.root_dir) / "raw" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return path

    def _handle_sensor(self, job: _Job) -> None:
        payload = job.payload if isinstance(job.payload, dict) else {}
        meta = dict(job.metadata)
        receipt = _safe_component(meta.get("receipt_id"), "camera")
        seq = _safe_component(meta.get("seq"), "0")
        stamp = float(meta.get("stamp") or 0.0)
        files: dict[str, str] = {}
        for key, directory, extension in (
            ("rgb", "camera/rgb", "jpg"),
            ("depth", "camera/depth", "png"),
        ):
            encoded = str(payload.get(key) or "")
            if not encoded:
                continue
            data = base64.b64decode(encoded, validate=False)
            name = f"{receipt}_{seq}.{extension}"
            path = self._write_bytes(f"{directory}/{name}", data)
            files[key] = str(path.relative_to(self._session_dir or path.parent))
        record = {
            **meta,
            "files": files,
            "session_time_s": max(0.0, float(meta.get("received_monotonic", 0.0)) - self._started_mono),
        }
        self._append_json("camera/manifest.jsonl", record, stage=job.stage)

    def _handle_map(self, job: _Job) -> None:
        value = job.payload if isinstance(job.payload, dict) else {}
        meta = dict(job.metadata)
        stage = str(job.stage)
        receipt = _safe_component(meta.get("receipt_id"), "map")
        data = value.get("data", [])
        # ROS-style payloads carry a flat row-major list.  Normalize it before
        # handing it to the lossless PNG encoder; retain the original payload
        # for the JSON fallback when geometry is malformed.
        if np is not None:
            try:
                width = int(value.get("width", 0) or 0)
                height = int(value.get("height", 0) or 0)
                flat = np.asarray(data, dtype=np.int32)
                if width > 0 and height > 0 and flat.size >= width * height:
                    data = flat[: width * height].reshape((height, width))
            except Exception:
                pass
        encoded = _encode_grid_values(data, stage)
        record = {
            **meta,
            "stage": stage,
            "receipt_id": receipt,
            "stamp_sec": float(meta.get("stamp") or 0.0),
            "session_time_s": max(0.0, float(meta.get("received_monotonic", 0.0)) - self._started_mono),
        }
        if encoded is not None:
            raster, offset, bit_depth, width, height = encoded
            ok, compressed = cv2.imencode(".png", raster)
            if not ok:
                raise RuntimeError(f"failed to encode map {stage}")
            relative = f"maps/{stage}/{receipt}.png"
            path = self._write_bytes(relative, bytes(compressed))
            record.update({
                "image": str(path.relative_to(self._session_dir or path.parent)),
                "width": int(value.get("width") or width),
                "height": int(value.get("height") or height),
                "png_value_offset": offset,
                "png_bit_depth": bit_depth,
            })
        else:
            # A JSON fallback keeps the receipt inspectable on hosts without
            # OpenCV; normal physical runs use the lossless PNG branch.
            relative = f"maps/{stage}/{receipt}.json"
            path = self._write_bytes(
                relative,
                json.dumps(value, ensure_ascii=False, default=_json_default).encode("utf-8"),
            )
            record["json"] = str(path.relative_to(self._session_dir or path.parent))
        # Keep the canonical manifest at raw/map_manifest.jsonl (the path used
        # by the offline format).  Individual rasters remain grouped under
        # raw/maps/<stage>/.
        self._append_json("map_manifest.jsonl", record, stage=stage)

    def _handle_json(self, job: _Job) -> None:
        value = job.payload
        stage = str(job.stage)
        if stage == "step_boundary":
            relative = "step_boundaries.jsonl"
        elif stage == "right_panel_snapshot":
            relative = "state/right_panel.jsonl"
        elif stage == "telemetry":
            relative = "telemetry.jsonl"
        elif stage == "mllm_event":
            relative = "semantic/mllm_events.jsonl"
            value = self._prepare_mllm(value)
        elif stage == "qwen_event":
            relative = "semantic/qwen_events.jsonl"
        elif stage.startswith("ros_state:"):
            name = _safe_component(stage.split(":", 1)[1])
            if name in {"global_plan", "local_global_plan", "local_plan", "odom", "subgoal"}:
                relative = f"navigation/{name}.jsonl"
            elif name in {"graph", "unified_graph"}:
                relative = "semantic/unified_graph.jsonl"
            elif name in {"candidates", "selection", "execution_state", "behavior_feedback", "interaction_result", "decision_trace", "explore_status", "current_subgoal"}:
                relative = f"semantic/{name}.jsonl"
            else:
                relative = f"state/{name}.jsonl"
        elif stage == "recording":
            relative = "recording_events.jsonl"
        else:
            relative = f"state/{_safe_component(stage)}.jsonl"
        if isinstance(value, dict):
            # Keep both clocks: ``_source_session_time_s`` is when the
            # gateway accepted the receipt, while ``_session_time_s`` is when
            # the background writer serialized it.  Offline replay uses the
            # source clock so a temporarily busy disk cannot shift a Qwen,
            # state, or step-boundary event relative to its video frame.
            received_monotonic = job.metadata.get("received_monotonic")
            source_time = None
            if received_monotonic is not None:
                try:
                    source_time = max(0.0, float(received_monotonic) - self._started_mono)
                except (TypeError, ValueError, OverflowError):
                    source_time = None
            value = {
                **value,
                "_recorded_at_wall": time.time(),
                "_session_time_s": max(0.0, time.monotonic() - self._started_mono),
            }
            if source_time is not None:
                value["_source_session_time_s"] = source_time
        self._append_json(relative, value, stage=stage)

    def _prepare_mllm(self, value: Any) -> Any:
        """Persist M1 image bytes separately while keeping event semantics."""

        if not isinstance(value, dict):
            return value
        result = copy.deepcopy(value)

        def visit(node: Any, path_hint: str = "event") -> Any:
            if isinstance(node, dict):
                output = {}
                for key, item in node.items():
                    if key == "m1_input_image_data_url" and isinstance(item, str) and "," in item:
                        header, encoded = item.split(",", 1)
                        try:
                            data = base64.b64decode(encoded, validate=True)
                            event = result.get("event") if isinstance(result.get("event"), dict) else result
                            # Request sequence numbers are sometimes reused
                            # after a policy/node restart.  Include the event
                            # timestamp in the filename so two valid M1
                            # inputs can never overwrite one another in a
                            # long recording.
                            event_key = event.get("event_id") or event.get("request_sequence") or "mllm"
                            try:
                                event_stamp = int(float(event.get("timestamp") or time.time()) * 1_000_000)
                            except (TypeError, ValueError, OverflowError):
                                event_stamp = int(time.time() * 1_000_000)
                            event_id = _safe_component(f"{event_key}_{event_stamp}", "mllm")
                            extension = "png" if "png" in header.lower() else "jpg"
                            relative = f"semantic/mllm_inputs/{event_id}.{extension}"
                            image_path = self._write_bytes(relative, data)
                            output["m1_input_image_path"] = str(image_path.relative_to(self._session_dir or image_path.parent))
                            output["m1_input_image_sha256"] = hashlib.sha256(data).hexdigest()
                            continue
                        except Exception:
                            # Keep a short diagnostic rather than dropping the
                            # whole MLLM event when an optional image is invalid.
                            output["m1_input_image_decode_error"] = True
                            continue
                    output[key] = visit(item, f"{path_hint}.{key}")
                return output
            if isinstance(node, list):
                return [visit(item, path_hint) for item in node]
            return node

        return visit(result)

    def _handle_panel(self, job: _Job) -> None:
        meta = dict(job.metadata)
        index = int(meta.get("panel_index", 0) or 0)
        stage = str(job.stage)
        sequence = _safe_component(meta.get("frame_seq"), "0")
        receipt_id = self._next_receipt(stage, "panel")
        # Receipt IDs intentionally use ``stage:counter`` for readability;
        # sanitize only the filesystem component so sessions remain portable.
        receipt = _safe_component(receipt_id, "panel")
        relative = f"panels/{stage}/{receipt}_{sequence}.jpg"
        path = self._write_bytes(relative, bytes(job.payload))
        record = {
            **meta,
            "receipt_id": receipt_id,
            "stage": stage,
            "panel_index": index,
            "image": str(path.relative_to(self._session_dir or path.parent)),
            "session_time_s": max(0.0, float(meta.get("received_monotonic", 0.0)) - self._started_mono),
        }
        self._append_json("panels/manifest.jsonl", record, stage=stage)

    def _handle_phone_frame(self, job: _Job) -> None:
        meta = dict(job.metadata)
        sequence = _safe_component(meta.get("sequence"), "0")
        receipt_id = self._next_receipt("phone_frame", "phone")
        receipt = _safe_component(receipt_id, "phone")
        relative = f"phone/frames/{receipt}_{sequence}.jpg"
        path = self._write_bytes(relative, bytes(job.payload))
        record = {
            **meta,
            "receipt_id": receipt_id,
            "image": str(path.relative_to(self._session_dir or path.parent)),
            "session_time_s": max(0.0, float(meta.get("received_monotonic", 0.0)) - self._started_mono),
        }
        self._append_json("phone/manifest.jsonl", record, stage=job.stage)

    def _handle_phone_audio(self, job: _Job) -> None:
        meta = dict(job.metadata)
        rate = int(meta.get("sample_rate") or self._audio_rate or 48_000)
        if rate != self._audio_rate and self._audio_bytes:
            # Keep one deterministic stream; a rate change is recorded and the
            # first rate remains the WAV header's rate.
            rate = self._audio_rate
        self._audio_rate = rate
        relative = "phone/audio.pcm"
        audio_path = (self._session_dir or self.root_dir) / "raw" / relative
        if self._audio_bytes == 0:
            # The first write is atomic; subsequent packets append in this
            # single writer thread.
            self._write_bytes(relative, bytes(job.payload))
        else:
            with audio_path.open("ab") as handle:
                handle.write(bytes(job.payload))
        self._audio_bytes += len(job.payload)
        record = {
            **meta,
            "sample_rate": self._audio_rate,
            "bytes": len(job.payload),
            "offset_bytes": self._audio_bytes - len(job.payload),
            "session_time_s": max(0.0, float(meta.get("received_monotonic", 0.0)) - self._started_mono),
        }
        self._append_json("phone/audio_manifest.jsonl", record, stage=job.stage)

    def _close_handles(self) -> None:
        for handle in list(self._handles.values()):
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        # Make the phone PCM immediately usable by ffmpeg/editors.
        if self._session_dir is not None:
            pcm = self._session_dir / "raw" / "phone" / "audio.pcm"
            wav_path = self._session_dir / "raw" / "phone" / "audio.wav"
            if pcm.is_file() and pcm.stat().st_size:
                try:
                    with pcm.open("rb") as source, wave.open(str(wav_path), "wb") as target:
                        target.setnchannels(1)
                        target.setsampwidth(2)
                        target.setframerate(int(self._audio_rate or 48_000))
                        while True:
                            chunk = source.read(1 << 20)
                            if not chunk:
                                break
                            target.writeframes(chunk)
                except Exception as exc:  # pragma: no cover - filesystem only
                    self._record_error(f"phone WAV conversion failed: {exc}")

    def _run(self) -> None:
        while True:
            jobs = self._queue
            if jobs is None:
                return
            job = jobs.get()
            try:
                if job is None:
                    return
                if job.kind == "sensor":
                    self._handle_sensor(job)
                elif job.kind == "map":
                    self._handle_map(job)
                elif job.kind == "json":
                    self._handle_json(job)
                elif job.kind == "panel":
                    self._handle_panel(job)
                elif job.kind == "phone_frame":
                    self._handle_phone_frame(job)
                elif job.kind == "phone_audio":
                    self._handle_phone_audio(job)
                else:
                    raise RuntimeError(f"unknown recording job kind: {job.kind}")
            except Exception as exc:  # Disk errors must be visible, not fatal to gateway.
                with self._lock:
                    self._note("write_failed", job.stage if job else "unknown")
                    self._stats.setdefault("errors", []).append(
                        f"{job.stage if job else 'unknown'}: {type(exc).__name__}: {exc}"
                    )
                    self._stats["errors"] = self._stats["errors"][-50:]
                    self._degraded = True
            finally:
                jobs.task_done()

    def close(self) -> dict[str, Any]:
        return self.stop(reason="close")

    def list_sessions(self) -> list[dict[str, Any]]:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        result: list[dict[str, Any]] = []
        for directory in sorted(self.root_dir.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            session_file = directory / "session.json"
            item: dict[str, Any] = {"session_id": directory.name, "record_dir": str(directory)}
            try:
                loaded = json.loads(session_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    for key in ("status", "mode", "started_at", "ended_at", "duration_s", "stop_reason", "stats"):
                        if key in loaded:
                            item[key] = loaded[key]
            except Exception:
                item["status"] = "incomplete"
            result.append(item)
        return result[:100]


__all__ = ["DEFAULT_RECORD_DIR", "PhysicalRawRecorder", "RECORDING_SCHEMA"]
