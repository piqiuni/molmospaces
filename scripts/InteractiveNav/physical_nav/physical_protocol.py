#!/usr/bin/env python3
"""Versioned WebSocket protocol for the read-only Go2 sensor link.

Images are JPEG/PNG encoded and base64 wrapped so the link has no ROS or DDS
dependency on the robot.  Every packet carries a monotonic sequence and wall
clock timestamp; the policy host must never infer motion commands from this
protocol.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Mapping
import zlib

PROTOCOL_VERSION = 1
WIRE_ZLIB_MAGIC = b"MSZ1"
# JPEG/PNG bytes are already entropy-coded.  The outer envelope mainly
# compresses JSON punctuation and base64 alphabet; the default LZ77 strategy
# needlessly searches megabytes of high-entropy image text and can consume a
# full 10 Hz period on the Go2 CPU. Huffman-only coding keeps the envelope
# wire-compatible while making this pass bounded and predictable.
WIRE_ZLIB_STRATEGY = zlib.Z_HUFFMAN_ONLY


def now_wall() -> float:
    return time.time()


def sample_clock_pair() -> tuple[float, float, float]:
    """Sample wall time bracketed by monotonic time.

    Under CPU pressure a thread can be descheduled between two clock reads.
    The midpoint and measured span let the receiver reject such a sample
    instead of mistaking scheduler latency for a host wall-clock step.
    """

    monotonic_before = time.monotonic()
    wall = time.time()
    monotonic_after = time.monotonic()
    return (
        wall,
        .5 * (monotonic_before + monotonic_after),
        max(0., monotonic_after - monotonic_before),
    )


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def encode_wire_packet(
    packet: Mapping[str, Any],
    *,
    compression_level: int = 1,
    compression_strategy: int = WIRE_ZLIB_STRATEGY,
) -> bytes:
    """Encode a packet without changing any RGB or uint16 depth samples.

    JPEG/PNG data is already compressed, but its base64 JSON representation
    adds roughly one third on the wire. A fast outer zlib pass recovers most
    of that overhead while keeping protocol-v1 payload semantics unchanged.
    """
    payload = json.dumps(dict(packet), separators=(",", ":")).encode("utf-8")
    compressor = zlib.compressobj(
        int(compression_level),
        zlib.DEFLATED,
        zlib.MAX_WBITS,
        8,
        int(compression_strategy),
    )
    return WIRE_ZLIB_MAGIC + compressor.compress(payload) + compressor.flush()


def decode_wire_packet(raw: str | bytes) -> dict[str, Any]:
    """Decode either the original text protocol or its compressed envelope."""
    if isinstance(raw, str):
        return json.loads(raw)
    payload = bytes(raw)
    if payload.startswith(WIRE_ZLIB_MAGIC):
        payload = zlib.decompress(payload[len(WIRE_ZLIB_MAGIC):])
    return json.loads(payload.decode("utf-8"))


def image_packet(
    *,
    seq: int,
    stamp: float,
    rgb_jpeg: bytes,
    depth_png: bytes,
    width: int,
    height: int,
    camera_frame: str,
    depth_frame: str | None = None,
    depth_scale: float,
    intrinsics: Mapping[str, float],
    color_depth_sync_ms: float,
    rgb_intrinsics: Mapping[str, Any] | None = None,
    depth_intrinsics: Mapping[str, Any] | None = None,
    depth_to_color_extrinsics: Mapping[str, Any] | None = None,
    camera_imu: Mapping[str, Any] | None = None,
    telemetry: Mapping[str, Any] | None = None,
    capture_timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "sensor_frame",
        "seq": int(seq),
        "stamp": float(stamp),
        "rgb": {"encoding": "jpeg", "data": _b64(rgb_jpeg)},
        "depth": {"encoding": "png16", "data": _b64(depth_png)},
        "width": int(width),
        "height": int(height),
        "camera_frame": str(camera_frame),
        "depth_frame": str(depth_frame or camera_frame),
        "depth_scale": float(depth_scale),
        "intrinsics": dict(intrinsics),
        "rgb_intrinsics": dict(rgb_intrinsics or intrinsics),
        "depth_intrinsics": dict(depth_intrinsics or intrinsics),
        "depth_to_color_extrinsics": dict(depth_to_color_extrinsics or {}),
        "camera_imu": dict(camera_imu or {}),
        **({"capture_timing": dict(capture_timing)} if capture_timing else {}),
        # Carry the body pose sampled with this capture.  The ROS bridge can
        # publish an odom/TF entry before the cloud, and YOLO can use the same
        # pose instead of waiting for the next independent telemetry packet.
        "telemetry": dict(telemetry or {}),
        "color_depth_sync_ms": float(color_depth_sync_ms),
    }


def hello_packet(*, host: str, streams: Mapping[str, Any]) -> dict[str, Any]:
    wall, monotonic, pair_span_s = sample_clock_pair()
    return {
        "v": PROTOCOL_VERSION,
        "type": "hello",
        "role": "go2_readonly_sensor",
        "host": host,
        "streams": dict(streams),
        "capabilities": ["rgb", "depth", "camera_info", "pose", "telemetry"],
        "actuation_enabled": False,
        "stamp": wall,
        # Paired source clocks let the receiver distinguish a real Go2 wall
        # clock step from WebSocket/TCP head-of-line delay.  The field is
        # additive and ignored by protocol-v1 peers that predate it.
        "source_monotonic": monotonic,
        "source_clock_pair_span_s": pair_span_s,
    }


def clock_probe_packet(*, seq: int) -> dict[str, Any]:
    """Small send-time envelope used only for cross-host clock estimation."""

    wall, monotonic, pair_span_s = sample_clock_pair()
    return {
        "v": PROTOCOL_VERSION,
        "type": "clock_probe",
        "seq": int(seq),
        "stamp": wall,
        "source_monotonic": monotonic,
        "source_clock_pair_span_s": pair_span_s,
        "read_only": True,
    }


def telemetry_packet(*, seq: int, telemetry: Mapping[str, Any]) -> dict[str, Any]:
    wall, monotonic, pair_span_s = sample_clock_pair()
    return {
        "v": PROTOCOL_VERSION,
        "type": "telemetry",
        "seq": int(seq),
        "stamp": wall,
        "source_monotonic": monotonic,
        "source_clock_pair_span_s": pair_span_s,
        "read_only": True,
        "telemetry": dict(telemetry),
    }


def command_blocked_packet(*, seq: int, command: Mapping[str, Any]) -> dict[str, Any]:
    """Explicitly acknowledge a keyboard intent without touching the robot."""
    return {
        "v": PROTOCOL_VERSION,
        "type": "read_only_blocked",
        "seq": int(seq),
        "stamp": now_wall(),
        "accepted": False,
        "reason": "physical_nav_phase1_read_only",
        "command": dict(command),
    }


def validate_packet(packet: Mapping[str, Any]) -> None:
    if int(packet.get("v", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported physical_nav protocol version")
    if not packet.get("type"):
        raise ValueError("packet has no type")
    if packet.get("type") == "sensor_frame":
        if int(packet.get("seq", -1)) < 0:
            raise ValueError("sensor frame sequence must be non-negative")
        for name in ("rgb", "depth", "intrinsics", "camera_frame"):
            if name not in packet:
                raise ValueError(f"sensor frame missing {name}")
