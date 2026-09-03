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


def now_wall() -> float:
    return time.time()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def encode_wire_packet(packet: Mapping[str, Any], *, compression_level: int = 1) -> bytes:
    """Encode a packet without changing any RGB or uint16 depth samples.

    JPEG/PNG data is already compressed, but its base64 JSON representation
    adds roughly one third on the wire. A fast outer zlib pass recovers most
    of that overhead while keeping protocol-v1 payload semantics unchanged.
    """
    payload = json.dumps(dict(packet), separators=(",", ":")).encode("utf-8")
    return WIRE_ZLIB_MAGIC + zlib.compress(payload, int(compression_level))


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
    depth_scale: float,
    intrinsics: Mapping[str, float],
    color_depth_sync_ms: float,
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
        "depth_scale": float(depth_scale),
        "intrinsics": dict(intrinsics),
        "color_depth_sync_ms": float(color_depth_sync_ms),
    }


def hello_packet(*, host: str, streams: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "hello",
        "role": "go2_readonly_sensor",
        "host": host,
        "streams": dict(streams),
        "capabilities": ["rgb", "depth", "camera_info", "pose", "telemetry"],
        "actuation_enabled": False,
        "stamp": now_wall(),
    }


def telemetry_packet(*, seq: int, telemetry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "telemetry",
        "seq": int(seq),
        "stamp": now_wall(),
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
