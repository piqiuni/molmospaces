#!/usr/bin/env python3
"""Shared wire protocol for policy-to-Go2 control messages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping


PROTOCOL_VERSION = 1
CONTROL_MODES = ("discrete", "continuous")
LIDAR_ACTIONS = ("toggle", "on", "off")
POSTURE_ACTIONS = ("stand_down", "stand_up")
DEFAULT_SPEECH_VOICE = "zh-CN-XiaoxiaoNeural"


class DiscreteAction(IntEnum):
    """Habitat-compatible discrete navigation actions."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    MOVE_BACKWARD = 4
    MOVE_LEFT = 5
    MOVE_RIGHT = 6


ACTION_NAMES = {action.name: action for action in DiscreteAction}


@dataclass(frozen=True)
class ControlCommand:
    seq: int
    control_mode: str
    ttl_ms: int
    action: DiscreteAction | None = None
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


@dataclass(frozen=True)
class LidarCommand:
    seq: int
    action: str


@dataclass(frozen=True)
class PostureCommand:
    seq: int
    action: str


@dataclass(frozen=True)
class SpeechCommand:
    seq: int
    text: str
    voice: str
    volume: int | None
    ttl_ms: int


def parse_action(value: Any) -> DiscreteAction:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("a discrete command must contain exactly one action")
        value = value[0]
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized.isdigit():
            value = int(normalized)
        else:
            try:
                return ACTION_NAMES[normalized]
            except KeyError as exc:
                raise ValueError(f"unsupported discrete action: {value!r}") from exc
    try:
        return DiscreteAction(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported discrete action: {value!r}") from exc


def infer_control_mode(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("control_mode")
    if explicit is not None:
        mode = str(explicit).strip().lower()
    elif "action" in payload or "discrete_action" in payload:
        mode = "discrete"
    elif "velocity" in payload:
        mode = "continuous"
    else:
        raise ValueError("cannot infer control_mode from message content")
    if mode not in CONTROL_MODES:
        raise ValueError(f"unsupported control_mode: {mode!r}")
    return mode


def _bounded_number(value: Any, limit: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return min(max(parsed, -limit), limit)


def parse_control_message(
    payload: Mapping[str, Any],
    *,
    last_seq: int,
    max_discrete_ttl_ms: int,
    max_continuous_ttl_ms: int,
    max_vx: float,
    max_vy: float,
    max_wz: float,
) -> ControlCommand:
    if int(payload.get("v", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    if payload.get("type") not in {"control", "cmd"}:
        raise ValueError("unsupported message type")

    seq = int(payload["seq"])
    if seq <= last_seq:
        raise ValueError("sequence number did not increase")
    mode = infer_control_mode(payload)
    ttl_ms = int(payload.get("ttl_ms", 0))
    max_ttl_ms = (
        max_discrete_ttl_ms if mode == "discrete" else max_continuous_ttl_ms
    )
    if not 0 < ttl_ms <= max_ttl_ms:
        raise ValueError(
            f"{mode} ttl_ms must be between 1 and {max_ttl_ms}"
        )
    if mode == "discrete":
        raw_action = payload.get("action", payload.get("discrete_action"))
        return ControlCommand(
            seq=seq,
            control_mode=mode,
            ttl_ms=ttl_ms,
            action=parse_action(raw_action),
        )

    velocity = payload.get("velocity")
    if not isinstance(velocity, Mapping):
        raise ValueError("continuous command requires a velocity object")
    return ControlCommand(
        seq=seq,
        control_mode=mode,
        ttl_ms=ttl_ms,
        vx=_bounded_number(velocity.get("vx", velocity.get("linear_x", 0.0)), max_vx, "vx"),
        vy=_bounded_number(velocity.get("vy", velocity.get("linear_y", 0.0)), max_vy, "vy"),
        wz=_bounded_number(velocity.get("wz", velocity.get("angular_z", 0.0)), max_wz, "wz"),
    )


def parse_lidar_message(
    payload: Mapping[str, Any],
    *,
    last_seq: int,
) -> LidarCommand:
    if int(payload.get("v", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    if payload.get("type") != "lidar":
        raise ValueError("unsupported message type")

    seq = int(payload["seq"])
    if seq <= last_seq:
        raise ValueError("sequence number did not increase")
    action = str(payload.get("action", "")).strip().lower()
    if action not in LIDAR_ACTIONS:
        raise ValueError(f"unsupported lidar action: {action!r}")
    return LidarCommand(seq=seq, action=action)


def parse_posture_message(
    payload: Mapping[str, Any],
    *,
    last_seq: int,
) -> PostureCommand:
    if int(payload.get("v", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    if payload.get("type") != "posture":
        raise ValueError("unsupported message type")

    seq = int(payload["seq"])
    if seq <= last_seq:
        raise ValueError("sequence number did not increase")
    action = str(payload.get("action", "")).strip().lower()
    if action not in POSTURE_ACTIONS:
        raise ValueError(f"unsupported posture action: {action!r}")
    return PostureCommand(seq=seq, action=action)


def parse_speech_message(
    payload: Mapping[str, Any],
    *,
    last_seq: int,
    max_text_chars: int,
    max_ttl_ms: int,
) -> SpeechCommand:
    if int(payload.get("v", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    if payload.get("type") != "speak":
        raise ValueError("unsupported message type")

    seq = int(payload["seq"])
    if seq <= last_seq:
        raise ValueError("sequence number did not increase")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("speech text must be a non-empty string")
    text = text.strip()
    if len(text) > max_text_chars:
        raise ValueError(f"speech text exceeds {max_text_chars} characters")

    voice = payload.get("voice", DEFAULT_SPEECH_VOICE)
    if not isinstance(voice, str) or not voice.strip() or len(voice) > 80:
        raise ValueError("speech voice must be a non-empty string up to 80 characters")
    voice = voice.strip()

    raw_volume = payload.get("volume")
    volume = None if raw_volume is None else int(raw_volume)
    if volume is not None and not 0 <= volume <= 10:
        raise ValueError("speech volume must be between 0 and 10")

    ttl_ms = int(payload.get("ttl_ms", 30000))
    if not 1000 <= ttl_ms <= max_ttl_ms:
        raise ValueError(f"speech ttl_ms must be between 1000 and {max_ttl_ms}")
    return SpeechCommand(
        seq=seq,
        text=text,
        voice=voice,
        volume=volume,
        ttl_ms=ttl_ms,
    )


def make_discrete_message(seq: int, action: Any, ttl_ms: int) -> dict[str, Any]:
    parsed = parse_action(action)
    return {
        "v": PROTOCOL_VERSION,
        "type": "control",
        "seq": int(seq),
        "control_mode": "discrete",
        "ttl_ms": int(ttl_ms),
        "action": parsed.name,
        "action_id": int(parsed),
    }


def make_continuous_message(
    seq: int,
    vx: float,
    vy: float,
    wz: float,
    ttl_ms: int,
) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "control",
        "seq": int(seq),
        "control_mode": "continuous",
        "ttl_ms": int(ttl_ms),
        "velocity": {
            "linear_x": float(vx),
            "linear_y": float(vy),
            "angular_z": float(wz),
        },
    }


def make_lidar_message(seq: int, action: str = "toggle") -> dict[str, Any]:
    normalized = str(action).strip().lower()
    if normalized not in LIDAR_ACTIONS:
        raise ValueError(f"unsupported lidar action: {action!r}")
    return {
        "v": PROTOCOL_VERSION,
        "type": "lidar",
        "seq": int(seq),
        "action": normalized,
    }


def make_posture_message(seq: int, action: str) -> dict[str, Any]:
    normalized = str(action).strip().lower()
    if normalized not in POSTURE_ACTIONS:
        raise ValueError(f"unsupported posture action: {action!r}")
    return {
        "v": PROTOCOL_VERSION,
        "type": "posture",
        "seq": int(seq),
        "action": normalized,
    }


def make_speech_message(
    seq: int,
    text: str,
    *,
    voice: str = DEFAULT_SPEECH_VOICE,
    volume: int | None = None,
    ttl_ms: int = 30000,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "type": "speak",
        "seq": int(seq),
        "text": str(text),
        "voice": str(voice),
        "ttl_ms": int(ttl_ms),
    }
    if volume is not None:
        payload["volume"] = int(volume)
    return payload
