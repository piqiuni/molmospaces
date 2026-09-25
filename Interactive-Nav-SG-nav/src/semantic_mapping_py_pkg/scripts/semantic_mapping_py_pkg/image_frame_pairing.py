"""Pair asynchronous detector payloads with the RGB frame they describe."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def payload_capture_step(payload: Mapping[str, Any]) -> int | None:
    """Return the public capture step, preferring the explicit envelope field."""

    value = payload.get("capture_step")
    if value is None:
        value = payload.get("frame_index")
    return _nonnegative_int(value)


def payload_image_sequence(payload: Mapping[str, Any]) -> int | None:
    """Return an explicit source RGB sequence, ignoring ROS's ambiguous zero."""

    for key in ("image_sequence", "source_image_sequence", "rgb_sequence"):
        value = _nonnegative_int(payload.get(key))
        if value is not None and value > 0:
            return value
    return None


def stamp_key_from_parts(seconds: Any, nanoseconds: Any = 0) -> tuple[int, int] | None:
    try:
        sec = int(seconds)
        nsec = int(nanoseconds)
    except (TypeError, ValueError, OverflowError):
        return None
    total_ns = sec * 1_000_000_000 + nsec
    normalized_sec, normalized_nsec = divmod(total_ns, 1_000_000_000)
    return int(normalized_sec), int(normalized_nsec)


def stamp_key_from_ros(stamp: Any) -> tuple[int, int] | None:
    """Normalize a ROS time-like value without comparing lossy floats."""

    if stamp is None:
        return None
    key = stamp_key_from_parts(getattr(stamp, "secs", None), getattr(stamp, "nsecs", 0))
    if key is not None and (
        getattr(stamp, "secs", None) is not None
        or getattr(stamp, "nsecs", None) is not None
    ):
        return key
    try:
        total_ns = int(round(float(stamp.to_sec()) * 1_000_000_000.0))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return divmod(total_ns, 1_000_000_000)


def stamp_key_from_payload(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the exact serialized ROS stamp carried by a detection envelope."""

    if "stamp_sec" not in payload and "capture_stamp_sec" not in payload:
        return None
    seconds = payload.get("stamp_sec")
    if seconds is None:
        seconds = payload.get("capture_stamp_sec")
    has_nanoseconds = "stamp_nsec" in payload or "capture_stamp_nsec" in payload
    nanoseconds = payload.get("stamp_nsec", payload.get("capture_stamp_nsec"))
    try:
        if has_nanoseconds and nanoseconds not in (None, ""):
            return stamp_key_from_parts(seconds, nanoseconds)
        total_ns = int(round(float(seconds) * 1_000_000_000.0))
    except (TypeError, ValueError, OverflowError):
        return None
    return divmod(total_ns, 1_000_000_000)


def payload_image_size(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    """Read a declared ``[width, height]`` source-image size."""

    value = payload.get("image_size")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def select_image_record(
    payload: Mapping[str, Any], records: Iterable[Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | None, str]:
    """Pair by source stamp/sequence, never by ROS header.seq as task step.

    rospy overwrites header.seq during serialization. Only an explicit
    capture_step on both envelopes can be compared as a task-step identity.
    Legacy float stamps allow rounding error, not asynchronous frame slop.
    """

    candidates = [record for record in records if isinstance(record, Mapping)]
    if not candidates:
        return None, "no_image_available"
    expected_step = payload_capture_step(payload)
    expected_sequence = payload_image_sequence(payload)
    expected_stamp = stamp_key_from_payload(payload)
    if expected_step is None and expected_sequence is None and expected_stamp is None:
        return None, "detection_frame_identity_missing"

    if expected_stamp is not None:
        tolerance_ns = 0
        if not any(payload.get(key) is not None for key in ("stamp_nsec", "capture_stamp_nsec")):
            seconds = payload.get("stamp_sec")
            if seconds is None:
                seconds = payload.get("capture_stamp_sec")
            tolerance_ns = max(1, math.ceil(math.ulp(float(seconds)) * 1_000_000_000))
        expected_ns = expected_stamp[0] * 1_000_000_000 + expected_stamp[1]
        stamp_matches = [
            record
            for record in candidates
            if isinstance(record.get("stamp_key"), (tuple, list))
            and len(record["stamp_key"]) == 2
            and abs(
                int(record["stamp_key"][0]) * 1_000_000_000
                + int(record["stamp_key"][1]) - expected_ns
            ) <= tolerance_ns
        ]
        if stamp_matches:
            candidates = stamp_matches
        else:
            return None, "image_stamp_mismatch"

    if expected_sequence is not None:
        candidates = [
            record for record in candidates
            if _nonnegative_int(record.get("image_sequence")) == expected_sequence
        ]
        if not candidates:
            return None, "image_sequence_mismatch"

    if expected_step is not None:
        candidates = [
            record for record in candidates
            if record.get("capture_step") is None
            or _nonnegative_int(record.get("capture_step")) == expected_step
        ]
        if not candidates:
            return None, "capture_step_image_not_found"
        if expected_stamp is None and expected_sequence is None:
            candidates = [
                record for record in candidates
                if _nonnegative_int(record.get("capture_step")) == expected_step
            ]
            if not candidates:
                return None, "capture_step_image_identity_unavailable"

    identities = {
        (tuple(record.get("stamp_key") or ()), record.get("image_sequence"))
        for record in candidates
    }
    if len(identities) > 1:
        return None, "image_frame_identity_ambiguous"
    return candidates[-1], "matched"


__all__ = [
    "payload_capture_step",
    "payload_image_sequence",
    "payload_image_size",
    "select_image_record",
    "stamp_key_from_payload",
    "stamp_key_from_ros",
]
