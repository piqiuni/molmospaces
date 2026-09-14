import json
import math


def stamp_to_json(stamp):
    return {
        "stamp_sec": int(stamp.secs),
        "stamp_nsec": int(stamp.nsecs),
    }


def observation_stamp_seconds(payload, fallback=0.0):
    """Read exact ROS parts or legacy floating capture seconds as one clock."""
    if not isinstance(payload, dict):
        return float(fallback)
    try:
        if payload.get("stamp_sec") is not None and payload.get("stamp_nsec") is not None:
            value = float(payload["stamp_sec"]) + float(payload["stamp_nsec"]) * 1e-9
        else:
            value = payload.get("capture_stamp_sec")
            if value is None:
                value = payload.get("stamp_sec", fallback)
            value = float(value)
        return value if math.isfinite(value) else float(fallback)
    except (TypeError, ValueError, OverflowError):
        return float(fallback)


def parse_json_list(data):
    if not data:
        return []
    try:
        value = json.loads(data)
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def parse_json_object_or_text(data):
    if not data:
        return {}
    try:
        value = json.loads(data)
    except ValueError:
        return {"scene_attribute": str(data)}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"scene_attribute": value}
    return {}


def dumps_compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
