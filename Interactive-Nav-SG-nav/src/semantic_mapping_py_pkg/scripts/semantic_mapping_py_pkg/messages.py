import json


def stamp_to_json(stamp):
    return {
        "stamp_sec": int(stamp.secs),
        "stamp_nsec": int(stamp.nsecs),
    }


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
