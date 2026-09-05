#!/usr/bin/env python3
"""Build presentation videos from a physical Go2 recording session.

The live gateway writes immutable receipts under
``/home/user/ldl/recordings/go2_physical/<session>/raw``.  This script is a
deliberately offline-only consumer: it never contacts Go2, ROS, Qwen, or the
web server.  Saved panel rasters are used when available; missing panels are
reconstructed from the recorded maps/graph/state with the same
``OfflineSixPanelRenderer`` used by the live adapter.

Examples
--------
    # Produce the dark showcase and mux the phone WAV when it exists.
    python build_physical_showcase_video.py go2-20260904-120000-ab12cd34

    # Render all three presentation themes and replace the D435i panel with a
    # separately edited phone video (the D435i source remains in the session).
    python build_physical_showcase_video.py /path/to/session --theme all \
        --phone-video /path/to/phone.mp4 --phone-mode replace
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit("opencv-python and numpy are required for offline synthesis") from exc

try:  # Pillow gives the offline compositor a CJK-capable font fallback.
    from PIL import Image as _PILImage
    from PIL import ImageDraw as _PILImageDraw
    from PIL import ImageFont as _PILImageFont
except ImportError:  # pragma: no cover - OpenCV-only installations
    _PILImage = _PILImageDraw = _PILImageFont = None


SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICAL_DIR = SCRIPT_DIR / "physical_nav"
# The canonical renderer lives beside this script while the physical gateway
# helpers live in ``physical_nav``.  Add both locations so the compositor can
# be imported as a library as well as executed as a CLI.
for _import_path in (SCRIPT_DIR, PHYSICAL_DIR):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

from offline_semantic_renderer import (  # noqa: E402
    OfflineSixPanelRenderer,
    RawGrid,
    TransformResolver,
    TransformSample,
    known_world_bounds,
    load_raw_grid,
)
from physical_six_panel_server import SixPanelRenderer  # noqa: E402


DEFAULT_RECORD_ROOT = Path("/home/user/ldl/recordings/go2_physical")
THEMES = ("dark", "light", "academic")
PANEL_SIZE = (640, 360)
_CJK_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
)
_CJK_FONT_CACHE: dict[tuple[str, int], Any] = {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _time_of(row: dict[str, Any], *, started_wall: float = 0.0) -> float:
    """Return session-relative seconds for any recorder receipt."""

    for key in ("_source_session_time_s", "session_time_s", "_session_time_s"):
        if row.get(key) is not None:
            return max(0.0, _number(row.get(key)))
    wall = row.get("received_wall", row.get("recorded_at", row.get("_recorded_at_wall")))
    if wall is not None and started_wall:
        return max(0.0, _number(wall) - started_wall)
    return max(0.0, _number(row.get("stamp_sec", row.get("stamp", 0.0))))


def _resolve_session(value: str | os.PathLike[str], root: Path = DEFAULT_RECORD_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    candidate = (root / path).resolve()
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"recording session not found: {value}")


def _resolve_artifact(session: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [session / path, session / "raw" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_session_json(session: Path) -> dict[str, Any]:
    path = session / "session.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _last_before(rows: list[dict[str, Any]], timestamp: float, *, started_wall: float = 0.0) -> dict[str, Any] | None:
    result = None
    for row in rows:
        if _time_of(row, started_wall=started_wall) <= timestamp + 1e-6:
            result = row
        else:
            break
    return result


class SessionData:
    """Small indexed view of one immutable raw session."""

    def __init__(self, session: Path) -> None:
        self.session = session
        self.info = _load_session_json(session)
        self.started_wall = _number(self.info.get("started_wall"))
        raw = session / "raw"
        self.boundaries = sorted(
            _jsonl(raw / "step_boundaries.jsonl"), key=lambda row: _time_of(row, started_wall=self.started_wall)
        )
        self.panels = sorted(
            _jsonl(raw / "panels" / "manifest.jsonl"), key=lambda row: _time_of(row, started_wall=self.started_wall)
        )
        self.camera = sorted(
            _jsonl(raw / "camera" / "manifest.jsonl"), key=lambda row: _time_of(row, started_wall=self.started_wall)
        )
        map_manifest = raw / "map_manifest.jsonl"
        if not map_manifest.is_file():
            map_manifest = raw / "maps" / "map_manifest.jsonl"
        self.maps = sorted(
            _jsonl(map_manifest), key=lambda row: _time_of(row, started_wall=self.started_wall)
        )
        self.right_state = sorted(
            _jsonl(raw / "state" / "right_panel.jsonl"), key=lambda row: _time_of(row, started_wall=self.started_wall)
        )
        # Phone uploads are retained as timestamped JPEG receipts.  They are
        # intentionally indexed separately from the optional editor-supplied
        # MP4 so an offline render can use the synchronized phone stream even
        # when no local browser recording was downloaded.
        self.phone_frames = sorted(
            _jsonl(raw / "phone" / "manifest.jsonl"),
            key=lambda row: _time_of(row, started_wall=self.started_wall),
        )
        self.phone_audio = sorted(
            _jsonl(raw / "phone" / "audio_manifest.jsonl"),
            key=lambda row: _time_of(row, started_wall=self.started_wall),
        )
        self.events: list[tuple[str, dict[str, Any]]] = []
        # All state/event JSONL files are intentionally read generically.  This
        # keeps replay compatible with both the first physical recorder schema
        # and future additions without changing the compositor.
        for directory in (raw / "navigation", raw / "semantic", raw / "state"):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.jsonl")):
                if path.name == "right_panel.jsonl":
                    continue
                name = path.stem
                self.events.extend((name, row) for row in _jsonl(path))
        self.events.sort(key=lambda item: _time_of(item[1], started_wall=self.started_wall))
        self._map_cache: dict[tuple[str, str], RawGrid | None] = {}
        self._event_cursor = 0
        self._state_cache: dict[str, Any] = {}

    @property
    def duration(self) -> float:
        candidates = [_time_of(row, started_wall=self.started_wall) for row in self.boundaries]
        candidates += [_time_of(row, started_wall=self.started_wall) for row in self.panels]
        candidates += [_time_of(row, started_wall=self.started_wall) for row in self.camera]
        candidates += [_time_of(row, started_wall=self.started_wall) for row in self.right_state]
        candidates += [_time_of(row, started_wall=self.started_wall) for row in self.phone_frames]
        candidates += [_time_of(row, started_wall=self.started_wall) for row in self.phone_audio]
        return max(candidates or [0.0])

    @property
    def phone_audio_start_s(self) -> float:
        """Session offset of the first phone PCM packet, if present."""

        if not self.phone_audio:
            return 0.0
        return max(0.0, _time_of(self.phone_audio[0], started_wall=self.started_wall))

    def frame_times(self, max_frames: int = 0) -> list[float]:
        if self.boundaries:
            values = [_time_of(row, started_wall=self.started_wall) for row in self.boundaries]
        else:
            values = sorted(
                {
                    round(_time_of(row, started_wall=self.started_wall), 4)
                    for row in self.panels + self.camera + self.right_state + self.phone_frames
                }
            )
        values = sorted(set(max(0.0, value) for value in values))
        if not values:
            values = [0.0]
        if max_frames and len(values) > max_frames:
            stride = max(1, math.ceil(len(values) / max_frames))
            values = values[::stride][:max_frames]
        return values

    def panel_record(self, index: int, timestamp: float) -> dict[str, Any] | None:
        rows = [row for row in self.panels if int(_number(row.get("panel_index"), -1)) == int(index)]
        return _last_before(rows, timestamp, started_wall=self.started_wall)

    def camera_record(self, timestamp: float) -> dict[str, Any] | None:
        return _last_before(self.camera, timestamp, started_wall=self.started_wall)

    def map_record(self, stage: str, timestamp: float) -> dict[str, Any] | None:
        rows = [row for row in self.maps if str(row.get("stage") or "") == stage]
        return _last_before(rows, timestamp, started_wall=self.started_wall)

    def map_grid(self, stage: str, timestamp: float, geometry: RawGrid | None = None) -> RawGrid | None:
        record = self.map_record(stage, timestamp)
        if not record:
            return None
        receipt = str(record.get("receipt_id") or record.get("image") or "")
        key = (stage, receipt)
        if key in self._map_cache:
            return self._map_cache[key]
        image = _resolve_artifact(self.session, record.get("image"))
        if image is not None:
            adjusted = dict(record)
            adjusted["image"] = str(image)
            grid = load_raw_grid(adjusted, geometry=geometry)
        else:
            grid = None
        self._map_cache[key] = grid
        return grid

    def snapshot(self, timestamp: float) -> dict[str, Any]:
        """Merge causal right-panel snapshots and ROS event streams."""

        snapshot = copy.deepcopy(_last_before(self.right_state, timestamp, started_wall=self.started_wall) or self._state_cache or {})
        # A right-panel JSON row has recorder metadata mixed into the object;
        # keep it harmlessly available but remove only internal bookkeeping.
        for key in ("_recorded_at_wall", "_session_time_s"):
            snapshot.pop(key, None)
        for name, row in self.events:
            if _time_of(row, started_wall=self.started_wall) > timestamp + 1e-6:
                break
            value = row.get("value", row)
            if name in {"detections", "mapped_detections", "graph", "consistency", "telemetry", "occupancy", "room_grid", "global_costmap", "local_costmap"}:
                snapshot[name] = copy.deepcopy(value)
            elif name in {"global_plan", "local_global_plan", "local_plan", "current_subgoal", "candidates", "selection", "execution_state", "behavior_feedback", "interaction_result", "decision_trace", "explore_status"}:
                snapshot.setdefault("navigation", {})[name] = copy.deepcopy(value)
            elif name == "mllm_events":
                events = value if isinstance(value, list) else [value]
                normalized_events = []
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    # The recorder wraps each trace as
                    # ``{source, event, receipt-clock}`` so it can retain
                    # transport metadata.  The live state endpoint exposes
                    # the inner event directly; flatten both forms for a
                    # faithful M1/M2/M3 replay and keep the receipt clock for
                    # causal debugging.
                    inner = event.get("event")
                    if isinstance(inner, dict):
                        flattened = copy.deepcopy(inner)
                        for key in ("source", "_recorded_at_wall", "_session_time_s", "_source_session_time_s"):
                            if key in event and key not in flattened:
                                flattened[key] = copy.deepcopy(event[key])
                        normalized_events.append(flattened)
                    else:
                        normalized_events.append(copy.deepcopy(event))
                # A right-panel snapshot intentionally carries a bounded
                # recent MLLM history, while the dedicated event stream also
                # contains every call.  Merge the two without rendering one
                # invocation twice.  Prefer the event-stream copy (it keeps
                # the full prompt/result and extracted image path).
                existing = snapshot.get("mllm_events")
                existing = existing if isinstance(existing, list) else []
                combined = existing + normalized_events
                deduplicated: list[dict[str, Any]] = []
                seen: set[tuple[Any, ...]] = set()
                for event in combined:
                    if not isinstance(event, dict):
                        continue
                    identity = None
                    for key in ("event_id", "request_id", "command_id"):
                        value_id = event.get(key)
                        if value_id not in (None, ""):
                            identity = (
                                key,
                                str(value_id),
                                str(event.get("stage") or event.get("module") or ""),
                                str(event.get("phase") or ""),
                                str(event.get("event_type") or ""),
                                str(event.get("request_sequence") or ""),
                                str(event.get("timestamp") or ""),
                            )
                            break
                    if identity is None:
                        identity = (
                            "fallback",
                            str(event.get("stage") or event.get("module") or ""),
                            str(event.get("request_sequence") or ""),
                            str(event.get("timestamp") or ""),
                            str(event.get("raw_text") or event.get("result") or ""),
                        )
                    if identity in seen:
                        # The later entry comes from the dedicated stream and
                        # is generally the richer representation.
                        for index, prior in enumerate(deduplicated):
                            if prior.get("_mllm_identity") == identity:
                                deduplicated[index] = event
                                deduplicated[index]["_mllm_identity"] = identity
                                break
                        continue
                    seen.add(identity)
                    event = copy.deepcopy(event)
                    event["_mllm_identity"] = identity
                    deduplicated.append(event)
                for event in deduplicated:
                    event.pop("_mllm_identity", None)
                snapshot["mllm_events"] = deduplicated[-200:]
        snapshot.setdefault("navigation_step", int(_number(snapshot.get("navigation_step"))))
        return snapshot


def _image_from_record(session: Path, record: dict[str, Any] | None) -> np.ndarray | None:
    if not record:
        return None
    path = _resolve_artifact(session, record.get("image"))
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def _fit(image: np.ndarray | None, size: tuple[int, int], *, background: tuple[int, int, int] = (20, 25, 32)) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    draw_w, draw_h = max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
    ox, oy = (width - draw_w) // 2, (height - draw_h) // 2
    canvas[oy : oy + draw_h, ox : ox + draw_w] = resized
    return canvas


def _draw_label(image: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = .65) -> None:
    value = str(text)[:100]
    # Hershey fonts used by OpenCV do not contain Chinese glyphs and render
    # them as ``?``.  Draw non-ASCII labels through a small Pillow crop so a
    # long replay keeps the copy cost local to the text rather than converting
    # the complete 1080p frame for every label.  ASCII labels retain the fast
    # OpenCV path.
    if _PILImage is None or not any(ord(char) > 127 for char in value):
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return
    font_path = next((path for path in _CJK_FONT_PATHS if Path(path).is_file()), None)
    if not font_path:
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return
    font_size = max(10, int(round(float(scale) * 32)))
    cache_key = (font_path, font_size)
    font = _CJK_FONT_CACHE.get(cache_key)
    if font is None:
        try:
            font = _PILImageFont.truetype(font_path, font_size)
        except Exception:
            cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
            return
        _CJK_FONT_CACHE[cache_key] = font
    probe = _PILImageDraw.Draw(_PILImage.new("RGB", (1, 1)))
    stroke = max(1, int(round(float(scale) * 1.2)))
    try:
        bounds = probe.textbbox((0, 0), value, font=font, anchor="ls", stroke_width=stroke)
        x, y = int(xy[0]), int(xy[1])
        pad = stroke + 3
        left = max(0, x + bounds[0] - pad)
        top = max(0, y + bounds[1] - pad)
        right = min(image.shape[1], x + bounds[2] + pad)
        bottom = min(image.shape[0], y + bounds[3] + pad)
        if right <= left or bottom <= top:
            return
        crop = image[top:bottom, left:right]
        pil_crop = _PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        drawer = _PILImageDraw.Draw(pil_crop)
        drawer.text(
            (x - left, y - top), value, font=font,
            fill=(int(color[2]), int(color[1]), int(color[0])),
            stroke_width=stroke,
            stroke_fill=(int(color[2]), int(color[1]), int(color[0])),
            anchor="ls",
        )
        image[top:bottom, left:right] = cv2.cvtColor(np.asarray(pil_crop), cv2.COLOR_RGB2BGR)
    except Exception:
        # A missing/old Pillow anchor implementation should not abort replay.
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _panel_from_raw(data: SessionData, timestamp: float, index: int, renderer: OfflineSixPanelRenderer) -> np.ndarray:
    """Reconstruct a missing panel from raw maps/state.

    This path is intentionally conservative.  If a source was not recorded,
    it displays an explicit placeholder instead of silently substituting a
    future frame or a browser crop.
    """

    snapshot = data.snapshot(timestamp)
    planning = data.map_grid("planning_occ", timestamp)
    room = data.map_grid("room_segment", timestamp, geometry=planning)
    global_grid = data.map_grid("global_costmap", timestamp, geometry=planning)
    local_grid = data.map_grid("local_costmap", timestamp, geometry=planning)
    if planning is None:
        planning = global_grid or local_grid
    if global_grid is None:
        global_grid = planning
    if local_grid is None:
        local_grid = planning
    step = SixPanelRenderer._physical_step(snapshot)
    step["step_index"] = int(_number(snapshot.get("navigation_step")))
    bounds = known_world_bounds(planning, margin_m=2.5) if planning is not None else None
    if index == 1:
        image = _image_from_record(data.session, data.camera_record(timestamp))
        if image is None:
            panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), (28, 34, 42), dtype=np.uint8)
            _draw_label(panel, "NO RECORDED RGB", (18, 38), (190, 200, 215))
            return panel
        SixPanelRenderer._draw_live_detections(
            image,
            [item for item in snapshot.get("detections", []) if isinstance(item, dict)],
            (image.shape[0], image.shape[1]),
            include_masks=False,
            include_labels=True,
        )
        return _fit(image, PANEL_SIZE, background=(235, 235, 235))
    if index == 2:
        return renderer.render_map_panel(
            planning, PANEL_SIZE, step, step["step_index"], title="OCC", kind="occupancy",
            world_bounds=bounds, draw_global_plan=True, draw_local_plan=False,
            draw_frontiers=True, draw_semantic_candidates=True,
            draw_interaction_target_links=True, view_scale=1.35,
        )
    if index == 3:
        return renderer.render_room_panel(
            planning, room, PANEL_SIZE, step, step["step_index"], bounds,
            view_scale=1.75, draw_global_plan=True,
        )
    if index == 4:
        left = renderer.render_map_panel(
            global_grid, (PANEL_SIZE[0] // 2, PANEL_SIZE[1]), step, step["step_index"],
            title="GLOBAL COSTMAP", kind="costmap", world_bounds=bounds,
            draw_global_plan=True, draw_local_plan=False, draw_frontiers=True,
            draw_semantic_candidates=True,
        )
        right = renderer.render_map_panel(
            local_grid, (PANEL_SIZE[0] - PANEL_SIZE[0] // 2, PANEL_SIZE[1]), step, step["step_index"],
            title="LOCAL COSTMAP", kind="costmap", draw_global_plan=False,
            draw_local_global_plan=True, draw_local_plan=True,
        )
        return np.concatenate([left, right], axis=1)
    if index == 5:
        return renderer.render_semantic_xy(
            planning, PANEL_SIZE, step, step["step_index"], bounds,
            view_scale=1.8, label_mode="all", draw_overview_inset=False,
        )
    if index == 6:
        return renderer.render_topology(PANEL_SIZE, step, step["step_index"])
    panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), (28, 34, 42), dtype=np.uint8)
    _draw_label(panel, f"PANEL {index} NOT RECORDED", (18, 38), (190, 200, 215))
    return panel


def _theme_palette(theme: str) -> dict[str, tuple[int, int, int]]:
    if theme == "dark":
        # OpenCV stores BGR (the values below are ordered accordingly).
        return {"bg": (22, 12, 5), "panel": (48, 29, 12), "line": (166, 111, 50), "text": (255, 245, 235), "muted": (198, 171, 145), "accent": (255, 200, 105), "good": (151, 232, 98), "warn": (255, 201, 95)}
    if theme == "academic":
        return {"bg": (255, 255, 255), "panel": (253, 251, 250), "line": (194, 181, 171), "text": (55, 37, 26), "muted": (126, 106, 91), "accent": (161, 87, 31), "good": (69, 130, 25), "warn": (0, 119, 183)}
    return {"bg": (252, 249, 247), "panel": (255, 255, 255), "line": (220, 204, 190), "text": (101, 57, 27), "muted": (130, 108, 91), "accent": (155, 83, 24), "good": (84, 139, 24), "warn": (0, 133, 196)}


def _metric_values(snapshot: dict[str, Any]) -> list[tuple[str, str]]:
    telemetry = snapshot.get("telemetry") if isinstance(snapshot.get("telemetry"), dict) else {}
    battery = telemetry.get("battery") if isinstance(telemetry.get("battery"), dict) else {}
    velocity = telemetry.get("velocity") or telemetry.get("linear_velocity") or []
    if isinstance(velocity, (list, tuple)):
        speed = math.hypot(*[_number(item) for item in velocity[:3]])
    else:
        speed = _number(telemetry.get("speed"))
    yaw = _number(telemetry.get("yaw"), float("nan"))
    return [
        ("BATTERY", f"{_number(battery.get('soc', telemetry.get('battery_soc')), 0):.0f}%"),
        ("SPEED", f"{speed:.2f} m/s"),
        ("YAW", "--" if not math.isfinite(yaw) else f"{math.degrees(yaw):.1f} deg"),
        ("MODE", str(telemetry.get("mode", "--"))),
        ("GRAPH", str((snapshot.get("graph") or {}).get("node_count", len((snapshot.get("graph") or {}).get("nodes", []))))),
        ("TARGETS", str(len(snapshot.get("detections") or []))),
    ]


def _event_text(event: dict[str, Any]) -> str:
    raw = event.get("raw_text") or event.get("result") or event.get("payload") or ""
    if isinstance(raw, dict):
        raw = raw.get("result") or raw.get("state") or raw.get("candidate_id") or json.dumps(raw, ensure_ascii=False)
    return " ".join(str(raw).split())[:110]


def _draw_right_rail(snapshot: dict[str, Any], size: tuple[int, int], theme: str, dog: np.ndarray | None = None) -> np.ndarray:
    palette = _theme_palette(theme)
    width, height = size
    image = np.full((height, width, 3), palette["bg"], dtype=np.uint8)
    margin = 18
    status_h = int(height * .30)
    cv2.rectangle(image, (margin, margin), (width - margin, status_h), palette["panel"], -1)
    cv2.rectangle(image, (margin, margin), (width - margin, status_h), palette["line"], 2)
    _draw_label(image, "Go2 CURRENT STATUS", (margin + 18, margin + 34), palette["accent"], .75)
    metrics = _metric_values(snapshot)
    cols = 3
    cell_w = (width - 2 * margin - 28) // cols
    for index, (name, value) in enumerate(metrics):
        row, col = divmod(index, cols)
        x = margin + 10 + col * cell_w
        y = margin + 54 + row * 64
        cv2.rectangle(image, (x, y), (x + cell_w - 10, y + 52), palette["bg"], -1)
        cv2.rectangle(image, (x, y), (x + cell_w - 10, y + 52), palette["line"], 1)
        _draw_label(image, name, (x + 8, y + 18), palette["muted"], .42)
        _draw_label(image, value, (x + 8, y + 42), palette["text"], .58)
    if dog is not None and status_h > 170:
        dog_max_h = max(40, status_h - margin - 54)
        dog_fit = _fit(dog, (min(220, width // 3), dog_max_h), background=palette["panel"])
        dx = width - margin - dog_fit.shape[1] - 12
        dy = margin + 42
        image[dy : min(status_h - 2, dy + dog_fit.shape[0]), dx : dx + dog_fit.shape[1]] = dog_fit[: max(0, status_h - 2 - dy)]

    agent_y = status_h + 32
    cv2.rectangle(image, (margin, agent_y), (width - margin, height - margin), palette["panel"], -1)
    cv2.rectangle(image, (margin, agent_y), (width - margin, height - margin), palette["line"], 2)
    _draw_label(image, "INTERACTIVE NAVIGATION AGENT", (margin + 18, agent_y + 34), palette["accent"], .72)
    y = agent_y + 63
    events = snapshot.get("mllm_events") or []
    if not isinstance(events, list):
        events = []
    events = [item for item in events if isinstance(item, dict)][-6:][::-1]
    _draw_label(image, "MLLM CALLS (M1 / M2 / M3)", (margin + 18, y), palette["muted"], .43)
    y += 15
    for event in events:
        stage = str(event.get("stage") or event.get("module") or "MLLM").upper()
        if y + 43 >= height * .74:
            break
        cv2.rectangle(image, (margin + 12, y), (width - margin - 12, y + 36), palette["bg"], -1)
        cv2.rectangle(image, (margin + 12, y), (width - margin - 12, y + 36), palette["line"], 1)
        _draw_label(image, stage, (margin + 22, y + 23), palette["accent"], .48)
        _draw_label(image, _event_text(event), (margin + 75, y + 23), palette["text"], .39)
        y += 42
    y = max(y + 14, int(height * .60))
    _draw_label(image, "BEHAVIOR TIMELINE", (margin + 18, y), palette["muted"], .43)
    y += 18
    nav = snapshot.get("navigation") if isinstance(snapshot.get("navigation"), dict) else {}
    execution = nav.get("execution_state") if isinstance(nav.get("execution_state"), dict) else {}
    behavior = str(execution.get("behavior_type") or execution.get("state") or "IDLE").upper()
    if "INTERACT" in behavior:
        behavior = "INTERACT"
    elif "EXPLORE" in behavior:
        behavior = "EXPLORE"
    elif "NAVIGAT" in behavior or "MOVE" in behavior:
        behavior = "NAVIGATE"
    else:
        behavior = "IDLE"
    timeline = [
        ("OBSERVE", "RGB-D + YOLOE"),
        ("DECIDE", str((nav.get("decision_trace") or {}).get("model_selected_candidate_id") or "semantic graph")),
        (behavior, str(execution.get("candidate_id") or execution.get("state") or "current state")),
        ("VERIFY", str((nav.get("interaction_result") or {}).get("status") or "read-only record")),
    ]
    for label, detail in timeline:
        if y + 35 >= height - margin:
            break
        cv2.circle(image, (margin + 30, y + 13), 7, palette["good"] if label in {"VERIFY", "INTERACT"} else palette["accent"], -1)
        _draw_label(image, label, (margin + 50, y + 11), palette["text"], .46)
        _draw_label(image, detail, (margin + 50, y + 29), palette["muted"], .36)
        y += 39
    return image


class PhoneReader:
    """Read an editor MP4 or the synchronized JPEG receipts from a session.

    ``VideoCapture.set(POS_MSEC)`` is convenient for an external clip but is
    expensive for some codecs.  Session receipts are already ordered by the
    causal recorder clock, so they use a monotonic cursor and decode each new
    JPEG at most once.  This keeps offline synthesis bounded even for a long
    experiment.
    """

    def __init__(
        self,
        path: Path | None,
        offset_s: float = 0.0,
        *,
        session: Path | None = None,
        phone_records: list[dict[str, Any]] | None = None,
        started_wall: float = 0.0,
    ) -> None:
        self.path = path
        self.offset_s = float(offset_s)
        self.capture = cv2.VideoCapture(str(path)) if path else None
        self.session = session
        self.phone_records = phone_records or []
        self.started_wall = float(started_wall or 0.0)
        self._phone_cursor = -1
        self._phone_image: np.ndarray | None = None

    def read(self, timestamp: float) -> np.ndarray | None:
        target = max(0.0, float(timestamp) + self.offset_s)
        if self.capture is not None:
            if not self.capture.isOpened():
                return None
            self.capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
            ok, frame = self.capture.read()
            return frame if ok else None
        if self.session is None or not self.phone_records:
            return None
        # Advance to the newest receipt at or before the requested display
        # time.  A frame arriving after the timestamp must never leak into a
        # causal replay.
        next_cursor = self._phone_cursor
        while next_cursor + 1 < len(self.phone_records):
            candidate = self.phone_records[next_cursor + 1]
            if _time_of(candidate, started_wall=self.started_wall) > target + 1e-6:
                break
            next_cursor += 1
        if next_cursor < 0:
            return None
        if next_cursor != self._phone_cursor:
            self._phone_cursor = next_cursor
            record = self.phone_records[next_cursor]
            path = _resolve_artifact(self.session, record.get("image"))
            self._phone_image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path else None
        return self._phone_image

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


def _apply_phone(panel: np.ndarray, phone: np.ndarray | None, mode: str) -> np.ndarray:
    if phone is None or mode == "empty":
        return panel
    phone_fit = _fit(phone, (panel.shape[1], panel.shape[0]), background=(0, 0, 0))
    if mode == "replace":
        _draw_label(phone_fit, "PHONE VIDEO", (12, 28), (80, 230, 150), .55)
        return phone_fit
    if mode == "side-by-side":
        half = panel.shape[1] // 2
        left = _fit(panel, (half, panel.shape[0]))
        right = _fit(phone, (panel.shape[1] - half, panel.shape[0]), background=(0, 0, 0))
        _draw_label(right, "PHONE", (10, 27), (80, 230, 150), .5)
        return np.concatenate([left, right], axis=1)
    # picture-in-picture
    pip_w = max(160, panel.shape[1] // 3)
    pip = _fit(phone, (pip_w, max(90, panel.shape[0] // 3)), background=(0, 0, 0))
    x, y = panel.shape[1] - pip.shape[1] - 10, 10
    panel = panel.copy()
    panel[y : y + pip.shape[0], x : x + pip.shape[1]] = pip
    cv2.rectangle(panel, (x, y), (x + pip.shape[1] - 1, y + pip.shape[0] - 1), (80, 230, 150), 2)
    _draw_label(panel, "PHONE", (x + 6, y + 21), (80, 230, 150), .42)
    return panel


def _load_dog() -> np.ndarray | None:
    for path in (
        PHYSICAL_DIR / "assets" / "go2-user-reference.png",
        PHYSICAL_DIR / "assets" / "go2-dark-reference.png",
    ):
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                if image.ndim == 3 and image.shape[2] == 4:
                    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
                    bg = np.full(image[:, :, :3].shape, 12, dtype=np.float32)
                    image = (image[:, :, :3].astype(np.float32) * alpha + bg * (1 - alpha)).astype(np.uint8)
                return image[:, :, :3]
    return None


def _write_video(path: Path, frames: Iterable[np.ndarray], fps: float, size: tuple[int, int]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    count = 0
    try:
        for frame in frames:
            if frame is None:
                continue
            writer.write(_fit(frame, size))
            count += 1
    finally:
        writer.release()
    return count


class _VideoSink:
    """Incremental MP4 writer used by long recordings.

    A physical experiment can run for hours.  Keeping decoded 1080p frames in
    Python lists until the end would consume several gigabytes and make the
    offline compositor look like it had stalled.  This sink writes each frame
    as soon as it is rendered and only retains the frame count.
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.size = size
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {path}")
        self.count = 0

    def write(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        self.writer.write(_fit(frame, self.size))
        self.count += 1

    def close(self) -> None:
        self.writer.release()


def _mux_audio(
    video: Path,
    audio: Path | None,
    output: Path,
    *,
    audio_offset_s: float = 0.0,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not audio or not audio.is_file() or not ffmpeg:
        shutil.copy2(video, output)
        return
    # ``audio_offset_s`` is measured from the recorder session start.  The
    # phone may join after the desktop starts, so delay that input instead of
    # letting its first PCM packet become time zero in the final movie.
    offset = max(0.0, float(audio_offset_s or 0.0))
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
    ]
    if offset > 1e-4:
        command.extend(["-itsoffset", f"{offset:.6f}"])
    command.extend([
        "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        # Phone capture can start after the navigation page.  Pad a short
        # phone track instead of truncating the complete visual experiment.
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-af", "apad", "-shortest",
        "-movflags", "+faststart", str(output),
    ])
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        shutil.copy2(video, output)


def _compose_frame(
    panels: dict[int, np.ndarray], snapshot: dict[str, Any], theme: str, dog: np.ndarray | None,
    phone: np.ndarray | None, phone_mode: str,
    output_size: tuple[int, int], rail: np.ndarray | None = None,
) -> np.ndarray:
    width, height = output_size
    palette = _theme_palette(theme)
    canvas = np.full((height, width, 3), palette["bg"], dtype=np.uint8)
    # Header and panel proportions scale with the requested output size.  The
    # CLI is useful for small review clips as well as the default 1920x1080;
    # fixed 475px panel heights would otherwise overflow a 720p/360p canvas.
    scale = max(0.38, min(1.0, height / 1080.0))
    header_y = max(34, int(round(height * .08)))
    _draw_label(
        canvas, "具身智能交互导航展示平台",
        (max(12, int(28 * scale)), max(20, int(42 * scale))),
        palette["text"], .95 * scale,
    )
    _draw_label(
        canvas, "Embodied Interactive Navigation on Physical Go2",
        (max(12, int(30 * scale)), max(34, int(65 * scale))),
        palette["muted"], .42 * scale,
    )
    _draw_label(
        canvas, "LIVE / OFFLINE REPLAY",
        (max(12, width - max(150, int(275 * scale))), max(20, int(43 * scale))),
        palette["good"], .48 * scale,
    )
    left_w = int(width * .675)
    right_x = left_w + 16
    # Main perception section.
    x0, y0 = max(6, int(round(12 * scale))), header_y
    bottom_margin = max(8, int(round(18 * scale)))
    gap = max(6, int(round(14 * scale)))
    available_main = max(40, height - y0 - bottom_margin)
    p1_height = max(24, int(round((available_main - gap) * .49)))
    p1 = _apply_phone(
        _fit(panels.get(1), (max(24, left_w - 2 * x0), p1_height), background=palette["panel"]),
        phone, phone_mode,
    )
    canvas[y0 : y0 + p1.shape[0], x0 : x0 + p1.shape[1]] = p1
    cv2.rectangle(canvas, (x0, y0), (x0 + p1.shape[1] - 1, y0 + p1.shape[0] - 1), palette["line"], 2)
    _draw_label(
        canvas, "01  实时语义感知",
        (x0 + max(8, int(14 * scale)), y0 + max(16, int(28 * scale))),
        palette["accent"], .58 * scale,
    )
    bottom_y = y0 + p1.shape[0] + gap
    half = max(12, (left_w - 2 * x0 - max(6, int(round(10 * scale)))) // 2)
    bottom_height = max(1, height - bottom_y - bottom_margin)
    for index, x in ((3, x0), (6, x0 + half + 10)):
        panel = _fit(panels.get(index), (half, bottom_height), background=palette["panel"])
        canvas[bottom_y : bottom_y + panel.shape[0], x : x + panel.shape[1]] = panel
        cv2.rectangle(canvas, (x, bottom_y), (x + panel.shape[1] - 1, bottom_y + panel.shape[0] - 1), palette["line"], 2)
        title = "03  空间理解" if index == 3 else "06  交互语义图"
        _draw_label(
            canvas, title,
            (x + max(7, int(12 * scale)), bottom_y + max(15, int(25 * scale))),
            palette["accent"], .52 * scale,
        )
    if rail is None:
        rail = _draw_right_rail(
            snapshot,
            (max(24, width - right_x - x0), max(24, height - header_y - bottom_margin)),
            theme,
            dog,
        )
    rail_x, rail_y = right_x, header_y
    rail_w = min(rail.shape[1], max(0, width - rail_x))
    rail_h = min(rail.shape[0], max(0, height - rail_y))
    if rail_w > 0 and rail_h > 0:
        canvas[rail_y : rail_y + rail_h, rail_x : rail_x + rail_w] = rail[:rail_h, :rail_w]
    return canvas


def render_theme(
    data: SessionData,
    theme: str,
    output_root: Path,
    *,
    fps: float,
    phone_video: Path | None,
    phone_mode: str,
    phone_offset_s: float,
    max_frames: int,
    width: int,
    height: int,
    audio_path: Path | None,
    rebuild_from_raw: bool,
) -> dict[str, Any]:
    theme_dir = output_root / f"showcase-{theme}"
    theme_dir.mkdir(parents=True, exist_ok=True)
    # Replay on a regular timeline.  Receipt/step timestamps are sparse
    # (typically 2--5 Hz) and must not be used as the video frame list,
    # otherwise an 8-second recording would be shortened to ~2 seconds at
    # 10 FPS.  Each sampled timestamp still selects the latest causal receipt.
    duration = max(0.0, data.duration)
    if duration > 0.0:
        count = max(1, int(math.ceil(duration * fps)))
        if max_frames and count > max_frames:
            count = max_frames
        times = [min(index / fps, duration) for index in range(count)]
    else:
        times = data.frame_times(max_frames=max_frames)
    canonical = OfflineSixPanelRenderer(
        transforms=TransformResolver([], map_frame="tf_frame_map", odom_frame="tf_frame_odom")
    )
    dog = _load_dog()
    phone = PhoneReader(
        phone_video,
        phone_offset_s,
        session=data.session,
        phone_records=data.phone_frames,
        started_wall=data.started_wall,
    )
    # Write every decoded frame immediately.  Keeping a list of 1920x1080
    # arrays is prohibitively expensive for a long physical experiment.
    panel_paths = {index: theme_dir / f"panel{index}.mp4" for index in range(1, 7)}
    right_size = (
        max(320, width - int(width * .675) - 28),
        max(180, height - 88),
    )
    sinks: dict[int, _VideoSink] = {}
    right_sink: _VideoSink | None = None
    final_sink: _VideoSink | None = None
    right_rows_handle = None
    frame_count = 0
    try:
        for index, path in panel_paths.items():
            sinks[index] = _VideoSink(path, fps, PANEL_SIZE)
        right_sink = _VideoSink(theme_dir / "right_rail.mp4", fps, right_size)
        final_sink = _VideoSink(theme_dir / "showcase_no_audio.mp4", fps, (width, height))
        right_rows_handle = (theme_dir / "right_rail.jsonl").open("w", encoding="utf-8")
        for timestamp in times:
            snapshot = data.snapshot(timestamp)
            panels: dict[int, np.ndarray] = {}
            # Use exact saved rasters first.  They are the receipt produced by
            # the live canonical renderer, so no browser crop/resize is added.
            for index in panel_paths:
                image = None if rebuild_from_raw else _image_from_record(data.session, data.panel_record(index, timestamp))
                if image is None:
                    image = _panel_from_raw(data, timestamp, index, canonical)
                panels[index] = _fit(image, PANEL_SIZE, background=_theme_palette(theme)["panel"])
                sinks[index].write(panels[index])
            phone_frame = phone.read(timestamp)
            right_frame = _draw_right_rail(snapshot, right_size, theme, dog)
            right_sink.write(right_frame)
            final_sink.write(
                _compose_frame(
                    panels, snapshot, theme, dog, phone_frame, phone_mode,
                    (width, height), rail=right_frame,
                )
            )
            if right_rows_handle is not None:
                right_rows_handle.write(
                    json.dumps(
                        {"session_time_s": timestamp, "snapshot": snapshot},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
                # A periodic flush keeps the sidecar useful if a long render
                # is interrupted, without forcing an fsync for every frame.
                if frame_count % 25 == 0:
                    right_rows_handle.flush()
            frame_count += 1
    finally:
        phone.close()
        if right_rows_handle is not None:
            right_rows_handle.flush()
            right_rows_handle.close()
        for sink in sinks.values():
            sink.close()
        if right_sink is not None:
            right_sink.close()
        if final_sink is not None:
            final_sink.close()

    panel_outputs = {str(index): str(path) for index, path in panel_paths.items()}
    right_path = theme_dir / "right_rail.mp4"
    raw_video = theme_dir / "showcase_no_audio.mp4"
    output_video = theme_dir / "showcase.mp4"
    # Only the recorder-generated phone WAV has a known session clock.  An
    # editor-supplied audio file is assumed to already be aligned and can be
    # shifted explicitly by replacing it or using a pre-aligned export.
    audio_offset_s = data.phone_audio_start_s if audio_path is not None and audio_path == data.session / "raw" / "phone" / "audio.wav" else 0.0
    _mux_audio(raw_video, audio_path, output_video, audio_offset_s=audio_offset_s)
    raw_video.unlink(missing_ok=True)
    manifest = {
        "schema": "physical_showcase_derived_v1",
        "source_session": str(data.session),
        "theme": theme,
        "fps": fps,
        "resolution": [width, height],
        "frame_count": frame_count,
        "phone_video": str(phone_video) if phone_video else "",
        "phone_source": "external_video" if phone_video else ("session_phone_frames" if data.phone_frames else "none"),
        "phone_mode": phone_mode,
        "phone_offset_s": phone_offset_s,
        "audio": str(audio_path) if audio_path else "",
        "audio_offset_s": audio_offset_s,
        "panels": panel_outputs,
        "showcase_video": str(output_video),
        "right_rail_video": str(right_path),
        "raw_sources_preserved": True,
        "panel_source_mode": "raw_rebuild" if rebuild_from_raw else "saved_panel_receipts",
    }
    (theme_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="session directory or id under the physical recording root")
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output", type=Path, default=None, help="derived output root (default: <session>/derived)")
    parser.add_argument("--theme", choices=[*THEMES, "all"], default="dark")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--phone-video", type=Path, default=None)
    parser.add_argument("--phone-mode", choices=["replace", "pip", "side-by-side", "empty"], default="empty")
    parser.add_argument("--phone-offset-s", type=float, default=0.0)
    parser.add_argument("--audio", type=Path, default=None, help="external phone WAV/MP4 audio; default raw/phone/audio.wav")
    parser.add_argument("--rebuild-from-raw", action="store_true", help="ignore saved panel JPEGs and redraw every panel from RGB-D/maps/events")
    args = parser.parse_args()
    session = _resolve_session(args.session, args.record_root)
    data = SessionData(session)
    output_root = (args.output or (session / "derived")).expanduser().resolve()
    audio = args.audio.expanduser().resolve() if args.audio else session / "raw" / "phone" / "audio.wav"
    if not audio.is_file():
        audio = None
    themes = THEMES if args.theme == "all" else (args.theme,)
    results = []
    for theme in themes:
        results.append(render_theme(
            data, theme, output_root, fps=max(.1, args.fps),
            phone_video=args.phone_video.expanduser().resolve() if args.phone_video else None,
            phone_mode=args.phone_mode, phone_offset_s=args.phone_offset_s,
            max_frames=max(0, args.max_frames), width=max(320, args.width), height=max(180, args.height),
            audio_path=audio,
            rebuild_from_raw=args.rebuild_from_raw,
        ))
    print(json.dumps({"session": str(session), "outputs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
