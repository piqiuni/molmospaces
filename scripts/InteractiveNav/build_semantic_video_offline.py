#!/usr/bin/env python3
"""Build an exact-step semantic six-panel video from saved runtime frames."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from offline_semantic_renderer import (
    GlobalCostmapReplay,
    OfflineSixPanelRenderer,
    TransformResolver,
    draw_camera_title,
    draw_task_subgoal_header,
    known_world_bounds,
    load_raw_grid,
    zoom_panel,
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return sorted(records, key=lambda record: int(record.get("step_index", 0)))


def load_recorder_frames(path: Path, *, source_seq_is_step_index: bool = False) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    source_sequences = [
        int(record["source_seq"])
        for record in records
        if record.get("source_seq") not in {None, ""}
    ]
    source_sequences_are_zero_based = bool(source_sequences) and min(source_sequences) == 0
    for record in records:
        source_seq = record.get("source_seq")
        record["source_step_value"] = (
            int(source_seq)
            if (
                (source_sequences_are_zero_based or source_seq_is_step_index)
                and source_seq not in {None, ""}
            )
            else max(0, int(source_seq) - 1)
            if source_seq not in {None, ""}
            else max(0, int(record.get("step_id") or 1) - 1)
        )
        record["image_stamp_value"] = float(record.get("image_stamp") or 0.0)
    return records


def index_recorder_frames(records: list[dict]) -> dict[int, dict]:
    indexed = {}
    duplicates = []
    for record in records:
        source_step = int(record["source_step_value"])
        if source_step in indexed:
            duplicates.append(source_step)
        indexed[source_step] = record
    if duplicates:
        raise RuntimeError(f"Duplicate recorder source steps: {sorted(set(duplicates))[:20]}")
    return indexed


def align_exact_sim_records(
    sim_records: list[dict], recorder_by_step: dict[int, dict]
) -> tuple[list[dict], list[int]]:
    sim_steps = [int(record.get("step_index", index)) for index, record in enumerate(sim_records)]
    missing_steps = [step for step in sim_steps if step not in recorder_by_step]
    if not missing_steps:
        return sim_records, []
    last_recorder_step = max(recorder_by_step)
    interior_missing = [step for step in missing_steps if step <= last_recorder_step]
    if interior_missing:
        raise RuntimeError(
            f"Missing exact recorder snapshots for {len(interior_missing)} simulator steps: "
            f"{interior_missing[:20]}"
        )
    aligned = [
        record
        for index, record in enumerate(sim_records)
        if int(record.get("step_index", index)) <= last_recorder_step
    ]
    return aligned, missing_steps


def align_latest_recorder_frames(
    sim_records: list[dict], recorder_by_step: dict[int, dict]
) -> dict[int, dict]:
    recorder_steps = sorted(recorder_by_step)
    if not recorder_steps:
        return {}
    aligned = {}
    recorder_index = 0
    for index, sim_record in enumerate(sim_records):
        sim_step = int(sim_record.get("step_index", index))
        while (
            recorder_index + 1 < len(recorder_steps)
            and recorder_steps[recorder_index + 1] <= sim_step
        ):
            recorder_index += 1
        state_step = recorder_steps[recorder_index]
        aligned[sim_step] = recorder_by_step[state_step]
    return aligned


def align_nearest_timestamp_recorder_frames(
    sim_records: list[dict], recorder_records: list[dict]
) -> dict[int, dict]:
    timestamped = sorted(
        (record for record in recorder_records if float(record.get("image_stamp_value") or 0.0) > 0.0),
        key=lambda record: float(record["image_stamp_value"]),
    )
    if not timestamped:
        return {}
    aligned = {}
    recorder_index = 0
    for index, sim_record in enumerate(sim_records):
        sim_step = int(sim_record.get("step_index", index))
        sim_stamp = float(sim_record.get("stamp_sec") or 0.0)
        while recorder_index + 1 < len(timestamped):
            current_delta = abs(float(timestamped[recorder_index]["image_stamp_value"]) - sim_stamp)
            next_delta = abs(float(timestamped[recorder_index + 1]["image_stamp_value"]) - sim_stamp)
            if next_delta > current_delta:
                break
            recorder_index += 1
        aligned[sim_step] = timestamped[recorder_index]
    return aligned


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_route_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(payload.get("events") or [], key=lambda event: float(event.get("wall_time", 0.0)))


def route_event_at_stamp(events: list[dict], stamp_sec: float) -> dict | None:
    selected = None
    for event in events:
        if float(event.get("wall_time", 0.0)) > stamp_sec:
            break
        selected = event
    return selected


def route_target_at_stamp(events: list[dict], stamp_sec: float) -> str:
    target_id = ""
    for event in events:
        if float(event.get("wall_time", 0.0)) > stamp_sec:
            break
        command = event.get("command") or {}
        result = event.get("result") or {}
        portal = event.get("portal") or {}
        target_id = str(
            command.get("object_id")
            or result.get("object_id")
            or portal.get("object_id")
            or event.get("target_root")
            or target_id
            or ""
        )
    return target_id


def gt_draw_spec(
    frame_shape: tuple[int, ...],
    payload: dict,
    observation: dict,
    target_object_id: str = "",
) -> dict | None:
    bbox = observation.get("bbox_2d") or []
    image_size = observation.get("image_size") or payload.get("image_size") or []
    if len(bbox) != 4 or len(image_size) != 2:
        return None
    frame_height, frame_width = frame_shape[:2]
    scale_x = float(frame_width) / max(1, int(image_size[0]))
    scale_y = float(frame_height) / max(1, int(image_size[1]))
    x0, y0, x1, y1 = [int(value) for value in bbox]
    start = (
        max(0, min(frame_width - 1, int(x0 * scale_x))),
        max(0, min(frame_height - 1, int(y0 * scale_y))),
    )
    end = (
        max(0, min(frame_width - 1, int((x1 + 1) * scale_x) - 1)),
        max(0, min(frame_height - 1, int((y1 + 1) * scale_y) - 1)),
    )
    if end[0] <= start[0] or end[1] <= start[1]:
        return None
    object_id = str(observation.get("id") or observation.get("instance_id") or "")
    name = str(observation.get("name") or observation.get("semantic_name") or "object")
    normalized_name = name.lower()
    is_target = bool(target_object_id and object_id == target_object_id)
    is_door = "door" in normalized_name
    is_container = any(
        token in normalized_name
        for token in ("drawer", "cabinet", "fridge", "refrigerator", "wardrobe", "cupboard")
    )
    color = (
        (235, 35, 210)
        if is_target
        else (238, 80, 50)
        if is_door
        else (170, 70, 220)
        if is_container
        else (20, 210, 210)
    )
    return {
        "start": start,
        "end": end,
        "color": color,
        "thickness": 4 if is_target else 2,
        "label": " ".join(
            value
            for value in (
                "INTERACT" if is_target else "",
                name,
                object_id if object_id.startswith("gt_") else "",
            )
            if value
        ),
    }


def draw_gt(frame, payload: dict | None, target_object_id: str = "") -> None:
    if not payload:
        return
    observations = list(payload.get("observations") or [])
    source_height, source_width = frame.shape[:2]
    for observation in observations:
        spec = gt_draw_spec(frame.shape, payload, observation, target_object_id)
        if spec is None:
            continue
        start = spec["start"]
        color = spec["color"]
        cv2.rectangle(frame, start, spec["end"], color, spec["thickness"], cv2.LINE_AA)
        cv2.putText(
            frame,
            spec["label"][:48],
            (start[0], max(18, start[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    gt_frame = payload.get("frame_index", "-")
    status_text = f"GT visible={len(observations)} frame={gt_frame} source=realtime_gt"
    cv2.putText(
        frame,
        status_text,
        (9, source_height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status_text,
        (8, source_height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def draw_route_event(frame, event: dict | None) -> None:
    if not event:
        return
    event_name = str(event.get("event") or "")
    labels = {
        "route_started": "ROUTE: START",
        "system_ready": "ROUTE: READY",
        "navigate_started": "ROUTE: NAVIGATE",
        "navigate_succeeded": "ROUTE: NAVIGATE OK",
        "closed_portal_verified": "ROUTE: CLOSED PORTAL",
        "interaction_started": "ROUTE: INTERACT OPEN",
        "interaction_succeeded": "ROUTE: INTERACT OK",
        "open_portal_verified": "ROUTE: OPEN PORTAL",
        "navigation_map_settle_started": "ROUTE: UPDATE MAP",
        "navigation_map_settled": "ROUTE: MAP READY",
        "route_succeeded": "ROUTE: COMPLETE",
        "route_failed": "ROUTE: FAILED",
    }
    label = labels.get(event_name, f"ROUTE: {event_name}")
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (6, 6), (min(width - 6, 430), 34), (255, 255, 255), -1)
    cv2.putText(
        frame,
        label,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def panel_names(panel_columns: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if panel_columns == 3:
        return (
            ("CAMERA", "OCC", "ROOM + INTERACTION"),
            ("GLOBAL + LOCAL", "SEMANTIC XY", "TOPOLOGY"),
        )
    return ("CAMERA", "OCC"), ("GLOBAL", "LOCAL")


PANEL_FIELDS = {
    "camera": "camera_frame",
    "occ": "map_frame",
    "global_costmap": "global_costmap_frame",
    "local_costmap": "local_costmap_frame",
    "room_interaction": "room_interaction_frame",
    "semantic_spatial": "semantic_spatial_frame",
    "semantic_topology": "semantic_topology_frame",
}


def build_panel_video(
    scene_dir: Path,
    debug_dir: Path,
    recorder_records: list[dict],
    panel: str,
    fps: float,
    output_stem: str,
) -> None:
    field = PANEL_FIELDS[panel]
    frames_dir = scene_dir / "videos" / f"{output_stem}_frames"
    videos_dir = scene_dir / "videos"
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    paths = [resolve_path(record.get(field, ""), debug_dir) for record in recorder_records]
    if not paths or any(not path.exists() for path in paths):
        missing = next((str(path) for path in paths if not path.exists()), "<none>")
        raise RuntimeError(f"Missing saved panel image for {panel}: {missing}")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read first saved panel image for {panel}: {paths[0]}")
    height, width = first.shape[:2]
    raw_video = videos_dir / f"{output_stem}_raw.mp4"
    temp_video = videos_dir / f"{output_stem}_h264_tmp.mp4"
    final_video = videos_dir / f"{output_stem}.mp4"
    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open offline panel video writer for {panel}")
    try:
        for index, path in enumerate(paths, start=1):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read saved panel image at frame {index}: {path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_path = frames_dir / f"frame_{index:06d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Cannot write offline panel frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    ffmpeg_log = videos_dir / f"{output_stem}_ffmpeg.log"
    command = [
        "ffmpeg", "-y", "-i", str(raw_video), "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
        "-movflags", "+faststart", str(temp_video),
    ]
    with ffmpeg_log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not temp_video.exists() or temp_video.stat().st_size <= 0:
        raise RuntimeError(f"Offline panel H264 encoding failed; see {ffmpeg_log}")
    temp_video.replace(final_video)
    raw_video.unlink(missing_ok=True)
    summary = {
        "panel": panel,
        "input_frame_count": len(paths),
        "output_frame_count": len(paths),
        "fps": float(fps),
        "video": str(final_video),
        "frame_dir": str(frames_dir),
    }
    (videos_dir / f"{output_stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def _raw_grid_panel(meta: dict | None, size: tuple[int, int], *, room: bool = False):
    width, height = size
    panel = np.full((height, width, 3), 238, dtype=np.uint8)
    if not meta:
        return panel
    image = cv2.imread(str(meta.get("image") or ""), cv2.IMREAD_UNCHANGED)
    if image is None:
        return panel
    values = image.astype(np.int16) - int(meta.get("png_value_offset", 1))
    if room:
        rgb = np.full((*values.shape, 3), 245, dtype=np.uint8)
        valid = values >= 0
        ids = values[valid].astype(np.int32)
        rgb[valid, 0] = (37 * ids + 70) % 210
        rgb[valid, 1] = (83 * ids + 110) % 210
        rgb[valid, 2] = (151 * ids + 45) % 210
    else:
        rgb = np.full((*values.shape, 3), 205, dtype=np.uint8)
        rgb[values < 0] = (180, 180, 180)
        rgb[(values >= 0) & (values <= 20)] = (250, 250, 250)
        rgb[values >= 50] = (20, 20, 20)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)


def _raw_world_to_panel(meta: dict, xy: list | tuple, size: tuple[int, int]):
    if not meta or len(xy) < 2:
        return None
    resolution = float(meta.get("resolution") or 0.0)
    origin = meta.get("origin") or {}
    if resolution <= 0.0:
        return None
    mx = (float(xy[0]) - float(origin.get("x", 0.0))) / resolution
    my = (float(xy[1]) - float(origin.get("y", 0.0))) / resolution
    source_width = max(1, int(meta.get("width") or 1))
    source_height = max(1, int(meta.get("height") or 1))
    px = int(round(mx * size[0] / source_width))
    py = int(round((source_height - 1 - my) * size[1] / source_height))
    if 0 <= px < size[0] and 0 <= py < size[1]:
        return px, py
    return None


def _draw_raw_overlays(panel, meta: dict | None, step: dict, *, plans: tuple[str, ...] = ()) -> None:
    if not meta:
        return
    size = (panel.shape[1], panel.shape[0])
    colors = {"global_plan": (20, 50, 235), "local_global_plan": (235, 90, 30), "local_plan": (30, 190, 50)}
    for name in plans:
        plan = step.get(name) or {}
        points = plan.get("points") or plan.get("poses") or []
        pixels = []
        for point in points:
            xy = point.get("xy") or [point.get("x"), point.get("y")] if isinstance(point, dict) else point
            if xy and len(xy) >= 2 and xy[0] is not None and xy[1] is not None:
                pixel = _raw_world_to_panel(meta, xy, size)
                if pixel is not None:
                    pixels.append(pixel)
        if len(pixels) >= 2:
            cv2.polylines(panel, [np.asarray(pixels, dtype=np.int32)], False, colors[name], 2, cv2.LINE_AA)
    pose = step.get("pose") or []
    pose_pixel = _raw_world_to_panel(meta, pose, size)
    if pose_pixel is not None:
        cv2.circle(panel, pose_pixel, 6, (230, 40, 210), -1, cv2.LINE_AA)
    goal = step.get("active_goal") or []
    goal_pixel = _raw_world_to_panel(meta, goal, size)
    if goal_pixel is not None:
        cv2.drawMarker(panel, goal_pixel, (20, 30, 235), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)


def _raw_topology_panel(step: dict, size: tuple[int, int]):
    width, height = size
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    graph = step.get("unified_graph") or {}
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    cv2.putText(panel, f"revision={int(graph.get('graph_revision', 0) or 0)} nodes={len(nodes)} edges={len(edges)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    y = 50
    for node in nodes[:12]:
        node_type = str(node.get("type") or "object")
        label = str(node.get("label") or node.get("id") or "")[:42]
        state = str((node.get("interaction") or {}).get("state") or "")
        cv2.putText(panel, f"{node_type:9s} {label} {state}"[:64], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
        y += 18
        if y > height - 12:
            break
    return panel


def _title_panel(panel, title: str, step_index: int) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 28), (255, 255, 255), -1)
    cv2.putText(panel, f"{title}  STEP={step_index + 1:04d}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def build_raw_overview(scene_dir: Path, debug_dir: Path, args, sim_records: list[dict]) -> dict:
    """Reconstruct the established six-panel renderer from raw PNG+JSON data."""

    raw_dir = debug_dir / "raw"
    steps = load_jsonl(raw_dir / "step_boundaries.jsonl")
    maps = load_jsonl(raw_dir / "map_manifest.jsonl")
    if not steps or not maps:
        raise RuntimeError(f"Raw PNG+JSON recording is incomplete under {raw_dir}")
    maps_by_id = {str(record.get("receipt_id")): record for record in maps}
    maps_by_stage: dict[str, list[dict]] = {}
    for record in maps:
        maps_by_stage.setdefault(str(record.get("stage") or ""), []).append(record)
    for records in maps_by_stage.values():
        records.sort(key=lambda record: (float(record.get("stamp_sec", 0.0) or 0.0), int(record.get("source_index", 0) or 0)))
    sim_by_step = {
        int(record.get("step_index", index)): record
        for index, record in enumerate(sim_records)
    }
    raw_step_indexes = [int(step.get("step_index", index)) for index, step in enumerate(steps)]
    missing_sim_step_indexes = sorted(set(raw_step_indexes).difference(sim_by_step))
    missing_raw_step_indexes = sorted(set(sim_by_step).difference(raw_step_indexes))
    if args.state_alignment == "exact" and (missing_sim_step_indexes or missing_raw_step_indexes):
        raise RuntimeError(
            "Raw/simulator step boundaries are not exact: "
            f"missing_sim={missing_sim_step_indexes[:8]} "
            f"missing_raw={missing_raw_step_indexes[:8]}"
        )

    def first_frame(stage: str) -> str:
        return next((str(record.get("frame_id") or "") for record in maps_by_stage.get(stage, []) if record.get("frame_id")), "")

    map_frame = first_frame("planning_occ") or first_frame("raw_occ")
    odom_frame = first_frame("local_costmap_full") or map_frame
    transforms = TransformResolver.from_step_boundaries(
        steps,
        map_frame=map_frame,
        odom_frame=odom_frame,
        fallback_csv=debug_dir / "map_to_odom.csv",
    )
    renderer = OfflineSixPanelRenderer(transforms=transforms)
    global_replay = GlobalCostmapReplay(maps_by_id)

    def receipt_meta(step: dict, stage: str) -> dict | None:
        receipt = str((step.get("receipts") or {}).get(stage) or "")
        record = maps_by_id.get(receipt)
        if record is not None:
            return record
        stamp = float(step.get("stamp_sec", 0.0) or 0.0)
        eligible = [
            item for item in maps_by_stage.get(stage, [])
            if float(item.get("stamp_sec", 0.0) or 0.0) <= stamp
        ]
        return eligible[-1] if eligible else None

    panel_size = (480, 270)
    videos_dir = scene_dir / "videos"
    frames_dir = videos_dir / "offline_composite_frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_video = videos_dir / f"{args.output_stem}_offline_raw.mp4"
    temp_video = videos_dir / f"{args.output_stem}_offline_h264_tmp.mp4"
    final_video = videos_dir / f"{args.output_stem}.mp4"
    alignment_path = scene_dir / "offline_render_alignment.jsonl"
    output_size = (panel_size[0] * 3, panel_size[1] * 2)
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, float(args.fps)), output_size,
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open raw offline video writer")
    written = 0
    component_post_observation_receipt_count = 0
    try:
        with alignment_path.open("w", encoding="utf-8") as alignment_file:
            for step in steps:
                step_index = int(step.get("step_index", written))
                sim_record = sim_by_step.get(step_index)
                if sim_record is None:
                    if args.state_alignment == "exact":
                        raise RuntimeError(f"Missing simulator frame for raw step {step_index}")
                    continue
                sim_stamp = float(sim_record.get("stamp_sec", step.get("stamp_sec", 0.0)) or 0.0)
                receipts = step.get("receipts") or {}
                planning_meta = receipt_meta(step, "planning_occ")
                room_meta = receipt_meta(step, "room_segmentation")
                global_full_meta = receipt_meta(step, "global_costmap_full")
                global_update_meta = receipt_meta(step, "global_costmap_update")
                local_meta = receipt_meta(step, "local_costmap_full")
                selected = {
                    "planning_occ": planning_meta,
                    "room_segmentation": room_meta,
                    "global_costmap_full": global_full_meta,
                    "global_costmap_update": global_update_meta,
                    "local_costmap_full": local_meta,
                }
                receipt_stamps = {
                    stage: float((meta or {}).get("stamp_sec", 0.0) or 0.0)
                    for stage, meta in selected.items()
                }
                component_post_observation_receipt_count += sum(
                    bool(sim_stamp and value and value > sim_stamp + 1e-6)
                    for value in receipt_stamps.values()
                )
                planning = load_raw_grid(planning_meta)
                room = load_raw_grid(room_meta)
                local = load_raw_grid(local_meta)
                global_grid = global_replay.grid_for(
                    global_full_meta,
                    (global_update_meta or {}).get("receipt_id") or receipts.get("global_costmap_update"),
                )
                visualization_config = step.get("visualization_config") or {}
                occ_crop_margin_m = float(
                    visualization_config.get("video_occ_crop_margin_m", 2.5) or 2.5
                )
                global_panel_scale = float(
                    visualization_config.get("video_global_panel_scale", 1.8) or 1.8
                )
                world_bounds = (
                    known_world_bounds(planning, margin_m=occ_crop_margin_m)
                    if planning is not None
                    else None
                )
                camera = cv2.imread(
                    str(resolve_path(str(sim_record.get("frame") or ""), scene_dir)),
                    cv2.IMREAD_COLOR,
                )
                if camera is None:
                    raise RuntimeError(f"Missing simulator camera image for step {step_index}")
                camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
                camera = cv2.resize(camera, panel_size, interpolation=cv2.INTER_AREA)
                selection = step.get("semantic_selection") or {}
                draw_gt(
                    camera,
                    sim_record.get("gt_observations"),
                    str(
                        selection.get("target_id")
                        or selection.get("object_id")
                        or selection.get("candidate_id")
                        or ""
                    ),
                )
                draw_camera_title(camera, step, step_index)
                occ = renderer.render_map_panel(
                    planning, panel_size, step, step_index, title="OCC", kind="occupancy",
                    world_bounds=world_bounds, draw_global_plan=True, draw_local_plan=True,
                    draw_frontiers=True, draw_semantic_candidates=True, draw_route_plan=True,
                )
                draw_task_subgoal_header(occ, step)
                room_panel = renderer.render_room_panel(
                    planning, room, panel_size, step, step_index, world_bounds
                )
                global_width = panel_size[0] // 2
                global_panel = renderer.render_map_panel(
                    global_grid, (global_width, panel_size[1]), step, step_index,
                    title="GLOBAL COSTMAP", kind="costmap", world_bounds=world_bounds,
                    draw_global_plan=True, draw_local_plan=False, draw_frontiers=False,
                )
                global_panel = zoom_panel(global_panel, global_panel_scale)
                local_panel = renderer.render_map_panel(
                    local, (panel_size[0] - global_width, panel_size[1]), step, step_index,
                    title="LOCAL COSTMAP", kind="costmap", draw_global_plan=False,
                    draw_local_global_plan=True, draw_local_plan=True, draw_frontiers=False,
                )
                costmaps = np.concatenate([global_panel, local_panel], axis=1)
                spatial = renderer.render_semantic_xy(
                    planning, panel_size, step, step_index, world_bounds
                )
                topology = renderer.render_topology(panel_size, step, step_index)
                frame = np.vstack([
                    np.concatenate([camera, occ, room_panel], axis=1),
                    np.concatenate([costmaps, spatial, topology], axis=1),
                ])
                output_path = frames_dir / f"frame_{written + 1:06d}_composite.png"
                encoded_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(output_path), encoded_frame):
                    raise RuntimeError(f"Failed to write offline frame {written + 1}")
                writer.write(encoded_frame)
                alignment_file.write(json.dumps({
                    "step_index": step_index,
                    "sim_stamp_sec": sim_stamp,
                    "receipts": {stage: (meta or {}).get("receipt_id", "") for stage, meta in selected.items()},
                    "receipt_stamp_sec": receipt_stamps,
                    "output_frame": str(output_path),
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
    finally:
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    codec = "mp4v"
    if ffmpeg:
        command = [
            ffmpeg, "-y", "-i", str(raw_video), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
            "-movflags", "+faststart", str(temp_video),
        ]
        log_path = videos_dir / f"{args.output_stem}_offline_ffmpeg.log"
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0 or not temp_video.exists() or temp_video.stat().st_size <= 0:
            raise RuntimeError(f"Raw offline H264 encoding failed; see {log_path}")
        temp_video.replace(final_video)
        raw_video.unlink(missing_ok=True)
        codec = "h264"
    else:
        raw_video.replace(final_video)
    summary = {
        "recording_format": "png_json_v1",
        "sim_frame_count": len(sim_records),
        "aligned_sim_frame_count": written,
        "state_frame_count": len(steps),
        "input_step_count": len(steps),
        "output_frame_count": written,
        "exact_step_match_count": written,
        "latest_state_match_count": 0,
        "timestamp_nearest_match_count": 0,
        "state_alignment": args.state_alignment,
        "missing_sim_step_indexes": missing_sim_step_indexes,
        "missing_raw_step_indexes": missing_raw_step_indexes,
        "max_stamp_delta_sec": 0.0,
        "component_post_observation_receipt_count": component_post_observation_receipt_count,
        "component_alignment": "step_boundary_receipts",
        "alignment_jsonl": str(alignment_path),
        "fps": float(args.fps),
        "codec": codec,
        "video": str(final_video),
        "frame_dir": str(frames_dir),
    }
    (scene_dir / "offline_video_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--route-result", default="")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-stamp-delta-sec", type=float, default=0.20)
    parser.add_argument(
        "--state-alignment",
        choices=["exact", "latest", "timestamp"],
        default="exact",
    )
    parser.add_argument("--output-stem", default="overview_6panel")
    parser.add_argument(
        "--panel",
        choices=("overview", *PANEL_FIELDS),
        default="overview",
        help="Build the full six-panel overview or one saved recorder panel.",
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else scene_dir / "debug"
    sim_manifest = scene_dir / "sim_step_frames" / "manifest.jsonl"
    if not sim_manifest.exists():
        sim_manifest = scene_dir / "step_frames" / "manifest.jsonl"
    sim_records = load_jsonl(sim_manifest)
    raw_step_manifest = debug_dir / "raw" / "step_boundaries.jsonl"
    raw_map_manifest = debug_dir / "raw" / "map_manifest.jsonl"
    if args.panel == "overview" and raw_step_manifest.exists() and raw_map_manifest.exists():
        if not sim_records:
            raise RuntimeError(f"No simulator step frames found under {scene_dir}")
        build_raw_overview(scene_dir, debug_dir, args, sim_records)
        return
    recorder_summary = load_json(debug_dir / "summary.json")
    source_seq_is_step_index = (
        recorder_summary.get("first_person_video_trigger") == "step_sync"
    )
    recorder_records = load_recorder_frames(
        debug_dir / "video_frames.csv",
        source_seq_is_step_index=source_seq_is_step_index,
    )
    if not recorder_records:
        recorder_records = load_recorder_frames(
            scene_dir / "video_frames.csv",
            source_seq_is_step_index=source_seq_is_step_index,
        )
    if not recorder_records:
        raise RuntimeError(f"No recorder state frames found under {debug_dir}")
    if args.panel != "overview":
        output_stem = (
            args.output_stem
            if args.output_stem != "overview_6panel"
            else f"{args.panel}_panel"
        )
        build_panel_video(
            scene_dir,
            debug_dir,
            recorder_records,
            args.panel,
            args.fps,
            output_stem,
        )
        return
    if not sim_records:
        raise RuntimeError(f"No simulator step frames found under {scene_dir}")

    original_sim_frame_count = len(sim_records)
    recorder_by_step = index_recorder_frames(recorder_records)
    if args.state_alignment == "exact":
        sim_records, trimmed_trailing_steps = align_exact_sim_records(
            sim_records, recorder_by_step
        )
        state_by_sim_step = {
            int(record.get("step_index", index)): recorder_by_step[
                int(record.get("step_index", index))
            ]
            for index, record in enumerate(sim_records)
        }
    elif args.state_alignment == "latest":
        trimmed_trailing_steps = []
        state_by_sim_step = align_latest_recorder_frames(sim_records, recorder_by_step)
    else:
        trimmed_trailing_steps = []
        state_by_sim_step = align_nearest_timestamp_recorder_frames(sim_records, recorder_records)
    sim_steps = [int(record.get("step_index", index)) for index, record in enumerate(sim_records)]

    route_path = Path(args.route_result).expanduser().resolve() if args.route_result else scene_dir / "route_result.json"
    route_events = load_route_events(route_path)
    first_record = state_by_sim_step[sim_steps[0]]
    first_state = cv2.imread(
        str(resolve_path(first_record["composite_frame"], debug_dir)), cv2.IMREAD_COLOR
    )
    if first_state is None:
        raise RuntimeError("Cannot read first recorder composite frame")
    output_height, output_width = first_state.shape[:2]
    panel_columns = 3 if output_width >= output_height * 2.4 else 2
    output_dir = scene_dir / "videos" / "offline_composite_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = scene_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    raw_video = videos_dir / f"{args.output_stem}_offline_raw.mp4"
    temp_h264 = videos_dir / f"{args.output_stem}_offline_h264_tmp.mp4"
    final_video = videos_dir / f"{args.output_stem}.mp4"
    sync_csv = scene_dir / "offline_video_sync.csv"
    panel_width = output_width // panel_columns
    panel_height = output_height // 2
    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, float(args.fps)),
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open offline video writer")

    titles = panel_names(panel_columns)
    sync_rows = []
    try:
        for frame_index, sim_record in enumerate(sim_records, start=1):
            sim_step_index = int(sim_record.get("step_index", frame_index - 1))
            state_record = state_by_sim_step[sim_step_index]
            stamp_sec = float(sim_record.get("stamp_sec") or 0.0)
            state_stamp = float(state_record.get("image_stamp_value") or 0.0)
            stamp_delta = abs(stamp_sec - state_stamp) if stamp_sec and state_stamp else 0.0
            state_step_index = int(state_record.get("source_step_value", sim_step_index))
            exact_state_match = state_step_index == sim_step_index
            if (
                args.state_alignment == "exact"
                and stamp_delta > max(0.0, float(args.max_stamp_delta_sec))
            ):
                raise RuntimeError(
                    f"Simulator/recorder stamp mismatch at step {sim_step_index}: {stamp_delta:.6f}s"
                )
            state_frame = cv2.imread(
                str(resolve_path(state_record["composite_frame"], debug_dir)), cv2.IMREAD_COLOR
            )
            camera_frame = cv2.imread(
                str(resolve_path(str(sim_record["frame"]), scene_dir)), cv2.IMREAD_COLOR
            )
            if state_frame is None or camera_frame is None:
                raise RuntimeError(f"Missing source image for simulator step {sim_step_index}")
            if state_frame.shape[:2] != (output_height, output_width):
                state_frame = cv2.resize(state_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            camera_frame = cv2.resize(camera_frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            draw_gt(
                camera_frame,
                sim_record.get("gt_observations"),
                route_target_at_stamp(route_events, stamp_sec),
            )
            draw_route_event(camera_frame, route_event_at_stamp(route_events, stamp_sec))
            sim_step = sim_step_index + 1
            state_step = state_step_index + 1
            cv2.putText(
                camera_frame,
                f"SIM STEP={sim_step:04d} STAMP={stamp_sec:.3f}",
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
            state_frame[:panel_height, :panel_width] = camera_frame
            sync_label = f"SIM={sim_step:04d} SRC={state_step:04d} dT={stamp_delta:.3f}s"
            for row in range(2):
                for column in range(panel_columns):
                    if row == 0 and column == 0:
                        continue
                    x0 = column * panel_width
                    y0 = row * panel_height
                    title = f"{titles[row][column]}  STEP={sim_step:04d}"
                    cv2.rectangle(
                        state_frame,
                        (x0, y0),
                        (x0 + panel_width - 1, y0 + 27),
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        state_frame,
                        title,
                        (x0 + 8, y0 + 19),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (20, 20, 20),
                        1,
                        cv2.LINE_AA,
                    )
                    text_size = cv2.getTextSize(sync_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                    text_x = x0 + panel_width - text_size[0] - 8
                    text_y = y0 + panel_height - 10
                    cv2.rectangle(
                        state_frame,
                        (text_x - 4, text_y - 15),
                        (x0 + panel_width - 4, text_y + 4),
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        state_frame,
                        sync_label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (20, 20, 20),
                        1,
                        cv2.LINE_AA,
                    )
            output_path = output_dir / f"frame_{frame_index:06d}_composite.png"
            if not cv2.imwrite(str(output_path), state_frame):
                raise RuntimeError(f"Failed to write offline frame {frame_index}")
            writer.write(state_frame)
            sync_rows.append(
                {
                    "frame_index": frame_index,
                    "sim_step": sim_step,
                    "sim_step_index": sim_step_index,
                    "state_step": state_step,
                    "sim_stamp": f"{stamp_sec:.9f}",
                    "state_stamp": f"{state_stamp:.9f}",
                    "stamp_delta_sec": f"{stamp_delta:.9f}",
                    "match_mode": (
                        "timestamp_nearest"
                        if args.state_alignment == "timestamp"
                        else "exact_step"
                        if exact_state_match
                        else "latest_state"
                    ),
                    "route_event": (route_event_at_stamp(route_events, stamp_sec) or {}).get("event", ""),
                    "output_frame": str(output_path),
                }
            )
    finally:
        writer.release()

    with sync_csv.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(sync_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(sync_rows)

    ffmpeg_log = videos_dir / f"{args.output_stem}_offline_ffmpeg.log"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_video),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(temp_h264),
    ]
    with ffmpeg_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not temp_h264.exists() or temp_h264.stat().st_size <= 0:
        raise RuntimeError(f"Offline H264 encoding failed; see {ffmpeg_log}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(temp_h264),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0 or probe.stdout.strip() != "h264":
            raise RuntimeError(f"Offline video codec verification failed: {probe.stdout.strip()!r}")
    else:
        capture = cv2.VideoCapture(str(temp_h264))
        readable = capture.isOpened()
        capture.release()
        if not readable:
            raise RuntimeError("Offline video cannot be opened and ffprobe is unavailable")
    temp_h264.replace(final_video)
    raw_video.unlink(missing_ok=True)
    summary = {
        "sim_frame_count": original_sim_frame_count,
        "aligned_sim_frame_count": len(sim_records),
        "state_frame_count": len(recorder_records),
        "output_frame_count": len(sync_rows),
        "exact_step_match_count": sum(
            row["match_mode"] == "exact_step" for row in sync_rows
        ),
        "latest_state_match_count": sum(
            row["match_mode"] == "latest_state" for row in sync_rows
        ),
        "timestamp_nearest_match_count": sum(
            row["match_mode"] == "timestamp_nearest" for row in sync_rows
        ),
        "state_alignment": args.state_alignment,
        "trimmed_trailing_sim_steps": trimmed_trailing_steps,
        "max_stamp_delta_sec": max(float(row["stamp_delta_sec"]) for row in sync_rows),
        "fps": float(args.fps),
        "codec": "h264",
        "video": str(final_video),
        "sync_csv": str(sync_csv),
        "frame_dir": str(output_dir),
        "route_result": str(route_path) if route_path.exists() else "",
    }
    (scene_dir / "offline_video_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
