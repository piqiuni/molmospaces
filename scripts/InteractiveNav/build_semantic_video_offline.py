#!/usr/bin/env python3
"""Build an exact-step semantic six-panel video from saved runtime frames."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2


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
    for record in records:
        source_seq = record.get("source_seq")
        record["source_step_value"] = (
            int(source_seq)
            if source_seq_is_step_index and source_seq not in {None, ""}
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
    cv2.rectangle(
        frame,
        (0, max(0, source_height - 28)),
        (min(source_width - 1, 470), source_height - 1),
        (255, 255, 255),
        -1,
    )
    gt_frame = payload.get("frame_index", "-")
    cv2.putText(
        frame,
        f"GT visible={len(observations)} frame={gt_frame} source=realtime_gt",
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
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else scene_dir / "debug"
    sim_manifest = scene_dir / "sim_step_frames" / "manifest.jsonl"
    if not sim_manifest.exists():
        sim_manifest = scene_dir / "step_frames" / "manifest.jsonl"
    sim_records = load_jsonl(sim_manifest)
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
    if not sim_records:
        raise RuntimeError(f"No simulator step frames found under {scene_dir}")
    if not recorder_records:
        raise RuntimeError(f"No recorder state frames found under {debug_dir}")

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
    probe = subprocess.run(
        [
            "ffprobe",
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
