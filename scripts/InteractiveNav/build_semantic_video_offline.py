#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import subprocess
from pathlib import Path

import cv2


def draw_outlined_text(frame, text, origin, scale=0.46, thickness=1) -> None:
    cv2.putText(
        frame,
        text,
        (origin[0] + 1, origin[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_recorder_frames(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        try:
            row["image_stamp_value"] = float(row.get("image_stamp") or 0.0)
        except (TypeError, ValueError):
            continue
        if not row.get("composite_frame"):
            continue
        records.append(row)
    return sorted(records, key=lambda row: row["image_stamp_value"])


def causal_recorder_frame(records: list[dict], stamp_sec: float) -> dict | None:
    selected = None
    for record in records:
        if record["image_stamp_value"] > stamp_sec:
            break
        selected = record
    if selected is None and records:
        selected = records[0]
    return selected


def nearest_sim_step(sim_stamps: list[float], stamp_sec: float) -> int:
    index = bisect.bisect_left(sim_stamps, stamp_sec)
    candidates = []
    if index < len(sim_stamps):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return 0
    nearest = min(candidates, key=lambda candidate: abs(sim_stamps[candidate] - stamp_sec))
    return nearest + 1


def draw_gt(frame, payload: dict | None) -> None:
    if not payload:
        return
    observations = list((payload or {}).get("observations") or [])
    source_height, source_width = frame.shape[:2]
    for observation in observations:
        bbox = observation.get("bbox_2d") or []
        image_size = observation.get("image_size") or [source_width, source_height]
        if len(bbox) != 4 or len(image_size) != 2:
            continue
        scale_x = float(source_width) / max(1, int(image_size[0]))
        scale_y = float(source_height) / max(1, int(image_size[1]))
        x0, y0, x1, y1 = [int(value) for value in bbox]
        start = (int(x0 * scale_x), int(y0 * scale_y))
        end = (int(x1 * scale_x), int(y1 * scale_y))
        color = (50, 80, 238) if observation.get("is_door") else (220, 70, 170) if observation.get("is_receptacle") else (210, 210, 20)
        cv2.rectangle(frame, start, end, color, 2, cv2.LINE_AA)
        label = f"{observation.get('semantic_name', 'obj')} {observation.get('instance_id', '')}"
        cv2.putText(frame, label[:48], (start[0], max(18, start[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    gt_frame = (payload or {}).get("frame_index", "-")
    draw_outlined_text(
        frame,
        f"GT visible={len(observations)} frame={gt_frame} source=realtime_gt",
        (8, source_height - 9),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--route-result", default="")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    debug_dir = (
        Path(args.debug_dir).expanduser().resolve()
        if args.debug_dir
        else scene_dir / "debug"
    )
    sim_records = load_jsonl(scene_dir / "sim_step_frames" / "manifest.jsonl")
    recorder_records = load_recorder_frames(debug_dir / "video_frames.csv")
    if not sim_records:
        raise RuntimeError("No simulator step frames found")
    if not recorder_records:
        raise RuntimeError("No recorder state frames found")
    sim_stamps = [float(record.get("stamp_sec") or 0.0) for record in sim_records]

    first_state = cv2.imread(str(recorder_records[0]["composite_frame"]), cv2.IMREAD_COLOR)
    if first_state is None:
        raise RuntimeError("Cannot read first recorder composite frame")
    output_height, output_width = first_state.shape[:2]
    panel_columns = 3 if output_width >= output_height * 2.4 else 2
    output_stem = args.output_stem or ("overview_6panel" if panel_columns == 3 else "overview_4panel")
    video_dir = debug_dir / "videos"
    output_dir = video_dir / "offline_composite_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_video = video_dir / f"{output_stem}_offline_raw.mp4"
    temp_h264 = video_dir / f"{output_stem}_offline_h264_tmp.mp4"
    final_video = video_dir / f"{output_stem}.mp4"
    sync_csv = debug_dir / "offline_video_sync.csv"
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

    sync_rows = []
    try:
        for frame_index, sim_record in enumerate(sim_records, start=1):
            stamp_sec = float(sim_record.get("stamp_sec") or 0.0)
            state_record = causal_recorder_frame(recorder_records, stamp_sec)
            state_frame = cv2.imread(str(state_record["composite_frame"]), cv2.IMREAD_COLOR)
            camera_frame = cv2.imread(str(sim_record["frame"]), cv2.IMREAD_COLOR)
            if state_frame is None or camera_frame is None:
                continue
            if state_frame.shape[1] != output_width or state_frame.shape[0] != output_height:
                state_frame = cv2.resize(state_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            camera_frame = cv2.resize(camera_frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            draw_gt(camera_frame, sim_record.get("gt_observations"))
            sim_step = int(sim_record.get("step_index", frame_index - 1)) + 1
            state_stamp = float(state_record.get("image_stamp_value") or 0.0)
            state_step = nearest_sim_step(sim_stamps, state_stamp)
            state_age_sec = max(0.0, stamp_sec - state_stamp)
            camera_label = f"STEP={sim_step:04d}"
            camera_size = cv2.getTextSize(
                camera_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )[0]
            draw_outlined_text(
                camera_frame,
                camera_label,
                (panel_width - camera_size[0] - 8, 24),
                scale=0.55,
                thickness=2,
            )
            state_frame[:panel_height, :panel_width] = camera_frame
            sync_label = f"TARGET={sim_step:04d} STATE={state_step:04d} AGE={state_age_sec:.2f}s"
            for row in range(2):
                for column in range(panel_columns):
                    if row == 0 and column == 0:
                        continue
                    x0 = column * panel_width
                    y0 = row * panel_height
                    text_size = cv2.getTextSize(sync_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                    text_x = x0 + panel_width - text_size[0] - 8
                    text_y = y0 + panel_height - 10
                    draw_outlined_text(
                        state_frame,
                        sync_label,
                        (text_x, text_y),
                        scale=0.42,
                    )
            output_path = output_dir / f"frame_{frame_index:06d}_composite.png"
            cv2.imwrite(str(output_path), state_frame)
            writer.write(state_frame)
            sync_rows.append(
                {
                    "frame_index": frame_index,
                    "sim_step": sim_step,
                    "sim_step_index": sim_step - 1,
                    "sim_stamp": f"{stamp_sec:.9f}",
                    "state_frame_index": int(state_record.get("frame_index") or 0),
                    "state_step": state_step,
                    "state_stamp": f"{state_stamp:.9f}",
                    "state_age_sec": f"{state_age_sec:.9f}",
                    "output_frame": str(output_path),
                }
            )
    finally:
        writer.release()

    with sync_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(sync_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(sync_rows)

    ffmpeg_log = video_dir / f"{output_stem}_offline_ffmpeg.log"
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
        "sim_frame_count": len(sim_records),
        "state_frame_count": len(recorder_records),
        "output_frame_count": len(sync_rows),
        "fps": float(args.fps),
        "codec": "h264",
        "video": str(final_video),
        "sync_csv": str(sync_csv),
        "route_result": str(Path(args.route_result).expanduser().resolve()) if args.route_result else "",
    }
    (debug_dir / "offline_video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
