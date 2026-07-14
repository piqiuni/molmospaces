#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2


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
    for row in rows:
        row["image_stamp_value"] = float(row.get("image_stamp") or 0.0)
    return sorted(rows, key=lambda row: row["image_stamp_value"])


def causal_recorder_frame(records: list[dict], stamp_sec: float) -> dict | None:
    selected = None
    for record in records:
        if record["image_stamp_value"] > stamp_sec:
            break
        selected = record
    if selected is None and records:
        selected = records[0]
    return selected


def draw_gt(frame, payload: dict | None) -> None:
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
    cv2.rectangle(frame, (0, max(0, source_height - 28)), (min(source_width - 1, 470), source_height - 1), (255, 255, 255), -1)
    gt_frame = (payload or {}).get("frame_index", "-")
    cv2.putText(frame, f"GT visible={len(observations)} frame={gt_frame} source=realtime_gt", (8, source_height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    sim_records = load_jsonl(scene_dir / "sim_step_frames" / "manifest.jsonl")
    recorder_records = load_recorder_frames(scene_dir / "video_frames.csv")
    if not sim_records:
        raise RuntimeError("No simulator step frames found")
    if not recorder_records:
        raise RuntimeError("No recorder state frames found")

    output_dir = scene_dir / "videos" / "offline_composite_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_video = scene_dir / "videos" / "overview_6panel_offline_raw.mp4"
    final_video = scene_dir / "videos" / "overview_6panel.mp4"
    sync_csv = scene_dir / "offline_video_sync.csv"

    first_state = cv2.imread(str(recorder_records[0]["composite_frame"]), cv2.IMREAD_COLOR)
    if first_state is None:
        raise RuntimeError("Cannot read first recorder composite frame")
    output_height, output_width = first_state.shape[:2]
    panel_width = output_width // 3
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
            cv2.putText(camera_frame, f"SIM STEP={int(sim_record.get('step_index', frame_index - 1)):04d} STAMP={stamp_sec:.3f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)
            state_frame[:panel_height, :panel_width] = camera_frame
            output_path = output_dir / f"frame_{frame_index:06d}_composite.png"
            cv2.imwrite(str(output_path), state_frame)
            writer.write(state_frame)
            state_stamp = float(state_record.get("image_stamp_value") or 0.0)
            sync_rows.append(
                {
                    "frame_index": frame_index,
                    "sim_step": int(sim_record.get("step_index", frame_index - 1)),
                    "sim_stamp": f"{stamp_sec:.9f}",
                    "state_frame_index": int(state_record.get("frame_index") or 0),
                    "state_stamp": f"{state_stamp:.9f}",
                    "state_age_sec": f"{max(0.0, stamp_sec - state_stamp):.9f}",
                    "output_frame": str(output_path),
                }
            )
    finally:
        writer.release()

    with sync_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(sync_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(sync_rows)

    ffmpeg_log = scene_dir / "videos" / "overview_6panel_offline_ffmpeg.log"
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
        str(final_video),
    ]
    with ffmpeg_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raw_video.replace(final_video)
    summary = {
        "sim_frame_count": len(sim_records),
        "state_frame_count": len(recorder_records),
        "output_frame_count": len(sync_rows),
        "fps": float(args.fps),
        "video": str(final_video),
        "sync_csv": str(sync_csv),
    }
    (scene_dir / "offline_video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
