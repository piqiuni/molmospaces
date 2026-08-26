#!/usr/bin/env python3
"""Build one exploration debug-video frame for every simulator step."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import cv2


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return sorted(records, key=lambda record: int(record.get("step_index", 0)))


def load_recorder_frames(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    for record in records:
        record["image_stamp_value"] = float(record.get("image_stamp") or 0.0)
        record["source_step_value"] = int(record.get("source_seq") or record.get("step_id") or 0)
    return sorted(records, key=lambda record: record["source_step_value"])


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def draw_panel_step_header(
    frame,
    panel_x: int,
    panel_y: int,
    panel_width: int,
    label: str,
    sim_step: int,
) -> None:
    header_width = min(panel_width - 1, 680)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + header_width, panel_y + 32),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        frame,
        f"{label}  SIM STEP={sim_step:04d}",
        (panel_x + 10, panel_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 if panel_width < 700 else 0.65,
        (20, 20, 20),
        1 if panel_width < 700 else 2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-stamp-delta-sec", type=float, default=0.05)
    args = parser.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    debug_dir = (args.debug_dir or scene_dir / "debug").expanduser().resolve()
    sim_records = load_jsonl(scene_dir / "sim_step_frames" / "manifest.jsonl")
    state_records = load_recorder_frames(debug_dir / "video_frames.csv")
    if not sim_records:
        raise RuntimeError("No simulator step frames found")
    if not state_records:
        raise RuntimeError("No recorder state frames found")

    states_by_step = {}
    duplicate_steps = []
    for record in state_records:
        source_step = int(record["source_step_value"])
        if source_step in states_by_step:
            duplicate_steps.append(source_step)
        states_by_step[source_step] = record
    if duplicate_steps:
        raise RuntimeError(f"Duplicate recorder source steps: {sorted(set(duplicate_steps))[:20]}")

    missing_steps = [
        int(record.get("step_index", index))
        for index, record in enumerate(sim_records)
        if int(record.get("step_index", index)) not in states_by_step
    ]
    if missing_steps:
        raise RuntimeError(
            f"Missing exact recorder snapshots for {len(missing_steps)} simulator steps: "
            f"{missing_steps[:20]}"
        )

    first_state = cv2.imread(str(resolve_path(state_records[0]["composite_frame"], debug_dir)), cv2.IMREAD_COLOR)
    if first_state is None:
        raise RuntimeError("Cannot read the first recorder composite frame")

    output_height, output_width = first_state.shape[:2]
    panel_width = int(state_records[0].get("panel_width") or output_width // 2)
    panel_height = int(state_records[0].get("panel_height") or output_height // 2)
    if panel_height > output_height or panel_width > output_width:
        raise RuntimeError("Recorder camera panel is larger than its composite frame")

    video_dir = debug_dir / "videos"
    frame_dir = video_dir / "full_step_composite_frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    raw_video = video_dir / "first_person_full_step_raw.mp4"
    temp_h264 = video_dir / "first_person_full_step_tmp.mp4"
    final_video = video_dir / "first_person.mp4"
    sampled_video = video_dir / "first_person_sampled.mp4"
    sync_csv = debug_dir / "full_step_video_sync.csv"
    summary_path = debug_dir / "full_step_video_summary.json"

    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, float(args.fps)),
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open full-step video writer")

    sync_rows = []
    try:
        for frame_index, sim_record in enumerate(sim_records, start=1):
            sim_step = int(sim_record.get("step_index", frame_index - 1))
            sim_stamp = float(sim_record.get("stamp_sec") or 0.0)
            state_record = states_by_step[sim_step]

            state_frame = cv2.imread(
                str(resolve_path(state_record["composite_frame"], debug_dir)),
                cv2.IMREAD_COLOR,
            )
            camera_frame = cv2.imread(
                str(resolve_path(str(sim_record["frame"]), scene_dir)),
                cv2.IMREAD_COLOR,
            )
            if state_frame is None or camera_frame is None:
                raise RuntimeError(f"Missing source image for simulator step {sim_step}")
            if state_frame.shape[:2] != (output_height, output_width):
                state_frame = cv2.resize(state_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            camera_frame = cv2.resize(camera_frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)

            state_step = int(state_record.get("step_id") or 0)
            state_stamp = float(state_record.get("image_stamp_value") or 0.0)
            stamp_delta = abs(sim_stamp - state_stamp)
            if stamp_delta > max(0.0, float(args.max_stamp_delta_sec)):
                raise RuntimeError(
                    f"Simulator/recorder stamp mismatch at step {sim_step}: {stamp_delta:.6f}s"
                )
            distance_m = float(state_record.get("distance_m") or 0.0)
            goal_count = int(state_record.get("goal_count") or 0)
            stuck_state = str(state_record.get("stuck_state") or "UNKNOWN")
            cv2.rectangle(camera_frame, (6, 6), (min(panel_width - 6, 620), 67), (255, 255, 255), -1)
            cv2.putText(
                camera_frame,
                f"FIRST PERSON  SIM STEP={sim_step:04d}",
                (16, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                camera_frame,
                f"dist={distance_m:.2f}m goal=#{goal_count:03d} stuck={stuck_state}",
                (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )
            state_frame[:panel_height, :panel_width] = camera_frame
            draw_panel_step_header(state_frame, panel_width, 0, panel_width, "OCC", sim_step)
            draw_panel_step_header(state_frame, 0, panel_height, panel_width, "GLOBAL COSTMAP", sim_step)
            draw_panel_step_header(
                state_frame,
                panel_width,
                panel_height,
                panel_width,
                "LOCAL COSTMAP",
                sim_step,
            )

            output_frame = frame_dir / f"frame_{frame_index:06d}_composite.png"
            if not cv2.imwrite(str(output_frame), state_frame):
                raise RuntimeError(f"Failed to write full-step frame {frame_index}")
            writer.write(state_frame)
            sync_rows.append(
                {
                    "frame_index": frame_index,
                    "sim_step": sim_step,
                    "sim_stamp": f"{sim_stamp:.9f}",
                    "state_frame_index": int(state_record.get("frame_index") or 0),
                    "state_step": state_step,
                    "state_stamp": f"{state_stamp:.9f}",
                    "state_age_sec": f"{stamp_delta:.9f}",
                    "match_mode": "exact_step",
                    "output_frame": str(output_frame),
                }
            )
    finally:
        writer.release()

    with sync_csv.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(sync_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(sync_rows)

    ffmpeg_log = video_dir / "first_person_full_step_ffmpeg.log"
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
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not temp_h264.exists() or temp_h264.stat().st_size <= 0:
        raise RuntimeError(f"Full-step H264 encoding failed; see {ffmpeg_log}")

    if final_video.exists() and not sampled_video.exists():
        final_video.replace(sampled_video)
    temp_h264.replace(final_video)
    raw_video.unlink(missing_ok=True)

    summary = {
        "sim_frame_count": len(sim_records),
        "state_frame_count": len(state_records),
        "output_frame_count": len(sync_rows),
        "exact_step_match_count": len(sync_rows),
        "missing_step_count": 0,
        "max_stamp_delta_sec": max(float(row["state_age_sec"]) for row in sync_rows),
        "first_sim_step": int(sim_records[0].get("step_index", 0)),
        "last_sim_step": int(sim_records[-1].get("step_index", len(sim_records) - 1)),
        "fps": float(args.fps),
        "video": str(final_video),
        "sampled_video": str(sampled_video) if sampled_video.exists() else "",
        "sync_csv": str(sync_csv),
        "frame_dir": str(frame_dir),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
