#!/usr/bin/env python3
"""Offline export of exact recorded six-panel JPEGs and native first-person RGB.

No ROS, robot, model service or live web server is needed. Frames are held
until their next recorded receipt; a future frame is never shown early.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict, deque
import json
import math
from pathlib import Path

import cv2
import numpy as np


def rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_at(times, timestamp):
    return bisect_right(times, timestamp) - 1


def export_stream(session, records, output, fps, duration, step_overlay=False):
    records = sorted(records, key=lambda row: float(row["session_time_s"]))
    if not records:
        raise ValueError(f"no recorded frames for {output.name}")
    times = [float(row["session_time_s"]) for row in records]
    def read(index):
        image = cv2.imread(str(session / records[index]["image"]))
        if image is None:
            raise ValueError(f"missing/unreadable image: {records[index]['image']}")
        if step_overlay:
            image = cv2.copyMakeBorder(image, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            label = f"STEP {records[index]['step_index']}   |   recorded t={times[index]:.2f}s"
            cv2.putText(image, label, (20, 41), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return image
    sample = read(0)
    height, width = sample.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {output}")
    previous = -1
    image = np.zeros_like(sample)
    count = max(1, math.ceil(duration * fps) + 1)
    try:
        for frame in range(count):
            index = frame_at(times, frame / fps)
            if index != previous and index >= 0:
                image = read(index)
                if image.shape != sample.shape:
                    raise ValueError("image resolution changed during recording")
                previous = index
            writer.write(image)
    finally:
        writer.release()
    return {"file": output.name, "frames": count, "source_frames": len(records),
            "width": width, "height": height, "fps": fps}


def build(session, output, fps=10.0, step_overlay=False, overview_only=False):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    info = json.loads((session / "session.json").read_text())
    if info.get("status") not in {"complete", "degraded"}:
        raise ValueError("stop recording before offline export")
    panels = [row for row in rows(session / "raw/panels/manifest.jsonl")
              if row.get("panel_index") == 0]
    camera = [{**row, "image": row["files"]["rgb"]}
              for row in rows(session / "raw/camera/manifest.jsonl") if row.get("files", {}).get("rgb")]
    if not panels or not camera:
        raise ValueError("requires a new recording containing overview panel 0 and raw RGB")
    if step_overlay:
        boundaries = rows(session / "raw/step_boundaries.jsonl")
        steps = defaultdict(deque)
        for row in boundaries:
            steps[row["frame_seq"]].append(row["step_index"])
        for panel in panels:
            # One camera frame can underlie several navigation/render steps.
            # Consume matching receipts in order instead of overwriting them.
            panel["step_index"] = steps[panel["frame_seq"]].popleft()
    duration = max(float(info.get("duration_s", 0)),
                   max(float(row["session_time_s"]) for row in panels + camera))
    # Never overwrite existing exports or the immutable source session.
    output.mkdir(parents=True, exist_ok=False)
    result = {"session": str(session), "source_status": info["status"],
              "source_stats": info.get("stats", {}), "duration_s": duration,
              "alignment": "latest receipt at or before output timestamp", "videos": []}
    streams = [("overview_6panel.mp4", panels)]
    if not overview_only:
        streams.append(("first_person.mp4", camera))
    result["step_overlay"] = step_overlay
    for name, records in streams:
        result["videos"].append(export_stream(session, records, output / name, fps, duration,
                                             step_overlay=step_overlay and name == "overview_6panel.mp4"))
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--step-overlay", action="store_true", help="add a header with the recorded step, matched by source frame_seq")
    parser.add_argument("--overview-only", action="store_true")
    args = parser.parse_args()
    session = args.session.resolve()
    print(json.dumps(build(session, args.output or session / "derived/six-panel", args.fps,
                           args.step_overlay, args.overview_only), indent=2))


if __name__ == "__main__":
    main()
