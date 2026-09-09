#!/usr/bin/env python3
"""Prepare two deterministic 90-frame replay inputs without altering sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--habitat-frames", type=Path, required=True)
    parser.add_argument("--mobile-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=90)
    args = parser.parse_args()
    habitat_out = args.output_dir / "habitat90"
    mobile_out = args.output_dir / "mobile90"
    habitat_out.mkdir(parents=True, exist_ok=True)
    mobile_out.mkdir(parents=True, exist_ok=True)
    habitat = sorted(args.habitat_frames.glob("*.png"))
    if len(habitat) != args.count:
        raise SystemExit(f"expected {args.count} Habitat PNGs, found {len(habitat)}")
    rows = []
    for index, source in enumerate(habitat, 1):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {source}")
        target = habitat_out / f"frame_{index:04d}.jpg"
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 96])
        rows.append({"sequence": "habitat90", "index": index, "source": str(source), "target": str(target)})

    capture = cv2.VideoCapture(str(args.mobile_video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {args.mobile_video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count < args.count:
        raise RuntimeError(f"mobile video has only {frame_count} frames")
    indices = np.linspace(0, frame_count - 1, args.count).round().astype(int)
    for index, source_index in enumerate(indices, 1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode mobile frame {source_index}")
        target = mobile_out / f"frame_{index:04d}.jpg"
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 96])
        rows.append({
            "sequence": "mobile90", "index": index,
            "source": str(args.mobile_video), "source_frame": int(source_index),
            "source_time_s": float(source_index / fps), "target": str(target),
        })
    capture.release()
    manifest = {
        "count_per_sequence": args.count,
        "mobile_frame_count": frame_count,
        "mobile_fps": fps,
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"habitat": len(habitat), "mobile": len(indices), "mobile_fps": fps, "mobile_frames": frame_count}, indent=2))


if __name__ == "__main__":
    main()
