#!/usr/bin/env python3
"""Render raw PF segmentation beside optimized PF box output."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    raw = sorted((args.raw_dir / "seg").glob("frame_*.jpg"))
    optimized = sorted((args.optimized_dir / "box").glob("frame_*.jpg"))
    if len(raw) != len(optimized) or not raw:
        raise SystemExit(f"frame mismatch raw={len(raw)} optimized={len(optimized)}")
    first = cv2.imread(str(raw[0]))
    width = int(round(args.height * first.shape[1] / first.shape[0] / 2) * 2)
    out_w, out_h = width * 2, args.height + 40
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for raw_path, opt_path in zip(raw, optimized):
        left = cv2.resize(cv2.imread(str(raw_path)), (width, args.height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(cv2.imread(str(opt_path)), (width, args.height), interpolation=cv2.INTER_AREA)
        footer = np.full((40, out_w, 3), 16, dtype=np.uint8)
        cv2.putText(footer, "LEFT: raw PF instance masks     RIGHT: optimized PF detections", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
        process.stdin.write(np.concatenate((np.concatenate((left, right), axis=1), footer), axis=0).tobytes())
    process.stdin.close()
    if process.wait():
        raise SystemExit("ffmpeg failed")
    print(args.output)


if __name__ == "__main__":
    main()
