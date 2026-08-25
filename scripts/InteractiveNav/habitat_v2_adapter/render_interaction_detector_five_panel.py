#!/usr/bin/env python3
"""Render detector replay panels as an H.264 comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np


def _paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("frame_*.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--gd-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--panel-height", type=int, default=540)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--layout", choices=("five", "three"), default="five")
    args = parser.parse_args()
    groups = [_paths(args.pf_dir / "box"), _paths(args.pf_dir / "seg")]
    if args.layout == "five":
        groups.extend((_paths(args.text_dir / "box"), _paths(args.text_dir / "seg")))
    groups.append(_paths(args.gd_dir / "box"))
    sizes = {len(group) for group in groups}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise SystemExit(f"panel frame count mismatch: {[len(group) for group in groups]}")
    first = cv2.imread(str(groups[0][0]))
    aspect = first.shape[1] / first.shape[0]
    panel_width = int(round(args.panel_height * aspect / 2.0) * 2)
    output_width = panel_width * len(groups)
    output_height = args.panel_height + 36
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{output_width}x{output_height}",
        "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for index, items in enumerate(zip(*groups), 1):
        panels = []
        for path in items:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read {path}")
            panels.append(cv2.resize(image, (panel_width, args.panel_height), interpolation=cv2.INTER_AREA))
        canvas = np.concatenate(panels, axis=1)
        footer = np.full((36, output_width, 3), 20, dtype=np.uint8)
        cv2.putText(footer, f"Replay frame {index:03d}/{len(groups[0]):03d}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
        process.stdin.write(np.concatenate((canvas, footer), axis=0).tobytes())
    process.stdin.close()
    code = process.wait()
    if code:
        raise SystemExit(f"ffmpeg exited with {code}")
    print(args.output)


if __name__ == "__main__":
    main()
