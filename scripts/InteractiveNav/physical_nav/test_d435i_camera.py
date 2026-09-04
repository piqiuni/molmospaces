#!/usr/bin/env python3
"""Inspect and record Intel RealSense D435i RGB/depth modes.

Run this on the Unitree computer (where ``pyrealsense2`` is installed).  The
script deliberately keeps native depth frames when ``--align none`` is used;
alignment is a resampling operation and cannot preserve pixels outside the
target camera's field of view.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def _intrinsics(value: Any) -> dict[str, Any]:
    return {
        "width": int(value.width),
        "height": int(value.height),
        "fx": float(value.fx),
        "fy": float(value.fy),
        "cx": float(value.ppx),
        "cy": float(value.ppy),
        "model": str(getattr(value, "model", "")),
        "coeffs": [float(x) for x in getattr(value, "coeffs", ())],
    }


def _fov(i: dict[str, Any]) -> dict[str, float]:
    # Intrinsics describe the calibrated active image rectangle.  This is the
    # correct FOV to compare between modes; do not infer it from aspect ratio.
    h = math.degrees(2.0 * math.atan(i["width"] / (2.0 * i["fx"])))
    v = math.degrees(2.0 * math.atan(i["height"] / (2.0 * i["fy"])))
    d = math.degrees(
        2.0
        * math.atan(
            math.sqrt(i["width"] ** 2 + i["height"] ** 2)
            / (2.0 * math.sqrt(i["fx"] ** 2 + i["fy"] ** 2))
        )
    )
    return {"horizontal_deg": h, "vertical_deg": v, "diagonal_deg": d}


def _mode_key(profile: Any) -> tuple[int, int, int, str]:
    return (int(profile.width()), int(profile.height()), int(profile.fps()), str(profile.format()))


def list_modes(rs: Any) -> None:
    devices = rs.context().query_devices()
    if len(devices) == 0:
        raise SystemExit("No RealSense device found")
    device = devices[0]
    print(f"device: {device.get_info(rs.camera_info.name)}")
    for sensor in device.query_sensors():
        print(f"\n{sensor.get_info(rs.camera_info.name)}")
        seen: set[tuple[int, int, int, str]] = set()
        for profile in sensor.get_stream_profiles():
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            key = _mode_key(video)
            if key not in seen:
                seen.add(key)
                print(f"  {key[0]}x{key[1]} @ {key[2]} fps {key[3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-modes", action="store_true", help="list device-supported video profiles and exit")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--color-width", type=int)
    parser.add_argument("--color-height", type=int)
    parser.add_argument("--color-fps", type=int)
    parser.add_argument("--depth-width", type=int)
    parser.add_argument("--depth-height", type=int)
    parser.add_argument("--depth-fps", type=int)
    parser.add_argument("--frames", type=int, default=30, help="number of frames to save")
    parser.add_argument("--output", type=Path, default=Path("d435i_test"))
    parser.add_argument("--align", choices=("none", "color", "depth"), default="none")
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup must be non-negative")

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(f"Missing camera dependency ({exc}); install/use pyrealsense2 on unitree") from exc

    pipeline = rs.pipeline()
    if args.list_modes:
        list_modes(rs)
        return

    color_width = args.color_width or args.width
    color_height = args.color_height or args.height
    color_fps = args.color_fps or args.fps
    depth_width = args.depth_width or args.width
    depth_height = args.depth_height or args.height
    depth_fps = args.depth_fps or args.fps
    config = rs.config()
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
    try:
        profile = pipeline.start(config)
    except Exception as exc:
        raise SystemExit(f"Could not start color {color_width}x{color_height}@{color_fps} + depth {depth_width}x{depth_height}@{depth_fps}: {exc}\nRun --list-modes first.") from exc

    align = None if args.align == "none" else rs.align(getattr(rs.stream, args.align))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "requested": {
            "color": {"width": color_width, "height": color_height, "fps": color_fps},
            "depth": {"width": depth_width, "height": depth_height, "fps": depth_fps},
        },
        "align": args.align,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {},
    }
    device = profile.get_device()
    for info in (rs.camera_info.name, rs.camera_info.serial_number, rs.camera_info.firmware_version):
        try:
            metadata["device"][str(info)] = device.get_info(info)
        except Exception:
            pass
    depth_sensor = device.first_depth_sensor()
    metadata["depth_scale_m_per_unit"] = float(depth_sensor.get_depth_scale())
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    metadata["depth"] = {"intrinsics": _intrinsics(depth_profile.get_intrinsics())}
    metadata["color"] = {"intrinsics": _intrinsics(color_profile.get_intrinsics())}
    metadata["depth"]["fov_deg"] = _fov(metadata["depth"]["intrinsics"])
    metadata["color"]["fov_deg"] = _fov(metadata["color"]["intrinsics"])
    try:
        ext = depth_profile.get_extrinsics_to(color_profile)
        metadata["depth_to_color_extrinsics"] = {"rotation": list(ext.rotation), "translation_m": list(ext.translation)}
    except Exception as exc:
        metadata["depth_to_color_extrinsics_error"] = str(exc)
    with (args.output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps({"depth_fov_deg": metadata["depth"]["fov_deg"], "color_fov_deg": metadata["color"]["fov_deg"], "align": args.align}, indent=2))
    print(f"saving {args.frames} frames to {args.output.resolve()}")
    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        for index in range(args.frames):
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                print(f"warning: incomplete frame {index}; skipping")
                continue
            rgb = np.asanyarray(color.get_data())
            depth_u16 = np.asanyarray(depth.get_data())
            cv2.imwrite(str(args.output / f"rgb_{index:06d}.jpg"), rgb)
            cv2.imwrite(str(args.output / f"depth_{index:06d}.png"), depth_u16)
            # A viewable false-colour preview is useful without destroying the
            # lossless uint16 depth output used for measurements.
            preview = cv2.applyColorMap(cv2.convertScaleAbs(depth_u16, alpha=0.08), cv2.COLORMAP_JET)
            cv2.imwrite(str(args.output / f"depth_preview_{index:06d}.jpg"), preview)
            # Keep a directly inspectable side-by-side image.  Native RGB and
            # depth streams can have different sizes, so only the preview is
            # resized for display; the saved depth PNG remains untouched.
            display_depth = cv2.resize(preview, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined = np.concatenate((rgb, display_depth), axis=1)
            cv2.imwrite(str(args.output / f"rgb_depth_combined_{index:06d}.jpg"), combined)
            if args.align == "depth":
                # ``align depth`` makes color and depth share the native depth
                # canvas.  Blend them to expose the optical-center parallax;
                # pixels without valid depth remain visible in the RGB layer.
                overlay = cv2.addWeighted(display_depth, 0.5, rgb, 0.5, 0.0)
                cv2.imwrite(str(args.output / f"rgb_on_depth_overlay_{index:06d}.jpg"), overlay)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
