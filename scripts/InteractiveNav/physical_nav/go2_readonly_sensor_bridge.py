#!/usr/bin/env python3
"""Go2-side D435i and state publisher.

This process intentionally imports only Unitree *subscriber* types.  It does
not construct SportClient, ObstaclesAvoidClient, a publisher, or any motion
API.  The only outbound channel is a WebSocket carrying sensor frames and
read-only telemetry to the policy machine.

The script is designed to run on the Go2 Jetson.  ``--dry-run`` produces a
synthetic RGB-D stream for protocol and web UI tests on a development machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import threading
import time
from typing import Any

import websocket

from physical_protocol import hello_packet, image_packet, telemetry_packet


class ReadOnlyState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.telemetry: dict[str, Any] = {}
        self.seq = 0

    def update(self, value: dict[str, Any]) -> None:
        with self.lock:
            self.telemetry = value

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.telemetry)


def _unitree_state_reader(state: ReadOnlyState, interface: str) -> None:
    """Subscribe to state topics without loading any command client."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
    except ImportError as exc:
        print(f"warning: Unitree SDK unavailable; telemetry disabled: {exc}", flush=True)
        return

    try:
        ChannelFactoryInitialize(0, interface)
        pose_lock = threading.Lock()
        latest: dict[str, Any] = {}

        def on_sport(msg: Any) -> None:
            try:
                q = [float(v) for v in msg.imu_state.quaternion]
                w, x, y, z = q
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                value = {
                    "received_at": time.time(),
                    "position": [float(v) for v in msg.position],
                    "velocity": [float(v) for v in msg.velocity],
                    "yaw_speed": float(msg.yaw_speed),
                    "mode": int(msg.mode),
                    "gait_type": int(msg.gait_type),
                    "body_height": float(msg.body_height),
                    "range_obstacle": [float(v) for v in msg.range_obstacle],
                    "foot_force": [int(v) for v in msg.foot_force],
                    "imu": {
                        "quaternion": q,
                        "gyroscope": [float(v) for v in msg.imu_state.gyroscope],
                        "accelerometer": [float(v) for v in msg.imu_state.accelerometer],
                        "rpy": [float(v) for v in msg.imu_state.rpy],
                        "temperature": int(msg.imu_state.temperature),
                    },
                    "yaw": yaw,
                }
                with pose_lock:
                    latest.update(value)
                    state.update(dict(latest))
            except Exception as exc:  # SDK message layouts differ by release.
                print(f"warning: could not decode rt/sportmodestate: {exc}", flush=True)

        def on_low(msg: Any) -> None:
            try:
                bms = msg.bms_state
                value = {
                    "received_at": time.time(),
                    "soc": int(bms.soc),
                    "voltage": float(msg.power_v),
                    "current": float(msg.power_a),
                    "power": float(msg.power_v) * float(msg.power_a),
                    "temperature_ntc1": int(msg.temperature_ntc1),
                    "temperature_ntc2": int(msg.temperature_ntc2),
                }
                with pose_lock:
                    latest["battery"] = value
                    state.update(dict(latest))
            except Exception as exc:
                print(f"warning: could not decode rt/lowstate: {exc}", flush=True)

        # Subscribers are the only Unitree objects constructed in this file.
        subscribers = [ChannelSubscriber("rt/sportmodestate", SportModeState_), ChannelSubscriber("rt/lowstate", LowState_)]
        subscribers[0].Init(on_sport, 10); subscribers[1].Init(on_low, 10)
        print("read-only Unitree state subscribers started", flush=True)
        while True:
            time.sleep(1.0)
    except Exception as exc:
        print(f"warning: state subscribers stopped: {exc}", flush=True)


class D435iSource:
    def __init__(self, width: int, height: int, fps: int) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        c = color_profile.get_intrinsics()
        self.intrinsics = {"fx": c.fx, "fy": c.fy, "cx": c.ppx, "cy": c.ppy, "width": c.width, "height": c.height}
        self.camera_frame = f"{color_profile.stream_name()}_frame"
        print(f"D435i started {c.width}x{c.height}@{fps}, depth_scale={self.depth_scale}", flush=True)

    def read(self) -> tuple[Any, Any, float]:
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("D435i returned an incomplete RGB-D frame")
        stamp = time.time()
        sync_ms = abs(float(color.get_timestamp()) - float(depth.get_timestamp()))
        import numpy as np
        return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), sync_ms

    def close(self) -> None:
        self.pipeline.stop()


def _encode(rgb: Any, depth: Any) -> tuple[bytes, bytes]:
    import cv2
    ok_rgb, rgb_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 82])
    ok_depth, depth_buf = cv2.imencode(".png", depth)
    if not ok_rgb or not ok_depth:
        raise RuntimeError("failed to encode D435i frame")
    return bytes(rgb_buf), bytes(depth_buf)


def _synthetic_frame(width: int, height: int) -> tuple[Any, Any, float, dict[str, float]]:
    import numpy as np
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 1] = 32
    rgb[height // 4 : height * 3 // 4, width // 3 : width * 2 // 3] = (60, 150, 230)
    depth = np.full((height, width), 1500, dtype=np.uint16)
    depth[height // 4 : height * 3 // 4, width // 3 : width * 2 // 3] = 1000
    intr = {"fx": width * 0.9, "fy": width * 0.9, "cx": width / 2, "cy": height / 2, "width": width, "height": height}
    return rgb, depth, 0.0, intr


async def publish(args: argparse.Namespace) -> None:
    state = ReadOnlyState()
    threading.Thread(target=_unitree_state_reader, args=(state, args.interface), daemon=True).start()
    source = None if args.dry_run else D435iSource(args.width, args.height, args.fps)
    seq = 0
    telemetry_seq = 0
    while True:
        try:
            ws = websocket.create_connection(args.url, timeout=args.connect_timeout, enable_multithread=True)
            ws.send(json.dumps(hello_packet(host=socket.gethostname(), streams={"camera": "d435i", "fps": args.fps})))
            ws.settimeout(0.0)
            last_telemetry = 0.0
            print(f"connected to policy WebSocket {args.url}", flush=True)
            while True:
                # The policy gateway acknowledges packets. Drain those small
                # messages so a long-running sensor stream cannot fill the TCP
                # receive window; no command is ever read or acted upon here.
                try:
                    while ws.recv():
                        pass
                except (websocket.WebSocketTimeoutException, websocket.WebSocketConnectionClosedException, OSError):
                    pass
                started = time.monotonic()
                if source is None:
                    rgb, depth, sync_ms, intr = _synthetic_frame(args.width, args.height)
                else:
                    rgb, depth, sync_ms = source.read()
                    intr = source.intrinsics
                rgb_jpeg, depth_png = _encode(rgb, depth)
                seq += 1
                ws.send(json.dumps(image_packet(seq=seq, stamp=time.time(), rgb_jpeg=rgb_jpeg, depth_png=depth_png,
                                                 width=rgb.shape[1], height=rgb.shape[0], camera_frame=args.camera_frame,
                                                 depth_scale=(args.depth_scale if source is None else source.depth_scale),
                                                 intrinsics=intr, color_depth_sync_ms=sync_ms), separators=(",", ":")))
                now = time.monotonic()
                if now - last_telemetry >= args.telemetry_period:
                    telemetry_seq += 1
                    ws.send(json.dumps(telemetry_packet(seq=telemetry_seq, telemetry=state.snapshot()), separators=(",", ":")))
                    last_telemetry = now
                elapsed = time.monotonic() - started
                await asyncio.sleep(max(0.0, 1.0 / args.fps - elapsed))
        except Exception as exc:
            print(f"sensor link disconnected: {exc}; retrying in {args.reconnect_s}s", flush=True)
            await asyncio.sleep(args.reconnect_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:12334")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--telemetry-period", type=float, default=0.2)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--reconnect-s", type=float, default=2.0)
    parser.add_argument("--camera-frame", default="d435i_color_optical_frame")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(publish(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
