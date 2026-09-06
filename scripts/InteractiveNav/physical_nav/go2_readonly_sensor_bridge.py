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

from physical_protocol import encode_wire_packet, hello_packet, image_packet, telemetry_packet


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


def _field_sequence(message: Any, name: str, *, cast: Any = float) -> list[Any]:
    """Read optional SDK sequences without dropping the whole pose packet."""
    try:
        values = getattr(message, name, None)
        return [] if values is None else [cast(value) for value in values]
    except (TypeError, ValueError):
        return []


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
                imu = getattr(msg, "imu_state")
                q = _field_sequence(imu, "quaternion")
                if len(q) != 4:
                    raise ValueError("IMU quaternion is missing or does not contain four values")
                w, x, y, z = q
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                position = (_field_sequence(msg, "position") + [0.0, 0.0, 0.0])[:3]
                velocity = (_field_sequence(msg, "velocity") + [0.0, 0.0, 0.0])[:3]
                value = {
                    "received_at": time.time(),
                    "error_code": int(getattr(msg, "error_code", 0)),
                    "position": position,
                    "velocity": velocity,
                    "yaw_speed": float(getattr(msg, "yaw_speed", 0.0)),
                    "mode": int(getattr(msg, "mode", 0)),
                    "progress": float(getattr(msg, "progress", 0.0)),
                    "gait_type": int(getattr(msg, "gait_type", 0)),
                    "foot_raise_height": float(getattr(msg, "foot_raise_height", 0.0)),
                    "body_height": float(getattr(msg, "body_height", 0.0)),
                    "range_obstacle": _field_sequence(msg, "range_obstacle"),
                    "foot_force": _field_sequence(msg, "foot_force", cast=int),
                    "imu": {
                        "quaternion": q,
                        "gyroscope": _field_sequence(imu, "gyroscope"),
                        "accelerometer": _field_sequence(imu, "accelerometer"),
                        "rpy": _field_sequence(imu, "rpy"),
                        "temperature": int(getattr(imu, "temperature", 0)),
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
                    "soc": int(getattr(bms, "soc", 0)),
                    "bms_current": int(getattr(bms, "current", 0)),
                    "cycle": int(getattr(bms, "cycle", 0)),
                    "cell_vol": [int(v) for v in getattr(bms, "cell_vol", [])],
                    "bq_ntc": [int(v) for v in getattr(bms, "bq_ntc", [])],
                    "mcu_ntc": [int(v) for v in getattr(bms, "mcu_ntc", [])],
                    "voltage": float(getattr(msg, "power_v", 0.0)),
                    "current": float(getattr(msg, "power_a", 0.0)),
                    "power": float(getattr(msg, "power_v", 0.0)) * float(getattr(msg, "power_a", 0.0)),
                    "temperature_ntc1": int(getattr(msg, "temperature_ntc1", 0)),
                    "temperature_ntc2": int(getattr(msg, "temperature_ntc2", 0)),
                    "fan_frequency": [int(v) for v in getattr(msg, "fan_frequency", [])],
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
    def __init__(self, width: int, height: int, fps: int, enable_motion: bool = False,
                 color_width: int | None = None, color_height: int | None = None,
                 color_fps: int | None = None, align_to: str = "depth") -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        depth_width, depth_height, depth_fps = width, height, fps
        color_width, color_height, color_fps = color_width or width, color_height or height, color_fps or fps
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
        self.motion_enabled = False
        try:
            # D435i exposes gyro/accelerometer streams, but not a standalone
            # visual-odometry pose stream.  Keep these measurements alongside
            # the Go2 body quaternion so the policy host can use dynamic tilt.
            if enable_motion:
                self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
                self.motion_enabled = True
        except Exception as exc:
            print(f"D435i motion streams unavailable: {exc}", flush=True)
        if enable_motion:
            # After a previous bridge restart the D435i motion endpoint can
            # remain in a stale state on the Jetson USB2 controller.  Reset
            # once before resolving the combined RGB-D+IMU requests.
            try:
                devices = self.rs.context().query_devices()
                if len(devices):
                    devices[0].hardware_reset()
                    time.sleep(3.0)
            except Exception as exc:
                print(f"D435i hardware reset skipped: {exc}", flush=True)
        self.profile = self.pipeline.start(self.config)
        if align_to not in ("none", "color", "depth"):
            raise ValueError("align_to must be none, color, or depth")
        self.align = None if align_to == "none" else rs.align(getattr(rs.stream, align_to))
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        c, d = color_profile.get_intrinsics(), depth_profile.get_intrinsics()
        self.intrinsics = {
            "fx": c.fx,
            "fy": c.fy,
            "cx": c.ppx,
            "cy": c.ppy,
            "width": c.width,
            "height": c.height,
            "distortion_model": str(getattr(c, "model", "")),
            "distortion": [float(v) for v in getattr(c, "coeffs", [])],
        }
        self.depth_intrinsics = {"fx": d.fx, "fy": d.fy, "cx": d.ppx, "cy": d.ppy,
                                 "width": d.width, "height": d.height, "distortion_model": str(getattr(d, "model", "")),
                                 "distortion": [float(v) for v in getattr(d, "coeffs", [])]}
        if align_to == "color":
            self.depth_intrinsics = dict(self.intrinsics)
        ext = depth_profile.get_extrinsics_to(color_profile)
        self.depth_to_color_extrinsics = {"rotation": [float(v) for v in ext.rotation], "translation": [float(v) for v in ext.translation]}
        target_stream = rs.stream.depth if align_to == "depth" else rs.stream.color
        self.camera_frame = f"{self.profile.get_stream(target_stream).stream_name()}_frame"
        self.latest_motion: dict[str, Any] = {}
        self._imu_rpy = [0.0, 0.0, 0.0]
        self._imu_reference_rpy = None
        self._imu_calibration_samples = 0
        self._imu_calibration_sum = [0.0, 0.0, 0.0]
        self._imu_last_gyro_ts = None
        print(f"D435i started color {c.width}x{c.height}@{color_fps} + depth {d.width}x{d.height}@{depth_fps}, align={align_to}, depth_scale={self.depth_scale}", flush=True)

    def read(self) -> tuple[Any, Any, float]:
        deadline = time.monotonic() + 0.5
        raw_frames = None
        while time.monotonic() < deadline:
            try:
                candidate = self.pipeline.wait_for_frames(100)
            except Exception:
                time.sleep(0.001)
                continue
            if self.motion_enabled:
                motion: dict[str, Any] = {"source": "d435i_imu", "received_at": time.time()}
                for stream_name, key, timestamp_key in ((self.rs.stream.gyro, "gyroscope", "gyro_timestamp_ms"), (self.rs.stream.accel, "accelerometer", "accel_timestamp_ms")):
                    try:
                        frame = candidate.first_or_default(stream_name)
                        if frame:
                            value = frame.as_motion_frame().get_motion_data()
                            motion[key] = [float(value.x), float(value.y), float(value.z)]
                            motion[timestamp_key] = float(frame.get_timestamp())
                    except Exception:
                        pass
                if "gyroscope" in motion or "accelerometer" in motion:
                    self._update_imu_orientation(motion)
                    self.latest_motion = motion
            if candidate.get_color_frame() and candidate.get_depth_frame():
                raw_frames = candidate
                break
        if raw_frames is None:
            raise RuntimeError("D435i RGB-D frame poll timeout")
        # Keep the complete native color image.  On D435i, retrieving color
        # from an ``align(depth)`` frameset can return a depth-sized canvas
        # with black padding/cropping.  Only depth should come from the
        # aligned frameset; RGB remains the original 1280x720 stream.
        color = raw_frames.get_color_frame()
        if self.motion_enabled:
            motion: dict[str, Any] = {"source": "d435i_imu", "received_at": time.time()}
            for stream_name, key, timestamp_key in ((self.rs.stream.gyro, "gyroscope", "gyro_timestamp_ms"), (self.rs.stream.accel, "accelerometer", "accel_timestamp_ms")):
                try:
                    frame = raw_frames.first_or_default(stream_name)
                    if frame:
                        value = frame.as_motion_frame().get_motion_data()
                        motion[key] = [float(value.x), float(value.y), float(value.z)]
                        motion[timestamp_key] = float(frame.get_timestamp())
                except Exception:
                    pass
            if "gyroscope" in motion or "accelerometer" in motion:
                self._update_imu_orientation(motion)
                self.latest_motion = motion
        frames = self.align.process(raw_frames) if self.align is not None else raw_frames
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("D435i returned an incomplete RGB-D frame")
        stamp = time.time()
        sync_ms = abs(float(color.get_timestamp()) - float(depth.get_timestamp()))
        import numpy as np
        return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), sync_ms

    def _update_imu_orientation(self, motion: dict[str, Any]) -> None:
        """Estimate camera roll/pitch from gravity and yaw by gyro integration.

        D435i motion frames do not provide an absolute quaternion.  The
        accelerometer stabilizes roll/pitch while the gyro supplies short-term
        dynamics; yaw is intentionally relative and reset on process start.
        """
        accel = motion.get("accelerometer")
        gyro = motion.get("gyroscope")
        if accel and len(accel) >= 3:
            ax, ay, az = (float(v) for v in accel[:3])
            norm = max((ax * ax + ay * ay + az * az) ** 0.5, 1e-6)
            ax, ay, az = ax / norm, ay / norm, az / norm
            gravity_roll = math.atan2(ay, az)
            gravity_pitch = math.atan2(-ax, max((ay * ay + az * az) ** 0.5, 1e-6))
            alpha = 0.98
            self._imu_rpy[0] = alpha * self._imu_rpy[0] + (1.0 - alpha) * gravity_roll
            self._imu_rpy[1] = alpha * self._imu_rpy[1] + (1.0 - alpha) * gravity_pitch
        if gyro and len(gyro) >= 3:
            ts = motion.get("gyro_timestamp_ms")
            if ts is not None and self._imu_last_gyro_ts is not None:
                dt = min(max((float(ts) - self._imu_last_gyro_ts) * 1e-3, 0.0), 0.05)
                self._imu_rpy[2] += float(gyro[2]) * dt
            if ts is not None:
                self._imu_last_gyro_ts = float(ts)
        r, p, y = self._imu_rpy
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        motion["rpy"] = [r, p, y]
        if self._imu_reference_rpy is None and accel:
            self._imu_calibration_samples += 1
            if self._imu_calibration_samples > 150:
                for index, value in enumerate((r, p, y)):
                    self._imu_calibration_sum[index] += value
            if self._imu_calibration_samples >= 250:
                self._imu_reference_rpy = [
                    value / max(self._imu_calibration_samples - 150, 1)
                    for value in self._imu_calibration_sum
                ]
        reference = self._imu_reference_rpy or [0.0, 0.0, 0.0]
        motion["imu_calibrating"] = self._imu_reference_rpy is None
        motion["correction_rpy"] = [0.0, 0.0, 0.0] if self._imu_reference_rpy is None else [r - reference[0], p - reference[1], y - reference[2]]
        motion["quaternion"] = [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]

    def close(self) -> None:
        self.pipeline.stop()


def _encode(
    rgb: Any,
    depth: Any,
    *,
    depth_png_compression: int = 4,
) -> tuple[bytes, bytes]:
    import cv2
    ok_rgb, rgb_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 82])
    # OpenCV's implicit PNG settings produce unusually large 16-bit D435
    # depth frames.  An explicit moderate compression level is still exactly
    # lossless, but nearly halves the bytes sent through the Go2 Wi-Fi link.
    ok_depth, depth_buf = cv2.imencode(
        ".png",
        depth,
        [cv2.IMWRITE_PNG_COMPRESSION, int(depth_png_compression)],
    )
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
    source = None if args.dry_run else D435iSource(
        args.depth_width, args.depth_height, args.depth_fps, enable_motion=args.enable_camera_imu,
        color_width=args.color_width, color_height=args.color_height, color_fps=args.color_fps,
        align_to=args.align_to,
    )
    # D435 capture must never wait for TCP/WebSocket writes.  When networking
    # slows down, keep draining the hardware stream and overwrite the pending
    # encoded frame.  The policy host only needs the newest observation.
    frame_lock = threading.Lock()
    frame_ready = threading.Event()
    latest_frame: dict[str, Any] = {}

    def capture_loop() -> None:
        seq = 0
        while True:
            started = time.monotonic()
            try:
                if source is None:
                    rgb, depth, sync_ms, intr = _synthetic_frame(args.width, args.height)
                else:
                    rgb, depth, sync_ms = source.read()
                    intr = source.intrinsics
                capture_stamp = time.time()
                rgb_jpeg, depth_png = _encode(
                    rgb,
                    depth,
                    depth_png_compression=args.depth_png_compression,
                )
                seq += 1
                frame = {
                    "seq": seq,
                    "stamp": capture_stamp,
                    "rgb_jpeg": rgb_jpeg,
                    "depth_png": depth_png,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "intrinsics": intr,
                    "rgb_intrinsics": source.intrinsics if source is not None else intr,
                    "depth_intrinsics": source.depth_intrinsics if source is not None else intr,
                    "depth_to_color_extrinsics": source.depth_to_color_extrinsics if source is not None else {},
                    "sync_ms": sync_ms,
                    "depth_scale": args.depth_scale if source is None else source.depth_scale,
                    "camera_imu": dict(source.latest_motion) if source is not None else {},
                }
                with frame_lock:
                    latest_frame.clear()
                    latest_frame.update(frame)
                    frame_ready.set()
            except Exception as exc:
                print(f"camera capture warning: {exc}", flush=True)
                time.sleep(0.1)
            if source is None:
                time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - started)))

    threading.Thread(target=capture_loop, daemon=True).start()
    telemetry_seq = 0
    while True:
        ws = None
        try:
            ws = websocket.create_connection(args.url, timeout=args.connect_timeout, enable_multithread=True)
            # A short connect timeout is useful, but applying the same timeout
            # to a large RGB-D send creates a reconnect storm on brief Wi-Fi
            # congestion.  Capture remains independent and latest-only while
            # this send waits, so allowing the TCP channel to drain is safe.
            ws.settimeout(args.send_timeout)
            raw_socket = getattr(ws.sock, "sock", ws.sock)
            try:
                raw_socket.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_SNDBUF,
                    int(args.send_buffer_kb) * 1024,
                )
                # Wake senders as soon as a small amount can leave the kernel
                # instead of accumulating several stale RGB-D frames.
                if hasattr(socket, "TCP_NOTSENT_LOWAT"):
                    raw_socket.setsockopt(
                        socket.IPPROTO_TCP,
                        socket.TCP_NOTSENT_LOWAT,
                        min(32 * 1024, int(args.send_buffer_kb) * 1024),
                    )
            except OSError as exc:
                print(f"sensor socket tuning warning: {exc}", flush=True)
            ws.send(json.dumps(hello_packet(host=socket.gethostname(), streams={"camera": "d435i", "fps": args.fps})))
            last_telemetry = 0.0
            last_sent_seq = -1
            next_publish = time.monotonic()
            diagnostic_started = time.monotonic()
            diagnostic_sends = 0
            diagnostic_send_s = 0.0
            diagnostic_max_send_s = 0.0
            diagnostic_encode_s = 0.0
            diagnostic_bytes = 0
            print(f"connected to policy WebSocket {args.url}", flush=True)
            while True:
                now = time.monotonic()
                if now < next_publish:
                    await asyncio.sleep(min(0.01, next_publish - now))
                    continue
                if not frame_ready.wait(timeout=0.02):
                    await asyncio.sleep(0)
                    continue
                with frame_lock:
                    frame = dict(latest_frame)
                if not frame or int(frame.get("seq", -1)) == last_sent_seq:
                    await asyncio.sleep(0.002)
                    continue
                packet = image_packet(
                    seq=int(frame["seq"]), stamp=float(frame["stamp"]),
                    rgb_jpeg=frame["rgb_jpeg"], depth_png=frame["depth_png"],
                    width=int(frame["width"]), height=int(frame["height"]),
                    camera_frame=args.camera_frame,
                    depth_frame=("d435i_depth_optical_frame" if source is not None and args.align_to == "depth" else args.camera_frame),
                    depth_scale=float(frame["depth_scale"]),
                    intrinsics=frame["intrinsics"], color_depth_sync_ms=frame["sync_ms"],
                    rgb_intrinsics=frame.get("rgb_intrinsics"), depth_intrinsics=frame.get("depth_intrinsics"),
                    depth_to_color_extrinsics=frame.get("depth_to_color_extrinsics"),
                    camera_imu=frame.get("camera_imu"),
                )
                encode_started = time.monotonic()
                payload = encode_wire_packet(packet, compression_level=1)
                diagnostic_encode_s += time.monotonic() - encode_started
                send_started = time.monotonic()
                ws.send_binary(payload)
                send_s = time.monotonic() - send_started
                diagnostic_sends += 1
                diagnostic_send_s += send_s
                diagnostic_max_send_s = max(diagnostic_max_send_s, send_s)
                diagnostic_bytes += len(payload)
                last_sent_seq = int(frame["seq"])
                now = time.monotonic()
                if now - last_telemetry >= args.telemetry_period:
                    telemetry_seq += 1
                    telemetry = state.snapshot()
                    if source is not None and source.latest_motion:
                        telemetry["camera_imu"] = dict(source.latest_motion)
                    ws.send(json.dumps(telemetry_packet(seq=telemetry_seq, telemetry=telemetry), separators=(",", ":")))
                    last_telemetry = now
                if now - diagnostic_started >= 10.0:
                    wall = max(now - diagnostic_started, 1e-6)
                    print(
                        "sensor transport "
                        f"send_hz={diagnostic_sends / wall:.2f} "
                        f"avg_send_ms={1000.0 * diagnostic_send_s / max(diagnostic_sends, 1):.1f} "
                        f"avg_encode_ms={1000.0 * diagnostic_encode_s / max(diagnostic_sends, 1):.1f} "
                        f"max_send_ms={1000.0 * diagnostic_max_send_s:.1f} "
                        f"wire_mbps={diagnostic_bytes * 8.0 / wall / 1e6:.2f} "
                        f"capture_seq={frame['seq']}",
                        flush=True,
                    )
                    diagnostic_started = now
                    diagnostic_sends = 0
                    diagnostic_send_s = 0.0
                    diagnostic_encode_s = 0.0
                    diagnostic_max_send_s = 0.0
                    diagnostic_bytes = 0
                # Schedule against the original cadence.  Adding the encode
                # time to ``now`` made a 10 Hz target degrade to ~6 Hz even
                # when capture and socket send were fast.
                next_publish += 1.0 / args.publish_fps
                if next_publish < now:
                    next_publish = now
        except Exception as exc:
            print(f"sensor link disconnected: {exc}; retrying in {args.reconnect_s}s", flush=True)
            await asyncio.sleep(args.reconnect_s)
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:12334")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--color-fps", type=int, default=10)
    parser.add_argument("--depth-width", type=int, default=848)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=10)
    parser.add_argument("--align-to", choices=("none", "color", "depth"), default="depth")
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--depth-png-compression", type=int, default=4)
    parser.add_argument("--telemetry-period", type=float, default=0.2)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--send-timeout", type=float, default=3.0)
    parser.add_argument("--send-buffer-kb", type=int, default=128)
    parser.add_argument("--reconnect-s", type=float, default=2.0)
    parser.add_argument("--camera-frame", default="d435i_color_optical_frame")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument(
        "--enable-camera-imu",
        action="store_true",
        help="also stream the D435i gyro/accelerometer (disabled by default for RGB-D stability)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.publish_fps <= 0:
        parser.error("--publish-fps must be positive")
    if not 0 <= args.depth_png_compression <= 9:
        parser.error("--depth-png-compression must be in [0, 9]")
    if args.send_timeout <= 0:
        parser.error("--send-timeout must be positive")
    if args.send_buffer_kb < 32:
        parser.error("--send-buffer-kb must be at least 32")
    try:
        asyncio.run(publish(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
