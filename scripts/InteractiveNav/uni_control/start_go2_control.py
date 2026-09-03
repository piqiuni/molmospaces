#!/usr/bin/env python3
"""Start and supervise the Go2→zgca_gpu SSH tunnel and control bridge."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class Launcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False
        self.tunnel: Optional[subprocess.Popen] = None
        self.bridge: Optional[subprocess.Popen] = None

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def tunnel_command(self) -> list[str]:
        forward = (
            f"{self.args.local_host}:{self.args.local_port}:"
            f"{self.args.server_host}:{self.args.server_port}"
        )
        return [
            "ssh",
            "-N",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"ConnectTimeout={self.args.connect_timeout}",
            "-o",
            f"ServerAliveInterval={self.args.server_alive_interval}",
            "-o",
            f"ServerAliveCountMax={self.args.server_alive_count_max}",
            "-L",
            forward,
            self.args.ssh_target,
        ]

    def bridge_command(self) -> list[str]:
        command = [
            sys.executable,
            "-u",
            str(Path(self.args.bridge).resolve()),
            "--url",
            f"ws://{self.args.local_host}:{self.args.local_port}",
            "--interface",
            self.args.interface,
            "--state-source",
            "go2",
            "--lidar-initial-state",
            self.args.lidar_initial_state,
            "--telemetry-period",
            str(self.args.telemetry_period),
            "--speech-primary-backend",
            self.args.speech_primary_backend,
            "--speech-fallback-backend",
            self.args.speech_fallback_backend,
            "--matcha-model-dir",
            self.args.matcha_model_dir,
            "--matcha-vocoder",
            self.args.matcha_vocoder,
            "--tts-threads",
            str(self.args.tts_threads),
            "--forward-distance",
            str(self.args.forward_distance),
            "--lateral-distance",
            str(self.args.lateral_distance),
            "--primitive-max-vx",
            str(self.args.primitive_max_vx),
            "--primitive-max-vy",
            str(self.args.primitive_max_vy),
            "--forward-kp",
            str(self.args.forward_kp),
            "--lateral-kp",
            str(self.args.lateral_kp),
            "--turn-angle-deg",
            str(self.args.turn_angle_deg),
            "--turn-control-mode",
            self.args.turn_control_mode,
            "--primitive-max-wz",
            str(self.args.primitive_max_wz),
            "--turn-kp",
            str(self.args.turn_kp),
        ]
        if self.args.turn_duration is not None:
            command.extend(["--turn-duration", str(self.args.turn_duration)])
        if self.args.enable_motion:
            command.append("--enable-motion")
        if self.args.bridge_ready_file:
            command.extend(["--ready-file", self.args.bridge_ready_file])
        for value in self.args.bridge_arg:
            command.append(value)
        return command

    @staticmethod
    def stop_process(
        process: Optional[subprocess.Popen],
        *,
        interrupt: bool,
        timeout: float,
    ) -> None:
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT if interrupt else signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def cleanup(self) -> None:
        # Bridge first: it sends zero velocity and releases remote command mode.
        self.stop_process(self.bridge, interrupt=True, timeout=5.0)
        self.stop_process(self.tunnel, interrupt=False, timeout=2.0)
        self.bridge = None
        self.tunnel = None

    def run_once(self) -> str:
        tunnel_command = self.tunnel_command()
        bridge_command = self.bridge_command()
        print("Starting SSH tunnel:", " ".join(tunnel_command), flush=True)
        self.tunnel = subprocess.Popen(tunnel_command)
        time.sleep(self.args.tunnel_start_delay)
        tunnel_code = self.tunnel.poll()
        if tunnel_code is not None:
            raise RuntimeError(f"SSH tunnel exited during startup with code {tunnel_code}")

        mode = (
            "VELOCITY + AUXILIARY CONTROL"
            if self.args.enable_motion
            else "TELEMETRY + AUXILIARY CONTROL (VELOCITY DISABLED)"
        )
        print(f"Starting Go2 bridge ({mode}):", " ".join(bridge_command), flush=True)
        self.bridge = subprocess.Popen(bridge_command, env=os.environ.copy())
        while not self.stop_requested:
            tunnel_code = self.tunnel.poll()
            if tunnel_code is not None:
                return f"SSH tunnel exited with code {tunnel_code}"
            bridge_code = self.bridge.poll()
            if bridge_code is not None:
                return f"Go2 bridge exited with code {bridge_code}"
            time.sleep(0.2)
        return "shutdown requested"

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        if self.args.print_command:
            print(" ".join(self.tunnel_command()))
            print(" ".join(self.bridge_command()))
            return

        while not self.stop_requested:
            reason = ""
            try:
                reason = self.run_once()
            except Exception as exc:
                reason = str(exc)
            finally:
                self.cleanup()
            print(reason, flush=True)
            if self.stop_requested or self.args.no_restart:
                break
            print(f"Restarting in {self.args.restart_delay:.1f}s", flush=True)
            end = time.monotonic() + self.args.restart_delay
            while not self.stop_requested and time.monotonic() < end:
                time.sleep(0.1)


def parse_args() -> argparse.Namespace:
    default_bridge = Path(__file__).with_name("go2_control_bridge.py")
    default_model_root = Path(__file__).resolve().parent / "vendor" / "local_tts" / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", default="zgca_gpu")
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--local-port", type=int, default=12333)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=12333)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--bridge", default=str(default_bridge))
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--speech-primary-backend",
        choices=("matcha", "edge"),
        default="matcha",
        help="preload local Matcha by default; use edge to skip local model loading",
    )
    parser.add_argument(
        "--speech-fallback-backend",
        choices=("edge", "disabled"),
        default="edge",
        help="fallback when Matcha loading or synthesis fails",
    )
    parser.add_argument(
        "--matcha-model-dir",
        default=str(default_model_root / "matcha-icefall-zh-baker"),
    )
    parser.add_argument(
        "--matcha-vocoder",
        default=str(default_model_root / "vocos-22khz-univ.onnx"),
    )
    parser.add_argument("--tts-threads", type=int, default=4)
    parser.add_argument(
        "--lidar-initial-state",
        choices=("unknown", "on", "off"),
        default="unknown",
        help="assumed LiDAR state; with unknown, the first keyboard toggle sends OFF",
    )
    parser.add_argument(
        "--forward-distance",
        type=float,
        default=0.25,
        help="distance in metres for one discrete w/s primitive",
    )
    parser.add_argument(
        "--lateral-distance",
        type=float,
        default=0.10,
        help="distance in metres for one discrete a/d primitive",
    )
    parser.add_argument(
        "--primitive-max-vx",
        type=float,
        default=1.0,
        help="maximum longitudinal speed for discrete primitives (m/s)",
    )
    parser.add_argument(
        "--primitive-max-vy",
        type=float,
        default=0.5,
        help="maximum lateral speed for discrete primitives (m/s)",
    )
    parser.add_argument(
        "--forward-kp",
        type=float,
        default=2.5,
        help="distance-proportional gain for discrete forward/backward motion",
    )
    parser.add_argument(
        "--lateral-kp",
        type=float,
        default=5.0,
        help="distance-proportional gain for discrete lateral motion",
    )
    parser.add_argument(
        "--turn-angle-deg",
        type=float,
        default=30.0,
        help="rotation angle in degrees for one discrete q/e primitive",
    )
    parser.add_argument(
        "--turn-control-mode",
        choices=("open_loop", "closed_loop"),
        default="open_loop",
        help="rotation completion mode",
    )
    parser.add_argument(
        "--turn-duration",
        type=float,
        default=None,
        help="optional open-loop q/e duration; default derives from angle and speed",
    )
    parser.add_argument(
        "--primitive-max-wz",
        type=float,
        default=1.5,
        help="maximum angular speed for discrete q/e primitives (rad/s)",
    )
    parser.add_argument(
        "--turn-kp",
        type=float,
        default=6.0,
        help="angle-proportional gain for discrete rotation",
    )
    parser.add_argument("--telemetry-period", type=float, default=0.20)
    parser.add_argument("--connect-timeout", type=int, default=5)
    parser.add_argument("--server-alive-interval", type=int, default=15)
    parser.add_argument("--server-alive-count-max", type=int, default=3)
    parser.add_argument("--tunnel-start-delay", type=float, default=1.0)
    parser.add_argument("--restart-delay", type=float, default=2.0)
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument(
        "--bridge-ready-file",
        default="",
        help="bridge-owned readiness PID file passed through to go2_control_bridge.py",
    )
    parser.add_argument("--bridge-arg", action="append", default=[])
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    if args.forward_distance <= 0 or args.lateral_distance <= 0:
        parser.error("discrete distances must be positive")
    if args.primitive_max_vx <= 0 or args.primitive_max_vy <= 0:
        parser.error("discrete speed limits must be positive")
    if args.forward_kp <= 0 or args.lateral_kp <= 0:
        parser.error("discrete proportional gains must be positive")
    if args.turn_angle_deg <= 0:
        parser.error("turn angle must be positive")
    if args.primitive_max_wz <= 0 or args.turn_kp <= 0:
        parser.error("discrete rotation speed and gain must be positive")
    if args.turn_duration is not None and args.turn_duration <= 0:
        parser.error("turn duration must be positive")
    if not 1 <= args.tts_threads <= 8:
        parser.error("--tts-threads must be between 1 and 8")
    return args


if __name__ == "__main__":
    Launcher(parse_args()).run()
