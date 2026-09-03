#!/usr/bin/env python3
"""Launcher for the Go2-side read-only D435i WebSocket client."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-host", default="127.0.0.1", help="policy host reachable from Go2")
    p.add_argument("--policy-port", type=int, default=12334)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--interface", default="eth0")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--publish-fps", type=float, default=10.0)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    url = f"ws://{args.policy_host}:{args.policy_port}"
    command = [sys.executable, str(root / "go2_readonly_sensor_bridge.py"), "--url", url, "--interface", args.interface, "--fps", str(args.fps), "--publish-fps", str(args.publish_fps)]
    if args.dry_run: command.append("--dry-run")
    # No SSH tunnel or ROS is started here: this process has one responsibility
    # and one outbound data channel, the policy WebSocket.
    os.execv(command[0], command)


if __name__ == "__main__": main()
