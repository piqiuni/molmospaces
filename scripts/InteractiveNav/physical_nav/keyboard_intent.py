#!/usr/bin/env python3
"""Keyboard teleoperation-intent source for the stationary Phase-1 run."""

from __future__ import annotations

import argparse
import json
import sys
import termios
import tty
import urllib.request


KEYS = {"w": "MOVE_FORWARD", "s": "MOVE_BACKWARD", "a": "TURN_LEFT", "d": "TURN_RIGHT", "q": "STOP", "x": "STOP"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--url", default="http://127.0.0.1:8765/api/teleop-intent"); args = p.parse_args()
    if not sys.stdin.isatty(): raise SystemExit("keyboard intent requires a TTY")
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd); tty.setcbreak(fd)
    print("Physical Phase 1 keyboard intent: w/s/a/d, q/x=STOP; every command is blocked and logged.", flush=True)
    try:
        while True:
            key = sys.stdin.read(1).lower()
            if key == "\x03": break
            action = KEYS.get(key)
            if not action: continue
            payload = json.dumps({"action": action, "source": "keyboard"}).encode()
            try:
                request = urllib.request.Request(args.url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=2) as response: print(response.read().decode(), flush=True)
            except Exception as exc: print(f"intent submit failed: {exc}", flush=True)
            if key == "q": break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__": main()
