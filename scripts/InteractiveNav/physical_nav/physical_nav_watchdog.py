#!/usr/bin/env python3
"""Monitor whether the physical-navigation gateway keeps making progress."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HealthLimits:
    frame_stale_s: float = 15.0
    perception_stale_s: float = 15.0
    require_perception: bool = True


def health_errors(payload: Any, limits: HealthLimits) -> list[str]:
    """Return actionable failures from one ``/api/health`` response."""
    if not isinstance(payload, dict):
        return ["health response is not a JSON object"]
    errors: list[str] = []
    generated_at = float(payload.get("generated_at") or time.time())
    frame_seq_value = payload.get("frame_seq", -1)
    frame_seq = int(frame_seq_value) if frame_seq_value is not None else -1
    frame_stamp = float(payload.get("frame_stamp") or 0.0)
    frame_age = max(0.0, generated_at - frame_stamp) if frame_stamp > 0 else float("inf")
    if payload.get("link_connected") is not True:
        errors.append("Go2 sensor link is disconnected")
    if frame_seq < 0:
        errors.append("no D435i frame has been received")
    elif frame_age > limits.frame_stale_s:
        errors.append(f"D435i frame is stale ({frame_age:.1f}s > {limits.frame_stale_s:.1f}s)")

    if limits.require_perception:
        perception_seq_value = payload.get("perception_seq", -1)
        perception_seq = int(perception_seq_value) if perception_seq_value is not None else -1
        perception_stamp = float(payload.get("perception_stamp") or 0.0)
        perception_age = (
            max(0.0, generated_at - perception_stamp)
            if perception_stamp > 0 else float("inf")
        )
        if perception_seq < 0:
            errors.append("YOLOE has not published a perception receipt")
        elif perception_age > limits.perception_stale_s:
            errors.append(
                "YOLOE receipt is stale "
                f"({perception_age:.1f}s > {limits.perception_stale_s:.1f}s)"
            )
    return errors


def fetch_health(url: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def monitor(args: argparse.Namespace) -> int:
    limits = HealthLimits(
        frame_stale_s=args.frame_stale_s,
        perception_stale_s=args.perception_stale_s,
        require_perception=args.require_perception,
    )
    started = time.monotonic()
    consecutive_failures = 0
    last_message = ""
    while True:
        try:
            payload = fetch_health(args.url, args.timeout_s)
            errors = health_errors(payload, limits)
        except Exception as exc:
            errors = [f"health request failed: {exc}"]

        in_startup_grace = time.monotonic() - started < args.startup_grace_s
        if errors and in_startup_grace:
            message = "waiting for physical pipeline: " + "; ".join(errors)
            if message != last_message:
                print(message, flush=True)
                last_message = message
            consecutive_failures = 0
        elif errors:
            consecutive_failures += 1
            message = (
                f"physical pipeline unhealthy [{consecutive_failures}/{args.failure_limit}]: "
                + "; ".join(errors)
            )
            print(message, flush=True)
            last_message = message
            if consecutive_failures >= args.failure_limit:
                if not args.keep_running:
                    print("physical pipeline watchdog exiting", flush=True)
                    return 1
                if consecutive_failures == args.failure_limit:
                    print(
                        "physical pipeline remains alive for automatic sensor recovery",
                        flush=True,
                    )
                consecutive_failures = args.failure_limit
        else:
            if last_message:
                print("physical pipeline health recovered", flush=True)
            last_message = ""
            consecutive_failures = 0
        time.sleep(args.interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/api/health")
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--startup-grace-s", type=float, default=60.0)
    parser.add_argument("--frame-stale-s", type=float, default=15.0)
    parser.add_argument("--perception-stale-s", type=float, default=15.0)
    parser.add_argument("--failure-limit", type=int, default=5)
    parser.add_argument("--require-perception", action="store_true")
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="report stale input without terminating the supervised ROS stack",
    )
    args = parser.parse_args()
    if min(
        args.interval_s,
        args.timeout_s,
        args.frame_stale_s,
        args.perception_stale_s,
    ) <= 0:
        parser.error("interval and timeout limits must be positive")
    if args.startup_grace_s < 0 or args.failure_limit <= 0:
        parser.error("startup grace must be non-negative and failure limit must be positive")
    raise SystemExit(monitor(args))


if __name__ == "__main__":
    main()
