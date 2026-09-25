#!/usr/bin/env python3
"""Monitor ROS pipeline progress independently of the optional dashboard."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HealthLimits:
    frame_stale_s: float = 15.0
    perception_stale_s: float = 15.0
    require_perception: bool = True


def _finite_float(value: Any, default: float | None = None) -> float | None:
    """Parse an optional health value without crashing the watchdog."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def _reported_or_derived_age(
    payload: dict[str, Any], age_key: str, stamp_key: str, generated_at: float
) -> float:
    """Prefer gateway-local age over mixing clocks from Go2 and this host."""

    reported = _finite_float(payload.get(age_key))
    if reported is not None and reported >= 0.0:
        return reported
    stamp = _finite_float(payload.get(stamp_key), 0.0) or 0.0
    return max(0.0, generated_at - stamp) if stamp > 0.0 else float("inf")


def health_errors(payload: Any, limits: HealthLimits) -> list[str]:
    """Return actionable failures from one ``/api/health`` response."""
    if not isinstance(payload, dict):
        return ["health response is not a JSON object"]
    errors: list[str] = []
    generated_at = _finite_float(payload.get("generated_at"), time.time())
    if generated_at is None:
        generated_at = time.time()
    frame_seq_value = payload.get("frame_seq", -1)
    try:
        frame_seq = int(frame_seq_value) if frame_seq_value is not None else -1
    except (TypeError, ValueError):
        frame_seq = -1
    frame_age = _reported_or_derived_age(
        payload, "frame_age_s", "frame_stamp", generated_at
    )
    if payload.get("link_connected") is not True:
        errors.append("Go2 sensor link is disconnected")
    if frame_seq < 0:
        errors.append("no D435i frame has been received")
    elif frame_age > limits.frame_stale_s:
        errors.append(f"D435i frame is stale ({frame_age:.1f}s > {limits.frame_stale_s:.1f}s)")

    if limits.require_perception:
        perception_seq_value = payload.get("perception_seq", -1)
        try:
            perception_seq = (
                int(perception_seq_value) if perception_seq_value is not None else -1
            )
        except (TypeError, ValueError):
            perception_seq = -1
        perception_age = _reported_or_derived_age(
            payload, "perception_age_s", "perception_stamp", generated_at
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


class PipelineProgress:
    """Bounded progress receipts; duplicates cannot keep a stalled worker alive."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._receipts = {}

    def observe(self, stream, seq, stamp, *, ros_now=None):
        stamp = _finite_float(stamp)
        try:
            seq = int(seq)
        except (TypeError, ValueError, OverflowError):
            return False
        if stream not in {"frame", "perception"} or seq < 0 or stamp is None or stamp <= 0:
            return False
        now = self._clock()
        ros_now = _finite_float(ros_now)
        # These stamps have already been mapped to the local ROS epoch by the
        # sensor bridge. Include source age so old queued frames are not healthy.
        source_age = max(0.0, ros_now - stamp) if ros_now is not None else 0.0
        with self._lock:
            previous = self._receipts.get(stream)
            if previous is not None and stamp <= previous[1]:
                return False
            self._receipts[stream] = (seq, stamp, now, source_age)
        return True

    def snapshot(self):
        now = self._clock()
        result = {"generated_at": time.time(), "source": "ros"}
        with self._lock:
            for stream in ("frame", "perception"):
                seq, stamp, received, source_age = self._receipts.get(
                    stream, (-1, 0.0, now, float("inf"))
                )
                result.update({f"{stream}_seq": seq, f"{stream}_stamp": stamp,
                               f"{stream}_age_s": max(0.0, now - received) + source_age})
        # This is end-to-end data liveness, not a claim about TCP socket state.
        result["link_connected"] = result["frame_seq"] >= 0
        return result


class RosHealthSource:
    def __init__(self, args):
        import rospy
        from std_msgs.msg import Header, String

        self.rospy = rospy
        self.progress = PipelineProgress()
        rospy.init_node("physical_nav_watchdog", anonymous=False, disable_signals=True)
        self.subscribers = [
            rospy.Subscriber(args.capture_context_topic, String, self._capture,
                             queue_size=1, tcp_nodelay=True),
            rospy.Subscriber(args.perception_heartbeat_topic, Header, self._perception,
                             queue_size=1, tcp_nodelay=True),
        ]

    def _capture(self, message):
        try:
            data = json.loads(message.data)
            self.progress.observe("frame", data.get("seq"), data.get("stamp"),
                                  ros_now=self.rospy.Time.now().to_sec())
        except (ValueError, TypeError, AttributeError):
            pass

    def _perception(self, message):
        self.progress.observe("perception", message.seq, message.stamp.to_sec(),
                              ros_now=self.rospy.Time.now().to_sec())

    def snapshot(self):
        return self.progress.snapshot()

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()


def monitor(args: argparse.Namespace) -> int:
    source = RosHealthSource(args) if args.source == "ros" else None
    try:
        return _monitor(args, source)
    finally:
        if source is not None:
            source.close()


def _monitor(args: argparse.Namespace, source=None) -> int:
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
            payload = source.snapshot() if source is not None else fetch_health(args.url, args.timeout_s)
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
    parser.add_argument("--source", choices=("ros", "http"), default="ros")
    parser.add_argument("--capture-context-topic", default="/physical_nav/capture_context")
    parser.add_argument("--perception-heartbeat-topic", default="/physical_nav/yolo/heartbeat")
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
