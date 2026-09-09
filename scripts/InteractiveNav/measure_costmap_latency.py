#!/usr/bin/env python3
"""Lightweight ROS1 timestamp probe for the depth-to-costmap path.

The callbacks only copy message metadata into memory.  PointCloud2 bytes and
OccupancyGrid contents are never serialized, so the probe is suitable for a
performance smoke test without adding a recorder/rendering workload.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Path as RosPath
from sensor_msgs.msg import PointCloud2


try:
    from map_msgs.msg import OccupancyGridUpdate
except ImportError:  # pragma: no cover - depends on the ROS installation
    OccupancyGridUpdate = None


def _stamp_ns(message: Any) -> int:
    try:
        return int(message.header.stamp.to_nsec())
    except (AttributeError, TypeError, ValueError):
        return 0


def _seq(message: Any) -> int:
    try:
        return int(message.header.seq)
    except (AttributeError, TypeError, ValueError):
        return -1


def _event(kind: str, message: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "kind": kind,
        "receipt_wall_ns": time.time_ns(),
        "receipt_mono_ns": time.monotonic_ns(),
        "header_stamp_ns": _stamp_ns(message),
        "header_seq": _seq(message),
    }
    if isinstance(message, PointCloud2):
        event.update(
            {
                "width": int(message.width),
                "height": int(message.height),
                "point_step": int(message.point_step),
                "row_step": int(message.row_step),
                "data_bytes": len(message.data),
            }
        )
    elif isinstance(message, OccupancyGrid):
        event.update(
            {
                "width": int(message.info.width),
                "height": int(message.info.height),
                "resolution_m": float(message.info.resolution),
                "data_cells": len(message.data),
            }
        )
    elif OccupancyGridUpdate is not None and isinstance(message, OccupancyGridUpdate):
        event.update(
            {
                "x": int(message.x),
                "y": int(message.y),
                "width": int(message.width),
                "height": int(message.height),
                "data_cells": len(message.data),
            }
        )
    elif isinstance(message, RosPath):
        event["pose_count"] = len(message.poses)
    elif isinstance(message, Twist):
        event.update(
            {
                "linear_x": float(message.linear.x),
                "linear_y": float(message.linear.y),
                "angular_z": float(message.angular.z),
            }
        )
    return event


def _percentiles(values_ns: list[int]) -> dict[str, float]:
    if not values_ns:
        return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values_ms = [value / 1e6 for value in values_ns]
    ordered = sorted(values_ms)

    def percentile(q: float) -> float:
        return float(ordered[round((len(ordered) - 1) * q)])

    return {
        "count": len(values_ms),
        "avg_ms": float(statistics.fmean(values_ms)),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": float(max(values_ms)),
    }


def _next_publish_deltas(events: list[dict[str, Any]], source_kind: str, target_kind: str) -> list[int]:
    """Return first subsequent target receipt for each source receipt.

    Costmap observation buffers are asynchronous, so this is an apparent
    source-to-next-publish latency rather than proof that the published grid
    contains exactly that one cloud.  It intentionally avoids repeatedly
    pairing shutdown-time map publishes with a stale final cloud.
    """
    sources = [event for event in events if event["kind"] == source_kind]
    targets = [event for event in events if event["kind"] == target_kind]
    deltas: list[int] = []
    target_index = 0
    for source in sources:
        source_mono = int(source["receipt_mono_ns"])
        while target_index < len(targets) and int(targets[target_index]["receipt_mono_ns"]) < source_mono:
            target_index += 1
        if target_index < len(targets):
            deltas.append(int(targets[target_index]["receipt_mono_ns"]) - source_mono)
    return deltas


def _next_publish_header_stamp_deltas(
    events: list[dict[str, Any]], source_kind: str, target_kind: str
) -> list[int]:
    """Return header-stamp deltas for the same first-subsequent pairing."""
    sources = [event for event in events if event["kind"] == source_kind]
    targets = [event for event in events if event["kind"] == target_kind]
    deltas: list[int] = []
    target_index = 0
    for source in sources:
        source_mono = int(source["receipt_mono_ns"])
        while target_index < len(targets) and int(targets[target_index]["receipt_mono_ns"]) < source_mono:
            target_index += 1
        if target_index >= len(targets):
            break
        source_stamp = int(source.get("header_stamp_ns", 0))
        target_stamp = int(targets[target_index].get("header_stamp_ns", 0))
        if source_stamp > 0 and target_stamp > 0:
            deltas.append(target_stamp - source_stamp)
    return deltas


def _header_to_receipt_deltas(events: list[dict[str, Any]], kind: str) -> list[int]:
    return [
        int(event["receipt_wall_ns"]) - int(event["header_stamp_ns"])
        for event in events
        if event["kind"] == kind and int(event.get("header_stamp_ns", 0)) > 0
    ]


class CostmapLatencyProbe:
    def __init__(self, output_dir: Path, topics: dict[str, str]) -> None:
        self.output_dir = output_dir
        self.topics = topics
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False
        self._subscribers = []

        self._subscribers.append(
            rospy.Subscriber(
                topics["raw_pointcloud"],
                PointCloud2,
                lambda message: self._record("raw_pointcloud", message),
                queue_size=1,
                buff_size=16 * 1024 * 1024,
                tcp_nodelay=True,
            )
        )
        self._subscribers.append(
            rospy.Subscriber(
                topics["filtered_pointcloud"],
                PointCloud2,
                lambda message: self._record("filtered_pointcloud", message),
                queue_size=1,
                buff_size=16 * 1024 * 1024,
                tcp_nodelay=True,
            )
        )
        self._subscribers.append(
            rospy.Subscriber(
                topics["local_costmap"],
                OccupancyGrid,
                lambda message: self._record("local_costmap", message),
                queue_size=1,
            )
        )
        if OccupancyGridUpdate is not None:
            self._subscribers.append(
                rospy.Subscriber(
                    topics["local_costmap_updates"],
                    OccupancyGridUpdate,
                    lambda message: self._record("local_costmap_update", message),
                    queue_size=1,
                )
            )
        self._subscribers.append(
            rospy.Subscriber(
                topics["local_plan"],
                RosPath,
                lambda message: self._record("local_plan", message),
                queue_size=1,
            )
        )
        self._subscribers.append(
            rospy.Subscriber(
                topics["cmd_vel"],
                Twist,
                lambda message: self._record("cmd_vel", message),
                queue_size=1,
            )
        )

    def _record(self, kind: str, message: Any) -> None:
        with self._lock:
            if not self._closed:
                self._events.append(_event(kind, message))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            events = sorted(self._events, key=lambda item: int(item["receipt_mono_ns"]))
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass

        self.output_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.output_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")

        periods = []
        costmap_events = [event for event in events if event["kind"] == "local_costmap"]
        for previous, current in zip(costmap_events, costmap_events[1:]):
            periods.append(int(current["receipt_mono_ns"]) - int(previous["receipt_mono_ns"]))
        summary = {
            "topics": self.topics,
            "event_counts": {
                kind: sum(event["kind"] == kind for event in events)
                for kind in sorted({event["kind"] for event in events})
            },
            "costmap_publish_period": _percentiles(periods),
            "raw_pointcloud_to_next_local_costmap": _percentiles(
                _next_publish_deltas(events, "raw_pointcloud", "local_costmap")
            ),
            "raw_pointcloud_to_next_filtered_pointcloud": _percentiles(
                _next_publish_deltas(events, "raw_pointcloud", "filtered_pointcloud")
            ),
            "filtered_pointcloud_to_next_local_costmap": _percentiles(
                _next_publish_deltas(events, "filtered_pointcloud", "local_costmap")
            ),
            "raw_pointcloud_header_to_probe_receipt": _percentiles(
                _header_to_receipt_deltas(events, "raw_pointcloud")
            ),
            "filtered_pointcloud_header_to_next_local_costmap_header": _percentiles(
                _next_publish_header_stamp_deltas(
                    events, "filtered_pointcloud", "local_costmap"
                )
            ),
            "pairing_note": (
                "Each value pairs a source receipt with the first subsequent target receipt; "
                "Costmap observation buffering means this is apparent publication latency, "
                "not an internal updateMap duration or a one-cloud causality proof. Header-to-"
                "header values are meaningful only when simulated ROS time shares the wall-clock "
                "epoch used by this probe."
            ),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-pointcloud-topic", default="/registered_scan")
    parser.add_argument("--filtered-pointcloud-topic", default="/filtered_pointcloud")
    parser.add_argument("--local-costmap-topic", default="/move_base/local_costmap/costmap")
    parser.add_argument(
        "--local-costmap-updates-topic",
        default="/move_base/local_costmap/costmap_updates",
    )
    parser.add_argument("--local-plan-topic", default="/move_base/DWAPlannerROS/local_plan")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("costmap_latency_probe", anonymous=True)
    topics = {
        "raw_pointcloud": args.raw_pointcloud_topic,
        "filtered_pointcloud": args.filtered_pointcloud_topic,
        "local_costmap": args.local_costmap_topic,
        "local_costmap_updates": args.local_costmap_updates_topic,
        "local_plan": args.local_plan_topic,
        "cmd_vel": args.cmd_vel_topic,
    }
    probe = CostmapLatencyProbe(Path(args.output_dir), topics)
    rospy.on_shutdown(probe.close)
    rospy.loginfo("Costmap latency probe started: %s", json.dumps(topics, sort_keys=True))
    rospy.spin()


if __name__ == "__main__":
    main()
