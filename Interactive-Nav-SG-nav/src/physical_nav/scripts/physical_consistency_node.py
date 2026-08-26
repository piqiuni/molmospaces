#!/usr/bin/env python3
"""Installed ROS entry point for read-only perception/map diagnostics."""
from __future__ import annotations
import json, threading
from typing import Any
import rospy
from std_msgs.msg import String
from physical_consistency import evaluate_frame

class ConsistencyNode:
    def __init__(self) -> None:
        self._lock = threading.Lock(); self._detections: list[dict[str, Any]] = []; self._graph: dict[str, Any] = {}
        self._pub = rospy.Publisher("/physical_nav/consistency", String, queue_size=1)
        rospy.Subscriber("/physical_nav/detections", String, self._detections_cb, queue_size=1); rospy.Subscriber("/physical_nav/unified_graph", String, self._graph_cb, queue_size=1)
        rospy.Timer(rospy.Duration(.2), self._publish)
    def _detections_cb(self, msg: Any) -> None:
        try:
            value = json.loads(msg.data); value = value.get("detections", value.get("objects", [])) if isinstance(value, dict) else value
            with self._lock: self._detections = list(value or [])
        except Exception as exc: rospy.logwarn_throttle(5.0, "invalid detections JSON: %s", exc)
    def _graph_cb(self, msg: Any) -> None:
        try:
            with self._lock: self._graph = json.loads(msg.data)
        except Exception as exc: rospy.logwarn_throttle(5.0, "invalid graph JSON: %s", exc)
    def _publish(self, _event: Any) -> None:
        with self._lock: detections, graph = list(self._detections), dict(self._graph)
        result = evaluate_frame(detections, graph=graph); result["graph_revision"] = graph.get("revision", graph.get("timestamp"))
        self._pub.publish(json.dumps(result, ensure_ascii=False, separators=(",", ":")))

def main() -> None:
    rospy.init_node("physical_nav_consistency", anonymous=False); ConsistencyNode(); rospy.spin()

if __name__ == "__main__": main()
