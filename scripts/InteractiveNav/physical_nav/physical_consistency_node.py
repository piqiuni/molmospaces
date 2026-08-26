#!/usr/bin/env python3
"""ROS1 adapter for the perception/map consistency evaluator."""

from __future__ import annotations

import json
import threading
from typing import Any

from physical_consistency import evaluate_frame
import rospy


class ConsistencyNode:
    def __init__(self) -> None:
        import rospy
        from std_msgs.msg import String
        self._lock = threading.Lock(); self._detections: list[dict[str, Any]] = []; self._graph: dict[str, Any] = {}
        self._pub = rospy.Publisher("/physical_nav/consistency", String, queue_size=1)
        rospy.Subscriber("/physical_nav/detections", String, self._detections_cb, queue_size=1)
        rospy.Subscriber("/physical_nav/unified_graph", String, self._graph_cb, queue_size=1)
        self._timer = rospy.Timer(rospy.Duration(0.2), self._publish)

    def _detections_cb(self, msg: Any) -> None:
        try:
            value = json.loads(msg.data)
            if isinstance(value, dict): value = value.get("detections", value.get("objects", []))
            with self._lock: self._detections = list(value or [])
        except Exception as exc: rospy.logwarn_throttle(5.0, "invalid detections JSON: %s", exc)

    def _graph_cb(self, msg: Any) -> None:
        try:
            with self._lock: self._graph = json.loads(msg.data)
        except Exception as exc: rospy.logwarn_throttle(5.0, "invalid graph JSON: %s", exc)

    def _publish(self, _event: Any) -> None:
        with self._lock: detections, graph = list(self._detections), dict(self._graph)
        report = evaluate_frame(detections, graph=graph)
        report["detection_count"] = len(detections)
        report["graph_revision"] = graph.get("revision", graph.get("timestamp"))
        self._pub.publish(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    import rospy
    rospy.init_node("physical_nav_consistency", anonymous=False)
    ConsistencyNode(); rospy.spin()


if __name__ == "__main__": main()
