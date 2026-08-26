#!/usr/bin/env python3
"""ROS1 adapter for the perception/map consistency evaluator."""

from __future__ import annotations

import json
import math
import threading
from typing import Any

from physical_consistency import evaluate_frame
import rospy
import tf2_ros


class ConsistencyNode:
    def __init__(self, args: Any) -> None:
        import rospy
        from std_msgs.msg import String
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo
        self._lock = threading.Lock(); self._detections: list[dict[str, Any]] = []; self._graph: dict[str, Any] = {}
        self._intrinsics: dict[str, Any] = {}; self._telemetry: dict[str, Any] = {}
        self._camera_translation = (args.camera_x, args.camera_y, args.camera_z)
        self._camera_rpy = (args.camera_roll, args.camera_pitch, args.camera_yaw)
        self._camera_frame = args.camera_frame
        self._world_frame = args.world_frame
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._pub = rospy.Publisher("/physical_nav/consistency", String, queue_size=1)
        rospy.Subscriber("/physical_nav/detections", String, self._detections_cb, queue_size=1)
        rospy.Subscriber("/physical_nav/unified_graph", String, self._graph_cb, queue_size=1)
        rospy.Subscriber("/physical_nav/camera_info", CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber("/physical_nav/odom", Odometry, self._odom_cb, queue_size=1)
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

    def _camera_info_cb(self, msg: Any) -> None:
        with self._lock:
            self._intrinsics = {
                "fx": float(msg.K[0]), "fy": float(msg.K[4]),
                "cx": float(msg.K[2]), "cy": float(msg.K[5]),
                "width": int(msg.width), "height": int(msg.height),
            }

    def _odom_cb(self, msg: Any) -> None:
        q = msg.pose.pose.orientation
        yaw = 2.0 * math.atan2(float(q.z), float(q.w))
        with self._lock:
            self._telemetry = {
                "position": [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(msg.pose.pose.position.z)],
                "velocity": [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y), float(msg.twist.twist.linear.z)],
                "yaw": yaw,
            }

    def _publish(self, _event: Any) -> None:
        with self._lock:
            detections, graph = list(self._detections), dict(self._graph)
            intrinsics, telemetry = dict(self._intrinsics), dict(self._telemetry)
        context = None
        if intrinsics and telemetry:
            context = {
                "intrinsics": intrinsics,
                "telemetry": telemetry,
                "camera_translation": self._camera_translation,
                "camera_rpy": self._camera_rpy,
                "image_size": (intrinsics.get("width", 0), intrinsics.get("height", 0)),
            }
        projection_source = "odom+camera_extrinsic_fallback"
        if intrinsics:
            try:
                transform = self._tf_buffer.lookup_transform(self._camera_frame, self._world_frame, rospy.Time(0), rospy.Duration(0.05))
                context = context or {"intrinsics": intrinsics, "telemetry": telemetry}
                context["world_to_camera"] = {
                    "translation": [float(transform.transform.translation.x), float(transform.transform.translation.y), float(transform.transform.translation.z)],
                    "quaternion": [float(transform.transform.rotation.x), float(transform.transform.rotation.y), float(transform.transform.rotation.z), float(transform.transform.rotation.w)],
                }
                projection_source = "tf_camera_from_map"
            except Exception:
                pass
        report = evaluate_frame(detections, graph=graph, projection_context=context)
        report["detection_count"] = len(detections)
        report["graph_revision"] = graph.get("revision", graph.get("timestamp"))
        report["projection"] = {
            "enabled": bool(context),
            "source": projection_source if context else "waiting_for_camera_info_and_odom",
            "camera_frame": self._camera_frame,
        }
        self._pub.publish(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    import rospy
    rospy.init_node("physical_nav_consistency", anonymous=False)
    args = type("ConsistencyArgs", (), {
        "camera_frame": str(rospy.get_param("~camera_frame", "d435i_color_optical_frame")),
        "world_frame": str(rospy.get_param("~world_frame", "tf_frame_map")),
        "camera_x": float(rospy.get_param("~camera_x", 0.0)),
        "camera_y": float(rospy.get_param("~camera_y", 0.0)),
        "camera_z": float(rospy.get_param("~camera_z", 0.0)),
        "camera_roll": float(rospy.get_param("~camera_roll", 0.0)),
        "camera_pitch": float(rospy.get_param("~camera_pitch", 0.0)),
        "camera_yaw": float(rospy.get_param("~camera_yaw", 0.0)),
    })()
    ConsistencyNode(args); rospy.spin()


if __name__ == "__main__": main()
