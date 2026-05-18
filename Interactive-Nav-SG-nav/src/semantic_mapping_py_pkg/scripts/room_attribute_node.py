#!/usr/bin/env python3
import time

import rospy
from std_msgs.msg import String

from semantic_mapping_py_pkg.messages import dumps_compact, parse_json_list, parse_json_object_or_text
from semantic_mapping_py_pkg.room_inference_backends import make_room_backend
from semantic_mapping_py_pkg.ros_params import get_nested_param, get_topics


class RoomAttributeNode:
    def __init__(self):
        rospy.init_node("semantic_room_attribute")
        topics = get_topics(rospy)
        config = get_nested_param(rospy, "room_inference", {}) or {}
        priors = get_nested_param(rospy, "object_room_priors", {}) or {}

        self.input_topic = topics.get("object_detections", "/semantic_mapping/object_detections")
        self.output_topic = topics.get("scene_attribute", "/semantic_mapping/scene_attribute")
        self.publish_rate = float(config.get("publish_rate", 2.0))
        self.evidence_window_sec = float(config.get("evidence_window_sec", 5.0))
        self.backend = make_room_backend(config.get("backend", "object_rules"), config, priors)
        self.recent_detections = []

        self.sub = rospy.Subscriber(self.input_topic, String, self.detections_callback, queue_size=10)
        self.pub = rospy.Publisher(self.output_topic, String, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.timer_callback)
        rospy.loginfo("[room_attribute_node] backend=%s input=%s output=%s",
                      config.get("backend", "object_rules"), self.input_topic, self.output_topic)

    def detections_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        detections = parsed.get("detections")
        if detections is None:
            detections = parse_json_list(msg.data)
        if not isinstance(detections, list):
            return
        now = time.time()
        for det in detections:
            if isinstance(det, dict):
                self.recent_detections.append((now, det))
        self._drop_old(now)

    def timer_callback(self, _event):
        now = time.time()
        self._drop_old(now)
        detections = [det for _, det in self.recent_detections]
        result = self.backend.infer(detections)
        stamp = rospy.Time.now()
        payload = {
            "scene_attribute": result.get("scene_attribute", "unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "evidence": result.get("evidence", []),
            "image_timestamp_sec": int(stamp.secs),
            "image_timestamp_nsec": int(stamp.nsecs),
        }
        self.pub.publish(String(data=dumps_compact(payload)))

    def _drop_old(self, now):
        cutoff = now - self.evidence_window_sec
        self.recent_detections = [(ts, det) for ts, det in self.recent_detections if ts >= cutoff]


if __name__ == "__main__":
    RoomAttributeNode()
    rospy.spin()
