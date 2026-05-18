#!/usr/bin/env python3
import threading

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from semantic_mapping_py_pkg.detector_backends import make_detector_backend
from semantic_mapping_py_pkg.messages import dumps_compact, stamp_to_json
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param, get_topics


class ObjectDetectionNode:
    def __init__(self):
        rospy.init_node("semantic_object_detection")
        self.bridge = CvBridge()
        topics = get_topics(rospy)
        frames = get_frames(rospy)
        config = get_nested_param(rospy, "object_detection", {}) or {}

        self.rgb_topic = topics.get("rgb_image", "/molmo_spaces/head_camera/image")
        self.depth_topic = topics.get("depth_image", "/molmo_spaces/head_camera/depth")
        self.camera_info_topic = topics.get("camera_info", "/molmo_spaces/head_camera/camera_info")
        self.output_topic = topics.get("object_detections", "/semantic_mapping/object_detections")
        self.publish_rate = float(config.get("publish_rate", 5.0))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))
        self.default_frame_id = frames.get("camera_frame", "tf_frame_lidar")
        self.backend = make_detector_backend(config.get("backend", "mock_empty"), config, frames=frames)

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_stamp = None
        self.latest_frame_id = self.default_frame_id

        self.pub = rospy.Publisher(self.output_topic, String, queue_size=10)
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.timer_callback)

        rospy.loginfo("[object_detection_node] backend=%s rgb=%s depth=%s output=%s",
                      config.get("backend", "mock_empty"), self.rgb_topic, self.depth_topic, self.output_topic)

    def rgb_callback(self, msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[object_detection_node] RGB conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_rgb = rgb
            self.latest_stamp = msg.header.stamp
            if msg.header.frame_id:
                self.latest_frame_id = msg.header.frame_id

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[object_detection_node] depth conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_depth = depth

    def camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def timer_callback(self, _event):
        with self.lock:
            rgb = self.latest_rgb
            depth = self.latest_depth
            camera_info = self.latest_camera_info
            stamp = self.latest_stamp or rospy.Time.now()
            frame_id = self.latest_frame_id or self.default_frame_id

        if rgb is None:
            return

        detections = self.backend.detect(rgb, depth, camera_info, stamp, frame_id)
        detections = [
            det for det in detections
            if float(det.get("confidence", det.get("conf", 0.0)) or 0.0) >= self.confidence_threshold
        ]
        payload = {
            **stamp_to_json(stamp),
            "detections": detections,
        }
        self.pub.publish(String(data=dumps_compact(payload)))


if __name__ == "__main__":
    ObjectDetectionNode()
    rospy.spin()
