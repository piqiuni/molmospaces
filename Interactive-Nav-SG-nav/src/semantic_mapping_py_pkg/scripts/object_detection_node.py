#!/usr/bin/env python3
import sys
import threading

import numpy as np
import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from semantic_mapping_py_pkg.detector_backends import make_detector_backend
from semantic_mapping_py_pkg.messages import dumps_compact, stamp_to_json
from semantic_mapping_py_pkg.object_debug_viz import (
    make_box_markers,
    make_box_markers_world,
    make_segmented_cloud,
    make_segmented_cloud_world,
    render_detection_overlay,
)
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param, get_topics


def _image_encoding_info(encoding):
    encoding = str(encoding or "").lower()
    if encoding in {"rgb8", "bgr8"}:
        return np.uint8, 3
    if encoding in {"rgba8", "bgra8"}:
        return np.uint8, 4
    if encoding in {"mono8", "8uc1"}:
        return np.uint8, 1
    if encoding in {"mono16", "16uc1"}:
        return np.uint16, 1
    if encoding == "32fc1":
        return np.float32, 1
    if encoding == "16sc1":
        return np.int16, 1
    if encoding.startswith("8uc"):
        return np.uint8, max(1, int(encoding[3:] or 1))
    if encoding.startswith("16uc"):
        return np.uint16, max(1, int(encoding[4:] or 1))
    if encoding.startswith("32fc"):
        return np.float32, max(1, int(encoding[4:] or 1))
    raise RuntimeError("unsupported image encoding: %s" % encoding)


def _image_msg_to_numpy(msg, desired_encoding="passthrough"):
    dtype, channels = _image_encoding_info(msg.encoding)
    dtype = np.dtype(dtype)
    data = np.frombuffer(msg.data, dtype=dtype)
    if bool(msg.is_bigendian) != (sys.byteorder == "big"):
        data = data.byteswap()
    row_items = int(msg.step) // dtype.itemsize
    if channels == 1:
        image = data.reshape(int(msg.height), row_items)[:, : int(msg.width)]
    else:
        image = data.reshape(int(msg.height), row_items // channels, channels)[:, : int(msg.width), :]
    image = np.ascontiguousarray(image)

    desired_encoding = str(desired_encoding or "passthrough").lower()
    source_encoding = str(msg.encoding or "").lower()
    if desired_encoding in {"", "passthrough"}:
        return image
    if desired_encoding == "rgb8":
        if source_encoding == "rgb8":
            return image
        if source_encoding == "bgr8":
            return np.ascontiguousarray(image[:, :, ::-1])
        if source_encoding == "rgba8":
            return np.ascontiguousarray(image[:, :, :3])
        if source_encoding == "bgra8":
            return np.ascontiguousarray(image[:, :, 2::-1])
        if source_encoding in {"mono8", "8uc1"}:
            return np.repeat(image[:, :, None], 3, axis=2)
    raise RuntimeError("cannot convert image encoding %s to %s" % (msg.encoding, desired_encoding))


def _numpy_to_image_msg(image, encoding, stamp, frame_id):
    image = np.ascontiguousarray(image)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(image.strides[0])
    msg.data = image.tobytes()
    return msg


class ObjectDetectionNode:
    def __init__(self):
        patch_roslogging_findcaller_for_py311()
        rospy.init_node("semantic_object_detection")
        topics = get_topics(rospy)
        frames = get_frames(rospy)
        config = get_nested_param(rospy, "object_detection", {}) or {}

        self.rgb_topic = topics.get("rgb_image", "/molmo_spaces/head_camera/image")
        self.depth_topic = topics.get("depth_image", "/molmo_spaces/head_camera/depth")
        self.camera_info_topic = topics.get("camera_info", "/molmo_spaces/head_camera/camera_info")
        self.output_topic = topics.get("object_detections", "/semantic_mapping/object_detections")
        self.publish_rate = float(config.get("publish_rate", 5.0))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.35))
        self.max_depth_m = float(config.get("max_depth_m", 8.0))
        self.point_stride = max(1, int(config.get("point_stride", 4)))
        self.default_frame_id = frames.get("camera_frame", "tf_frame_lidar")
        self.world_frame = frames.get("world_frame", "tf_frame_map")
        self.backend = make_detector_backend(config.get("backend", "mock_empty"), config, frames=frames)
        self.publish_debug_markers = bool(config.get("publish_debug_markers", False))
        self.publish_debug_segmented_cloud = bool(config.get("publish_debug_segmented_cloud", False))
        self.publish_debug_detection_image = bool(config.get("publish_debug_detection_image", False))
        self.publish_debug_world_markers = bool(config.get("publish_debug_world_markers", False))
        self.publish_debug_world_segmented_cloud = bool(config.get("publish_debug_world_segmented_cloud", False))
        self.debug_marker_topic = config.get("debug_marker_topic", "/semantic_mapping/debug/boxes_3d")
        self.debug_segmented_cloud_topic = config.get(
            "debug_segmented_cloud_topic", "/semantic_mapping/debug/segmented_object_cloud"
        )
        self.debug_detection_image_topic = config.get(
            "debug_detection_image_topic", "/semantic_mapping/debug/detections_2d"
        )
        self.debug_world_marker_topic = config.get(
            "debug_world_marker_topic", "/semantic_mapping/debug/boxes_3d_world"
        )
        self.debug_world_segmented_cloud_topic = config.get(
            "debug_world_segmented_cloud_topic", "/semantic_mapping/debug/segmented_object_cloud_world"
        )

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_stamp = None
        self.latest_frame_id = self.default_frame_id
        self.latest_camera_frame_id = ""

        self.pub = rospy.Publisher(self.output_topic, String, queue_size=10)
        self.marker_pub = None
        self.segmented_cloud_pub = None
        self.debug_image_pub = None
        self.world_marker_pub = None
        self.world_segmented_cloud_pub = None
        self.tf_listener = None
        if self.publish_debug_markers:
            self.marker_pub = rospy.Publisher(self.debug_marker_topic, MarkerArray, queue_size=1)
        if self.publish_debug_segmented_cloud:
            self.segmented_cloud_pub = rospy.Publisher(self.debug_segmented_cloud_topic, PointCloud2, queue_size=1)
        if self.publish_debug_detection_image:
            self.debug_image_pub = rospy.Publisher(self.debug_detection_image_topic, Image, queue_size=1)
        if self.publish_debug_world_markers:
            self.world_marker_pub = rospy.Publisher(self.debug_world_marker_topic, MarkerArray, queue_size=1)
        if self.publish_debug_world_segmented_cloud:
            self.world_segmented_cloud_pub = rospy.Publisher(
                self.debug_world_segmented_cloud_topic, PointCloud2, queue_size=1
            )
        if self.publish_debug_world_markers or self.publish_debug_world_segmented_cloud:
            self.tf_listener = tf.TransformListener()
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.timer_callback)

        rospy.loginfo("[object_detection_node] backend=%s rgb=%s depth=%s output=%s",
                      config.get("backend", "mock_empty"), self.rgb_topic, self.depth_topic, self.output_topic)

    def rgb_callback(self, msg):
        try:
            rgb = _image_msg_to_numpy(msg, desired_encoding="rgb8")
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
            depth = _image_msg_to_numpy(msg, desired_encoding="passthrough")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[object_detection_node] depth conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_depth = depth

    def camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = msg
            if msg.header.frame_id:
                self.latest_camera_frame_id = msg.header.frame_id

    def timer_callback(self, _event):
        with self.lock:
            rgb = self.latest_rgb
            depth = self.latest_depth
            camera_info = self.latest_camera_info
            stamp = self.latest_stamp or rospy.Time.now()
            frame_id = self.latest_camera_frame_id or self.latest_frame_id or self.default_frame_id

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

        if self.marker_pub is not None and depth is not None and camera_info is not None:
            marker_msg = make_box_markers(detections, depth, camera_info, frame_id, stamp, {"label_height": 0.12})
            self.marker_pub.publish(marker_msg)
        if self.segmented_cloud_pub is not None and depth is not None and camera_info is not None:
            cloud_msg = make_segmented_cloud(
                detections, rgb, depth, camera_info, frame_id, stamp, self.point_stride, self.max_depth_m
            )
            self.segmented_cloud_pub.publish(cloud_msg)
        if (
            self.world_marker_pub is not None
            and depth is not None
            and camera_info is not None
            and self.tf_listener is not None
        ):
            world_marker_msg = make_box_markers_world(
                detections,
                depth,
                camera_info,
                frame_id,
                self.world_frame,
                stamp,
                {"label_height": 0.12},
                self.tf_listener,
            )
            self.world_marker_pub.publish(world_marker_msg)
        if (
            self.world_segmented_cloud_pub is not None
            and depth is not None
            and camera_info is not None
            and self.tf_listener is not None
        ):
            world_cloud_msg = make_segmented_cloud_world(
                detections,
                rgb,
                depth,
                camera_info,
                frame_id,
                self.world_frame,
                stamp,
                self.point_stride,
                self.max_depth_m,
                self.tf_listener,
            )
            self.world_segmented_cloud_pub.publish(world_cloud_msg)
        if self.debug_image_pub is not None:
            overlay = render_detection_overlay(rgb, detections)
            overlay_msg = _numpy_to_image_msg(overlay, "rgb8", stamp, frame_id)
            self.debug_image_pub.publish(overlay_msg)


if __name__ == "__main__":
    ObjectDetectionNode()
    rospy.spin()
