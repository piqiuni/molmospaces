#!/usr/bin/env python3
import argparse
import math
import os
import socket
import struct
import sys
import time
import xmlrpc.client
import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from visualization_msgs.msg import MarkerArray

from semantic_mapping_py_pkg.detector_backends import make_detector_backend
from semantic_mapping_py_pkg.messages import dumps_compact, stamp_to_json
from semantic_mapping_py_pkg.object_debug_viz import (
    get_confidence,
    make_box_markers,
    make_segmented_cloud,
)
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param


def _print_status(message):
    print(message, flush=True)


def _ros_master_reachable(timeout=1.0):
    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    try:
        parsed = xmlrpc.client.ServerProxy(master_uri)
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(float(timeout))
        try:
            parsed.getSystemState("/object_detection_visual_test_probe")
        finally:
            socket.setdefaulttimeout(old_timeout)
        return True, master_uri
    except Exception:
        return False, master_uri


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run one RGB-D image pair through the object detector backend and publish RViz visualization topics.",
    )
    parser.add_argument("--rgb_path", default="", help="Path to an RGB image.")
    parser.add_argument("--depth_path", default="", help="Path to a depth image, PNG/TIFF or .npy.")
    parser.add_argument("--backend", default="", help="Detector backend kind.")
    parser.add_argument("--provider", default="", help="Raw detection provider kind.")
    parser.add_argument("--external_url", default="", help="External HTTP detector URL.")
    parser.add_argument("--model_path", default="", help="Path to a local detector model.")
    parser.add_argument("--class_mapping", default="", help="Path to a JSON class mapping file.")
    parser.add_argument("--include_depth", default="", help="Whether to include depth in HTTP payload.")
    parser.add_argument("--confidence_threshold", default="", help="Confidence threshold.")
    parser.add_argument("--frame_id", default="", help="RViz fixed/camera frame.")
    parser.add_argument("--depth_scale", default="", help="Depth scale, e.g. 0.001 for millimeters.")
    parser.add_argument("--publish_rate", default="", help="Visualization publish rate.")
    parser.add_argument("--max_depth", default="", help="Maximum depth in meters for visualization.")
    parser.add_argument("--point_stride", default="", help="Point cloud sampling stride.")
    args, _unknown = parser.parse_known_args(rospy.myargv()[1:])
    return args


def _apply_cli_overrides(config, detector_config, args):
    for key in ("rgb_path", "depth_path", "frame_id"):
        value = getattr(args, key)
        if value:
            config[key] = value
    for key in ("depth_scale", "publish_rate", "max_depth"):
        value = getattr(args, key)
        if value:
            config[key] = float(value)
    if args.point_stride:
        config["point_stride"] = int(args.point_stride)

    for key in ("backend", "provider", "external_url", "model_path", "class_mapping"):
        value = getattr(args, key)
        if value:
            detector_config[key] = value
    if args.include_depth:
        detector_config["include_depth"] = _str_to_bool(args.include_depth)
    if args.confidence_threshold:
        detector_config["confidence_threshold"] = float(args.confidence_threshold)


def _load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to read RGB image: %s" % path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_depth(path, depth_scale):
    if path.lower().endswith(".npy"):
        depth = np.load(path)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("failed to read depth image: %s" % path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32) * float(depth_scale)


def _camera_info(width, height, config):
    fx = float(config.get("fx", 0.0))
    fy = float(config.get("fy", 0.0))
    cx = float(config.get("cx", -1.0))
    cy = float(config.get("cy", -1.0))
    if fx <= 0.0 or fy <= 0.0:
        fov = math.radians(float(config.get("fov_deg", 69.0)))
        fx = fy = width / (2.0 * math.tan(fov * 0.5))
    if cx < 0.0:
        cx = (width - 1.0) * 0.5
    if cy < 0.0:
        cy = (height - 1.0) * 0.5

    msg = CameraInfo()
    msg.width = int(width)
    msg.height = int(height)
    msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def _camera_point(u, v, depth, info):
    fx, fy = info.K[0], info.K[4]
    cx, cy = info.K[2], info.K[5]
    # Camera-centered frame for RViz: x forward, y left, z up.
    return float(depth), float(-(u - cx) * depth / fx), float(-(v - cy) * depth / fy)


def _pack_rgb_float(r, g, b):
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


def _make_cloud(rgb, depth, info, frame_id, stamp, stride, max_depth):
    height, width = depth.shape[:2]
    points = []
    for v in range(0, height, stride):
        for u in range(0, width, stride):
            z = float(depth[v, u])
            if not np.isfinite(z) or z <= 0.0 or z > max_depth:
                continue
            x_fwd, y_left, z_up = _camera_point(u, v, z, info)
            r, g, b = rgb[v, u]
            points.append([x_fwd, y_left, z_up, _pack_rgb_float(r, g, b)])

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]
    header = Header(stamp=stamp, frame_id=frame_id)
    return pc2.create_cloud(header, fields, points)


def _depth_viz(depth, max_depth):
    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.zeros_like(depth, dtype=np.float32)
    clipped[valid] = np.minimum(depth[valid], max_depth)
    normalized = np.zeros_like(clipped, dtype=np.uint8)
    if max_depth > 0.0:
        normalized = np.uint8(np.clip(clipped / max_depth, 0.0, 1.0) * 255.0)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _image_msg(image, encoding, frame_id, stamp):
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        raise RuntimeError("Image encoding %s expects uint8 input, got %s" % (encoding, image.dtype))
    if encoding in ("rgb8", "bgr8") and (image.ndim != 3 or image.shape[2] != 3):
        raise RuntimeError("Image encoding %s expects HxWx3 input, got shape %s" % (encoding, image.shape))

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


def _timed_publish(name, publisher, msg):
    start_time = time.perf_counter()
    publisher.publish(msg)
    return name, max(time.perf_counter() - start_time, 0.0)


class ObjectDetectionVisualTest:
    def __init__(self):
        patch_roslogging_findcaller_for_py311()
        ok, master_uri = _ros_master_reachable()
        if not ok:
            raise RuntimeError(
                "ROS master is not reachable at %s. Start `roscore` first, "
                "or use `roslaunch semantic_mapping_py_pkg object_detection_visual_test.launch ...`."
                % master_uri
            )
        _print_status("ROS master reachable at %s" % master_uri)
        _print_status(
            "ROS env: ROS_MASTER_URI=%s ROS_IP=%s ROS_HOSTNAME=%s" % (
                os.environ.get("ROS_MASTER_URI", ""),
                os.environ.get("ROS_IP", ""),
                os.environ.get("ROS_HOSTNAME", ""),
            )
        )
        _print_status("Calling rospy.init_node...")
        rospy.init_node(
            "object_detection_visual_test",
            anonymous=True,
            disable_signals=True,
            disable_rosout=True,
        )
        _print_status("rospy.init_node returned")
        args = _parse_cli_args()
        _print_status("ObjectDetectionVisualTest initialized")
        self.config = get_nested_param(rospy, "visual_test", {}) or {}
        frames = get_frames(rospy)
        detector_config = get_nested_param(rospy, "object_detection", {}) or {}
        _apply_cli_overrides(self.config, detector_config, args)

        rgb_path = self.config.get("rgb_path", "")
        depth_path = self.config.get("depth_path", "")
        if not rgb_path or not depth_path:
            raise RuntimeError(
                "RGB/depth input is missing. Use roslaunch args rgb_path/depth_path, "
                "or run directly with --rgb_path and --depth_path."
            )
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            raise RuntimeError("input image path does not exist: rgb=%s depth=%s" % (rgb_path, depth_path))

        self.frame_id = self.config.get("frame_id", "semantic_test_camera")
        self.publish_rate = float(self.config.get("publish_rate", 1.0))
        self.max_depth = float(self.config.get("max_depth", 6.0))
        self.stride = max(1, int(self.config.get("point_stride", 2)))

        self.rgb = _load_rgb(rgb_path)
        self.depth = _load_depth(depth_path, self.config.get("depth_scale", 0.001))
        if self.depth.shape[:2] != self.rgb.shape[:2]:
            self.depth = cv2.resize(self.depth, (self.rgb.shape[1], self.rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.info = _camera_info(self.rgb.shape[1], self.rgb.shape[0], self.config)

        self.stamp = rospy.Time.now()
        self.backend_name = detector_config.get("backend", "mock_empty")
        self.backend = make_detector_backend(self.backend_name, detector_config, frames=frames)
        self.confidence_threshold = float(detector_config.get("confidence_threshold", 0.0))
        _print_status(
            "Running backend=%s provider=%s rgb=%s depth=%s" % (
                self.backend_name,
                detector_config.get("provider", "external_http"),
                rgb_path,
                depth_path,
            )
        )

        self.cloud_msg = _make_cloud(self.rgb, self.depth, self.info, self.frame_id, self.stamp, self.stride, self.max_depth)
        self.depth_msg = _image_msg(_depth_viz(self.depth, self.max_depth), "bgr8", self.frame_id, self.stamp)
        self.rgb_msg = _image_msg(self.rgb, "rgb8", self.frame_id, self.stamp)
        self.detections = []

        self.cloud_pub = rospy.Publisher(self.config.get("cloud_topic", "/semantic_mapping/test/rgb_depth_cloud"),
                                         PointCloud2, queue_size=1, latch=True)
        self.depth_pub = rospy.Publisher(self.config.get("depth_viz_topic", "/semantic_mapping/test/depth_viz"),
                                         Image, queue_size=1, latch=True)
        self.rgb_pub = rospy.Publisher(self.config.get("rgb_topic", "/semantic_mapping/test/rgb_image"),
                                       Image, queue_size=1, latch=True)
        self.box_pub = rospy.Publisher(self.config.get("box_topic", "/semantic_mapping/test/boxes_3d"),
                                       MarkerArray, queue_size=1, latch=True)
        self.segmented_cloud_pub = rospy.Publisher(
            self.config.get("segmented_cloud_topic", "/semantic_mapping/test/segmented_object_cloud"),
            PointCloud2,
            queue_size=1,
            latch=True,
        )
        self.det_pub = rospy.Publisher(self.config.get("detections_topic", "/semantic_mapping/test/detections"),
                                       String, queue_size=1, latch=True)

        rospy.loginfo("[object_detection_visual_test] backend=%s frame=%s cloud_points=%d",
                      self.backend_name, self.frame_id, self.cloud_msg.width)

    def _detect_latest(self, stamp):
        start_time = time.perf_counter()
        detections = self.backend.detect(self.rgb, self.depth, self.info, stamp, self.frame_id)
        detections = [det for det in detections if get_confidence(det) >= self.confidence_threshold]
        elapsed_s = max(time.perf_counter() - start_time, 1e-9)
        self.detections = detections
        box_msg = make_box_markers(detections, self.depth, self.info, self.frame_id, stamp, self.config)
        segmented_cloud_msg = make_segmented_cloud(
            detections, self.rgb, self.depth, self.info, self.frame_id, stamp, self.stride, self.max_depth
        )
        det_msg = String(data=dumps_compact({**stamp_to_json(stamp), "detections": detections}))
        return detections, box_msg, segmented_cloud_msg, det_msg, elapsed_s

    def spin(self):
        _print_status("Looping backend.detect() and publishing latest RViz topics. Fixed frame: %s" % self.frame_id)
        
        iteration = 0
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            detections, box_msg, segmented_cloud_msg, det_msg, detect_elapsed_s = self._detect_latest(stamp)
            for msg in (self.cloud_msg, self.depth_msg, self.rgb_msg, segmented_cloud_msg):
                msg.header.stamp = stamp
            for marker in box_msg.markers:
                marker.header.stamp = stamp
            publish_timings = [
                _timed_publish("cloud", self.cloud_pub, self.cloud_msg),
                _timed_publish("depth_viz", self.depth_pub, self.depth_msg),
                _timed_publish("rgb", self.rgb_pub, self.rgb_msg),
                _timed_publish("boxes", self.box_pub, box_msg),
                _timed_publish("segmented_cloud", self.segmented_cloud_pub, segmented_cloud_msg),
                _timed_publish("detections", self.det_pub, det_msg),
            ]
            publish_total_s = sum(elapsed_s for _name, elapsed_s in publish_timings)
            iteration += 1
            timing_text = " ".join("%s=%.4fs" % (name, elapsed_s) for name, elapsed_s in publish_timings)
            _print_status(
                "Iteration %d: detect=%.4fs (%.2f Hz) publish_total=%.4fs %s detections=%d"
                % (iteration, detect_elapsed_s, 1.0 / detect_elapsed_s, publish_total_s, timing_text, len(detections))
            )
            


if __name__ == "__main__":
    _print_status("object_detection_visual_test.py loaded")
    try:
        ObjectDetectionVisualTest().spin()
    except KeyboardInterrupt:
        _print_status("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        try:
            rospy.logerr("[object_detection_visual_test] %s", exc)
        except Exception:
            pass
        _print_status("[object_detection_visual_test] %s" % exc)
        raise
