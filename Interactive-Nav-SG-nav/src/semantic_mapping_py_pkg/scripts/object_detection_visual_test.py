#!/usr/bin/env python3
import argparse
import logging
import math
import os
import socket
import struct
import sys
import xmlrpc.client
import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

from semantic_mapping_py_pkg.detector_backends import make_detector_backend
from semantic_mapping_py_pkg.messages import dumps_compact, stamp_to_json
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param


def _print_status(message):
    print(message, flush=True)


def _patch_roslogging_findcaller_for_py311():
    if sys.version_info < (3, 11):
        return
    try:
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_semantic_mapping_py_safe", False):
        return

    def _safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 3:
            return result[0], result[1], result[2], None
        return result

    _safe_find_caller._semantic_mapping_py_safe = True
    roslogging.RospyLogger.findCaller = _safe_find_caller
    _print_status("Applied Python 3.11 rospy logging workaround")


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

    for key in ("backend", "provider", "external_url"):
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


def _get_label(det):
    return str(det.get("semantic_class") or det.get("class") or det.get("label") or
               det.get("semantic_name") or det.get("name") or "object")


def _get_confidence(det):
    return float(det.get("confidence", det.get("conf", 0.0)) or 0.0)


def _bbox_xyxy(det):
    box = det.get("bbox_xyxy") or det.get("bbox") or det.get("box_2d") or det.get("xyxy")
    if isinstance(box, dict):
        return [box.get("x1", box.get("xmin")), box.get("y1", box.get("ymin")),
                box.get("x2", box.get("xmax")), box.get("y2", box.get("ymax"))]
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [box[0], box[1], box[2], box[3]]
    return None


def _point_dict_to_list(value):
    if isinstance(value, dict):
        return [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [value[0], value[1], value[2]]
    return None


def _optical_to_rviz_camera(point):
    # Backend depth projection uses optical coordinates: x right, y down, z forward.
    return [float(point[2]), float(-point[0]), float(-point[1])]


def _optical_size_to_rviz_camera(size):
    return [float(size[2]), float(size[0]), float(size[1])]


def _box_from_detection(det, depth, info, config):
    center = _point_dict_to_list(det.get("box3d_center"))
    size = _point_dict_to_list(det.get("box3d_size") or det.get("size"))
    if center is not None and size is not None:
        return _optical_to_rviz_camera(center), _optical_size_to_rviz_camera(size)

    box3d = det.get("box_3d") or det.get("bbox_3d")
    if isinstance(box3d, dict):
        center = box3d.get("center") or box3d.get("position")
        size = box3d.get("size") or box3d.get("dimensions") or box3d.get("extent")
        if isinstance(center, dict):
            center = [center.get("x", 0.0), center.get("y", 0.0), center.get("z", 0.0)]
        if isinstance(size, dict):
            size = [size.get("x", 0.2), size.get("y", 0.2), size.get("z", 0.2)]
        if isinstance(center, (list, tuple)) and len(center) >= 3 and isinstance(size, (list, tuple)) and len(size) >= 3:
            return [float(center[0]), float(center[1]), float(center[2])], [float(size[0]), float(size[1]), float(size[2])]

    bbox = _bbox_xyxy(det)
    if bbox is None:
        pos = det.get("camera_position") or det.get("position_3d") or det.get("position")
        center = _point_dict_to_list(pos)
        if center is not None:
            if det.get("projection_method"):
                center = _optical_to_rviz_camera(center)
            size = [float(config.get("fallback_box_depth", 0.25))] * 3
            return center, size
        return None, None

    height, width = depth.shape[:2]
    if any(v is None for v in bbox):
        return None, None
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    if config.get("bbox_format", "xyxy") == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2
    x1, x2 = sorted([max(0, min(width - 1, x1)), max(0, min(width - 1, x2))])
    y1, y2 = sorted([max(0, min(height - 1, y1)), max(0, min(height - 1, y2))])
    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = depth[y1:y2 + 1, x1:x2 + 1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size == 0:
        return None, None
    z = float(np.median(valid))
    center_u = 0.5 * (x1 + x2)
    center_v = 0.5 * (y1 + y2)
    center = list(_camera_point(center_u, center_v, z, info))
    fx, fy = info.K[0], info.K[4]
    width_m = max(0.05, abs((x2 - x1) * z / fx))
    height_m = max(0.05, abs((y2 - y1) * z / fy))
    depth_m = float(config.get("fallback_box_depth", 0.25))
    return center, [depth_m, width_m, height_m]


def _make_box_markers(detections, depth, info, frame_id, stamp, config):
    markers = MarkerArray()
    for idx, det in enumerate(detections):
        center, size = _box_from_detection(det, depth, info, config)
        if center is None:
            continue

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "semantic_detection_boxes"
        marker.id = idx * 2
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(0.02, size[0])
        marker.scale.y = max(0.02, size[1])
        marker.scale.z = max(0.02, size[2])
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.32
        marker.lifetime = rospy.Duration(0.0)
        markers.markers.append(marker)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = frame_id
        text.ns = "semantic_detection_labels"
        text.id = idx * 2 + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(center[0], center[1], center[2] + 0.5 * marker.scale.z + 0.08)
        text.pose.orientation.w = 1.0
        text.scale.z = float(config.get("label_height", 0.12))
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = "%s %.2f" % (_get_label(det), _get_confidence(det))
        text.lifetime = rospy.Duration(0.0)
        markers.markers.append(text)
    return markers


class ObjectDetectionVisualTest:
    def __init__(self):
        _patch_roslogging_findcaller_for_py311()
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
        backend_name = detector_config.get("backend", "mock_empty")
        backend = make_detector_backend(backend_name, detector_config, frames=frames)
        _print_status(
            "Running backend=%s provider=%s rgb=%s depth=%s" % (
                backend_name,
                detector_config.get("provider", "external_http"),
                rgb_path,
                depth_path,
            )
        )
        self.detections = backend.detect(self.rgb, self.depth, self.info, self.stamp, self.frame_id)
        threshold = float(detector_config.get("confidence_threshold", 0.0))
        self.detections = [det for det in self.detections if _get_confidence(det) >= threshold]
        _print_status("Detections: %s" % self.detections)

        self.cloud_msg = _make_cloud(self.rgb, self.depth, self.info, self.frame_id, self.stamp, self.stride, self.max_depth)
        self.depth_msg = _image_msg(_depth_viz(self.depth, self.max_depth), "bgr8", self.frame_id, self.stamp)
        self.rgb_msg = _image_msg(self.rgb, "rgb8", self.frame_id, self.stamp)
        self.box_msg = _make_box_markers(self.detections, self.depth, self.info, self.frame_id, self.stamp, self.config)
        self.det_msg = String(data=dumps_compact({**stamp_to_json(self.stamp), "detections": self.detections}))

        self.cloud_pub = rospy.Publisher(self.config.get("cloud_topic", "/semantic_mapping/test/rgb_depth_cloud"),
                                         PointCloud2, queue_size=1, latch=True)
        self.depth_pub = rospy.Publisher(self.config.get("depth_viz_topic", "/semantic_mapping/test/depth_viz"),
                                         Image, queue_size=1, latch=True)
        self.rgb_pub = rospy.Publisher(self.config.get("rgb_topic", "/semantic_mapping/test/rgb_image"),
                                       Image, queue_size=1, latch=True)
        self.box_pub = rospy.Publisher(self.config.get("box_topic", "/semantic_mapping/test/boxes_3d"),
                                       MarkerArray, queue_size=1, latch=True)
        self.det_pub = rospy.Publisher(self.config.get("detections_topic", "/semantic_mapping/test/detections"),
                                       String, queue_size=1, latch=True)

        rospy.loginfo("[object_detection_visual_test] backend=%s detections=%d frame=%s cloud_points=%d",
                      backend_name, len(self.detections), self.frame_id, self.cloud_msg.width)

    def spin(self):
        _print_status("Publishing RViz topics. Fixed frame: %s" % self.frame_id)
        rate = rospy.Rate(max(self.publish_rate, 0.1))
        last_print = rospy.Time.now()
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            for msg in (self.cloud_msg, self.depth_msg, self.rgb_msg):
                msg.header.stamp = stamp
            for marker in self.box_msg.markers:
                marker.header.stamp = stamp
            self.cloud_pub.publish(self.cloud_msg)
            self.depth_pub.publish(self.depth_msg)
            self.rgb_pub.publish(self.rgb_msg)
            self.box_pub.publish(self.box_msg)
            self.det_pub.publish(self.det_msg)
            if (stamp - last_print).to_sec() >= 2.0:
                _print_status("Published %d detections" % len(self.detections))
                last_print = stamp
            rate.sleep()


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
