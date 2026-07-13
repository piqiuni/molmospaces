import struct

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .geometry_utils import transform_point_best_effort, transform_point_with_snapshot

try:
    import cv2
except ImportError:
    cv2 = None


def depth_to_meters(depth_values):
    depth = np.asarray(depth_values, dtype=np.float32)
    if depth.size == 0:
        return depth
    finite_positive = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite_positive.size == 0:
        return depth
    if float(np.median(finite_positive)) > 20.0:
        return depth / 1000.0
    return depth


def point_dict_to_list(value):
    if isinstance(value, dict):
        return [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [value[0], value[1], value[2]]
    return None


def get_label(det):
    return str(
        det.get("semantic_class")
        or det.get("semantic_class_raw")
        or det.get("class")
        or det.get("label")
        or det.get("semantic_name")
        or det.get("name")
        or "object"
    )


def get_confidence(det):
    return float(det.get("confidence", det.get("conf", 0.0)) or 0.0)


def bbox_xyxy(det):
    box = det.get("bbox_xyxy") or det.get("bbox") or det.get("box_2d") or det.get("xyxy")
    if isinstance(box, dict):
        return [box.get("x1", box.get("xmin")), box.get("y1", box.get("ymin")), box.get("x2", box.get("xmax")), box.get("y2", box.get("ymax"))]
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [box[0], box[1], box[2], box[3]]
    return None


def optical_to_rviz_camera(point):
    return [float(point[2]), float(-point[0]), float(-point[1])]


def optical_size_to_rviz_camera(size):
    return [float(size[2]), float(size[0]), float(size[1])]


def is_virtual_camera_frame(frame_id):
    return str(frame_id or "") == "semantic_test_camera"


def is_optical_frame(frame_id):
    return "optical" in str(frame_id or "").lower()


def camera_point_from_uvd(u, v, depth, info, frame_id=None):
    fx, fy = info.K[0], info.K[4]
    cx, cy = info.K[2], info.K[5]
    x = float((u - cx) * depth / fx)
    y = float((v - cy) * depth / fy)
    z = float(depth)
    if is_virtual_camera_frame(frame_id) or not is_optical_frame(frame_id):
        return z, -x, -y
    return x, y, z


def optical_point_from_uvd(u, v, depth, info):
    fx, fy = info.K[0], info.K[4]
    cx, cy = info.K[2], info.K[5]
    x = float((u - cx) * depth / fx)
    y = float((v - cy) * depth / fy)
    z = float(depth)
    return x, y, z


def optical_to_robot_frame(point):
    return float(point[2]), float(-point[0]), float(-point[1])


def pack_rgb_float(r, g, b):
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


def color_for_index(index):
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (30, 144, 255),
        (255, 215, 0),
        (186, 85, 211),
        (255, 140, 0),
        (64, 224, 208),
        (220, 20, 60),
    ]
    return palette[index % len(palette)]


def mask_rows_cols(mask, height, width):
    if mask is None:
        return None, None
    if isinstance(mask, dict):
        rows = np.asarray(mask.get("rows", []), dtype=np.int32)
        cols = np.asarray(mask.get("cols", []), dtype=np.int32)
    else:
        mask_array = np.asarray(mask)
        if mask_array.ndim != 2 or mask_array.shape[0] != height or mask_array.shape[1] != width:
            return None, None
        rows, cols = np.nonzero(mask_array > 0)
        rows = rows.astype(np.int32)
        cols = cols.astype(np.int32)
    if rows.size != cols.size or rows.size == 0:
        return None, None
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[valid]
    cols = cols[valid]
    if rows.size == 0:
        return None, None
    return rows, cols


def _paint_mask_pixels(image, rows, cols, color, alpha=0.65):
    color = np.asarray(color, dtype=np.float32)
    painted = image[rows, cols].astype(np.float32)
    image[rows, cols] = (painted * (1.0 - alpha) + color * alpha).astype(np.uint8)


def render_detection_overlay(rgb, detections):
    image = np.ascontiguousarray(rgb).copy()
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    if cv2 is None:
        height, width = image.shape[:2]
        for idx, det in enumerate(detections):
            color = np.asarray(color_for_index(idx), dtype=np.uint8)
            bbox = bbox_xyxy(det)
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            image[y1:y2, x1:x1 + 2] = color
            image[y1:y2, max(x2 - 2, x1):x2] = color
            image[y1:y1 + 2, x1:x2] = color
            image[max(y2 - 2, y1):y2, x1:x2] = color
            rows, cols = mask_rows_cols(det.get("mask"), height, width)
            if rows is not None:
                _paint_mask_pixels(image, rows, cols, color, alpha=0.65)
        return image

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    for idx, det in enumerate(detections):
        color = color_for_index(idx)
        bbox = bbox_xyxy(det)
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{get_label(det)} {get_confidence(det):.2f}"
        cv2.putText(
            image_bgr,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        rows, cols = mask_rows_cols(det.get("mask"), height, width)
        if rows is not None:
            _paint_mask_pixels(image_bgr, rows, cols, color, alpha=0.7)
            contours_mask = np.zeros((height, width), dtype=np.uint8)
            contours_mask[rows, cols] = 255
            contours, _hierarchy = cv2.findContours(contours_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(image_bgr, contours, -1, color, 2)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def sparse_mask_pixels(det, image_shape, stride):
    height, width = image_shape[:2]
    stride = max(1, int(stride))
    rows, cols = mask_rows_cols(det.get("mask"), height, width)
    if rows is not None:
        return np.stack([cols[::stride], rows[::stride]], axis=1)
    bbox = bbox_xyxy(det)
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    cols = np.arange(x1, x2, stride, dtype=np.int32)
    rows = np.arange(y1, y2, stride, dtype=np.int32)
    if cols.size == 0 or rows.size == 0:
        return None
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    return np.stack([grid_cols.reshape(-1), grid_rows.reshape(-1)], axis=1)


def sparse_mask_pixels_no_bbox_fallback(det, image_shape, stride):
    height, width = image_shape[:2]
    stride = max(1, int(stride))
    rows, cols = mask_rows_cols(det.get("mask"), height, width)
    if rows is not None:
        return np.stack([cols[::stride], rows[::stride]], axis=1)
    return None


def make_segmented_cloud(detections, rgb, depth, info, frame_id, stamp, stride, max_depth):
    points = []
    height, width = depth.shape[:2]
    depth_m = depth_to_meters(depth)
    for idx, det in enumerate(detections):
        pixels = sparse_mask_pixels(det, depth.shape, stride)
        if pixels is None or pixels.shape[0] == 0:
            continue
        color = color_for_index(idx)
        cols = np.clip(pixels[:, 0], 0, width - 1)
        rows = np.clip(pixels[:, 1], 0, height - 1)
        sampled_depths = depth_m[rows, cols].astype(np.float32)
        valid = np.isfinite(sampled_depths) & (sampled_depths > 0.0) & (sampled_depths <= float(max_depth))
        cols = cols[valid]
        rows = rows[valid]
        sampled_depths = sampled_depths[valid]
        for col, row, depth_value in zip(cols, rows, sampled_depths):
            px, py, pz = camera_point_from_uvd(col, row, depth_value, info, frame_id=frame_id)
            points.append([px, py, pz, pack_rgb_float(*color)])

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]
    header = Header(stamp=stamp, frame_id=frame_id)
    return pc2.create_cloud(header, fields, points)


def world_points_from_detection_mask(
    det,
    depth,
    info,
    source_frame,
    target_frame,
    stamp,
    stride,
    max_depth,
    tf_listener,
    tf_snapshot=None,
):
    world_points = []
    height, width = depth.shape[:2]
    depth_m = depth_to_meters(depth)
    pixels = sparse_mask_pixels_no_bbox_fallback(det, depth.shape, stride)
    if pixels is None or pixels.shape[0] == 0:
        return world_points
    cols = np.clip(pixels[:, 0], 0, width - 1)
    rows = np.clip(pixels[:, 1], 0, height - 1)
    sampled_depths = depth_m[rows, cols].astype(np.float32)
    valid = np.isfinite(sampled_depths) & (sampled_depths > 0.0) & (sampled_depths <= float(max_depth))
    cols = cols[valid]
    rows = rows[valid]
    sampled_depths = sampled_depths[valid]
    for col, row, depth_value in zip(cols, rows, sampled_depths):
        px, py, pz = camera_point_from_uvd(col, row, depth_value, info, frame_id=source_frame)
        if tf_snapshot is not None:
            try:
                wx, wy, wz = transform_point_with_snapshot(tf_snapshot, (px, py, pz))
            except Exception:
                continue
        else:
            try:
                (wx, wy, wz), _used_stamp = transform_point_best_effort(
                    tf_listener, target_frame, source_frame, stamp, (px, py, pz)
                )
            except Exception:
                continue
        world_points.append((float(wx), float(wy), float(wz)))
    return world_points


def make_segmented_cloud_world(
    detections,
    rgb,
    depth,
    info,
    source_frame,
    target_frame,
    stamp,
    stride,
    max_depth,
    tf_listener,
    tf_snapshot=None,
):
    points = []
    height, width = depth.shape[:2]
    depth_m = depth_to_meters(depth)
    for idx, det in enumerate(detections):
        color = color_for_index(idx)
        world_points = world_points_from_detection_mask(
            det, depth, info, source_frame, target_frame, stamp, stride, max_depth, tf_listener, tf_snapshot=tf_snapshot
        )
        if not world_points:
            continue
        for wx, wy, wz in world_points:
            points.append([wx, wy, wz, pack_rgb_float(*color)])

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]
    header = Header(stamp=stamp, frame_id=target_frame)
    return pc2.create_cloud(header, fields, points)


def box_from_detection(det, depth, info, frame_id, config):
    depth_m = depth_to_meters(depth)
    center = point_dict_to_list(det.get("box3d_center"))
    size = point_dict_to_list(det.get("box3d_size") or det.get("size"))
    if center is not None and size is not None:
        if is_virtual_camera_frame(frame_id):
            return optical_to_rviz_camera(center), optical_size_to_rviz_camera(size)
        return [float(center[0]), float(center[1]), float(center[2])], [float(size[0]), float(size[1]), float(size[2])]

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

    bbox = bbox_xyxy(det)
    if bbox is None:
        pos = det.get("camera_position") or det.get("position_3d") or det.get("position")
        center = point_dict_to_list(pos)
        if center is not None:
            if det.get("projection_method") and is_virtual_camera_frame(frame_id):
                center = optical_to_rviz_camera(center)
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

    roi = depth_m[y1:y2 + 1, x1:x2 + 1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size == 0:
        return None, None
    z = float(np.median(valid))
    center_u = 0.5 * (x1 + x2)
    center_v = 0.5 * (y1 + y2)
    center = list(camera_point_from_uvd(center_u, center_v, z, info, frame_id=frame_id))
    fx, fy = info.K[0], info.K[4]
    width_m = max(0.05, abs((x2 - x1) * z / fx))
    height_m = max(0.05, abs((y2 - y1) * z / fy))
    depth_m = float(config.get("fallback_box_depth", 0.25))
    return center, [depth_m, width_m, height_m]


def make_box_markers(detections, depth, info, frame_id, stamp, config):
    markers = MarkerArray()
    clear = Marker()
    clear.header.stamp = stamp
    clear.header.frame_id = frame_id
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    for idx, det in enumerate(detections):
        center, size = box_from_detection(det, depth, info, frame_id, config)
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
        text.text = "%s %.2f" % (get_label(det), get_confidence(det))
        text.lifetime = rospy.Duration(0.0)
        markers.markers.append(text)
    return markers


def make_box_markers_world(detections, depth, info, source_frame, target_frame, stamp, config, tf_listener, tf_snapshot=None):
    markers = MarkerArray()
    clear = Marker()
    clear.header.stamp = stamp
    clear.header.frame_id = target_frame
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    for idx, det in enumerate(detections):
        world_points = world_points_from_detection_mask(
            det,
            depth,
            info,
            source_frame,
            target_frame,
            stamp,
            config.get("point_stride", 4),
            config.get("max_depth_m", 8.0),
            tf_listener,
            tf_snapshot=tf_snapshot,
        )
        if not world_points:
            continue
        world_points_np = np.asarray(world_points, dtype=np.float32)
        mins = world_points_np.min(axis=0)
        maxs = world_points_np.max(axis=0)
        center = (0.5 * (mins + maxs)).tolist()
        size = np.maximum(maxs - mins, 0.0).tolist()

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = target_frame
        marker.ns = "semantic_detection_boxes_world"
        marker.id = idx * 2
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(0.02, float(size[0]))
        marker.scale.y = max(0.02, float(size[1]))
        marker.scale.z = max(0.02, float(size[2]))
        marker.color.r = 1.0
        marker.color.g = 0.45
        marker.color.b = 0.1
        marker.color.a = 0.28
        marker.lifetime = rospy.Duration(0.0)
        markers.markers.append(marker)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = target_frame
        text.ns = "semantic_detection_labels_world"
        text.id = idx * 2 + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(float(center[0]), float(center[1]), float(center[2]) + 0.5 * marker.scale.z + 0.08)
        text.pose.orientation.w = 1.0
        text.scale.z = float(config.get("label_height", 0.12))
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = "%s %.2f" % (get_label(det), get_confidence(det))
        text.lifetime = rospy.Duration(0.0)
        markers.markers.append(text)
    return markers
