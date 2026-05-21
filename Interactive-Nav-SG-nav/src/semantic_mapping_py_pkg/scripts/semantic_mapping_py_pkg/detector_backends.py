import base64
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .geometry_utils import point_dict


def _normalize_label(value):
    return str(value or "").strip().lower().replace(" ", "_")


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASS_MAPPING_PATH = PACKAGE_ROOT / "config" / "object_class_mapping.json"


DEFAULT_CLASS_MAPPING = {
    "bed": "bed",
    "bunk_bed": "bed",
    "playpen": "bed",
    "guzheng": "bed",
    "quilting": "bed",
    "chair": "chair",
    "electric_chair": "chair",
    "office_chair": "chair",
    "bowl": "bowl",
    "plate": "plate",
    "glass_plate": "plate",
    "cup": "cup",
    "bottle": "bottle",
    "kitchen_table": "table",
    "dinning_table": "table",
    "dining_table": "table",
    "table": "table",
    "cabinet": "cabinet",
    "drawer": "drawer",
    "fridge": "fridge",
    "refrigerator": "fridge",
    "microwave": "microwave",
    "shelf": "shelf",
    "shelfs": "shelf",
    "storage_box": "box",
    "box": "box",
    "sofa": "sofa",
    "couch": "sofa",
    "toilet": "toilet",
    "sink": "sink",
    "door": "door",
    "window_sill": "window",
}


DEFAULT_CLASS_KEYWORDS = [
    ("bed", "bed"),
    ("chair", "chair"),
    ("bowl", "bowl"),
    ("bottle", "bottle"),
    ("table", "table"),
    ("cabinet", "cabinet"),
    ("drawer", "drawer"),
    ("fridge", "fridge"),
    ("refrigerator", "fridge"),
    ("microwave", "microwave"),
    ("shelf", "shelf"),
    ("sofa", "sofa"),
    ("couch", "sofa"),
    ("toilet", "toilet"),
    ("sink", "sink"),
    ("door", "door"),
    ("window", "window"),
    ("cup", "cup"),
    ("plate", "plate"),
    ("box", "box"),
]


def _bbox_from_detection(det):
    bbox = det.get("bbox") or det.get("box_2d") or det.get("xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _sparse_mask_from_array(mask_array):
    if mask_array is None:
        return None, 0
    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        return None, 0
    rows, cols = np.nonzero(mask > 0)
    if rows.size == 0:
        return None, 0
    return {
        "rows": rows.astype(np.int32).tolist(),
        "cols": cols.astype(np.int32).tolist(),
    }, int(rows.size)


def _resolve_mapping_path(raw_path):
    if not isinstance(raw_path, str):
        return None
    text = raw_path.strip()
    if not text:
        return None
    package_token = "$(find semantic_mapping_py_pkg)"
    if package_token in text:
        text = text.replace(package_token, str(PACKAGE_ROOT))
    return Path(text).expanduser()


def _load_mapping_config(raw_mapping):
    if isinstance(raw_mapping, dict):
        return {_normalize_label(k): _normalize_label(v) for k, v in raw_mapping.items() if v}
    path = _resolve_mapping_path(raw_mapping) if raw_mapping else DEFAULT_CLASS_MAPPING_PATH
    if path is not None and path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {_normalize_label(k): _normalize_label(v) for k, v in data.items() if v}
        except Exception:
            return {}
    return {}


def _map_open_vocabulary_class(raw_name, mapping_config=None):
    raw_label = _normalize_label(raw_name)
    if not raw_label:
        return "unknown_open_set"
    merged = dict(DEFAULT_CLASS_MAPPING)
    if mapping_config:
        merged.update(mapping_config)
    if raw_label in merged:
        return merged[raw_label]
    for needle, normalized in DEFAULT_CLASS_KEYWORDS:
        if needle in raw_label:
            return normalized
    return "unknown_open_set"


def _clip_bbox(bbox, image_shape):
    if bbox is None:
        return None
    height, width = image_shape[:2]
    x1 = max(0, min(width - 1, bbox[0]))
    y1 = max(0, min(height - 1, bbox[1]))
    x2 = max(x1 + 1, min(width, bbox[2]))
    y2 = max(y1 + 1, min(height, bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _depth_to_meters(depth_values):
    if depth_values is None:
        return None
    depth = np.asarray(depth_values, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return None
    positive = depth[finite & (depth > 0.0)]
    if positive.size == 0:
        return None
    if positive.mean() > 20.0:
        depth = depth / 1000.0
    return depth


def _camera_intrinsics(camera_info, image_shape):
    if camera_info is not None and len(camera_info.K) >= 9 and camera_info.K[0] > 0.0 and camera_info.K[4] > 0.0:
        return {
            "fx": float(camera_info.K[0]),
            "fy": float(camera_info.K[4]),
            "cx": float(camera_info.K[2]),
            "cy": float(camera_info.K[5]),
        }
    height, width = image_shape[:2]
    fx = fy = 0.5 * float(width)
    return {
        "fx": fx,
        "fy": fy,
        "cx": 0.5 * float(width - 1),
        "cy": 0.5 * float(height - 1),
    }


def _pixels_to_camera_points(pixels, depths, intrinsics):
    fx = max(intrinsics["fx"], 1e-6)
    fy = max(intrinsics["fy"], 1e-6)
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    cols = pixels[:, 0].astype(np.float32)
    rows = pixels[:, 1].astype(np.float32)
    z = depths.astype(np.float32)
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _trim_points(points, trim_ratio):
    if points.shape[0] < 4:
        return points
    trim_ratio = min(max(float(trim_ratio), 0.0), 0.45)
    if trim_ratio <= 0.0:
        return points
    mins = np.quantile(points, trim_ratio, axis=0)
    maxs = np.quantile(points, 1.0 - trim_ratio, axis=0)
    mask = np.all((points >= mins) & (points <= maxs), axis=1)
    trimmed = points[mask]
    return trimmed if trimmed.shape[0] >= 4 else points


def _transform_point(tf_listener, target_frame, source_frame, stamp, point):
    from geometry_msgs.msg import PointStamped

    point_msg = PointStamped()
    point_msg.header.frame_id = source_frame
    point_msg.header.stamp = stamp
    point_msg.point.x = float(point[0])
    point_msg.point.y = float(point[1])
    point_msg.point.z = float(point[2])
    transformed = tf_listener.transformPoint(target_frame, point_msg)
    return np.array([transformed.point.x, transformed.point.y, transformed.point.z], dtype=np.float32)


def _transform_point_best_effort(tf_listener, target_frame, source_frame, stamp, point):
    import rospy

    candidate_stamps = []
    if stamp is not None:
        candidate_stamps.append(stamp)
    candidate_stamps.append(rospy.Time(0))
    last_exc = None
    for candidate_stamp in candidate_stamps:
        try:
            return _transform_point(tf_listener, target_frame, source_frame, candidate_stamp, point), candidate_stamp
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no TF candidate stamp available")


def _sample_center_point(depth_image_m, bbox, kernel_size):
    kernel_size = max(1, int(kernel_size))
    x1, y1, x2, y2 = bbox
    cx = int(round(0.5 * (x1 + x2 - 1)))
    cy = int(round(0.5 * (y1 + y2 - 1)))
    half = kernel_size // 2
    xs = slice(max(0, cx - half), min(depth_image_m.shape[1], cx + half + 1))
    ys = slice(max(0, cy - half), min(depth_image_m.shape[0], cy + half + 1))
    window = depth_image_m[ys, xs]
    valid = window[np.isfinite(window) & (window > 0.0)]
    if valid.size == 0:
        return None
    return cx, cy, float(np.median(valid))


def _sample_box_points(depth_image_m, bbox, point_stride):
    x1, y1, x2, y2 = bbox
    cols = np.arange(x1, x2, max(1, int(point_stride)), dtype=np.int32)
    rows = np.arange(y1, y2, max(1, int(point_stride)), dtype=np.int32)
    if cols.size == 0 or rows.size == 0:
        return None, None
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    depths = depth_image_m[grid_rows, grid_cols]
    valid = np.isfinite(depths) & (depths > 0.0)
    if not valid.any():
        return None, None
    pixels = np.stack([grid_cols[valid], grid_rows[valid]], axis=1)
    values = depths[valid]
    return pixels, values


def _mask_to_pixels(mask, bbox, image_shape, point_stride):
    if mask is None:
        return None
    height, width = image_shape[:2]
    if isinstance(mask, dict):
        rows = np.asarray(mask.get("rows", []), dtype=np.int32)
        cols = np.asarray(mask.get("cols", []), dtype=np.int32)
        if rows.size == cols.size and rows.size > 0:
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            if valid.any():
                return np.stack([cols[valid], rows[valid]], axis=1)
    if isinstance(mask, list):
        mask_array = np.asarray(mask)
        if mask_array.ndim == 2 and mask_array.shape == (height, width):
            rows, cols = np.nonzero(mask_array)
            if rows.size > 0:
                stride = max(1, int(point_stride))
                return np.stack([cols[::stride], rows[::stride]], axis=1)
    return None


def _normalize_detection(
    det,
    bbox,
    projection_method,
    camera_center,
    world_center,
    size,
    source_frame,
    world_frame=None,
    world_stamp=None,
):
    confidence = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
    semantic_class = det.get("semantic_class") or det.get("class") or det.get("semantic_name")
    semantic_class_raw = det.get("semantic_class_raw") or semantic_class
    normalized = {
        "semantic_class": _normalize_label(semantic_class),
        "semantic_class_raw": _normalize_label(semantic_class_raw),
        "confidence": confidence,
        "bbox": [int(v) for v in bbox] if bbox is not None else None,
        "position": point_dict(camera_center[0], camera_center[1], camera_center[2]),
        "projection_method": projection_method,
        "source_frame": str(source_frame or ""),
    }
    if world_center is not None:
        normalized["world_position"] = point_dict(world_center[0], world_center[1], world_center[2])
        normalized["world_frame"] = str(world_frame or "")
        if world_stamp is not None:
            normalized["world_position_stamp"] = {
                "secs": int(world_stamp.secs),
                "nsecs": int(world_stamp.nsecs),
                "is_latest_tf": bool(getattr(world_stamp, "is_zero", lambda: False)()),
            }
    if size is not None:
        normalized["size"] = point_dict(size[0], size[1], size[2])
        normalized["box3d_center"] = point_dict(camera_center[0], camera_center[1], camera_center[2])
        normalized["box3d_size"] = point_dict(size[0], size[1], size[2])
    instance_id = det.get("instance_id")
    if instance_id:
        normalized["instance_id"] = str(instance_id)
    if "mask_area" in det:
        normalized["mask_area"] = int(det["mask_area"])
    if "source_model" in det:
        normalized["source_model"] = str(det["source_model"])
    return normalized


class ObjectDetectorBackend:
    def detect(self, rgb_image, depth_image, camera_info, stamp, frame_id):
        raise NotImplementedError


class RawDetectionProvider:
    def detect_2d(self, rgb_image, depth_image, camera_info, stamp):
        raise NotImplementedError


class NoDetectionDetector(ObjectDetectorBackend):
    def detect(self, rgb_image, depth_image, camera_info, stamp, frame_id):
        return []


class MockEmptyDetector(NoDetectionDetector):
    pass


class MockEmptyProvider(RawDetectionProvider):
    def detect_2d(self, rgb_image, depth_image, camera_info, stamp):
        return []


class ExternalHttpProvider(RawDetectionProvider):
    def __init__(self, url, timeout=5.0, include_depth=False):
        self.url = url
        self.timeout = float(timeout)
        self.include_depth = bool(include_depth)

    def detect_2d(self, rgb_image, depth_image, camera_info, stamp):
        import cv2

        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        if not ok:
            return []

        payload = {
            "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "stamp_sec": int(stamp.secs),
            "stamp_nsec": int(stamp.nsecs),
        }
        if self.include_depth and depth_image is not None:
            depth = np.asarray(depth_image)
            payload["depth"] = {
                "dtype": str(depth.dtype),
                "shape": list(depth.shape),
                "data_b64": base64.b64encode(depth.tobytes(order="C")).decode("ascii"),
            }
        if camera_info is not None:
            payload["camera_info"] = {
                "width": int(camera_info.width),
                "height": int(camera_info.height),
                "K": list(camera_info.K),
            }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ValueError, TimeoutError):
            return []

        if isinstance(data, dict):
            data = data.get("detections", [])
        return data if isinstance(data, list) else []


class YoloeLocalProvider(RawDetectionProvider):
    def __init__(
        self,
        model_path,
        confidence_threshold=0.35,
        iou_threshold=0.7,
        imgsz=640,
        device="cuda:0",
        max_detections=50,
        keep_unknown_open_set=False,
        class_mapping=None,
    ):
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.max_detections = max(1, int(max_detections))
        self.keep_unknown_open_set = bool(keep_unknown_open_set)
        self.class_mapping = _load_mapping_config(class_mapping)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLOE

            self._model = YOLOE(self.model_path)
        return self._model

    def detect_2d(self, rgb_image, depth_image, camera_info, stamp):
        model = self._get_model()
        results = model.predict(
            source=rgb_image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            agnostic_nms=True,
            max_det=self.max_detections,
            verbose=False,
            save=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names or {}
        masks = result.masks
        if boxes is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        mask_array = None
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_array = masks.data.detach().cpu().numpy()

        detections = []
        for idx in range(len(xyxy)):
            raw_name = str(names.get(int(classes[idx]), str(classes[idx])))
            semantic_class_raw = _normalize_label(raw_name)
            semantic_class = _map_open_vocabulary_class(raw_name, self.class_mapping)
            if semantic_class == "unknown_open_set" and not self.keep_unknown_open_set:
                continue

            sparse_mask = None
            mask_area = 0
            if mask_array is not None and idx < len(mask_array):
                sparse_mask, mask_area = _sparse_mask_from_array(mask_array[idx] > 0.5)

            detections.append(
                {
                    "semantic_class_raw": semantic_class_raw,
                    "semantic_class": semantic_class,
                    "confidence": float(confs[idx]),
                    "bbox": [float(v) for v in xyxy[idx].tolist()],
                    "mask": sparse_mask,
                    "mask_area": int(mask_area),
                    "source_model": Path(self.model_path).name,
                }
            )
        return detections


class ExternalHttpDetector(ObjectDetectorBackend):
    def __init__(self, provider):
        self.provider = provider

    def detect(self, rgb_image, depth_image, camera_info, stamp, frame_id):
        detections = []
        for det in self.provider.detect_2d(rgb_image, depth_image, camera_info, stamp):
            bbox = _clip_bbox(_bbox_from_detection(det), rgb_image.shape)
            semantic_class = det.get("semantic_class") or det.get("class") or det.get("semantic_name")
            if not semantic_class:
                continue
            normalized = {
                "semantic_class": _normalize_label(semantic_class),
                "semantic_class_raw": _normalize_label(det.get("semantic_class_raw") or semantic_class),
                "confidence": float(det.get("confidence", det.get("conf", 0.0)) or 0.0),
                "bbox": bbox,
                "projection_method": det.get("projection_method", "provider_passthrough"),
                "source_frame": str(frame_id or ""),
            }
            for key in (
                "position",
                "world_position",
                "size",
                "instance_id",
                "mask_area",
                "box3d_center",
                "box3d_size",
                "source_model",
            ):
                if key in det:
                    normalized[key] = det[key]
            detections.append(normalized)
        return detections


class CenterProjectionDetector(ObjectDetectorBackend):
    def __init__(self, provider, world_frame, point_stride=4, max_depth_m=8.0, center_kernel_size=5):
        import rospy
        import tf

        self.provider = provider
        self.world_frame = world_frame
        self.point_stride = max(1, int(point_stride))
        self.max_depth_m = float(max_depth_m)
        self.center_kernel_size = max(1, int(center_kernel_size))
        self.rospy = rospy
        self.tf_listener = tf.TransformListener()

    def detect(self, rgb_image, depth_image, camera_info, stamp, frame_id):
        if depth_image is None:
            return []
        depth_image_m = _depth_to_meters(depth_image)
        if depth_image_m is None:
            return []
        intrinsics = _camera_intrinsics(camera_info, rgb_image.shape)
        detections = []
        for det in self.provider.detect_2d(rgb_image, depth_image, camera_info, stamp):
            bbox = _clip_bbox(_bbox_from_detection(det), rgb_image.shape)
            if bbox is None:
                continue
            sample = _sample_center_point(depth_image_m, bbox, self.center_kernel_size)
            if sample is None:
                continue
            col, row, depth_m = sample
            if depth_m <= 0.0 or depth_m > self.max_depth_m:
                continue
            camera_center = _pixels_to_camera_points(
                np.asarray([[col, row]], dtype=np.float32),
                np.asarray([depth_m], dtype=np.float32),
                intrinsics,
            )[0]
            world_center = None
            world_stamp = None
            if frame_id:
                try:
                    world_center, used_stamp = _transform_point_best_effort(
                        self.tf_listener, self.world_frame, frame_id, stamp, camera_center
                    )
                    world_stamp = used_stamp
                    if used_stamp == self.rospy.Time(0):
                        self.rospy.logwarn_throttle(
                            2.0,
                            "[object_detection_node] TF center projection fallback to latest transform for %s <- %s",
                            self.world_frame,
                            frame_id,
                        )
                except Exception as exc:
                    self.rospy.logwarn_throttle(2.0, "[object_detection_node] TF center projection failed: %s", exc)
            detections.append(
                _normalize_detection(
                    det=det,
                    bbox=bbox,
                    projection_method="bbox_center_projection",
                    camera_center=camera_center,
                    world_center=world_center,
                    size=None,
                    source_frame=frame_id,
                    world_frame=self.world_frame,
                    world_stamp=world_stamp,
                )
            )
        return detections


class SamBox3DDetector(ObjectDetectorBackend):
    def __init__(
        self,
        provider,
        world_frame,
        point_stride=4,
        max_depth_m=8.0,
        min_valid_points=12,
        trim_ratio=0.1,
    ):
        import rospy
        import tf

        self.provider = provider
        self.world_frame = world_frame
        self.point_stride = max(1, int(point_stride))
        self.max_depth_m = float(max_depth_m)
        self.min_valid_points = max(4, int(min_valid_points))
        self.trim_ratio = float(trim_ratio)
        self.rospy = rospy
        self.tf_listener = tf.TransformListener()
        self._tf_unavailable_warned = False

    def _try_transform_center(self, frame_id, stamp, camera_center):
        if not self.world_frame or not frame_id or self.world_frame == frame_id:
            return None
        try:
            try:
                if self.tf_listener.canTransform(self.world_frame, frame_id, stamp):
                    world_center, used_stamp = _transform_point_best_effort(
                        self.tf_listener, self.world_frame, frame_id, stamp, camera_center
                    )
                    return world_center, used_stamp
            except Exception:
                pass
            world_center, used_stamp = _transform_point_best_effort(
                self.tf_listener, self.world_frame, frame_id, stamp, camera_center
            )
            if used_stamp == self.rospy.Time(0) and not self._tf_unavailable_warned:
                self.rospy.logwarn(
                    "[object_detection_node] TF %s <- %s unavailable at exact stamp; fallback to latest transform",
                    self.world_frame,
                    frame_id,
                )
                self._tf_unavailable_warned = True
            return world_center, used_stamp
        except Exception as exc:
            if not self._tf_unavailable_warned:
                self.rospy.logwarn(
                    "[object_detection_node] TF mask box projection skipped; using camera-frame box3d_center only: %s",
                    exc,
                )
                self._tf_unavailable_warned = True
            return None, None

    def detect(self, rgb_image, depth_image, camera_info, stamp, frame_id):
        if depth_image is None:
            return []
        depth_image_m = _depth_to_meters(depth_image)
        if depth_image_m is None:
            return []
        intrinsics = _camera_intrinsics(camera_info, rgb_image.shape)
        detections = []
        for det in self.provider.detect_2d(rgb_image, depth_image, camera_info, stamp):
            bbox = _clip_bbox(_bbox_from_detection(det), rgb_image.shape)
            if bbox is None:
                continue
            pixels = _mask_to_pixels(det.get("mask"), bbox, rgb_image.shape, self.point_stride)
            if pixels is None or pixels.shape[0] == 0:
                pixels, values = _sample_box_points(depth_image_m, bbox, self.point_stride)
            else:
                cols = np.clip(pixels[:, 0], 0, depth_image_m.shape[1] - 1)
                rows = np.clip(pixels[:, 1], 0, depth_image_m.shape[0] - 1)
                values = depth_image_m[rows, cols]
                valid = np.isfinite(values) & (values > 0.0)
                pixels = pixels[valid]
                values = values[valid]
            if pixels is None or values is None or values.size < self.min_valid_points:
                continue
            valid_depth = values <= self.max_depth_m
            pixels = pixels[valid_depth]
            values = values[valid_depth]
            if values.size < self.min_valid_points:
                continue
            points = _pixels_to_camera_points(pixels, values, intrinsics)
            points = _trim_points(points, self.trim_ratio)
            if points.shape[0] < self.min_valid_points:
                continue
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            camera_center = 0.5 * (mins + maxs)
            size = np.maximum(maxs - mins, 0.0)
            world_center, world_stamp = self._try_transform_center(frame_id, stamp, camera_center)
            enriched = dict(det)
            enriched["mask_area"] = int(points.shape[0])
            detections.append(
                _normalize_detection(
                    det=enriched,
                    bbox=bbox,
                    projection_method="mask_box3d_projection",
                    camera_center=camera_center,
                    world_center=world_center,
                    size=size,
                    source_frame=frame_id,
                    world_frame=self.world_frame,
                    world_stamp=world_stamp,
                )
            )
        return detections


def make_raw_detection_provider(kind, config):
    kind = str(kind or "external_http").strip().lower()
    if kind == "yoloe_local":
        return YoloeLocalProvider(
            model_path=config.get(
                "model_path",
                "/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt",
            ),
            confidence_threshold=config.get("confidence_threshold", 0.35),
            iou_threshold=config.get("iou_threshold", 0.7),
            imgsz=config.get("imgsz", 640),
            device=config.get("device", "cuda:0"),
            max_detections=config.get("max_detections", 50),
            keep_unknown_open_set=config.get("keep_unknown_open_set", False),
            class_mapping=config.get("class_mapping", {}),
        )
    if kind == "external_http":
        return ExternalHttpProvider(
            config.get("external_url", "http://127.0.0.1:8000/detect"),
            timeout=config.get("timeout", 5.0),
            include_depth=config.get("include_depth", False),
        )
    return MockEmptyProvider()


def make_detector_backend(kind, config, frames=None):
    kind = str(kind or "mock_empty").strip().lower()
    frames = frames or {}
    if kind in ("no_detection", "none", "disabled", "off"):
        return NoDetectionDetector()
    if kind == "mock_empty":
        return MockEmptyDetector()

    provider = make_raw_detection_provider(config.get("provider", "external_http"), config)
    world_frame = frames.get("world_frame", "tf_frame_map")

    if kind == "external_http":
        return ExternalHttpDetector(provider)
    if kind == "yolo_world_center_projection":
        return CenterProjectionDetector(
            provider=provider,
            world_frame=world_frame,
            point_stride=config.get("point_stride", 4),
            max_depth_m=config.get("max_depth_m", 8.0),
            center_kernel_size=config.get("center_kernel_size", 5),
        )
    if kind == "yolo_world_sam_box3d":
        return SamBox3DDetector(
            provider=provider,
            world_frame=world_frame,
            point_stride=config.get("point_stride", 4),
            max_depth_m=config.get("max_depth_m", 8.0),
            min_valid_points=config.get("min_valid_points", 12),
            trim_ratio=config.get("trim_ratio", 0.1),
        )
    if kind == "yoloe_pf_box3d":
        return SamBox3DDetector(
            provider=provider,
            world_frame=world_frame,
            point_stride=config.get("point_stride", 4),
            max_depth_m=config.get("max_depth_m", 8.0),
            min_valid_points=config.get("min_valid_points", 12),
            trim_ratio=config.get("trim_ratio", 0.1),
        )
    if kind == "sam3_box3d":
        return SamBox3DDetector(
            provider=provider,
            world_frame=world_frame,
            point_stride=config.get("point_stride", 4),
            max_depth_m=config.get("max_depth_m", 8.0),
            min_valid_points=config.get("min_valid_points", 12),
            trim_ratio=config.get("trim_ratio", 0.1),
        )
    return MockEmptyDetector()
