#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from semantic_mllm_py_pkg import load_env_file
from semantic_mllm_py_pkg.client import MLLMClient
from semantic_mllm_py_pkg.env import client_config_from_env
from semantic_mllm_py_pkg.schemas import validate_attribute_patch


class InteractionAttributeInferenceNode:
    def __init__(self) -> None:
        load_env_file(os.environ.get("SEMANTIC_DECISION_ENV_FILE"))
        rospy.init_node("interaction_attribute_inference_node")
        topics = rospy.get_param("~topics", {}) or {}
        self.image_topic = topics.get("rgb_image", "/molmo_spaces/head_camera/image")
        self.detection_topic = topics.get(
            "object_detections", "/semantic_mapping/object_detections"
        )
        self.gt_observations_topic = topics.get(
            "gt_observations", "/semantic_mapping/gt_observations"
        )
        self.output_topic = topics.get(
            "attribute_updates", "/semantic_mapping/attribute_updates"
        )
        self.min_interval_s = float(rospy.get_param("~min_interval_s", 2.0))
        self.max_pending = max(1, int(rospy.get_param("~max_pending", 1)))
        self.model_name = str(rospy.get_param("~model_name", "") or "")
        self.bridge = CvBridge()
        self.client = MLLMClient(client_config_from_env(model=self.model_name or None))
        self.publisher = rospy.Publisher(self.output_topic, String, queue_size=2)
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_stamp = 0.0
        self.last_request: dict[str, float] = {}
        self.pending: set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=self.max_pending)
        rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)
        rospy.Subscriber(self.detection_topic, String, self._detection_callback, queue_size=1)
        rospy.Subscriber(
            self.gt_observations_topic, String, self._detection_callback, queue_size=1
        )
        rospy.loginfo(
            "[interaction_attribute_inference] image=%s detections=%s output=%s model=%s",
            self.image_topic,
            self.detection_topic,
            self.output_topic,
            self.client.config.model,
        )

    def _image_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "attribute image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image.copy()
            self.latest_stamp = message.header.stamp.to_sec() or time.time()

    def _detection_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        detections = (
            payload.get("detections") or payload.get("observations")
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(detections, list):
            return
        with self.lock:
            image = None if self.latest_image is None else self.latest_image.copy()
            image_stamp = self.latest_stamp
        if image is None:
            return
        episode_id = str(payload.get("episode_id") or "") if isinstance(payload, dict) else ""
        frame_id = str(payload.get("frame_index") or "") if isinstance(payload, dict) else ""
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            object_id = str(
                detection.get("instance_id")
                or detection.get("object_id")
                or detection.get("source_object_name")
                or detection.get("name")
                or ""
            )
            if not object_id or not self._try_reserve(object_id):
                continue
            crop = self._crop(image, detection)
            if crop is None:
                self._release(object_id)
                continue
            self.executor.submit(
                self._infer,
                object_id,
                detection,
                crop,
                episode_id,
                frame_id,
                image_stamp,
            )

    def _try_reserve(self, object_id: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if object_id in self.pending:
                return False
            if now - self.last_request.get(object_id, 0.0) < self.min_interval_s:
                return False
            self.pending.add(object_id)
            return True

    def _release(self, object_id: str) -> None:
        with self.lock:
            self.pending.discard(object_id)

    def _infer(
        self,
        object_id: str,
        detection: dict,
        crop,
        episode_id: str,
        frame_id: str,
        stamp: float,
    ) -> None:
        try:
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return
            image_data = "data:image/jpeg;base64," + __import__("base64").b64encode(encoded).decode("ascii")
            response = self.client.request_json(
                role="attribute_inference",
                instruction=(
                    "Infer only visible semantic interaction attributes. Do not invent exact "
                    "joint axes, ranges, or control trajectories. Return JSON fields object_id, "
                    "interactable, interaction_class, coarse_state, interaction_parts, "
                    "affordances, expected_effect, confidence, and evidence_frame_ids."
                ),
                context={
                    "object_id": object_id,
                    "name": detection.get("name") or detection.get("semantic_name"),
                    "category": detection.get("category"),
                    "geometry": {
                        "position": detection.get("position"),
                        "aabb_size": detection.get("aabb_size"),
                    },
                    "episode_id": episode_id,
                    "frame_id": frame_id,
                },
                images=[image_data],
            )
            if response.error or response.payload is None:
                return
            patch = validate_attribute_patch(response.payload)
            patch.update(
                {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stamp_sec": stamp,
                    "source": "mllm_attribute_inference",
                    "model_name": self.client.config.model,
                    "evidence_frame_ids": [frame_id] if frame_id else [],
                }
            )
            self.publisher.publish(
                String(
                    data=json.dumps(
                        {
                            "episode_id": episode_id,
                            "stamp_sec": stamp,
                            "updates": [patch],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            )
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "attribute inference failed: %s", exc)
        finally:
            with self.lock:
                self.pending.discard(object_id)
                self.last_request[object_id] = time.monotonic()

    @staticmethod
    def _crop(image, detection):
        box = detection.get("projected_bbox_2d") or detection.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        height, width = image.shape[:2]
        x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return image[y0:y1, x0:x1]


if __name__ == "__main__":
    InteractionAttributeInferenceNode()
    rospy.spin()
