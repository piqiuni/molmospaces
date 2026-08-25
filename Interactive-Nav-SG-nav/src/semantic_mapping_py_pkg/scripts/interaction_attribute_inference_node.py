#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import threading
import time
from io import BytesIO

import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import cv2
except ImportError:  # ROS may run under a Python interpreter without OpenCV.
    cv2 = None
    from PIL import Image as PILImage

from semantic_mllm_py_pkg import load_env_file
from semantic_mllm_py_pkg.client import MLLMClient
from semantic_mllm_py_pkg.env import client_config_from_env
from semantic_mllm_py_pkg.schemas import (
    build_attribute_patch_response_schema,
    validate_attribute_patch,
    validate_room_attribute_patch,
)
from semantic_mapping_py_pkg.attribute_filter import (
    DEFAULT_EXCLUDE_KEYWORDS,
    DEFAULT_INTERACTION_KEYWORDS,
    is_interaction_attribute_candidate,
)
from semantic_mapping_py_pkg.attribute_inference_queue import LatestPriorityRequestQueue
from semantic_mapping_py_pkg.graph_rules import bbox_area, segmentation_pixel_count
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311
from semantic_mapping_py_pkg.ros_params import get_nested_param


class InteractionAttributeInferenceNode:
    def __init__(self) -> None:
        env_path = os.environ.get("SEMANTIC_DECISION_ENV_FILE")
        # The explicit evaluator env file is authoritative; otherwise an
        # inherited API key can silently authenticate against the wrong account.
        load_env_file(env_path, override=bool(env_path))
        patch_roslogging_findcaller_for_py311()
        rospy.init_node("interaction_attribute_inference_node")
        topics = get_nested_param(rospy, "topics", {}) or {}
        attribute_config = get_nested_param(rospy, "attribute_inference", {}) or {}
        room_mllm_config = get_nested_param(rospy, "room_mllm", {}) or {}
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
        self.targeted_refresh_topic = topics.get(
            "attribute_refresh_requests",
            "/semantic_mapping/attribute_refresh_requests",
        )
        self.room_request_topic = topics.get(
            "room_attribute_requests", "/semantic_mapping/room_attribute_requests"
        )
        self.room_output_topic = topics.get(
            "room_attribute_updates", "/semantic_mapping/room_attribute_updates"
        )
        self.interaction_result_topic = topics.get(
            "interaction_result", "/semantic_mapping/interaction_result"
        )
        self.status_topic = topics.get(
            "attribute_inference_status",
            "/semantic_mapping/attribute_inference_status",
        )
        self.min_interval_s = float(rospy.get_param("~min_interval_s", 5.0))
        self.success_refresh_interval_s = max(
            0.0, float(rospy.get_param("~success_refresh_interval_s", 120.0))
        )
        # Unknown/weak portal judgments are safety-gated by the decision
        # layer, so keeping them cached for the normal 120 s would make a
        # nearby, better view unusable.  Retry at a bounded short interval;
        # material view buckets in _state_signature may also trigger a fresh
        # request, but min_interval_s prevents flooding.
        self.uncertain_portal_refresh_interval_s = max(
            0.0,
            float(rospy.get_param("~uncertain_portal_refresh_interval_s", 5.0)),
        )
        self.uncertain_portal_confidence = max(
            0.0,
            min(
                1.0,
                float(rospy.get_param("~uncertain_portal_confidence", 0.65)),
            ),
        )
        self.worker_count = max(1, int(rospy.get_param("~worker_count", 1)))
        self.max_queue_size = max(
            self.worker_count, int(rospy.get_param("~max_queue_size", 8))
        )
        self.min_visible_fraction = max(
            0.0, float(rospy.get_param("~min_visible_fraction", 0.20))
        )
        self.min_visible_pixels = max(
            1, int(rospy.get_param("~min_visible_pixels", 64))
        )
        self.min_bbox_area_px = max(
            1, int(rospy.get_param("~min_bbox_area_px", 512))
        )
        self.max_distance_m = max(
            0.0, float(rospy.get_param("~max_distance_m", 6.0))
        )
        self.required_consecutive_observations = max(
            1, int(rospy.get_param("~required_consecutive_observations", 2))
        )
        self.include_keywords = tuple(
            str(value).casefold()
            for value in rospy.get_param(
                "~include_keywords", list(DEFAULT_INTERACTION_KEYWORDS)
            )
            if str(value).strip()
        )
        self.exclude_keywords = tuple(
            str(value).casefold()
            for value in rospy.get_param(
                "~exclude_keywords", list(DEFAULT_EXCLUDE_KEYWORDS)
            )
            if str(value).strip()
        )
        # Keep Module 1 on the same explicitly selected local/remote model as
        # Modules 2 and 3.  Legacy launch arguments may still contain an old
        # model name, but an env-file choice is intentional and takes priority.
        self.model_name = str(
            os.environ.get("SEMANTIC_MODEL_NAME")
            or rospy.get_param("~model_name", "")
            or ""
        )
        self.client = MLLMClient(client_config_from_env(model=self.model_name or None))
        self.request_timeout_s = max(
            0.1, float(rospy.get_param("~request_timeout_s", 8.0))
        )
        self.max_output_tokens = max(
            32,
            int(rospy.get_param("~max_output_tokens", min(self.client.config.max_tokens, 256))),
        )
        self.crop_margin_ratio = max(
            0.0, float(rospy.get_param("~crop_margin_ratio", 0.08))
        )
        self.visual_evidence_max_side_px = max(
            0,
            int(
                rospy.get_param(
                    "~visual_evidence_max_side_px",
                    attribute_config.get("visual_evidence_max_side_px", 1024),
                )
            ),
        )
        # Explicit decision-layer refreshes must overtake discovery work, but
        # they still wait for a later RGB + detection observation.
        self.targeted_refresh_priority = max(
            1.0,
            float(
                rospy.get_param(
                    "~targeted_refresh_priority",
                    attribute_config.get("targeted_refresh_priority", 1000.0),
                )
            ),
        )
        self.room_enabled = bool(room_mllm_config.get("enabled", True))
        self.room_worker_count = max(
            1, int(room_mllm_config.get("worker_count", 1))
        )
        self.room_max_queue_size = max(
            self.room_worker_count,
            int(room_mllm_config.get("max_queue_size", 4)),
        )
        self.room_min_interval_s = max(
            0.0, float(room_mllm_config.get("min_interval_s", self.min_interval_s))
        )
        self.room_success_refresh_interval_s = max(
            0.0,
            float(
                room_mllm_config.get(
                    "success_refresh_interval_s", self.success_refresh_interval_s
                )
            ),
        )
        self.room_request_timeout_s = max(
            0.1, float(room_mllm_config.get("request_timeout_s", self.request_timeout_s))
        )
        self.room_max_output_tokens = max(
            32,
            int(
                room_mllm_config.get(
                    "max_output_tokens", min(self.client.config.max_tokens, 96)
                )
            ),
        )
        self.publisher = rospy.Publisher(self.output_topic, String, queue_size=2)
        self.room_publisher = rospy.Publisher(
            self.room_output_topic, String, queue_size=2
        )
        self.status_publisher = rospy.Publisher(
            self.status_topic, String, queue_size=1, latch=True
        )
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_stamp = 0.0
        self.latest_image_sequence = 0
        self.pending_detection_payload: dict | None = None
        self.filter_counts = {
            "messages_received": 0,
            "received": 0,
            "eligible": 0,
            "enqueued": 0,
            "coalesced": 0,
            "started": 0,
            "completed": 0,
            "stale": 0,
            "expired": 0,
            "failed": 0,
            "filtered": 0,
            "missing_image": 0,
            "targeted_refresh_received": 0,
            "targeted_refresh_armed": 0,
            "targeted_refresh_matched": 0,
            "targeted_refresh_enqueued": 0,
            "targeted_refresh_rejected": 0,
        }
        self.room_counts = {
            "messages_received": 0,
            "received": 0,
            "eligible": 0,
            "enqueued": 0,
            "coalesced": 0,
            "started": 0,
            "completed": 0,
            "stale": 0,
            "expired": 0,
            "failed": 0,
            "filtered": 0,
        }
        self.current_episode_id = ""
        self.request_sequence = 0
        self.last_request: dict[str, float] = {}
        self.pending: dict[str, dict] = {}
        self.completed: dict[str, dict] = {}
        self.generations: dict[str, int] = {}
        self.aliases: dict[str, str] = {}
        self.targeted_refresh_sequence = 0
        self.targeted_refresh_requests: dict[str, dict] = {}
        self.request_queue = LatestPriorityRequestQueue(self.max_queue_size)
        self.room_request_sequence = 0
        self.room_last_request: dict[str, float] = {}
        self.room_pending: dict[str, dict] = {}
        self.room_completed: dict[str, dict] = {}
        self.room_generations: dict[str, int] = {}
        self.room_request_queue = LatestPriorityRequestQueue(self.room_max_queue_size)
        self.shutdown_event = threading.Event()
        self.workers = [
            threading.Thread(target=self._worker_loop, daemon=True)
            for _index in range(self.worker_count)
        ]
        for worker in self.workers:
            worker.start()
        self.room_workers = []
        if self.room_enabled:
            self.room_workers = [
                threading.Thread(target=self._room_worker_loop, daemon=True)
                for _index in range(self.room_worker_count)
            ]
            for worker in self.room_workers:
                worker.start()
        rospy.on_shutdown(self._shutdown)
        rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=2)
        rospy.Subscriber(self.detection_topic, String, self._detection_callback, queue_size=4)
        rospy.Subscriber(
            self.gt_observations_topic, String, self._detection_callback, queue_size=4
        )
        rospy.Subscriber(
            self.interaction_result_topic,
            String,
            self._interaction_result_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.targeted_refresh_topic,
            String,
            self._targeted_refresh_callback,
            queue_size=10,
        )
        if self.room_enabled:
            rospy.Subscriber(
                self.room_request_topic,
                String,
                self._room_request_callback,
                queue_size=4,
            )
        rospy.loginfo(
            "[interaction_attribute_inference] image=%s detections=%s object_output=%s targeted_refresh=%s room=%s room_output=%s model=%s object_workers=%d room_workers=%d",
            self.image_topic,
            self.detection_topic,
            self.output_topic,
            self.targeted_refresh_topic,
            self.room_enabled,
            self.room_output_topic,
            self.client.config.model,
            self.worker_count,
            self.room_worker_count,
        )
        self._publish_status()

    def _image_callback(self, message: Image) -> None:
        try:
            image = self._decode_ros_image(message)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "attribute image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image.copy()
            self.latest_stamp = message.header.stamp.to_sec() or time.time()
            self.latest_image_sequence += 1
        self._process_pending_detections()

    @staticmethod
    def _decode_ros_image(message: Image):
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
        }
        encoding = str(message.encoding or "").casefold()
        channels = channels_by_encoding.get(encoding)
        if channels is None or message.height <= 0 or message.width <= 0:
            raise ValueError(f"unsupported image encoding: {message.encoding}")
        row_width = int(message.step or message.width * channels)
        raw = np.frombuffer(message.data, dtype=np.uint8).reshape(
            int(message.height), row_width
        )
        image = raw[:, : int(message.width) * channels].reshape(
            int(message.height), int(message.width), channels
        )
        if encoding == "rgb8":
            if cv2 is not None:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image[..., ::-1].copy()
        if encoding == "rgba8":
            if cv2 is not None:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image[..., :3][..., ::-1].copy()
        if encoding == "bgra8":
            if cv2 is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image[..., :3].copy()
        return image

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> bytes:
        if cv2 is not None:
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            return encoded.tobytes() if ok else b""
        if image.ndim == 2:
            pil_image = PILImage.fromarray(image, mode="L")
        else:
            pil_image = PILImage.fromarray(image[..., ::-1], mode="RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()

    def _detection_callback(self, message: String) -> None:
        with self.lock:
            self.filter_counts["messages_received"] += 1
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
            self.pending_detection_payload = dict(payload) if isinstance(payload, dict) else {
                "detections": detections
            }
            has_image = self.latest_image is not None
        if has_image:
            self._process_pending_detections()

    def _process_pending_detections(self) -> None:
        with self.lock:
            payload = self.pending_detection_payload
            image = None if self.latest_image is None else self.latest_image.copy()
            image_stamp = self.latest_stamp
            image_sequence = int(self.latest_image_sequence)
            if isinstance(payload, dict) and image is not None:
                self.pending_detection_payload = None
        if not isinstance(payload, dict) or image is None:
            if isinstance(payload, dict):
                with self.lock:
                    self.filter_counts["missing_image"] += 1
                self._publish_status()
            return
        detections = payload.get("detections") or payload.get("observations")
        if not isinstance(detections, list):
            return
        episode_id = str(payload.get("episode_id") or "") if isinstance(payload, dict) else ""
        capture_step = self._capture_step(payload)
        frame_id = "" if capture_step is None else str(capture_step)
        observation_stamp = self._observation_stamp(payload, image_stamp)
        if episode_id:
            self._set_episode(episode_id)
        for expired in self.request_queue.drop_expired():
            self._expire_attribute_request(expired)
        requests = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            with self.lock:
                self.filter_counts["received"] += 1
            object_id = str(
                detection.get("instance_id")
                or detection.get("id")
                or detection.get("object_id")
                or detection.get("source_object_name")
                or detection.get("name")
                or ""
            )
            if not object_id:
                with self.lock:
                    self.filter_counts["filtered"] += 1
                continue
            self._register_aliases(object_id, detection)
            targeted_refresh = self._targeted_refresh_for_detection(
                object_id,
                detection,
                episode_id=episode_id,
                capture_step=capture_step,
                image_sequence=image_sequence,
            )
            if not self._passes_observation_filter(detection):
                with self.lock:
                    self.filter_counts["filtered"] += 1
                continue
            with self.lock:
                self.filter_counts["eligible"] += 1
            signature = self._state_signature(detection)
            visual_evidence = self._compose_attribute_visual_evidence(
                image,
                detection,
                margin_ratio=self.crop_margin_ratio,
            )
            if visual_evidence is None:
                continue
            if targeted_refresh is None:
                self._invalidate_if_state_changed(object_id, signature, episode_id)
                reservation = self._try_reserve(object_id, signature)
            else:
                reservation = self._force_reserve_targeted_refresh(
                    object_id,
                    signature,
                    episode_id,
                    targeted_refresh,
                )
            if reservation is None:
                continue
            enqueued_at = time.monotonic()
            requests.append(
                {
                    "priority": (
                        self.targeted_refresh_priority
                        if targeted_refresh is not None
                        else self._priority(detection)
                    ),
                    "object_id": object_id,
                    "detection": dict(detection),
                    "visual_evidence": visual_evidence,
                    "episode_id": episode_id,
                    "frame_id": frame_id,
                    "image_sequence": image_sequence,
                    "stamp": observation_stamp,
                    "signature": signature,
                    "generation": reservation["generation"],
                    "request_sequence": reservation["request_sequence"],
                    "enqueued_at": enqueued_at,
                    "deadline_monotonic": enqueued_at + self.request_timeout_s,
                    "targeted_refresh": dict(targeted_refresh or {}),
                }
            )
        for expired in self.request_queue.drop_expired():
            self._expire_attribute_request(expired)
        for request_payload in sorted(
            requests, key=lambda item: (-float(item["priority"]), item["object_id"])
        ):
            if self._remaining_request_timeout(
                request_payload.get("deadline_monotonic"), self.request_timeout_s
            ) <= 0.0:
                self._expire_attribute_request(request_payload)
                continue
            accepted, displaced = self.request_queue.put(request_payload)
            if accepted:
                with self.lock:
                    self.filter_counts["enqueued"] += 1
                    if request_payload.get("targeted_refresh"):
                        self.filter_counts["targeted_refresh_enqueued"] += 1
                if request_payload.get("targeted_refresh"):
                    self._consume_targeted_refresh(request_payload["targeted_refresh"])
                if displaced is not None:
                    same_object = str(displaced.get("object_id") or "") == str(
                        request_payload.get("object_id") or ""
                    )
                    if same_object:
                        with self.lock:
                            self.filter_counts["coalesced"] = (
                                self.filter_counts.get("coalesced", 0) + 1
                            )
                    self._release(
                        str(displaced.get("object_id") or ""),
                        int(displaced.get("request_sequence", 0) or 0),
                    )
                    self._publish_updates(
                        str(displaced.get("episode_id") or ""),
                        float(displaced.get("stamp", request_payload["stamp"])),
                        [
                            self._attribute_status_patch(
                                displaced,
                                "stale",
                                error=(
                                    "queue_replaced_by_newer_object_evidence"
                                    if same_object
                                    else "queue_replaced_by_higher_priority_request"
                                ),
                            )
                        ],
                    )
                self._publish_updates(
                    str(request_payload["episode_id"]),
                    float(request_payload["stamp"]),
                    [self._attribute_status_patch(request_payload, "pending")],
                )
            else:
                if request_payload.get("targeted_refresh"):
                    with self.lock:
                        self.filter_counts["targeted_refresh_rejected"] += 1
                # The deadline can elapse after the pre-admission check but
                # before the queue lock is acquired.  Preserve the terminal
                # expiry status instead of silently releasing the reservation.
                if self._remaining_request_timeout(
                    request_payload.get("deadline_monotonic"), self.request_timeout_s
                ) <= 0.0:
                    self._expire_attribute_request(request_payload)
                else:
                    self._release(
                        str(request_payload["object_id"]),
                        int(request_payload["request_sequence"]),
                    )
        self._publish_status()
        rospy.loginfo_throttle(
            10.0,
            "[interaction_attribute_inference] received=%d eligible=%d enqueued=%d "
            "coalesced=%d expired=%d filtered=%d missing_image=%d queue=%d",
            self.filter_counts["received"],
            self.filter_counts["eligible"],
            self.filter_counts["enqueued"],
            self.filter_counts.get("coalesced", 0),
            self.filter_counts.get("expired", 0),
            self.filter_counts["filtered"],
            self.filter_counts["missing_image"],
            len(self.request_queue),
        )

    def _room_request_callback(self, message: String) -> None:
        """Queue no-image room classification independently of object crops."""

        if not self.room_enabled:
            return
        with self.lock:
            self.room_counts["messages_received"] += 1
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        rooms = payload.get("rooms") or []
        if not isinstance(rooms, list):
            return
        episode_id = str(payload.get("episode_id") or "")
        if episode_id:
            self._set_episode(episode_id)
        for expired in self.room_request_queue.drop_expired():
            self._expire_room_request(expired)
        capture_step = self._capture_step(payload)
        stamp = self._observation_stamp(payload, time.time())
        requests = []
        for room in rooms:
            with self.lock:
                self.room_counts["received"] += 1
            if not isinstance(room, dict):
                with self.lock:
                    self.room_counts["filtered"] += 1
                continue
            try:
                room_id = int(room.get("room_id"))
            except (TypeError, ValueError):
                with self.lock:
                    self.room_counts["filtered"] += 1
                continue
            room_node_id = str(room.get("room_node_id") or f"room_{room_id}")
            objects = self._room_objects(room.get("objects"))
            if not objects:
                with self.lock:
                    self.room_counts["filtered"] += 1
                continue
            room_key = room_node_id or f"room_{room_id}"
            signature = self._room_signature(room_id, objects)
            self._invalidate_room_if_state_changed(room_key, signature, episode_id)
            reservation = self._try_reserve_room(room_key, signature)
            if reservation is None:
                continue
            with self.lock:
                self.room_counts["eligible"] += 1
            enqueued_at = time.monotonic()
            requests.append(
                {
                    # LatestPriorityRequestQueue deliberately uses object_id as
                    # its generic de-dup key.  This is a room-node ID here,
                    # and the room queue is completely separate from object
                    # image requests.
                    "object_id": room_key,
                    "priority": self._room_priority(objects),
                    "room_id": room_id,
                    "room_node_id": room_node_id,
                    "objects": objects,
                    "episode_id": episode_id,
                    "capture_step": capture_step,
                    "stamp": stamp,
                    "signature": signature,
                    "generation": reservation["generation"],
                    "request_sequence": reservation["request_sequence"],
                    "enqueued_at": enqueued_at,
                    "deadline_monotonic": enqueued_at + self.room_request_timeout_s,
                }
            )
        for expired in self.room_request_queue.drop_expired():
            self._expire_room_request(expired)
        for request_payload in sorted(
            requests,
            key=lambda item: (-float(item["priority"]), item["object_id"]),
        ):
            if self._remaining_request_timeout(
                request_payload.get("deadline_monotonic"), self.room_request_timeout_s
            ) <= 0.0:
                self._expire_room_request(request_payload)
                continue
            accepted, displaced = self.room_request_queue.put(request_payload)
            room_key = str(request_payload["object_id"])
            if not accepted:
                # Match the object lane: an item can expire while waiting for
                # the queue lock, and must be reported as expired rather than
                # disappearing as an unclassified rejection.
                if self._remaining_request_timeout(
                    request_payload.get("deadline_monotonic"), self.room_request_timeout_s
                ) <= 0.0:
                    self._expire_room_request(request_payload)
                else:
                    self._release_room(room_key, int(request_payload["request_sequence"]))
                continue
            with self.lock:
                self.room_counts["enqueued"] += 1
            if displaced is not None:
                displaced_key = str(displaced.get("object_id") or "")
                displaced_sequence = int(displaced.get("request_sequence", 0) or 0)
                if displaced_key == room_key:
                    with self.lock:
                        self.room_counts["coalesced"] = (
                            self.room_counts.get("coalesced", 0) + 1
                        )
                self._release_room(displaced_key, displaced_sequence)
                self._publish_room_updates(
                    str(displaced.get("episode_id") or ""),
                    float(displaced.get("stamp", stamp) or stamp),
                    [
                        self._room_status_patch(
                            displaced,
                            "stale",
                            error="queue_replaced_by_newer_room_evidence",
                        )
                    ],
                )
            self._publish_room_updates(
                str(request_payload["episode_id"]),
                float(request_payload["stamp"]),
                [self._room_status_patch(request_payload, "pending")],
            )
        self._publish_status()

    @staticmethod
    def _room_objects(raw_objects) -> list[dict]:
        """Allow only room-member object metadata into the no-image lane."""

        if not isinstance(raw_objects, list):
            return []
        normalized = []
        for raw in raw_objects:
            if not isinstance(raw, dict):
                continue
            object_id = str(raw.get("object_id") or raw.get("node_id") or "")
            name = str(raw.get("name") or raw.get("category") or "object")
            if not object_id or not name:
                continue
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            normalized.append(
                {
                    "object_id": object_id,
                    "name": name,
                    "category": str(raw.get("category") or ""),
                    "type": str(raw.get("type") or "object"),
                    "confidence": confidence,
                    "currently_visible": bool(raw.get("currently_visible", False)),
                }
            )
        return sorted(normalized, key=lambda item: (item["object_id"], item["name"]))

    @staticmethod
    def _room_signature(room_id: int, objects: list[dict]) -> str:
        return json.dumps(
            {
                "room_id": int(room_id),
                "objects": [
                    {
                        "object_id": item["object_id"],
                        "name": item["name"].casefold(),
                        "category": item["category"].casefold(),
                        "type": item["type"].casefold(),
                    }
                    for item in objects
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _room_priority(objects: list[dict]) -> float:
        return float(len(objects)) + 0.25 * float(
            sum(1 for item in objects if item.get("currently_visible"))
        )

    def _invalidate_room_if_state_changed(
        self, room_key: str, signature: str, episode_id: str
    ) -> None:
        request_sequence = 0
        with self.lock:
            pending = self.room_pending.get(room_key)
            if pending is None or str(pending.get("signature") or "") == signature:
                return
            if episode_id and str(pending.get("episode_id") or "") not in {"", episode_id}:
                return
            request_sequence = int(pending.get("request_sequence", 0) or 0)
            self.room_generations[room_key] = self.room_generations.get(room_key, 0) + 1
            self.room_pending.pop(room_key, None)
            self.room_last_request.pop(room_key, None)
        discarded = self.room_request_queue.discard(room_key, request_sequence)
        self._publish_discarded_room_requests(
            discarded,
            error="queue_replaced_by_newer_room_evidence",
        )

    def _try_reserve_room(self, room_key: str, signature: str) -> dict | None:
        now = time.monotonic()
        with self.lock:
            if room_key in self.room_pending:
                return None
            completed = self.room_completed.get(room_key)
            if completed is not None and completed.get("signature") == signature:
                if (
                    self.room_success_refresh_interval_s <= 0.0
                    or now - float(completed.get("completed_at", 0.0))
                    < self.room_success_refresh_interval_s
                ):
                    return None
            if now - self.room_last_request.get(room_key, 0.0) < self.room_min_interval_s:
                return None
            self.room_request_sequence += 1
            self.room_pending[room_key] = {
                "request_sequence": self.room_request_sequence,
                "signature": signature,
                "generation": self.room_generations.get(room_key, 0),
                "episode_id": self.current_episode_id,
            }
            return {
                "generation": self.room_generations.get(room_key, 0),
                "request_sequence": self.room_request_sequence,
            }

    def _release_room(self, room_key: str, request_sequence: int | None = None) -> None:
        if not room_key:
            return
        with self.lock:
            pending = self.room_pending.get(room_key)
            if request_sequence is None or int(
                (pending or {}).get("request_sequence", 0) or 0
            ) == int(request_sequence):
                self.room_pending.pop(room_key, None)

    @staticmethod
    def _room_status_patch(request_payload: dict, status: str, error: str = "") -> dict:
        return {
            "room_id": int(request_payload["room_id"]),
            "room_node_id": str(request_payload["room_node_id"]),
            "room_attribute_status": status,
            "observation_capture_step": request_payload.get("capture_step"),
            "observation_stamp_sec": float(request_payload["stamp"]),
            "observation_signature": str(request_payload["signature"]),
            "request_sequence": int(request_payload["request_sequence"]),
            "source": "mllm_room_attribute_inference",
            "error": str(error)[:240],
        }

    def _publish_discarded_room_requests(
        self,
        requests: list[dict] | None,
        *,
        error: str,
    ) -> None:
        """Give directly discarded queued room requests a terminal status."""

        discarded = [dict(item) for item in (requests or []) if isinstance(item, dict)]
        if not discarded:
            return
        with self.lock:
            self.room_counts["coalesced"] = (
                self.room_counts.get("coalesced", 0) + len(discarded)
            )
        for request_payload in discarded:
            self._publish_room_updates(
                str(request_payload.get("episode_id") or ""),
                float(request_payload.get("stamp", time.time()) or time.time()),
                [self._room_status_patch(request_payload, "stale", error=error)],
            )

    def _publish_status(self) -> None:
        with self.lock:
            payload = {
                "episode_id": self.current_episode_id,
                "filter_counts": dict(self.filter_counts),
                "queue_size": len(self.request_queue),
                "pending_requests": len(self.pending),
                "room_enabled": self.room_enabled,
                "room_counts": dict(self.room_counts),
                "room_queue_size": len(self.room_request_queue),
                "room_pending_requests": len(self.room_pending),
                "has_latest_image": self.latest_image is not None,
                "latest_image_sequence": int(self.latest_image_sequence),
                "has_pending_detection": self.pending_detection_payload is not None,
                "targeted_refresh_topic": self.targeted_refresh_topic,
                "targeted_refresh_pending": len(self.targeted_refresh_requests),
            }
        self.status_publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _set_episode(self, episode_id: str) -> None:
        stale_request_ids = []
        stale_room_request_ids = []
        with self.lock:
            if not episode_id or episode_id == self.current_episode_id:
                return
            stale_request_ids = [
                (object_id, int(payload.get("request_sequence", 0) or 0))
                for object_id, payload in self.pending.items()
            ]
            stale_room_request_ids = [
                (room_key, int(payload.get("request_sequence", 0) or 0))
                for room_key, payload in self.room_pending.items()
            ]
            self.current_episode_id = episode_id
            self.last_request.clear()
            self.pending.clear()
            self.completed.clear()
            self.generations.clear()
            self.aliases.clear()
            self.room_last_request.clear()
            self.room_pending.clear()
            self.room_completed.clear()
            self.room_generations.clear()
            self.targeted_refresh_requests.clear()
        for object_id, request_sequence in stale_request_ids:
            self.request_queue.discard(object_id, request_sequence)
        for room_key, request_sequence in stale_room_request_ids:
            self.room_request_queue.discard(room_key, request_sequence)

    def _register_aliases(self, object_id: str, detection: dict) -> None:
        if not object_id:
            return
        aliases = {
            object_id,
            str(detection.get("id") or ""),
            str(detection.get("instance_id") or ""),
            str(detection.get("object_id") or ""),
            str(detection.get("source_object_name") or ""),
            str(detection.get("name") or ""),
        }
        with self.lock:
            for alias in aliases:
                if alias:
                    self.aliases[alias] = object_id

    def _interaction_result_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        identifiers = {
            str(payload.get("object_id") or ""),
            str(payload.get("instance_id") or ""),
            str(payload.get("node_id") or ""),
            str(payload.get("source_object_name") or ""),
        }
        identifiers.discard("")
        if not identifiers:
            return
        with self.lock:
            object_ids = {self.aliases.get(identifier, identifier) for identifier in identifiers}
            for object_id in object_ids:
                self.generations[object_id] = self.generations.get(object_id, 0) + 1
                self.completed.pop(object_id, None)
                self.last_request.pop(object_id, None)
                self.pending.pop(object_id, None)
        for object_id in object_ids:
            self.request_queue.discard(object_id)

    @staticmethod
    def _parse_targeted_refresh_payload(payload: object) -> dict | None:
        """Validate the compact public request for a fresh Module-1 view.

        The request intentionally carries no image, pose, category, or private
        simulator data.  It only identifies the object and the earliest
        acceptable public capture step; the node waits for a later local RGB +
        detection pair before creating an inference request.
        """

        if not isinstance(payload, dict):
            return None
        object_id = str(payload.get("object_id") or "").strip()
        if not object_id:
            return None
        try:
            minimum_capture_step = int(payload.get("minimum_capture_step"))
        except (TypeError, ValueError):
            return None
        if minimum_capture_step < 0:
            return None
        expected_node_type = str(payload.get("expected_node_type") or "").strip().casefold()
        if expected_node_type not in {"", "container", "portal"}:
            return None
        return {
            "object_id": object_id,
            "episode_id": str(payload.get("episode_id") or "").strip(),
            "minimum_capture_step": minimum_capture_step,
            "reason": str(payload.get("reason") or "targeted_refresh").strip()[:160]
            or "targeted_refresh",
            "request_id": str(payload.get("request_id") or "").strip()[:96],
            "expected_node_type": expected_node_type,
        }

    def _targeted_refresh_callback(self, message: String) -> None:
        with self.lock:
            self.filter_counts["targeted_refresh_received"] += 1
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            with self.lock:
                self.filter_counts["targeted_refresh_rejected"] += 1
            self._publish_status()
            return
        request = self._parse_targeted_refresh_payload(payload)
        if request is None:
            with self.lock:
                self.filter_counts["targeted_refresh_rejected"] += 1
            self._publish_status()
            return
        requested_episode = str(request["episode_id"])
        with self.lock:
            current_episode = self.current_episode_id
        if requested_episode and current_episode and requested_episode != current_episode:
            with self.lock:
                self.filter_counts["targeted_refresh_rejected"] += 1
            rospy.logwarn(
                "[interaction_attribute_inference] rejected targeted refresh for %s: "
                "episode %s != current %s",
                request["object_id"],
                requested_episode,
                current_episode,
            )
            self._publish_status()
            return
        if requested_episode and not current_episode:
            self._set_episode(requested_episode)
        with self.lock:
            canonical_object_id = self.aliases.get(
                request["object_id"], request["object_id"]
            )
            self.targeted_refresh_sequence += 1
            refresh_sequence = self.targeted_refresh_sequence
            self.targeted_refresh_requests[canonical_object_id] = {
                **request,
                "request_key": canonical_object_id,
                "refresh_sequence": refresh_sequence,
                # Require an RGB image that arrived after this request, even
                # if an old detection packet is still waiting to be consumed.
                "minimum_image_sequence": int(self.latest_image_sequence),
                "armed_at_monotonic": time.monotonic(),
            }
            self.filter_counts["targeted_refresh_armed"] += 1
        rospy.loginfo(
            "[interaction_attribute_inference] armed targeted refresh object=%s "
            "minimum_capture_step=%d reason=%s",
            request["object_id"],
            request["minimum_capture_step"],
            request["reason"],
        )
        self._publish_status()

    def _targeted_refresh_for_detection(
        self,
        object_id: str,
        detection: dict,
        *,
        episode_id: str,
        capture_step: int | None,
        image_sequence: int,
    ) -> dict | None:
        """Return an armed request only for a strictly later RGB/detection view."""

        if capture_step is None:
            return None
        identifiers = {
            str(object_id),
            str(detection.get("instance_id") or ""),
            str(detection.get("id") or ""),
            str(detection.get("object_id") or ""),
            str(detection.get("source_object_name") or ""),
            str(detection.get("name") or ""),
        }
        identifiers.discard("")
        with self.lock:
            canonical_identifiers = {
                self.aliases.get(identifier, identifier) for identifier in identifiers
            }
            for request_key, request in sorted(
                self.targeted_refresh_requests.items(),
                key=lambda item: int(item[1].get("refresh_sequence", 0) or 0),
                reverse=True,
            ):
                requested_episode = str(request.get("episode_id") or "")
                if requested_episode and requested_episode != str(episode_id or ""):
                    continue
                requested_object_id = str(request.get("object_id") or "")
                if (
                    request_key not in canonical_identifiers
                    and requested_object_id not in identifiers
                    and requested_object_id not in canonical_identifiers
                ):
                    continue
                try:
                    minimum_capture_step = int(request.get("minimum_capture_step", -1))
                    minimum_image_sequence = int(
                        request.get("minimum_image_sequence", -1)
                    )
                except (TypeError, ValueError):
                    continue
                if int(capture_step) <= minimum_capture_step:
                    continue
                if int(image_sequence) <= minimum_image_sequence:
                    continue
                return dict(request)
        return None

    def _force_reserve_targeted_refresh(
        self,
        object_id: str,
        signature: str,
        episode_id: str,
        refresh: dict,
    ) -> dict | None:
        """Replace stale discovery work with the explicitly requested view."""

        previous_request_sequence: int | None = None
        previous_snapshot: dict | None = None
        with self.lock:
            if episode_id and self.current_episode_id and episode_id != self.current_episode_id:
                return None
            previous = self.pending.get(object_id)
            if previous is not None:
                previous_snapshot = dict(previous)
                previous_request_sequence = int(
                    previous.get("request_sequence", 0) or 0
                )
            # A running old request cannot be removed from a worker thread, so
            # move the generation first.  Its response is then rejected by the
            # normal current-request guard before it can publish stale state.
            self.generations[object_id] = self.generations.get(object_id, 0) + 1
            generation = self.generations[object_id]
            self.completed.pop(object_id, None)
            self.last_request.pop(object_id, None)
            self.request_sequence += 1
            request_sequence = self.request_sequence
            self.pending[object_id] = {
                "request_sequence": request_sequence,
                "signature": signature,
                "generation": generation,
                "episode_id": self.current_episode_id,
                "targeted_refresh": dict(refresh),
            }
            self.filter_counts["targeted_refresh_matched"] += 1
        if previous_request_sequence is not None:
            discarded = self.request_queue.discard(
                object_id, previous_request_sequence
            )
            if not discarded and previous_snapshot is not None:
                discarded = [
                    {
                        **previous_snapshot,
                        "object_id": object_id,
                        "episode_id": previous_snapshot.get("episode_id")
                        or episode_id,
                        "stamp": previous_snapshot.get("stamp", time.time()),
                        "frame_id": previous_snapshot.get("frame_id", ""),
                        "signature": previous_snapshot.get("signature", ""),
                        "targeted_refresh": previous_snapshot.get("targeted_refresh")
                        or {},
                    }
                ]
            self._publish_discarded_attribute_requests(
                discarded,
                error="queue_replaced_by_newer_object_evidence",
            )
        return {"generation": generation, "request_sequence": request_sequence}

    def _consume_targeted_refresh(self, refresh: dict) -> None:
        request_key = str(refresh.get("request_key") or "")
        refresh_sequence = int(refresh.get("refresh_sequence", 0) or 0)
        if not request_key or refresh_sequence <= 0:
            return
        with self.lock:
            current = self.targeted_refresh_requests.get(request_key) or {}
            if int(current.get("refresh_sequence", 0) or 0) == refresh_sequence:
                self.targeted_refresh_requests.pop(request_key, None)

    @classmethod
    def _attribute_status_patch(
        cls, request_payload: dict, status: str, error: str = ""
    ) -> dict:
        patch = {
            "object_id": str(request_payload.get("object_id") or ""),
            "attribute_status": str(status),
            "observation_capture_step": cls._frame_index(
                str(request_payload.get("frame_id") or "")
            ),
            "request_sequence": int(request_payload.get("request_sequence", 0) or 0),
            "observation_signature": str(request_payload.get("signature") or ""),
            "source": "mllm_attribute_inference",
            "error": str(error)[:240],
        }
        refresh = request_payload.get("targeted_refresh") or {}
        if isinstance(refresh, dict) and refresh:
            patch.update(
                {
                    "targeted_refresh": True,
                    "targeted_refresh_request_id": str(
                        refresh.get("request_id") or ""
                    ),
                    "targeted_refresh_sequence": int(
                        refresh.get("refresh_sequence", 0) or 0
                    ),
                    "targeted_refresh_reason": str(refresh.get("reason") or ""),
                    "targeted_refresh_minimum_capture_step": int(
                        refresh.get("minimum_capture_step", -1)
                    ),
                    "targeted_refresh_image_sequence": int(
                        request_payload.get("image_sequence", -1)
                    ),
                }
            )
        return patch

    def _publish_discarded_attribute_requests(
        self,
        requests: list[dict] | None,
        *,
        error: str,
    ) -> None:
        """Give directly discarded queued M1 requests a terminal status."""

        discarded = [dict(item) for item in (requests or []) if isinstance(item, dict)]
        if not discarded:
            return
        with self.lock:
            self.filter_counts["coalesced"] = (
                self.filter_counts.get("coalesced", 0) + len(discarded)
            )
        for request_payload in discarded:
            self._publish_updates(
                str(request_payload.get("episode_id") or ""),
                float(request_payload.get("stamp", time.time()) or time.time()),
                [self._attribute_status_patch(request_payload, "stale", error=error)],
            )

    @staticmethod
    def _observation_stamp(payload: object, fallback: float) -> float:
        if isinstance(payload, dict):
            try:
                return float(payload.get("stamp_sec", fallback) or fallback)
            except (TypeError, ValueError):
                pass
        return float(fallback)

    @staticmethod
    def _capture_step(payload: object) -> int | None:
        """Prefer the evaluator capture step over legacy frame-index aliases."""

        if not isinstance(payload, dict):
            return None
        value = payload.get("capture_step")
        if value is None:
            value = payload.get("frame_index")
        try:
            capture_step = int(value)
        except (TypeError, ValueError):
            return None
        return capture_step if capture_step >= 0 else None

    def _invalidate_if_state_changed(
        self, object_id: str, signature: str, episode_id: str
    ) -> None:
        if not object_id:
            return
        request_sequence = 0
        pending_snapshot: dict | None = None
        with self.lock:
            pending = self.pending.get(object_id)
            if pending is None or str(pending.get("signature") or "") == signature:
                return
            if episode_id and str(pending.get("episode_id") or "") not in {"", episode_id}:
                return
            pending_snapshot = dict(pending)
            request_sequence = int(pending.get("request_sequence", 0) or 0)
            self.generations[object_id] = self.generations.get(object_id, 0) + 1
            self.pending.pop(object_id, None)
            self.last_request.pop(object_id, None)
        discarded = self.request_queue.discard(object_id, request_sequence)
        if not discarded and pending_snapshot is not None:
            # A worker may already have removed the item from the local queue.
            # Publish a synthetic terminal record so the public state cannot
            # remain ``pending`` while the in-flight response is suppressed by
            # the generation guard.
            discarded = [
                {
                    **pending_snapshot,
                    "object_id": object_id,
                    "episode_id": pending_snapshot.get("episode_id") or episode_id,
                    "stamp": pending_snapshot.get("stamp", time.time()),
                    "frame_id": pending_snapshot.get("frame_id", ""),
                    "signature": pending_snapshot.get("signature", ""),
                    "targeted_refresh": pending_snapshot.get("targeted_refresh") or {},
                }
            ]
        self._publish_discarded_attribute_requests(
            discarded,
            error="queue_replaced_by_newer_object_evidence",
        )

    def _passes_observation_filter(self, detection: dict) -> bool:
        if not is_interaction_attribute_candidate(
            detection, self.include_keywords, self.exclude_keywords
        ):
            return False
        box = detection.get("bbox_2d") or detection.get("projected_bbox_2d") or detection.get("bbox")
        segmentation = detection.get("segmentation")
        if segmentation is None:
            segmentation = detection.get("mask")
        if segmentation is None:
            segmentation = detection.get("mask_rle")
        if segmentation is None:
            segmentation = detection.get("segmentation_rle")
        visible_pixels = int(
            detection.get("visible_pixels", segmentation_pixel_count(segmentation))
            or 0
        )
        visible_fraction = detection.get("visible_fraction")
        if visible_fraction is None:
            area = bbox_area(box)
            visible_fraction = min(1.0, visible_pixels / area) if area > 0.0 else 0.0
        visible_fraction = float(visible_fraction or 0.0)
        if visible_fraction < self.min_visible_fraction:
            return False
        if visible_pixels < self.min_visible_pixels:
            return False
        distance_m = detection.get("distance_m")
        if (
            distance_m is not None
            and self.max_distance_m > 0.0
            and float(distance_m) > self.max_distance_m
        ):
            return False
        consecutive = int(detection.get("consecutive_observations", 0) or 0)
        if consecutive and consecutive < self.required_consecutive_observations:
            return False
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return False
        x0, y0, x1, y1 = [float(value) for value in box[:4]]
        return abs(x1 - x0) * abs(y1 - y0) >= self.min_bbox_area_px

    @staticmethod
    def _public_detection_bbox(detection: dict) -> list[float] | None:
        """Return one finite public detector box for a completed M1 patch."""

        raw = (
            detection.get("bbox_2d")
            or detection.get("projected_bbox_2d")
            or detection.get("bbox")
        )
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            return None
        try:
            x0, y0, x1, y1 = (float(value) for value in raw[:4])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            return None
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 1.0 or bottom - top < 1.0:
            return None
        return [left, top, right, bottom]

    @staticmethod
    def _state_signature(detection: dict) -> str:
        """Use stable identity plus coarse, material view evidence.

        The old label+ID signature cached an unknown door verdict for the
        whole success refresh window even after the robot approached it.  Do
        not include the raw box or frame index (that would flood the queue);
        these buckets change only for a substantially better/worse view.
        """

        box = (
            detection.get("bbox_2d")
            or detection.get("projected_bbox_2d")
            or detection.get("bbox")
        )
        area = max(1.0, bbox_area(box))
        try:
            distance_m = max(0.0, float(detection.get("distance_m", 0.0) or 0.0))
        except (TypeError, ValueError):
            distance_m = 0.0
        try:
            visible_fraction = max(
                0.0, min(1.0, float(detection.get("visible_fraction", 0.0) or 0.0))
            )
        except (TypeError, ValueError):
            visible_fraction = 0.0
        return json.dumps(
            {
                "label": str(
                    detection.get("semantic_name")
                    or detection.get("category")
                    or detection.get("name")
                    or ""
                ).casefold(),
                "object_id": str(
                    detection.get("instance_id")
                    or detection.get("id")
                    or detection.get("object_id")
                    or ""
                ),
                "view": {
                    "area_log2": int(math.log2(area)),
                    "distance_half_m": int(distance_m * 2.0),
                    "visible_quarter": int(visible_fraction * 4.0),
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _attribute_refresh_interval(self, detection: dict, patch: dict) -> float:
        """Return a short bounded retry interval for uncertain portal views."""

        semantic_text = " ".join(
            str(detection.get(key) or "").casefold()
            for key in ("semantic_name", "category", "name")
        )
        is_portal_candidate = any(
            token in semantic_text for token in ("door", "gate", "barrier", "portal")
        )
        if not is_portal_candidate:
            return self.success_refresh_interval_s
        try:
            confidence = float(patch.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        interaction_class = str(patch.get("interaction_class") or "unknown").casefold()
        coarse_state = str(patch.get("coarse_state") or "unknown").casefold()
        uncertain = (
            interaction_class not in {"portal"}
            or coarse_state in {"", "unknown", "none"}
            or confidence < self.uncertain_portal_confidence
        )
        return (
            self.uncertain_portal_refresh_interval_s
            if uncertain
            else self.success_refresh_interval_s
        )

    def _priority(self, detection: dict) -> float:
        distance_m = max(0.0, float(detection.get("distance_m", 0.0) or 0.0))
        visible_fraction = max(
            0.0, float(detection.get("visible_fraction", 0.0) or 0.0)
        )
        semantic_text = " ".join(
            str(detection.get(key) or "").casefold()
            for key in ("semantic_name", "category", "name")
        )
        score = 10.0 * float(any(token in semantic_text for token in ("door", "gate", "barrier")))
        score += 6.0 * float(
            any(
                token in semantic_text
                for token in ("fridge", "cabinet", "drawer", "dresser", "wardrobe", "closet")
            )
        )
        score += 8.0 * float(bool(detection.get("target_relevant")))
        score += 4.0 * visible_fraction
        score += max(0.0, self.max_distance_m - distance_m)
        return score

    def _try_reserve(self, object_id: str, signature: str) -> dict | None:
        if not object_id:
            return None
        now = time.monotonic()
        with self.lock:
            if object_id in self.pending:
                return None
            completed = self.completed.get(object_id)
            if completed is not None and completed.get("signature") == signature:
                refresh_interval_s = float(
                    completed.get(
                        "refresh_interval_s", self.success_refresh_interval_s
                    )
                    or 0.0
                )
                if (
                    refresh_interval_s <= 0.0
                    or now - float(completed.get("completed_at", 0.0))
                    < refresh_interval_s
                ):
                    return None
            if now - self.last_request.get(object_id, 0.0) < self.min_interval_s:
                return None
            self.request_sequence += 1
            self.pending[object_id] = {
                "request_sequence": self.request_sequence,
                "signature": signature,
                "generation": self.generations.get(object_id, 0),
                "episode_id": self.current_episode_id,
            }
            return {
                "generation": self.generations.get(object_id, 0),
                "request_sequence": self.request_sequence,
            }

    def _release(self, object_id: str, request_sequence: int | None = None) -> None:
        with self.lock:
            pending = self.pending.get(object_id)
            if request_sequence is None or int(
                (pending or {}).get("request_sequence", 0) or 0
            ) == int(request_sequence):
                self.pending.pop(object_id, None)

    @staticmethod
    def _remaining_request_timeout(
        deadline_monotonic: float | None,
        fallback_timeout_s: float,
    ) -> float:
        """Return the remaining end-to-end budget, including queue wait."""

        if deadline_monotonic is None:
            return max(0.01, float(fallback_timeout_s))
        try:
            return max(0.0, float(deadline_monotonic) - time.monotonic())
        except (TypeError, ValueError):
            return 0.0

    def _expire_attribute_request(self, request_payload: dict) -> None:
        """Drop a stale queued M1 request without contacting the model."""

        object_id = str(request_payload.get("object_id") or "")
        episode_id = str(request_payload.get("episode_id") or "")
        generation = int(request_payload.get("generation", 0) or 0)
        request_sequence = int(request_payload.get("request_sequence", 0) or 0)
        current = False
        now = time.monotonic()
        with self.lock:
            current = self._is_current_request_locked(
                object_id, episode_id, generation, request_sequence
            )
            if current:
                self.pending.pop(object_id, None)
            self.filter_counts["expired"] = self.filter_counts.get("expired", 0) + 1
        if current:
            failure_payload = {
                "object_id": object_id,
                "frame_id": request_payload.get("frame_id", ""),
                "image_sequence": request_payload.get("image_sequence", -1),
                "request_sequence": request_sequence,
                "signature": str(request_payload.get("signature") or ""),
                "targeted_refresh": request_payload.get("targeted_refresh") or {},
            }
            enqueued_at = float(request_payload.get("enqueued_at", now) or now)
            stamp = float(request_payload.get("stamp", time.time()) or time.time())
            self._publish_updates(
                episode_id,
                stamp,
                [
                    self._attribute_status_patch(
                        failure_payload,
                        "failed",
                        error="queue_deadline_expired_before_send",
                    )
                    | {
                        "queue_lag_sec": max(0.0, now - enqueued_at),
                        "response_lag_sec": 0.0,
                        "total_lag_sec": max(0.0, now - enqueued_at),
                    }
                ],
            )
        self._publish_status()

    def _expire_room_request(self, request_payload: dict) -> None:
        """Drop a stale queued room request without contacting the model."""

        room_key = str(
            request_payload.get("object_id")
            or request_payload.get("room_node_id")
            or ""
        )
        episode_id = str(request_payload.get("episode_id") or "")
        generation = int(request_payload.get("generation", 0) or 0)
        request_sequence = int(request_payload.get("request_sequence", 0) or 0)
        current = False
        now = time.monotonic()
        with self.lock:
            current = self._is_current_room_request_locked(
                room_key, episode_id, generation, request_sequence
            )
            if current:
                self.room_pending.pop(room_key, None)
            self.room_counts["expired"] = self.room_counts.get("expired", 0) + 1
        if current:
            request_payload = dict(request_payload)
            enqueued_at = float(request_payload.get("enqueued_at", now) or now)
            stamp = float(request_payload.get("stamp", time.time()) or time.time())
            self._publish_room_updates(
                episode_id,
                stamp,
                [
                    self._room_status_patch(
                        request_payload,
                        "failed",
                        error="queue_deadline_expired_before_send",
                    )
                    | {
                        "queue_lag_sec": max(0.0, now - enqueued_at),
                        "response_lag_sec": 0.0,
                        "total_lag_sec": max(0.0, now - enqueued_at),
                    }
                ],
            )
        self._publish_status()

    def _worker_loop(self) -> None:
        while not self.shutdown_event.is_set() and not rospy.is_shutdown():
            request_payload = self.request_queue.get(timeout_s=0.5)
            if request_payload is None:
                continue
            request_payload.pop("priority", None)
            if self._remaining_request_timeout(
                request_payload.get("deadline_monotonic"), self.request_timeout_s
            ) <= 0.0:
                self._expire_attribute_request(request_payload)
                continue
            self._infer(**request_payload)

    def _room_worker_loop(self) -> None:
        while not self.shutdown_event.is_set() and not rospy.is_shutdown():
            request_payload = self.room_request_queue.get(timeout_s=0.5)
            if request_payload is None:
                continue
            request_payload.pop("priority", None)
            if self._remaining_request_timeout(
                request_payload.get("deadline_monotonic"), self.room_request_timeout_s
            ) <= 0.0:
                self._expire_room_request(request_payload)
                continue
            room_key = str(request_payload.pop("object_id") or "")
            self._infer_room(room_key=room_key, **request_payload)

    def _shutdown(self) -> None:
        self.shutdown_event.set()
        self.request_queue.close()
        self.room_request_queue.close()
        for worker in self.workers:
            if worker is not threading.current_thread():
                worker.join(timeout=self.request_timeout_s + 0.5)
        for worker in self.room_workers:
            if worker is not threading.current_thread():
                worker.join(timeout=self.room_request_timeout_s + 0.5)

    def _publish_updates(self, episode_id: str, stamp: float, updates: list[dict]) -> None:
        self.publisher.publish(
            String(
                data=json.dumps(
                    {
                        "episode_id": episode_id,
                        "stamp_sec": stamp,
                        "updates": updates,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )

    def _publish_room_updates(
        self, episode_id: str, stamp: float, updates: list[dict]
    ) -> None:
        self.room_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "episode_id": episode_id,
                        "stamp_sec": stamp,
                        "updates": updates,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )

    def _infer(
        self,
        object_id: str,
        detection: dict,
        visual_evidence: np.ndarray,
        episode_id: str,
        frame_id: str,
        image_sequence: int,
        stamp: float,
        signature: str,
        generation: int,
        request_sequence: int,
        enqueued_at: float,
        targeted_refresh: dict,
        deadline_monotonic: float | None = None,
    ) -> None:
        request_started = time.monotonic()
        queue_lag_sec = max(0.0, request_started - float(enqueued_at))
        succeeded = False
        deadline_expired = False
        model_call_started = False
        outcome_status = "failed"
        outcome_error = ""
        refresh_interval_s = self.success_refresh_interval_s
        try:
            with self.lock:
                self.filter_counts["started"] += 1
            if not self._is_current_request(
                object_id, episode_id, generation, request_sequence
            ):
                outcome_status = "stale"
                return
            encoded = self._encode_jpeg(
                self._resize_visual_evidence(
                    visual_evidence, self.visual_evidence_max_side_px
                )
            )
            if not encoded:
                return
            image_data = "data:image/jpeg;base64," + __import__("base64").b64encode(encoded).decode("ascii")
            remaining_timeout_s = self._remaining_request_timeout(
                deadline_monotonic, self.request_timeout_s
            )
            if remaining_timeout_s <= 0.0:
                deadline_expired = True
                outcome_error = "queue_deadline_expired_before_send"
                return
            expected_node_type = str(
                targeted_refresh.get("expected_node_type") or ""
            ).strip().casefold()
            container_refresh = expected_node_type == "container"
            expected_type_instruction = (
                "This is a targeted re-observation whose public expected_node_type is "
                "container. It constrains class output only: interaction_class MUST be "
                "container, portal_morphology and portal_aperture_evidence MUST be null, "
                "and coarse_state MUST be open, closed, ajar, or unknown. Do not promote "
                "this target to a portal. Still determine state, frontality, and reobserve "
                "need from image pixels alone; do not assume a front view or hidden state. "
                if container_refresh
                else ""
            )
            class_instruction = (
                "interaction_class is container for this targeted refresh. "
                if container_refresh
                else "interaction_class is portal, container, none, or unknown. "
            )
            portal_instruction = (
                "For a non-portal, set portal_morphology and "
                "portal_aperture_evidence to null. "
                if container_refresh
                else (
                    "For a portal only, portal_morphology is {door_leaf: absent|present|unknown, "
                    "confidence: 0..1}; report absent only when the image visibly shows a clear "
                    "opening with no door leaf. It is visual morphology only: never infer "
                    "traversability, simulator joints, asset names, or hidden geometry. "
                    "For a portal only, portal_aperture_evidence is {open_aperture: "
                    "visible|not_visible|unknown, confidence: 0..1}; use visible only when "
                    "a real gap/open passage is directly visible in the image. Do not claim "
                    "open or ajar from the class label, handle, or a guessed hidden state. "
                    "For a non-portal, set portal_morphology and "
                    "portal_aperture_evidence to null. "
                )
            )
            instruction = "".join(
                (
                    expected_type_instruction,
                    "Infer the outlined target's pre-interaction visual attributes using "
                    "only pixels in the one supplied composite image. The composite shows "
                    "the complete head-camera image, a target outline, and a padded target "
                    "crop inset. Do not use object IDs, prior state, category names, map "
                    "geometry, simulator knowledge, or hidden properties. Return exactly one "
                    "compact, single-line JSON object with only object_id, interactable, "
                    "interaction_class, coarse_state, portal_morphology, "
                    "portal_aperture_evidence, view_state, view_state_confidence, "
                    "front_surface_visible, front_surface_confidence, approach_ready, "
                    "needs_reobserve, action_regions, interaction_parts, and confidence. ",
                    class_instruction,
                    portal_instruction,
                    "view_state is front, oblique, side_or_back, occluded, or unknown and "
                    "describes only the current camera view of the outlined target. Judge "
                    "frontality primarily from the target's horizontal perspective: use front "
                    "for a nearly head-on, broad usable face whose left/right extent and visible "
                    "drawer or door fronts/handles face the camera without a dominant receding "
                    "side plane or strong horizontal foreshortening. The head camera can be "
                    "higher than the target while looking approximately horizontally, so the "
                    "target may sit low in the image, its lower boundary may leave the frame, "
                    "and some top surface may be visible. Those vertical framing/height cues "
                    "alone are not evidence of an oblique view and must not override otherwise "
                    "frontal horizontal evidence; do not require the top or bottom boundary to "
                    "be fully visible. A dominant side plane, lateral left/right clipping, a "
                    "strongly foreshortened or narrow front, a single dark slab, or a guess from "
                    "object shape is not front. "
                    "front_surface_visible is true only when its usable front surface is "
                    "directly visible. For a container, set approach_ready true only for that "
                    "direct front view; an oblique, side_or_back, occluded, or unknown view "
                    "must set approach_ready false and needs_reobserve true. "
                    "action_regions is an ordered list of visible actionable centers, in "
                    "normalized coordinates of the padded target-crop inset (x=0 left, y=0 top). "
                    "For a visibly front-facing drawer-like container, include each visible "
                    "drawer front or handle from top to bottom. A vertically clipped lower "
                    "drawer does not invalidate frontality, but include only action regions "
                    "that are actually visible. For side, occluded, unknown, or non-drawer "
                    "targets, return an empty list. Never invent hidden drawers or use full-image "
                    "coordinates. "
                    "interaction_parts has at most one item with part_id, type, state, "
                    "handle_visible, and confidence. Use a short generic part_id such as "
                    "part_1; never copy simulator body or joint identifiers. Do not output "
                    "markdown, explanations, geometry, axes, ranges, trajectories, or extra keys.",
                )
            )
            model_call_started = True
            response = self.client.request_json(
                role="attribute_inference",
                instruction=instruction,
                context={
                    # Keep semantic identifiers and scene geometry outside the
                    # M1 prompt.  ``target`` is an opaque response-routing token
                    # which the caller replaces with the real object ID.
                    "object_id": "target",
                    **(
                        {"expected_node_type": "container"}
                        if container_refresh
                        else {}
                    ),
                },
                images=[image_data],
                response_schema=build_attribute_patch_response_schema(
                    "target", expected_node_type=expected_node_type or None
                ),
                timeout_s=remaining_timeout_s,
                max_tokens=self.max_output_tokens,
                metrics_context={
                    "episode_id": episode_id,
                    "object_id": object_id,
                    "observation_frame_index": self._frame_index(frame_id),
                    "observation_capture_step": self._frame_index(frame_id),
                    "observation_image_sequence": int(image_sequence),
                    "request_sequence": request_sequence,
                    "queue_lag_sec": queue_lag_sec,
                    "targeted_refresh": bool(targeted_refresh),
                },
            )
            if response.error or response.payload is None:
                outcome_error = str(response.error or "empty_model_response")
                return
            patch = validate_attribute_patch(response.payload)
            if container_refresh and patch.get("interaction_class") != "container":
                # A compatible endpoint should enforce the strict schema.  Keep
                # this response-side guard for older/command backends: preserve
                # its image-derived state/frontality, but never let a targeted
                # container refresh turn into a portal update.
                patch["m1_reported_interaction_class"] = str(
                    patch.get("interaction_class") or "unknown"
                )
                patch["interaction_class"] = "container"
                if str(patch.get("coarse_state") or "").casefold() == "static_open":
                    patch["coarse_state"] = "unknown"
                patch.pop("portal_morphology", None)
                patch.pop("portal_aperture_evidence", None)
            refresh_interval_s = self._attribute_refresh_interval(detection, patch)
            with self.lock:
                if episode_id and episode_id != self.current_episode_id:
                    outcome_status = "stale"
                    return
                if generation != self.generations.get(object_id, 0):
                    outcome_status = "stale"
                    return
            patch.update(
                {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "stamp_sec": stamp,
                    "observation_stamp_sec": stamp,
                    "observation_frame_index": self._frame_index(frame_id),
                    "observation_capture_step": self._frame_index(frame_id),
                    "observation_image_sequence": int(image_sequence),
                    "observation_signature": signature,
                    "source": "mllm_attribute_inference",
                    "model_name": self.client.config.model,
                    "evidence_frame_ids": [frame_id] if frame_id else [],
                    "request_sequence": request_sequence,
                    "attribute_status": "ready",
                    "queue_lag_sec": queue_lag_sec,
                    "response_lag_sec": max(0.0, time.monotonic() - request_started),
                    "total_lag_sec": max(0.0, time.monotonic() - float(enqueued_at)),
                }
            )
            # M1 does not receive detector geometry in its prompt.  A targeted
            # drawer scan nevertheless needs the exact public box paired with
            # this RGB frame, so attach that sensor-side evidence only after
            # model inference returns.
            observed_bbox = self._public_detection_bbox(detection)
            if observed_bbox is not None:
                patch["observed_bbox_2d"] = observed_bbox
            # Keep this detector fact separate from the M1 answer.  Portal
            # state still treats any clipped side as incomplete evidence.  For
            # a container contact view, the executor can distinguish a missing
            # *lateral* boundary (unsafe frontality) from a low camera view
            # that clips only the top/bottom while retaining a grounded handle
            # region.
            truncated_edges = self._detection_bbox_border_edges(
                visual_evidence, detection
            )
            patch["visual_evidence_truncated_edges"] = truncated_edges
            patch["visual_evidence_truncated"] = bool(truncated_edges)
            patch["is_currently_visible"] = True
            patch.update(
                self._attribute_status_patch(
                    {
                        "object_id": object_id,
                        "frame_id": frame_id,
                        "image_sequence": image_sequence,
                        "request_sequence": request_sequence,
                        "signature": signature,
                        "targeted_refresh": targeted_refresh,
                    },
                    "ready",
                )
            )
            self._publish_updates(episode_id, stamp, [patch])
            succeeded = True
        except Exception as exc:
            outcome_error = str(exc)
            rospy.logwarn_throttle(5.0, "attribute inference failed: %s", exc)
        finally:
            publish_status = False
            with self.lock:
                current_request = self._is_current_request_locked(
                    object_id, episode_id, generation, request_sequence
                )
                if current_request:
                    self.pending.pop(object_id, None)
                    if model_call_started:
                        self.last_request[object_id] = time.monotonic()
                if succeeded and current_request:
                    self.completed[object_id] = {
                        "signature": signature,
                        "completed_at": time.monotonic(),
                        "request_sequence": request_sequence,
                        "refresh_interval_s": refresh_interval_s,
                    }
                elif (
                    current_request
                ):
                    publish_status = True
            if publish_status:
                failure_payload = {
                    "object_id": object_id,
                    "frame_id": frame_id,
                    "image_sequence": image_sequence,
                    "request_sequence": request_sequence,
                    "signature": signature,
                    "targeted_refresh": targeted_refresh,
                }
                self._publish_updates(
                    episode_id,
                    stamp,
                    [
                        self._attribute_status_patch(
                            failure_payload,
                            outcome_status,
                            error=outcome_error,
                        )
                        | {
                            "queue_lag_sec": queue_lag_sec,
                            "response_lag_sec": max(0.0, time.monotonic() - request_started),
                            "total_lag_sec": max(
                                0.0, time.monotonic() - float(enqueued_at)
                            ),
                        }
                    ],
                )
            with self.lock:
                if outcome_status == "stale":
                    self.filter_counts["stale"] += 1
                elif succeeded:
                    self.filter_counts["completed"] += 1
                elif deadline_expired:
                    self.filter_counts["expired"] = (
                        self.filter_counts.get("expired", 0) + 1
                    )
                else:
                    self.filter_counts["failed"] += 1
            self._publish_status()

    def _infer_room(
        self,
        room_key: str,
        room_id: int,
        room_node_id: str,
        objects: list[dict],
        episode_id: str,
        capture_step: int | None,
        stamp: float,
        signature: str,
        generation: int,
        request_sequence: int,
        enqueued_at: float,
        deadline_monotonic: float | None = None,
    ) -> None:
        """Run text-only room inference in a queue independent of RGB crops."""

        request_started = time.monotonic()
        queue_lag_sec = max(0.0, request_started - float(enqueued_at))
        succeeded = False
        deadline_expired = False
        model_call_started = False
        outcome_status = "failed"
        outcome_error = ""
        request_payload = {
            "room_id": int(room_id),
            "room_node_id": str(room_node_id),
            "capture_step": capture_step,
            "stamp": float(stamp),
            "signature": signature,
            "request_sequence": int(request_sequence),
            "episode_id": episode_id,
        }
        try:
            with self.lock:
                self.room_counts["started"] += 1
            if not self._is_current_room_request(
                room_key, episode_id, generation, request_sequence
            ):
                outcome_status = "stale"
                return
            remaining_timeout_s = self._remaining_request_timeout(
                deadline_monotonic, self.room_request_timeout_s
            )
            if remaining_timeout_s <= 0.0:
                deadline_expired = True
                outcome_error = "queue_deadline_expired_before_send"
                return
            model_call_started = True
            response = self.client.request_json(
                role="room_attribute_inference",
                instruction=(
                    "Infer the likely room attribute using only the supplied room ID and "
                    "its currently known in-room object labels. Return exactly one compact, "
                    "single-line JSON object with only room_id, room_attribute, confidence, "
                    "and evidence_object_ids. Include at most two strongest evidence_object_ids. "
                    "Do not assume an image was provided. Do not "
                    "output markdown, explanations, geometry, poses, simulator identifiers, "
                    "or any extra keys. If the object evidence is insufficient, return unknown "
                    "with low confidence."
                ),
                context={
                    "room_id": int(room_id),
                    "capture_step": capture_step,
                    "objects": objects,
                    "episode_id": episode_id,
                },
                timeout_s=remaining_timeout_s,
                max_tokens=self.room_max_output_tokens,
                metrics_context={
                    "episode_id": episode_id,
                    "room_id": int(room_id),
                    "capture_step": capture_step,
                    "request_sequence": request_sequence,
                    "queue_lag_sec": queue_lag_sec,
                    "inference_lane": "room",
                },
            )
            if response.error or response.payload is None:
                outcome_error = str(response.error or "empty_model_response")
                return
            patch = validate_room_attribute_patch(response.payload)
            if int(patch["room_id"]) != int(room_id):
                outcome_error = "room_id_mismatch"
                return
            with self.lock:
                if episode_id and episode_id != self.current_episode_id:
                    outcome_status = "stale"
                    return
                if generation != self.room_generations.get(room_key, 0):
                    outcome_status = "stale"
                    return
            patch.update(
                {
                    "room_id": int(room_id),
                    "room_node_id": str(room_node_id),
                    "episode_id": episode_id,
                    "stamp_sec": float(stamp),
                    "observation_stamp_sec": float(stamp),
                    "observation_capture_step": capture_step,
                    "observation_signature": signature,
                    "source": "mllm_room_attribute_inference",
                    "model_name": self.client.config.model,
                    "request_sequence": int(request_sequence),
                    "room_attribute_status": "ready",
                    "queue_lag_sec": queue_lag_sec,
                    "response_lag_sec": max(0.0, time.monotonic() - request_started),
                    "total_lag_sec": max(0.0, time.monotonic() - float(enqueued_at)),
                }
            )
            self._publish_room_updates(episode_id, float(stamp), [patch])
            succeeded = True
        except Exception as exc:
            outcome_error = str(exc)
            rospy.logwarn_throttle(5.0, "room attribute inference failed: %s", exc)
        finally:
            publish_status = False
            with self.lock:
                current_request = self._is_current_room_request_locked(
                    room_key, episode_id, generation, request_sequence
                )
                if current_request:
                    self.room_pending.pop(room_key, None)
                    if model_call_started:
                        self.room_last_request[room_key] = time.monotonic()
                if succeeded and current_request:
                    self.room_completed[room_key] = {
                        "signature": signature,
                        "completed_at": time.monotonic(),
                        "request_sequence": request_sequence,
                    }
                elif current_request:
                    publish_status = True
            if publish_status:
                self._publish_room_updates(
                    episode_id,
                    float(stamp),
                    [
                        self._room_status_patch(
                            request_payload,
                            outcome_status,
                            error=outcome_error,
                        )
                        | {
                            "queue_lag_sec": queue_lag_sec,
                            "response_lag_sec": max(
                                0.0, time.monotonic() - request_started
                            ),
                            "total_lag_sec": max(
                                0.0, time.monotonic() - float(enqueued_at)
                            ),
                        }
                    ],
                )
            with self.lock:
                if outcome_status == "stale":
                    self.room_counts["stale"] += 1
                elif succeeded:
                    self.room_counts["completed"] += 1
                elif deadline_expired:
                    self.room_counts["expired"] = (
                        self.room_counts.get("expired", 0) + 1
                    )
                else:
                    self.room_counts["failed"] += 1
            self._publish_status()

    def _is_current_room_request(
        self, room_key: str, episode_id: str, generation: int, request_sequence: int
    ) -> bool:
        with self.lock:
            return self._is_current_room_request_locked(
                room_key, episode_id, generation, request_sequence
            )

    def _is_current_room_request_locked(
        self, room_key: str, episode_id: str, generation: int, request_sequence: int
    ) -> bool:
        pending = self.room_pending.get(room_key) or {}
        pending_generation = pending.get("generation", -1)
        return (
            (not episode_id or episode_id == self.current_episode_id)
            and int(pending.get("request_sequence", 0) or 0) == int(request_sequence)
            and int(-1 if pending_generation is None else pending_generation)
            == int(generation)
            and int(self.room_generations.get(room_key, 0)) == int(generation)
        )

    @staticmethod
    def _frame_index(frame_id: str) -> int | None:
        try:
            return int(frame_id)
        except (TypeError, ValueError):
            return None

    def _is_current_request(
        self, object_id: str, episode_id: str, generation: int, request_sequence: int
    ) -> bool:
        with self.lock:
            return self._is_current_request_locked(
                object_id, episode_id, generation, request_sequence
            )

    def _is_current_request_locked(
        self, object_id: str, episode_id: str, generation: int, request_sequence: int
    ) -> bool:
        pending = self.pending.get(object_id) or {}
        pending_generation = pending.get("generation", -1)
        return (
            (not episode_id or episode_id == self.current_episode_id)
            and int(pending.get("request_sequence", 0) or 0) == int(request_sequence)
            and int(
                -1 if pending_generation is None else pending_generation
            ) == int(generation)
            and int(self.generations.get(object_id, 0)) == int(generation)
        )

    @staticmethod
    def _bbox_pixels(image: np.ndarray, detection: dict) -> tuple[int, int, int, int] | None:
        box = (
            detection.get("bbox_2d")
            or detection.get("projected_bbox_2d")
            or detection.get("bbox")
        )
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        height, width = image.shape[:2]
        x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @classmethod
    def _detection_bbox_touches_image_border(
        cls,
        image: np.ndarray,
        detection: dict,
        *,
        margin_px: int = 2,
    ) -> bool:
        """Return whether the public detection is clipped by this RGB frame."""

        return bool(cls._detection_bbox_border_edges(image, detection, margin_px=margin_px))

    @classmethod
    def _detection_bbox_border_edges(
        cls,
        image: np.ndarray,
        detection: dict,
        *,
        margin_px: int = 2,
    ) -> list[str]:
        """Return public image sides touched by a detector box.

        This remains a sensor-side fact rather than an M1 judgement.  ``unknown``
        deliberately fails closed for callers that cannot establish a valid
        detector rectangle.
        """

        bbox = cls._bbox_pixels(image, detection)
        if bbox is None:
            return ["unknown"]
        height, width = image.shape[:2]
        x0, y0, x1, y1 = bbox
        margin = max(0, int(margin_px))
        edges: list[str] = []
        if x0 <= margin:
            edges.append("left")
        if x1 >= max(0, width - margin):
            edges.append("right")
        if y0 <= margin:
            edges.append("top")
        if y1 >= max(0, height - margin):
            edges.append("bottom")
        return edges

    @classmethod
    def _crop(cls, image, detection, margin_ratio=0.0):
        bbox = cls._bbox_pixels(image, detection)
        if bbox is None:
            return None
        height, width = image.shape[:2]
        x0, y0, x1, y1 = bbox
        margin_x = int(round(abs(x1 - x0) * max(0.0, float(margin_ratio))))
        margin_y = int(round(abs(y1 - y0) * max(0.0, float(margin_ratio))))
        x0 -= margin_x
        x1 += margin_x
        y0 -= margin_y
        y1 += margin_y
        x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return image[y0:y1, x0:x1]

    @staticmethod
    def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize without making M1 depend on an OpenCV-enabled ROS build."""

        source_height, source_width = image.shape[:2]
        if source_height <= 0 or source_width <= 0:
            return image
        width = max(1, int(width))
        height = max(1, int(height))
        if width == source_width and height == source_height:
            return image.copy()
        if cv2 is not None:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        source_y = np.linspace(0, source_height - 1, height).astype(np.intp)
        source_x = np.linspace(0, source_width - 1, width).astype(np.intp)
        return image[source_y][:, source_x].copy()

    @classmethod
    def _resize_visual_evidence(
        cls, image: np.ndarray, max_side_px: int
    ) -> np.ndarray:
        max_side_px = int(max_side_px)
        if max_side_px <= 0:
            return image
        height, width = image.shape[:2]
        largest_side = max(height, width)
        if largest_side <= max_side_px:
            return image
        scale = float(max_side_px) / float(largest_side)
        return cls._resize_nearest(
            image,
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )

    @staticmethod
    def _draw_rectangle(
        image: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        height, width = image.shape[:2]
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        if right <= left or bottom <= top:
            return
        thickness = max(1, min(int(thickness), max(1, (right - left) // 2), max(1, (bottom - top) // 2)))
        image[top : min(bottom, top + thickness), left:right] = color
        image[max(top, bottom - thickness) : bottom, left:right] = color
        image[top:bottom, left : min(right, left + thickness)] = color
        image[top:bottom, max(left, right - thickness) : right] = color

    @staticmethod
    def _rect_overlap_area(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> int:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        return max(0, right - left) * max(0, bottom - top)

    @classmethod
    def _compose_attribute_visual_evidence(
        cls, image: np.ndarray, detection: dict, *, margin_ratio: float
    ) -> np.ndarray | None:
        """Build M1's one-image visual observation without textual/GT context.

        The target outline anchors which item the model must judge, while a
        padded crop inset preserves detail that would be too small in the full
        head-camera view.  The source image itself remains the full camera
        observation, so the model can use surrounding visual context only.
        """

        if not isinstance(image, np.ndarray) or image.ndim != 3:
            return None
        bbox = cls._bbox_pixels(image, detection)
        crop = cls._crop(image, detection, margin_ratio=margin_ratio)
        if bbox is None or crop is None or crop.size == 0:
            return None
        composite = image.copy()
        height, width = composite.shape[:2]
        target_left, target_top, target_right, target_bottom = bbox
        outline_color = (0, 255, 255)
        cls._draw_rectangle(
            composite,
            target_left,
            target_top,
            target_right,
            target_bottom,
            outline_color,
            thickness=max(2, min(width, height) // 240),
        )

        max_panel_width = max(1, min(width, max(32, int(round(width * 0.38)))))
        max_panel_height = max(1, min(height, max(32, int(round(height * 0.46)))))
        crop_height, crop_width = crop.shape[:2]
        scale = min(
            float(max_panel_width) / float(max(1, crop_width)),
            float(max_panel_height) / float(max(1, crop_height)),
        )
        panel_width = max(1, min(max_panel_width, int(round(crop_width * scale))))
        panel_height = max(1, min(max_panel_height, int(round(crop_height * scale))))
        resized_crop = cls._resize_nearest(crop, panel_width, panel_height)
        border = 3
        right_origin = max(0, width - panel_width)
        bottom_origin = max(0, height - panel_height)
        candidates = [
            (0, 0),
            (right_origin, 0),
            (0, bottom_origin),
            (right_origin, bottom_origin),
        ]
        panel_left, panel_top = min(
            candidates,
            key=lambda item: cls._rect_overlap_area(
                (item[0], item[1], item[0] + panel_width, item[1] + panel_height),
                bbox,
            ),
        )
        panel_left = max(0, panel_left)
        panel_top = max(0, panel_top)
        panel_right = min(width, panel_left + panel_width)
        panel_bottom = min(height, panel_top + panel_height)
        composite[panel_top:panel_bottom, panel_left:panel_right] = resized_crop[
            : panel_bottom - panel_top, : panel_right - panel_left
        ]
        cls._draw_rectangle(
            composite,
            panel_left - border,
            panel_top - border,
            panel_right + border,
            panel_bottom + border,
            outline_color,
            thickness=border,
        )
        return composite


if __name__ == "__main__":
    InteractionAttributeInferenceNode()
    rospy.spin()
