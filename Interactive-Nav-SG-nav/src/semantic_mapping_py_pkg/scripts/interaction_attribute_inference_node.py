#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from semantic_mllm_py_pkg import load_env_file
from semantic_mllm_py_pkg.client import MLLMClient
from semantic_mllm_py_pkg.env import client_config_from_env
from semantic_mllm_py_pkg.schemas import validate_attribute_patch
from semantic_mapping_py_pkg.attribute_filter import (
    DEFAULT_EXCLUDE_KEYWORDS,
    DEFAULT_INTERACTION_KEYWORDS,
    is_interaction_attribute_candidate,
)
from semantic_mapping_py_pkg.attribute_inference_queue import LatestPriorityRequestQueue
from semantic_mapping_py_pkg.graph_rules import bbox_area, segmentation_pixel_count
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311


class InteractionAttributeInferenceNode:
    def __init__(self) -> None:
        env_file = os.environ.get("SEMANTIC_DECISION_ENV_FILE")
        load_env_file(env_file, override=bool(env_file))
        patch_roslogging_findcaller_for_py311()
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
        self.model_name = str(rospy.get_param("~model_name", "") or "")
        self.client = MLLMClient(client_config_from_env(model=self.model_name or None))
        self.request_timeout_s = max(
            0.1, float(rospy.get_param("~request_timeout_s", 8.0))
        )
        self.max_output_tokens = max(
            32,
            int(rospy.get_param("~max_output_tokens", min(self.client.config.max_tokens, 160))),
        )
        self.crop_margin_ratio = max(
            0.0, float(rospy.get_param("~crop_margin_ratio", 0.08))
        )
        self.publisher = rospy.Publisher(self.output_topic, String, queue_size=2)
        self.status_publisher = rospy.Publisher(
            self.status_topic, String, queue_size=1, latch=True
        )
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_stamp = 0.0
        self.pending_detection_payload: dict | None = None
        self.filter_counts = {
            "messages_received": 0,
            "received": 0,
            "eligible": 0,
            "enqueued": 0,
            "started": 0,
            "completed": 0,
            "stale": 0,
            "failed": 0,
            "filtered": 0,
            "missing_image": 0,
        }
        self.current_episode_id = ""
        self.request_sequence = 0
        self.last_request: dict[str, float] = {}
        self.pending: dict[str, dict] = {}
        self.completed: dict[str, dict] = {}
        self.generations: dict[str, int] = {}
        self.aliases: dict[str, str] = {}
        self.request_queue = LatestPriorityRequestQueue(self.max_queue_size)
        self.shutdown_event = threading.Event()
        self.workers = [
            threading.Thread(target=self._worker_loop, daemon=True)
            for _index in range(self.worker_count)
        ]
        for worker in self.workers:
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
        rospy.loginfo(
            "[interaction_attribute_inference] image=%s detections=%s output=%s model=%s workers=%d queue=%d",
            self.image_topic,
            self.detection_topic,
            self.output_topic,
            self.client.config.model,
            self.worker_count,
            self.max_queue_size,
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
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

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
        frame_id = str(payload.get("frame_index") or "") if isinstance(payload, dict) else ""
        observation_stamp = self._observation_stamp(payload, image_stamp)
        if episode_id:
            self._set_episode(episode_id)
        requests = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            with self.lock:
                self.filter_counts["received"] += 1
            if not self._passes_observation_filter(detection):
                with self.lock:
                    self.filter_counts["filtered"] += 1
                continue
            with self.lock:
                self.filter_counts["eligible"] += 1
            object_id = str(
                detection.get("instance_id")
                or detection.get("id")
                or detection.get("object_id")
                or detection.get("source_object_name")
                or detection.get("name")
                or ""
            )
            self._register_aliases(object_id, detection)
            signature = self._state_signature(detection)
            self._invalidate_if_state_changed(object_id, signature, episode_id)
            reservation = self._try_reserve(object_id, signature)
            if not object_id or reservation is None:
                continue
            crop = self._crop(image, detection, self.crop_margin_ratio)
            if crop is None:
                self._release(object_id, reservation["request_sequence"])
                continue
            requests.append(
                {
                    "priority": self._priority(detection),
                    "object_id": object_id,
                    "detection": dict(detection),
                    "crop": crop,
                    "episode_id": episode_id,
                    "frame_id": frame_id,
                    "stamp": observation_stamp,
                    "signature": signature,
                    "generation": reservation["generation"],
                    "request_sequence": reservation["request_sequence"],
                    "enqueued_at": time.monotonic(),
                }
            )
        for request_payload in sorted(
            requests, key=lambda item: (-float(item["priority"]), item["object_id"])
        ):
            accepted, displaced = self.request_queue.put(request_payload)
            if accepted:
                with self.lock:
                    self.filter_counts["enqueued"] += 1
                if displaced is not None:
                    self._release(
                        str(displaced.get("object_id") or ""),
                        int(displaced.get("request_sequence", 0) or 0),
                    )
                self._publish_updates(
                    str(request_payload["episode_id"]),
                    float(request_payload["stamp"]),
                    [
                        {
                            "object_id": request_payload["object_id"],
                            "attribute_status": "pending",
                            "request_sequence": request_payload["request_sequence"],
                            "source": "mllm_attribute_inference",
                        }
                    ],
                )
            else:
                self._release(
                    str(request_payload["object_id"]),
                    int(request_payload["request_sequence"]),
                )
        self._publish_status()
        rospy.loginfo_throttle(
            10.0,
            "[interaction_attribute_inference] received=%d eligible=%d enqueued=%d filtered=%d missing_image=%d queue=%d",
            self.filter_counts["received"],
            self.filter_counts["eligible"],
            self.filter_counts["enqueued"],
            self.filter_counts["filtered"],
            self.filter_counts["missing_image"],
            len(self.request_queue),
        )

    def _publish_status(self) -> None:
        with self.lock:
            payload = {
                "episode_id": self.current_episode_id,
                "filter_counts": dict(self.filter_counts),
                "queue_size": len(self.request_queue),
                "pending_requests": len(self.pending),
                "has_latest_image": self.latest_image is not None,
                "has_pending_detection": self.pending_detection_payload is not None,
            }
        self.status_publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _set_episode(self, episode_id: str) -> None:
        stale_request_ids = []
        with self.lock:
            if not episode_id or episode_id == self.current_episode_id:
                return
            stale_request_ids = [
                (object_id, int(payload.get("request_sequence", 0) or 0))
                for object_id, payload in self.pending.items()
            ]
            self.current_episode_id = episode_id
            self.last_request.clear()
            self.pending.clear()
            self.completed.clear()
            self.generations.clear()
            self.aliases.clear()
        for object_id, request_sequence in stale_request_ids:
            self.request_queue.discard(object_id, request_sequence)

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
    def _observation_stamp(payload: object, fallback: float) -> float:
        if isinstance(payload, dict):
            try:
                return float(payload.get("stamp_sec", fallback) or fallback)
            except (TypeError, ValueError):
                pass
        return float(fallback)

    def _invalidate_if_state_changed(
        self, object_id: str, signature: str, episode_id: str
    ) -> None:
        if not object_id:
            return
        request_sequence = 0
        with self.lock:
            pending = self.pending.get(object_id)
            if pending is None or str(pending.get("signature") or "") == signature:
                return
            if episode_id and str(pending.get("episode_id") or "") not in {"", episode_id}:
                return
            request_sequence = int(pending.get("request_sequence", 0) or 0)
            self.generations[object_id] = self.generations.get(object_id, 0) + 1
            self.pending.pop(object_id, None)
            self.last_request.pop(object_id, None)
        self.request_queue.discard(object_id, request_sequence)

    def _passes_observation_filter(self, detection: dict) -> bool:
        if not is_interaction_attribute_candidate(
            detection, self.include_keywords, self.exclude_keywords
        ):
            return False
        box = detection.get("bbox_2d") or detection.get("projected_bbox_2d") or detection.get("bbox")
        visible_pixels = int(
            detection.get(
                "visible_pixels",
                segmentation_pixel_count(
                    detection.get("segmentation")
                    if detection.get("segmentation") is not None
                    else detection.get("mask")
                ),
            )
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
    def _state_signature(detection: dict) -> str:
        return "|".join(
            (
                str(
                    detection.get("semantic_name")
                    or detection.get("category")
                    or detection.get("name")
                    or ""
                ).casefold(),
                str(
                    detection.get("instance_id")
                    or detection.get("id")
                    or detection.get("object_id")
                    or ""
                ),
            )
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
                if (
                    self.success_refresh_interval_s <= 0.0
                    or now - float(completed.get("completed_at", 0.0))
                    < self.success_refresh_interval_s
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

    def _worker_loop(self) -> None:
        while not self.shutdown_event.is_set() and not rospy.is_shutdown():
            request_payload = self.request_queue.get(timeout_s=0.5)
            if request_payload is None:
                continue
            request_payload.pop("priority", None)
            self._infer(**request_payload)

    def _shutdown(self) -> None:
        self.shutdown_event.set()
        self.request_queue.close()
        for worker in self.workers:
            if worker is not threading.current_thread():
                worker.join(timeout=self.request_timeout_s + 0.5)

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

    def _infer(
        self,
        object_id: str,
        detection: dict,
        crop,
        episode_id: str,
        frame_id: str,
        stamp: float,
        signature: str,
        generation: int,
        request_sequence: int,
        enqueued_at: float,
    ) -> None:
        request_started = time.monotonic()
        queue_lag_sec = max(0.0, request_started - float(enqueued_at))
        succeeded = False
        outcome_status = "failed"
        outcome_error = ""
        try:
            with self.lock:
                self.filter_counts["started"] += 1
            if not self._is_current_request(
                object_id, episode_id, generation, request_sequence
            ):
                outcome_status = "stale"
                return
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return
            image_data = "data:image/jpeg;base64," + __import__("base64").b64encode(encoded).decode("ascii")
            response = self.client.request_json(
                role="attribute_inference",
                instruction=(
                    "Infer only visible interaction attributes. Return exactly one compact, "
                    "single-line JSON object with only object_id, interactable, "
                    "interaction_class, coarse_state, interaction_parts, and confidence. "
                    "interaction_class is portal, container, none, or unknown. "
                    "interaction_parts has at most one item with part_id, type, state, "
                    "handle_visible, and confidence. Use a short generic part_id such as "
                    "part_1; never copy simulator body or joint identifiers. Do not output "
                    "markdown, explanations, geometry, axes, ranges, trajectories, or extra keys."
                ),
                context={
                    "object_id": object_id,
                    "name": detection.get("name") or detection.get("semantic_name"),
                    "category": detection.get("category"),
                    "geometry": {
                        "position": detection.get("position")
                        or (detection.get("box_3d") or {}).get("center"),
                        "aabb_size": detection.get("aabb_size")
                        or (detection.get("box_3d") or {}).get("size"),
                    },
                    "episode_id": episode_id,
                    "frame_id": frame_id,
                },
                images=[image_data],
                timeout_s=self.request_timeout_s,
                max_tokens=self.max_output_tokens,
                metrics_context={
                    "episode_id": episode_id,
                    "object_id": object_id,
                    "observation_frame_index": self._frame_index(frame_id),
                    "request_sequence": request_sequence,
                    "queue_lag_sec": queue_lag_sec,
                },
            )
            if response.error or response.payload is None:
                outcome_error = str(response.error or "empty_model_response")
                return
            patch = validate_attribute_patch(response.payload)
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
                    self.last_request[object_id] = time.monotonic()
                if succeeded and current_request:
                    self.completed[object_id] = {
                        "signature": signature,
                        "completed_at": time.monotonic(),
                        "request_sequence": request_sequence,
                    }
                elif (
                    current_request
                ):
                    publish_status = True
            if publish_status:
                self._publish_updates(
                    episode_id,
                    stamp,
                    [
                        {
                            "object_id": object_id,
                            "attribute_status": outcome_status,
                            "request_sequence": request_sequence,
                            "source": "mllm_attribute_inference",
                            "error": outcome_error[:240],
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
                else:
                    self.filter_counts["failed"] += 1
            self._publish_status()

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
    def _crop(image, detection, margin_ratio=0.0):
        box = (
            detection.get("bbox_2d")
            or detection.get("projected_bbox_2d")
            or detection.get("bbox")
        )
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        height, width = image.shape[:2]
        x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
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


if __name__ == "__main__":
    InteractionAttributeInferenceNode()
    rospy.spin()
