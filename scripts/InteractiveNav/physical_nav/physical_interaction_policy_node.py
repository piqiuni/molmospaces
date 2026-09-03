#!/usr/bin/env python3
"""Physical human-assist interaction policy with visual M3 verification."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import String

from interaction_policy import InteractionRequest, build_interaction_policy
from qwen_client import QwenClient


class PhysicalInteractionPolicyNode:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest_image = ""
        self._active_command = ""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="physical-policy")
        self._cancel = threading.Event()
        self._speech_condition = threading.Condition()
        self._speech_sequence = 0
        self._speech_status: dict[str, dict[str, Any]] = {}
        interaction_timeout_s = float(self._param("interaction_timeout_s", 20.0))
        self._qwen = QwenClient(
            base_url=str(rospy.get_param("~qwen_base_url", "http://127.0.0.1:18080/v1")),
            model=str(rospy.get_param("~qwen_model", "qwen3.6-35b-a3b-fp8")),
            timeout_s=min(
                interaction_timeout_s,
                float(rospy.get_param("~qwen_timeout_s", 20.0)),
            ),
        )
        self._speech_pub = rospy.Publisher(
            str(self._param("speech_topic", "/physical_nav/speech_request")), String, queue_size=4
        )
        self._result_pub = rospy.Publisher(
            str(self._param("result_topic", "/physical_nav/interaction_result")), String, queue_size=4, latch=True
        )
        self._event_pub = rospy.Publisher(
            str(self._param("mllm_events_topic", "/physical_nav/mllm_events")), String, queue_size=8
        )
        rospy.Subscriber(str(self._param("image_topic", "/physical_nav/rgb/image_raw")), Image, self._image_callback, queue_size=1)
        rospy.Subscriber(str(self._param("command_topic", "/physical_nav/interaction_command")), String, self._command_callback, queue_size=4)
        rospy.Subscriber(str(self._param("cancel_topic", "/physical_nav/interaction_policy/cancel")), String, self._cancel_callback, queue_size=2)
        rospy.Subscriber(
            str(self._param("speech_status_topic", "/physical_nav/speech_status")),
            String,
            self._speech_status_callback,
            queue_size=4,
        )

    @staticmethod
    def _param(name: str, default: Any) -> Any:
        direct = f"~{name}"
        nested = f"~interaction_policy/{name}"
        if rospy.has_param(direct):
            return rospy.get_param(direct)
        return rospy.get_param(nested, default)

    def _image_callback(self, message: Image) -> None:
        try:
            encoding = str(message.encoding or "rgb8").casefold()
            mode = "L" if encoding in {"mono8", "8uc1"} else "RGB"
            raw_mode = "L" if mode == "L" else ("BGR" if encoding == "bgr8" else "RGB")
            image = PILImage.frombytes(
                mode,
                (int(message.width), int(message.height)),
                bytes(message.data),
                "raw",
                raw_mode,
                int(message.step),
                1,
            )
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=85)
            with self._lock:
                self._latest_image = "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical policy image conversion failed: %s", exc)

    def _image_provider(self) -> str | None:
        with self._lock:
            return self._latest_image or None

    def _emit(self, event: dict[str, Any]) -> None:
        self._event_pub.publish(String(data=json.dumps(event, ensure_ascii=False, separators=(",", ":"))))

    def _speak(self, text: str, wait: bool) -> dict[str, Any]:
        with self._speech_condition:
            self._speech_sequence += 1
            request_id = f"speech-{int(time.time() * 1000)}-{self._speech_sequence}"
        payload = {
            "request_id": request_id,
            "text": text,
            # Human interaction time starts only after the request has actually
            # finished playing on Go2, not when it merely enters the queue.
            "wait": bool(wait),
            "volume": int(self._param("speech_volume", 4)),
        }
        deadline = time.monotonic() + max(
            0.0, float(self._param("speech_subscriber_wait_s", 2.0))
        )
        while self._speech_pub.get_num_connections() <= 0 and time.monotonic() < deadline:
            if rospy.is_shutdown():
                break
            time.sleep(0.05)
        if self._speech_pub.get_num_connections() <= 0:
            return {
                "accepted": False,
                "topic": self._speech_pub.name,
                "reason": "no_speech_subscriber",
            }
        self._speech_pub.publish(String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
        ack_deadline = time.monotonic() + max(0.1, float(self._param("speech_ack_timeout_s", 4.0)))
        with self._speech_condition:
            while request_id not in self._speech_status and time.monotonic() < ack_deadline:
                self._speech_condition.wait(timeout=min(0.1, ack_deadline - time.monotonic()))
            status = self._speech_status.pop(request_id, None)
        if not status:
            return {"accepted": False, "topic": self._speech_pub.name, "reason": "speech_bridge_ack_timeout"}
        return {**status, "topic": self._speech_pub.name}

    def _speech_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
            request_id = str(status.get("request_id") or "")
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not request_id:
            return
        with self._speech_condition:
            self._speech_status[request_id] = dict(status)
            self._speech_condition.notify_all()

    def _qwen_request(self, prompt: str, *, image_data_url: str, max_tokens: int) -> dict[str, Any]:
        return self._qwen.chat(
            prompt,
            image_data_url=image_data_url,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    def _cancel_callback(self, _message: String) -> None:
        self._cancel.set()

    def _command_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            request = InteractionRequest.from_payload(payload)
        except Exception as exc:
            rospy.logwarn("physical interaction command rejected: %s", exc)
            return
        with self._lock:
            if self._active_command:
                self._cancel.set()
                self._emit({"module": "POLICY", "stage": "REJECTED_BUSY", "command_id": request.command_id})
                return
            self._active_command = request.command_id
            self._cancel.clear()
        self._executor.submit(self._run, request)

    def _run(self, request: InteractionRequest) -> None:
        try:
            profile = str(self._param("profile", "physical_human"))
            policy = build_interaction_policy(
                profile,
                speak=self._speak,
                image_provider=self._image_provider,
                request_json=self._qwen_request,
                emit=self._emit,
                options={
                    "speech_retry_count": int(self._param("speech_retry_count", 2)),
                    "speech_retry_interval_s": float(self._param("speech_retry_interval_s", 12.0)),
                    "stable_open_s": float(self._param("stable_open_s", 3.0)),
                    "m3_sample_period_s": float(self._param("m3_sample_period_s", 1.0)),
                    "interaction_timeout_s": float(self._param("interaction_timeout_s", 20.0)),
                    "m3_min_confidence": float(self._param("m3_min_confidence", 0.60)),
                    "temporary_skip_s": float(self._param("temporary_skip_s", 30.0)),
                },
            )
            result = policy.execute(request, self._cancel.is_set)
            temporary_skip_s = float(result.verification.get("temporary_skip_s", 0.0) or 0.0)
            self._result_pub.publish(String(data=json.dumps({
                "command_id": result.command_id,
                "candidate_id": request.candidate_id,
                "object_id": request.target_id,
                "source_object_name": request.target_name,
                "target_kind": request.target_kind,
                "success": result.success,
                "status": result.status,
                "pre_state": result.pre_state,
                "post_state": result.post_state,
                "verification_source": "physical_visual_mllm",
                "verification": result.verification,
                "detail": result.detail,
                "temporary_skip_s": temporary_skip_s,
                "stamp_sec": result.timestamp,
            }, ensure_ascii=False, separators=(",", ":"))))
        except Exception as exc:
            self._result_pub.publish(String(data=json.dumps({"command_id": request.command_id, "success": False, "status": "FAILED", "failure_reason": str(exc)[:200], "verification_source": "physical_visual_mllm"}, ensure_ascii=False, separators=(",", ":"))))
        finally:
            with self._lock:
                self._active_command = ""


if __name__ == "__main__":
    print("physical interaction policy: initializing ROS", flush=True)
    rospy.init_node("physical_interaction_policy")
    print("physical interaction policy: constructing node", flush=True)
    PhysicalInteractionPolicyNode()
    print("physical interaction policy: ready", flush=True)
    rospy.spin()
