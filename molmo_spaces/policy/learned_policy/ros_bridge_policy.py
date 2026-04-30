import json
import threading
import time
from typing import Any
import time

import numpy as np

from molmo_spaces.policy.base_policy import BasePolicy
from molmo_spaces.tasks.task import BaseMujocoTask


class RosBridgePolicy(BasePolicy):
    """
    ROS bridge policy for closed-loop external control.

    - Publishes observations to a ROS topic.
    - Receives actions from another ROS node.
    """

    def __init__(
        self,
        config,
        task: BaseMujocoTask | None = None,
        observation_topic: str = "/molmo_spaces/observation",
        action_topic: str = "/molmo_spaces/action",
        action_timeout_s: float = 1.0,
        queue_size: int = 1,
    ) -> None:
        super().__init__(config, task)
        self.observation_topic = observation_topic
        self.action_topic = action_topic
        self.action_timeout_s = action_timeout_s
        self.queue_size = queue_size

        self._step_idx = 0
        self._latest_action: dict[str, Any] | None = None
        self._latest_action_step: int = -1
        self._last_consumed_action_step: int = -1
        self._lock = threading.Lock()

        self._patch_rosgraph_logger_for_py311()
        import rospy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self._rospy = rospy
        self._Image = Image
        self._String = String
        if not rospy.core.is_initialized():
            rospy.init_node("molmo_spaces_ros_policy", anonymous=True, disable_signals=True)

        self._obs_pub = rospy.Publisher(self.observation_topic, Image, queue_size=self.queue_size)
        self._action_sub = rospy.Subscriber(self.action_topic, String, self._action_callback)

    @staticmethod
    def _patch_rosgraph_logger_for_py311() -> None:
        """
        Patch rosgraph's custom logger to avoid a known infinite-loop edge case.

        In some Python 3.11 environments with ROS Noetic, rosgraph's RospyLogger.findCaller
        can loop forever when walking stack frames if no matching frame is found and f_back is None.
        """
        try:
            import os
            import sys
            import inspect
            import rosgraph.roslogging as roslogging
        except Exception:
            # If rosgraph isn't importable yet, let rospy import handle errors.
            return

        if not hasattr(roslogging, "RospyLogger"):
            return

        def _safe_find_caller(self, *args, **kwargs):
            file_name, lineno, func_name = super(roslogging.RospyLogger, self).findCaller(
                *args, **kwargs
            )[:3]
            file_name = os.path.normcase(file_name)

            f = inspect.currentframe()
            if f is not None:
                f = f.f_back
            while hasattr(f, "f_code"):
                co = f.f_code
                filename = os.path.normcase(co.co_filename)
                if filename == file_name and f.f_lineno == lineno and co.co_name == func_name:
                    break
                if f.f_back:
                    f = f.f_back
                else:
                    # Critical fix: break instead of infinite loop.
                    break

            if f is None or not hasattr(f, "f_code"):
                if sys.version_info > (3, 2):
                    return file_name, lineno, func_name, None
                return file_name, lineno, func_name

            if f.f_back and f.f_code and f.f_code.co_name == "_base_logger":
                f = f.f_back
                if f.f_back:
                    f = f.f_back
            co = f.f_code
            func_name2 = co.co_name
            try:
                class_name = f.f_locals["self"].__class__.__name__
                func_name2 = f"{class_name}.{func_name2}"
            except KeyError:
                pass

            if sys.version_info > (3, 2):
                return co.co_filename, f.f_lineno, func_name2, None
            return co.co_filename, f.f_lineno, func_name2

        roslogging.RospyLogger.findCaller = _safe_find_caller

    def reset(self):
        self._step_idx = 0
        with self._lock:
            self._latest_action = None
            self._latest_action_step = -1
            self._last_consumed_action_step = -1

    def _action_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self._rospy.logwarn("RosBridgePolicy: received non-JSON action payload, ignored.")
            return

        step = int(payload.get("step", -1))
        raw_action = payload.get("action", payload)
        action = self._coerce_action(raw_action)
        with self._lock:
            self._latest_action = action
            self._latest_action_step = step

    def _coerce_action(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._coerce_action(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if all(isinstance(v, (int, float)) for v in obj):
                return np.asarray(obj, dtype=np.float32)
            return [self._coerce_action(v) for v in obj]
        return obj

    def _sanitize_observation(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._sanitize_observation(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_observation(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._sanitize_observation(v) for v in obj]
        if isinstance(obj, np.ndarray):
            # Keep payloads bounded for ROS String transport.
            if obj.size <= 256:
                return obj.tolist()
            return {
                "__ndarray_summary__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "min": float(np.nanmin(obj)) if obj.size > 0 else None,
                "max": float(np.nanmax(obj)) if obj.size > 0 else None,
            }
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _build_noop_action(self) -> dict[str, Any]:
        if self.task is not None:
            robot_view = self.task.env.current_robot.robot_view
            return {**robot_view.get_noop_ctrl_dict(["base"]), "done": False}
        return {"done": False}

    def _extract_image_from_observation(self, observation: Any) -> np.ndarray | None:
        if isinstance(observation, list) and len(observation) > 0:
            observation = observation[0]
        if not isinstance(observation, dict):
            return None

        # Prefer common camera keys, then fallback to first image-like tensor.
        preferred_keys = ("head_camera", "exo_camera_1", "wrist_camera", "rgb", "image")
        for key in preferred_keys:
            value = observation.get(key)
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                return value

        for value in observation.values():
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                return value
        return None

    def _to_image_msg(self, frame: np.ndarray):
        img = frame
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                # Best effort normalization for float images.
                max_val = float(np.nanmax(img)) if img.size > 0 else 1.0
                if max_val <= 1.0:
                    img = np.clip(img, 0.0, 1.0) * 255.0
                img = np.nan_to_num(img, nan=0.0)
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            h, w = img.shape
            encoding = "mono8"
            step = w
            data = img.tobytes()
        elif img.ndim == 3 and img.shape[2] >= 3:
            # Keep first 3 channels as rgb8.
            img = img[:, :, :3]
            h, w, _ = img.shape
            encoding = "rgb8"
            step = w * 3
            data = img.tobytes()
        else:
            return None

        msg = self._Image()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = "molmo_spaces_camera"
        msg.height = h
        msg.width = w
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = step
        msg.data = data
        return msg

    def get_action(self, observation):
        frame = self._extract_image_from_observation(observation)
        if frame is not None:
            msg = self._to_image_msg(frame)
            if msg is not None:
                self._obs_pub.publish(msg)
            else:
                self._rospy.logwarn_throttle(2.0, "RosBridgePolicy: failed to encode image.")
        else:
            self._rospy.logwarn_throttle(
                2.0,
                "RosBridgePolicy: no image-like tensor found in observation."
            )

        deadline = time.monotonic() + self.action_timeout_s
        chosen_action = None
        while time.monotonic() < deadline and not self._rospy.is_shutdown():
            with self._lock:
                if self._latest_action is not None:
                    if self._latest_action_step < 0:
                        # Allow action payloads without explicit step field.
                        chosen_action = self._latest_action
                        self._last_consumed_action_step += 1
                        break
                    if self._latest_action_step > self._last_consumed_action_step:
                        chosen_action = self._latest_action
                        self._last_consumed_action_step = self._latest_action_step
                        break
            time.sleep(0.005)

        if chosen_action is None:
            chosen_action = self._build_noop_action()
            self._rospy.logwarn_throttle(2.0, "RosBridgePolicy: action timeout, using noop action.")

        if isinstance(chosen_action, dict):
            # Keep downstream navigation task behavior consistent.
            chosen_action.setdefault("done", False)
            if self.task is not None and "base" not in chosen_action:
                chosen_action["base"] = self.task.env.current_robot.robot_view.get_noop_ctrl_dict(
                    ["base"]
                )["base"]

        self._step_idx += 1
        return chosen_action

    def close(self):
        if hasattr(self, "_action_sub") and self._action_sub is not None:
            self._action_sub.unregister()
            self._action_sub = None
