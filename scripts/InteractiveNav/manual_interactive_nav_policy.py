from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from molmo_spaces.policy.base_policy import BasePolicy


@dataclass(frozen=True)
class ManualControlEvent:
    name: str
    timestamp: float


@dataclass(frozen=True)
class CameraControlCommand:
    forward: float = 0.0
    right: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


class ManualInteractiveNavPolicy(BasePolicy):
    """Independent keyboard policy for InteractiveNav scene inspection.

    The RBY1M open/close configuration uses a relative holonomic base
    controller, so the returned base action is ``[dx_world, dy_world, d_yaw]``.
    Camera and articulation requests are emitted as side-channel events and are
    handled by the manual scene runner instead of being mixed into robot actions.
    """

    MOVEMENT_KEYS = frozenset({"w", "s", "a", "d"})
    CAMERA_KEYS = frozenset({"i", "k", "j", "l", ";", "'", ".", "/"})
    EDGE_EVENTS = {
        "o": "open_nearest",
        "p": "close_nearest",
        "r": "reset_robot",
        "c": "reset_camera",
        "v": "capture_frame",
        "space": "toggle_pause",
        "esc": "quit",
    }

    def __init__(
        self,
        config: Any = None,
        task: Any = None,
        *,
        env: Any = None,
        linear_step_m: float = 0.035,
        angular_step_rad: float = math.radians(2.5),
        camera_translation_step_m: float = 0.12,
        camera_rotation_step_rad: float = math.radians(2.0),
        start_listener: bool = True,
    ) -> None:
        super().__init__(config, task)
        self.env = env if env is not None else getattr(task, "env", None)
        self.linear_step_m = float(linear_step_m)
        self.angular_step_rad = float(angular_step_rad)
        self.camera_translation_step_m = float(camera_translation_step_m)
        self.camera_rotation_step_rad = float(camera_rotation_step_rad)
        self._pressed: set[str] = set()
        self._events: deque[ManualControlEvent] = deque()
        self._lock = threading.Lock()
        self._listener = None
        if start_listener:
            self.start_listener()

    @staticmethod
    def normalize_key(key: Any) -> str | None:
        if isinstance(key, str):
            value = key.lower()
            if value in {"space", "esc"}:
                return value
            return value if len(value) == 1 else None
        char = getattr(key, "char", None)
        if char is not None:
            return str(char).lower()
        name = getattr(key, "name", None)
        if name in {"space", "esc"}:
            return str(name)
        return None

    def start_listener(self) -> None:
        if self._listener is not None:
            return
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _on_press(self, key: Any) -> None:
        normalized = self.normalize_key(key)
        if normalized is not None:
            self.press_key(normalized)

    def _on_release(self, key: Any) -> None:
        normalized = self.normalize_key(key)
        if normalized is not None:
            self.release_key(normalized)

    def press_key(self, key: str) -> None:
        key = str(key).lower()
        with self._lock:
            first_press = key not in self._pressed
            self._pressed.add(key)
            if first_press and key in self.EDGE_EVENTS:
                self._events.append(
                    ManualControlEvent(self.EDGE_EVENTS[key], time.time())
                )

    def release_key(self, key: str) -> None:
        with self._lock:
            self._pressed.discard(str(key).lower())

    def pressed_keys(self) -> set[str]:
        with self._lock:
            return set(self._pressed)

    def combined_pressed_keys(self, extra_pressed: set[str] | None = None) -> set[str]:
        pressed = self.pressed_keys()
        if extra_pressed:
            pressed.update(str(key).lower() for key in extra_pressed)
        return pressed

    def drain_events(self) -> list[ManualControlEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def yaw_from_pose(pose: np.ndarray) -> float:
        return float(math.atan2(pose[1, 0], pose[0, 0]))

    @classmethod
    def base_delta_from_keys(
        cls,
        pressed: set[str],
        *,
        yaw: float,
        linear_step_m: float,
        angular_step_rad: float,
    ) -> np.ndarray:
        forward = float("w" in pressed) - float("s" in pressed)
        turn = float("a" in pressed) - float("d" in pressed)
        dx = forward * float(linear_step_m) * math.cos(yaw)
        dy = forward * float(linear_step_m) * math.sin(yaw)
        d_yaw = turn * float(angular_step_rad)
        return np.asarray([dx, dy, d_yaw], dtype=np.float32)

    def get_action(
        self,
        observation: Any = None,
        *,
        extra_pressed: set[str] | None = None,
    ) -> dict[str, Any]:
        del observation
        if self.env is None:
            raise RuntimeError("ManualInteractiveNavPolicy requires an environment")
        robot_view = self.env.current_robot.robot_view
        yaw = self.yaw_from_pose(np.asarray(robot_view.base.pose, dtype=float))
        delta = self.base_delta_from_keys(
            self.combined_pressed_keys(extra_pressed),
            yaw=yaw,
            linear_step_m=self.linear_step_m,
            angular_step_rad=self.angular_step_rad,
        )
        return {"base": delta, "done": False}

    def get_camera_command(
        self, extra_pressed: set[str] | None = None
    ) -> CameraControlCommand:
        pressed = self.combined_pressed_keys(extra_pressed)
        return CameraControlCommand(
            forward=(float("i" in pressed) - float("k" in pressed))
            * self.camera_translation_step_m,
            right=(float("l" in pressed) - float("j" in pressed))
            * self.camera_translation_step_m,
            yaw=(float(";" in pressed) - float("'" in pressed))
            * self.camera_rotation_step_rad,
            pitch=(float("." in pressed) - float("/" in pressed))
            * self.camera_rotation_step_rad,
        )

    def reset(self) -> None:
        with self._lock:
            self._pressed.clear()
            self._events.clear()

    def get_phase(self) -> str:
        return "manual_interactive_nav"

    def get_info(self) -> dict[str, Any]:
        return {
            "policy_name": "manual_interactive_nav",
            "pressed_keys": sorted(self.pressed_keys()),
            "camera_command": asdict(self.get_camera_command()),
        }

    def close(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.stop()
