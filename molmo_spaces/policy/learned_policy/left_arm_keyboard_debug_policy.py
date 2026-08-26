import logging
import threading
from typing import Any

import numpy as np

from molmo_spaces.policy.base_policy import BasePolicy
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


class LeftArmKeyboardDebugPolicy(BasePolicy):
    """
    Debug policy: publish only left_arm action.

    Controls:
    - i: increase current joint value
    - k: decrease current joint value
    - j: select previous joint index
    - l: select next joint index
    """

    def __init__(
        self,
        config,
        task: BaseMujocoTask | None = None,
        joint_delta: float = 0.05,
        initial_left_arm_qpos: list[float] | None = None,
    ) -> None:
        super().__init__(config, task)
        from pynput import keyboard

        self._keyboard = keyboard
        self.joint_delta = float(joint_delta)
        self._lock = threading.Lock()
        self._joint_inc_count = 0
        self._joint_dec_count = 0
        self._joint_prev_count = 0
        self._joint_next_count = 0
        self._joint_idx = 0
        self._left_arm_cmd: np.ndarray | None = None
        self._initial_left_arm_qpos = (
            None
            if initial_left_arm_qpos is None
            else np.asarray(initial_left_arm_qpos, dtype=np.float32).reshape(-1)
        )

        # 0.28303343, -0.02953514,  0.01789065, -0.6400702 ,  0.39170685,   -0.2636598 , -0.03980027
        self._initial_left_arm_qpos = np.array([0.28, 0.0, 0.0, -0.64, 0.39, -0.26, -0.04])
        

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()
        log.info(
            "LeftArmKeyboardDebugPolicy ready: i/k adjust value, j/l switch joint, delta=%.3f",
            self.joint_delta,
        )

    def _on_press(self, key) -> None:
        char = getattr(key, "char", None)
        if char is None:
            return
        with self._lock:
            if char == "i":
                self._joint_inc_count += 1
            elif char == "k":
                self._joint_dec_count += 1
            elif char == "j":
                self._joint_prev_count += 1
            elif char == "l":
                self._joint_next_count += 1

    def reset(self):
        self._joint_idx = 0
        self._left_arm_cmd = None
        with self._lock:
            self._joint_inc_count = 0
            self._joint_dec_count = 0
            self._joint_prev_count = 0
            self._joint_next_count = 0

    def _drain_events(self) -> tuple[int, int, int, int]:
        with self._lock:
            vals = (
                self._joint_inc_count,
                self._joint_dec_count,
                self._joint_prev_count,
                self._joint_next_count,
            )
            self._joint_inc_count = 0
            self._joint_dec_count = 0
            self._joint_prev_count = 0
            self._joint_next_count = 0
            return vals

    def _ensure_left_arm_cmd(self) -> None:
        if self.task is None:
            return
        if self._left_arm_cmd is not None:
            return
        robot_view = self.task.env.current_robot.robot_view
        noop_left = np.asarray(robot_view.get_noop_ctrl_dict(["left_arm"])["left_arm"], dtype=np.float32)
        if self._initial_left_arm_qpos is not None and self._initial_left_arm_qpos.shape == noop_left.shape:
            self._left_arm_cmd = self._initial_left_arm_qpos.copy()
        else:
            self._left_arm_cmd = noop_left.copy()

    def get_action(self, observation: Any):
        _ = observation
        if self.task is None:
            return {"done": False}

        self._ensure_left_arm_cmd()
        if self._left_arm_cmd is None:
            return {"done": False}
        robot_view = self.task.env.current_robot.robot_view

        inc_n, dec_n, prev_n, next_n = self._drain_events()
        dof = int(self._left_arm_cmd.size)
        if dof <= 0:
            return {"done": False}

        if prev_n > 0:
            self._joint_idx = (self._joint_idx - prev_n) % dof
        if next_n > 0:
            self._joint_idx = (self._joint_idx + next_n) % dof
        if inc_n > 0:
            self._left_arm_cmd[self._joint_idx] += self.joint_delta * inc_n
        if dec_n > 0:
            self._left_arm_cmd[self._joint_idx] -= self.joint_delta * dec_n

        if inc_n or dec_n or prev_n or next_n:
            log.info(
                "Left arm debug: joint=%d value=%.4f cmd=%s",
                self._joint_idx,
                float(self._left_arm_cmd[self._joint_idx]),
                np.array2string(self._left_arm_cmd, precision=3, separator=","),
            )

        # Start from full noop command so base/right arm/grippers/head stay still.
        action = robot_view.get_noop_ctrl_dict()
        action["left_arm"] = self._left_arm_cmd.copy()
        action["right_arm"] = self._initial_left_arm_qpos.copy()
        action["done"] = False
        return action

    def close(self) -> None:
        if hasattr(self, "_listener") and self._listener is not None:
            self._listener.stop()
            self._listener = None

