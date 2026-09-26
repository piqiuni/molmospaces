"""Exclude disabled intervals without erasing previously consumed progress budgets."""

from dataclasses import dataclass
import time


@dataclass
class ProgressPause:
    enabled: bool = True
    last_step: int | None = None
    last_time: float | None = None

    def advance(
        self, enabled: bool, step: int | None = None, now: float | None = None,
    ) -> tuple[int, float]:
        # Exclude transition intervals conservatively (at most one observer tick).
        paused = not enabled or not self.enabled
        steps = (
            max(0, step - self.last_step)
            if paused and step is not None and self.last_step is not None else 0
        )
        seconds = (
            max(0.0, now - self.last_time)
            if paused and now is not None and self.last_time is not None else 0.0
        )
        self.enabled = bool(enabled)
        if step is not None:
            self.last_step = step if self.last_step is None else max(step, self.last_step)
        if now is not None:
            self.last_time = now if self.last_time is None else max(now, self.last_time)
        return steps, seconds


class ExecutionEnableState:
    """Optional ROS heartbeat; absent configuration preserves simulator behavior."""

    def __init__(self, rospy, config):
        from std_msgs.msg import Bool

        self.topic = str(config.get("progress_execution_enabled_topic", ""))
        self.required = bool(config.get("execution_pause_enabled", False))
        self.timeout = max(
            0.1, float(config.get("progress_execution_enabled_timeout_s", 2.0))
        )
        self.state = (False, float("-inf"))
        self.subscriber = None
        if self.topic:
            self.subscriber = rospy.Subscriber(self.topic, Bool, self._callback, queue_size=1)

    def _callback(self, message):
        self.state = (bool(message.data), time.monotonic())

    @property
    def enabled(self):
        enabled, received = self.state
        return (not self.topic and not self.required) or bool(
            self.topic and enabled and 0.0 <= time.monotonic() - received < self.timeout
        )
