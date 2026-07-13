from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "InteractiveNav"))

from scripts.InteractiveNav.run_nav_ros_sim import (
    NavRosRolloutRunner,
    SceneExecutionTimeout,
    resolve_action_timeout_s,
)


class TimeoutPolicy:
    scene_timeout_s = 60.0
    max_consecutive_action_timeouts = 2
    sim_timing_log_every_n_steps = 0
    step_log_every_n_steps = 0

    def prepare_episode_reset(self):
        pass

    def reset(self):
        pass

    def get_action(self, _observation):
        self.last_action_timed_out = True
        return {"base": np.zeros(3, dtype=np.float32)}


class TimeoutTask:
    def __init__(self):
        self.env = SimpleNamespace(
            current_model=SimpleNamespace(opt=SimpleNamespace(enableflags=0))
        )
        self.config = SimpleNamespace(policy_dt_ms=200.0)
        self.step_count = 0

    def reset(self):
        return {}, {}

    def is_done(self):
        return False

    def step(self, _action):
        self.step_count += 1
        return {}, 0.0, False, False, [{}]


def test_scene_aborts_after_consecutive_action_timeouts():
    task = TimeoutTask()
    with pytest.raises(SceneExecutionTimeout, match="2 consecutive waits"):
        NavRosRolloutRunner.run_single_rollout(
            episode_seed=1,
            task=task,
            policy=TimeoutPolicy(),
        )
    assert task.step_count == 1


def test_scene_timeout_enables_finite_action_wait():
    assert resolve_action_timeout_s(0.0, 600.0) == 5.0
    assert resolve_action_timeout_s(2.0, 600.0) == 2.0
    assert resolve_action_timeout_s(0.0, 0.0) == 0.0
