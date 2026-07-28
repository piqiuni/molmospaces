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

    def set_history_retention(self, _enabled):
        pass

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


class CompletionAfterActionMonitor:
    def __init__(self) -> None:
        self.requested = False
        self.prepared = 0
        self.finalized_steps = []

    def prepare(self) -> None:
        self.prepared += 1

    def should_stop(self, _completed_steps: int) -> bool:
        return self.requested

    def finalize(self, completed_steps: int) -> None:
        self.finalized_steps.append(completed_steps)


class CompletionAfterActionPolicy(TimeoutPolicy):
    max_consecutive_action_timeouts = 1

    def __init__(self) -> None:
        self.completion_monitor = CompletionAfterActionMonitor()

    def get_action(self, _observation):
        # Model a semantic terminal update arriving while the bridge is
        # blocked waiting for a ROS navigation command.
        self.last_action_timed_out = True
        self.completion_monitor.requested = True
        return {"base": np.zeros(3, dtype=np.float32)}


def test_completion_arriving_during_action_wait_preempts_timeout_guard():
    task = TimeoutTask()
    policy = CompletionAfterActionPolicy()

    success = NavRosRolloutRunner.run_single_rollout(
        episode_seed=1,
        task=task,
        policy=policy,
    )

    assert success is False
    assert task.step_count == 0
    assert policy.completion_monitor.prepared == 1
    assert policy.completion_monitor.finalized_steps == [0]


class FinishPolicy(TimeoutPolicy):
    max_consecutive_action_timeouts = 0

    def __init__(self):
        self.finish_calls = []
        self.last_action_timed_out = False

    def get_action(self, _observation):
        self.last_action_timed_out = False
        return {"base": np.zeros(3, dtype=np.float32)}

    def finish_episode(self, step_index=None, *, reason=""):
        self.finish_calls.append((step_index, reason))


class FinishTask(TimeoutTask):
    def is_done(self):
        return self.step_count >= 1

    def judge_success(self):
        return False


def test_rollout_finishes_episode_before_returning():
    task = FinishTask()
    policy = FinishPolicy()

    success = NavRosRolloutRunner.run_single_rollout(
        episode_seed=1,
        task=task,
        policy=policy,
    )

    assert success is False
    assert policy.finish_calls == [(1, "task_done")]
