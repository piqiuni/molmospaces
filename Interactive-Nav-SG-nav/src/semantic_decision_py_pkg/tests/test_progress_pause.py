from types import SimpleNamespace

import pytest

from semantic_decision_py_pkg.behavior_execution import (
    NavigationProgressWatchdog, SemanticNavigationProgressSupervisor,
)
from semantic_decision_py_pkg.progress_pause import ExecutionEnableState, ProgressPause


def test_older_worker_sample_cannot_double_count_paused_steps():
    pause = ProgressPause()
    pause.advance(False, 10)
    assert pause.advance(False, 20)[0] == 10
    assert pause.advance(False, 15)[0] == 0
    assert pause.advance(False, 21)[0] == 1


@pytest.mark.parametrize("use_steps", [False, True])
def test_watchdog_preserves_used_budget_during_pause(use_steps):
    watchdog = NavigationProgressWatchdog(timeout_s=10, timeout_task_steps=10 if use_steps else None)
    pose = (0, 0, 0)
    watchdog.reset(pose, 0, task_step_index=0)
    assert not watchdog.observe(pose, 4, task_step_index=4)
    watchdog.set_execution_enabled(False, 4, 4)
    assert not watchdog.observe(pose, 100, task_step_index=100)
    watchdog.set_execution_enabled(True, 104, 104)
    assert not watchdog.observe(pose, 109, task_step_index=109)
    assert watchdog.observe(pose, 110, task_step_index=110)


def test_supervisor_pauses_both_subgoal_and_mission_without_reset():
    supervisor = SemanticNavigationProgressSupervisor(subgoal_timeout_task_steps=10, mission_timeout_task_steps=20)
    def observe(step, enabled=True):
        return supervisor.observe(subgoal_key="a", pose=(0, 0), task_step_index=step, execution_enabled=enabled)
    observe(0)
    observe(4)
    observe(4, False)
    assert observe(100, False)["semantic_progress_paused"]
    assert not observe(104)["subgoal_stalled"]
    assert not observe(109)["subgoal_stalled"]
    assert observe(110)["subgoal_stalled"]
    assert supervisor.mission_reference_step_index == 100
    assert supervisor.subgoal_reference_step_index == 100


def test_disabled_start_never_initializes_budget():
    supervisor = SemanticNavigationProgressSupervisor()
    for step in (0, 500, 1000):
        assert not supervisor.observe(subgoal_key="idle", pose=(0, 0), task_step_index=step, execution_enabled=False)["mission_stalled"]
    assert supervisor.mission_reference_step_index is None
    supervisor.observe(subgoal_key="idle", pose=(0, 0), task_step_index=1001)
    assert supervisor.mission_reference_step_index == 1001


def test_execution_heartbeat_is_opt_in_and_fail_closed(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("semantic_decision_py_pkg.progress_pause.time.monotonic", lambda: now[0])
    ros = SimpleNamespace(Subscriber=lambda *args, **kwargs: object())
    assert ExecutionEnableState(ros, {}).enabled
    state = ExecutionEnableState(ros, {"progress_execution_enabled_topic": "/enabled"})
    assert not state.enabled
    state._callback(SimpleNamespace(data=True))
    assert state.enabled
    now[0] = 3
    assert not state.enabled
    state._callback(SimpleNamespace(data=False))
    assert not state.enabled


def test_execution_pause_requires_fresh_explicit_enable_even_when_topic_missing(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("semantic_decision_py_pkg.progress_pause.time.monotonic", lambda: now[0])
    ros = SimpleNamespace(Subscriber=lambda *args, **kwargs: object())
    assert not ExecutionEnableState(ros, {"execution_pause_enabled": True}).enabled
    state = ExecutionEnableState(ros, {"execution_pause_enabled": True,
                                      "progress_execution_enabled_topic": "/enabled"})
    assert not state.enabled
    state._callback(SimpleNamespace(data=True))
    assert state.enabled
    now[0] = 2.1
    assert not state.enabled
    state._callback(SimpleNamespace(data=False))
    assert not state.enabled
    now[0] = 1000
    assert not state.enabled
