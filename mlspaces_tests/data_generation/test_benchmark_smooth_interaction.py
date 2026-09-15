from types import SimpleNamespace

from scripts.InteractiveNav.evaluation import smooth_interaction


def test_goal_terminal_interrupts_scan_without_closing_or_restoring_view(monkeypatch):
    class Controller:
        def __init__(self, **kwargs):
            self._events = []
            self._restore_view_pending = True
        def enqueue_command(self, command):
            pass
        def before_step(self, task, index):
            pass
        def after_step(self, task, index):
            raise AssertionError("must not advance articulation after goal terminal")
    monkeypatch.setattr(smooth_interaction, "AtomicForceInteractionController", Controller)
    execution = smooth_interaction.run_smooth_interaction(
        object(), {}, object_name="drawer", step=lambda controller, index: True)
    assert execution["result"]["interrupted_by_goal_status"]
    assert execution["task_steps_consumed"] == 1


def test_smooth_scan_records_open_dwell_before_close_and_keeps_empty_regions(monkeypatch):
    recorded = []
    transitions = ["open"] + ["observe"] * 10 + ["close"]

    class Controller:
        def __init__(self, **kwargs):
            assert kwargs["interaction_execution_mode"] == "smooth"
            assert kwargs["drawer_observation_steps"] == 10
            assert kwargs["bypass_unsafe_open_sweep"] is True
            self._restore_view_pending = False
            self._events = []

        def enqueue_command(self, command):
            assert command["sequence_type"] == "drawer_scan"
            assert command["open_regions"] == []

        def before_step(self, task, index):
            task.phase = transitions[index]

        def after_step(self, task, index):
            assert recorded[-1] == transitions[index]
            if index == len(transitions) - 1:
                return {"success": False, "reason": "close_failed"}

        def after_task_step(self):
            pass

    monkeypatch.setattr(smooth_interaction, "AtomicForceInteractionController", Controller)
    task = SimpleNamespace()
    result = smooth_interaction.run_smooth_interaction(
        task, {"sequence_type": "drawer_scan", "open_regions": []},
        object_name="private_container", step=lambda controller, index: recorded.append(task.phase),
    )
    assert recorded == transitions
    assert result["task_steps_consumed"] == 12
    assert result["result"]["success"] is False


def test_smooth_target_discovery_keeps_group_index_for_result_serialization(monkeypatch):
    from scripts.InteractiveNav.evaluation import benchmark_runner as runner

    controller = SimpleNamespace(_pending={"phase": "observe", "group_index": 2},
                                 view_torso_target=lambda: None)
    def run(task, command, *, object_name, step):
        controller._pending["phase"] = "open"
        assert step(controller, 0) is False
        controller._pending["phase"] = "observe"
        for index in range(1, 11):
            assert step(controller, index) is (index == 10)
        return {"result": {"success": True}, "task_steps_consumed": 11, "events": []}
    monkeypatch.setattr(smooth_interaction, "run_smooth_interaction", run)
    for name in ["_report_interaction_progress", "_publish_restricted_ros_frame",
                 "_capture_head_frame", "_discard_task_rollout_cache", "_atomic_json"]:
        monkeypatch.setattr(runner, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "joint_open_fraction", lambda *args: .98)
    monkeypatch.setattr(runner, "_drawer_scan_target_evidence", lambda *args: {"visible": True})
    monkeypatch.setattr(runner, "_poll_restricted_goal_status", lambda **kwargs: ("target_found", None, {}))
    bridge = SimpleNamespace(get_action=lambda *a, **k: {}, _step_idx=10)
    task = SimpleNamespace(env=object(), get_observations=lambda: {}, step=lambda action: None)
    from pathlib import Path
    runtime = SimpleNamespace(smooth_bridge=bridge, goal_status_observer=object())
    result = runner._execute_native_smooth(task, runtime,
        SimpleNamespace(public_command={}, sequence_type="drawer_scan"), "private",
        [SimpleNamespace(joint_name="drawer")], {},
        SimpleNamespace(record_video=False, policy_dt_ms=200, output_dir=Path("/home/ldl/tmp")), 0, [])
    assert result["target_discovery"]["group_index"] == 2
    assert result["metadata"]["target_discovery"] == result["target_discovery"]
    assert runtime.pending_goal_terminal[0] == "target_found"
