from __future__ import annotations

import base64
import importlib.util
import json
import math
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import cv2
import numpy as np
import yaml


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MLLM_SCRIPTS = PACKAGE_SCRIPTS.parents[1] / "semantic_mllm_py_pkg" / "scripts"
for path in (PACKAGE_SCRIPTS, MLLM_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_interaction_step_sync_pauses_navigation_clock(executor_module):
    from semantic_decision_py_pkg.behavior_execution import SemanticNavigationProgressSupervisor
    executor = executor_module.SemanticBehaviorExecutor.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = SimpleNamespace(state=executor_module.STATE_INTERACTING)
    executor.startup_scan_enabled = False
    executor.rear_goal_prerotate_step_sync_enabled = False
    executor.interaction_final_align_enabled = False
    supervisor = SemanticNavigationProgressSupervisor(mission_timeout_task_steps=18)
    executor._semantic_navigation_progress = supervisor
    supervisor.observe(subgoal_key="old", pose=(0.0, 0.0), task_step_index=1)
    executor._step_sync_callback(SimpleNamespace(data=json.dumps({"step_index": 100, "action_source": "navigation_hold"})))
    detail = supervisor.observe(subgoal_key="new", pose=(0.0, 0.0), task_step_index=101)
    assert detail["mission_elapsed_task_steps"] == 1
    assert not detail["mission_stalled"]


def _stub_module(monkeypatch, name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_ros_import_stubs(monkeypatch) -> None:
    """Load the executor's pure callback logic without a ROS installation."""

    class _Placeholder:
        pass

    class _String:
        def __init__(self, data: str = "") -> None:
            self.data = data

    _stub_module(monkeypatch, "actionlib", SimpleActionClient=_Placeholder)
    _stub_module(monkeypatch, "rospy.numpy_msg", numpy_msg=lambda message_type: message_type)
    _stub_module(
        monkeypatch,
        "rospy",
        Time=_Placeholder,
        Duration=_Placeholder,
        Publisher=_Placeholder,
        Subscriber=_Placeholder,
        Timer=_Placeholder,
        ROSException=RuntimeError,
        is_shutdown=lambda: False,
        loginfo=lambda *_args, **_kwargs: None,
        logwarn=lambda *_args, **_kwargs: None,
    )
    _stub_module(monkeypatch, "tf", TransformListener=_Placeholder)
    _stub_module(monkeypatch, "actionlib_msgs")
    _stub_module(
        monkeypatch,
        "actionlib_msgs.msg",
        GoalStatus=type(
            "GoalStatus",
            (),
            {
                "PREEMPTED": 1,
                "SUCCEEDED": 2,
                "ABORTED": 3,
                "REJECTED": 4,
                "RECALLED": 5,
                "LOST": 6,
                "PENDING": 10,
                "ACTIVE": 11,
                "PREEMPTING": 12,
                "RECALLING": 13,
            },
        ),
        GoalStatusArray=_Placeholder,
    )
    _stub_module(monkeypatch, "geometry_msgs")
    _stub_module(
        monkeypatch,
        "geometry_msgs.msg",
        PointStamped=_Placeholder,
        PoseStamped=_Placeholder,
        Twist=_Placeholder,
        TwistStamped=_Placeholder,
    )
    _stub_module(monkeypatch, "map_msgs")
    _stub_module(monkeypatch, "map_msgs.msg", OccupancyGridUpdate=_Placeholder)
    _stub_module(monkeypatch, "move_base_msgs")
    _stub_module(
        monkeypatch,
        "move_base_msgs.msg",
        MoveBaseAction=_Placeholder,
        MoveBaseGoal=_Placeholder,
    )
    _stub_module(monkeypatch, "nav_msgs")
    _stub_module(monkeypatch, "nav_msgs.msg", OccupancyGrid=_Placeholder, Path=_Placeholder)
    _stub_module(monkeypatch, "nav_msgs.srv", GetPlan=_Placeholder)
    _stub_module(monkeypatch, "sensor_msgs")
    _stub_module(monkeypatch, "sensor_msgs.msg", Image=_Placeholder)
    _stub_module(monkeypatch, "std_msgs")
    _stub_module(monkeypatch, "std_msgs.msg", String=_String)


@pytest.fixture
def executor_module(monkeypatch):
    try:
        import actionlib  # noqa: F401
        import rospy  # noqa: F401
        import tf  # noqa: F401
    except ImportError:
        _install_ros_import_stubs(monkeypatch)

    module_name = "_semantic_behavior_executor_static_test"
    module_path = PACKAGE_SCRIPTS / "semantic_behavior_executor.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _portal_selection() -> dict:
    return {
        "decision_id": "decision_static",
        "candidate_id": "interaction:portal_static:open",
        "behavior_type": "INTERACT",
        "target_id": "portal_static",
        "target_name": "door_0003",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "metadata": {"node_type": "portal", "requires_approach": False},
        "interaction_command": {
            "node_id": "portal_static",
            "object_id": "door_0003",
            "node_type": "portal",
            "action": "open",
            "expected_state": "open",
        },
    }


@pytest.mark.parametrize(
    "timeout_reason",
    ("navigation_timeout", "interaction_navigation_timeout"),
)
def test_tick_leaves_navigation_timeout_to_step_synchronized_worker(
    executor_module, timeout_reason: str
) -> None:
    """The periodic ROS callback must not preempt a live navigation worker.

    Navigation workers have an evaluator-step budget and report the final
    timeout themselves.  If ``_tick`` consumes the wall-clock reason first,
    an interaction can be aborted while its terminal-yaw controller is still
    making progress (the H1 regression).
    """

    class _Machine:
        state = executor_module.STATE_APPROACH_INTERACTION
        config = SimpleNamespace(interaction_timeout_s=30.0)

        def summary(self):
            return {}

        def fail_timeout(self, reason):
            pytest.fail(f"_tick unexpectedly consumed {reason!r}")

    class _Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class _MoveBase:
        def __init__(self):
            self.cancel_count = 0

        def cancel_goal(self):
            self.cancel_count += 1

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = _Machine()
    executor.selection = {"decision_id": "decision_live"}
    executor._active_navigation_run_tokens = {"decision_live": 1}
    executor._effective_timeout_reason_locked = lambda: timeout_reason
    executor._explore_reservation_publish_count = 0
    executor._explore_feedback_received_count = 0
    executor._explore_feedback_matched_count = 0
    executor._explore_feedback_ignored_count = 0
    executor._last_explore_feedback = {}
    executor.external_recovery_request_enabled = False
    executor._external_recovery_requests = {}
    executor._external_recovery_received_count = 0
    executor._external_recovery_accepted_count = 0
    executor._external_recovery_rejected_count = 0
    executor._external_recovery_consumed_count = 0
    executor._external_recovery_last_feedback = {}
    executor._drawer_scan_wait_contexts = {}
    executor._startup_scan_progress = {}
    executor._post_interaction_visual_audits = []
    executor._drawer_scan_execution_wait_summary_locked = lambda: {}
    executor._container_inner_corridor_summary_locked = lambda: {}
    executor.state_pub = _Publisher()
    executor.move_base = _MoveBase()
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)

    executor._tick(None)

    assert executor.move_base.cancel_count == 0
    assert dispatched == []
    assert len(executor.state_pub.messages) == 1


def test_tick_releases_observation_request_before_retry(
    executor_module,
) -> None:
    class _Machine:
        state = executor_module.STATE_WAITING_FOR_INTERACTION_OBSERVATION
        config = SimpleNamespace(interaction_timeout_s=30.0)

        def __init__(self):
            self.timeout_calls = []

        def summary(self):
            return {}

        def fail_timeout(self, reason, task_step_index=None):
            self.timeout_calls.append((reason, task_step_index))
            return []

    class _Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = _Machine()
    executor.selection = {"decision_id": "decision_observation"}
    executor._interaction_observation_requests = {
        "decision_observation": {"request_id": "old_request"}
    }
    executor._latest_step_sync_index = 120
    executor._effective_timeout_reason_locked = (
        lambda: "interaction_observation_timeout"
    )
    executor._explore_reservation_publish_count = 0
    executor._explore_feedback_received_count = 0
    executor._explore_feedback_matched_count = 0
    executor._explore_feedback_ignored_count = 0
    executor._last_explore_feedback = {}
    executor.external_recovery_request_enabled = False
    executor._external_recovery_requests = {}
    executor._external_recovery_received_count = 0
    executor._external_recovery_accepted_count = 0
    executor._external_recovery_rejected_count = 0
    executor._external_recovery_consumed_count = 0
    executor._external_recovery_last_feedback = {}
    executor._drawer_scan_wait_contexts = {}
    executor._startup_scan_progress = {}
    executor._post_interaction_visual_audits = []
    executor._late_interaction_observations = []
    executor._drawer_scan_execution_wait_summary_locked = lambda: {}
    executor._container_inner_corridor_summary_locked = lambda: {}
    executor.state_pub = _Publisher()
    executor.move_base = SimpleNamespace(cancel_goal=lambda: None)
    executor._dispatch = lambda _commands: None

    executor._tick(None)

    assert executor._interaction_observation_requests == {}
    assert executor.machine.timeout_calls == [
        ("interaction_observation_timeout", 120)
    ]
    assert len(executor.state_pub.messages) == 1


def test_observation_only_terminal_publishes_mapper_result(
    executor_module,
) -> None:
    selection = _portal_selection()
    selection["episode_id"] = "episode_1"
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = dict(selection)
    executor.latest_graph = {
        "nodes": [
            {
                "id": "portal_static",
                "type": "portal",
                "attributes": {
                    "portal_aperture_evidence": {
                        "open_aperture": "visible",
                        "confidence": 0.95,
                    },
                    "visual_evidence_truncated": False,
                },
            }
        ]
    }
    executor._latest_attribute_updates = {}
    executor._latest_step_sync_index = 88
    executor.machine = SimpleNamespace(
        state=executor_module.STATE_INTERACTING,
        reset=lambda: None,
    )
    executor.model_events = []
    executor._navigation_failure_recovery_attempts = {}
    executor._drawer_scan_wait_records = {}
    executor._drawer_scan_wait_contexts = {}
    executor._clear_drawer_scan_execution_wait_locked = lambda: None
    executor._publish_feedback = lambda *_args: None
    executor.interaction_result_pub = SimpleNamespace(
        messages=[],
        publish=lambda message: executor.interaction_result_pub.messages.append(
            json.loads(message.data)
        ),
    )
    executor.move_base = SimpleNamespace(cancel_goal=lambda: None)

    executor._finish_terminal(
        {
            "success": True,
            "detail": {
                "action": "open",
                "state": "open",
                "post_state": "open",
                "observation_outcome": "finish_without_action",
                "action_executed": False,
                "attribute_source": "mllm_attribute_inference",
                "portal_state_consensus": {
                    "accepted": True,
                    "stable_state": "open",
                },
                "observation_capture_step": 87,
            },
        }
    )

    assert len(executor.interaction_result_pub.messages) == 1
    result = executor.interaction_result_pub.messages[0]
    assert result["node_id"] == "portal_static"
    assert result["object_id"] == "door_0003"
    assert result["observation_outcome"] == "finish_without_action"
    assert result["action_executed"] is False
    assert result["portal_aperture_evidence"]["open_aperture"] == "visible"
    assert result["capture_step"] == 87


def test_remembered_portal_reobserve_terminal_publishes_mapper_result(
    executor_module,
) -> None:
    selection = {
        "decision_id": "decision_reobserve_result",
        "candidate_id": "reobserve_portal:portal_static",
        "behavior_type": "NAVIGATE",
        "target_id": "portal_static",
        "target_name": "door_0003",
        "episode_id": "episode_1",
        "metadata": {
            "node_type": "portal",
            "observation_only_reobserve": True,
            "reobserve_object_id": "door_0003",
            "last_interaction_observation": {
                "attribute_source": "mllm_attribute_inference",
                "attribute_status": "ready",
                "is_currently_visible": True,
                "state": "closed",
                "observation_capture_step": 87,
            },
        },
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = dict(selection)
    executor.latest_graph = {
        "episode_id": "episode_1",
        "nodes": [
            {
                "id": "portal_static",
                "type": "portal",
                "attributes": {"instance_id": "door_0003"},
            }
        ],
    }
    executor._latest_attribute_updates = {}
    executor._latest_step_sync_index = 88
    executor.machine = SimpleNamespace(
        state="SUCCEEDED",
        reset=lambda: None,
    )
    executor.model_events = []
    executor._navigation_failure_recovery_attempts = {}
    executor._drawer_scan_wait_records = {}
    executor._drawer_scan_wait_contexts = {}
    executor._clear_drawer_scan_execution_wait_locked = lambda: None
    executor._publish_feedback = lambda *_args: None
    executor.interaction_result_pub = SimpleNamespace(
        messages=[],
        publish=lambda message: executor.interaction_result_pub.messages.append(
            json.loads(message.data)
        ),
    )
    executor.move_base = SimpleNamespace(cancel_goal=lambda: None)

    executor._finish_terminal(
        {
            "success": True,
            "detail": {
                "action": "open",
                "state": "closed",
                "post_state": "closed",
                "observation_outcome": "finish_without_action",
                "reobserve_only": True,
                "action_executed": False,
                "attribute_source": "mllm_attribute_inference",
                "observation_capture_step": 87,
            },
        }
    )

    assert len(executor.interaction_result_pub.messages) == 1
    result = executor.interaction_result_pub.messages[0]
    assert result["node_id"] == "portal_static"
    assert result["object_id"] == "door_0003"
    assert result["observation_outcome"] == "finish_without_action"
    assert result["action_executed"] is False
    assert result["state"] == "closed"
    assert result["capture_step"] == 87


def test_remembered_portal_navigation_completion_enters_m1_barrier(
    executor_module,
) -> None:
    candidate = {
        "decision_id": "decision_reobserve_path",
        "candidate_id": "reobserve_portal:portal_static",
        "behavior_type": "NAVIGATE",
        "target_id": "portal_static",
        "target_name": "door",
        "goal_xyyaw": [1.0, 2.0, 0.5],
        "metadata": {"reobserve_interaction_target": True},
    }

    class _Machine:
        def __init__(self) -> None:
            self.candidate = dict(candidate)
            self.state = executor_module.STATE_NAVIGATING
            self.calls = []

        def on_remembered_portal_navigation_result(
            self,
            success,
            detail=None,
            object_id="",
            task_step_index=None,
        ):
            assert success is True
            self.calls.append(
                {
                    "object_id": object_id,
                    "task_step_index": task_step_index,
                }
            )
            self.state = executor_module.STATE_WAITING_FOR_INTERACTION_OBSERVATION
            return [
                {
                    "kind": "request_interaction_observation",
                    "candidate": self.candidate,
                    "object_id": object_id,
                }
            ]

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = dict(candidate)
    executor.latest_graph = {
        "nodes": [
            {
                "id": "portal_static",
                "type": "portal",
                "attributes": {
                    "instance_id": "door_0003",
                    "source_object_name": "door_0003",
                },
            }
        ]
    }
    executor.machine = _Machine()
    executor._active_navigation_run_tokens = {"decision_reobserve_path": 1}
    executor._latest_step_sync_index = 273
    executor._latest_rgb_step_seq = 273
    executor._rear_goal_turn_locks = {}
    executor.stuck_recovery_enabled = False
    executor.ablation = SimpleNamespace(module3="rule_verified")
    executor._navigation_run_is_active = lambda *_args: True
    executor._clear_rear_dwa_monitor = lambda *_args: None
    executor._consume_container_inner_corridor_result = lambda *_args, **_kwargs: False
    executor._attempt_single_goal_navigation_recovery = lambda *_args: (None, {})
    executor._maybe_run_stuck_recovery = lambda *_args: {}
    executor._needs_fresh_drawer_scan_locked = lambda: False
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)

    executor._handle_navigation_result(
        "decision_reobserve_path",
        True,
        {"status": "SUCCEEDED"},
        navigation_run_token=1,
    )

    assert executor.machine.calls == [
        {"object_id": "door_0003", "task_step_index": 273}
    ]
    assert [command["kind"] for command in dispatched] == [
        "request_interaction_observation"
    ]
    assert dispatched[0]["object_id"] == "door_0003"


def test_static_portal_result_skips_mllm_continuation_and_costmap_baseline(
    executor_module,
) -> None:
    selection = _portal_selection()
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = selection
    executor.ablation = SimpleNamespace(module3="mllm_skill_verified")
    executor.active_skill_plan = {"visual_operation_plan": {}}
    executor.pending_skill_actions = [{"action": "close"}]
    executor.evaluator_opaque_open_only = False
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.machine.start(selection, now=0.0)
    # Populate baseline fields so a regression that reaches the articulated
    # code path has a meaningful observable side effect instead of failing on
    # a missing test-only attribute.
    executor._post_interaction_costmap_baselines = {}
    executor._post_interaction_raw_map_barriers = {}
    executor._post_interaction_planning_map_barriers = {}
    executor._global_costmap_received_count = 10
    executor._latest_global_costmap_header_seq = 10
    executor._global_costmap_update_received_count = 11
    executor._latest_global_costmap_update_header_seq = 11
    executor._raw_occupancy_received_count = 12
    executor._latest_raw_occupancy_header_seq = 12
    executor._latest_raw_occupancy_header_stamp_sec = 1.0
    executor._planning_occupancy_received_count = 13
    executor._latest_planning_occupancy_header_seq = 13
    executor._latest_planning_occupancy_header_stamp_sec = 1.0
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)
    executor._candidate_for_skill_action = lambda *_args: pytest.fail(
        "static portal result must not advance an MLLM subaction"
    )

    payload = {
        "command_id": "decision_static:interaction:portal_static:open",
        "decision_id": "decision_static",
        "candidate_id": "interaction:portal_static:open",
        "event_id": "decision_static_interaction_001",
        "node_id": "portal_static",
        "object_id": "door_0003",
        "node_type": "portal",
        "action": "open",
        "success": True,
        "status": "SUCCEEDED",
        "interaction_capability": "static",
        "state": "static_open",
        "post_state": "static_open",
        "source": "executor_static_portal",
    }
    executor._interaction_result_callback(
        SimpleNamespace(data=json.dumps(payload, separators=(",", ":")))
    )

    assert executor.pending_skill_actions == []
    assert executor._post_interaction_costmap_baselines == {}
    assert executor.machine.state == "SUCCEEDED"
    assert len(dispatched) == 1
    assert dispatched[0]["kind"] == "terminal"
    assert dispatched[0]["success"] is True
    assert dispatched[0]["detail"]["verification_mode"] == "direct_static_portal_feedback"


def test_failed_interaction_navigation_is_marked_for_bounded_reachability(
    executor_module,
) -> None:
    selection = _portal_selection()
    selection["metadata"] = {**selection["metadata"], "requires_approach": True}
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = selection
    executor.ablation = SimpleNamespace(module3="rule_verified")
    executor.stuck_recovery_enabled = False
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.machine.start(selection, now=0.0)
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)

    executor._handle_navigation_result(
        "decision_static", False, {"reason": "navigation_timeout"}
    )

    assert len(dispatched) == 1
    assert dispatched[0]["kind"] == "terminal"
    assert dispatched[0]["success"] is False
    assert dispatched[0]["detail"]["failure_stage"] == (
        "interaction_approach_exhausted"
    )
    assert dispatched[0]["detail"]["reason"] == (
        "interaction_approach_options_exhausted"
    )


def test_single_active_aborted_goal_uses_one_safe_executor_recovery(
    executor_module,
) -> None:
    """One ABORTED goal recovers before the legacy multi-subgoal watchdog."""

    candidate = {
        "decision_id": "decision_recovery",
        "candidate_id": "frontier:4:8",
        "behavior_type": "NAVIGATE",
        "goal_xyyaw": [2.0, 0.0, 0.0],
        "metadata": {"frame_id": "map"},
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = dict(candidate)
    executor.machine = SimpleNamespace(
        candidate=dict(candidate), state=executor_module.STATE_NAVIGATING
    )
    executor.map_frame = "map"
    executor.navigation_failure_recovery_enabled = True
    executor.navigation_failure_recovery_max_attempts = 1
    executor._navigation_failure_recovery_attempts = {}
    executor._last_rear_goal_recovery_detail = {}
    executor._navigation_is_current = lambda decision_id: decision_id == "decision_recovery"
    executor._attempt_rear_goal_reverse = lambda decision_id: (
        decision_id == "decision_recovery",
        {"reason": "rear_goal_reverse_complete", "costmap_age_s": 0.01},
    )
    turns = []
    executor._prerotate_for_rear_goal = (
        lambda *args, **kwargs: turns.append((args, kwargs)) or True
    )

    restart, detail = executor._attempt_single_goal_navigation_recovery(
        "decision_recovery",
        False,
        {
            "reason": "navigation_terminal_failure",
            "status": "ABORTED",
        },
    )

    assert restart is not None
    assert restart["candidate"]["candidate_id"] == "frontier:4:8"
    assert restart["start_goal_option_index"] == 0
    assert detail["recovery_owner"] == "semantic_behavior_executor"
    assert detail["recovered"] is True
    assert turns[0][1]["allow_reverse"] is False
    # The one-attempt budget prevents a stuck single goal from becoming a
    # reverse/replan loop when the retried action aborts again.
    restart_again, detail_again = executor._attempt_single_goal_navigation_recovery(
        "decision_recovery",
        False,
        {"reason": "navigation_stagnation"},
    )
    assert restart_again is None
    assert detail_again == {}


def test_portal_aperture_observation_is_forwarded_without_private_metadata(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.evaluator_opaque_open_only = False
    executor.interaction_command_sequence = 0
    executor.latest_image_sequence = 0
    executor._command_id = lambda _candidate: "portal-command"
    published = []
    executor.interaction_command_pub = SimpleNamespace(
        publish=lambda message: published.append(json.loads(message.data))
    )
    candidate = _portal_selection()
    candidate["interaction_command"]["portal_aperture_observation"] = {
        "door_leaf": "absent",
        "connectivity": "open",
        "confidence": 0.8,
        "private_joint_name": "must_not_escape",
    }

    executor._publish_interaction_command(candidate)

    assert published[0]["portal_aperture_observation"] == {
        "door_leaf": "absent",
        "connectivity": "open",
        "confidence": 0.8,
    }
    assert "private_joint_name" not in published[0]["portal_aperture_observation"]


def test_interaction_front_tolerances_survive_bridge_payload_compaction(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.evaluator_opaque_open_only = False
    executor.interaction_command_sequence = 0
    executor.latest_image_sequence = 0
    executor._command_id = lambda _candidate: "container-command"
    published = []
    executor.interaction_command_pub = SimpleNamespace(
        publish=lambda message: published.append(json.loads(message.data))
    )
    candidate = _portal_selection()
    candidate["interaction_command"].update(
        {
            "interaction_front_position_tolerance_rad": 0.261799,
            "interaction_front_yaw_tolerance_rad": 0.261799,
            "interaction_ready_yaw_tolerance_rad": 0.20,
        }
    )

    executor._publish_interaction_command(candidate)

    assert published[0]["interaction_front_position_tolerance_rad"] == pytest.approx(
        0.261799
    )
    assert published[0]["interaction_front_yaw_tolerance_rad"] == pytest.approx(
        0.261799
    )
    assert published[0]["navigation_goal_tolerance_contract_explicit"] is False
    assert published[0]["navigation_goal_position_tolerance_m"] == pytest.approx(
        published[0]["interaction_ready_distance_m"]
    )
    assert published[0]["navigation_goal_yaw_tolerance_rad"] == pytest.approx(0.20)


def test_physical_retries_publish_unique_command_ids(executor_module) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.evaluator_opaque_open_only = False
    executor.interaction_command_sequence = 0
    executor.latest_image_sequence = 0
    executor._command_id = lambda _candidate: "decision:candidate"
    published = []
    executor.interaction_command_pub = SimpleNamespace(
        publish=lambda message: published.append(json.loads(message.data))
    )
    candidate = _portal_selection()

    executor._publish_interaction_command(candidate)
    executor._publish_interaction_command(candidate)

    assert [payload["command_id"] for payload in published] == [
        "decision:candidate:interaction:001",
        "decision:candidate:interaction:002",
    ]
    assert published[0]["candidate_id"] == published[1]["candidate_id"]


def test_unknown_portal_waits_for_its_matching_fresh_m1_update(
    executor_module,
) -> None:
    selection = _portal_selection()
    selection["episode_id"] = "episode_1"
    selection["metadata"] = {
        **selection["metadata"],
        "observation_required": True,
        "reobserve": True,
        "interaction_observation_source": "mllm_attribute_inference",
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = selection
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    commands = executor.machine.start(selection, now=0.0)
    assert executor.machine.state == "WAITING_FOR_INTERACTION_OBSERVATION"
    assert commands[0]["kind"] == "request_interaction_observation"
    executor.latest_graph = {"capture_step": 42, "episode_id": "episode_1"}
    executor._latest_rgb_step_seq = 42
    executor.interaction_observation_sequence = 0
    executor._interaction_observation_requests = {}
    executor._latest_attribute_updates = {}
    published = []
    executor.attribute_refresh_request_pub = SimpleNamespace(
        publish=lambda message: published.append(json.loads(message.data))
    )
    executor._publish_interaction_observation_request(commands[0])

    assert published == [
        {
            "object_id": "door_0003",
            "episode_id": "episode_1",
            "minimum_capture_step": 42,
            "reason": "mllm_portal_state_unknown",
            "request_id": "decision_static:m1:001",
        }
    ]
    # A duplicate dispatch while this targeted M1 request is unresolved must
    # leave its token authoritative. Otherwise the valid first response becomes
    # unmatchable and a later visual fallback can advance the interaction.
    executor._publish_interaction_observation_request(commands[0])
    assert len(published) == 1
    assert executor.interaction_observation_sequence == 1
    assert executor._interaction_observation_requests["decision_static"][
        "request_id"
    ] == "decision_static:m1:001"
    dispatched = []
    executor._dispatch = lambda next_commands: dispatched.extend(next_commands)

    # A pending acknowledgement and an unrelated targeted response cannot
    # authorize the interaction.
    executor._attribute_update_callback(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "episode_1",
                    "updates": [
                        {
                            "object_id": "door_0003",
                            "attribute_status": "pending",
                            "targeted_refresh": True,
                            "targeted_refresh_request_id": "decision_static:m1:001",
                        }
                    ],
                }
            )
        )
    )
    assert dispatched == []
    executor._attribute_update_callback(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "episode_1",
                    "updates": [
                        {
                            "object_id": "door_0003",
                            "attribute_status": "ready",
                            "targeted_refresh": True,
                            "targeted_refresh_request_id": "other-request",
                            "observation_capture_step": 43,
                            "source": "mllm_attribute_inference",
                            "coarse_state": "closed",
                        }
                    ],
                }
            )
        )
    )
    assert dispatched == []

    executor._attribute_update_callback(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "episode_1",
                    "updates": [
                        {
                            "object_id": "door_0003",
                            "attribute_status": "ready",
                            "targeted_refresh": True,
                            "targeted_refresh_request_id": "decision_static:m1:001",
                            "observation_capture_step": 43,
                            "source": "mllm_attribute_inference",
                            "coarse_state": "closed",
                        }
                    ],
                }
            )
        )
    )
    assert executor.machine.state == "INTERACTING"
    assert [command["kind"] for command in dispatched] == ["interact"]
    assert executor._interaction_observation_requests == {}


def test_two_stage_container_staging_holds_until_its_matching_m1_request_resolves(
    executor_module, monkeypatch
) -> None:
    """Residual DWA motion cannot move an outer M1 capture after it is armed."""

    candidate = {
        "decision_id": "decision-staging-hold",
        "candidate_id": "interaction:container-staging-hold:open",
        "behavior_type": "INTERACT",
        "target_id": "container-staging-hold",
        "metadata": {
            "requires_approach": False,
            "observation_required": True,
            "container_pre_action_observation": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
        },
        "interaction_command": {
            "node_id": "container-staging-hold",
            "object_id": "container-staging-hold",
            "action": "open",
            "expected_state": "open",
        },
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = dict(candidate)
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.machine.start(candidate, now=0.0)
    assert executor.machine.state == executor_module.STATE_WAITING_FOR_INTERACTION_OBSERVATION
    executor._interaction_observation_requests = {
        "decision-staging-hold": {"request_id": "decision-staging-hold:m1:001"}
    }
    executor.interaction_observation_poll_interval_s = 0.01
    cancelled = []
    executor.move_base = SimpleNamespace(cancel_goal=lambda: cancelled.append(True))
    initial_stops = []
    executor.cmd_vel_pub = SimpleNamespace(
        publish=lambda message: initial_stops.append(message)
    )
    started = []

    class _Thread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.args, self.daemon))

    monkeypatch.setattr(executor_module.threading, "Thread", _Thread)

    assert executor._begin_container_staging_observation_hold(
        "decision-staging-hold", "decision-staging-hold:m1:001"
    )
    assert cancelled == [True]
    assert len(initial_stops) == 1
    assert len(started) == 1

    hold_target, hold_args, daemon = started[0]
    assert daemon is True

    # A superseding request must end the old worker immediately; it cannot
    # keep publishing a stale zero/hold after the next M1 request is armed.
    executor._interaction_observation_requests["decision-staging-hold"] = {
        "request_id": "decision-staging-hold:m1:002"
    }
    stale_stops = []
    executor._publish_container_staging_hold_stop = lambda: stale_stops.append(True)
    hold_target(*hold_args)
    assert stale_stops == []

    # While this exact request remains pending, it repeatedly overrides any
    # late DWA command. Removing the matching request models the response path
    # consuming it, after which the worker terminates without another motion.
    executor._interaction_observation_requests["decision-staging-hold"] = {
        "request_id": "decision-staging-hold:m1:002"
    }
    active_stops = []

    def hold_stop() -> None:
        active_stops.append(True)
        if len(active_stops) == 2:
            executor._interaction_observation_requests.pop("decision-staging-hold")

    executor._publish_container_staging_hold_stop = hold_stop
    monkeypatch.setattr(executor_module.time, "sleep", lambda _seconds: None)
    executor._run_container_staging_observation_hold(
        "decision-staging-hold", "decision-staging-hold:m1:002"
    )
    assert active_stops == [True, True]


@pytest.mark.parametrize("confirmation_count", [None, 2])
def test_container_direct_front_confirmation_count_is_configurable(
    executor_module,
    confirmation_count: int | None,
) -> None:
    selection = _portal_selection()
    selection.update(
        {
            "target_id": "container_static",
            "target_name": "fridge_static",
            "candidate_id": "interaction:container_static:open",
            "metadata": {
                "node_type": "container",
                "requires_approach": False,
                "observation_required": True,
                "reobserve": True,
                "container_pre_action_observation": True,
            },
            "interaction_command": {
                "node_id": "container_static",
                "object_id": "fridge_static",
                "node_type": "container",
                "action": "open",
                "expected_state": "open",
            },
        }
    )
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = selection
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    commands = executor.machine.start(selection, now=0.0)
    assert executor.machine.state == "WAITING_FOR_INTERACTION_OBSERVATION"
    executor.latest_graph = {"capture_step": 42, "episode_id": "episode_1"}
    executor._latest_rgb_step_seq = 42
    executor.interaction_observation_sequence = 0
    executor._interaction_observation_requests = {}
    executor._latest_attribute_updates = {}
    if confirmation_count is not None:
        executor.container_pre_action_confirmation_count = confirmation_count
    executor.container_pre_action_require_direct_front = True
    published = []
    executor.attribute_refresh_request_pub = SimpleNamespace(
        publish=lambda message: published.append(json.loads(message.data))
    )
    executor._publish_interaction_observation_request(commands[0])
    dispatched = []

    assert published[-1]["expected_node_type"] == "container"

    def dispatch(next_commands):
        dispatched.extend(next_commands)
        for command in next_commands:
            if command.get("kind") == "request_interaction_observation":
                executor._publish_interaction_observation_request(command)

    executor._dispatch = dispatch
    first_request = published[-1]["request_id"]
    first_ready = {
        "object_id": "fridge_static",
        "attribute_status": "ready",
        "targeted_refresh": True,
        "targeted_refresh_request_id": first_request,
        "observation_capture_step": 43,
        "source": "mllm_attribute_inference",
        "coarse_state": "closed",
        "view_state": "front",
        "front_surface_visible": True,
        "approach_ready": True,
        "needs_reobserve": False,
        "observed_bbox_2d": [200, 100, 400, 500],
    }
    executor._attribute_update_callback(
        SimpleNamespace(data=json.dumps({"episode_id": "episode_1", "updates": [first_ready]}))
    )

    if confirmation_count is None:
        # The executor fallback is the production default: one strict fresh
        # front view at the safe outer pose advances to action (or the paired
        # inner pose for a two-stage candidate), not a second M1 vote.
        assert executor.machine.state == "INTERACTING"
        assert [command["kind"] for command in dispatched] == ["interact"]
        return

    assert executor.machine.state == "WAITING_FOR_INTERACTION_OBSERVATION"
    assert [command["kind"] for command in dispatched] == [
        "request_interaction_observation"
    ]
    assert published[-1]["request_id"] == "decision_static:m1:002"
    assert executor._interaction_observation_requests["decision_static"][
        "confirmation_count"
    ] == 1

    second_ready = {
        **first_ready,
        "targeted_refresh_request_id": published[-1]["request_id"],
        "observation_capture_step": 44,
    }
    executor._attribute_update_callback(
        SimpleNamespace(data=json.dumps({"episode_id": "episode_1", "updates": [second_ready]}))
    )

    assert executor.machine.state == "INTERACTING"
    assert [command["kind"] for command in dispatched] == [
        "request_interaction_observation",
        "interact",
    ]


def test_container_m1_capture_evidence_uses_staging_pose_without_tf(
    executor_module,
) -> None:
    """Reduced executor mocks accept evidence at their selected safe staging pose."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.tf_listener = None
    executor.map_frame = "map"
    executor.container_m1_capture_pose_tolerance_m = 0.18
    executor.container_m1_capture_yaw_tolerance_rad = 0.25
    # A TF-less test double may still expose an incomplete pose helper.  The
    # capture path must not call it or mistake its absence for stale evidence.
    executor._current_pose = lambda _frame: pytest.fail("TF-less fallback called pose reader")
    candidate = {
        "metadata": {
            "frame_id": "map",
            "effective_interaction_approach_pose_xyyaw": [1.25, -0.5, 0.4],
            "container_m1_front_axis_from_capture": True,
            "container_geometry_anchor_xy": [0.25, -0.5],
        },
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [1.25, -0.5, 0.4],
        },
    }
    update = {
        "observation_capture_step": 73,
        "view_state": "front",
        "front_surface_visible": True,
        "approach_ready": True,
    }
    request = {"observation_pose_xyyaw": [1.25, -0.5, 0.4]}

    evidence, reason = executor._container_m1_capture_evidence_locked(
        candidate, update, request
    )

    assert reason == "ready"
    assert evidence is not None
    assert evidence["capture_step"] == 73
    assert evidence["capture_pose_xyyaw"] == [1.25, -0.5, 0.4]
    assert evidence["staging_pose_xyyaw"] == [1.25, -0.5, 0.4]
    assert evidence["m1_front_axis_xy"] == [1.0, 0.0]
    assert evidence["m1_front_axis_source"] == "m1_confirmed_capture_pose"
    assert evidence["pose_validation"]["valid"] is True
    assert executor._container_m1_evidence_still_at_capture_pose_locked(
        candidate, evidence
    )


def test_container_m1_capture_uses_selected_montage_view_pose_and_face(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.tf_listener = None
    executor.map_frame = "map"
    executor.container_m1_capture_pose_tolerance_m = 0.18
    executor.container_m1_capture_yaw_tolerance_rad = 0.25
    candidate = {
        "metadata": {
            "frame_id": "map",
            "effective_interaction_approach_pose_xyyaw": [1.0, 2.0, 0.3],
            "container_m1_front_axis_from_capture": True,
            "container_m1_face_selection_enabled": True,
            "container_face_axis_xy_by_staging_index": [[0.0, 1.0], [1.0, 0.0]],
        },
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [1.0, 2.0, 0.3],
        },
    }
    update = {
        "observation_capture_step": 20,
        "selected_evidence_capture_step": 10,
        "selected_evidence_observation_pose_xyyaw": [1.12, 2.04, 0.42],
        "selected_evidence_anchor_face_id": "aabb_face_pos_x",
        "selected_evidence_anchor_face_index": 1,
        "selected_evidence_anchor_face_axis_xy": [1.0, 0.0],
        "selected_view_id": "view_1",
        "view_state": "front",
        "front_surface_visible": True,
        "approach_ready": True,
    }
    request = {"observation_pose_xyyaw": [9.0, 9.0, 0.0]}

    evidence, reason = executor._container_m1_capture_evidence_locked(
        candidate, update, request
    )

    assert reason == "ready"
    assert evidence is not None
    assert evidence["capture_step"] == 10
    assert evidence["capture_pose_xyyaw"] == [1.12, 2.04, 0.42]
    assert evidence["staging_pose_xyyaw"] == [1.0, 2.0, 0.3]
    assert evidence["m1_front_axis_xy"] == [1.0, 0.0]
    assert evidence["m1_front_staging_index"] == 1
    assert evidence["m1_front_face_id"] == "aabb_face_pos_x"
    assert (
        evidence["m1_front_axis_source"]
        == "m1_selected_montage_view_aabb_cardinal_face"
    )


def test_outer_container_staging_reuses_navigation_pose_before_m1_poll(
    executor_module,
) -> None:
    """A same-transition safe staging validation must not be lost after cancel."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor._latest_step_sync_index = 91
    executor.interaction_approach_pose_poll_max_attempts = 5
    executor._current_pose = lambda _frame: pytest.fail(
        "outer staging must reuse the navigation validation before a new TF poll"
    )
    candidate = {
        "behavior_type": "INTERACT",
        "metadata": {
            "m1_observation_staging_required": True,
            "m1_safe_staging_outer_offset_m": 0.30,
            "m1_safe_staging_arrival_tolerance_m": 0.25,
        },
        "interaction_command": {
            "interaction_ready_distance_m": 0.25,
            "interaction_ready_yaw_tolerance_rad": 0.55,
        },
    }
    selected_goal = (1.0, 2.0, 0.0)
    detail = {
        "interaction_arrival_step_index": 91,
        "interaction_pose_validation": executor_module.interaction_pose_validation(
            selected_goal,
            [1.23, 2.0, 0.10],
            distance_tolerance_m=0.25,
            yaw_tolerance_rad=0.55,
        ),
    }

    arrival_sample = executor._container_safe_staging_arrival_sample(
        candidate, selected_goal, detail
    )
    ready, poll_detail = executor._poll_interaction_approach_pose(
        "decision-safe-staging",
        candidate,
        selected_goal,
        arrival_sample=arrival_sample,
    )

    assert ready is True
    assert poll_detail["interaction_pose_poll_count"] == 0
    assert poll_detail["interaction_pose_poll_used_navigation_arrival"] is True
    assert poll_detail["interaction_pose_validation"]["sample_source"] == (
        "navigation_arrival_pose"
    )
    assert poll_detail["interaction_pose_validation"]["step_index"] == 91

    # A successful approach still enters the regular targeted-M1 barrier; the
    # reused pose sample never turns into a physical interaction command.
    candidate["metadata"].update(
        {
            "requires_approach": True,
            "observation_required": True,
            "container_pre_action_observation": True,
        }
    )
    machine = executor_module.BehaviorExecutionStateMachine()
    machine.start(candidate, now=0.0)
    commands = machine.on_navigation_result(True, detail=detail, now=1.0)
    assert machine.state == "WAITING_FOR_INTERACTION_OBSERVATION"
    assert [command["kind"] for command in commands] == [
        "request_interaction_observation"
    ]


def test_outer_container_staging_capture_uses_only_declared_safe_envelope(
    executor_module,
) -> None:
    """The wider outer-ring tolerance cannot apply to an ordinary container pose."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.tf_listener = object()
    executor.map_frame = "map"
    executor.container_m1_capture_pose_tolerance_m = 0.18
    executor.container_m1_capture_yaw_tolerance_rad = 0.25
    executor._current_pose = lambda _frame: (0.23, 0.0, 0.10)
    safe_candidate = {
        "metadata": {
            "frame_id": "map",
            "m1_observation_staging_required": True,
            "m1_safe_staging_outer_offset_m": 0.30,
            "m1_safe_staging_arrival_tolerance_m": 0.25,
            "effective_interaction_approach_pose_xyyaw": [0.0, 0.0, 0.0],
        },
        "interaction_command": {
            "interaction_ready_distance_m": 0.25,
            "interaction_approach_pose_xyyaw": [0.0, 0.0, 0.0],
        },
    }
    update = {"observation_capture_step": 73}

    evidence, reason = executor._container_m1_capture_evidence_locked(
        safe_candidate, update, {}
    )

    assert reason == "ready"
    assert evidence is not None
    assert evidence["pose_validation"]["distance_tolerance_m"] == pytest.approx(0.25)

    unsafe_candidate = {
        **safe_candidate,
        "metadata": {
            **safe_candidate["metadata"],
            "m1_safe_staging_outer_offset_m": 0.0,
        },
    }
    evidence, reason = executor._container_m1_capture_evidence_locked(
        unsafe_candidate, update, {}
    )
    assert evidence is None
    assert reason == "m1_capture_pose_mismatch"


def test_two_stage_outer_staging_tolerance_stays_separate_from_inner_bridge(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.container_m1_capture_pose_tolerance_m = 0.18
    staging_candidate = {
        "metadata": {
            "m1_observation_staging_required": True,
            "m1_safe_staging_outer_offset_m": 0.35,
            "m1_safe_staging_arrival_tolerance_m": 0.30,
            "container_two_stage_phase": "staging",
        },
        "interaction_command": {"interaction_ready_distance_m": 0.30},
    }
    inner_candidate = {
        "metadata": {
            "m1_observation_staging_required": False,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
        },
        "interaction_command": {"interaction_ready_distance_m": 0.18},
    }

    assert executor._interaction_navigation_pose_tolerance_m(
        staging_candidate
    ) == pytest.approx(0.30)
    assert executor._interaction_navigation_pose_tolerance_m(
        inner_candidate
    ) == pytest.approx(0.18)


def test_mapped_m1_capture_rejects_generic_ready_pose_before_request(
    executor_module,
) -> None:
    """A different scheduled M1 view must not reuse the last camera pose."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.tf_listener = object()
    executor.map_frame = "map"
    executor.container_m1_capture_pose_tolerance_m = 0.25
    executor.container_m1_capture_yaw_tolerance_rad = 0.35
    executor.container_m1_distinct_view_arrival_tolerance_m = 0.05
    executor.container_m1_distinct_view_arrival_yaw_tolerance_rad = 0.08
    # This pose was accepted by the former .25 m/.35 rad generic envelope, but
    # is not the requested second capture target.
    executor._current_pose = lambda _frame: (0.118, 0.0, 0.172)
    candidate = {
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "m1_capture",
            "container_two_stage_capture_pose_xyyaw": [0.0, 0.0, 0.0],
            # The first direct M1 capture uses the same exact contract as a
            # later viewpoint; this marker is audit-only.
            "container_m1_capture_requires_distinct_view_arrival": False,
            "m1_observation_staging_required": True,
        },
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [0.0, 0.0, 0.0],
            "container_m1_capture_ready_distance_m": 0.25,
            "interaction_ready_yaw_tolerance_rad": 0.35,
        },
    }

    assert executor._interaction_navigation_pose_tolerance_m(candidate) == pytest.approx(
        0.05
    )
    assert executor._interaction_navigation_yaw_tolerance_rad(candidate) == pytest.approx(
        0.35
    )
    evidence, reason = executor._container_m1_capture_evidence_locked(
        candidate,
        {"observation_capture_step": 101},
        {"observation_pose_xyyaw": [0.0, 0.0, 0.0]},
    )
    assert evidence is None
    assert reason == "m1_capture_pose_mismatch"

    # The marker requires the direct-capture metadata produced by the new
    # mapping. Old two-stage recordings therefore retain their legacy envelope.
    legacy = {
        **candidate,
        "metadata": {
            **candidate["metadata"],
            "container_two_stage_capture_pose_xyyaw": [],
        },
    }
    assert executor._interaction_navigation_pose_tolerance_m(legacy) == pytest.approx(
        0.25
    )
    assert executor._interaction_navigation_yaw_tolerance_rad(legacy) == pytest.approx(
        0.35
    )


def test_direct_m1_capture_dwa_profile_tightens_and_token_safely_restores(
    executor_module,
) -> None:
    """Only the direct capture lease may change and later restore live DWA tolerances."""

    class _DynamicClient:
        def __init__(self) -> None:
            self.cached_configuration = {
                "xy_goal_tolerance": 0.25,
                "yaw_goal_tolerance": 0.20,
            }
            self.configuration = dict(self.cached_configuration)
            self.updates: list[dict] = []
            self.read_count = 0

        def get_configuration(self, timeout=None) -> dict:
            del timeout
            self.read_count += 1
            # DynamicReconfigure Client can receive its subscriber update after
            # the synchronous service reply.  Keep this cache deliberately
            # stale to prove activation validates that reply directly.
            return dict(self.cached_configuration)

        def update_configuration(self, update: dict) -> dict:
            self.updates.append(dict(update))
            self.configuration.update(update)
            return dict(self.configuration)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._container_m1_capture_dwa_profile_lock = threading.RLock()
    executor._container_m1_capture_dwa_profile_active = None
    executor._container_m1_capture_dwa_profile_sequence = 0
    executor.container_m1_capture_dwa_profile_enabled = True
    executor.container_m1_capture_dwa_reconfigure_timeout_s = 1.0
    executor.container_m1_distinct_view_arrival_tolerance_m = 0.05
    executor.container_m1_distinct_view_arrival_yaw_tolerance_rad = 0.08
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    client = _DynamicClient()
    executor._container_m1_capture_dwa_reconfigure_client = client
    candidate = {
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": "m1_capture",
            "container_two_stage_capture_pose_xyyaw": [1.0, 2.0, 0.0],
        }
    }

    ready, detail = executor._activate_container_m1_capture_dwa_profile(
        "decision-capture",
        17,
        candidate,
        # XY tightens to the direct capture contract; the existing 0.20-rad DWA
        # yaw is already stricter than this configured capture envelope.
        direct_distance_tolerance_m=0.25,
        direct_yaw_tolerance_rad=0.35,
    )

    assert ready is True
    assert detail["default_xy_goal_tolerance"] == pytest.approx(0.25)
    assert detail["default_yaw_goal_tolerance"] == pytest.approx(0.20)
    assert client.read_count == 1
    assert client.updates == [
        {"xy_goal_tolerance": 0.05, "yaw_goal_tolerance": 0.20}
    ]
    assert client.configuration == {
        "xy_goal_tolerance": 0.05,
        "yaw_goal_tolerance": 0.20,
    }

    # A stale worker cannot restore a profile it no longer owns.
    stale_release = executor._release_container_m1_capture_dwa_profile(
        "decision-capture", 16
    )
    assert stale_release["released"] is False
    assert stale_release["reason"] == "not_profile_owner"
    assert len(client.updates) == 1

    release = executor._release_container_m1_capture_dwa_profile(
        "decision-capture", 17
    )
    assert release["released"] is True
    assert release["restored"] is True
    assert client.updates[-1] == {
        "xy_goal_tolerance": 0.25,
        "yaw_goal_tolerance": 0.20,
    }
    assert client.configuration == {
        "xy_goal_tolerance": 0.25,
        "yaw_goal_tolerance": 0.20,
    }


def test_interaction_dwa_profile_applies_explicit_pair_even_when_wider(
    executor_module,
) -> None:
    """Generated INTERACT goals share one exact executor/DWA arrival contract."""

    class _DynamicClient:
        def __init__(self) -> None:
            self.configuration = {
                "xy_goal_tolerance": 0.25,
                "yaw_goal_tolerance": 0.20,
            }
            self.updates: list[dict] = []

        def get_configuration(self, timeout=None) -> dict:
            del timeout
            return dict(self.configuration)

        def update_configuration(self, update: dict) -> dict:
            self.updates.append(dict(update))
            self.configuration.update(update)
            return dict(self.configuration)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._container_m1_capture_dwa_profile_lock = threading.RLock()
    executor._container_m1_capture_dwa_profile_active = None
    executor._container_m1_capture_dwa_profile_sequence = 0
    executor.container_m1_capture_dwa_profile_enabled = True
    executor.container_m1_capture_dwa_reconfigure_timeout_s = 1.0
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    client = _DynamicClient()
    executor._container_m1_capture_dwa_reconfigure_client = client

    candidate = {
        "behavior_type": "INTERACT",
        "metadata": {
            "interaction_command": {
                "navigation_goal_position_tolerance_m": 0.30,
                "navigation_goal_yaw_tolerance_rad": 0.40,
            }
        },
    }
    ready, detail = executor._activate_container_m1_capture_dwa_profile(
        "decision-portal",
        9,
        candidate,
        direct_distance_tolerance_m=0.30,
        direct_yaw_tolerance_rad=0.40,
    )

    assert ready is True
    assert detail["profile"] == "interaction_goal_tolerance_contract"
    assert client.updates == [
        {"xy_goal_tolerance": 0.30, "yaw_goal_tolerance": 0.40}
    ]
    assert detail["xy_goal_tolerance"] == pytest.approx(0.30)
    assert detail["yaw_goal_tolerance"] == pytest.approx(0.40)

    restored = executor._release_container_m1_capture_dwa_profile(
        "decision-portal", 9
    )
    assert restored["released"] is True
    assert client.updates[-1] == {
        "xy_goal_tolerance": 0.25,
        "yaw_goal_tolerance": 0.20,
    }


def test_non_capture_successor_cancels_and_confirms_capture_before_restoring_profile(
    executor_module, monkeypatch
) -> None:
    """A successor restores defaults only after the old capture goal is quiescent."""

    class _DynamicClient:
        def __init__(self) -> None:
            self.configuration = {
                "xy_goal_tolerance": 0.05,
                "yaw_goal_tolerance": 0.08,
            }
            self.updates: list[dict] = []

        def update_configuration(self, update: dict) -> dict:
            self.updates.append(dict(update))
            self.configuration.update(update)
            return dict(self.configuration)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._container_m1_capture_dwa_profile_lock = threading.RLock()
    executor._navigation_run_is_active = lambda decision_id, token: (
        (decision_id, token) == ("decision-successor", 22)
    )
    client = _DynamicClient()
    executor._container_m1_capture_dwa_profile_active = {
        "decision_id": "decision-successor",
        "navigation_run_token": 21,
        "client": client,
        "restore_configuration": {
            "xy_goal_tolerance": 0.25,
            "yaw_goal_tolerance": 0.20,
        },
        "restore_required": True,
        "detail": {},
    }

    class _MoveBase:
        def __init__(self) -> None:
            self.state = 1  # ACTIVE
            self.cancel_count = 0
            self.waits: list[float] = []

        def get_state(self) -> int:
            return self.state

        def cancel_goal(self) -> None:
            self.cancel_count += 1
            self.state = 2  # terminal for this isolated action-client stub

        def wait_for_result(self, timeout) -> None:
            self.waits.append(timeout)

    move_base = _MoveBase()
    executor.move_base = move_base
    executor.final_align_cancel_wait_s = 0.5
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)

    restored, detail = (
        executor._restore_container_m1_capture_dwa_profile_before_non_capture_dispatch(
            "decision-successor", 22
        )
    )

    assert restored is True
    assert detail["restored_before_non_capture_dispatch"] is True
    assert client.updates == [
        {"xy_goal_tolerance": 0.25, "yaw_goal_tolerance": 0.20}
    ]
    assert executor._container_m1_capture_dwa_profile_active is None
    assert move_base.cancel_count == 1
    assert move_base.waits == [0.5]
    assert detail["capture_goal_quiescence"] == {
        "state_before": 1,
        "state_after": 2,
        "cancel_issued": True,
        "server_status_confirmed": False,
        "status_receipts_observed": 0,
        "owned_goal_id": "",
        "goal_generation": 0,
    }


def test_capture_successor_waits_for_server_preempted_after_client_done(
    executor_module,
) -> None:
    """A local DONE state cannot outrun move_base's PREEMPTING status."""

    class _MoveBase:
        def get_state(self) -> int:
            return 2  # locally terminal in this isolated stub

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.move_base = _MoveBase()
    executor._move_base_status_condition = threading.Condition()
    executor._move_base_status_received_count = 4
    executor._move_base_active_goal_statuses = (("capture-goal", 6),)
    executor._move_base_owned_goal_id = "capture-goal"
    executor._move_base_goal_generation = 4
    executor.container_m1_capture_successor_quiescence_timeout_s = 0.5

    def _publish_server_terminal() -> None:
        time.sleep(0.01)
        with executor._move_base_status_condition:
            executor._move_base_status_received_count += 1
            executor._move_base_active_goal_statuses = (("capture-goal", 6),)
            executor._move_base_status_condition.notify_all()
        with executor._move_base_status_condition:
            executor._move_base_status_received_count += 1
            executor._move_base_active_goal_statuses = ()
            executor._move_base_status_condition.notify_all()

    publisher = threading.Thread(target=_publish_server_terminal)
    publisher.start()
    inactive, detail = (
        executor._confirm_container_m1_capture_goal_inactive_before_profile_restore()
    )
    publisher.join(timeout=1.0)

    assert inactive is True
    assert detail["state_before"] == 2
    assert detail["server_status_confirmed"] is True
    assert detail["status_receipts_observed"] >= 1


def test_terminal_client_with_empty_server_activity_is_immediately_quiescent(
    executor_module,
) -> None:
    class _MoveBase:
        def get_state(self) -> int:
            return 2  # ROS GoalStatus.PREEMPTED

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.move_base = _MoveBase()
    executor._move_base_status_condition = threading.Condition()
    executor._move_base_status_received_count = 9
    executor._move_base_active_goal_statuses = ()
    executor._move_base_owned_goal_id = "latest-goal"
    executor._move_base_goal_generation = 3
    executor.move_base_successor_quiescence_timeout_s = 0.5

    quiescent, detail = executor._confirm_move_base_goal_quiescent_before_successor()

    assert quiescent is True
    assert detail["status_receipts_observed"] == 0
    assert detail["owned_goal_id"] == "latest-goal"
    assert executor._move_base_owned_goal_id == ""


def test_terminal_client_ignores_historical_recalling_from_other_goal(
    executor_module,
) -> None:
    class _MoveBase:
        def get_state(self) -> int:
            return executor_module.GoalStatus.SUCCEEDED

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.move_base = _MoveBase()
    executor._move_base_status_condition = threading.Condition()
    executor._move_base_status_received_count = 12
    executor._move_base_active_goal_statuses = (("historical-goal", 7),)
    executor._move_base_owned_goal_id = "latest-goal"
    executor._move_base_goal_generation = 5
    executor.move_base_successor_quiescence_timeout_s = 0.5

    quiescent, detail = executor._confirm_move_base_goal_quiescent_before_successor()

    assert quiescent is True
    assert detail["status_receipts_observed"] == 0
    assert detail["owned_goal_id"] == "latest-goal"


def test_direct_capture_profile_switch_confirms_old_goal_is_inactive(
    executor_module, monkeypatch
) -> None:
    """A next M1 capture cannot replace global DWA tolerances under an old goal."""

    class _DynamicClient:
        def __init__(self) -> None:
            self.configuration = {
                "xy_goal_tolerance": 0.05,
                "yaw_goal_tolerance": 0.08,
            }
            self.updates: list[dict] = []

        def get_configuration(self, timeout=None) -> dict:
            del timeout
            return dict(self.configuration)

        def update_configuration(self, update: dict) -> dict:
            self.updates.append(dict(update))
            self.configuration.update(update)
            return dict(self.configuration)

    class _MoveBase:
        def __init__(self) -> None:
            self.state = 1
            self.cancel_count = 0
            self.waits: list[float] = []

        def get_state(self) -> int:
            return self.state

        def cancel_goal(self) -> None:
            self.cancel_count += 1
            self.state = 2

        def wait_for_result(self, timeout) -> None:
            self.waits.append(timeout)

    client = _DynamicClient()
    move_base = _MoveBase()
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._container_m1_capture_dwa_profile_lock = threading.RLock()
    executor._container_m1_capture_dwa_profile_active = {
        "decision_id": "decision-capture",
        "navigation_run_token": 10,
        "client": client,
        "restore_configuration": {
            "xy_goal_tolerance": 0.25,
            "yaw_goal_tolerance": 0.20,
        },
        "restore_required": True,
        "detail": {},
    }
    executor._container_m1_capture_dwa_profile_sequence = 0
    executor._container_m1_capture_dwa_reconfigure_client = client
    executor.container_m1_capture_dwa_profile_enabled = True
    executor.container_m1_capture_dwa_reconfigure_timeout_s = 1.0
    executor.container_m1_distinct_view_arrival_tolerance_m = 0.05
    executor.container_m1_distinct_view_arrival_yaw_tolerance_rad = 0.08
    executor.final_align_cancel_wait_s = 0.5
    executor.move_base = move_base
    executor._navigation_run_is_active = lambda decision_id, token: (
        (decision_id, token) == ("decision-capture", 11)
    )
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    candidate = {
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": "m1_capture",
            "container_two_stage_capture_pose_xyyaw": [1.0, 0.0, 0.0],
        }
    }

    ready, detail = executor._activate_container_m1_capture_dwa_profile(
        "decision-capture",
        11,
        candidate,
        direct_distance_tolerance_m=0.05,
        direct_yaw_tolerance_rad=0.08,
    )

    assert ready is True
    assert detail["active"] is True
    assert move_base.cancel_count == 1
    assert move_base.waits == [0.5]
    assert client.updates == [
        {"xy_goal_tolerance": 0.25, "yaw_goal_tolerance": 0.20},
        {"xy_goal_tolerance": 0.05, "yaw_goal_tolerance": 0.08},
    ]
    assert executor._container_m1_capture_dwa_profile_active["navigation_run_token"] == 11


def test_capture_profile_successor_fails_closed_when_old_goal_stays_active(
    executor_module, monkeypatch
) -> None:
    """Never restore broad DWA tolerances if capture cancellation is unconfirmed."""

    class _DynamicClient:
        def __init__(self) -> None:
            self.updates: list[dict] = []

        def update_configuration(self, update: dict) -> dict:
            self.updates.append(dict(update))
            return dict(update)

    class _MoveBase:
        def __init__(self) -> None:
            self.cancel_count = 0

        def get_state(self) -> int:
            return 1  # remains ACTIVE even after cancel

        def cancel_goal(self) -> None:
            self.cancel_count += 1

        def wait_for_result(self, _timeout) -> None:
            return None

    client = _DynamicClient()
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._container_m1_capture_dwa_profile_lock = threading.RLock()
    executor._container_m1_capture_dwa_profile_active = {
        "decision_id": "decision-successor",
        "navigation_run_token": 30,
        "client": client,
        "restore_configuration": {
            "xy_goal_tolerance": 0.25,
            "yaw_goal_tolerance": 0.20,
        },
        "restore_required": True,
        "detail": {},
    }
    executor._navigation_run_is_active = lambda decision_id, token: (
        (decision_id, token) == ("decision-successor", 31)
    )
    executor.move_base = _MoveBase()
    executor.final_align_cancel_wait_s = 0.5
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)

    restored, detail = (
        executor._restore_container_m1_capture_dwa_profile_before_non_capture_dispatch(
            "decision-successor", 31
        )
    )

    assert restored is False
    assert detail["reason"] == (
        "container_m1_capture_dwa_profile_goal_still_active_before_"
        "successor_dispatch"
    )
    assert executor.move_base.cancel_count == 1
    assert client.updates == []
    assert executor._container_m1_capture_dwa_profile_active is not None


@pytest.mark.parametrize("phase", ["staging", "physical_action"])
def test_dwa_capture_profile_excludes_non_capture_container_phases(
    executor_module, phase
) -> None:
    """Outer navigation and physical action must never mutate DWA tolerance."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._container_m1_capture_dwa_profile_lock = threading.RLock()
    executor._container_m1_capture_dwa_profile_active = None
    executor.container_m1_capture_dwa_profile_enabled = True

    class _UnexpectedClient:
        def get_configuration(self, *args, **kwargs):
            raise AssertionError("non-capture phase must not read DWA configuration")

    executor._container_m1_capture_dwa_reconfigure_client = _UnexpectedClient()
    candidate = {
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": phase,
        }
    }

    ready, detail = executor._activate_container_m1_capture_dwa_profile(
        "decision-non-capture",
        3,
        candidate,
        direct_distance_tolerance_m=0.05,
        direct_yaw_tolerance_rad=0.08,
    )

    assert ready is True
    assert detail == {"active": False, "reason": "not_container_m1_capture"}
    assert executor._container_m1_capture_dwa_profile_active is None


def test_direct_m1_capture_strict_dwa_profile_owns_only_its_terminal_pose(
    executor_module,
) -> None:
    """Only an active mapped capture profile may suppress generic final yaw."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    capture = {
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": "m1_capture",
            "container_two_stage_capture_pose_xyyaw": [1.0, 2.0, 0.0],
        }
    }
    physical = {
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
            "container_two_stage_capture_pose_xyyaw": [1.0, 2.0, 0.0],
        }
    }
    strict_profile = {
        "active": True,
        "xy_goal_tolerance": 0.05,
        "yaw_goal_tolerance": 0.08,
    }

    assert executor._container_m1_capture_dwa_owns_terminal_pose(
        capture,
        strict_profile,
        distance_tolerance_m=0.05,
        yaw_tolerance_rad=0.08,
    )
    assert not executor._container_m1_capture_dwa_owns_terminal_pose(
        capture,
        {**strict_profile, "active": False},
        distance_tolerance_m=0.05,
        yaw_tolerance_rad=0.08,
    )
    assert not executor._container_m1_capture_dwa_owns_terminal_pose(
        capture,
        {**strict_profile, "yaw_goal_tolerance": 0.09},
        distance_tolerance_m=0.05,
        yaw_tolerance_rad=0.08,
    )
    assert not executor._container_m1_capture_dwa_owns_terminal_pose(
        physical,
        strict_profile,
        distance_tolerance_m=0.05,
        yaw_tolerance_rad=0.08,
    )


def _direct_m1_capture_navigation_executor(
    executor_module,
    monkeypatch,
    *,
    profile_detail: dict,
    state_sequence: list[int],
    pose_reader,
    on_state_read=None,
):
    """Build a small token-current direct-capture navigation worker harness."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def __init__(self) -> None:
            self.state_reads = 0
            self.sent_goals = []
            self.cancel_count = 0

        def wait_for_server(self, _timeout) -> bool:
            return True

        def send_goal(self, goal) -> None:
            self.sent_goals.append(goal)

        def get_state(self) -> int:
            self.state_reads += 1
            if on_state_read is not None:
                on_state_read(self.state_reads)
            return int(state_sequence[min(self.state_reads - 1, len(state_sequence) - 1)])

        def get_goal_status_text(self) -> str:
            return "test-state"

        def cancel_goal(self) -> None:
            self.cancel_count += 1

    class _Watchdog:
        def __init__(self, **kwargs) -> None:
            trace["watchdog_kwargs"].append(dict(kwargs))

        def reset(self, *_args, **_kwargs) -> None:
            return None

        def observe(self, *_args, **_kwargs) -> bool:
            trace["watchdog_observe_calls"] += 1
            return False

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    monkeypatch.setattr(
        executor_module.rospy, "Time", SimpleNamespace(now=lambda: 0.0)
    )
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor_module, "NavigationProgressWatchdog", _Watchdog)

    trace = {
        "current": True,
        "generic_interaction_align_calls": [],
        "generic_terminal_align_calls": [],
        "completed": [],
        "watchdog_kwargs": [],
        "watchdog_observe_calls": 0,
        "rear_oscillation_calls": [],
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._latest_step_sync_index = 0
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.machine = SimpleNamespace(
        config=SimpleNamespace(
            interaction_navigation_timeout_s=10.0,
            navigation_timeout_s=10.0,
        )
    )
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor.container_m1_distinct_view_arrival_tolerance_m = 0.05
    executor.container_m1_distinct_view_arrival_yaw_tolerance_rad = 0.08
    # These focused tests exercise the strict 0.08-rad terminal-yaw controller;
    # runtime full-M1 configuration separately verifies the new 30-degree gate.
    executor.container_m1_capture_yaw_tolerance_rad = 0.08
    executor.navigation_stagnation_timeout_s = 12.0
    executor.navigation_stagnation_distance_m = 0.10
    executor.navigation_stagnation_yaw_rad = 0.15
    executor.navigation_stagnation_goal_distance_reduction_m = 0.02
    executor.final_align_enabled = True
    executor.final_align_max_distance_m = 0.12
    executor.final_align_yaw_tolerance_rad = 0.15
    executor.final_align_trigger_delay_s = 0.0
    executor.interaction_final_align_enabled = True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    executor._navigation_is_current = lambda _decision_id: trace["current"]
    executor._current_pose = pose_reader
    executor._preflight_navigation_plan = lambda *_args: (
        True,
        (1.0, 0.0),
        "reachable",
    )
    executor._prerotate_for_rear_goal = lambda *_args, **_kwargs: True
    executor._set_effective_interaction_approach = lambda *_args, **_kwargs: None
    executor._activate_container_m1_capture_dwa_profile = (
        lambda *_args, **_kwargs: (True, dict(profile_detail))
    )
    executor._start_rear_dwa_monitor = lambda *_args, **_kwargs: None
    executor._has_fresh_local_plan = lambda *_args, **_kwargs: False
    executor._rear_dwa_oscillation_detail = (
        lambda *_args, **_kwargs: trace["rear_oscillation_calls"].append(True)
        or None
    )
    executor._final_align_interaction_goal = (
        lambda *_args, **_kwargs: trace["generic_interaction_align_calls"].append(
            True
        )
        or True
    )
    executor._final_align_goal = (
        lambda *_args, **_kwargs: trace["generic_terminal_align_calls"].append(True)
        or True
    )
    executor._complete_interaction_approach_navigation = (
        lambda *_args, **kwargs: trace["completed"].append(dict(kwargs))
    )
    candidate = {
        "candidate_id": "interaction:container:direct-capture",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "m1_capture",
            "container_two_stage_capture_pose_xyyaw": [1.0, 0.0, 0.0],
        },
        "interaction_command": {
            "container_m1_capture_ready_distance_m": 0.05,
            "interaction_ready_yaw_tolerance_rad": 0.08,
        },
    }
    return executor, candidate, trace


def test_strict_direct_m1_capture_leaves_position_ready_final_yaw_to_dwa(
    executor_module, monkeypatch
) -> None:
    """A strict leased capture must not cancel DWA for generic final alignment."""

    trace = {"current": True}

    def stop_after_active_state(state_reads: int) -> None:
        if state_reads >= 2:
            trace["current"] = False

    executor, candidate, worker_trace = _direct_m1_capture_navigation_executor(
        executor_module,
        monkeypatch,
        profile_detail={
            "active": True,
            "xy_goal_tolerance": 0.05,
            "yaw_goal_tolerance": 0.08,
        },
        # The import stub uses GoalStatus.PREEMPTED=1, so use its non-terminal
        # pending state to keep this worker in the active-loop test path.
        state_sequence=[0],
        pose_reader=lambda _frame_id: (1.0, 0.0, 0.20),
        on_state_read=stop_after_active_state,
    )
    # Share the worker's current token with the state callback only after its
    # harness has been built.
    trace = worker_trace

    executor._run_navigation_impl(
        "decision-strict-active",
        candidate,
        navigation_run_token=7,
    )

    assert executor.move_base.cancel_count == 0
    assert worker_trace["generic_interaction_align_calls"] == []
    assert worker_trace["generic_terminal_align_calls"] == []
    assert worker_trace["watchdog_kwargs"][0]["allow_yaw_progress"] is False
    assert worker_trace["watchdog_observe_calls"] == 0
    assert worker_trace["rear_oscillation_calls"] == []


def test_strict_direct_m1_capture_terminal_yaw_settle_timeout_retries(
    executor_module, monkeypatch
) -> None:
    """The strict-capture settle budget advances only with simulator steps."""

    holder = {}

    def advance_public_step(_state_reads: int) -> None:
        holder["executor"]._latest_step_sync_index += 1

    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module,
        monkeypatch,
        profile_detail={
            "active": True,
            "xy_goal_tolerance": 0.05,
            "yaw_goal_tolerance": 0.08,
        },
        state_sequence=[0],
        pose_reader=lambda _frame_id: (1.0, 0.0, 0.20),
        on_state_read=advance_public_step,
    )
    holder["executor"] = executor
    executor.container_m1_capture_dwa_terminal_yaw_settle_max_task_steps = 5
    retries = []
    executor._retry_interaction_approach = (
        lambda *_args: retries.append(dict(_args[-1])) or True
    )

    executor._run_navigation_impl(
        "decision-strict-settle-timeout",
        candidate,
        navigation_run_token=71,
    )

    assert executor.move_base.cancel_count == 1
    assert len(retries) == 1
    yaw_detail = retries[0]["container_m1_capture_dwa_terminal_yaw"]
    assert yaw_detail["timed_out"] is True
    assert yaw_detail["elapsed_task_steps"] == 5
    assert yaw_detail["settle_max_task_steps"] == 5
    assert yaw_detail["best_yaw_error_rad"] == pytest.approx(0.20)
    assert trace["generic_interaction_align_calls"] == []
    assert trace["watchdog_observe_calls"] == 0


def test_position_latch_survives_drift_until_yaw_ready(executor_module, monkeypatch):
    holder = {}

    def advance(reads):
        holder["executor"]._latest_step_sync_index = reads

    def pose(_frame):
        return (1.30, 0.0, 0.04) if holder["executor"]._latest_step_sync_index >= 3 else (1.0, 0.0, 0.20)

    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module, monkeypatch,
        profile_detail={"active": True, "xy_goal_tolerance": 0.05, "yaw_goal_tolerance": 0.08},
        state_sequence=[0], pose_reader=pose, on_state_read=advance,
    )
    holder["executor"] = executor
    executor._run_navigation_impl("latched-drift", candidate, navigation_run_token=1)
    detail = trace["completed"][0]["detail"]["interaction_pose_validation"]
    assert detail["valid"]
    assert detail["position_tolerance_latched"]
    assert detail["position_error_m"] == pytest.approx(0.30)
    assert trace["rear_oscillation_calls"] == []


def test_position_ready_goal_skips_rear_xy_turn(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._current_pose = lambda _frame: (1.0, 0.068, 1.8)
    assert executor._prerotate_for_rear_goal("near", "map", 1.0, 0.0, heading_target_xy=None, position_tolerance_m=0.15)
    assert executor._last_rear_goal_recovery_detail["reason"] == "rear_goal_position_already_ready"


def test_same_anchor_resume_checks_only_original_goal_and_reuses_attempt(executor_module, monkeypatch):
    moving = {"arrived": False}
    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module, monkeypatch,
        profile_detail={"active": True, "xy_goal_tolerance": .05, "yaw_goal_tolerance": .08},
        state_sequence=[0], pose_reader=lambda _frame: (1., 0., .04) if moving["arrived"] else (0., 0., 0.),
        on_state_read=lambda _reads: moving.update(arrived=True),
    )
    candidate["metadata"]["goal_xyyaw_candidates"] = [[1., 0., 0.], [2., 0., 0.]]
    preflight_goals = []
    executor._preflight_navigation_plan = lambda _frame, x, y, yaw: (preflight_goals.append([x, y, yaw]) or True, (x, y), "reachable")
    history = [{"index": 0, "navigation_attempt": 1, "resume_same_anchor": True}]
    effective_attempts = []
    executor._set_effective_interaction_approach = lambda *args: effective_attempts.extend(args[-1])
    executor._run_navigation_impl("same-anchor", candidate, interaction_approach_attempts=history, navigation_run_token=1)
    assert preflight_goals == [[1., 0., 0.]]
    assert trace["completed"]
    attempts = effective_attempts
    assert len(attempts) == 1
    assert attempts[0]["navigation_attempt"] == 1
    assert attempts[0]["clearance_resend_count"] == 1
    assert "resume_same_anchor" not in attempts[0]


def test_same_anchor_resume_seeds_missing_attempt_history(executor_module):
    history = []
    selected_attempt = {
        "index": 2,
        "goal_xyyaw": [1.0, 2.0, 0.3],
        "reachable": True,
        "preflight_reason": "reachable",
    }
    clearance = {"reason": "container_anchor_center_blocked", "confirmed": True}

    executor_module.SemanticBehaviorExecutor._mark_same_anchor_resume(
        history,
        selected_attempt=selected_attempt,
        selected_goal_option_index=2,
        selected_goal=(1.0, 2.0, 0.3),
        updates={
            "clearance_recheck_resend": True,
            "clearance_confirmation": clearance,
        },
    )

    assert history == [
        {
            **selected_attempt,
            "navigation_attempt": 1,
            "resume_same_anchor": True,
            "clearance_recheck_resend": True,
            "clearance_confirmation": clearance,
        }
    ]


def test_unchanged_costmap_is_inconclusive_not_a_blocked_anchor(executor_module, monkeypatch):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._latest_step_sync_index = 10
    executor._navigation_is_current = lambda _decision: True
    executor.move_base = SimpleNamespace(cancel_goal=lambda: None)
    initial = {"reason": "container_anchor_center_blocked", "costmap_received_at": 10}
    executor._container_anchor_local_clearance = lambda *_args: (False, initial)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _s: setattr(executor, "_latest_step_sync_index", executor._latest_step_sync_index+1))
    clear, detail = executor._confirm_container_anchor_blocked("d", {}, "map", (1., 0., 0.), initial)
    assert not clear and detail["clearance_confirmation_inconclusive"]
    assert executor._latest_step_sync_index == 30
    assert len(detail["clearance_confirmation_samples"]) == 1


def test_disconnected_inflated_goal_never_calls_make_plan(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.make_plan_preflight_enabled = True
    executor._current_pose = lambda _frame: (-1., 0., 0.)
    values = np.zeros((100, 100), dtype=np.int16)
    values[:, 50] = 99
    grid = executor_module.ArrivalClearanceGrid(values, .1, -5., -5., 0., "map", 100, 99)
    executor._global_clearance_snapshot = (grid, time.monotonic())
    executor.make_plan_client = lambda **_kwargs: pytest.fail("disconnected endpoint must not dispatch")
    assert executor._preflight_navigation_plan("map", 1., 0., 0.) == (False, None, "path_disconnected")


def test_frontier_recovery_scan_refuses_unsafe_rotation(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.startup_scan_angle_rad = 2*math.pi
    executor.startup_scan_angular_speed_rad_s = 1.25
    executor.startup_scan_control_dt_s = .2
    executor.startup_scan_max_control_steps = 40
    executor.startup_scan_timeout_s = 15.
    executor.startup_scan_step_sync_stall_timeout_s = 60.
    executor._startup_scan_gate = executor_module.StepCommandGate(max_pair_age_s=10.)
    executor._startup_scan_is_current = lambda _decision: True
    executor.map_frame = "map"
    executor._current_pose = lambda _frame: (0., 0., 0.)
    executor._rear_goal_rotation_command_safe = lambda: False
    executor._startup_scan_detail = lambda **kwargs: kwargs
    executor._publish_startup_scan_progress = lambda **_kwargs: None
    rotations, results = [], []
    executor._publish_rotation = rotations.append
    executor._handle_startup_scan_result = lambda _decision, success, detail: results.append((success, detail))
    executor._run_startup_scan("recovery", {"metadata": {"frontier_recovery_scan": True}})
    assert rotations == [0.]
    assert results[0][0] is False
    assert results[0][1]["reason"] == "frontier_recovery_scan_unsafe"


@pytest.mark.parametrize("recovers", [False, True])
def test_changed_anchor_stops_before_bounded_fresh_map_recheck(executor_module, monkeypatch, recovers):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._latest_step_sync_index = 10
    executor._navigation_is_current = lambda _decision: True
    cancelled = []
    executor.move_base = SimpleNamespace(cancel_goal=lambda: cancelled.append(True))
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)

    def sleep(_seconds):
        assert cancelled
        executor._latest_step_sync_index += 1

    def clearance(*_args):
        clear = recovers and executor._latest_step_sync_index >= 12
        return clear, {"reason": "clear" if clear else "container_anchor_center_blocked", "costmap_received_at": executor._latest_step_sync_index}

    monkeypatch.setattr(executor_module.time, "sleep", sleep)
    executor._container_anchor_local_clearance = clearance
    clear, detail = executor._confirm_container_anchor_blocked("d", {}, "map", (1, 0, 0), {"reason": "container_anchor_center_blocked", "costmap_received_at": 10})
    assert clear is recovers
    assert cancelled == [True]
    assert executor._latest_step_sync_index <= 13
    assert len(detail["clearance_confirmation_samples"]) >= 3


def test_non_strict_direct_m1_capture_keeps_active_dwa_terminal_control(
    executor_module, monkeypatch
) -> None:
    """A live move_base goal is not cancelled into a second cmd_vel turn."""

    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module,
        monkeypatch,
        profile_detail={
            "active": True,
            "xy_goal_tolerance": 0.05,
            "yaw_goal_tolerance": 0.09,
        },
        state_sequence=[0],
        pose_reader=lambda _frame_id: (1.0, 0.0, 0.20),
    )

    executor._run_navigation_impl(
        "decision-nonstrict-active",
        candidate,
        navigation_run_token=8,
    )

    assert trace["generic_interaction_align_calls"] == []


def test_strict_direct_m1_capture_terminal_success_skips_generic_final_align(
    executor_module, monkeypatch
) -> None:
    """SUCCEEDED still reaches M1 only through the strict terminal TF poll."""

    poses = iter(
        [
            (1.0, 0.0, 0.20),  # direct precheck: distance-ready, yaw not ready
            (1.0, 0.0, 0.20),  # navigation start sample
            (1.0, 0.0, 0.00),  # move_base SUCCEEDED terminal TF sample
        ]
    )
    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module,
        monkeypatch,
        profile_detail={
            "active": True,
            "xy_goal_tolerance": 0.05,
            "yaw_goal_tolerance": 0.08,
        },
        state_sequence=[int(executor_module.GoalStatus.SUCCEEDED)],
        pose_reader=lambda _frame_id: next(poses),
    )

    executor._run_navigation_impl(
        "decision-strict-terminal",
        candidate,
        navigation_run_token=9,
    )

    assert trace["generic_interaction_align_calls"] == []
    assert trace["generic_terminal_align_calls"] == []
    assert len(trace["completed"]) == 1
    detail = trace["completed"][0]["detail"]
    assert detail["interaction_pose_validation_source"] == "move_base_terminal"
    assert detail["interaction_pose_validation"]["valid"] is True
    assert detail["interaction_pose_validation"]["yaw_tolerance_rad"] == pytest.approx(
        0.08
    )
    assert detail["container_m1_capture_dwa_terminal_yaw"]["reason"] == (
        "strict_dwa_move_base_terminal"
    )


def test_navigation_wrapper_releases_capture_profile_on_early_exit(
    executor_module,
) -> None:
    """The worker-level finally covers cancellation, timeout, and every return path."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    released = []
    executor._register_navigation_run = lambda _decision_id, _candidate: 29
    executor._release_container_m1_capture_dwa_profile = (
        lambda decision_id, token: released.append(("profile", decision_id, token))
    )
    executor._release_navigation_run = (
        lambda decision_id, token, candidate: released.append(
            ("navigation", decision_id, token, candidate)
        )
    )

    def early_exit(*_args, **_kwargs) -> None:
        return None

    executor._run_navigation_impl = early_exit
    candidate = {"candidate_id": "direct-capture"}
    executor._run_navigation("decision-early-exit", candidate)

    assert released == [
        ("profile", "decision-early-exit", 29),
        ("navigation", "decision-early-exit", 29, candidate),
    ]


def test_direct_capture_rechecks_navigation_token_after_profile_setup(
    executor_module, monkeypatch
) -> None:
    """A profile RPC may race preemption, but the stale worker must not send its goal."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def __init__(self) -> None:
            self.sent_goals = []

        def wait_for_server(self, _timeout) -> bool:
            return True

        def send_goal(self, goal) -> None:
            self.sent_goals.append(goal)

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    monkeypatch.setattr(
        executor_module.rospy, "Time", SimpleNamespace(now=lambda: 0.0)
    )
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.machine = SimpleNamespace(
        config=SimpleNamespace(
            interaction_navigation_timeout_s=10.0,
            navigation_timeout_s=10.0,
        )
    )
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor.container_m1_distinct_view_arrival_tolerance_m = 0.05
    executor.container_m1_distinct_view_arrival_yaw_tolerance_rad = 0.08
    executor._register_navigation_run = lambda _decision_id, _candidate: 41
    executor._release_container_m1_capture_dwa_profile = lambda *_args: None
    executor._release_navigation_run = lambda *_args: None
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    current = [True]
    executor._navigation_is_current = lambda _decision_id: current[0]
    executor._current_pose = lambda _frame_id: (0.0, 0.0, 0.0)
    executor._preflight_navigation_plan = lambda *_args: (
        True,
        (1.0, 0.0),
        "reachable",
    )
    executor._prerotate_for_rear_goal = lambda *_args, **_kwargs: True
    executor._set_effective_interaction_approach = lambda *_args, **_kwargs: None
    activated = []

    def activate(*_args, **_kwargs):
        activated.append(True)
        current[0] = False
        return True, {"lease_token": 1, "active": True}

    executor._activate_container_m1_capture_dwa_profile = activate
    candidate = {
        "candidate_id": "interaction:container:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "m1_capture",
            "container_two_stage_capture_pose_xyyaw": [1.0, 0.0, 0.0],
        },
        "interaction_command": {
            "container_m1_capture_ready_distance_m": 0.50,
            "interaction_ready_yaw_tolerance_rad": 0.35,
        },
    }

    executor._run_navigation("decision-profile-race", candidate)

    assert activated == [True]
    assert executor.move_base.sent_goals == []


def test_outer_m1_capture_uses_declared_staging_yaw_contract(
    executor_module,
) -> None:
    """Outer M1 may use its navigation yaw envelope, never the inner gate."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.tf_listener = object()
    executor.map_frame = "map"
    executor.container_m1_capture_pose_tolerance_m = 0.18
    executor.container_m1_capture_yaw_tolerance_rad = 0.25
    executor._current_pose = lambda _frame: (0.0, 0.0, 0.30)
    candidate = {
        "metadata": {
            "frame_id": "map",
            "m1_observation_staging_required": True,
            "m1_safe_staging_outer_offset_m": 0.35,
            "m1_safe_staging_arrival_tolerance_m": 0.30,
            "effective_interaction_approach_pose_xyyaw": [0.0, 0.0, 0.0],
        },
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [0.0, 0.0, 0.0],
            "interaction_ready_yaw_tolerance_rad": 0.35,
        },
    }
    update = {"observation_capture_step": 93}

    evidence, reason = executor._container_m1_capture_evidence_locked(
        candidate, update, {"observation_pose_xyyaw": [0.0, 0.0, 0.0]}
    )
    assert reason == "ready"
    assert evidence is not None
    assert evidence["pose_validation"]["yaw_tolerance_rad"] == pytest.approx(0.35)

    executor._current_pose = lambda _frame: (0.0, 0.0, 0.36)
    evidence, reason = executor._container_m1_capture_evidence_locked(
        candidate, update, {"observation_pose_xyyaw": [0.0, 0.0, 0.0]}
    )
    assert evidence is None
    assert reason == "m1_capture_pose_mismatch"

    ordinary = {
        **candidate,
        "metadata": {
            **candidate["metadata"],
            "m1_observation_staging_required": False,
        },
    }
    executor._current_pose = lambda _frame: (0.0, 0.0, 0.30)
    evidence, reason = executor._container_m1_capture_evidence_locked(
        ordinary, update, {"observation_pose_xyyaw": [0.0, 0.0, 0.0]}
    )
    assert evidence is None
    assert reason == "m1_capture_pose_mismatch"


def test_two_stage_inner_drawer_bypasses_close_range_regrounding(executor_module) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.selection = {
        "behavior_type": "INTERACT",
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
        },
        "interaction_command": {
            "sequence_type": "drawer_scan",
            "open_regions": [{"center": [0.5, 0.5], "confidence": 0.9}],
        },
    }

    assert executor._needs_fresh_drawer_scan_locked() is False


def test_start_inner_corridor_dispatches_one_private_navigation_waypoint(
    executor_module, monkeypatch
) -> None:
    """A verified inner segment must not dispatch the physical bridge pose yet."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.container_inner_corridor_enabled = True
    executor.container_inner_corridor_max_segments = 4
    executor.container_inner_corridor_segment_m = 0.30
    executor.container_inner_corridor_arrival_tolerance_m = 0.10
    executor._preflight_navigation_path = lambda *_args: (
        True,
        [(0.0, 0.0), (0.15, 0.0), (0.35, 0.0), (1.0, 0.0)],
        "reachable",
    )
    executor._current_pose = lambda _frame: (0.0, 0.0, 0.0)
    physical_candidate = {
        "decision_id": "decision-corridor-start",
        "candidate_id": "interaction:fridge:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
            "goal_xyyaw_candidates": [[1.1, 0.1, 0.0], [1.1, -0.1, 0.0]],
        },
        "interaction_command": {"action": "open", "node_type": "container"},
    }
    launched = []

    class _Thread:
        def __init__(self, *, target, args, daemon):
            launched.append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(executor_module.threading, "Thread", _Thread)

    assert executor._start_container_inner_corridor(
        "decision-corridor-start",
        physical_candidate,
        goal_frame="map",
        final_goal=(1.0, 0.0, 0.0),
        final_goal_option_index=0,
        interaction_approach_attempts=[{"index": 0, "phase": "physical_action"}],
    ) is True

    assert len(launched) == 1
    _target, args, daemon = launched[0]
    assert daemon is True
    assert args[0] == "decision-corridor-start"
    corridor_candidate = args[1]
    assert args[2:] == (0, [])
    # The worker receives one short NAVIGATE target, not the physical action
    # target or its tangent alternatives.
    assert corridor_candidate["behavior_type"] == "NAVIGATE"
    assert corridor_candidate["goal_xyyaw"] == [0.35, 0.0, 0.0]
    assert executor_module.navigation_goal_options(corridor_candidate) == [
        (0.35, 0.0, 0.0)
    ]
    assert corridor_candidate["metadata"]["goal_xyyaw_candidates"] == []
    assert physical_candidate["behavior_type"] == "INTERACT"
    assert physical_candidate["metadata"]["goal_xyyaw_candidates"] == [
        [1.1, 0.1, 0.0],
        [1.1, -0.1, 0.0],
    ]

    corridor_run_id = corridor_candidate["metadata"]["container_inner_corridor_run_id"]
    assert corridor_run_id > 0
    context = executor._container_inner_corridors["decision-corridor-start"]
    assert context["corridor_run_id"] == corridor_run_id
    assert context["candidate"]["behavior_type"] == "INTERACT"
    assert context["candidate"]["goal_xyyaw"] == [1.0, 0.0, 0.0]
    assert context["waypoint_xyyaw"] == [0.35, 0.0, 0.0]


def test_container_successor_waits_for_predecessor_worker_release(
    executor_module, monkeypatch
) -> None:
    """A callback successor must not race the predecessor actionlib worker."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._active_navigation_run_tokens = {"decision-race": 7}
    launched = []

    class _Thread:
        def __init__(self, *, target, args, daemon):
            launched.append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(executor_module.threading, "Thread", _Thread)
    executor._schedule_navigation_successor(
        "decision-race",
        {"decision_id": "decision-race", "behavior_type": "NAVIGATE"},
        2,
        [{"index": 1}],
    )

    assert len(launched) == 1
    target, args, daemon = launched[0]
    assert daemon is True
    assert target == executor._run_navigation_successor_after_release
    assert args[0] == "decision-race"
    assert args[2:] == (2, [{"index": 1}], 7)


def test_container_successor_crosses_quiescence_before_final_dispatch(
    executor_module
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.move_base_successor_quiescence_timeout_s = 0.1
    active = iter([True, False])
    executor._navigation_run_is_active = lambda *_args: next(active, False)
    executor._navigation_is_current = lambda _decision_id: True
    calls = []
    executor._confirm_move_base_goal_quiescent_before_successor = lambda: (
        calls.append("quiescence") or True,
        {"state_after": 2},
    )
    executor._run_navigation = lambda *args: calls.append(("run", args))
    executor._handle_navigation_result = lambda *_args, **_kwargs: pytest.fail(
        "a quiescent successor must dispatch instead of terminalizing"
    )

    executor._run_navigation_successor_after_release(
        "decision-serialized",
        {"decision_id": "decision-serialized", "behavior_type": "INTERACT"},
        3,
        [{"index": 2}],
        9,
    )

    assert calls[0] == "quiescence"
    assert calls[1][0] == "run"
    assert calls[1][1][2:] == (3, [{"index": 2}])


def test_container_successor_quiescence_failure_returns_feedback(
    executor_module
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.move_base_successor_quiescence_timeout_s = 0.1
    executor._navigation_run_is_active = lambda *_args: False
    executor._navigation_is_current = lambda _decision_id: True
    executor._confirm_move_base_goal_quiescent_before_successor = lambda: (
        False,
        {"state_after": 1},
    )
    feedback = []
    executor._handle_navigation_result = lambda *args, **kwargs: feedback.append(
        (args, kwargs)
    )
    executor._run_navigation = lambda *_args: pytest.fail(
        "a non-quiescent move_base server must not receive the successor goal"
    )
    candidate = {
        "decision_id": "decision-not-quiescent",
        "behavior_type": "INTERACT",
    }

    executor._run_navigation_successor_after_release(
        "decision-not-quiescent", candidate, 0, [], 4
    )

    assert len(feedback) == 1
    detail = feedback[0][0][2]
    assert detail["reason"] == "move_base_successor_not_quiescent"
    assert detail["retryable"] is True
    assert feedback[0][1]["source_candidate"] == candidate


def test_shared_staging_corridor_is_one_shot_after_local_execution_failure(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.container_inner_corridor_enabled = True
    executor.container_inner_corridor_max_segments = 4
    executor._current_pose = lambda _frame: (0.0, 0.0, 0.0)
    calls = []
    executor._preflight_navigation_path = lambda *_args: (
        calls.append(_args) or (False, [], "empty_plan")
    )
    candidate = {
        "behavior_type": "INTERACT",
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
            "container_anchor_shared_pose": True,
        },
    }

    assert executor._start_container_inner_corridor(
        "decision-shared-staging",
        candidate,
        goal_frame="map",
        final_goal=(1.0, 0.0, 0.0),
        final_goal_option_index=0,
        interaction_approach_attempts=[],
    ) is False
    assert len(calls) == 1

    candidate["metadata"]["container_inner_corridor_segments_completed"] = 1
    assert executor._start_container_inner_corridor(
        "decision-shared-staging",
        candidate,
        goal_frame="map",
        final_goal=(1.0, 0.0, 0.0),
        final_goal_option_index=0,
        interaction_approach_attempts=[],
    ) is False
    assert len(calls) == 1


def test_inner_corridor_summary_is_recorder_safe_and_minimal(executor_module) -> None:
    """Execution-state snapshots expose only private corridor bookkeeping."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.selection = {"decision_id": "decision-corridor-summary"}
    executor._container_inner_corridors = {
        "decision-corridor-summary": {
            "corridor_run_id": 7,
            "segments_completed": 2,
            "waypoint_xyyaw": [0.35, 0.1, -0.2, 999.0],
            "final_goal_option_index": 3,
            # The retained candidate must never be copied into this payload.
            "candidate": {"interaction_command": {"action": "open"}},
        }
    }

    summary = executor._container_inner_corridor_summary_locked()

    assert summary == {
        "active": True,
        "run_id": 7,
        "segment_index": 2,
        "waypoint_xyyaw": [0.35, 0.1, -0.2],
        "final_goal_option_index": 3,
    }
    assert "candidate" not in summary

    executor.selection = {"decision_id": "decision-without-corridor"}
    assert executor._container_inner_corridor_summary_locked() == {"active": False}


def test_inner_corridor_completion_relaunches_canonical_physical_goal_without_bridge(
    executor_module, monkeypatch
) -> None:
    """A private corridor waypoint cannot become an interaction success."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = {"decision_id": "decision-corridor"}
    executor._container_inner_corridors = {
        "decision-corridor": {
            "corridor_run_id": 1,
            "candidate": {
                "decision_id": "decision-corridor",
                "behavior_type": "INTERACT",
                "goal_xyyaw": [2.0, 0.0, 0.0],
                "metadata": {
                    "container_two_stage_approach": True,
                    "container_two_stage_phase": "physical_action",
                },
            },
            "final_goal_option_index": 1,
            "interaction_approach_attempts": [{"index": 1}],
            "waypoint_xyyaw": [1.0, 0.0, 0.0],
        }
    }
    executor._navigation_is_current = lambda decision_id: decision_id == "decision-corridor"
    launched = []

    class _Thread:
        def __init__(self, *, target, args, daemon):
            launched.append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(executor_module.threading, "Thread", _Thread)
    executor._handle_navigation_result = lambda *_args, **_kwargs: pytest.fail(
        "a corridor waypoint must not fall through to bridge/state-machine result"
    )

    assert executor._consume_container_inner_corridor_result(
        "decision-corridor",
        True,
        {"status": "SUCCEEDED"},
        candidate={
            "metadata": {
                "container_inner_corridor_navigation": True,
                "container_inner_corridor_run_id": 1,
            }
        },
    ) is True
    assert executor._container_inner_corridors == {}
    assert len(launched) == 1
    _target, args, _daemon = launched[0]
    assert args[0] == "decision-corridor"
    assert args[2] == 1
    assert args[1]["goal_xyyaw"] == [2.0, 0.0, 0.0]
    assert args[1]["metadata"]["container_inner_corridor_segments_completed"] == 1


def test_inner_corridor_failure_retries_original_physical_option(executor_module) -> None:
    """A failed waypoint preserves the physical-option index for tangent retry."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = {"decision_id": "decision-corridor-fail"}
    candidate = {
        "decision_id": "decision-corridor-fail",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [2.0, 0.0, 0.0],
        "metadata": {
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
            "container_two_stage_staging_goal_option_index": 0,
            "container_staging_goal_xyyaw_candidates": [[3.0, 0.0, 0.0]],
            "container_action_goal_xyyaw_options_by_staging_index": [
                [[2.0, 0.0, 0.0], [2.1, 0.1, 0.0], [2.1, -0.1, 0.0]]
            ],
        },
    }
    executor._container_inner_corridors = {
        "decision-corridor-fail": {
            "corridor_run_id": 2,
            "candidate": candidate,
            "final_goal_option_index": 1,
            "interaction_approach_attempts": [{"index": 1}],
            "waypoint_xyyaw": [1.0, 0.0, 0.0],
        }
    }
    executor._navigation_is_current = lambda _decision_id: True
    retry_calls = []
    executor._retry_interaction_approach = lambda *args: retry_calls.append(args) or True
    executor._handle_navigation_result = lambda *_args, **_kwargs: pytest.fail(
        "successful bounded retry must not terminalize the physical candidate"
    )

    assert executor._consume_container_inner_corridor_result(
        "decision-corridor-fail",
        False,
        {"reason": "navigation_stagnation"},
        candidate={
            "metadata": {
                "container_inner_corridor_navigation": True,
                "container_inner_corridor_run_id": 2,
            }
        },
    ) is True
    assert len(retry_calls) == 1
    assert retry_calls[0][2] == 1
    assert retry_calls[0][4] == 3
    assert retry_calls[0][3][-1]["index"] == 1
    assert retry_calls[0][3][-1]["outcome"] == "container_inner_corridor_failed"


def test_non_corridor_result_cannot_consume_same_decision_corridor(executor_module) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._container_inner_corridors = {
        "decision-corridor-stale": {"candidate": {"behavior_type": "INTERACT"}}
    }

    assert executor._consume_container_inner_corridor_result(
        "decision-corridor-stale",
        True,
        {},
        candidate={"metadata": {}},
    ) is False
    assert "decision-corridor-stale" in executor._container_inner_corridors


def test_stale_inner_corridor_run_id_cannot_consume_replacement_context(
    executor_module,
) -> None:
    """An old private worker must not consume a newer waypoint's context."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    context = {
        "corridor_run_id": 22,
        "candidate": {"behavior_type": "INTERACT", "goal_xyyaw": [2.0, 0.0, 0.0]},
        "final_goal_option_index": 0,
        "interaction_approach_attempts": [],
        "waypoint_xyyaw": [1.0, 0.0, 0.0],
    }
    executor._container_inner_corridors = {"decision-corridor-replaced": context}
    executor._navigation_is_current = lambda _decision_id: True
    executor._retry_interaction_approach = lambda *_args: pytest.fail(
        "a stale private worker must not retry the replacement context"
    )
    executor._handle_navigation_result = lambda *_args, **_kwargs: pytest.fail(
        "a stale private worker must not terminalize the replacement context"
    )

    assert executor._consume_container_inner_corridor_result(
        "decision-corridor-replaced",
        True,
        {"status": "SUCCEEDED"},
        candidate={
            "metadata": {
                "container_inner_corridor_navigation": True,
                "container_inner_corridor_run_id": 21,
            }
        },
    ) is True
    assert executor._container_inner_corridors["decision-corridor-replaced"] is context
    assert context["corridor_run_id"] == 22


def test_inner_corridor_never_consumes_physical_tangents_or_final_yaw(
    executor_module,
) -> None:
    """Private waypoints have one goal and defer every retry to their context."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._navigation_is_current = lambda _decision_id: True
    corridor = {
        "behavior_type": "NAVIGATE",
        "goal_xyyaw": [1.0, 0.0, 0.2],
        "metadata": {
            "container_inner_corridor_navigation": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
            "goal_xyyaw_candidates": [],
        },
    }

    assert executor._container_inner_corridor_marker(corridor) is True
    assert executor._retry_interaction_approach(
        "decision-corridor", corridor, 0, [], 1, {"reason": "navigation_stagnation"}
    ) is False
    assert executor_module.navigation_requires_final_yaw(
        "NAVIGATE", True, [1.0, 0.0, 0.2]
    ) is True
    assert executor._requires_final_yaw_for_navigation(
        corridor, "NAVIGATE", True, [1.0, 0.0, 0.2]
    ) is False


def test_inner_corridor_bypasses_rear_goal_direct_control(executor_module) -> None:
    corridor = {
        "metadata": {"container_inner_corridor_navigation": True},
    }
    ordinary = {"metadata": {}}
    assert executor_module.SemanticBehaviorExecutor._container_inner_corridor_marker(
        corridor
    ) is True
    assert executor_module.navigation_should_prerotate("NAVIGATE") is True
    # The run loop combines the existing predicate with this marker.  Keep the
    # marker itself explicit in a small regression, rather than calling direct
    # control helpers in a static test fixture.
    assert not executor_module.SemanticBehaviorExecutor._container_inner_corridor_marker(
        ordinary
    )


@pytest.mark.parametrize("elapsed", [5.1, 14.4])
def test_slow_first_physical_step_does_not_release_backend_ownership(executor_module, elapsed):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.machine = SimpleNamespace(state=executor_module.STATE_INTERACTING)
    executor.selection = {"decision_id": "drawer"}
    executor._interaction_command_sent_id = "command"
    executor._interaction_command_sent_at = 100.0
    executor._latest_step_sync_index = 431
    executor._interaction_execution_progress = {
        "command_id": "command", "received_at": 100.0 + elapsed,
        "progress_age_s": elapsed, "execution_step": 432,
    }
    assert executor._effective_timeout_reason_locked(now=100.0 + elapsed) == ""
    assert not getattr(executor, "_interaction_execution_terminal", {})
    assert executor.machine.state == executor_module.STATE_INTERACTING


def test_missing_backend_ack_ends_episode_without_dispatching_new_goal(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.machine = SimpleNamespace(state=executor_module.STATE_INTERACTING)
    executor.selection = {"decision_id": "drawer"}
    executor._interaction_command_sent_id = "command"
    executor._interaction_command_sent_at = 100.0
    executor._latest_step_sync_index = 431
    published = []
    executor._navigation_terminal_pub = SimpleNamespace(publish=lambda msg: published.append(json.loads(msg.data)))
    assert executor._effective_timeout_reason_locked(now=161.0) == ""
    assert executor.machine.state == executor_module.STATE_INTERACTING
    assert executor._interaction_command_sent_id == "command"
    assert published[0]["status"] == "EXPLORATION_STALLED"
    assert published[0]["detail"]["backend_ownership_retained"]
    assert published[0]["detail"]["reason"] == "interaction_backend_ack_missing"


@pytest.mark.parametrize("status,expected", [
    ("waiting_for_view", "interaction_observation_timeout"), ("pending", ""), ("in_flight", "")
])
def test_short_view_wait_does_not_timeout_a_submitted_m1(executor_module, status, expected):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.machine = SimpleNamespace(
        state=executor_module.STATE_WAITING_FOR_INTERACTION_OBSERVATION,
        timeout_reason=lambda **kwargs: "",
    )
    executor.selection = {"decision_id": "door"}
    executor._latest_step_sync_index = 291
    executor._interaction_observation_requests = {
        "door": {"minimum_capture_step": 285, "request_status": status}
    }
    assert executor._effective_timeout_reason_locked(now=120.0) == expected


def test_late_waiting_ack_cannot_demote_inflight_m1(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.Lock()
    executor.machine = SimpleNamespace(state=executor_module.STATE_WAITING_FOR_INTERACTION_OBSERVATION)
    executor.selection = {"decision_id": "door"}
    executor._latest_attribute_updates = {}
    executor._interaction_observation_requests = {
        "door": {"request_id": "refresh", "object_id": "door_1", "request_status": "in_flight"}
    }
    executor._attribute_update_callback(executor_module.String(data=json.dumps({"updates": [{
        "object_id": "door_1", "targeted_refresh": True,
        "targeted_refresh_request_id": "refresh", "attribute_status": "waiting_for_view"
    }]})))
    assert executor._interaction_observation_requests["door"]["request_status"] == "in_flight"


def test_drawer_scan_wait_uses_finite_simulator_step_budget_not_wall_time(
    executor_module,
) -> None:
    candidate = {
        "decision_id": "decision-drawer-budget",
        "candidate_id": "interaction:drawer-budget:scan",
        "behavior_type": "INTERACT",
        "metadata": {"requires_approach": False},
        "interaction_command": {"sequence_type": "drawer_scan"},
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.selection = candidate
    executor.machine = executor_module.BehaviorExecutionStateMachine(
        executor_module.ExecutionConfig(interaction_timeout_s=30.0)
    )
    executor.machine.start(candidate, now=0.0)
    assert executor.machine.state == executor_module.STATE_INTERACTING
    executor.drawer_scan_execution_step_budget_authoritative = True
    executor.drawer_scan_execution_max_task_steps = 120
    executor.drawer_scan_execution_step_budget_margin = 8
    executor.drawer_scan_execution_step_sync_stall_timeout_s = 5.0
    executor.drawer_scan_execution_wall_cap_s = 180.0
    executor._latest_step_sync_index = 141
    executor._latest_step_sync_received_at = 55.0
    executor._drawer_scan_execution_wait = {
        "command_id": "drawer-command",
        "decision_id": "decision-drawer-budget",
        "candidate_id": "interaction:drawer-budget:scan",
        "sequence_type": "drawer_scan",
        "started_step_index": 100,
        "started_at_monotonic_s": 0.0,
        "max_task_steps": 120,
        "step_budget_margin": 8,
    }

    # The exact house2 macro is still progressing after 41 simulator steps,
    # despite having exceeded the old 30-second host-time timeout.
    assert executor.machine.timeout_reason(now=55.0) == "interaction_timeout"
    assert executor._drawer_scan_execution_timeout_reason_locked(now=55.0) == ""
    assert executor._effective_timeout_reason_locked(now=55.0) == ""

    executor._latest_step_sync_index = 229
    executor._latest_step_sync_received_at = 56.0
    assert (
        executor._drawer_scan_execution_timeout_reason_locked(now=56.0)
        == "drawer_scan_execution_step_budget_exhausted"
    )
    assert (
        executor._effective_timeout_reason_locked(now=56.0)
        == "drawer_scan_execution_step_budget_exhausted"
    )


def test_drawer_scan_wait_ignores_stale_callback_timestamp_after_step_progress(
    executor_module,
) -> None:
    candidate = {
        "decision_id": "decision-drawer-stale-callback",
        "candidate_id": "interaction:drawer-stale:scan",
        "behavior_type": "INTERACT",
        "metadata": {"requires_approach": False},
        "interaction_command": {"sequence_type": "drawer_scan"},
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.selection = candidate
    executor.machine = executor_module.BehaviorExecutionStateMachine(
        executor_module.ExecutionConfig(interaction_timeout_s=30.0)
    )
    executor.machine.start(candidate, now=0.0)
    executor.drawer_scan_execution_step_sync_stall_timeout_s = 5.0
    executor.drawer_scan_execution_wall_cap_s = 180.0
    executor._latest_step_sync_index = 141
    executor._latest_step_sync_received_at = 1.0
    executor._drawer_scan_execution_wait = {
        "decision_id": "decision-drawer-stale-callback",
        "started_step_index": 100,
        "started_at_monotonic_s": 0.0,
        "max_task_steps": 120,
        "step_budget_margin": 8,
    }
    # Even a host delay beyond the emergency wall cap is not a failure while
    # the public evaluator clock is advancing below its finite step budget.
    assert executor._drawer_scan_execution_timeout_reason_locked(now=181.0) == ""


def test_drawer_scan_wait_active_only_for_matching_sent_command(executor_module) -> None:
    candidate = {
        "decision_id": "decision-drawer-owner",
        "candidate_id": "interaction:drawer-owner:scan",
        "behavior_type": "INTERACT",
        "metadata": {"requires_approach": False},
        "interaction_command": {"sequence_type": "drawer_scan"},
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.selection = candidate
    executor.machine = executor_module.BehaviorExecutionStateMachine(
        executor_module.ExecutionConfig(interaction_timeout_s=30.0)
    )
    executor.machine.start(candidate, now=0.0)
    executor._interaction_command_sent_id = "drawer-command"
    executor._drawer_scan_execution_wait = {
        "command_id": "drawer-command",
        "decision_id": "decision-drawer-owner",
        "candidate_id": "interaction:drawer-owner:scan",
        "sequence_type": "drawer_scan",
    }
    assert executor._drawer_scan_execution_wait_active_locked() is True

    executor._interaction_command_sent_id = "different-command"
    assert executor._drawer_scan_execution_wait_active_locked() is False
    executor._interaction_command_sent_id = "drawer-command"
    executor._drawer_scan_execution_wait["candidate_id"] = "other-candidate"
    assert executor._drawer_scan_execution_wait_active_locked() is False
    executor._drawer_scan_execution_wait["candidate_id"] = "interaction:drawer-owner:scan"
    executor._drawer_scan_execution_wait["sequence_type"] = "ordinary"
    assert executor._drawer_scan_execution_wait_active_locked() is False


def test_executor_inner_navigation_failure_dispatches_next_outer_staging(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.interaction_approach_fallback_max_attempts = 4
    executor.container_two_stage_fallback_max_attempts = 12
    executor._navigation_is_current = lambda _decision_id: True
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)
    executor.selection = None
    executor._container_m1_last_accepted_evidence = {
        "decision-inner": {"capture_step": 11}
    }
    candidate = {
        "decision_id": "decision-inner",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [3.0, 2.0, 0.0],
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [3.0, 2.0, 0.0],
            "interaction_ready_distance_m": 0.18,
            "container_staging_ready_distance_m": 0.30,
            "container_physical_action_ready_distance_m": 0.18,
        },
        "metadata": {
            "requires_approach": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
            # The generic portal budget is four fallbacks, but a two-stage
            # container must be able to reach a later advertised outer face.
            "container_two_stage_staging_goal_option_index": 4,
            "container_staging_goal_xyyaw_candidates": [
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 1.57],
                [3.0, 2.0, 3.14],
                [4.0, 2.0, -1.57],
                [5.0, 2.0, 0.0],
                [6.0, 2.0, 1.57],
            ],
            "container_staging_pose_labels": [
                "safe_outer",
                "safe_left",
                "safe_right",
                "safe_back",
                "safe_far",
                "safe_far_left",
            ],
            "container_action_goal_xyyaw_by_staging_index": [
                [3.0, 2.0, 0.0],
                [4.0, 2.0, 1.57],
                [5.0, 2.0, 3.14],
                [6.0, 2.0, -1.57],
                [7.0, 2.0, 0.0],
                [8.0, 2.0, 1.57],
            ],
            "container_two_stage_staging_observation_required": True,
            "container_two_stage_staging_container_pre_action_observation": True,
            "container_two_stage_staging_drawer_pre_action_observation": False,
            "observation_required": False,
            "container_pre_action_observation": False,
            "drawer_pre_action_observation": False,
            "m1_observation_staging_required": False,
        },
    }
    executor.machine.start(candidate, now=0.0)
    executor.selection = dict(executor.machine.candidate)

    retried = executor._retry_interaction_approach(
        "decision-inner",
        candidate,
        0,
        [
            {"index": 0, "phase": "staging"},
            {"index": 1, "phase": "staging"},
            {"index": 2, "phase": "staging"},
            {"index": 3, "phase": "staging"},
            {"index": 4, "phase": "physical_action"},
        ],
        1,
        {"reason": "navigation_stagnation"},
    )

    assert retried is True
    assert [command["kind"] for command in dispatched] == ["navigate"]
    assert dispatched[0]["start_goal_option_index"] == 5
    assert executor.machine.candidate["metadata"]["container_two_stage_phase"] == (
        "staging"
    )
    assert executor.machine.candidate["metadata"][
        "container_pre_action_observation"
    ] is True
    assert executor._container_m1_last_accepted_evidence == {}


def test_direct_m1_capture_failure_returns_to_next_outer_anchor(executor_module) -> None:
    """A failed visual capture may not authorize M1 from an outer anchor."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.interaction_approach_fallback_max_attempts = 8
    executor.container_two_stage_fallback_max_attempts = 4
    executor._navigation_is_current = lambda _decision_id: True
    executor._container_m1_last_accepted_evidence = {}
    executor._confirm_container_m1_capture_goal_inactive_before_profile_restore = (
        lambda: (True, {"server_status_confirmed": True})
    )
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)
    candidate = {
        "decision_id": "decision-capture",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.10, 2.0, 0.0],
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [1.10, 2.0, 0.0],
            "interaction_ready_distance_m": 0.30,
            "container_staging_ready_distance_m": 0.30,
            "container_physical_action_ready_distance_m": 0.18,
        },
        "metadata": {
            "requires_approach": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "m1_capture",
            "container_two_stage_staging_goal_option_index": 0,
            "container_staging_goal_xyyaw_candidates": [
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 1.57],
                [3.0, 2.0, 3.14],
            ],
            "container_m1_capture_goal_xyyaw_by_staging_index": [
                [1.10, 2.0, 0.0],
                [2.10, 2.0, 1.57],
                [3.10, 2.0, 3.14],
            ],
            "container_staging_pose_labels": ["anchor_0", "anchor_1", "anchor_2"],
            # Capture failures use the evidence scheduler, not legacy index +1.
            "container_m1_viewpoint_order": [0, 2, 1],
            "container_two_stage_staging_observation_required": True,
            "container_two_stage_staging_container_pre_action_observation": True,
            "container_two_stage_staging_drawer_pre_action_observation": False,
            "m1_observation_staging_required": True,
            "observation_required": True,
            "container_pre_action_observation": True,
        },
    }
    executor.machine.start(candidate, now=0.0)
    executor.selection = dict(executor.machine.candidate)

    assert executor._retry_interaction_approach(
        "decision-capture",
        candidate,
        0,
        [{"index": 0, "phase": "m1_capture"}],
        1,
        {"reason": "navigation_terminal_failure"},
    )
    assert [command["kind"] for command in dispatched] == ["navigate"]
    assert dispatched[0]["start_goal_option_index"] == 2
    assert executor.machine.candidate["metadata"]["container_two_stage_phase"] == "staging"
    assert executor.machine.candidate["goal_xyyaw"] == [1.0, 2.0, 0.0]
    assert executor.machine.candidate["metadata"]["m1_observation_staging_required"] is True


def test_outer_staging_failure_uses_m1_viewpoint_order(executor_module) -> None:
    """A failed anchor must not fall through to the old linear ring order."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.interaction_approach_fallback_max_attempts = 8
    executor.container_two_stage_fallback_max_attempts = 4
    executor._navigation_is_current = lambda _decision_id: True
    executor._confirm_move_base_goal_quiescent_before_successor = lambda: (
        True,
        {"server_status_confirmed": True},
    )
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)
    candidate = {
        "decision_id": "decision-staging-order",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [2.0, 0.0, 0.0],
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [2.0, 0.0, 0.0],
            "interaction_ready_distance_m": 0.30,
            "container_staging_ready_distance_m": 0.30,
        },
        "metadata": {
            "requires_approach": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "container_two_stage_staging_goal_option_index": 2,
            "interaction_approach_goal_option_index": 2,
            "container_staging_goal_xyyaw_candidates": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            "container_m1_viewpoint_order": [0, 2, 1, 3],
            "container_staging_pose_labels": ["face_0", "face_1", "face_2", "face_3"],
            "container_two_stage_staging_observation_required": True,
            "container_two_stage_staging_container_pre_action_observation": True,
            "container_two_stage_staging_drawer_pre_action_observation": False,
            "m1_observation_staging_required": True,
            "observation_required": True,
            "container_pre_action_observation": True,
            "interaction_observation_attempts": 2,
            "interaction_observation_viewpoint_staging_indices": [0],
        },
    }
    executor.machine.start(candidate, now=0.0)
    executor.selection = dict(executor.machine.candidate)

    assert executor._retry_interaction_approach(
        "decision-staging-order",
        candidate,
        2,
        [{"index": 2, "phase": "staging"}],
        4,
        {"reason": "navigation_terminal_failure"},
    )
    assert [command["kind"] for command in dispatched] == ["navigate"]
    # Evidence order is 0 -> 2 -> 1 -> 3, so index 1—not legacy index 3—is
    # the next usable capture anchor after index 2 fails.
    assert dispatched[0]["start_goal_option_index"] == 1
    assert executor.machine.candidate["metadata"][
        "container_m1_unavailable_staging_indices"
    ] == [2]


def test_outer_m1_staging_batches_all_remaining_anchor_preflights(
    executor_module, monkeypatch
) -> None:
    """Empty plans are scanned in one worker without per-anchor redispatch."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def wait_for_server(self, _timeout) -> bool:
            return True

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    executor._current_pose = lambda _frame_id: None
    preflight_goals = []

    def preflight(_frame_id, x, y, _yaw):
        preflight_goals.append((x, y))
        return False, None, "empty_plan"

    executor._preflight_navigation_plan = preflight
    retry_calls = []
    executor._retry_interaction_approach = (
        lambda _decision_id, _candidate, selected_index, attempts, count, detail: (
            retry_calls.append((selected_index, attempts, count, detail)) or True
        )
    )
    candidate = {
        "candidate_id": "interaction:drawer:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "interaction_command": {
            "interaction_ready_distance_m": 0.30,
            "container_staging_ready_distance_m": 0.30,
        },
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
            "container_staging_goal_xyyaw_candidates": [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            "goal_xyyaw_candidates": [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
        },
    }

    executor._run_navigation_impl(
        "decision-m1-anchor-only",
        candidate,
        start_goal_option_index=0,
        interaction_approach_attempts=[],
        navigation_run_token=1,
    )

    assert preflight_goals == [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    assert len(retry_calls) == 1
    assert retry_calls[0][0] == 2
    assert retry_calls[0][1][-1]["index"] == 2
    assert [item["index"] for item in retry_calls[0][3]["attempted_goals"]] == [
        0,
        1,
        2,
    ]


def test_outer_staging_exhaustion_reports_navigation_not_m1_failure(executor_module) -> None:
    """No reachable follow-up view may permanently exclude an M1-inconclusive drawer."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.interaction_approach_fallback_max_attempts = 8
    executor.container_two_stage_fallback_max_attempts = 2
    executor._navigation_is_current = lambda _decision_id: True
    dispatched = []
    executor._dispatch = lambda commands: dispatched.extend(commands)
    candidate = {
        "decision_id": "decision-staging-defer",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [1.0, 0.0, 0.0],
            "interaction_ready_distance_m": 0.30,
            "container_staging_ready_distance_m": 0.30,
        },
        "metadata": {
            "requires_approach": True,
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "container_two_stage_staging_goal_option_index": 1,
            "interaction_approach_goal_option_index": 1,
            "container_staging_goal_xyyaw_candidates": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "container_m1_viewpoint_order": [0, 1],
            "container_staging_pose_labels": ["face_0", "face_1"],
            "container_two_stage_staging_observation_required": True,
            "container_two_stage_staging_container_pre_action_observation": False,
            "container_two_stage_staging_drawer_pre_action_observation": True,
            "m1_observation_staging_required": True,
            "observation_required": True,
            "drawer_pre_action_observation": True,
            "interaction_observation_attempts": 2,
            "interaction_observation_viewpoint_staging_indices": [0],
        },
    }
    executor.machine.start(candidate, now=0.0)
    executor.selection = dict(executor.machine.candidate)

    assert executor._retry_interaction_approach(
        "decision-staging-defer",
        candidate,
        1,
        [{"index": 1, "phase": "staging"}],
        2,
        {"reason": "navigation_terminal_failure"},
    )
    assert [command["kind"] for command in dispatched] == ["terminal"]
    detail = dispatched[0]["detail"]
    assert detail["m1_evidence_inconclusive"] is False
    assert detail["retryable"] is True
    assert detail["terminal_candidate_exclusion"] is False
    assert detail["m1_viewpoint_navigation_inconclusive"] is True
    assert detail["reason"] == "container_approach_navigation_unreachable"
    assert detail["failure_stage"] == "interaction_approach_navigation"


def test_inner_preflight_exhaustion_uses_last_tangent_before_outer_retry(
    executor_module, monkeypatch
) -> None:
    """A failed tangent preflight must not restart the inner sequence at zero."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def wait_for_server(self, _timeout) -> bool:
            return True

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor._navigation_is_current = lambda _decision_id: True
    executor._current_pose = lambda _frame_id: None
    executor._preflight_navigation_plan = lambda *_args: (
        False,
        None,
        "empty_plan",
    )
    retries = []
    executor._retry_interaction_approach = (
        lambda _decision_id, _candidate, selected_index, attempts, _count, detail: (
            retries.append((selected_index, attempts, detail)) or True
        )
    )

    candidate = {
        "decision_id": "decision-inner-preflight",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "interaction_command": {
            "interaction_ready_distance_m": 0.25,
            "interaction_ready_yaw_tolerance_rad": 0.35,
        },
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
            "interaction_approach_pose_labels": [
                "physical_primary",
                "physical_tangent_left",
                "physical_tangent_right",
            ],
            "goal_xyyaw_candidates": [
                [1.0, 0.22, 0.0],
                [1.0, -0.22, 0.0],
            ],
        },
    }

    # Start at tangent-left.  The real preflight loop then also checks
    # tangent-right; its failure must advance from index 2 to outer staging,
    # not restart index 1 via the old hard-coded zero.
    executor._run_navigation(
        "decision-inner-preflight",
        candidate,
        start_goal_option_index=1,
        interaction_approach_attempts=[{"index": 0, "phase": "physical_action"}],
    )

    assert len(retries) == 1
    selected_index, attempts, detail = retries[0]
    assert selected_index == 2
    assert attempts[-1]["index"] == 2
    assert attempts[-1]["goal_xyyaw"] == [1.0, -0.22, 0.0]
    assert attempts[-1]["approach_pose_label"] == "physical_tangent_right"
    assert detail["reason"] == "make_plan_unreachable"


def test_bridge_inner_pose_precondition_returns_next_outer_staging(executor_module) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    executor.interaction_approach_fallback_max_attempts = 4
    executor._container_m1_last_accepted_evidence = {
        "decision-bridge": {"capture_step": 11}
    }
    candidate = {
        "decision_id": "decision-bridge",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [3.0, 2.0, 0.0],
        "interaction_command": {
            "interaction_approach_pose_xyyaw": [3.0, 2.0, 0.0],
            "interaction_ready_distance_m": 0.18,
            "container_staging_ready_distance_m": 0.30,
            "container_physical_action_ready_distance_m": 0.18,
        },
        "metadata": {
            "requires_approach": True,
            "node_type": "container",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "physical_action",
            "container_two_stage_staging_goal_option_index": 0,
            "interaction_approach_goal_option_index": 0,
            "interaction_approach_attempts": [{"index": 0, "phase": "physical_action"}],
            "container_staging_goal_xyyaw_candidates": [
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 1.57],
            ],
            "container_staging_pose_labels": ["safe_outer", "safe_far"],
            "container_action_goal_xyyaw_by_staging_index": [
                [3.0, 2.0, 0.0],
                [4.0, 2.0, 1.57],
            ],
            "container_two_stage_staging_observation_required": True,
            "container_two_stage_staging_container_pre_action_observation": True,
            "container_two_stage_staging_drawer_pre_action_observation": False,
            "observation_required": False,
            "container_pre_action_observation": False,
            "drawer_pre_action_observation": False,
            "m1_observation_staging_required": False,
        },
    }
    executor.machine.start(candidate, now=0.0)
    executor.machine.state = executor_module.STATE_INTERACTING
    executor.selection = dict(executor.machine.candidate)

    commands = executor._retry_interaction_approach_after_pose_failure_locked(
        {"failure_reason": "unsafe_open_sweep", "recommended_retreat_m": 0.2}
    )

    assert [command["kind"] for command in commands] == ["navigate"]
    assert commands[0]["start_goal_option_index"] == 0
    assert executor.machine.candidate["metadata"]["container_two_stage_phase"] == (
        "physical_action"
    )
    assert executor.machine.candidate["metadata"][
        "unsafe_open_sweep_retreat_attempted"
    ] is True
    assert executor.machine.candidate["goal_xyyaw"] == pytest.approx(
        [2.8, 2.0, 0.0]
    )
    assert executor.machine.candidate["interaction_command"][
        "interaction_ready_distance_m"
    ] == pytest.approx(0.18)
    assert "decision-bridge" in executor._container_m1_last_accepted_evidence


def test_fresh_m1_drawer_plan_uses_sequential_scan_contract(executor_module) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.container_pre_action_require_direct_front = True
    candidate = {
        "metadata": {"drawer_pre_action_observation": True},
        "interaction_command": {
            "node_id": "container_drawers",
            "object_id": "drawer_unit",
            "node_type": "container",
            "action": "open",
        },
    }
    planned, reason = executor._drawer_candidate_from_m1_update_locked(
        candidate,
        {
            "attribute_status": "ready",
            "is_currently_visible": True,
            "observation_capture_step": 22,
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "needs_reobserve": False,
            "observed_bbox_2d": [10, 20, 110, 220],
            "action_regions": [
                {"center": [0.5, 0.20], "confidence": 0.9},
                {"center": [0.5, 0.80], "confidence": 0.9},
            ],
            "source": "mllm_attribute_inference",
        },
        {"minimum_capture_step": 21},
    )
    assert reason == "ready"
    command = planned["interaction_command"]
    assert command["action"] == "scan"
    assert command["sequence_type"] == "drawer_scan"
    assert command["visual_operation_plan"]["drawer_sequence_type"] == "drawer_scan"


def test_low_view_drawer_scan_contract_allows_empty_regions_but_not_drawer_open(
    executor_module,
) -> None:
    """A valid front/bbox M1 frame may request the scan-all fallback only."""
    executor_cls = executor_module.SemanticBehaviorExecutor
    scan = {
        "sequence_type": "drawer_scan",
        "open_regions": [],
        "drawer_scan_fallback_to_all": True,
        "drawer_container_bbox_2d": [10, 20, 110, 220],
        "drawer_container_capture_step": 22,
    }
    assert executor_cls._has_valid_drawer_visual_contract(scan) is True

    open_command = {
        **scan,
        "sequence_type": "drawer_open",
    }
    assert executor_cls._has_valid_drawer_visual_contract(open_command) is False


def test_low_view_drawer_m1_without_regions_builds_scan_all_candidate(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.container_pre_action_require_direct_front = True
    candidate = {
        "metadata": {"drawer_pre_action_observation": True},
        "interaction_command": {
            "node_id": "container_drawers",
            "object_id": "drawer_unit",
            "node_type": "container",
            "action": "open",
        },
    }
    planned, reason = executor._drawer_candidate_from_m1_update_locked(
        candidate,
        {
            "attribute_status": "ready",
            "is_currently_visible": True,
            "observation_capture_step": 22,
            "view_state": "front",
            "front_surface_visible": True,
            "approach_ready": True,
            "needs_reobserve": False,
            "observed_bbox_2d": [10, 20, 110, 220],
            # Low camera view: M1 identifies the drawer unit but cannot provide
            # a reliable crop-relative region. The sealed scan must still be
            # allowed to enumerate every simulator slide joint.
            "action_regions": [],
            "source": "mllm_attribute_inference",
        },
        {"minimum_capture_step": 21},
    )
    assert reason == "ready"
    assert planned is not None
    command = planned["interaction_command"]
    assert command["sequence_type"] == "drawer_scan"
    assert command["open_regions"] == []
    assert command["drawer_scan_fallback_to_all"] is True


def test_drawer_contact_allows_vertical_crop_but_rejects_lateral_crop(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.container_pre_action_require_direct_front = True
    candidate = {
        "metadata": {"drawer_pre_action_observation": True},
        "interaction_command": {
            "node_id": "container_drawers",
            "object_id": "drawer_unit",
            "node_type": "container",
            "action": "open",
        },
    }
    base = {
        "attribute_status": "ready",
        "is_currently_visible": True,
        "observation_capture_step": 22,
        "view_state": "front",
        "front_surface_visible": True,
        "approach_ready": True,
        "needs_reobserve": False,
        "observed_bbox_2d": [10, 20, 110, 220],
        "action_regions": [{"center": [0.5, 0.5], "confidence": 0.9}],
        "source": "mllm_attribute_inference",
    }
    vertical_crop = {
        **base,
        "visual_evidence_truncated": True,
        "visual_evidence_truncated_edges": ["bottom"],
    }
    planned, reason = executor._drawer_candidate_from_m1_update_locked(
        candidate, vertical_crop, {"minimum_capture_step": 21}
    )
    assert reason == "ready"
    assert planned["interaction_command"]["sequence_type"] == "drawer_scan"

    lateral_crop = {
        **base,
        "visual_evidence_truncated": True,
        "visual_evidence_truncated_edges": ["right"],
    }
    planned, reason = executor._drawer_candidate_from_m1_update_locked(
        candidate, lateral_crop, {"minimum_capture_step": 21}
    )
    assert planned is None
    assert reason == "m1_visual_evidence_laterally_truncated"


def test_mllm_module3_dispatches_interaction_without_preaction_planning(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.ablation = SimpleNamespace(module3="mllm_skill_verified")
    published = []
    executor._publish_interaction_command = lambda candidate: published.append(candidate)
    executor._plan_and_publish_interaction = lambda *_args: pytest.fail(
        "M3 must not plan a pre-action approach"
    )
    candidate = _portal_selection()

    executor._dispatch([{"kind": "interact", "candidate": candidate}])

    assert published == [candidate]


def test_m3_composite_evidence_uses_one_image_with_full_frame_and_target_box(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.latest_image = np.full((120, 200, 3), (12, 34, 56), dtype=np.uint8)
    executor.mllm_crop_margin_ratio = 0.10
    executor.mllm_crop_max_side_px = 512
    executor.mllm_full_image_max_side_px = 960
    node = {
        "attributes": {
            "projected_bbox_2d": [85, 35, 125, 85],
        }
    }

    images, bbox, full_size = executor._visual_interaction_images_locked(node)

    assert len(images) == 1
    assert images[0].startswith("data:image/jpeg;base64,")
    assert bbox == [85, 35, 125, 85]
    assert full_size == [200, 120]
    encoded = images[0].split(",", 1)[1]
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape[:2] == (120, 200)
    # A distant untouched pixel proves the full headcam remains the canvas.
    assert np.allclose(decoded[110, 180], (12, 34, 56), atol=8)
    # The outlined target survives in the composite (yellow in BGR).
    assert decoded[35, 85, 1] > 140
    assert decoded[35, 85, 2] > 140
    assert decoded[35, 85, 0] < 120
    # The crop inset is visibly framed in the same single image.
    assert decoded[4, 4].min() > 150


def test_rear_rotation_sweep_and_reverse_helpers_fail_closed_on_obstacle(
    executor_module,
) -> None:
    width = height = 20
    resolution = 0.1
    free = [0] * (width * height)
    assert executor_module.circular_costmap_rotation_sweep_is_clear(
        free,
        width,
        height,
        resolution,
        (0.0, 0.0),
        0.0,
        (1.0, 1.0),
        math.pi,
        0.1,
        0.20,
        0.05,
    )
    # The obstacle is on the straight reverse ray and must prevent the whole
    # requested backoff, rather than allowing a blind partial command.
    blocked = list(free)
    blocked[10 * width + 7] = 100
    assert executor_module.circular_costmap_linear_sweep_distance(
        blocked,
        width,
        height,
        resolution,
        (0.0, 0.0),
        0.0,
        (1.0, 1.0, 0.0),
        -1.0,
        0.5,
        0.20,
        0.05,
    ) == pytest.approx(0.0)

    # A regular inflation value is a DWA scoring cost, not a collision cell.
    # The executor's explicit footprint sweep must match that distinction.
    soft_inflation = list(free)
    soft_inflation[10 * width + 10] = 99
    assert executor_module.circular_costmap_rotation_sweep_is_clear(
        soft_inflation,
        width,
        height,
        resolution,
        (0.0, 0.0),
        0.0,
        (1.0, 1.0),
        math.pi / 2.0,
        0.1,
        0.20,
        0.05,
        occupied_threshold=253,
    )
    assert not executor_module.circular_costmap_rotation_sweep_is_clear(
        soft_inflation,
        width,
        height,
        resolution,
        (0.0, 0.0),
        0.0,
        (1.0, 1.0),
        math.pi / 2.0,
        0.1,
        0.20,
        0.05,
        occupied_threshold=50,
    )


def test_rear_turn_choice_locks_deterministic_side_on_fresh_costmap(
    executor_module,
) -> None:
    class _Orientation:
        x = y = z = 0.0
        w = 1.0

    occupancy = SimpleNamespace(
        data=[0] * (20 * 20),
        header=SimpleNamespace(frame_id="map"),
        info=SimpleNamespace(
            width=20,
            height=20,
            resolution=0.1,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0),
                orientation=_Orientation(),
            ),
        ),
    )
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._latest_occupancy = occupancy
    executor._latest_occupancy_received_at = executor_module.time.monotonic()
    executor._rear_goal_turn_locks = {}
    executor.rear_goal_local_costmap_max_age_s = 1.0
    executor.rear_goal_enter_angle_rad = math.pi / 2.0
    executor.rear_goal_exit_angle_rad = 0.20
    executor.rear_goal_rotate_speed_rad_s = 1.25
    executor.rear_goal_prerotate_control_dt_s = 0.2
    executor.rear_goal_prerotate_max_control_steps = 14
    executor.rear_goal_robot_radius_m = 0.20
    executor.rear_goal_safety_margin_m = 0.05
    executor.rear_goal_costmap_occupied_threshold = 50
    executor.rear_goal_unknown_is_blocked = True
    executor.rear_goal_rotation_sweep_step_rad = 0.10
    executor.rear_goal_pi_turn_sign = -1
    executor._current_pose = lambda _frame: (1.0, 1.0, 0.0)

    choice, detail = executor._rear_goal_rotation_choice(
        "decision-rear", "map", (-1.0, 1.0)
    )
    assert choice is not None
    assert choice["direction"] == "cw"
    assert detail["selected_turn"] == "cw"
    assert detail["shortest_angle_enforced"] is True
    choice_again, _detail_again = executor._rear_goal_rotation_choice(
        "decision-rear", "map", (-1.0, 1.0)
    )
    assert choice_again is not None
    assert choice_again["turn_sign"] == choice["turn_sign"]


def test_rear_turn_fails_closed_when_shortest_side_is_blocked(
    monkeypatch, executor_module
) -> None:
    """A blocked shortest turn must not trigger a long collision-free spin."""

    class _Orientation:
        x = y = z = 0.0
        w = 1.0

    occupancy = SimpleNamespace(
        data=[0] * (30 * 30),
        header=SimpleNamespace(frame_id="map"),
        info=SimpleNamespace(
            width=30,
            height=30,
            resolution=0.1,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0), orientation=_Orientation()
            ),
        ),
    )
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._latest_occupancy = occupancy
    executor._latest_occupancy_received_at = executor_module.time.monotonic()
    executor._rear_goal_turn_locks = {}
    executor.rear_goal_local_costmap_max_age_s = 1.0
    executor.rear_goal_enter_angle_rad = math.pi / 2.0
    executor.rear_goal_exit_angle_rad = 0.20
    executor.rear_goal_rotate_speed_rad_s = 1.25
    executor.rear_goal_prerotate_control_dt_s = 0.2
    # Enough for a wrapped fallback, which must no longer be selected.
    executor.rear_goal_prerotate_max_control_steps = 28
    executor.rear_goal_robot_radius_m = 0.20
    executor.rear_goal_safety_margin_m = 0.05
    executor.rear_goal_costmap_occupied_threshold = 50
    executor.rear_goal_unknown_is_blocked = True
    executor.rear_goal_rotation_sweep_step_rad = 0.10
    executor.rear_goal_pi_turn_sign = -1
    executor._current_pose = lambda _frame: (1.5, 1.5, 0.0)
    # Model the House4 condition: the short CW sweep collides, but the longer
    # CCW sweep was checked and is clear.
    monkeypatch.setattr(
        executor_module,
        "circular_costmap_rotation_sweep_is_clear",
        lambda _data, _w, _h, _res, _origin, _yaw, _pose, arc, *_args, **_kwargs: arc
        > math.pi,
    )
    target_yaw = -2.04
    target = (1.5 + math.cos(target_yaw), 1.5 + math.sin(target_yaw))

    choice, detail = executor._rear_goal_rotation_choice(
        "decision-wrapped", "map", target
    )

    assert choice is None
    assert detail["reason"] == "rear_goal_shortest_turn_sweep_blocked"
    assert detail["shortest_angle_enforced"] is True


def test_rear_dwa_detector_requires_flips_and_no_progress(executor_module) -> None:
    alternating = [
        {"angular_z": 1.0},
        {"angular_z": -1.0},
        {"angular_z": 1.0},
        {"angular_z": -1.0},
    ]
    assert executor_module.rear_dwa_oscillation_detected(
        alternating,
        minimum_samples=4,
        minimum_sign_flips=2,
        displacement_m=0.01,
        maximum_displacement_m=0.05,
        goal_distance_reduction_m=0.0,
        minimum_goal_distance_reduction_m=0.02,
    )
    assert not executor_module.rear_dwa_oscillation_detected(
        alternating,
        minimum_samples=4,
        minimum_sign_flips=2,
        displacement_m=0.10,
        maximum_displacement_m=0.05,
        goal_distance_reduction_m=0.0,
        minimum_goal_distance_reduction_m=0.02,
    )


def test_rear_prerotate_refuses_blind_command_without_fresh_costmap(
    executor_module,
) -> None:
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.rear_goal_prerotate_enabled = True
    executor.rear_goal_enter_angle_rad = math.pi / 2.0
    executor.rear_goal_local_costmap_max_age_s = 0.75
    executor._latest_occupancy = None
    executor._latest_occupancy_received_at = 0.0
    executor._rear_goal_turn_locks = {}
    executor._last_rear_goal_recovery_detail = {}
    executor._current_pose = lambda _frame: (1.0, 1.0, 0.0)

    assert not executor._prerotate_for_rear_goal(
        "decision-no-map",
        "map",
        -1.0,
        1.0,
        heading_target_xy=(-1.0, 1.0),
    )
    assert executor._last_rear_goal_recovery_detail["reason"] == (
        "rear_goal_local_costmap_unavailable"
    )


def test_rear_prerotate_pairs_rgb_with_the_fresh_bridge_window(
    executor_module,
) -> None:
    """Rear pre-turns must publish after the bridge opens its fresh-cmd gate."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.startup_scan_enabled = False
    executor.rear_goal_prerotate_step_sync_enabled = True
    executor.interaction_final_align_enabled = True
    executor.interaction_final_align_step_sync_enabled = True
    executor._startup_scan_gate = executor_module.StepCommandGate(max_pair_age_s=10.0)
    executor._rear_goal_prerotate_gate = executor_module.StepCommandGate(
        max_pair_age_s=10.0
    )
    executor._interaction_final_align_gate = executor_module.StepCommandGate(
        max_pair_age_s=10.0
    )
    executor._latest_step_sync_index = None
    executor._latest_step_sync_received_at = 0.0
    executor.latest_image_sequence = 0
    executor.latest_image = None
    executor._latest_rgb_step_seq = None
    executor._latest_rgb_step_received_at = 0.0
    image = SimpleNamespace(
        header=SimpleNamespace(seq=93, stamp=SimpleNamespace(to_sec=lambda: 100.25)),
        encoding="bgr8",
        height=1,
        width=1,
        step=3,
        data=bytes((10, 20, 30)),
    )

    executor._image_callback(image)
    assert executor._rear_goal_prerotate_gate.consume_step() is None
    executor._fresh_command_gate_callback(
        SimpleNamespace(data=json.dumps({"step_index": 37, "stamp_sec": 100.25}))
    )
    assert executor._rear_goal_prerotate_gate.consume_step() == 37
    assert executor._interaction_final_align_gate.consume_step() == 37

    # The bridge's step-sync acknowledgement unlocks exactly one later window.
    executor._step_sync_callback(
        SimpleNamespace(
            data=json.dumps({"step_index": 37, "action_source": "cmd_vel"})
        )
    )
    acknowledgements = executor._rear_goal_prerotate_gate.take_acks()
    assert len(acknowledgements) == 1
    assert acknowledgements[0].command_applied
    interaction_acknowledgements = executor._interaction_final_align_gate.take_acks()
    assert len(interaction_acknowledgements) == 1
    assert interaction_acknowledgements[0].command_applied


def test_rear_prerotate_converts_one_fresh_window_to_one_yaw_step(
    executor_module,
) -> None:
    """Minimal bridge/executor reproduction for the former timeout-noop race."""

    class PrimedGate(executor_module.StepCommandGate):
        prime_on_reset = False

        def reset(self) -> None:
            super().reset()
            if self.prime_on_reset:
                self.record_rgb(41)
                self.record_fresh_gate(41)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    gate = PrimedGate(max_pair_age_s=10.0)
    gate.prime_on_reset = True
    executor._rear_goal_prerotate_gate = gate
    executor.rear_goal_prerotate_delivery_retry_steps = 0
    executor.rear_goal_pi_tie_tolerance_rad = 0.20
    executor.rear_goal_pi_turn_sign = -1
    executor._navigation_is_current = lambda _decision_id: True
    pose = [0.0, 0.0, 0.0]
    executor._current_pose = lambda _frame_id: tuple(pose)
    published = []

    def publish_rotation(angular_z: float) -> None:
        published.append(float(angular_z))
        if abs(angular_z) <= 1e-9:
            return
        # This is the RosBridgePolicy fixed-dt yaw target that the simulator
        # applies after accepting the fresh cmd_vel in evaluator step 41.
        pose[2] += float(angular_z) * 0.2
        gate.record_step_sync(41, action_source="cmd_vel")

    executor._publish_rotation = publish_rotation

    assert executor._rotate_to_yaw(
        "decision-41",
        "tf_frame_map",
        target_yaw=0.25,
        tolerance_rad=0.02,
        speed_rad_s=1.25,
        timeout_s=1.0,
        turn_sign=1,
        max_prerotate_control_steps=1,
        step_sync_stall_timeout_s=0.2,
    )
    assert published == [1.25, 0.0]
    assert pose[2] == pytest.approx(0.25)


def test_interaction_arrival_tolerance_completes_selected_fallback_when_plan_is_stale(
    executor_module, monkeypatch
) -> None:
    """A ready fallback pose must not be retried solely for a stale DWA plan."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def __init__(self) -> None:
            self.cancel_count = 0
            self.sent_goals = []

        def wait_for_server(self, _timeout) -> bool:
            return True

        def send_goal(self, goal) -> None:
            self.sent_goals.append(goal)

        def get_state(self) -> int:
            return 0  # ACTIVE, intentionally non-terminal.

        def cancel_goal(self) -> None:
            self.cancel_count += 1

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    monkeypatch.setattr(
        executor_module.rospy, "Time", SimpleNamespace(now=lambda: 0.0)
    )
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _seconds: None)

    # The primary candidate is intentionally different.  Navigation starts at
    # fallback #1 and the actual base pose is only valid for that selected
    # fallback, which guards against accepting an approach on the wrong side.
    primary = [2.0, 0.0, 0.0]
    selected_fallback = [1.0, 0.0, 0.0]
    candidate = {
        "candidate_id": "interaction:door:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": primary,
        "metadata": {
            "frame_id": "map",
            "goal_xyyaw_candidates": [primary, selected_fallback],
        },
        "interaction_command": {
            "interaction_ready_distance_m": 0.45,
            "interaction_ready_yaw_tolerance_rad": 0.55,
        },
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.machine = SimpleNamespace(
        config=SimpleNamespace(
            interaction_navigation_timeout_s=10.0,
            navigation_timeout_s=10.0,
        )
    )
    executor.navigation_stagnation_timeout_s = 12.0
    executor.navigation_stagnation_distance_m = 0.10
    executor.navigation_stagnation_yaw_rad = 0.15
    executor.navigation_stagnation_goal_distance_reduction_m = 0.02
    executor.final_align_enabled = True
    executor.final_align_max_distance_m = 0.12
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor._navigation_is_current = lambda _decision_id: True
    executor._preflight_navigation_plan = lambda *_args: (
        True,
        (1.0, 0.0),
        "reachable",
    )
    executor._prerotate_for_rear_goal = lambda *_args, **_kwargs: True
    executor._set_effective_interaction_approach = lambda *_args, **_kwargs: None
    executor._has_fresh_local_plan = lambda *_args, **_kwargs: False
    poses = iter(
        [
            (0.0, 0.0, 0.0),  # initial direct-arrival check: not ready
            (0.80, 0.0, 0.0),  # start-pose bookkeeping after sending the goal
            (0.80, 0.0, 0.0),  # ready for fallback #1, local plan stale
        ]
    )
    executor._current_pose = lambda _frame_id: next(poses)
    completed = []
    executor._complete_interaction_approach_navigation = (
        lambda *args, **kwargs: completed.append((args, kwargs))
    )

    executor._run_navigation("decision-1", candidate, start_goal_option_index=1)

    assert executor.move_base.cancel_count == 1
    assert len(completed) == 1
    _args, kwargs = completed[0]
    assert kwargs["selected_goal"] == tuple(selected_fallback)
    assert kwargs["selected_goal_option_index"] == 1
    assert kwargs["detail"]["reason"] == "interaction_approach_pose_tolerance"
    assert kwargs["detail"]["local_plan_fresh"] is False
    assert kwargs["detail"]["interaction_pose_validation"]["valid"] is True


def test_interaction_missing_path_heading_retries_next_safe_staging_pose(
    executor_module, monkeypatch
) -> None:
    """A fail-open preflight must not terminalize M1 staging without a path yaw."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def wait_for_server(self, _timeout) -> bool:
            return True

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    monkeypatch.setattr(
        executor_module.rospy, "Time", SimpleNamespace(now=lambda: 0.0)
    )

    candidate = {
        "candidate_id": "interaction:container:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
            "container_staging_goal_xyyaw_candidates": [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            "goal_xyyaw_candidates": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        },
        "interaction_command": {
            "interaction_ready_distance_m": 0.30,
            "interaction_ready_yaw_tolerance_rad": 0.30,
        },
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.machine = SimpleNamespace(
        config=SimpleNamespace(
            interaction_navigation_timeout_s=10.0,
            navigation_timeout_s=10.0,
        )
    )
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor.final_align_enabled = True
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _run_token: True
    executor._preflight_navigation_plan = lambda *_args: (
        True,
        None,
        "service_unavailable",
    )
    executor._current_pose = lambda _frame_id: None
    executor._prerotate_for_rear_goal = lambda *_args, **_kwargs: False
    executor._set_effective_interaction_approach = lambda *_args, **_kwargs: None
    executor._last_rear_goal_recovery_detail = {
        "reason": "rear_goal_heading_unavailable",
        "trigger_source": "initial_rear_goal",
    }
    retry_calls = []
    executor._retry_interaction_approach = (
        lambda _decision_id, _candidate, selected_index, attempts, count, detail: (
            retry_calls.append((selected_index, attempts, count, detail)) or True
        )
    )
    terminal_results = []
    executor._handle_navigation_result = (
        lambda *_args, **_kwargs: terminal_results.append((_args, _kwargs))
    )

    executor._run_navigation("decision-missing-heading", candidate)

    assert len(retry_calls) == 1
    selected_index, attempts, option_count, detail = retry_calls[0]
    assert selected_index == 0
    assert attempts[-1]["index"] == 0
    assert option_count == 2
    assert detail["reason"] == "navigation_terminal_failure"
    assert detail["failure_reason"] == "rear_goal_heading_unavailable"
    assert detail["interaction_approach_reposition"] is True
    assert terminal_results == []


def _run_container_staging_rear_failure(
    executor_module, monkeypatch, *, phase: str, retry_result: bool
) -> tuple[list[tuple], list[tuple]]:
    """Exercise the rear-turn exit before a move_base goal is sent."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def wait_for_server(self, _timeout) -> bool:
            return True

        def send_goal(self, _goal) -> None:
            return None

        def get_state(self) -> int:
            return executor_module.GoalStatus.ABORTED

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    candidate = {
        "candidate_id": "interaction:container:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": [1.0, 0.0, 0.0],
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": phase,
            "m1_observation_staging_required": phase == "staging",
            "container_staging_goal_xyyaw_candidates": [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            "goal_xyyaw_candidates": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        },
        "interaction_command": {
            "interaction_ready_distance_m": 0.25,
            "interaction_ready_yaw_tolerance_rad": 0.30,
        },
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.machine = SimpleNamespace(
        config=SimpleNamespace(
            interaction_navigation_timeout_s=10.0,
            navigation_timeout_s=10.0,
        )
    )
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor.final_align_enabled = True
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _run_token: True
    executor._preflight_navigation_plan = lambda *_args: (
        True,
        (1.0, 0.0),
        "reachable",
    )
    # This test isolates the outer/physical rear-turn branch itself; a private
    # inner corridor has separate ownership and result-routing coverage.
    executor._start_container_inner_corridor = lambda *_args, **_kwargs: False
    executor._current_pose = lambda _frame_id: None
    executor._prerotate_for_rear_goal = lambda *_args, **_kwargs: False
    executor._set_effective_interaction_approach = lambda *_args, **_kwargs: None
    executor._last_rear_goal_recovery_detail = {
        "reason": "rear_goal_turn_failed",
        "turn_failure_detail": {
            "reason": "rear_goal_step_budget_exhausted",
            "delivered_step_count": 14,
        },
    }
    retry_calls = []
    executor._retry_interaction_approach = (
        lambda _decision_id, _candidate, selected_index, attempts, count, detail: (
            retry_calls.append((selected_index, attempts, count, detail)) or retry_result
        )
    )
    terminal_results = []
    executor._handle_navigation_result = (
        lambda *_args, **_kwargs: terminal_results.append((_args, _kwargs))
    )

    executor._run_navigation_impl(
        "decision-turn-failed", candidate, navigation_run_token=1
    )
    return retry_calls, terminal_results


def test_two_stage_outer_staging_turn_failure_retries_next_safe_staging_pose(
    executor_module, monkeypatch
) -> None:
    retry_calls, terminal_results = _run_container_staging_rear_failure(
        executor_module, monkeypatch, phase="staging", retry_result=True
    )

    assert len(retry_calls) == 1
    selected_index, attempts, option_count, detail = retry_calls[0]
    assert selected_index == 0
    assert attempts[-1]["index"] == 0
    assert option_count == 2
    assert detail["reason"] == "navigation_terminal_failure"
    assert detail["failure_reason"] == "rear_goal_turn_failed"
    assert detail["turn_failure_detail"]["reason"] == "rear_goal_step_budget_exhausted"
    assert detail["rear_goal_turn_retry_to_next_outer_staging"] is True
    assert terminal_results == []


def test_two_stage_physical_rear_turn_failure_does_not_retry_outer_staging(
    executor_module, monkeypatch
) -> None:
    retry_calls, terminal_results = _run_container_staging_rear_failure(
        executor_module, monkeypatch, phase="physical_action", retry_result=False
    )

    # A physical-action candidate must not enter the new outer-staging retry
    # policy.  It remains a direct fail-closed terminal result here.
    assert retry_calls == []
    assert len(terminal_results) == 1
    args, _kwargs = terminal_results[0]
    assert args[1] is False
    assert args[2]["reason"] == "rear_goal_turn_failed"
    assert "rear_goal_turn_retry_to_next_outer_staging" not in args[2]


def test_two_stage_outer_staging_turn_failure_terminalizes_after_retry_exhaustion(
    executor_module, monkeypatch
) -> None:
    retry_calls, terminal_results = _run_container_staging_rear_failure(
        executor_module, monkeypatch, phase="staging", retry_result=False
    )

    assert len(retry_calls) == 1
    assert len(terminal_results) == 1
    args, _kwargs = terminal_results[0]
    assert args[1] is False
    assert args[2]["reason"] == "rear_goal_turn_failed"
    assert args[2]["failure_reason"] == "rear_goal_turn_failed"
    assert args[2]["rear_goal_turn_retry_to_next_outer_staging"] is True


def test_terminal_move_base_success_preserves_safe_staging_arrival_pose(
    executor_module, monkeypatch
) -> None:
    """A terminal action-client success must retain its valid staging sample.

    The executor used to sample the valid outer container pose only in the
    non-terminal loop.  If move_base completed between loop samples, the later
    post-cancel polling path could lose that pose and consume its bounded
    retries before targeted M1 was requested.
    """

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def wait_for_server(self, _timeout) -> bool:
            return True

        def send_goal(self, _goal) -> None:
            return None

        def get_state(self) -> int:
            return executor_module.GoalStatus.SUCCEEDED

        def get_goal_status_text(self) -> str:
            return "succeeded"

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    monkeypatch.setattr(
        executor_module.rospy, "Time", SimpleNamespace(now=lambda: 0.0)
    )
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)

    primary = [2.0, 0.0, 0.0]
    selected_staging_goal = [1.0, 0.0, 0.0]
    candidate = {
        "candidate_id": "interaction:container:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": primary,
        "metadata": {
            "frame_id": "map",
            "m1_observation_staging_required": True,
            "m1_safe_staging_outer_offset_m": 0.30,
            "m1_safe_staging_arrival_tolerance_m": 0.25,
            "goal_xyyaw_candidates": [primary, selected_staging_goal],
        },
        "interaction_command": {
            "interaction_ready_distance_m": 0.25,
            "interaction_ready_yaw_tolerance_rad": 0.25,
        },
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.machine = SimpleNamespace(
        config=SimpleNamespace(
            interaction_navigation_timeout_s=10.0,
            navigation_timeout_s=10.0,
        )
    )
    executor.navigation_stagnation_timeout_s = 12.0
    executor.navigation_stagnation_distance_m = 0.10
    executor.navigation_stagnation_yaw_rad = 0.15
    executor.navigation_stagnation_goal_distance_reduction_m = 0.02
    executor.final_align_enabled = False
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor._latest_step_sync_index = 73
    executor._navigation_is_current = lambda _decision_id: True
    executor._preflight_navigation_plan = lambda *_args: (
        True,
        (1.0, 0.0),
        "reachable",
    )
    executor._prerotate_for_rear_goal = lambda *_args, **_kwargs: True
    executor._set_effective_interaction_approach = lambda *_args, **_kwargs: None
    executor._start_rear_dwa_monitor = lambda *_args, **_kwargs: None
    poses = iter(
        [
            (0.0, 0.0, 0.0),  # direct-arrival check: not ready
            (0.0, 0.0, 0.0),  # start-pose bookkeeping after send_goal
            (1.0, 0.0, 0.0),  # terminal move_base success at staging pose
        ]
    )
    executor._current_pose = lambda _frame_id: next(poses)
    completed = []
    executor._complete_interaction_approach_navigation = (
        lambda *args, **kwargs: completed.append((args, kwargs))
    )

    executor._run_navigation("decision-terminal", candidate, start_goal_option_index=1)

    assert len(completed) == 1
    _args, kwargs = completed[0]
    assert kwargs["selected_goal"] == tuple(selected_staging_goal)
    assert kwargs["detail"]["interaction_pose_validation_source"] == (
        "move_base_terminal"
    )
    assert kwargs["detail"]["interaction_arrival_step_index"] == 73
    assert kwargs["detail"]["interaction_pose_validation"]["valid"] is True
    assert kwargs["detail"]["goal_distance_m"] == pytest.approx(0.0)


def test_interaction_arrival_tolerance_still_requires_selected_heading() -> None:
    """Position tolerance alone must not accept an approach from the wrong heading."""

    from semantic_decision_py_pkg.behavior_execution import interaction_pose_validation

    validation = interaction_pose_validation(
        [1.0, 0.0, 0.0],
        [0.80, 0.0, 0.60],
        distance_tolerance_m=0.45,
        yaw_tolerance_rad=0.55,
    )
    assert validation["position_error_m"] < validation["distance_tolerance_m"]
    assert validation["yaw_error_rad"] > validation["yaw_tolerance_rad"]
    assert validation["valid"] is False


def test_interaction_ready_standoff_runs_bounded_selected_yaw_alignment(
    executor_module, monkeypatch
) -> None:
    """At a ready standoff, rotate toward the selected fallback before acting."""

    class _Goal:
        def __init__(self) -> None:
            self.target_pose = SimpleNamespace(
                header=SimpleNamespace(frame_id="", stamp=None),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0),
                    orientation=SimpleNamespace(z=0.0, w=1.0),
                ),
            )

    class _MoveBase:
        def __init__(self) -> None:
            self.cancel_count = 0
            self.cancel_waits = []

        def wait_for_server(self, _timeout) -> bool:
            return True

        def send_goal(self, _goal) -> None:
            return None

        def get_state(self) -> int:
            return 0  # ACTIVE, intentionally non-terminal.

        def cancel_goal(self) -> None:
            self.cancel_count += 1

        def wait_for_result(self, timeout) -> None:
            self.cancel_waits.append(timeout)

    monkeypatch.setattr(executor_module, "MoveBaseGoal", _Goal)
    monkeypatch.setattr(executor_module.rospy, "Duration", lambda seconds: seconds)
    monkeypatch.setattr(
        executor_module.rospy, "Time", SimpleNamespace(now=lambda: 0.0)
    )
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _seconds: None)

    primary = [2.0, 0.0, 0.0]
    selected_fallback = [1.0, 0.0, math.pi]
    candidate = {
        "candidate_id": "interaction:door:open",
        "behavior_type": "INTERACT",
        "goal_xyyaw": primary,
        "metadata": {
            "frame_id": "map",
            "goal_xyyaw_candidates": [primary, selected_fallback],
        },
        "interaction_command": {
            "interaction_ready_distance_m": 0.45,
            "interaction_ready_yaw_tolerance_rad": 0.55,
        },
    }
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor.move_base = _MoveBase()
    executor.machine = SimpleNamespace(
        config=SimpleNamespace(
            interaction_navigation_timeout_s=10.0,
            navigation_timeout_s=10.0,
        )
    )
    executor.navigation_stagnation_timeout_s = 12.0
    executor.navigation_stagnation_distance_m = 0.10
    executor.navigation_stagnation_yaw_rad = 0.15
    executor.navigation_stagnation_goal_distance_reduction_m = 0.02
    # Generic NAVIGATE/EXPLORE final alignment stays disabled.  The selected
    # INTERACT fallback must use only its independent controller.
    executor.final_align_enabled = False
    executor.final_align_max_distance_m = 0.12
    executor.interaction_final_align_enabled = True
    executor.interaction_final_align_max_distance_m = 0.35
    executor.interaction_final_align_yaw_tolerance_rad = 0.15
    executor.interaction_final_align_rotate_speed_rad_s = 0.30
    executor.interaction_final_align_timeout_s = 15.0
    executor.interaction_final_align_step_sync_enabled = True
    executor.interaction_final_align_control_dt_s = 0.2
    executor.interaction_final_align_max_control_steps = 56
    executor.interaction_final_align_control_step_margin_steps = 4
    executor.interaction_final_align_step_sync_stall_timeout_s = 2.0
    executor.interaction_final_align_delivery_retry_steps = 2
    executor._interaction_final_align_gate = executor_module.StepCommandGate(
        max_pair_age_s=10.0
    )
    executor.final_align_cancel_wait_s = 0.0
    executor.post_interaction_traversal_make_plan_retry_window_s = 0.0
    executor._navigation_is_current = lambda _decision_id: True
    executor._preflight_navigation_plan = lambda *_args: (
        True,
        (1.0, 0.0),
        "reachable",
    )
    executor._prerotate_for_rear_goal = lambda *_args, **_kwargs: True
    executor._set_effective_interaction_approach = lambda *_args, **_kwargs: None
    executor._has_fresh_local_plan = lambda *_args, **_kwargs: False
    # 18.6 cm from selected fallback, but 0.60 rad away from its required yaw:
    # inside the bridge distance contract and outside its yaw contract.
    ready_but_misaligned = (0.814, 0.0, 2.54)
    poses = iter(
        [
            (0.0, 0.0, 0.0),
            ready_but_misaligned,
            ready_but_misaligned,
            (0.814, 0.0, math.pi),
        ]
    )
    executor._current_pose = lambda _frame_id: next(poses)
    rotate_calls = []
    executor._rotate_to_yaw = (
        lambda *args, **kwargs: rotate_calls.append((args, kwargs)) or True
    )
    completed = []
    executor._complete_interaction_approach_navigation = (
        lambda *args, **kwargs: completed.append((args, kwargs))
    )

    executor._run_navigation("decision-2", candidate, start_goal_option_index=1)

    assert executor.move_base.cancel_count == 1
    assert executor.move_base.cancel_waits == []
    assert rotate_calls == []
    assert len(completed) == 1
    _args, kwargs = completed[0]
    assert kwargs["selected_goal"] == tuple(selected_fallback)
    assert kwargs["detail"]["reason"] == "interaction_approach_pose_tolerance"
    assert kwargs["detail"]["interaction_pose_validation"]["valid"] is True


def test_interaction_final_align_rotation_waits_for_its_own_fresh_step_gate(
    executor_module,
) -> None:
    """Final-align cmd_vel must not reuse a window from rear pre-rotation."""

    class PrimedGate(executor_module.StepCommandGate):
        prime_on_reset = False

        def reset(self) -> None:
            super().reset()
            if self.prime_on_reset:
                self.record_rgb(73)
                self.record_fresh_gate(73)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    gate = PrimedGate(max_pair_age_s=10.0)
    # An old pair exists before final alignment starts.  reset() must discard
    # it and wait for the newly paired step 73 instead.
    gate.record_rgb(41)
    gate.record_fresh_gate(41)
    gate.prime_on_reset = True
    executor._interaction_final_align_gate = gate
    executor.rear_goal_prerotate_delivery_retry_steps = 0
    executor.rear_goal_pi_tie_tolerance_rad = 0.20
    executor.rear_goal_pi_turn_sign = -1
    executor._navigation_is_current = lambda _decision_id: True
    pose = [0.0, 0.0, 0.0]
    executor._current_pose = lambda _frame_id: tuple(pose)
    published = []

    def publish_rotation(angular_z: float) -> None:
        published.append(float(angular_z))
        if abs(angular_z) <= 1e-9:
            return
        pose[2] += float(angular_z) * 0.2
        gate.record_step_sync(73, action_source="cmd_vel")

    executor._publish_rotation = publish_rotation

    assert executor._rotate_to_yaw(
        "decision-73",
        "tf_frame_map",
        target_yaw=0.25,
        tolerance_rad=0.02,
        speed_rad_s=1.25,
        timeout_s=1.0,
        turn_sign=1,
        max_prerotate_control_steps=1,
        step_sync_stall_timeout_s=0.2,
        step_command_gate=gate,
        delivery_retry_steps=0,
        rotation_label="interaction-final-align",
    )
    assert published == [1.25, 0.0]
    assert pose[2] == pytest.approx(0.25)
    diagnostics = gate.diagnostics()
    assert diagnostics["last_sent_step"] == 73
    assert diagnostics["awaiting_ack_step"] is None


def test_interaction_final_align_uses_acknowledged_step_budget_over_wall_timeout(
    executor_module, monkeypatch
) -> None:
    """Slow simulator-step delivery must not truncate an active gated turn."""

    class PrimedGate(executor_module.StepCommandGate):
        def reset(self) -> None:
            super().reset()
            self.record_rgb(1)
            self.record_fresh_gate(1)

    clock = {"now": 0.0}
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        executor_module.time,
        "sleep",
        lambda _seconds: clock.__setitem__("now", clock["now"] + 1.0),
    )
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.rear_goal_prerotate_delivery_retry_steps = 0
    executor.rear_goal_pi_tie_tolerance_rad = 0.20
    executor.rear_goal_pi_turn_sign = -1
    executor._navigation_is_current = lambda _decision_id: True
    gate = PrimedGate(max_pair_age_s=120.0)
    pose = [0.0, 0.0, 0.0]
    executor._current_pose = lambda _frame_id: tuple(pose)
    step = [1]
    published = []

    def publish_rotation(angular_z: float) -> None:
        published.append(float(angular_z))
        if abs(angular_z) <= 1e-9:
            return
        pose[2] += float(angular_z) * 0.2
        gate.record_step_sync(step[0], action_source="cmd_vel")
        step[0] += 1
        gate.record_rgb(step[0])
        gate.record_fresh_gate(step[0])

    executor._publish_rotation = publish_rotation

    # ceil((1.960 - .150) / (.300 * .200)) == 31.  Each acknowledged
    # simulator action takes one synthetic wall-second, deliberately exceeding
    # the legacy 15 s timeout while making real step progress.
    assert executor._rotate_to_yaw(
        "decision-slow-steps",
        "tf_frame_map",
        target_yaw=1.960,
        tolerance_rad=0.150,
        speed_rad_s=0.300,
        timeout_s=15.0,
        turn_sign=1,
        max_prerotate_control_steps=31,
        step_sync_stall_timeout_s=2.0,
        step_command_gate=gate,
        delivery_retry_steps=0,
        rotation_label="interaction-final-align",
        step_sync_budget_authoritative=True,
    )
    assert len([value for value in published if abs(value) > 1e-9]) == 31
    assert clock["now"] > 15.0
    assert abs(1.960 - pose[2]) <= 0.150


def test_interaction_final_align_acknowledged_no_progress_stays_step_bounded(
    executor_module, monkeypatch
) -> None:
    """Authoritative step timing does not make a non-moving turn unbounded."""

    class PrimedGate(executor_module.StepCommandGate):
        def reset(self) -> None:
            super().reset()
            self.record_rgb(1)
            self.record_fresh_gate(1)

    clock = {"now": 0.0}
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        executor_module.time,
        "sleep",
        lambda _seconds: clock.__setitem__("now", clock["now"] + 1.0),
    )
    monkeypatch.setattr(executor_module.rospy, "is_shutdown", lambda: False)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.rear_goal_prerotate_delivery_retry_steps = 0
    executor.rear_goal_pi_tie_tolerance_rad = 0.20
    executor.rear_goal_pi_turn_sign = -1
    executor._navigation_is_current = lambda _decision_id: True
    gate = PrimedGate(max_pair_age_s=120.0)
    executor._current_pose = lambda _frame_id: (0.0, 0.0, 0.0)
    step = [1]
    published = []

    def publish_without_motion(angular_z: float) -> None:
        published.append(float(angular_z))
        if abs(angular_z) <= 1e-9:
            return
        gate.record_step_sync(step[0], action_source="cmd_vel")
        step[0] += 1
        gate.record_rgb(step[0])
        gate.record_fresh_gate(step[0])

    executor._publish_rotation = publish_without_motion

    assert not executor._rotate_to_yaw(
        "decision-no-progress",
        "tf_frame_map",
        target_yaw=1.960,
        tolerance_rad=0.150,
        speed_rad_s=0.300,
        timeout_s=15.0,
        turn_sign=1,
        max_prerotate_control_steps=3,
        step_sync_stall_timeout_s=2.0,
        step_command_gate=gate,
        delivery_retry_steps=0,
        rotation_label="interaction-final-align",
        step_sync_budget_authoritative=True,
    )
    assert len([value for value in published if abs(value) > 1e-9]) == 3
    assert clock["now"] <= 4.0


def test_interaction_final_align_observes_bounded_pose_settle_after_budget(
    executor_module, monkeypatch
) -> None:
    """The last applied yaw action may reach TF after its bridge acknowledgement."""

    class PrimedGate(executor_module.StepCommandGate):
        def reset(self) -> None:
            super().reset()
            self.record_rgb(1)
            self.record_fresh_gate(1)

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.rear_goal_prerotate_delivery_retry_steps = 0
    executor.rear_goal_pi_tie_tolerance_rad = 0.20
    executor.rear_goal_pi_turn_sign = -1
    executor._navigation_is_current = lambda _decision_id: True
    gate = PrimedGate(max_pair_age_s=10.0)
    pose = [0.0, 0.0, 0.0]
    executor._current_pose = lambda _frame_id: tuple(pose)
    command_step = [1]
    commands_sent = [0]
    post_budget_noop_steps = [0]
    published = []

    def publish_rotation(angular_z: float) -> None:
        published.append(float(angular_z))
        if abs(angular_z) <= 1e-9:
            return
        commands_sent[0] += 1
        gate.record_step_sync(command_step[0], action_source="cmd_vel")
        command_step[0] += 1
        gate.record_rgb(command_step[0])
        gate.record_fresh_gate(command_step[0])

    def sleep(seconds: float) -> None:
        # The bridge can keep taking timeout-noop simulator steps after the
        # final cmd_vel.  TF only reflects the last applied yaw after three of
        # those causal boundaries.
        if commands_sent[0] != 3 or seconds > 0.01:
            return
        gate.record_step_sync(command_step[0], action_source="timeout_noop")
        command_step[0] += 1
        gate.record_rgb(command_step[0])
        gate.record_fresh_gate(command_step[0])
        post_budget_noop_steps[0] += 1
        if post_budget_noop_steps[0] == 3:
            pose[2] = 0.60

    monkeypatch.setattr(executor_module.time, "sleep", sleep)
    executor._publish_rotation = publish_rotation

    assert executor._rotate_to_yaw(
        "decision-pose-settle",
        "tf_frame_map",
        target_yaw=0.60,
        tolerance_rad=0.05,
        speed_rad_s=1.0,
        timeout_s=1.0,
        turn_sign=1,
        max_prerotate_control_steps=3,
        step_sync_stall_timeout_s=1.0,
        step_command_gate=gate,
        delivery_retry_steps=0,
        post_budget_settle_steps=3,
        rotation_label="interaction-final-align",
        step_sync_budget_authoritative=True,
    )
    assert len([value for value in published if abs(value) > 1e-9]) == 3
    assert post_budget_noop_steps == [3]
    assert pose[2] == pytest.approx(0.60)


def test_full_mllm_interaction_final_align_is_opted_in_without_generic_align() -> None:
    """The full MLLM lane enables only the interaction-specific controller."""

    default_config = yaml.safe_load(
        (PACKAGE_SCRIPTS.parent / "config" / "default.yaml").read_text()
    )
    override_config = yaml.safe_load(
        (
            PACKAGE_SCRIPTS.parents[3]
            / "scripts"
            / "InteractiveNav"
            / "configs"
            / "semantic_decision"
            / "full_mllm_interactive_exploration.yaml"
        ).read_text()
    )
    assert default_config["executor"]["final_align_enabled"] is False
    assert default_config["executor"]["interaction_final_align_enabled"] is False
    executor_override = override_config["executor"]
    assert executor_override["interaction_final_align_enabled"] is True
    assert executor_override["interaction_final_align_max_distance_m"] == 0.35
    assert executor_override["interaction_final_align_yaw_tolerance_rad"] == 0.15
    assert executor_override["interaction_final_align_rotate_speed_rad_s"] == 0.30
    assert executor_override["interaction_final_align_timeout_s"] == 15.0
    assert executor_override["interaction_final_align_step_sync_enabled"] is True
    assert executor_override["interaction_final_align_control_dt_s"] == 0.2
    assert executor_override["interaction_final_align_max_control_steps"] == 56
    assert executor_override["interaction_final_align_control_step_margin_steps"] == 4
    assert executor_override["interaction_final_align_max_control_steps"] >= (
        math.ceil(math.pi / (0.30 * 0.20)) + 3
    )
    assert executor_override["container_m1_capture_dwa_profile_enabled"] is True
    assert executor_override["container_m1_capture_dwa_reconfigure_server"] == (
        "/move_base/DWAPlannerROS"
    )
    assert executor_override["container_m1_capture_dwa_reconfigure_timeout_s"] == 1.0
    assert (
        executor_override[
            "container_m1_capture_dwa_terminal_yaw_settle_max_task_steps"
        ]
        == 100
    )
    assert executor_override["container_m1_capture_yaw_tolerance_rad"] == pytest.approx(
        math.radians(30.0)
    )
    assert (
        executor_override[
            "container_m1_capture_successor_quiescence_timeout_s"
        ]
        == 1.0
    )
    assert executor_override["move_base_successor_quiescence_timeout_s"] == 1.0
    assert executor_override["navigation_failure_recovery_enabled"] is False
    assert executor_override["navigation_failure_recovery_max_attempts"] == 0
    assert executor_override["navigation_stagnation_post_rotation_grace_s"] == 15.0


def test_interaction_preflight_debug_keeps_skipped_ring_options_separate_from_retry_budget(
    executor_module,
) -> None:
    """Every ring is visible in diagnostics without counting as a navigation try."""

    options = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    ]
    debug = executor_module.SemanticBehaviorExecutor._interaction_preflight_debug_attempts(
        options,
        ["safe_outer", "safe_far", "safe_farthest", "safe_max"],
        start_goal_option_index=1,
        attempted_goals=[
            {
                "index": 1,
                "goal_xyyaw": list(options[1]),
                "reachable": True,
                "preflight_reason": "reachable",
                "preflight_attempts": 1,
            }
        ],
        selected_goal_option_index=1,
    )

    assert [item["approach_pose_label"] for item in debug] == [
        "safe_outer",
        "safe_far",
        "safe_farthest",
        "safe_max",
    ]
    assert debug[0]["preflight_reason"] == "skipped_prior_retry"
    assert debug[0]["preflight_skipped"] is True
    assert debug[1]["preflight_checked"] is True
    assert debug[2]["preflight_reason"] == "skipped_after_reachable_option"
    assert debug[3]["preflight_reason"] == "skipped_after_reachable_option"


def test_outer_staging_terminal_abort_retries_same_pose_once_after_fresh_plan(
    executor_module, monkeypatch
) -> None:
    """A reachable outer M1 stance gets one fresh-plan same-index resend only."""

    started = []

    class _Thread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            started.append((self.target, self.args, self.daemon))

    monkeypatch.setattr(executor_module.threading, "Thread", _Thread)
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor._container_outer_staging_terminal_retry_ledger = set()
    executor._navigation_is_current = lambda decision_id: decision_id == "decision-abort"
    executor._navigation_run_is_active = lambda decision_id, _token: decision_id == "decision-abort"
    executor._confirm_move_base_goal_quiescent_before_successor = lambda: (
        True,
        {"server_status_confirmed": True},
    )
    executor._wait_for_outer_staging_abort_global_costmap_receipt = lambda _decision_id, _token: (
        True,
        {"fresh_source": "global_costmap_update"},
    )
    executor._wait_for_outer_staging_abort_reachable_plan = (
        lambda decision_id, _token, frame_id, x, y, yaw: (
            fresh_calls.append((frame_id, x, y, yaw)) or True,
            (x, y),
            "reachable",
            {"attempts": 2, "reason": "reachable"},
        )
    )
    fresh_calls = []
    executor._preflight_navigation_plan = (
        lambda frame_id, x, y, yaw: (
            fresh_calls.append((frame_id, x, y, yaw)) or True,
            (x, y),
            "reachable",
        )
    )
    candidate = {
        "candidate_id": "interaction:container:open",
        "behavior_type": "INTERACT",
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
        },
    }
    selected_goal = (1.2, 3.4, -0.5)
    detail = {
        "reason": "navigation_terminal_failure",
        "status_code": executor_module.GoalStatus.ABORTED,
        "status": "ABORTED",
    }

    assert executor._retry_outer_staging_after_terminal_abort(
        "decision-abort",
        candidate,
        navigation_run_token=17,
        selected_goal_option_index=2,
        selected_goal=selected_goal,
        selected_preflight_reachable=True,
        interaction_approach_attempts=[{"index": 2, "reachable": True}],
        terminal_detail=detail,
    )
    assert fresh_calls == [("map", *selected_goal)]
    assert len(started) == 1
    _target, args, daemon = started[0]
    assert daemon is True
    assert args[0] == "decision-abort"
    assert args[2] == 2
    retry_metadata = args[1]["metadata"]["container_outer_staging_terminal_retry"]
    assert retry_metadata["goal_option_index"] == 2
    assert retry_metadata["goal_xyyaw"] == list(selected_goal)
    assert retry_metadata["fresh_preflight_reason"] == "reachable"
    assert retry_metadata["fresh_plan"]["attempts"] == 2
    assert retry_metadata["global_costmap_receipt"]["fresh_source"] == "global_costmap_update"

    # This decision cannot spawn a second same-pose worker or even consume
    # another make_plan request after a repeated terminal callback.
    assert not executor._retry_outer_staging_after_terminal_abort(
        "decision-abort",
        candidate,
        navigation_run_token=17,
        selected_goal_option_index=2,
        selected_goal=selected_goal,
        selected_preflight_reachable=True,
        interaction_approach_attempts=[{"index": 2, "reachable": True}],
        terminal_detail=detail,
    )
    assert fresh_calls == [("map", *selected_goal)]
    assert len(started) == 1

    # Bound the recovery at decision scope, rather than giving every one of a
    # twenty-view container's staging points its own six-second retry window.
    assert not executor._retry_outer_staging_after_terminal_abort(
        "decision-abort",
        candidate,
        navigation_run_token=17,
        selected_goal_option_index=3,
        selected_goal=(9.0, 8.0, 0.0),
        selected_preflight_reachable=True,
        interaction_approach_attempts=[{"index": 3, "reachable": True}],
        terminal_detail=detail,
    )
    assert fresh_calls == [("map", *selected_goal)]
    assert len(started) == 1


def test_outer_staging_abort_waits_for_goal_quiescence_before_make_plan(
    executor_module,
) -> None:
    """PREEMPTING is a transport fence, not an unreachable next viewpoint."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor._container_outer_staging_terminal_retry_ledger = set()
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    executor._wait_for_outer_staging_abort_global_costmap_receipt = lambda *_args: (
        True,
        {"fresh_source": "global_costmap_update"},
    )
    executor._confirm_move_base_goal_quiescent_before_successor = lambda: (
        False,
        {"reason": "move_base_successor_server_still_active"},
    )
    executor._wait_for_outer_staging_abort_reachable_plan = lambda *_args: pytest.fail(
        "make_plan must not run while move_base is PREEMPTING"
    )
    candidate = {
        "behavior_type": "INTERACT",
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
        },
    }

    assert not executor._retry_outer_staging_after_terminal_abort(
        "decision-preempting",
        candidate,
        navigation_run_token=3,
        selected_goal_option_index=0,
        selected_goal=(1.0, 2.0, 0.0),
        selected_preflight_reachable=True,
        interaction_approach_attempts=[],
        terminal_detail={
            "reason": "navigation_terminal_failure",
            "status_code": executor_module.GoalStatus.ABORTED,
            "status": "ABORTED",
        },
    )


def test_outer_staging_terminal_abort_requires_new_costmap_before_replan(
    executor_module
) -> None:
    """An old costmap cannot turn an outer ABORT into a blind same-pose resend."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor._container_outer_staging_terminal_retry_ledger = set()
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    executor._wait_for_outer_staging_abort_global_costmap_receipt = lambda _decision_id, _token: (
        False,
        {"reason": "container_outer_staging_abort_replan_global_costmap_timeout"},
    )
    executor._preflight_navigation_plan = lambda *_args: pytest.fail(
        "make_plan must wait for a newer costmap receipt"
    )
    candidate = {
        "candidate_id": "interaction:container:open",
        "behavior_type": "INTERACT",
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": "staging",
            "m1_observation_staging_required": True,
        },
    }
    detail = {
        "reason": "navigation_terminal_failure",
        "status_code": executor_module.GoalStatus.ABORTED,
        "status": "ABORTED",
    }

    assert not executor._retry_outer_staging_after_terminal_abort(
        "decision-no-fresh-map",
        candidate,
        navigation_run_token=1,
        selected_goal_option_index=0,
        selected_goal=(1.0, 0.0, 0.0),
        selected_preflight_reachable=True,
        interaction_approach_attempts=[],
        terminal_detail=detail,
    )


def test_outer_staging_abort_plan_wait_requires_two_healthy_plans(
    executor_module, monkeypatch
) -> None:
    """A fresh receipt needs two spaced real paths before the same-pose resend."""

    class _Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            self.now += float(duration)

    clock = _Clock()
    monkeypatch.setattr(executor_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(executor_module.time, "sleep", clock.sleep)
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.container_outer_staging_abort_replan_plan_retry_window_s = 1.0
    executor.container_outer_staging_abort_replan_plan_retry_interval_s = 0.20
    executor.container_outer_staging_abort_replan_plan_health_confirmations = 2
    executor.container_outer_staging_abort_replan_plan_health_interval_s = 0.30
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    calls = []
    outcomes = iter(
        [
            (False, None, "empty_plan"),
            (True, (1.1, 2.2), "reachable"),
            (True, (1.3, 2.4), "reachable"),
        ]
    )

    def preflight(frame_id, x, y, yaw):
        calls.append((frame_id, x, y, yaw))
        return next(outcomes)

    executor._preflight_navigation_plan = preflight
    reachable, lookahead, reason, detail = (
        executor._wait_for_outer_staging_abort_reachable_plan(
            "decision-replan", 1, "map", 1.0, 2.0, -0.5
        )
    )

    assert reachable is True
    assert lookahead == (1.3, 2.4)
    assert reason == "reachable"
    assert detail["attempts"] == 3
    assert detail["planner_health_required_confirmations"] == 2
    assert detail["planner_health_confirmations"] == 2
    assert clock.now == pytest.approx(0.50)
    assert calls == [("map", 1.0, 2.0, -0.5)] * 3


def test_outer_staging_abort_plan_wait_resets_health_after_transient_failure(
    executor_module, monkeypatch
) -> None:
    """A solitary reachable reply cannot pair with one before an empty plan."""

    class _Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            self.now += float(duration)

    clock = _Clock()
    monkeypatch.setattr(executor_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(executor_module.time, "sleep", clock.sleep)
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.container_outer_staging_abort_replan_plan_retry_window_s = 1.5
    executor.container_outer_staging_abort_replan_plan_retry_interval_s = 0.20
    executor.container_outer_staging_abort_replan_plan_health_confirmations = 2
    executor.container_outer_staging_abort_replan_plan_health_interval_s = 0.30
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    outcomes = iter(
        [
            (True, (1.0, 2.0), "reachable"),
            (False, None, "empty_plan"),
            (True, (1.2, 2.2), "reachable"),
            (True, (1.4, 2.4), "reachable"),
        ]
    )
    executor._preflight_navigation_plan = lambda *_args: next(outcomes)

    reachable, lookahead, reason, detail = (
        executor._wait_for_outer_staging_abort_reachable_plan(
            "decision-reset", 1, "map", 1.0, 2.0, -0.5
        )
    )

    assert reachable is True
    assert lookahead == (1.4, 2.4)
    assert reason == "reachable"
    assert detail["attempts"] == 4
    assert detail["planner_health_confirmations"] == 2
    assert clock.now == pytest.approx(0.80)


def test_outer_staging_abort_plan_wait_times_out_without_reachable_plan(
    executor_module, monkeypatch
) -> None:
    """The bounded same-pose recovery cannot wait forever on empty plans."""

    class _Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            self.now += float(duration)

    clock = _Clock()
    monkeypatch.setattr(executor_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(executor_module.time, "sleep", clock.sleep)
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.container_outer_staging_abort_replan_plan_retry_window_s = 0.40
    executor.container_outer_staging_abort_replan_plan_retry_interval_s = 0.20
    executor.container_outer_staging_abort_replan_plan_health_confirmations = 2
    executor.container_outer_staging_abort_replan_plan_health_interval_s = 0.30
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    calls = []
    executor._preflight_navigation_plan = lambda frame_id, x, y, yaw: (
        calls.append((frame_id, x, y, yaw)) or False,
        None,
        "empty_plan",
    )

    reachable, lookahead, reason, detail = (
        executor._wait_for_outer_staging_abort_reachable_plan(
            "decision-timeout", 1, "map", 1.0, 2.0, -0.5
        )
    )

    assert reachable is False
    assert lookahead is None
    assert reason == "empty_plan"
    assert detail["reason"] == "container_outer_staging_abort_replan_plan_timeout"
    assert detail["attempts"] == len(calls)
    assert len(calls) == 3


def test_outer_staging_abort_plan_wait_stops_when_navigation_token_is_replaced(
    executor_module, monkeypatch
) -> None:
    """A stale ABORT worker cannot preflight or spawn after a newer run owns it."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.container_outer_staging_abort_replan_plan_retry_window_s = 1.0
    executor.container_outer_staging_abort_replan_plan_retry_interval_s = 0.20
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: False
    executor._preflight_navigation_plan = lambda *_args: pytest.fail(
        "a superseded worker must not preflight"
    )

    reachable, lookahead, reason, detail = (
        executor._wait_for_outer_staging_abort_reachable_plan(
            "decision-stale", 4, "map", 1.0, 2.0, -0.5
        )
    )

    assert reachable is False
    assert lookahead is None
    assert reason == "preempted"
    assert detail["reason"] == "container_outer_staging_abort_replan_preempted"
    assert detail["attempts"] == 0


@pytest.mark.parametrize(
    ("phase", "initial_preflight_reachable"),
    [
        ("physical_action", True),
        ("staging", False),
    ],
)
def test_terminal_abort_same_pose_retry_excludes_inner_and_unreachable_staging(
    executor_module, phase, initial_preflight_reachable
) -> None:
    """No inner retry or fail-open resend may bypass outer M1/preflight gates."""

    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor._container_outer_staging_terminal_retry_ledger = set()
    executor._navigation_is_current = lambda _decision_id: True
    executor._navigation_run_is_active = lambda _decision_id, _token: True
    fresh_calls = []
    executor._preflight_navigation_plan = lambda *_args: (
        fresh_calls.append(_args) or True,
        (1.0, 0.0),
        "reachable",
    )
    candidate = {
        "candidate_id": "interaction:container:open",
        "behavior_type": "INTERACT",
        "metadata": {
            "frame_id": "map",
            "container_two_stage_approach": True,
            "container_two_stage_phase": phase,
            "m1_observation_staging_required": phase == "staging",
        },
    }
    detail = {
        "reason": "navigation_terminal_failure",
        "status_code": executor_module.GoalStatus.ABORTED,
        "status": "ABORTED",
    }

    assert not executor._retry_outer_staging_after_terminal_abort(
        "decision-excluded",
        candidate,
        navigation_run_token=1,
        selected_goal_option_index=0,
        selected_goal=(1.0, 0.0, 0.0),
        selected_preflight_reachable=initial_preflight_reachable,
        interaction_approach_attempts=[],
        terminal_detail=detail,
    )
    assert fresh_calls == []


def _reconnecting_executor(module, monkeypatch, *, status=(), changed=False, exited=False):
    executor = object.__new__(module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._move_base_worker_lock = threading.RLock()
    executor._move_base_status_condition = threading.Condition()
    executor._move_base_action_name = "/move_base"
    executor._move_base_owned_goal_id = "owned-old"
    executor._move_base_server_identity = ("http://localhost:1234", 123)
    executor._move_base_client_generation = 0
    executor._move_base_goal_generation = 1
    executor._move_base_reconnect_attempts = 0
    executor._move_base_reconnect_limit = 3
    executor._move_base_transport_terminal = {}
    executor.move_base_successor_quiescence_timeout_s = 0.1
    executor.make_plan_service = "/move_base/make_plan"
    executor.selection = {"decision_id": "decision"}
    executor._probe_move_base_server_identity = lambda: (
        ("http://localhost:5678", 456) if changed else executor._move_base_server_identity
    )
    executor._move_base_previous_process_exited = lambda _identity: exited
    executor._wait_for_move_base_server = lambda *_args: True
    trace = {"created": [], "cancels": [], "stopped": [], "terminal": []}

    class Transport:
        pub_cancel = SimpleNamespace(publish=lambda message: trace["cancels"].append(message.id))

        @property
        def last_status_msg(self):
            return SimpleNamespace(status_list=[
                SimpleNamespace(goal_id=SimpleNamespace(id=goal), status=state)
                for goal, state in status
            ])

        def stop(self):
            trace["stopped"].append(self)

    class Client:
        def __init__(self, *_args):
            self.action_client = Transport()
            trace["created"].append(self)

        def stop_tracking_goal(self):
            pass

        def send_goal(self, *_args):
            pytest.fail("Recovery must never dispatch a goal")

    executor.move_base = Client()
    trace["created"].clear()
    executor._navigation_terminal_pub = SimpleNamespace(
        publish=lambda message: trace["terminal"].append(json.loads(message.data))
    )
    monkeypatch.setattr(module.actionlib, "SimpleActionClient", Client)
    monkeypatch.setattr(sys.modules["actionlib_msgs.msg"], "GoalID", SimpleNamespace, raising=False)
    monkeypatch.setattr(module.rospy, "ServiceProxy", lambda *_args: object(), raising=False)
    return executor, trace


@pytest.mark.parametrize("changed,exited", [(False, False), (True, True)])
def test_move_base_reconnect_requires_fresh_idle_without_sending_goal(
    executor_module, monkeypatch, changed, exited
):
    executor, trace = _reconnecting_executor(executor_module, monkeypatch, changed=changed, exited=exited)
    old = executor.move_base
    ready, detail = executor._recover_move_base_client({"reason": "recalling"})
    assert ready and detail["server_status_confirmed"]
    assert executor.move_base is not old
    assert executor._move_base_client_generation == 1
    assert executor._move_base_owned_goal_id == ""
    assert trace["cancels"] == ([] if exited else ["owned-old"])
    assert old.action_client in trace["stopped"]
    assert executor._move_base_lifecycle_audit[-1]["event"] == "client_reconnected"
    assert executor._move_base_lifecycle_audit[-1]["client_generation"] == 1


def test_move_base_replacement_cannot_overlap_unproven_old_server(executor_module, monkeypatch):
    executor, trace = _reconnecting_executor(executor_module, monkeypatch, changed=True)
    ready, detail = executor._recover_move_base_client({"reason": "server_changed"})
    assert not ready
    assert detail["recovery_reason"] == "previous_server_not_proven_exited"
    assert detail["navigation_transport_terminal"]
    assert not trace["created"] and not trace["cancels"]


def test_move_base_permanent_recalling_latches_run_terminal_once(executor_module, monkeypatch):
    executor, trace = _reconnecting_executor(executor_module, monkeypatch, status=(("owned-old", 7),))
    old = executor.move_base
    for _ in range(6):
        assert not executor._recover_move_base_client({"reason": "recalling"})[0]
    assert executor._move_base_reconnect_attempts == 3
    assert len(trace["created"]) == 3
    assert len(trace["stopped"]) == 3
    assert executor.move_base is old
    assert len(trace["terminal"]) == 1
    assert trace["terminal"][0]["status"] == "EXPLORATION_STALLED"
    assert trace["terminal"][0]["detail"]["retryable"] is False


def test_move_base_reconnect_cannot_replace_busy_worker_client(executor_module, monkeypatch):
    executor, trace = _reconnecting_executor(executor_module, monkeypatch)
    executor._move_base_worker_lock = threading.Lock()
    with executor._move_base_worker_lock:
        assert not executor._recover_move_base_client({"reason": "busy"})[0]
    assert executor._move_base_reconnect_attempts == 0
    assert not trace["created"]


def test_move_base_connection_timeout_does_not_depend_on_ros_clock(executor_module, monkeypatch):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._move_base_action_name = "/move_base"
    clock = iter([0.0, 0.01, 0.02, 0.03, 0.2])
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)
    client = SimpleNamespace(action_client=SimpleNamespace(last_status_msg=None))
    assert not executor._wait_for_move_base_server(client, 0.1)


def test_move_base_terminal_audit_ignores_previous_goal_generation(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._move_base_status_condition = threading.Condition()
    executor._move_base_action_name = "/move_base"
    executor._move_base_owned_goal_id = "new"
    executor._move_base_goal_generation = 2
    executor._move_base_client_generation = 1
    message = SimpleNamespace(status_list=[SimpleNamespace(goal_id=SimpleNamespace(id="old"), status=3)])
    executor._move_base_status_callback(message)
    assert not getattr(executor, "_move_base_lifecycle_audit", ())
    message.status_list[0].goal_id.id = "new"
    executor._move_base_status_callback(message)
    executor._move_base_status_callback(message)
    assert len(executor._move_base_lifecycle_audit) == 1
    assert executor._move_base_lifecycle_audit[0]["goal_generation"] == 2


def _anchor_candidate(phase="staging"):
    return {
        "behavior_type": "INTERACT", "goal_xyyaw": [0.0, 0.0, 0.0],
        "interaction_command": {"container_kind": "drawer"},
        "metadata": {
            "frame_id": "map", "container_two_stage_approach": True,
            "container_two_stage_phase": phase,
            "m1_observation_staging_required": phase == "staging",
            "container_two_stage_staging_goal_option_index": 1,
        },
    }


def _local_anchor_executor(module):
    executor = object.__new__(module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.rear_goal_local_costmap_max_age_s = 0.75
    executor._latest_occupancy_received_at = time.monotonic()
    executor._latest_occupancy = SimpleNamespace(
        header=SimpleNamespace(frame_id="map"),
        info=SimpleNamespace(
            width=100, height=100, resolution=0.05,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=-2.5, y=-2.5),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        data=[0] * 10000,
    )
    return executor


@pytest.mark.parametrize("value,col,expected,reason", [
    (99, 55, True, "container_anchor_clearance_confirmed"),
    (99, 50, False, "container_anchor_center_blocked"),
    (100, 55, False, "container_anchor_arrival_region_blocked"),
    (-1, 55, False, "container_anchor_arrival_region_blocked"),
    (100, 70, True, "container_anchor_clearance_confirmed"),
])
def test_anchor_clearance_does_not_double_inflate_inscribed_cost(
    executor_module, value, col, expected, reason
):
    executor = _local_anchor_executor(executor_module)
    executor._latest_occupancy.data[50 * 100 + col] = value
    clear, detail = executor._container_anchor_local_clearance(_anchor_candidate(), "map", (0.0, 0.0, 0.0))
    assert clear is expected
    assert detail["reason"] == reason


def test_anchor_clearance_requires_global_evidence_outside_local_window(executor_module):
    executor = _local_anchor_executor(executor_module)
    clear, detail = executor._container_anchor_local_clearance(_anchor_candidate(), "map", (10.0, 0.0, 0.0))
    assert not clear and detail["reason"] == "container_anchor_global_map_not_fresh"
    import copy
    executor._latest_portal_planning_occupancy = copy.deepcopy(executor._latest_occupancy)
    executor._latest_portal_planning_occupancy.info.origin.position.x = 7.5
    executor._latest_planning_occupancy_received_at = time.monotonic()
    clear, detail = executor._container_anchor_local_clearance(_anchor_candidate(), "map", (10.0, 0.0, 0.0))
    assert clear and detail["local_clearance_deferred"]
    assert detail["costmap_source"] == "planning_occupancy"
    executor._latest_occupancy_received_at -= 2.0
    clear, detail = executor._container_anchor_local_clearance(_anchor_candidate(), "map", (0.0, 0.0, 0.0))
    assert not clear and detail["reason"] == "container_anchor_local_costmap_not_fresh"


@pytest.mark.parametrize("phase", ["staging", "m1_capture"])
def test_anchor_rejects_viewed_indices_and_actual_pose_aliases(executor_module, phase):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    candidate = _anchor_candidate(phase)
    candidate["metadata"]["interaction_observation_viewpoint_staging_indices"] = [1]
    assert not executor._container_anchor_eligible(candidate, 1, (1.0, 0.0, 0.0))[0]
    candidate["metadata"]["interaction_observation_viewpoint_staging_indices"] = []
    candidate["metadata"]["container_m1_sampled_capture_poses_xyyaw"] = [[0.0, 0.0, math.pi]]
    assert not executor._container_anchor_eligible(candidate, 1, (0.1, 0.0, -math.pi + 0.1))[0]
    assert executor._container_anchor_eligible(candidate, 1, (0.3, 0.0, math.pi))[0]


def test_physical_action_is_not_excluded_by_observation_history(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    candidate = _anchor_candidate("physical_action")
    candidate["metadata"].update({
        "interaction_observation_viewpoint_staging_indices": [1],
        "container_m1_sampled_capture_poses_xyyaw": [[0.0, 0.0, 0.0]],
    })
    assert executor._container_anchor_eligible(candidate, 1, (0.0, 0.0, 0.0))[0]


def test_same_face_yaw_failure_keeps_face_and_retries_at_current_xy(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    candidate = _anchor_candidate("physical_action")
    candidate["interaction_command"].update(navigation_goal_tolerance_contract_explicit=True, navigation_goal_yaw_tolerance_rad=0.2)
    candidate["metadata"]["m1_front_yaw"] = 1.57
    candidate["metadata"]["container_m1_rejected_face_staging_indices"] = []
    executor.selection = candidate
    executor.map_frame = "map"
    executor._current_pose = lambda _frame: (0.66, 7.69, 1.81)
    retries = []
    executor.machine = SimpleNamespace(candidate=candidate, retry_interaction_approach=lambda **kwargs: retries.append(kwargs) or [{"kind": "navigate"}])
    commands = executor._retry_interaction_approach_after_pose_failure_locked({
        "failure_reason": "interaction_orientation_misaligned",
        "reject_selected_face": False,
    })
    assert commands == [{"kind": "navigate"}]
    repaired = executor.machine.candidate
    assert repaired["goal_xyyaw"] == pytest.approx([0.66, 7.69, 1.57])
    assert repaired["metadata"]["container_m1_rejected_face_staging_indices"] == []
    assert repaired["metadata"]["interaction_same_face_yaw_retry_attempted"]
    assert repaired["interaction_command"]["interaction_ready_yaw_tolerance_rad"] == 0.08
    assert executor._interaction_navigation_yaw_tolerance_rad(repaired) == 0.08
    assert len(retries) == 1


def test_same_face_recovery_bypasses_corridor_and_make_plan(executor_module, monkeypatch):
    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module, monkeypatch, profile_detail={}, state_sequence=[0],
        pose_reader=lambda _frame: (1.0, 0.0, 0.2),
    )
    candidate["metadata"]["container_two_stage_phase"] = "physical_action"
    candidate["metadata"]["interaction_same_face_yaw_retry_pending"] = True
    calls = []
    executor._run_same_face_yaw_recovery = lambda *args: calls.append(args) or (True, {})
    executor._start_container_inner_corridor = lambda *_args, **_kwargs: pytest.fail("rotation must not request a corridor")
    executor._preflight_navigation_plan = lambda *_args: pytest.fail("rotation must not request make_plan")
    executor._run_navigation_impl("same-face", candidate, navigation_run_token=1)
    assert len(calls) == 1
    assert executor.move_base.sent_goals == []


def test_same_face_recovery_rotates_with_step_gate_and_keeps_xy_latched(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.map_frame = "map"
    executor._latest_step_sync_index = 10
    candidate = _anchor_candidate("physical_action")
    candidate["metadata"]["interaction_same_face_yaw_retry_pending"] = True
    candidate["interaction_command"].update(interaction_ready_distance_m=0.15, interaction_ready_yaw_tolerance_rad=0.08)
    pose = {"value": (1.0, 0.0, 0.3)}
    executor._current_pose = lambda _frame: pose["value"]
    executor._container_anchor_local_clearance = lambda *_args: (True, {})
    executor._set_effective_interaction_approach = lambda *_args: None
    lease = []
    executor._acquire_rear_goal_cmd_vel_lease = lambda *_args, **_kwargs: lease.append("acquire") or True
    executor._release_rear_goal_cmd_vel_lease = lambda: lease.append("release")
    executor.interaction_final_align_max_control_steps = 56
    executor.interaction_final_align_control_step_margin_steps = 4
    executor.interaction_final_align_rotate_speed_rad_s = 0.3
    executor.interaction_final_align_control_dt_s = 0.2
    executor.interaction_final_align_timeout_s = 15.0
    executor.interaction_final_align_step_sync_stall_timeout_s = 2.0
    executor.interaction_final_align_delivery_retry_steps = 2
    executor.interaction_final_align_post_budget_settle_steps = 3
    executor._interaction_final_align_gate = object()

    def rotate(*args, **kwargs):
        assert args[3] == 0.08
        assert kwargs["step_sync_budget_authoritative"]
        assert kwargs["step_command_gate"] is executor._interaction_final_align_gate
        assert kwargs["command_guard"]()
        assert kwargs["max_prerotate_control_steps"] <= 56
        pose["value"] = (1.3, 0.0, 0.48)
        return True

    executor._rotate_to_yaw = rotate
    completed = []
    executor._complete_interaction_approach_navigation = lambda *_args, **kwargs: completed.append(kwargs)
    assert executor._run_same_face_yaw_recovery("same-face", candidate, (1.0, 0.0, 0.5), 0, [], 1)[0]
    validation = completed[0]["detail"]["interaction_pose_validation"]
    assert validation["valid"] and validation["position_tolerance_latched"]
    assert validation["position_error_m"] == pytest.approx(0.3)
    assert lease == ["acquire", "release"]


def test_position_ready_container_skips_inner_corridor_preflight(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.container_inner_corridor_enabled = True
    executor._current_pose = lambda _frame: (1.0, 0.0, 1.0)
    executor._preflight_navigation_path = lambda *_args: pytest.fail("same-XY goal must not call make_plan")
    candidate = _anchor_candidate("physical_action")
    assert not executor._start_container_inner_corridor("d", candidate, goal_frame="map", final_goal=(1.0, 0.0, 0.0), final_goal_option_index=0, interaction_approach_attempts=[])


def test_latched_arrival_sample_is_not_rejected_by_post_cancel_xy_drift(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    candidate = _anchor_candidate("physical_action")
    candidate["interaction_command"]["interaction_ready_yaw_tolerance_rad"] = 0.08
    goal = (1.0, 0.0, 0.0)
    latch = executor_module.PositionToleranceLatch(goal, 0.15)
    latch.observe((0.86, 0.0, 0.3), 10)
    validation = latch.apply(executor_module.interaction_pose_validation(
        list(goal), [0.70, 0.0, 0.04], distance_tolerance_m=0.15, yaw_tolerance_rad=0.08,
    ))
    sample = executor._container_safe_staging_arrival_sample(candidate, goal, {"interaction_pose_validation": validation})
    assert sample["valid"]
    assert sample["position_tolerance_latched"]
    assert sample["position_error_m"] > 0.15


def test_arrival_rejects_actual_duplicate_even_when_new_goal_is_distinct(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    candidate = _anchor_candidate()
    candidate["metadata"]["container_m1_sampled_capture_poses_xyyaw"] = [[0.0, 0.0, 0.0]]
    executor._container_safe_staging_arrival_sample = lambda *_args: None
    executor._poll_interaction_approach_pose = lambda *_args, **_kwargs: (
        True, {"interaction_pose_validation": {"actual_pose_xyyaw": [0.02, 0.0, 0.0]}}
    )
    executor._retry_interaction_approach = lambda *_args: False
    results = []
    executor._handle_navigation_result = lambda _id, success, detail, **_kwargs: results.append((success, detail))
    executor._complete_interaction_approach_navigation(
        "decision", candidate, selected_goal=(0.3, 0.0, 0.0),
        selected_goal_option_index=1, interaction_approach_attempts=[], goal_option_count=2, detail={},
    )
    assert results[0][0] is False
    assert results[0][1]["reason"] == "container_anchor_viewpoint_duplicate"


def test_drawer_m1_fallback_publishes_sealed_same_capture_scan(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.latest_graph = {"capture_step": 42}
    executor._latest_step_sync_index = 42
    executor._selected_graph_node_locked = lambda _: {
        "is_currently_visible": True,
        "attributes": {"bbox_capture_step": 42, "projected_bbox_2d": [10, 20, 100, 150], "joints": ["private"]},
    }
    candidate = _anchor_candidate("physical_action")
    candidate["metadata"].update({
        "interaction_observation_fallback_used": True,
        "last_interaction_observation": {"failure_reason": "timeout"},
        "container_two_stage_action_geometry_source": "m1_failure_planned_geometry",
    })
    executor.evaluator_opaque_open_only = False
    executor.interaction_command_sequence = 0
    executor.latest_image_sequence = 0
    executor._command_id = lambda _: "drawer-command"
    executor._arm_drawer_scan_execution_wait_locked = lambda _: None
    published = []
    executor.interaction_command_pub = SimpleNamespace(publish=lambda message: published.append(json.loads(message.data)))
    executor._publish_interaction_command(candidate)
    assert len(published) == 1
    assert published[0]["sequence_type"] == "drawer_scan"
    assert published[0]["drawer_container_capture_step"] == 42
    assert published[0]["drawer_container_bbox_2d"] == [10, 20, 100, 150]
    assert published[0]["open_regions"] == []
    assert "private" not in json.dumps(published)
    assert candidate["metadata"]["last_interaction_observation"] == {"failure_reason": "timeout"}


@pytest.mark.parametrize("attributes,visible,reason", [
    ({"bbox_capture_step": 41, "last_observation_frame_index": 42}, True, "drawer_fallback_bbox_capture_not_fresh"),
    ({"bbox_capture_step": None, "last_observation_frame_index": 42}, True, "drawer_fallback_bbox_capture_missing"),
    ({}, True, "drawer_fallback_bbox_capture_missing"),
    ({"bbox_capture_step": 42}, False, "drawer_fallback_visible_bbox_missing"),
])
def test_drawer_fallback_waits_for_missing_stale_or_hidden_bbox(executor_module, attributes, visible, reason):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.latest_graph = {"capture_step": 42}
    executor._latest_step_sync_index = 42
    executor._selected_graph_node_locked = lambda _: {
        "is_currently_visible": visible,
        "attributes": {**attributes, "projected_bbox_2d": [1, 2, 50, 80]},
    }
    candidate = _anchor_candidate("physical_action")
    candidate["metadata"]["interaction_observation_fallback_used"] = True
    rejected = []
    executor._start_drawer_fallback_bbox_wait = lambda _candidate, why: rejected.append(why)
    executor._publish_interaction_command(candidate)
    assert rejected == [reason]


@pytest.mark.parametrize("fresh,owned", [(True, True), (False, True), (True, False)])
def test_drawer_bbox_wait_is_frame_bound_bounded_and_decision_owned(executor_module, monkeypatch, fresh, owned):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor._latest_step_sync_index = 266
    executor.latest_graph = {"capture_step": 264}
    executor.drawer_fallback_bbox_wait_task_steps = 2
    executor._interaction_is_current = lambda _: owned
    executor._selected_graph_node_locked = lambda _: {
        "is_currently_visible": True,
        "attributes": {"bbox_capture_step": executor.latest_graph["capture_step"],
                       "projected_bbox_2d": [475, 340, 526, 420]},
    }
    def hold():
        executor._latest_step_sync_index += 1
        if fresh:
            executor.latest_graph["capture_step"] = 267
    executor._publish_container_staging_hold_stop = hold
    published, rejected = [], []
    executor._publish_interaction_command = published.append
    executor._reject_drawer_command_without_visual_contract = lambda *args, **kw: rejected.append(kw)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)
    context = {"started_step": 266, "started_at": time.monotonic()}
    executor._drawer_fallback_bbox_waits = {"d": context}
    executor._wait_for_drawer_fallback_bbox("d", _anchor_candidate("physical_action"), context)
    assert not executor._drawer_fallback_bbox_waits
    assert len(published) == int(fresh and owned)
    assert rejected == ([{"retryable": True}] if owned and not fresh else [])
    if published:
        assert published[0]["interaction_command"]["drawer_container_capture_step"] == 267


@pytest.mark.parametrize("latched", [False, True])
def test_container_clearance_excludes_arrival_margin_before_and_after_latch(executor_module, latched):
    executor = _local_anchor_executor(executor_module)
    candidate = _anchor_candidate()
    candidate["interaction_command"]["navigation_goal_tolerance_contract_explicit"] = True
    candidate["interaction_command"]["navigation_goal_position_tolerance_m"] = 0.15
    candidate["metadata"]["container_clearance_position_latched"] = latched
    # 0.426 m is outside the footprint safety radius even before arrival.
    executor._latest_occupancy.data[50 * 100 + 58] = 100
    clear, detail = executor._container_anchor_local_clearance(candidate, "map", (0., 0., 0.))
    assert clear and detail["arrival_region_radius_m"] == 0.0
    assert detail["clearance_radius_m"] == pytest.approx(.25 + .05 + .05/math.sqrt(2))
    assert candidate["interaction_command"]["navigation_goal_position_tolerance_m"] == .15


@pytest.mark.parametrize("pose", [(1.1, 2.2, 0.3), None])
def test_observation_request_records_actual_pose_and_clears_stale_pose(executor_module, pose):
    selection = _portal_selection()
    selection["metadata"].update({
        "observation_required": True, "reobserve": True,
        "interaction_observation_source": "mllm_attribute_inference",
        "interaction_observation_actual_pose_xyyaw": [99.0, 99.0, 0.0],
    })
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor.lock = threading.RLock()
    executor.selection = selection
    executor.machine = executor_module.BehaviorExecutionStateMachine()
    commands = executor.machine.start(selection, now=0.0)
    executor.latest_graph = {"capture_step": 42}
    executor._latest_rgb_step_seq = 44
    executor.interaction_observation_sequence = 0
    executor._interaction_observation_requests = {}
    executor._latest_attribute_updates = {}
    executor.tf_listener = object()

    def current_pose(_frame):
        assert executor.lock._is_owned()
        return pose

    executor._current_pose = current_pose
    published = []
    executor.attribute_refresh_request_pub = SimpleNamespace(publish=lambda message: published.append(json.loads(message.data)))
    executor._publish_interaction_observation_request(commands[0])
    metadata = executor.machine.candidate["metadata"]
    if pose is not None:
        assert metadata["interaction_observation_actual_pose_xyyaw"] == list(pose)
        assert published[0]["observation_pose_xyyaw"] == list(pose)
    else:
        assert "interaction_observation_actual_pose_xyyaw" not in metadata


def test_blocked_anchor_preflight_never_calls_make_plan_or_reconfigure(executor_module, monkeypatch):
    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module, monkeypatch, profile_detail={}, state_sequence=[executor_module.GoalStatus.ACTIVE],
        pose_reader=lambda _: None,
    )
    executor.container_anchor_clearance_enabled = True
    executor._container_anchor_local_clearance = lambda *_args: (False, {"reason": "container_anchor_center_blocked"})
    executor._preflight_navigation_plan = lambda *_args: pytest.fail("blocked anchor must skip make_plan")
    executor._activate_container_m1_capture_dwa_profile = lambda *_args, **_kwargs: pytest.fail("blocked anchor must skip reconfigure")
    executor._retry_interaction_approach = lambda *_args, **_kwargs: False
    results = []
    executor._handle_navigation_result = lambda _id, success, detail, **_kwargs: results.append((success, detail))
    executor._run_navigation_impl("decision", candidate, navigation_run_token=1)
    assert not executor.move_base.sent_goals
    assert results and results[0][0] is False


def test_entering_local_window_cancels_before_capture_profile_successor(executor_module, monkeypatch):
    executor, candidate, trace = _direct_m1_capture_navigation_executor(
        executor_module, monkeypatch, profile_detail={}, state_sequence=[executor_module.GoalStatus.ACTIVE],
        pose_reader=lambda _: (0.0, 0.0, 0.0),
    )
    executor.container_anchor_clearance_enabled = True
    clearances = iter([True, True, False])
    executor._container_anchor_local_clearance = lambda *_args: (
        True, {"local_clearance_deferred": next(clearances)}
    )
    executor._activate_container_m1_capture_dwa_profile = lambda *_args, **_kwargs: pytest.fail("transit goal must stop before reconfigure")
    successors = []
    executor._schedule_navigation_successor = lambda *_args: successors.append(executor.move_base.cancel_count)
    executor._run_navigation_impl("decision", candidate, navigation_run_token=1)
    assert len(executor.move_base.sent_goals) == 1
    assert successors == [1]


def test_dwa_restore_is_deferred_while_old_server_goal_is_active(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._move_base_action_name = "/move_base"
    executor._check_move_base_goal_quiescent_before_successor = lambda: (False, {"reason": "recalling"})
    snapshot = {
        "restore_required": True, "restore_configuration": {"xy_goal_tolerance": 0.45},
        "client": SimpleNamespace(update_configuration=lambda _: pytest.fail("active goal cannot be reconfigured")),
    }
    restored, detail = executor._restore_container_m1_capture_dwa_profile_locked(snapshot)
    assert not restored and detail["reason"] == "container_m1_capture_dwa_restore_deferred"


def test_successor_fence_does_not_cancel_another_active_worker(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._move_base_worker_lock = threading.Lock()
    executor._check_move_base_goal_quiescent_before_successor = lambda: pytest.fail("must not cancel another worker")
    with executor._move_base_worker_lock:
        ready, detail = executor._confirm_move_base_goal_quiescent_before_successor()
    assert not ready and detail["reason"] == "move_base_predecessor_worker_active"


def test_run_transport_terminal_suppresses_queued_motion_and_interactions(executor_module):
    executor = object.__new__(executor_module.SemanticBehaviorExecutor)
    executor._move_base_transport_terminal = {"navigation_transport_terminal": True}
    executor._dispatch([
        {"kind": kind} for kind in ("navigate", "interact", "scan", "request_interaction_observation", "publish_drawer_scan")
    ])


def _clearance_portal_candidate():
    from test_portal_approach import portal_candidate
    return portal_candidate()


@pytest.mark.parametrize("obstacle", [100, -1])
def test_portal_clearance_excludes_arrival_margin_but_keeps_footprint(executor_module, obstacle):
    executor = _local_anchor_executor(executor_module)
    candidate = _clearance_portal_candidate()
    executor._latest_occupancy.data[50*100+79] = obstacle
    clear, detail = executor._container_anchor_local_clearance(candidate, "map", (1., 0., math.pi))
    assert clear and detail["arrival_region_radius_m"] == 0.
    assert detail["clearance_radius_m"] == pytest.approx(.25 + .05 + .05/math.sqrt(2))
    executor._latest_occupancy.data[50*100+76] = obstacle
    clear, detail = executor._container_anchor_local_clearance(candidate, "map", (1., 0., math.pi))
    assert not clear
    assert detail["center_cost"] == 0
    assert detail["reason"] == "portal_anchor_arrival_region_blocked"
    shifted = (1., -.1, math.atan2(.1, -1.))
    clear, detail = executor._container_anchor_local_clearance(candidate, "map", shifted)
    assert clear
    assert detail["arrival_region_radius_m"] == 0.


def test_portal_clearance_does_not_double_inflate_99(executor_module):
    executor = _local_anchor_executor(executor_module)
    executor._latest_occupancy.data[50*100+75] = 99
    clear, _ = executor._container_anchor_local_clearance(_clearance_portal_candidate(), "map", (1., 0., math.pi))
    assert clear
    executor._latest_occupancy.data[50*100+70] = 99
    clear, detail = executor._container_anchor_local_clearance(_clearance_portal_candidate(), "map", (1., 0., math.pi))
    assert not clear and detail["reason"] == "portal_anchor_center_blocked"


def test_portal_near_goal_never_uses_global_map_to_bypass_stale_local(executor_module):
    executor = _local_anchor_executor(executor_module)
    executor._latest_occupancy_received_at -= 2
    executor._latest_portal_planning_occupancy = executor._latest_occupancy
    executor._latest_planning_occupancy_received_at = time.monotonic()
    clear, detail = executor._container_anchor_local_clearance(_clearance_portal_candidate(), "map", (1., 0., math.pi))
    assert not clear and detail["reason"] == "portal_anchor_local_costmap_not_fresh"


def test_portal_far_goal_requires_fresh_global_map_and_then_local_confirmation(executor_module):
    import copy
    executor = _local_anchor_executor(executor_module)
    candidate = _clearance_portal_candidate()
    goal = (6., 0., math.pi)
    assert not executor._container_anchor_local_clearance(candidate, "map", goal)[0]
    global_map = copy.deepcopy(executor._latest_occupancy)
    global_map.info.width = 200
    global_map.data = [0]*20000
    executor._latest_portal_planning_occupancy = global_map
    executor._latest_planning_occupancy_received_at = time.monotonic()
    clear, detail = executor._container_anchor_local_clearance(candidate, "map", goal)
    assert clear and detail["local_clearance_deferred"]
    assert detail["costmap_source"] == "planning_occupancy"
    candidate["metadata"].update(portal_clearance_position_latched=True, effective_interaction_approach_pose_xyyaw=list(goal))
    assert not executor._container_anchor_local_clearance(candidate, "map", goal)[0]


def test_portal_latched_clearance_checks_actual_footprint_without_rechecking_xy(executor_module):
    executor = _local_anchor_executor(executor_module)
    candidate = _clearance_portal_candidate()
    candidate["metadata"].update(portal_clearance_position_latched=True, effective_interaction_approach_pose_xyyaw=[1., 0., math.pi])
    actual = (.8, .2, math.pi/2)
    clear, detail = executor._container_anchor_local_clearance(candidate, "map", actual)
    assert clear and detail["arrival_region_radius_m"] == 0
    executor._latest_occupancy.data[53*100+65] = 100
    assert not executor._container_anchor_local_clearance(candidate, "map", actual)[0]


def test_portal_preflight_ranks_all_safe_goals_and_binds_chosen_tolerance(executor_module, monkeypatch):
    executor, _, trace = _direct_m1_capture_navigation_executor(
        executor_module, monkeypatch, profile_detail={}, state_sequence=[3],
        pose_reader=lambda _frame: (-2., 0., 0.),
    )
    candidate = _clearance_portal_candidate()
    goals = [[1., 0., math.pi], [1.2, -.1, math.atan2(.1, -1.2)], [1.5, 0., math.pi]]
    candidate["metadata"]["goal_xyyaw_candidates"] = goals
    checked = []
    def clearance(_candidate, _frame, goal):
        index = goals.index(list(goal))
        checked.append(index)
        return True, {"obstacle_clearance_m": [.5, .7, .6][index], "reason": "portal_anchor_clearance_confirmed"}
    executor._container_anchor_local_clearance = clearance
    bound = []
    executor._set_effective_interaction_approach = lambda *args: bound.append(args)
    class Selected(Exception):
        pass
    def stop_at_prerotate(*args, **kwargs):
        assert (args[2], args[3]) == tuple(goals[1][:2])
        raise Selected()
    executor._prerotate_for_rear_goal = stop_at_prerotate
    with pytest.raises(Selected):
        executor._run_navigation_impl("portal", candidate, start_goal_option_index=0,
                                      interaction_approach_attempts=[], navigation_run_token=1)
    assert checked == [0, 1, 2]
    chosen = bound[0][1]
    assert chosen["interaction_command"]["interaction_approach_pose_xyyaw"] == goals[1]
    assert chosen["interaction_command"]["navigation_goal_position_tolerance_m"] < .15
    assert chosen["interaction_command"]["interaction_approach_axis_xy"] == [1., 0.]
