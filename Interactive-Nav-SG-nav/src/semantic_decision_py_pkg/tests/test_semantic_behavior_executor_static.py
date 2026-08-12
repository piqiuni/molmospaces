from __future__ import annotations

import base64
import importlib.util
import json
import math
import sys
import threading
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
            },
        ),
    )
    _stub_module(monkeypatch, "geometry_msgs")
    _stub_module(
        monkeypatch,
        "geometry_msgs.msg",
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
    assert evidence["pose_validation"]["valid"] is True
    assert executor._container_m1_evidence_still_at_capture_pose_locked(
        candidate, evidence
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
    assert commands[0]["start_goal_option_index"] == 1
    assert executor.machine.candidate["metadata"]["container_two_stage_phase"] == (
        "staging"
    )
    assert executor.machine.candidate["interaction_command"][
        "interaction_ready_distance_m"
    ] == pytest.approx(0.30)
    assert executor._container_m1_last_accepted_evidence == {}


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
    # A later re-evaluation retains the selected side even though both arcs
    # remain geometrically equivalent.
    choice_again, _detail_again = executor._rear_goal_rotation_choice(
        "decision-rear", "map", (-1.0, 1.0)
    )
    assert choice_again is not None
    assert choice_again["turn_sign"] == choice["turn_sign"]


def test_rear_turn_uses_finite_wrapped_side_when_short_turn_is_blocked(
    monkeypatch, executor_module
) -> None:
    """A costmap-safe full alternate turn must not be rejected at the pi cap."""

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
    # Enough for a wrapped 2*pi fallback, while still a strict finite budget.
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

    assert choice is not None
    assert choice["direction"] == "ccw"
    assert choice["required_control_steps"] > 14
    assert choice["required_control_steps"] <= 28
    assert detail["selected_turn"] == "ccw"


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
        header=SimpleNamespace(seq=37),
        encoding="bgr8",
        height=1,
        width=1,
        step=3,
        data=bytes((10, 20, 30)),
    )

    executor._image_callback(image)
    assert executor._rear_goal_prerotate_gate.consume_step() is None
    executor._fresh_command_gate_callback(
        SimpleNamespace(data=json.dumps({"step_index": 37}))
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
            ready_but_misaligned,
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
    assert executor.move_base.cancel_waits == [0.0]
    assert len(rotate_calls) == 1
    rotate_args, rotate_kwargs = rotate_calls[0]
    assert rotate_args[2] == pytest.approx(math.pi)
    assert rotate_args[3] == pytest.approx(0.15)
    assert rotate_kwargs["step_command_gate"] is executor._interaction_final_align_gate
    assert rotate_kwargs["rotation_label"] == "interaction-final-align"
    assert rotate_kwargs["max_prerotate_control_steps"] == 12
    assert rotate_kwargs["step_sync_budget_authoritative"] is True
    assert len(completed) == 1
    _args, kwargs = completed[0]
    assert kwargs["selected_goal"] == tuple(selected_fallback)
    assert kwargs["detail"]["reason"] == "interaction_approach_final_yaw_alignment"
    before = kwargs["detail"]["interaction_pose_validation_before_final_align"]
    assert before["position_error_m"] < before["distance_tolerance_m"]
    assert before["yaw_error_rad"] > before["yaw_tolerance_rad"]
    assert before["valid"] is False


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
