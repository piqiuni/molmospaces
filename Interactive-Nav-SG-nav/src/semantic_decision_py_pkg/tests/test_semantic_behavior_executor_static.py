from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    _stub_module(monkeypatch, "geometry_msgs.msg", PoseStamped=_Placeholder, Twist=_Placeholder)
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
