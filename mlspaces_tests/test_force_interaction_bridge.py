from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav import force_interaction_runtime
from scripts.InteractiveNav.force_interaction_bridge import (
    AtomicForceInteractionController,
    ground_drawer_open_regions,
)


def test_controller_emits_success_result_and_behavior_feedback(monkeypatch) -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    published = []
    prepare_calls = []

    def prepare(_env, root_name, **kwargs):
        prepare_calls.append((root_name, kwargs))
        return {
            "group": {"root_body_name": root_name},
            "targets": {"left_hinge": 1.0},
            "selected_joint_names": ["left_hinge"],
            "closed_joint_names": [],
            "pre_joint_infos": [],
        }

    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_force",
        prepare,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.complete_articulation_force",
        lambda _env, _plan, config: {
            "pre_state": "closed",
            "post_state": "open",
            "joint_infos": [],
            "selected_joint_names": ["left_hinge"],
            "closed_joint_names": [],
            "success": True,
            "physics_substeps": 123,
            "task_steps_consumed": 1,
        },
    )
    monkeypatch.setattr(controller, "_publish", lambda publisher, payload: published.append(payload))

    assert controller.enqueue_command(
        {
            "command_id": "command_1",
            "candidate_id": "portal_1",
            "decision_id": "decision_1",
            "object_id": "double_door_root",
            "action": "open",
            "joint_names": ["planner_must_not_select_this"],
            "interaction_groups": [{"group_id": "forbidden"}],
        }
    )
    task = SimpleNamespace(env=SimpleNamespace(current_model=object(), current_data=object()))
    assert controller.before_step(task, step=17) is None
    result = controller.after_step(task, step=17)

    assert result is not None
    assert result["success"] is True
    assert result["status"] == "SUCCEEDED"
    assert result["sim_steps_consumed"] == 1
    assert result["physics_substeps"] == 123
    assert result["object_id"] == "double_door_root"
    assert prepare_calls == [("double_door_root", {})]
    assert "joint_names" not in result
    assert "joint_infos" not in result
    assert "force_result" not in result
    assert len(published) == 2
    assert published[0]["source"] == "force_atomic_interaction"
    assert published[1]["behavior_type"] == "INTERACT"
    assert published[1]["status"] == "SUCCEEDED"


def test_controller_deduplicates_command_ids() -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    command = {
        "command_id": "same_command",
        "object_id": "door_root",
        "action": "open",
    }
    assert controller.enqueue_command(command)
    assert not controller.enqueue_command(command)


def test_non_articulated_object_returns_static_failure_without_crashing(
    monkeypatch,
) -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    published = []
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_force",
        lambda _env, root_name, **_kwargs: {
            "supported": False,
            "reason": "non_articulated",
            "interaction_capability": "static",
            "object_name": root_name,
        },
    )
    monkeypatch.setattr(
        controller, "_publish", lambda _publisher, payload: published.append(payload)
    )
    assert controller.enqueue_command(
        {
            "command_id": "static_doorframe",
            "candidate_id": "portal_doorframe",
            "node_id": "portal_doorframe",
            "object_id": "doorframe_static_1",
            "action": "open",
        }
    )

    result = controller.before_step(
        SimpleNamespace(env=SimpleNamespace()), step=9
    )

    assert result is not None
    assert result["status"] == "FAILED"
    assert result["reason"] == "non_articulated"
    assert result["interaction_capability"] == "static"
    assert result["interactable"] is False
    assert len(published) == 2
    assert published[1]["interaction_result"] == result


def test_non_articulated_portal_is_marked_static_open_from_graph_type(
    monkeypatch,
) -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    published = []
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_force",
        lambda _env, root_name, **_kwargs: {
            "supported": False,
            "reason": "non_articulated",
            "interaction_capability": "static",
            "object_name": root_name,
        },
    )
    monkeypatch.setattr(
        controller, "_publish", lambda _publisher, payload: published.append(payload)
    )
    assert controller.enqueue_command(
        {
            "command_id": "static_portal",
            "candidate_id": "portal_candidate",
            "node_id": "graph_node_17",
            "node_type": "portal",
            "object_id": "visual_instance_17",
            "action": "open",
        }
    )

    result = controller.before_step(SimpleNamespace(env=SimpleNamespace()), step=10)

    assert result is not None
    assert result["status"] == "SUCCEEDED"
    assert result["success"] is True
    assert result["post_state"] == "static_open"
    assert result["source"] == "executor_static_portal"
    assert published[1]["status"] == "SUCCEEDED"


def test_prepare_articulation_force_reports_missing_group_as_static(monkeypatch) -> None:
    monkeypatch.setattr(
        force_interaction_runtime, "collect_articulation_groups", lambda _env: {}
    )

    plan = force_interaction_runtime.prepare_articulation_force(
        SimpleNamespace(), "doorframe_static_1"
    )

    assert plan == {
        "supported": False,
        "reason": "non_articulated",
        "interaction_capability": "static",
        "object_name": "doorframe_static_1",
        "available_object_names": [],
    }

def test_controller_requires_canonical_object_id() -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    with pytest.raises(ValueError, match="requires object_id"):
        controller.enqueue_command(
            {"command_id": "legacy", "source_object_name": "door_root"}
        )


def test_missing_portal_articulation_is_marked_static_open(monkeypatch) -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    published = []
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_force",
        lambda _env, object_id: (_ for _ in ()).throw(
            ValueError(f"Articulated object not found: {object_id}")
        ),
    )
    monkeypatch.setattr(
        controller,
        "_publish",
        lambda _publisher, payload: published.append(payload),
    )
    controller._head_view_controller.command = lambda *_args, **_kwargs: {"applied": True}
    controller._head_view_controller.restore = lambda *_args, **_kwargs: {"applied": True}
    assert controller.enqueue_command(
        {
            "command_id": "missing_door",
            "candidate_id": "interaction:missing_door:open",
            "decision_id": "decision_missing",
            "node_id": "portal_doorframe_without_joint",
            "object_id": "doorframe_without_joint",
            "node_type": "portal",
            "action": "open",
        }
    )
    task = SimpleNamespace(env=SimpleNamespace(current_model=object(), current_data=object()))

    result = controller.before_step(task, step=210)

    assert result is not None
    assert result["status"] == "SUCCEEDED"
    assert result["success"] is True
    assert result["post_state"] == "static_open"
    assert result["source"] == "executor_static_portal"
    assert result["verification_source"] == "simulator_no_articulation"
    assert result["object_id"] == "doorframe_without_joint"
    assert controller.should_pause_navigation() is False
    assert controller.after_step(task, step=210) is None
    assert len(published) == 2
    assert published[1]["behavior_type"] == "INTERACT"
    assert published[1]["status"] == "SUCCEEDED"


def test_missing_container_articulation_remains_a_recoverable_failure(monkeypatch) -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    published = []
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_force",
        lambda _env, object_id: (_ for _ in ()).throw(
            ValueError(f"Articulated object not found: {object_id}")
        ),
    )
    monkeypatch.setattr(
        controller,
        "_publish",
        lambda _publisher, payload: published.append(payload),
    )
    controller._head_view_controller.command = lambda *_args, **_kwargs: {"applied": True}
    controller._head_view_controller.restore = lambda *_args, **_kwargs: {"applied": True}
    assert controller.enqueue_command(
        {
            "command_id": "missing_container",
            "candidate_id": "interaction:missing_container:open",
            "decision_id": "decision_missing_container",
            "node_id": "container_missing",
            "object_id": "cabinet_without_joint",
            "node_type": "container",
            "action": "open",
        }
    )
    task = SimpleNamespace(env=SimpleNamespace(current_model=object(), current_data=object()))

    result = controller.before_step(task, step=211)

    assert result is not None
    assert result["status"] == "FAILED"
    assert result["success"] is False
    assert result["post_state"] == "unknown"
    assert result["failure_reason"] == "articulation_resolution_failed"
    assert result["verification_source"] == "executor_resolution_failure"
    assert controller.should_pause_navigation() is False
    assert len(published) == 2


def test_visual_drawer_regions_are_grounded_to_slide_joints_by_height() -> None:
    groups = ground_drawer_open_regions(
        [
            {"joint_name": "middle", "joint_type": "slide"},
            {"joint_name": "door_hinge", "joint_type": "hinge"},
            {"joint_name": "bottom", "joint_type": "slide"},
            {"joint_name": "top", "joint_type": "slide"},
        ],
        [
            {"center": [0.51, 0.82], "confidence": 0.9},
            {"center": [0.49, 0.18], "confidence": 0.8},
        ],
        {"top": 1.0, "middle": 0.6, "bottom": 0.2},
    )

    assert [group["joint_names"] for group in groups] == [["top"], ["bottom"]]
    assert [group["open_region"]["center"][1] for group in groups] == [0.18, 0.82]
    assert all(group["grounding_source"] == "visual_region_vertical_order" for group in groups)


def test_empty_visual_regions_fall_back_to_all_slide_joints() -> None:
    groups = ground_drawer_open_regions(
        [
            {"joint_name": "bottom", "joint_type": "slide"},
            {"joint_name": "top", "joint_type": "slide"},
        ],
        [],
        {"top": 1.0, "bottom": 0.2},
    )

    assert [group["joint_names"] for group in groups] == [["top"], ["bottom"]]
    assert all(group["grounding_source"] == "simulator_all_slide_joints" for group in groups)


def test_controller_discovers_drawer_joints_for_visual_plan(monkeypatch) -> None:
    joints = [
        {"joint_name": "bottom", "joint_type": "slide", "joint_id": 0},
        {"joint_name": "top", "joint_type": "slide", "joint_id": 1},
        {"joint_name": "middle", "joint_type": "slide", "joint_id": 2},
    ]
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.collect_articulation_groups",
        lambda _env: {"dresser_root": {"joints": joints}},
    )
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    model = SimpleNamespace(jnt_bodyid=[0, 1, 2])
    data = SimpleNamespace(xpos=[[0.0, 0.0, 0.2], [0.0, 0.0, 1.0], [0.0, 0.0, 0.6]])
    task = SimpleNamespace(env=SimpleNamespace(current_model=model, current_data=data))
    command = {
        "command_id": "visual_drawer_scan",
        "object_id": "dresser_root",
        "action": "scan",
        "sequence_type": "drawer_scan",
        "approach_goal_xyyaw": [1.0, 2.0, 0.5],
        "open_regions": [
            {"center": [0.5, 0.18], "confidence": 0.9},
            {"center": [0.5, 0.52], "confidence": 0.8},
            {"center": [0.5, 0.83], "confidence": 0.9},
        ],
    }

    controller._start_drawer_sequence(task, command, step=12)

    assert [group["joint_names"] for group in controller._pending["groups"]] == [
        ["top"],
        ["middle"],
        ["bottom"],
    ]
    assert controller._pending["all_joint_names"] == ["top", "middle", "bottom"]

def test_smooth_door_or_fridge_interaction_uses_task_steps_without_low_view(
    monkeypatch,
) -> None:
    advances = []
    view_profiles = []
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_force",
        lambda _env, root_name, **_kwargs: {
            "group": {"root_body_name": root_name, "joints": []},
            "targets": {"hinge": 1.0},
            "selected_joint_names": ["hinge"],
            "closed_joint_names": [],
            "pre_joint_infos": [{"joint_name": "hinge", "joint_value": 0.0}],
        },
    )

    def advance(_env, _plan, progress, **_kwargs):
        advances.append(progress)
        return {
            "progress": progress,
            "physics_substeps": 2,
            "fallback": False,
        }

    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.advance_articulation_force",
        advance,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.finalize_articulation_force_transition",
        lambda _env, _plan, **kwargs: {
            "pre_state": "closed",
            "post_state": "open",
            "joint_infos": [],
            "selected_joint_names": ["hinge"],
            "closed_joint_names": [],
            "success": True,
            "physics_substeps": kwargs["physics_substeps"],
            "task_steps_consumed": kwargs["task_steps_consumed"],
        },
    )
    controller = AtomicForceInteractionController(
        close_all_doors_on_prepare=False,
        interaction_execution_mode="smooth",
        interaction_transition_steps=3,
    )
    controller._head_view_controller.command = (
        lambda _env, profile, **_kwargs: view_profiles.append(profile) or {"applied": True}
    )
    controller._publish = lambda *_args, **_kwargs: None
    assert controller.enqueue_command(
        {
            "command_id": "smooth_fridge",
            "object_id": "fridge_root",
            "action": "open",
            "view_profile": "default",
        }
    )
    task = SimpleNamespace(env=SimpleNamespace(current_model=object(), current_data=object()))

    result = None
    for step in range(3):
        controller.before_step(task, step=step)
        result = controller.after_step(task, step=step)

    assert result is not None
    assert advances == [1.0 / 3.0, 2.0 / 3.0, 1.0]
    assert view_profiles == ["default"]
    assert result["interaction_execution_mode"] == "smooth"
    assert result["interaction_transition_steps"] == 3
    assert result["task_steps_consumed"] == 3
    assert result["source"] == "force_smooth_interaction"


def test_drawer_scan_fast_mode_combines_transitions_and_observations(monkeypatch) -> None:
    state = {"drawer_top": 0.0, "drawer_bottom": 0.0}
    published = []
    prepared_targets = []
    lock_calls = []
    joints = [
        {"joint_name": "drawer_top", "joint_type": "slide", "joint_id": 0},
        {"joint_name": "drawer_bottom", "joint_type": "slide", "joint_id": 1},
    ]

    def prepare(_env, _root, open_joint_names=None, close_joint_names=None):
        prepared_targets.append(
            (tuple(open_joint_names or []), tuple(close_joint_names or []))
        )
        targets = {
            name: 1.0 for name in open_joint_names or []
        }
        targets.update({name: 0.0 for name in close_joint_names or []})
        return {
            "group": {"joints": []},
            "targets": targets,
            "selected_joint_names": list(open_joint_names or []),
            "closed_joint_names": list(close_joint_names or []),
            "pre_joint_infos": [
                {"joint_name": name, "joint_value": state[name]}
                for name in targets
            ],
        }

    def advance(_env, plan, **_kwargs):
        callback = _kwargs.get("robot_lock_callback")
        if callback is not None:
            callback()
        state.update(plan["targets"])
        return {"physics_substeps": 1, "fallback": False}

    def infos(_env, _root):
        return [
            {"joint_name": name, "open_fraction": value, "joint_value": value}
            for name, value in state.items()
        ]

    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_state_force",
        prepare,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.advance_articulation_force",
        advance,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.articulation_joint_infos",
        infos,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.collect_articulation_groups",
        lambda _env: {"dresser_root": {"joints": joints}},
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge._capture_robot_lock",
        lambda _env: {"locked": True},
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge._apply_robot_lock",
        lambda _env, snapshot: lock_calls.append(snapshot),
    )
    controller = AtomicForceInteractionController(
        close_all_doors_on_prepare=False,
        drawer_execution_mode="fast",
        drawer_observation_steps=3,
    )
    controller._head_view_controller.command = lambda *_args, **_kwargs: {"applied": True}
    controller._head_view_controller.restore = lambda *_args, **_kwargs: {"applied": True}
    monkeypatch.setattr(controller, "_publish", lambda _publisher, payload: published.append(payload))
    command = {
        "command_id": "drawer_scan_fast",
        "candidate_id": "drawer_scan",
        "decision_id": "decision_1",
        "object_id": "dresser_root",
        "action": "scan",
        "sequence_type": "drawer_scan",
        "approach_goal_xyyaw": [1.0, 2.0, 0.5],
        "open_regions": [
            {"center": [0.5, 0.2]},
            {"center": [0.5, 0.8]},
        ],
        "interaction_groups": [
            {"group_id": "bad", "joint_names": ["planner_must_not_select_this"]},
        ],
    }
    assert controller.enqueue_command(command)
    model = SimpleNamespace(jnt_bodyid=[0, 1])
    data = SimpleNamespace(xpos=[[0.0, 0.0, 1.0], [0.0, 0.0, 0.2]])
    task = SimpleNamespace(env=SimpleNamespace(current_model=model, current_data=data))

    for step in range(10):
        controller.before_step(task, step=step)
        result = controller.after_step(task, step=step)

    assert result is not None
    assert result["success"] is True
    assert result["task_steps_consumed"] == 10
    assert result["drawer_execution_mode"] == "fast"
    assert result["drawer_observation_steps"] == 3
    assert result["approach_goal_xyyaw"] == [1.0, 2.0, 0.5]
    assert [item["observation_step"] for item in result["region_results"]] == [3, 8]
    assert "interaction_group_results" not in result
    assert "joint_names" not in result
    assert "joint_infos" not in result
    assert result["source"] == "force_container_sequence"
    assert len(published) == 2
    # A completed close phase now separates the two drawers.  In particular,
    # the bottom drawer is never opened in the same transition that closes the
    # top one, even in the low-latency ``fast`` mode.
    assert prepared_targets == [
        (("drawer_top",), ()),
        ((), ("drawer_top",)),
        (("drawer_bottom",), ()),
        ((), ("drawer_bottom",)),
    ]
    # The lock is applied before/after each phase and passed down to the force
    # routine so real MuJoCo substeps also reassert the robot pose.
    assert len(lock_calls) >= 12


def test_drawer_scan_smooth_mode_uses_configured_transition_steps(monkeypatch) -> None:
    state = {"drawer_top": 0.0, "drawer_bottom": 0.0}
    joints = [
        {"joint_name": "drawer_top", "joint_type": "slide", "joint_id": 0},
        {"joint_name": "drawer_bottom", "joint_type": "slide", "joint_id": 1},
    ]

    def prepare(_env, _root, open_joint_names=None, close_joint_names=None):
        targets = {name: 1.0 for name in open_joint_names or []}
        targets.update({name: 0.0 for name in close_joint_names or []})
        return {
            "group": {"joints": []},
            "targets": targets,
            "selected_joint_names": list(open_joint_names or []),
            "closed_joint_names": list(close_joint_names or []),
            "pre_joint_infos": [
                {"joint_name": name, "joint_value": state[name]}
                for name in targets
            ],
        }

    def advance(_env, plan, progress, **_kwargs):
        for name, target in plan["targets"].items():
            state[name] = target if progress >= 1.0 else state[name]
        return {"physics_substeps": 1, "fallback": False}

    def infos(_env, _root):
        return [
            {"joint_name": name, "open_fraction": value, "joint_value": value}
            for name, value in state.items()
        ]

    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_state_force",
        prepare,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.advance_articulation_force",
        advance,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.articulation_joint_infos",
        infos,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.collect_articulation_groups",
        lambda _env: {"dresser_root": {"joints": joints}},
    )
    controller = AtomicForceInteractionController(
        close_all_doors_on_prepare=False,
        drawer_execution_mode="smooth",
        drawer_transition_steps=3,
    )
    controller._head_view_controller.command = lambda *_args, **_kwargs: {"applied": True}
    controller._head_view_controller.restore = lambda *_args, **_kwargs: {"applied": True}
    controller._publish = lambda *_args, **_kwargs: None
    assert controller.enqueue_command(
        {
            "command_id": "drawer_scan_smooth",
                "object_id": "dresser_root",
                "action": "scan",
                "sequence_type": "drawer_scan",
                "open_regions": [
                    {"center": [0.5, 0.2]},
                    {"center": [0.5, 0.8]},
                ],
            }
    )
    model = SimpleNamespace(jnt_bodyid=[0, 1])
    data = SimpleNamespace(xpos=[[0.0, 0.0, 1.0], [0.0, 0.0, 0.2]])
    task = SimpleNamespace(env=SimpleNamespace(current_model=model, current_data=data))
    result = None
    for step in range(30):
        controller.before_step(task, step=step)
        result = controller.after_step(task, step=step)
        if result is not None:
            break

    assert result is not None
    assert result["success"] is True
    assert result["drawer_transition_steps"] == 3
    assert result["task_steps_consumed"] == 14


def test_drawer_scan_reconciles_rebound_after_final_task_step(monkeypatch) -> None:
    state = {"drawer_top": 0.0, "drawer_bottom": 0.8}
    prepared_close_names = []
    published = []
    command = {
        "command_id": "drawer_rebound",
        "candidate_id": "drawer_scan",
        "decision_id": "decision_1",
        "object_id": "dresser_root",
        "action": "scan",
        "sequence_type": "drawer_scan",
    }

    def infos(_env, _root):
        return [
            {"joint_name": name, "open_fraction": value, "joint_value": value}
            for name, value in state.items()
        ]

    def prepare(_env, _root, close_joint_names=None, **_kwargs):
        names = list(close_joint_names or [])
        prepared_close_names.append(tuple(names))
        return {
            "group": {"joints": []},
            "targets": {name: 0.0 for name in names},
            "selected_joint_names": [],
            "closed_joint_names": names,
            "pre_joint_infos": infos(None, None),
        }

    def complete(_env, plan, **_kwargs):
        for name in plan["closed_joint_names"]:
            state[name] = 0.0
        return {
            # Close-only force plans historically report false here because
            # they have no open-selection.  The bridge must verify simulator
            # state instead of trusting that summary flag.
            "success": False,
            "atomic_fallback": True,
            "physics_substeps": 7,
        }

    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.articulation_joint_infos",
        infos,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_state_force",
        prepare,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.complete_articulation_force",
        complete,
    )
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    controller._head_view_controller.restore = lambda _env: {"applied": True}
    monkeypatch.setattr(
        controller,
        "_publish",
        lambda _publisher, payload: published.append(payload),
    )
    controller._commands.put(command)
    controller._commands.get_nowait()
    controller._pending = {
        "kind": "drawer_sequence",
        "command": command,
        "step": 10,
        "phase": "close",
        "phase_step": 1,
        "group_index": 1,
        "groups": [
            {"group_id": "top", "joint_names": ["drawer_top"]},
            {"group_id": "bottom", "joint_names": ["drawer_bottom"]},
        ],
        "all_joint_names": ["drawer_top", "drawer_bottom"],
        "transition_steps": 1,
        "observation_steps": 1,
        "group_results": [
            {"region_id": "top", "success": True},
            {"region_id": "bottom", "success": True},
        ],
        "transition_log": [],
        "physics_substeps": 20,
        "view_result": {"applied": True},
        "view_restore_result": None,
        "robot_lock_snapshot": None,
    }
    task = SimpleNamespace(env=SimpleNamespace())

    result = controller._advance_drawer_sequence_after_step(task, step=11)

    assert result is not None
    assert result["status"] == "SUCCEEDED"
    assert result["final_close_success"] is True
    assert result["final_close_recovery"] == {
        "attempted": True,
        "success": True,
        "physics_substeps": 7,
        "atomic_fallback": True,
    }
    assert prepared_close_names == [("drawer_top", "drawer_bottom")]
    assert result["physics_substeps"] == 27
    assert result["transition_log"][-1]["phase"] == "final_close_recovery"
    assert len(published) == 2


def test_drawer_close_verification_requires_every_selected_joint() -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)

    assert not controller._drawer_close_is_confirmed(
        [{"joint_name": "drawer_top", "open_fraction": 0.0}],
        ["drawer_top", "drawer_bottom"],
    )


def test_drawer_scan_failure_best_effort_closes_and_restores_view(monkeypatch) -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    command = {
        "command_id": "drawer_failure",
        "object_id": "dresser_root",
        "action": "scan",
        "sequence_type": "drawer_scan",
    }
    # Mirror a command already taken from the queue by ``before_step``.
    controller._commands.put(command)
    controller._commands.get_nowait()
    controller._pending = {
        "kind": "drawer_sequence",
        "command": command,
        "all_joint_names": ["drawer_top", "drawer_bottom"],
        "robot_lock_snapshot": None,
        "view_result": {"applied": True},
    }
    prepared_close_names = []
    published = []

    def prepare(_env, _root, close_joint_names=None, **_kwargs):
        prepared_close_names.append(tuple(close_joint_names or []))
        return {"targets": {name: 0.0 for name in close_joint_names or []}}

    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_state_force",
        prepare,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.complete_articulation_force",
        lambda *_args, **_kwargs: {"success": True},
    )
    controller._head_view_controller.restore = lambda _env: {"restored": True}
    monkeypatch.setattr(controller, "_publish", lambda _publisher, payload: published.append(payload))

    result = controller._abort_drawer_sequence(
        SimpleNamespace(env=SimpleNamespace()),
        step=9,
        exc=RuntimeError("render failed"),
    )

    assert prepared_close_names == [("drawer_top", "drawer_bottom")]
    assert result["status"] == "FAILED"
    assert result["failure_reason"] == "drawer_scan_execution_failed"
    assert result["view_restore_result"] == {"restored": True}
    assert controller._pending is None
    assert controller.should_pause_navigation() is False
    assert len(published) == 2
