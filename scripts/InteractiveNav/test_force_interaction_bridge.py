import math

import numpy as np
import pytest

import force_interaction_bridge as bridge
from force_interaction_bridge import AtomicForceInteractionController


class _Base:
    pose = np.eye(4, dtype=float)


class _RobotView:
    base = _Base()


class _Robot:
    robot_view = _RobotView()


class _Env:
    current_robot = _Robot()


class _Task:
    env = _Env()


class _DrawerEnv(_Env):
    current_model = object()
    current_data = object()


class _DrawerTask:
    env = _DrawerEnv()


def _set_pose(x: float, y: float, yaw: float) -> None:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    _Base.pose = np.asarray(
        [
            [cosine, -sine, 0.0, x],
            [sine, cosine, 0.0, y],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_drawer_scan_defaults_to_every_slide_joint(monkeypatch) -> None:
    controller = AtomicForceInteractionController()
    joints = [
        {"joint_name": f"drawer_{index}", "joint_type": "slide"}
        for index in range(4)
    ]
    monkeypatch.setattr(
        bridge,
        "collect_articulation_groups",
        lambda _env: {"dresser": {"joints": joints}},
    )

    controller._start_drawer_sequence(
        _DrawerTask(),
        {
            "object_id": "dresser",
            "sequence_type": "drawer_scan",
            "open_regions": [
                {"center": [0.5, 0.35]},
                {"center": [0.5, 0.65]},
            ],
        },
        step=10,
    )

    assert len(controller._pending["groups"]) == 4
    assert all(
        group["grounding_source"] == "simulator_all_slide_joints"
        for group in controller._pending["groups"]
    )


def test_interaction_pose_validation_accepts_front_pose() -> None:
    _set_pose(8.20, 1.10, math.pi - 0.05)
    result = AtomicForceInteractionController._validate_interaction_pose(
        _Task(),
        {
            "interaction_approach_pose_xyyaw": [8.25, 1.05, math.pi],
            "interaction_ready_distance_m": 0.45,
            "interaction_ready_yaw_tolerance_rad": 0.55,
        },
    )
    assert result["valid"] is True


def test_interaction_pose_validation_rejects_side_pose_with_feedback_detail() -> None:
    _set_pose(7.00, 2.46, -math.pi / 2.0)
    command = {
        "interaction_approach_pose_xyyaw": [8.25, 1.05, math.pi],
        "interaction_ready_distance_m": 0.45,
        "interaction_ready_yaw_tolerance_rad": 0.55,
    }
    with pytest.raises(ValueError, match="Interaction pose invalid"):
        AtomicForceInteractionController._validate_interaction_pose(_Task(), command)
    assert command["interaction_pose_validation"]["valid"] is False
    assert command["interaction_pose_validation"]["position_error_m"] > 1.0


def test_interaction_pose_validation_rejects_m1_face_normal_mismatch() -> None:
    # The expected pose is deliberately close enough for the legacy distance
    # and yaw checks.  The independent face contract must still reject a
    # robot that arrived on the side of the selected AABB face.
    _set_pose(1.0, 1.35, 0.0)
    command = {
        "interaction_approach_pose_xyyaw": [1.0, 1.35, 0.0],
        "interaction_ready_distance_m": 0.45,
        "interaction_ready_yaw_tolerance_rad": 0.55,
        "interaction_approach_axis_xy": [0.0, 1.0],
        "interaction_target_center_xy": [1.0, 1.0],
        "interaction_front_axis_validation_required": True,
    }
    with pytest.raises(ValueError, match="face_checked=True"):
        AtomicForceInteractionController._validate_interaction_pose(_Task(), command)
    validation = command["interaction_pose_validation"]
    assert validation["face_checked"] is True
    assert validation["face_valid"] is False
    assert validation["face_yaw_error_rad"] > 0.35


def test_interaction_pose_validation_accepts_m1_confirmed_face() -> None:
    _set_pose(1.0, 1.35, -math.pi / 2.0)
    result = AtomicForceInteractionController._validate_interaction_pose(
        _Task(),
        {
            "interaction_approach_pose_xyyaw": [1.0, 1.35, -math.pi / 2.0],
            "interaction_ready_distance_m": 0.45,
            "interaction_ready_yaw_tolerance_rad": 0.55,
            "interaction_approach_axis_xy": [0.0, 1.0],
            "interaction_target_center_xy": [1.0, 1.0],
            "interaction_front_axis_validation_required": True,
        },
    )
    assert result["face_checked"] is True
    assert result["face_valid"] is True


def test_interaction_pose_validation_rejects_m1_face_that_is_not_physical_front(
    monkeypatch,
) -> None:
    _set_pose(1.0, 1.35, -math.pi / 2.0)
    monkeypatch.setattr(
        bridge,
        "infer_articulation_front_axis_xy",
        lambda _env, _object_id: {
            "checked": True,
            "axis_xy": [1.0, 0.0],
            "source": "slide_open_travel",
        },
    )
    command = {
        "node_type": "container",
        "object_id": "drawer_1",
        "_execution_object_id": "private_drawer_1",
        "interaction_approach_pose_xyyaw": [1.0, 1.35, -math.pi / 2.0],
        "interaction_ready_distance_m": 0.45,
        "interaction_ready_yaw_tolerance_rad": 0.55,
        "interaction_approach_axis_xy": [0.0, 1.0],
        "interaction_target_center_xy": [1.0, 1.0],
        "interaction_front_axis_validation_required": True,
    }

    with pytest.raises(ValueError, match="Interaction physical front invalid"):
        AtomicForceInteractionController._validate_interaction_pose(_Task(), command)

    validation = command["interaction_pose_validation"]
    assert validation["face_valid"] is True
    assert validation["physical_front_checked"] is True
    assert validation["physical_front_valid"] is False


def _enqueue_fridge_open(controller: AtomicForceInteractionController) -> None:
    controller.enqueue_command(
        {
            "command_id": "fridge-open-1",
            "object_id": "public-fridge-1",
            "node_id": "container-1",
            "node_type": "container",
            "container_kind": "fridge",
            "action": "open",
        }
    )


def _stub_interaction_setup(
    monkeypatch: pytest.MonkeyPatch,
    controller: AtomicForceInteractionController,
) -> None:
    monkeypatch.setattr(
        controller,
        "_validate_interaction_pose",
        lambda _task, _command: {"valid": True},
    )
    monkeypatch.setattr(
        controller._head_view_controller,
        "command",
        lambda *_args, **_kwargs: {"applied": False},
    )
    monkeypatch.setattr(
        bridge,
        "prepare_articulation_force",
        lambda *_args, **_kwargs: {"supported": True},
    )


def test_unsafe_open_sweep_bypass_runs_real_backend_and_audits_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AtomicForceInteractionController(bypass_unsafe_open_sweep=True)
    _stub_interaction_setup(monkeypatch, controller)
    monkeypatch.setattr(
        bridge,
        "_refrigerator_open_sweep_preflight",
        lambda *_args, **_kwargs: pytest.fail("bypassed sweep must not run"),
    )
    backend_calls = []

    def _complete(*_args, **_kwargs):
        backend_calls.append(True)
        # A bypass must not turn a real backend failure into success.
        return {
            "success": False,
            "pre_state": "closed",
            "post_state": "closed",
            "physics_substeps": 7,
            "task_steps_consumed": 1,
        }

    monkeypatch.setattr(bridge, "complete_articulation_force", _complete)
    _enqueue_fridge_open(controller)

    assert controller.before_step(_Task(), step=10) is None
    assert controller._pending is not None
    result = controller.after_step(_Task(), step=10)

    assert backend_calls == [True]
    assert result is not None
    assert result["success"] is False
    assert result["failure_reason"] == "force_target_not_reached"
    assert result["open_sweep_preflight_bypassed"] is True


def test_unsafe_open_sweep_still_rejects_when_bypass_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AtomicForceInteractionController(bypass_unsafe_open_sweep=False)
    _stub_interaction_setup(monkeypatch, controller)
    monkeypatch.setattr(
        bridge,
        "_refrigerator_open_sweep_preflight",
        lambda *_args, **_kwargs: {
            "checked": True,
            "safe": False,
            "reason": "unsafe_open_sweep",
            "recommended_retreat_m": 0.25,
        },
    )
    monkeypatch.setattr(
        bridge,
        "complete_articulation_force",
        lambda *_args, **_kwargs: pytest.fail("unsafe sweep must reject before force"),
    )
    _enqueue_fridge_open(controller)

    result = controller.before_step(_Task(), step=11)

    assert result is not None
    assert result["success"] is False
    assert result["failure_reason"] == "unsafe_open_sweep"
    assert result["physics_substeps"] == 0
    assert controller._pending is None
