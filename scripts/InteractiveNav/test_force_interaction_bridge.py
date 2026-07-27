import math

import numpy as np
import pytest

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
