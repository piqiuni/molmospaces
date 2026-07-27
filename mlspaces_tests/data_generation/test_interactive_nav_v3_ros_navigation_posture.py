"""Regression coverage for the ROS-only V3 navigation reset posture."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.InteractiveNav.evaluation.benchmark_runner import (
    BenchmarkEvaluationConfig,
    ROS_NAVIGATION_ARM_QPOS,
    _apply_ros_navigation_arm_posture,
    _build_replay_config,
)


def test_ros_navigation_posture_replaces_only_supported_arm_groups() -> None:
    spec = SimpleNamespace(
        robot=SimpleNamespace(
            init_qpos={
                "left_arm": [9.0] * 7,
                "right_arm": [8.0] * 7,
                "head": [0.0, 0.0],
            }
        )
    )

    _apply_ros_navigation_arm_posture(spec)

    assert spec.robot.init_qpos["left_arm"] == list(ROS_NAVIGATION_ARM_QPOS["left_arm"])
    assert spec.robot.init_qpos["right_arm"] == list(ROS_NAVIGATION_ARM_QPOS["right_arm"])
    assert spec.robot.init_qpos["head"] == [0.0, 0.0]


def test_ros_navigation_posture_does_not_add_unsupported_arm_groups() -> None:
    spec = SimpleNamespace(robot=SimpleNamespace(init_qpos={"head": [0.0, 0.0]}))

    _apply_ros_navigation_arm_posture(spec)

    assert spec.robot.init_qpos == {"head": [0.0, 0.0]}


def test_ros_replay_config_uses_the_same_noise_free_navigation_posture() -> None:
    config = BenchmarkEvaluationConfig(
        benchmark=Path("benchmark.json"),
        output_dir=Path("output"),
        policy="ros_bridge",
    )

    replay = _build_replay_config(config, Path("output"))

    for name, qpos in ROS_NAVIGATION_ARM_QPOS.items():
        np.testing.assert_array_equal(replay.robot_config.init_qpos[name], np.asarray(qpos, dtype=float))
        np.testing.assert_array_equal(replay.robot_config.init_qpos_noise_range[name], np.zeros(len(qpos)))
