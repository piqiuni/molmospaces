"""Unit coverage for the GT-free ROS adapter boundary.

These tests intentionally do not import rospy or start MuJoCo.  They validate
the small protocol translation that must remain stable before a live ROS smoke
test is attempted.
"""

from __future__ import annotations

import inspect

from scripts.InteractiveNav.evaluation.benchmark_policies import (
    RosBridgePolicyAdapter,
    build_ros_bridge_policy,
)
from scripts.InteractiveNav.evaluation.benchmark_types import PolicyObservation, PublicEpisode
from scripts.InteractiveNav.evaluation.ros_navigation_factory import (
    CURRENT_ROS_NAVIGATION_FACTORY,
    create_current_ros_navigation_policy,
)


class _FakeRosBridge:
    def __init__(self, response):
        self.response = response
        self.reset_count = 0
        self.closed = False
        self.observation = None

    def reset(self) -> None:
        self.reset_count += 1

    def get_action(self, observation):
        self.observation = observation
        return self.response

    def close(self) -> None:
        self.closed = True


def _public_episode() -> PublicEpisode:
    return PublicEpisode(
        house_index=1,
        scene_dataset="procthor-10k",
        data_split="val",
        instruction="find the target",
        task_type="nav_to_obj",
        camera_names=["head_camera"],
        image_resolution=(640, 480),
    )


def test_current_ros_factory_is_importable_without_ros_runtime() -> None:
    assert CURRENT_ROS_NAVIGATION_FACTORY.endswith(":create_current_ros_navigation_policy")
    assert "public_episode" in inspect.signature(create_current_ros_navigation_policy).parameters


def test_v3_bridge_keeps_cmd_vel_fresh_for_control_step_budgets() -> None:
    source = inspect.getsource(build_ros_bridge_policy)
    assert "require_fresh_cmd_vel=True" in source


def test_ros_bridge_adapter_normalizes_navigation_and_timeout_without_task() -> None:
    live_observation = {"head_camera": object(), "robot_base_pose": [0.0] * 7}
    policy_observation = PolicyObservation(
        observation=live_observation,
        instruction="find the target",
        step_index=3,
        elapsed_seconds=0.6,
        previous_action=None,
    )
    bridge = _FakeRosBridge({"base": [1.0, 2.0, 0.3], "done": False})
    adapter = RosBridgePolicyAdapter(bridge, name="fake_ros")
    adapter.reset(_public_episode())
    action = adapter.act(policy_observation)

    assert bridge.reset_count == 1
    # The bridge receives only the live sensor payload, not the public episode
    # and never a private V3 replay/task object.
    assert bridge.observation is live_observation
    assert action.kind == "base"
    assert action.base_action == {"base": [1.0, 2.0, 0.3]}

    bridge.response = {"done": False}
    timeout_action = adapter.act(policy_observation)
    assert timeout_action.kind == "observe"
    assert timeout_action.metadata["reason"] == "ros_bridge_no_fresh_action"
