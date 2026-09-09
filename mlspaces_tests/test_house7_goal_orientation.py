from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav import force_interaction_runtime
from scripts.InteractiveNav import run_nav_ros_sim
from scripts.InteractiveNav.run_house7_goal_orientation import (
    angle_error,
    build_orientation_goal,
    normalize_angle,
)
from scripts.InteractiveNav.run_nav_ros_sim import apply_initial_door_state


def test_orientation_goal_reverses_the_frozen_route_heading() -> None:
    route = {"far_goal_xyyaw": [7.0, 3.0, -0.25]}
    goal = build_orientation_goal(route, math.pi)
    assert goal[:2] == [7.0, 3.0]
    assert angle_error(goal[2], route["far_goal_xyyaw"][2]) == pytest.approx(math.pi)
    assert -math.pi <= normalize_angle(goal[2]) <= math.pi


def test_angle_error_wraps_across_pi() -> None:
    assert angle_error(math.pi - 0.05, -math.pi + 0.05) == pytest.approx(0.10)


def test_all_open_state_targets_each_door_open_limit(monkeypatch) -> None:
    positions = {"left": 0.0, "right": 0.0}

    class FakeDoor:
        def __init__(self, name, _data) -> None:
            self.name = name

        def get_joint_position(self, _index) -> float:
            return positions[self.name]

        def set_joint_position(self, _index, value) -> None:
            positions[self.name] = float(value)

    groups = {
        "double_door": {
            "leaves": [
                {
                    "leaf_body_name": "left",
                    "hinge_joint_index": 0,
                    "hinge_joint_name": "left_hinge",
                    "joint_range": [0.0, 1.5],
                },
                {
                    "leaf_body_name": "right",
                    "hinge_joint_index": 0,
                    "hinge_joint_name": "right_hinge",
                    "joint_range": [-1.5, 0.0],
                },
            ]
        }
    }
    monkeypatch.setattr(force_interaction_runtime, "Door", FakeDoor)
    monkeypatch.setattr(
        force_interaction_runtime, "collect_door_root_groups", lambda _env: groups
    )
    monkeypatch.setattr(force_interaction_runtime.mujoco, "mj_forward", lambda *_args: None)
    task = SimpleNamespace(
        env=SimpleNamespace(current_model=object(), current_data=object())
    )

    transitions = force_interaction_runtime.set_all_door_roots_open(task.env)

    assert positions == {"left": 1.5, "right": -1.5}
    assert transitions[0]["state"] == "open"


def test_initial_door_state_dispatches_to_open_helper(monkeypatch) -> None:
    expected = [{"root_body_name": "door", "state": "open"}]
    task = SimpleNamespace(env=object())
    monkeypatch.setattr(
        run_nav_ros_sim, "set_all_door_roots_open", lambda _env: expected
    )
    assert apply_initial_door_state(task, "open") == expected


def test_navigation_launch_exposes_initial_door_state() -> None:
    launch_path = (
        REPO_ROOT
        / "Interactive-Nav-SG-nav"
        / "src"
        / "nav_pkg"
        / "launch"
        / "molmospaces_nav_system.launch"
    )
    root = ET.parse(launch_path).getroot()
    args = {element.attrib["name"]: element.attrib for element in root.iter("arg")}
    assert args["initial_door_state"]["default"] == "unchanged"
    simulator_node = next(
        element for element in root.iter("node") if element.attrib.get("name") == "molmospaces_nav_ros_sim"
    )
    assert "--initial_door_state $(arg initial_door_state)" in simulator_node.attrib["args"]
