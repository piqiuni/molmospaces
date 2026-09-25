import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from semantic_decision_py_pkg.behavior_execution import interaction_pose_validation
from semantic_decision_py_pkg.navigation_arrival import PositionToleranceLatch


def validation(pose, goal=(1.0, 0.0, 0.0)):
    return interaction_pose_validation(list(goal), pose, distance_tolerance_m=0.15, yaw_tolerance_rad=0.08)


def test_latched_position_does_not_recheck_distance_during_rotation():
    latch = PositionToleranceLatch((1.0, 0.0, 0.0), 0.15)
    assert not latch.observe((0.82, 0.0, 1.0), 1)
    assert latch.observe((0.86, 0.0, 1.0), 2)
    assert latch.observe((0.70, 0.0, 0.2), 3)
    assert not latch.apply(validation([0.70, 0.0, 0.2]))["valid"]
    ready = latch.apply(validation([0.70, 0.0, 0.05]))
    assert ready["valid"]
    assert ready["position_error_m"] > 0.15
    assert ready["position_latch_step_index"] == 2
    assert ready["position_latch_pose_xyyaw"] == [0.86, 0.0, 1.0]


def test_latch_is_goal_local_and_new_goal_requires_new_position_arrival():
    latch = PositionToleranceLatch((1.0, 0.0, 0.0), 0.15)
    latch.observe((1.0, 0.0, 0.0), 1)
    assert not latch.apply(validation([1.0, 0.0, 0.0], goal=(2.0, 0.0, 0.0)))["valid"]
    successor = PositionToleranceLatch((2.0, 0.0, 0.0), 0.15)
    assert not successor.observe((1.0, 0.0, 0.0), 2)


def test_missing_pose_never_completes_latched_rotation():
    latch = PositionToleranceLatch((1.0, 0.0, 0.0), 0.15)
    assert not latch.observe((math.nan, 0.0, 0.0), 1)
    assert not latch.observe(None, 2)
    latch.observe((1.0, 0.0, 0.0), 3)
    assert not latch.apply(validation(None))["valid"]


def test_noetic_position_latch_uses_move_base_private_namespace():
    root = Path(__file__).resolve().parents[4]
    config = yaml.safe_load((root / "scripts/InteractiveNav/configs/semantic_decision/semantic_interaction_nav.yaml").read_text())
    assert config["latch_xy_goal_tolerance"] is True
    assert config["GlobalPlanner"]["orientation_mode"] == 0
    launch = ET.parse(root / "Interactive-Nav-SG-nav/src/nav_pkg/launch/nav.launch")
    overrides = [node for node in launch.iter("param") if node.get("name") == "GlobalPlanner/orientation_mode"]
    assert overrides[-1].get("value") == "0"
    assert "OrientedGlobalPlanner" in overrides[-1].get("if", "")
