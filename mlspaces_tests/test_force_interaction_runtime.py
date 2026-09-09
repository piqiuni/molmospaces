from __future__ import annotations

import sys
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav.force_interaction_runtime import (
    ForceDriveConfig,
    drive_joint_group_to_targets,
    joint_closed_open_values,
    joint_open_fraction,
)


DOUBLE_HINGE_XML = """
<mujoco>
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <body name="left_leaf" pos="0 0 0">
      <joint name="left_hinge" type="hinge" axis="0 0 1" range="0 90" damping="1"/>
      <geom type="box" size="0.4 0.03 0.8" mass="1" pos="0.4 0 0"/>
    </body>
    <body name="right_leaf" pos="0 1 0">
      <joint name="right_hinge" type="hinge" axis="0 0 -1" range="-90 0" damping="1"/>
      <geom type="box" size="0.4 0.03 0.8" mass="1" pos="0.4 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_closed_open_values_support_positive_and_negative_ranges() -> None:
    assert joint_closed_open_values([0.0, 1.5]) == (0.0, 1.5)
    assert joint_closed_open_values([-1.5, 0.0]) == (0.0, -1.5)
    assert joint_open_fraction(-0.75, [-1.5, 0.0]) == 0.5


def test_group_force_drive_opens_two_hinges_together() -> None:
    model = mujoco.MjModel.from_xml_string(DOUBLE_HINGE_XML)
    data = mujoco.MjData(model)
    result = drive_joint_group_to_targets(
        model,
        data,
        {
            "left_hinge": float(model.jnt_range[model.joint("left_hinge").id][1]),
            "right_hinge": float(model.jnt_range[model.joint("right_hinge").id][0]),
        },
        config=ForceDriveConfig(max_physics_substeps=2500),
    )

    assert result["success"] is True
    assert result["physics_substeps"] > 0
    assert {joint["joint_name"] for joint in result["joints"]} == {
        "left_hinge",
        "right_hinge",
    }
    assert all(joint["open_fraction"] >= 0.99 for joint in result["joints"])
