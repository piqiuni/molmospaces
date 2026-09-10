import mujoco
import numpy as np
import pytest

from molmo_spaces.robots.robot_views.abstract import HoloJointsRobotBaseGroup
from molmo_spaces.robots.robot_views.rby1_view import RBY1HoloBaseGroup


def _make_group(mode: str) -> HoloJointsRobotBaseGroup:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <site name="world"/>
            <body name="base">
              <joint name="base_x" type="slide" axis="1 0 0"/>
              <joint name="base_y" type="slide" axis="0 1 0"/>
              <joint name="base_theta" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.1"/>
              <site name="base_site"/>
            </body>
          </worldbody>
          <actuator>
            <position name="base_x_act" joint="base_x" ctrlrange="-100 100"/>
            <position name="base_y_act" joint="base_y" ctrlrange="-100 100"/>
            <position name="base_theta_act" joint="base_theta" ctrlrange="-10 10"/>
          </actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    return HoloJointsRobotBaseGroup(
        data,
        model.site("world").id,
        model.site("base_site").id,
        [model.joint(name).id for name in ("base_x", "base_y", "base_theta")],
        [
            model.actuator(name).id
            for name in ("base_x_act", "base_y_act", "base_theta_act")
        ],
        model.body("base").id,
        yaw_control_mode=mode,
    )


def test_nearest_equivalent_yaw_keeps_current_qpos_branch() -> None:
    group = _make_group("nearest_equivalent")
    theta_address = group.mj_model.jnt_qposadr[group._joint_ids[2]]
    group.mj_data.qpos[theta_address] = 3.1

    group.ctrl = np.array([0.0, 0.0, -3.1])

    assert group.mj_data.qpos[theta_address] == pytest.approx(3.1)
    assert group.ctrl[2] == pytest.approx(2.0 * np.pi - 3.1)


def test_legacy_yaw_moves_qpos_to_target_branch() -> None:
    group = _make_group("legacy_branch_reset")
    theta_address = group.mj_model.jnt_qposadr[group._joint_ids[2]]
    group.mj_data.qpos[theta_address] = 3.1

    group.ctrl = np.array([0.0, 0.0, -3.1])

    assert group.mj_data.qpos[theta_address] == pytest.approx(-3.1)
    assert group.ctrl[2] == pytest.approx(-3.1)


def test_holo_base_rejects_unknown_yaw_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported holonomic base yaw mode"):
        _make_group("unknown")


def test_rby1_holo_base_forwards_yaw_mode() -> None:
    generic_group = _make_group("nearest_equivalent")

    group = RBY1HoloBaseGroup(
        generic_group.mj_data,
        yaw_control_mode="legacy_branch_reset",
    )

    assert group._yaw_control_mode == "legacy_branch_reset"
