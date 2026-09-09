from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from molmo_spaces.robots.rby1 import RBY1


def _minimal_holo_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <body name="robot_0/base">
              <joint name="robot_0/base_x" type="slide" range="-25 25"/>
              <joint name="robot_0/base_y" type="slide" range="-25 25"/>
            </body>
          </worldbody>
          <actuator>
            <position name="robot_0/base_x_act" joint="robot_0/base_x"
                      ctrlrange="-25 25"/>
            <position name="robot_0/base_y_act" joint="robot_0/base_y"
                      ctrlrange="-25 25"/>
          </actuator>
        </mujoco>
        """
    )


def _config(limit_m: float = 100.0, *, use_holo_base: bool = True):
    return SimpleNamespace(
        gravcomp=False,
        K_stiffness=None,
        K_damping=None,
        use_holo_base=use_holo_base,
        holo_base_position_limit_m=limit_m,
        robot_namespace="robot_0/",
    )


def test_rby1_holo_base_override_expands_joint_and_actuator_ranges() -> None:
    spec = _minimal_holo_spec()

    RBY1.apply_control_overrides(spec, _config())

    expected = np.array([-100.0, 100.0])
    for axis in ("x", "y"):
        np.testing.assert_allclose(spec.joint(f"robot_0/base_{axis}").range, expected)
        np.testing.assert_allclose(
            spec.actuator(f"robot_0/base_{axis}_act").ctrlrange, expected
        )


def test_rby1_holo_base_override_defaults_to_100m_for_legacy_config() -> None:
    config = _config()
    del config.holo_base_position_limit_m
    spec = _minimal_holo_spec()

    RBY1.apply_control_overrides(spec, config)

    np.testing.assert_allclose(spec.joint("robot_0/base_y").range, [-100.0, 100.0])


def test_rby1_holo_base_override_leaves_non_holonomic_model_unchanged() -> None:
    spec = _minimal_holo_spec()

    RBY1.apply_control_overrides(spec, _config(use_holo_base=False))

    np.testing.assert_allclose(spec.joint("robot_0/base_y").range, [-25.0, 25.0])


@pytest.mark.parametrize("limit_m", [0.0, -1.0, float("nan"), float("inf")])
def test_rby1_holo_base_override_rejects_invalid_limit(limit_m: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RBY1.apply_control_overrides(_minimal_holo_spec(), _config(limit_m))
