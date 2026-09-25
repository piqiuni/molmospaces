from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

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


def test_force_config_preserves_legacy_positional_arguments() -> None:
    config = ForceDriveConfig(2500, 12)
    assert config.max_physics_substeps == 2500
    assert config.stable_substeps == 12


@pytest.mark.parametrize('kind,progress,enabled,active', [
    ('slide', .2, True, True), ('slide', .8, True, True),
    ('slide', 1., True, False), ('slide', 0., True, False),
    ('hinge', .2, True, False), ('slide', .2, False, False),
])
def test_position_only_settle_is_limited_to_intermediate_slides(monkeypatch, kind, progress, enabled, active):
    from types import SimpleNamespace
    from scripts.InteractiveNav import force_interaction_runtime as runtime
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body>
      <joint name="j" type="{kind}" axis="1 0 0" range="0 1"/>
      <geom type="sphere" size=".1" mass="1"/>
    </body></worldbody></mujoco>''')
    configs = []
    def drive(*args, config, **kwargs):
        configs.append(config)
        return {'success': True, 'physics_substeps': 1}
    monkeypatch.setattr(runtime, 'drive_joint_group_to_targets', drive)
    config = ForceDriveConfig(intermediate_position_only=enabled)
    result = runtime.advance_articulation_force(
        SimpleNamespace(current_model=model, current_data=mujoco.MjData(model)),
        {'targets': {'j': .2}, 'group': {'joints': []}}, progress, {'j': 0}, 5, config)
    assert result['position_only_settle'] is active
    assert configs[0].position_tolerance == config.position_tolerance
    assert configs[0].max_physics_substeps == 300
    assert configs[0].stable_substeps == 8
    assert configs[0].velocity_tolerance == .01
    assert configs[0].position_only_settle is active


def test_position_only_settle_keeps_full_rule_on_robot_contact():
    xml = '''<mujoco><option gravity="0 0 0" timestep=".002"/>
      <worldbody><body name="drawer"><joint name="j" type="slide" axis="1 0 0" range="0 .3"/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="robot_0" pos=".19 0 0"><geom type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>'''
    outcomes = []
    for enabled in (False, True):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        result = drive_joint_group_to_targets(model, data, {'j': .001},
            ForceDriveConfig(max_physics_substeps=100, position_only_settle=enabled))
        assert result['robot_target_contacts_before']['count'] > 0
        outcomes.append((result['physics_substeps'], data.qpos.copy()))
    assert outcomes[0][0] == outcomes[1][0]
    np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])


def test_position_only_settle_detects_new_contact_before_early_exit(monkeypatch):
    from scripts.InteractiveNav import force_interaction_runtime as runtime
    calls = 0
    def contacts(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {'count': 0 if calls == 1 else 1, 'minimum_distance': 0.0}
    monkeypatch.setattr(runtime, '_robot_articulation_contact_stats', contacts)
    model = mujoco.MjModel.from_xml_string(DOUBLE_HINGE_XML)
    result = drive_joint_group_to_targets(model, mujoco.MjData(model), {'left_hinge': 0.0},
        ForceDriveConfig(position_only_settle=True))
    assert result['physics_substeps'] == 8
    assert result['position_only_contact_guarded'] is True


def test_intermediate_settle_saves_steps_without_endpoint_fallback():
    from types import SimpleNamespace
    from scripts.InteractiveNav.force_interaction_runtime import advance_articulation_force
    totals = []
    for enabled in (False, True):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0" timestep=".002"/>
          <worldbody><body><joint name="j" type="slide" axis="1 0 0" range="0 .3" damping="5"/>
          <geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        env = SimpleNamespace(current_model=model, current_data=data)
        total = 0
        for target in (.25, 0.):
            start = float(data.qpos[0])
            for progress in (.2, .4, .6, .8, 1.):
                result = advance_articulation_force(env, {'targets': {'j': target}, 'group': {'joints': []}},
                    progress, {'j': start}, 5, ForceDriveConfig(intermediate_position_only=enabled))
                assert not result['fallback']
                total += result['physics_substeps']
            assert abs(data.qpos[0] - target) <= .01
            assert data.qvel[0] == 0
        totals.append(total)
    assert totals[1] < totals[0]


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


def test_coalesced_robot_lock_preserves_force_trajectory():
    outcomes = []
    for optimized in (False, True):
        model = mujoco.MjModel.from_xml_string(DOUBLE_HINGE_XML)
        data = mujoco.MjData(model)
        count = 0
        def lock():
            nonlocal count
            count += 1
            mujoco.mj_forward(model, data)
        trajectory = []
        for target in (0.5, 1.0, 0.0):
            result = drive_joint_group_to_targets(model, data, {"left_hinge": target},
                config=ForceDriveConfig(max_physics_substeps=2500, coalesce_robot_lock=optimized),
                robot_lock_callback=lock)
            trajectory.append((data.qpos.copy(), result))
        outcomes.append((trajectory, count))
    for (base_qpos, base), (fast_qpos, fast) in zip(outcomes[0][0], outcomes[1][0]):
        np.testing.assert_allclose(fast_qpos, base_qpos, atol=1e-10, rtol=0)
        assert fast["success"] == base["success"]
        assert fast["physics_substeps"] == base["physics_substeps"]
        assert fast["robot_target_max_contact_count"] == base["robot_target_max_contact_count"]
    assert outcomes[1][1] == outcomes[0][1] // 2 + 3


def test_coalesced_lock_preserves_robot_contact_and_drawer_trajectory():
    xml = '''<mujoco><option gravity="0 0 0" timestep="0.002"/>
    <worldbody>
      <body name="drawer"><joint name="slide" type="slide" axis="1 0 0" range="0 .3" damping="5"/>
        <geom type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="robot_0" pos="-.19 0 0"><joint name="robot_0/lock" type="slide" axis="1 0 0"/>
        <geom name="robot_0/geom" type="box" size=".1 .1 .1" mass="10"/></body>
    </worldbody></mujoco>'''
    runs = []
    for optimized in (False, True):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        robot = model.joint("robot_0/lock")
        def lock():
            data.qpos[robot.qposadr] = 0
            data.qvel[robot.dofadr] = 0
            mujoco.mj_forward(model, data)
        lock()
        trajectory = []
        for target in (.05, .15, .3, .1, 0):
            result = drive_joint_group_to_targets(model, data, {"slide": target},
                config=ForceDriveConfig(coalesce_robot_lock=optimized), robot_lock_callback=lock)
            trajectory.append((data.qpos.copy(), result))
        runs.append(trajectory)
    for (q0, baseline), (q1, candidate) in zip(*runs):
        np.testing.assert_allclose(q1, q0, atol=1e-8, rtol=0)
        assert candidate["success"] == baseline["success"]
        assert candidate["physics_substeps"] == baseline["physics_substeps"]
        assert candidate["robot_target_max_contact_count"] == baseline["robot_target_max_contact_count"]
    assert any(result["robot_target_max_contact_count"] > 0 for _, result in runs[0])
