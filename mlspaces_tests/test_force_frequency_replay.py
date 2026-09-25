import json
import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from scripts.InteractiveNav import force_interaction_runtime as runtime
from scripts.InteractiveNav.fixed_interaction_frequency import ForceFrequencyReplay


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="0 0 0" timestep=".002"/>
      <worldbody><body name="drawer">
        <joint name="slide" type="slide" axis="1 0 0" range="0 1" damping="2"/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def schedule(tmp_path, steps):
    baseline = tmp_path / 'baseline'
    output = tmp_path / 'candidate'
    baseline.mkdir()
    output.mkdir()
    (baseline / 'execution.json').write_text(json.dumps({'result': {
        'physics_substeps': sum(steps),
        'transition_log': [{'physics_substeps': count} for count in steps],
    }}))
    return baseline, output


def test_factor_one_matches_recorded_normal_trajectory(scene, tmp_path):
    model, data = scene
    config = runtime.ForceDriveConfig(max_physics_substeps=1000)
    normal_trace = []
    original = runtime.drive_joint_group_to_targets
    normal = original(model, data, {'slide': .2}, config,
                      robot_lock_callback=lambda: normal_trace.append(data.qpos.copy()))
    final = np.r_[data.qpos, data.qvel].copy()
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    baseline, output = schedule(tmp_path, [normal['physics_substeps']])
    replay_trace = []
    with ForceFrequencyReplay(runtime, baseline, 1, output):
        replay = runtime.drive_joint_group_to_targets(
            model, data, {'slide': .2}, config,
            robot_lock_callback=lambda: replay_trace.append(data.qpos.copy()))
    assert runtime.drive_joint_group_to_targets is original
    assert normal['physics_substeps'] == replay['physics_substeps']
    assert normal['success'] == replay['success']
    np.testing.assert_array_equal(normal_trace, replay_trace)
    np.testing.assert_array_equal(final, np.r_[data.qpos, data.qvel])


@pytest.mark.parametrize('factor', [2, 4])
def test_lower_frequency_preserves_duration_including_residual(scene, tmp_path, factor):
    model, data = scene
    baseline, output = schedule(tmp_path, [11])
    original = runtime.drive_joint_group_to_targets
    sampled_dt = []
    with ForceFrequencyReplay(runtime, baseline, factor, output) as replay:
        result = runtime.drive_joint_group_to_targets(
            model, data, {'slide': 0},
            runtime.ForceDriveConfig(max_physics_substeps=1, stable_substeps=1),
            robot_lock_callback=lambda: sampled_dt.append(float(model.opt.timestep)))
    assert runtime.drive_joint_group_to_targets is original
    assert model.opt.timestep == .002
    assert data.time == pytest.approx(.022, abs=1e-12)
    assert result['physics_substeps'] == math.ceil(11 / factor)
    assert min(sampled_dt) == pytest.approx((11 % factor) * .002)
    assert max(sampled_dt) == pytest.approx(factor * .002)
    assert replay.records[0]['actual_seconds'] == pytest.approx(.022)
    assert replay.records[0]['warning_deltas'] == [0] * len(data.warning)


def test_replay_does_not_stop_after_early_convergence(scene):
    model, data = scene
    config = runtime.ForceDriveConfig(max_physics_substeps=2, stable_substeps=1)
    normal = runtime.drive_joint_group_to_targets(model, data, {'slide': 0}, config)
    assert normal['physics_substeps'] == 1
    start = data.time
    replay = runtime.drive_joint_group_to_targets(
        model, data, {'slide': 0},
        replace(config, replay_duration_seconds=.026, replay_stable_seconds=.002))
    assert replay['physics_substeps'] == 13
    assert data.time - start == pytest.approx(.026)
    assert replay['success']


@pytest.mark.parametrize('duration', [0, -.01, float('nan'), float('inf')])
def test_invalid_duration_fails_without_advancing(scene, duration):
    model, data = scene
    with pytest.raises(ValueError, match='duration'):
        runtime.drive_joint_group_to_targets(
            model, data, {'slide': .2},
            runtime.ForceDriveConfig(replay_duration_seconds=duration))
    assert data.time == 0
    assert model.opt.timestep == .002


def test_exception_restores_timestep_and_driver(scene, tmp_path):
    model, data = scene
    baseline, output = schedule(tmp_path, [11])
    original = runtime.drive_joint_group_to_targets

    def fail():
        raise RuntimeError('injected callback failure')

    with pytest.raises(RuntimeError, match='injected callback failure'):
        with ForceFrequencyReplay(runtime, baseline, 4, output):
            runtime.drive_joint_group_to_targets(
                model, data, {'slide': .2}, robot_lock_callback=fail)
    assert runtime.drive_joint_group_to_targets is original
    assert model.opt.timestep == .002
    assert not np.any(data.xfrc_applied)


def test_incomplete_context_restores_driver(tmp_path):
    baseline, output = schedule(tmp_path, [11])
    original = runtime.drive_joint_group_to_targets
    with pytest.raises(RuntimeError, match='did not replay all'):
        with ForceFrequencyReplay(runtime, baseline, 2, output):
            pass
    assert runtime.drive_joint_group_to_targets is original


def test_context_body_exception_restores_driver(tmp_path):
    baseline, output = schedule(tmp_path, [11])
    original = runtime.drive_joint_group_to_targets
    with pytest.raises(ValueError, match='body failure'):
        with ForceFrequencyReplay(runtime, baseline, 2, output):
            raise ValueError('body failure')
    assert runtime.drive_joint_group_to_targets is original
