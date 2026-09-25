import json
import math

import numpy as np
import pytest

from scripts.InteractiveNav.run_fixed_interaction_test import (
    fixed_robot_pose, load_case, compare, synchronize_hold_targets, INITIAL_CONDITION_KEYS,
    run_view_cycle)


@pytest.mark.parametrize('converges, restore_steps', [(True, 4), (False, 24)])
def test_view_preflight_requires_physical_convergence(monkeypatch, converges, restore_steps):
    from types import SimpleNamespace
    from scripts.InteractiveNav import force_interaction_runtime as runtime

    class View:
        checks = 0

        def command(self, env, *args, **kwargs):
            self.target = [.35]

        def torso_target(self):
            return self.target

        def restore(self, env):
            self.target = [0.0]

        def restore_convergence(self, env):
            self.checks += 1
            return {'converged': converges and self.checks >= 4}

    monkeypatch.setattr(runtime, 'HeadViewController', View)
    phases = []
    result = run_view_cycle(task=SimpleNamespace(env=object()), command={}, object_name='unused',
                           step=lambda controller, i: phases.append(controller._pending['phase']))
    assert phases == ['low_view'] * 10 + ['restore_settle'] * restore_steps
    assert result['result']['success'] is converges
    assert result['task_steps_consumed'] == len(phases)


def test_sync_resets_controller_targets_without_resetting_robot_pose():
    from types import SimpleNamespace
    actual = np.array([.1, .4, 0.0])
    controller = SimpleNamespace(target=np.zeros(3))
    controller.reset = lambda: setattr(controller, 'target', actual.copy())
    commands = []
    robot = SimpleNamespace(controllers={'torso': controller},
                            compute_control=lambda: commands.append(controller.target.copy()))
    synchronize_hold_targets(robot)
    np.testing.assert_array_equal(controller.target, actual)
    np.testing.assert_array_equal(commands[0], actual)


def test_fixed_pose_uses_scalar_first_quaternion_and_preserves_height():
    pose = fixed_robot_pose([0, 0, .25, 1, 0, 0, 0], [2, 3, math.pi / 2])
    np.testing.assert_allclose(pose, [2, 3, .25, 2**-.5, 0, 0, 2**-.5])


def test_fixed_pose_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        fixed_robot_pose([0] * 7, [0, 0, float('nan')])


def test_case_requires_pose_and_preserves_joint_identity(tmp_path):
    path = tmp_path / 'macro.json'
    path.write_text(json.dumps({'result': {}, 'observations': [{'joint_fractions': {'drawer': 0}}]}))
    with pytest.raises(ValueError):
        load_case({'episode': 1, 'macro_path': str(path)})
    case = load_case({'episode': 1, 'macro_path': str(path), 'pose_xyyaw': [1, 2, 0]})
    assert case['joints'] == ['drawer']
    assert case['command']['sequence_type'] == 'drawer_scan'


@pytest.mark.parametrize('delta, target_delta, equivalent', [(0, 0, True), (.01, 0, False), (0, 1, False)])
def test_comparison_requires_physical_equivalence(tmp_path, delta, target_delta, equivalent):
    for variant in ('baseline', 'optimized'):
        directory = tmp_path / variant / '1'
        directory.mkdir(parents=True)
        ready = dict.fromkeys(INITIAL_CONDITION_KEYS, 'same')
        ready['controller_targets'] = target_delta if variant == 'optimized' else 0
        (directory / 'ready.json').write_text(json.dumps(ready))
        (directory / 'summary.json').write_text(json.dumps(dict(phases=['open'], physics_substeps=5,
            inner_steps=1, state='closed', result_success=True, elapsed_seconds=1)))
        np.savez(directory / 'trajectory.npz', qpos=np.array([[delta if variant == 'optimized' else 0]]),
                 qvel=np.zeros((1, 1)), contacts=np.zeros((1, 2)), final_qpos=np.zeros(1), final_qvel=np.zeros(1))
    assert compare(tmp_path, [{'episode': 1}])['all_equivalent'] is equivalent


@pytest.mark.parametrize('regression', [None, 'visibility', 'position', 'fallback', 'contact',
                                      'penetration', 'initial', 'observations', 'failure', 'slower', 'steps',
                                      'relocated_fallback', 'view_restore', 'one_pixel'])
def test_effect_checks_reject_regressions(tmp_path, regression):
    from scripts.InteractiveNav.fixed_interaction_quality import compare_effects
    for variant in ('baseline', 'optimized'):
        candidate = variant == 'optimized'
        directory = tmp_path / variant / '1'
        directory.mkdir(parents=True)
        ready = dict.fromkeys(INITIAL_CONDITION_KEYS, 'same')
        if candidate and regression == 'initial':
            ready['controller_targets'] = 'different'
        (directory / 'ready.json').write_text(json.dumps(ready))
        summary = dict(result_success=not(candidate and regression == 'failure'), state='closed',
                       physics_substeps=5 if candidate and regression != 'steps' else 10,
                       elapsed_seconds=1 if candidate and regression != 'slower' else 2)
        (directory / 'summary.json').write_text(json.dumps(summary))
        execution = {'result': dict(view_restore_convergence={'converged': not(candidate and regression == 'view_restore')},
            transition_log=[{'fallback': regression == 'relocated_fallback' or (candidate and regression == 'fallback'),
                             'phase': 'open', 'group_id': '1' if candidate and regression == 'relocated_fallback' else '0'}])}
        (directory / 'execution.json').write_text(json.dumps(execution))
        frame = dict(phase='restore' if candidate and regression == 'observations' else 'observe',
                     quality=dict(group_index=0, target_fraction=0 if candidate and regression == 'visibility' else .1,
                                  target_visible=True, robot_contacts=dict(
                                      count=1 if candidate and regression == 'contact' else 0,
                                      penetration=.001 if candidate and regression == 'penetration' else 0)))
        if regression == 'one_pixel':
            frame['quality']['target_fraction'] = (29 if candidate else 30) / (640 * 480)
        (directory / 'frames.jsonl').write_text(json.dumps(frame) + '\n')
        position = .011 if candidate and regression == 'position' else 0
        np.savez(directory / 'trajectory.npz', observed_qpos=np.array([[position]]),
                 observed_qvel=np.zeros((1, 1)), final_qpos=np.zeros(1), final_qvel=np.zeros(1))
    report = compare_effects(tmp_path, [{'episode': 1}])
    assert report['all_effect_checks_pass'] is (regression in (None, 'one_pixel'))


def test_baseline_reuse_validates_flags_and_does_not_copy_or_overwrite(tmp_path):
    from scripts.InteractiveNav.run_fixed_interaction_test import reuse_baseline
    source, output = tmp_path / 'source', tmp_path / 'new'
    episode = source / 'baseline' / '1'
    episode.mkdir(parents=True)
    output.mkdir()
    cases, flags = [{'episode': 1}], {'coalesce_robot_lock': True}
    (source / 'run_config.json').write_text(json.dumps({'baseline': flags}))
    (source / 'manifest.json').write_text(json.dumps(cases))
    (source / 'wave_times.json').write_text(json.dumps({'baseline': 12}))
    (episode / 'summary.json').write_text(json.dumps({'result_success': True, 'mode': 'full_drawer_scan'}))
    for name in ('ready.json', 'frames.jsonl', 'trajectory.npz', 'execution.json'):
        (episode / name).write_bytes(b'preserved')
    with pytest.raises(ValueError, match='configuration'):
        reuse_baseline(source, output, cases, {'coalesce_robot_lock': False})
    assert not (output / 'baseline').exists()
    assert reuse_baseline(source, output, cases, flags) == {'baseline': 12}
    assert (output / 'baseline').is_symlink()
    assert (episode / 'trajectory.npz').read_bytes() == b'preserved'


@pytest.mark.parametrize('regression', [None, 'force_clock', 'full_clock', 'warning'])
def test_frequency_comparison_requires_clocks_and_no_warnings(tmp_path, monkeypatch, regression):
    from scripts.InteractiveNav import fixed_interaction_quality as quality
    from scripts.InteractiveNav.fixed_interaction_frequency import compare_frequency
    monkeypatch.setattr(quality, 'compare_effects', lambda *args: {
        'episodes': [{'episode': 1, 'checks': {'effects': True}, 'passed': True}]})
    baseline, candidate = [tmp_path / v / '1' for v in ('baseline', 'optimized')]
    for path in (baseline, candidate):
        path.mkdir(parents=True)
    (baseline / 'summary.json').write_text(json.dumps(dict(
        physics_substeps=10, inner_steps=1, cpu_seconds=2)))
    (candidate / 'summary.json').write_text(json.dumps(dict(
        simulated_seconds=.23 if regression == 'full_clock' else .22, cpu_seconds=1)))
    (candidate / 'frequency_config.json').write_text(json.dumps(dict(
        baseline_timestep=.002, ordinary_control_dt_ms=10, control_steps_per_policy=20)))
    (candidate / 'frequency_drives.jsonl').write_text(json.dumps(dict(
        actual_seconds=.01 if regression == 'force_clock' else .02, expected_seconds=.02,
        warning_deltas=[1 if regression == 'warning' else 0])) + '\n')
    report = compare_frequency(tmp_path, [{'episode': 1}])
    assert report['all_effect_checks_pass'] is (regression is None)


def test_observation_hashes_cover_nested_arrays_and_detect_pixel_change():
    from scripts.InteractiveNav.run_fixed_interaction_test import observation_hashes
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    original = observation_hashes({'camera': [rgb], 'label': 'not an array'})
    assert original['/camera/0']['shape'] == [2, 2, 3]
    assert original == observation_hashes({'camera': [rgb.copy()]})
    rgb[0, 0, 0] = 1
    assert original != observation_hashes({'camera': [rgb]})


@pytest.mark.parametrize('optimized', [False, True])
def test_compute_worker_flags_accept_string_experiment(monkeypatch, tmp_path, optimized):
    import os
    from types import SimpleNamespace
    from scripts.InteractiveNav import run_fixed_interaction_test as runner
    keys = ('INTERACTIVE_NAV_COALESCE_ROBOT_LOCK', 'INTERACTIVE_NAV_INTERMEDIATE_POSITION_ONLY',
            'INTERACTIVE_NAV_LOCK_POSITION_FORWARD', 'INTERACTIVE_NAV_LOCK_GEOMETRY_FORWARD',
            'INTERACTIVE_NAV_SCENE_MIRROR', 'MUJOCO_GL', 'PYOPENGL_PLATFORM')
    for key in keys:
        monkeypatch.setenv(key, '')
    def stop_before_imports(seed):
        raise RuntimeError('flags configured')
    monkeypatch.setattr(runner.random, 'seed', stop_before_imports)
    args = SimpleNamespace(optimized=optimized, adaptive=False, position_forward=False,
                           force_frequency_factor=None, lock_experiment='combined', output=tmp_path)
    with pytest.raises(RuntimeError, match='flags configured'):
        runner.worker(args)
    assert os.environ['INTERACTIVE_NAV_COALESCE_ROBOT_LOCK'] == '1'
    assert os.environ['INTERACTIVE_NAV_LOCK_POSITION_FORWARD'] == '1'
    assert os.environ['INTERACTIVE_NAV_LOCK_GEOMETRY_FORWARD'] == str(int(optimized))
