"""Paired interaction-only diagnostics; never report these as navigation SR."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
INITIAL_CONDITION_KEYS = ('initial_state_sha256', 'model_sha256', 'case', 'object_name',
                          'controller_types', 'controller_targets')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def observation_hashes(observation):
    """Fingerprint array-valued observations without retaining large images."""
    import numpy as np
    hashes = {}
    def visit(value, path):
        if isinstance(value, np.ndarray):
            hashes[path] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                sha256=hashlib.sha256(value.tobytes()).hexdigest())
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f'{path}/{key}')
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f'{path}/{index}')
    visit(observation, '')
    return hashes


def load_case(case):
    macro_path = Path(case['macro_path']).resolve()
    macro = json.loads(macro_path.read_text())
    result = macro['result']
    pose = case.get('pose_xyyaw', result.get('approach_goal_xyyaw'))
    if pose is None or len(pose) != 3:
        raise ValueError('A fixed interaction pose is required')
    joints = list(macro['observations'][0]['joint_fractions'])
    if not joints:
        raise ValueError('Historical interaction has no joint identities')
    command = dict(object_id='fixed_container', action='open', sequence_type='drawer_scan',
                   interaction_mode='drawer_scan', command_id='fixed_scan',
                   approach_goal_xyyaw=pose, view_profile='drawer_low_view',
                   open_regions=result.get('open_regions', []))
    return dict(episode=int(case['episode']), pose_xyyaw=pose, joints=joints,
                command=command, source=str(macro_path),
                pose_source=case.get('pose_source', 'manifest override or historical approach goal'),
                source_sha256=hashlib.sha256(macro_path.read_bytes()).hexdigest())


def fixed_robot_pose(original, xyyaw):
    if len(original) != 7 or len(xyyaw) != 3 or not all(math.isfinite(x) for x in xyyaw):
        raise ValueError('Expected finite x/y/yaw and a seven-component robot pose')
    x, y, yaw = xyyaw
    return [x, y, original[2], math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def synchronize_hold_targets(robot):
    """Sampling can move joints after controller construction; reset targets only."""
    for controller in robot.controllers.values():
        controller.reset()
    robot.compute_control()


def run_view_cycle(*, task, command, object_name, step, timing_sink=None):
    """Cheap preflight for low-view control and physical restoration; no drawers."""
    from scripts.InteractiveNav.force_interaction_runtime import HeadViewController
    view = HeadViewController()
    view.view_torso_target = view.torso_target
    view.command(task.env, 'drawer_low_view', tilt_rad=.3, torso_pitch_rad=.35)
    count = 0
    for _ in range(10):
        view._pending = {'phase': 'low_view'}
        step(view, count)
        count += 1
    view.restore(task.env)
    for _ in range(24):
        view._pending = {'phase': 'restore_settle'}
        step(view, count)
        count += 1
        convergence = view.restore_convergence(task.env)
        if convergence['converged']:
            break
    return dict(result=dict(success=convergence['converged'], state='view_restored',
                            physics_substeps=0, view_restore_convergence=convergence),
                task_steps_consumed=count, execution_mode='view_cycle_preflight')


def fingerprint_model(model, np):
    arrays = {}
    for name in sorted(dir(model)):
        if name.startswith('_'):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            arrays[name] = hashlib.sha256(value.tobytes()).hexdigest()
    arrays['options'] = hashlib.sha256(str(model.opt).encode()).hexdigest()
    return hashlib.sha256(json.dumps(arrays, sort_keys=True).encode()).hexdigest(), arrays


def worker(args):
    os.environ['INTERACTIVE_NAV_COALESCE_ROBOT_LOCK'] = str(int(bool(args.optimized or args.adaptive or args.position_forward or args.force_frequency_factor or args.lock_experiment)))
    os.environ['INTERACTIVE_NAV_INTERMEDIATE_POSITION_ONLY'] = str(int(args.adaptive and args.optimized))
    os.environ['INTERACTIVE_NAV_LOCK_POSITION_FORWARD'] = str(int(bool(args.lock_experiment) or (args.position_forward and args.optimized)))
    os.environ['INTERACTIVE_NAV_LOCK_GEOMETRY_FORWARD'] = str(int(args.optimized and args.lock_experiment in ('geometry_only', 'combined')))
    os.environ['INTERACTIVE_NAV_SCENE_MIRROR'] = str(args.output / 'mirror')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    import mujoco
    from scipy.spatial.transform import Rotation
    from scripts.InteractiveNav.evaluation import benchmark_runner as br
    from scripts.InteractiveNav.evaluation.interaction_profile import profile_interaction
    from scripts.InteractiveNav.evaluation.smooth_interaction import run_smooth_interaction
    from scripts.InteractiveNav.force_interaction_runtime import (
        joint_closed_open_values, _robot_articulation_contact_stats, _build_robot_contact_lookup)
    import molmo_spaces.tasks.task_sampler as ts

    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(json.loads(args.manifest.read_text())[args.case_index])
    episode = json.loads(args.benchmark.read_text())[case['episode']]
    config = br.BenchmarkEvaluationConfig(benchmark=args.benchmark, output_dir=args.output,
                                         policy='ros_object_goal_rule', max_steps=10000)
    spec = br.EpisodeSpec.model_validate(episode)
    pose = fixed_robot_pose(spec.task['robot_base_pose'], case['pose_xyyaw'])
    spec.task['robot_base_pose'] = pose
    spec.cameras = [c for c in spec.cameras if c.name == 'head_camera']
    spec.img_resolution = (640, 480)
    for camera in spec.cameras:
        camera.record_depth = True
    br._apply_ros_navigation_arm_posture(spec)
    replay = br._build_replay_config(config, args.output, task_horizon=10000)
    sampler = br.V3BenchmarkTaskSampler(replay, spec, episode['interactive_nav'])
    # Assets are already installed locally. Avoid modifying the shared dataset.
    ts.install_scene_with_objects_and_grasps_from_path = lambda *a, **kw: None
    variants = sampler._get_dataset_index_map()[spec.data_split][spec.house_index]
    source = variants['base']
    variants['base'] = br.probe.prepare_writable_scene_path(Path(source))
    try:
        task = sampler.sample_task(house_index=spec.house_index)
        task.reset()
        model, data = task.env.current_model, task.env.current_data
        # Match the official evaluator's post-reset frozen scene handoff.
        modifications = dict(episode.get('scene_modifications') or {})
        body_names = {model.body(i).name for i in range(model.nbody)}
        poses = modifications.get('object_poses', {})
        critical = set(br.target_candidate_names(episode))
        critical.update(row['object_name'] for row in episode['interactive_nav'].get('interactions', [])
                        if row.get('object_name'))
        dropped = sorted(set(poses) - body_names)
        if critical.intersection(dropped):
            raise RuntimeError(f'Missing critical scene bodies: {critical.intersection(dropped)}')
        modifications['object_poses'] = {k: v for k, v in poses.items() if k in body_names}
        scene_state = br.probe.apply_episode_scene_state(task.env, {**episode, 'scene_modifications': modifications})
        write_json(args.output / 'scene_state.json', dict(applied=scene_state, dropped_decorative_bodies=dropped))
        actual_pose = task.env.current_robot.robot_view.base.pose
        expected_rotation = Rotation.from_euler('z', case['pose_xyyaw'][2]).as_matrix()
        write_json(args.output / 'pose_audit.json', dict(expected_pose_wxyz=pose,
                                                      actual_matrix=actual_pose.tolist()))
        if not (np.allclose(actual_pose[:2, 3], pose[:2], atol=1e-6, rtol=0)
                and np.allclose(actual_pose[:3, :3], expected_rotation, atol=1e-6, rtol=0)):
            raise RuntimeError('Robot did not initialize at the fixed interaction pose')
        joint_ids = [model.joint(name).id for name in case['joints']]
        roots = {int(model.body_rootid[model.jnt_bodyid[j]]) for j in joint_ids}
        if len(roots) != 1:
            raise ValueError('Fixed joints must belong to one object')
        root_id = roots.pop()
        object_name = model.body(root_id).name
        for j in joint_ids:
            data.qpos[model.jnt_qposadr[j]] = joint_closed_open_values(model.jnt_range[j])[0]
            data.qvel[model.jnt_dofadr[j]] = 0
        robot = task.env.current_robot
        for group_name in robot.robot_view.move_group_ids():
            group = robot.robot_view.get_move_group(group_name)
            group.joint_vel = np.zeros_like(group.joint_vel)
        synchronize_hold_targets(robot)
        head = robot.robot_view.get_move_group('head')
        head.ctrl = head.noop_ctrl
        mujoco.mj_forward(model, data)
        task.env.camera_manager.registry.update_all_cameras(task.env)
        task.get_observations()  # Warm renderer equally, outside the measured region.
        signature = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, signature))
        mujoco.mj_getState(model, data, state, signature)
        np.save(args.output / 'initial_state.npy', state)
        model_hash, arrays = fingerprint_model(model, np)
        write_json(args.output / 'model_hashes.json', arrays)
        write_json(args.output / 'joint_layout.json', [dict(
            name=model.joint(i).name, body=model.body(int(model.jnt_bodyid[i])).name,
            root=model.body(int(model.body_rootid[model.jnt_bodyid[i]])).name,
            type=int(model.jnt_type[i]), qposadr=int(model.jnt_qposadr[i]),
            dofadr=int(model.jnt_dofadr[i])) for i in range(model.njnt)])
        ready = dict(case=case, object_name=object_name, initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                     model_sha256=model_hash, robot_pose=task.env.current_robot.robot_view.base.pose.tolist(),
                     initial_contacts=int(data.ncon), initial_nefc=int(data.nefc),
                     initial_robot_object_contacts=_robot_articulation_contact_stats(model, data, root_id),
                     controller_types={k: type(c).__name__ for k, c in robot.controllers.items()},
                     controller_targets={k: np.asarray(c.target).tolist() for k, c in robot.controllers.items()},
                     optimized=args.optimized)
        write_json(args.output / 'ready.json', ready)
        if args.prepare_only:
            return
        deadline = time.monotonic() + 600
        while not args.gate.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Waiting for synchronized wave start')
            time.sleep(.1)
        qpos, qvel, phases, contacts = [], [], [], []
        observed_qpos, observed_qvel = [], []
        contact_lookup = _build_robot_contact_lookup(model)
        frames = (args.output / 'frames.jsonl').open('w', buffering=1)
        image_hashes = []

        def step(controller, index):
            phases.append((controller._pending or {}).get('phase', 'restore'))
            observation = task.get_observations()
            if args.lock_experiment:
                image_hashes.append(observation_hashes(observation))
            quality = {}
            if args.adaptive or args.position_forward or args.force_frequency_factor or args.lock_experiment:
                from scripts.InteractiveNav.fixed_interaction_quality import robot_contacts
                observed_qpos.append(data.qpos.copy())
                observed_qvel.append(data.qvel.copy())
                if phases[-1] == 'observe':
                    task.invalidate_private_realtime_gt_segmentation_snapshot()
                    visible, distance, fraction = br.target_metrics(task, episode)
                    quality.update(target_visible=bool(visible), target_distance=float(distance),
                                   target_fraction=float(fraction))
                quality.update(robot_contacts=robot_contacts(data, contact_lookup),
                               group_index=int((controller._pending or {}).get('group_index', -1)))
            torso = controller.view_torso_target()
            action = {} if torso is None else {'torso': np.asarray(torso, dtype=float)}
            # Keep ordinary robot control/physics and camera observations, but not
            # navigation rewards, task-success early exits, or ROS/Qwen waiting.
            task._apply_action(action)
            observation = task.get_observations()
            if args.lock_experiment:
                image_hashes.append(observation_hashes(observation))
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            contacts.append([int(data.ncon), int(data.nefc)])
            torso_group = robot.robot_view.get_move_group('torso')
            frames.write(json.dumps(dict(index=index, phase=phases[-1], ncon=int(data.ncon),
                nefc=int(data.nefc), solver_iterations=int(data.solver_niter.max()),
                torso_qpos=torso_group.joint_pos.tolist(), torso_ctrl=torso_group.ctrl.tolist(),
                torso_target=torso,
                quality=quality,
                selected_qpos=[float(data.qpos[model.jnt_qposadr[j]]) for j in joint_ids])) + '\n')
            br._discard_task_rollout_cache(task)
            return False

        started = time.perf_counter()
        cpu_started = time.process_time()
        sim_started = float(data.time)
        frequency_context = nullcontext()
        if args.optimized and args.lock_experiment in ('camera_batch', 'combined'):
            from scripts.InteractiveNav import force_interaction_runtime as runtime
            from scripts.InteractiveNav.fixed_interaction_compute import BatchForceCameras
            frequency_context = BatchForceCameras(runtime, task.env)
        if args.force_frequency_factor:
            from scripts.InteractiveNav import force_interaction_runtime as runtime
            from scripts.InteractiveNav.fixed_interaction_frequency import ForceFrequencyReplay
            frequency_context = ForceFrequencyReplay(runtime,
                args.baseline_from / 'baseline' / str(case['episode']),
                args.force_frequency_factor, args.output)
            write_json(args.output / 'frequency_config.json', dict(
                baseline_timestep=float(model.opt.timestep), factor=args.force_frequency_factor,
                integrator=int(model.opt.integrator),
                geom_solref_below_two_candidate_steps=int(np.sum(
                    (model.geom_solref[:, 0] > 0) &
                    (model.geom_solref[:, 0] < 2 * model.opt.timestep * args.force_frequency_factor))),
                ordinary_control_dt_ms=task._ctrl_dt_ms,
                ordinary_sim_steps_per_control=task._n_sim_steps_per_ctrl,
                control_steps_per_policy=task._n_ctrl_steps_per_policy,
                scope='Only force drives; exact baseline drive durations, residual final step; force PD rate also reduced'))
        try:
            execute = run_view_cycle if args.view_only else run_smooth_interaction
            with frequency_context:
                execution = profile_interaction(execute, args.output / 'profile',
                    task=task, command=case['command'], object_name=object_name, step=step)
        finally:
            frames.close()
        elapsed = time.perf_counter() - started
        cpu = time.process_time() - cpu_started
        if args.lock_experiment:
            write_json(args.output / 'observation_hashes.json', image_hashes)
        np.savez_compressed(args.output / 'trajectory.npz', qpos=qpos, qvel=qvel, contacts=contacts,
                            observed_qpos=observed_qpos, observed_qvel=observed_qvel,
                            final_qpos=data.qpos.copy(), final_qvel=data.qvel.copy())
        write_json(args.output / 'execution.json', execution)
        write_json(args.output / 'summary.json', dict(episode=case['episode'], optimized=args.optimized,
            mode='view_cycle_preflight' if args.view_only else 'full_drawer_scan',
            elapsed_seconds=elapsed, cpu_seconds=cpu, phases=phases,
            simulated_seconds=float(data.time) - sim_started,
            inner_steps=execution['task_steps_consumed'], result_success=execution['result'].get('success'),
            physics_substeps=execution['result'].get('physics_substeps'),
            state=execution['result'].get('state'), joint_count=len(joint_ids)))
    finally:
        variants['base'] = source
        if sampler._env is not None:
            sampler._env.close()


def compare(output, manifest):
    import numpy as np
    rows = []
    for index, case in enumerate(manifest):
        base = output / 'baseline' / str(case['episode'])
        opt = output / 'optimized' / str(case['episode'])
        a, b = [json.loads((p / 'summary.json').read_text()) for p in (base, opt)]
        initial_a, initial_b = [json.loads((p / 'ready.json').read_text()) for p in (base, opt)]
        initial_mismatches = [key for key in INITIAL_CONDITION_KEYS
                              if key not in initial_a or key not in initial_b
                              or initial_a[key] != initial_b[key]]
        x, y = [np.load(p / 'trajectory.npz') for p in (base, opt)]
        differences = {}
        for key in x.files:
            differences[key] = (float(np.max(np.abs(x[key] - y[key]), initial=0))
                                if x[key].shape == y[key].shape else None)
        frames_equal = None
        if all((p / 'frames.jsonl').exists() for p in (base, opt)):
            frames_a, frames_b = [[json.loads(s) for s in (p / 'frames.jsonl').read_text().splitlines()]
                                  for p in (base, opt)]
            frames_equal = frames_a == frames_b
        observations_equal = None
        if all((p / 'observation_hashes.json').exists() for p in (base, opt)):
            image_a, image_b = [json.loads((p / 'observation_hashes.json').read_text()) for p in (base, opt)]
            observations_equal = bool(image_a) and all(image_a) and image_a == image_b
        equivalent = (not initial_mismatches
                      and frames_equal is not False
                      and observations_equal is not False
                      and a['phases'] == b['phases'] and a['physics_substeps'] == b['physics_substeps']
                      and a['inner_steps'] == b['inner_steps'] and a['state'] == b['state']
                      and a['result_success'] is True and b['result_success'] is True
                      and all(v is not None and v <= 1e-8 for v in differences.values()))
        rows.append(dict(episode=case['episode'], baseline_seconds=a['elapsed_seconds'],
                         optimized_seconds=b['elapsed_seconds'], equivalent=equivalent,
                         max_differences=differences, inner_steps=a['inner_steps'],
                         frames_equal=frames_equal,
                         observation_arrays_equal=observations_equal,
                         initial_mismatches=initial_mismatches,
                         speedup=a['elapsed_seconds'] / b['elapsed_seconds']))
    report = dict(protocol='fixed_interaction_only_v1', episodes=rows,
                  all_equivalent=all(x['equivalent'] for x in rows))
    write_json(output / 'comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def reuse_baseline(source, output, cases, expected_flags, wave='baseline'):
    source = source.resolve()
    config = json.loads((source / 'run_config.json').read_text())
    if wave == 'optimized' and (config.get('lock_experiment') or config.get('force_frequency_factor')):
        raise ValueError('Experimental candidate cannot be reused as an unmodified baseline')
    if config['baseline' if wave == 'baseline' else 'candidate'] != expected_flags or json.loads((source / 'manifest.json').read_text()) != cases:
        raise ValueError('Baseline configuration or manifest mismatch')
    for case in cases:
        path = source / wave / str(case['episode'])
        summary = json.loads((path / 'summary.json').read_text())
        if summary['result_success'] is not True or summary.get('mode') != 'full_drawer_scan':
            raise ValueError('Only successful full-scan baselines can be reused')
        for name in ('ready.json', 'frames.jsonl', 'trajectory.npz', 'execution.json'):
            if not (path / name).is_file():
                raise ValueError(f'Incomplete baseline: {path / name}')
    baseline_seconds = json.loads((source / 'wave_times.json').read_text())[wave]
    (output / 'baseline').symlink_to(source / wave, target_is_directory=True)
    write_json(output / 'baseline_reference.json', dict(source=str(source),
        wave=wave, note='Reused prior measurements; not a newly executed baseline wave'))
    return {'baseline': baseline_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--view-only', action='store_true', help='Preflight low-view/restore only, no drawer motion')
    parser.add_argument('--adaptive', action='store_true', help='Compare intermediate position-only settling; both waves coalesce locks')
    parser.add_argument('--position-forward', action='store_true', help='Keep physics steps; only refresh position geometry during robot locks')
    parser.add_argument('--baseline-from', type=Path, help='Reuse an explicitly configured, successful baseline read-only')
    parser.add_argument('--baseline-wave', choices=('baseline', 'optimized'), default='baseline')
    parser.add_argument('--lock-experiment', choices=('reference', 'camera_batch', 'geometry_only', 'combined'))
    parser.add_argument('--force-frequency-factor', type=int, choices=(1, 2, 4),
                        help='Replay baseline force durations with larger force-loop timestep; requires --baseline-from')
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    args.benchmark = args.benchmark.resolve()
    if args.adaptive and args.position_forward:
        parser.error('Compare one optimization at a time')
    if args.lock_experiment and (args.adaptive or args.position_forward or args.force_frequency_factor):
        parser.error('Compute experiments cannot be combined with other experiment modes')
    if args.baseline_wave != 'baseline' and (not args.baseline_from or args.force_frequency_factor):
        parser.error('Alternative baseline wave requires reuse and is not supported for frequency replay')
    if args.force_frequency_factor and (not args.baseline_from or args.adaptive or args.position_forward
                                       or args.prepare_only or args.view_only):
        parser.error('Force frequency replay requires a baseline and a full scan with no other variant')
    if args.baseline_from and (args.prepare_only or args.view_only):
        parser.error('Baseline reuse is only for full scans')
    if args.worker:
        worker(args)
        return
    cases = json.loads(args.manifest.read_text())
    if len(cases) != args.workers or len({c['episode'] for c in cases}) != len(cases):
        parser.error('This synchronized test requires one distinct case per worker')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'manifest.json', cases)
    baseline_flags = dict(coalesce_robot_lock=bool(args.adaptive or args.position_forward or args.force_frequency_factor or args.lock_experiment),
                          intermediate_position_only=False, lock_position_forward=bool(args.lock_experiment))
    write_json(args.output / 'run_config.json', dict(baseline=baseline_flags,
        candidate=dict(coalesce_robot_lock=True, intermediate_position_only=args.adaptive,
                       lock_position_forward=bool(args.position_forward or args.lock_experiment)),
        force_frequency_factor=args.force_frequency_factor, lock_experiment=args.lock_experiment))
    wave_times = reuse_baseline(args.baseline_from, args.output, cases, baseline_flags, args.baseline_wave) if args.baseline_from else {}
    for wave in (('optimized',) if args.baseline_from else ('baseline', 'optimized')):
        directory = args.output / wave
        directory.mkdir()
        gate = directory / 'start'
        children, logs = [], []
        try:
            for index, case in enumerate(cases):
                log = (directory / f"{case['episode']}.log").open('wb')
                logs.append(log)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                           '--manifest', str(args.manifest), '--benchmark', str(args.benchmark),
                           '--output', str(directory / str(case['episode'])), '--case-index', str(index),
                           '--gate', str(gate)]
                if wave == 'optimized':
                    command.append('--optimized')
                if args.prepare_only:
                    command.append('--prepare-only')
                if args.view_only:
                    command.append('--view-only')
                if args.adaptive:
                    command.append('--adaptive')
                if args.position_forward:
                    command.append('--position-forward')
                if args.lock_experiment:
                    command.extend(['--lock-experiment', args.lock_experiment])
                if args.force_frequency_factor:
                    command.extend(['--force-frequency-factor', str(args.force_frequency_factor),
                                    '--baseline-from', str(args.baseline_from.resolve())])
                children.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + 600
            while not all((directory / str(c['episode']) / 'ready.json').exists() for c in cases):
                if any(p.poll() is not None and p.returncode != 0 for p in children):
                    raise RuntimeError(f'{wave} worker failed during setup; inspect logs')
                if time.monotonic() > deadline:
                    raise TimeoutError('Scene preparation timeout')
                time.sleep(1)
            if wave == 'optimized':
                for case in cases:
                    a, b = [json.loads((args.output / v / str(case['episode']) / 'ready.json').read_text())
                            for v in ('baseline', 'optimized')]
                    for key in INITIAL_CONDITION_KEYS:
                        if a[key] != b[key]:
                            raise RuntimeError(f"Initial condition mismatch: episode {case['episode']} {key}")
            action = 'preparation verified' if args.prepare_only else 'starting synchronized interaction'
            print(f'{wave}: all {len(cases)} workers ready; {action}', flush=True)
            started = time.perf_counter()
            gate.write_text('start\n')
            while any(p.poll() is None for p in children):
                if any(p.poll() is not None and p.returncode != 0 for p in children):
                    raise RuntimeError(f'{wave} worker failed during interaction; inspect logs')
                if time.perf_counter() - started > 3600:
                    raise TimeoutError(f'{wave} interaction wave exceeded one hour')
                time.sleep(2)
            if any(p.returncode for p in children):
                raise RuntimeError(f'{wave} worker failed; inspect logs')
            wave_times[wave] = time.perf_counter() - started
            write_json(args.output / 'wave_times.json', wave_times)
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
            for p in children:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for log in logs:
                log.close()
    if not args.prepare_only:
        if args.force_frequency_factor:
            from scripts.InteractiveNav.fixed_interaction_frequency import compare_frequency
            if not compare_frequency(args.output, cases)['all_effect_checks_pass']:
                raise SystemExit('Frequency effect/clock checks failed; do not enable the candidate')
        elif args.adaptive:
            from scripts.InteractiveNav.fixed_interaction_quality import compare_effects
            if not compare_effects(args.output, cases)['all_effect_checks_pass']:
                raise SystemExit('Effect checks failed; do not enable the candidate')
        elif not compare(args.output, cases)['all_equivalent']:
            raise SystemExit('State/frame/observation equivalence failed; do not claim equivalent speedup')


if __name__ == '__main__':
    main()
