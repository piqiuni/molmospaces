"""Conservative fixed-scan acceptance checks; these do not establish navigation SR."""
import json
from collections import Counter
import numpy as np


def robot_contacts(data, lookup):
    geoms = np.asarray(data.contact.geom)[:int(data.ncon)]
    selected = np.flatnonzero(lookup.robot_geoms[geoms].any(axis=1))
    distances = [float(data.contact[int(i)].dist) for i in selected]
    return dict(count=len(selected), penetration=max([0.0] + [-x for x in distances]))


def max_delta(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return float(np.max(np.abs(a - b), initial=0))


def within(value, tolerance):
    return value is not None and value <= tolerance


def compare_effects(output, manifest):
    from scripts.InteractiveNav.run_fixed_interaction_test import INITIAL_CONDITION_KEYS, write_json
    rows = []
    for case in manifest:
        paths = [output / v / str(case['episode']) for v in ('baseline', 'optimized')]
        a, b = [json.loads((p / 'summary.json').read_text()) for p in paths]
        ready = [json.loads((p / 'ready.json').read_text()) for p in paths]
        executions = [json.loads((p / 'execution.json').read_text())['result'] for p in paths]
        frames = [[json.loads(line) for line in (p / 'frames.jsonl').read_text().splitlines()] for p in paths]
        arrays = [np.load(p / 'trajectory.npz') for p in paths]
        indexes = [[i for i, f in enumerate(fs) if f['phase'] == 'observe'] for fs in frames]
        observations = [[fs[i] for i in idx] for fs, idx in zip(frames, indexes)]
        groups = [[f['quality']['group_index'] for f in fs] for fs in observations]
        fallback = [sum(bool(t['fallback']) for t in r['transition_log']) for r in executions]
        fallback_groups = [Counter((t.get('group_id'), t.get('phase'))
                                   for t in r['transition_log'] if t['fallback']) for r in executions]
        max_contacts = [max(f['quality']['robot_contacts']['count'] for f in fs) for fs in frames]
        max_penetration = [max(f['quality']['robot_contacts']['penetration'] for f in fs) for fs in frames]
        deltas = {key: max_delta(x[key][idx], y[key][idy])
                  for key in ('observed_qpos', 'observed_qvel')
                  for x, y, idx, idy in [(arrays[0], arrays[1], indexes[0], indexes[1])]}
        deltas.update({key: max_delta(arrays[0][key], arrays[1][key])
                       for key in ('final_qpos', 'final_qvel')})
        visibility_losses = sum(
            x['quality']['target_fraction'] - y['quality']['target_fraction'] > 1 / (640 * 480) + 1e-12
            or (x['quality']['target_visible'] and not y['quality']['target_visible'])
            for x, y in zip(*observations))
        checks = dict(
            initial_conditions=all(k in ready[0] and k in ready[1] and ready[0][k] == ready[1][k]
                                   for k in INITIAL_CONDITION_KEYS),
            both_succeeded=a['result_success'] is True and b['result_success'] is True,
            final_state=a['state'] == b['state'] == 'closed',
            observations=bool(groups[0]) and groups[0] == groups[1],
            view_restored=all(r['view_restore_convergence']['converged'] for r in executions),
            fallback_not_increased=fallback[1] <= fallback[0],
            per_group_fallback_not_increased=all(count <= fallback_groups[0][key]
                                                 for key, count in fallback_groups[1].items()),
            visibility_not_reduced=visibility_losses == 0,
            observation_positions=within(deltas['observed_qpos'], .01),
            observation_velocities=within(deltas['observed_qvel'], .08),
            final_positions=within(deltas['final_qpos'], .01),
            final_velocities=within(deltas['final_qvel'], .08),
            robot_contact_count=max_contacts[1] <= max_contacts[0],
            robot_penetration=max_penetration[1] <= max_penetration[0] + .0001,
            fewer_physics_steps=b['physics_substeps'] < a['physics_substeps'],
            faster=b['elapsed_seconds'] < a['elapsed_seconds'],
        )
        rows.append(dict(episode=case['episode'], checks=checks, passed=all(checks.values()),
                         baseline_seconds=a['elapsed_seconds'], candidate_seconds=b['elapsed_seconds'],
                         baseline_physics_steps=a['physics_substeps'], candidate_physics_steps=b['physics_substeps'],
                         fallback_counts=fallback, deltas=deltas, visibility_losses=visibility_losses,
                         robot_max_contacts=max_contacts, robot_max_penetration=max_penetration))
    report = dict(protocol='fixed_effect_checks_v1', episodes=rows,
                  all_effect_checks_pass=all(row['passed'] for row in rows),
                  scope='task-step samples; not continuous collision or full navigation SR validation')
    write_json(output / 'effect_comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report
