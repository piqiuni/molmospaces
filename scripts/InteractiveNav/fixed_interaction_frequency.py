"""Interaction-only timestep experiment with recorded per-drive clock replay."""
from dataclasses import replace
import json
import math


class ForceFrequencyReplay:
    def __init__(self, runtime, baseline, factor, output):
        if factor not in (1, 2, 4):
            raise ValueError('Expected timestep factor 1, 2 or 4')
        self.runtime, self.factor, self.output = runtime, factor, output
        result = json.loads((baseline / 'execution.json').read_text())['result']
        self.schedule = [int(t['physics_substeps']) for t in result['transition_log']
                         if int(t.get('physics_substeps', 0)) > 0]
        if sum(self.schedule) != result['physics_substeps'] or not self.schedule:
            raise ValueError('Incomplete baseline force schedule')
        self.original = runtime.drive_joint_group_to_targets
        self.records = []

    def __enter__(self):
        self.stream = (self.output / 'frequency_drives.jsonl').open('w', buffering=1)
        self.runtime.drive_joint_group_to_targets = self.drive
        return self

    def drive(self, model, data, targets, config=None, **kwargs):
        index = len(self.records)
        if index >= len(self.schedule):
            raise RuntimeError('Candidate requested an unexpected extra force drive')
        config = config or self.runtime.ForceDriveConfig()
        dt = float(model.opt.timestep)
        duration = self.schedule[index] * dt
        start = float(data.time)
        warnings = [int(w.number) for w in data.warning]
        candidate = replace(config, replay_duration_seconds=duration,
                            replay_stable_seconds=config.stable_substeps * dt)
        try:
            model.opt.timestep = dt * self.factor
            result = self.original(model, data, targets, candidate, **kwargs)
        finally:
            model.opt.timestep = dt
        elapsed = float(data.time) - start
        if not math.isclose(elapsed, duration, rel_tol=0, abs_tol=1e-7):
            raise RuntimeError(f'Force clock mismatch: {elapsed} versus {duration}')
        row = dict(index=index, targets=dict(targets), baseline_dt=dt,
                   candidate_dt=dt * self.factor, expected_seconds=duration,
                   actual_seconds=elapsed, baseline_steps=self.schedule[index],
                   candidate_steps=result['physics_substeps'], success=result['success'],
                   warning_deltas=[int(w.number) - n for w, n in zip(data.warning, warnings)])
        self.records.append(row)
        self.stream.write(json.dumps(row) + '\n')
        return result

    def __exit__(self, exc_type, exc, tb):
        self.runtime.drive_joint_group_to_targets = self.original
        self.stream.close()
        if exc_type is None and len(self.records) != len(self.schedule):
            raise RuntimeError('Candidate did not replay all baseline force drives')


def compare_frequency(output, cases):
    from scripts.InteractiveNav.fixed_interaction_quality import compare_effects
    from scripts.InteractiveNav.run_fixed_interaction_test import write_json
    report = compare_effects(output, cases)
    for row in report['episodes']:
        path = output / 'optimized' / str(row['episode'])
        records = [json.loads(s) for s in (path / 'frequency_drives.jsonl').read_text().splitlines()]
        config = json.loads((path / 'frequency_config.json').read_text())
        summary = json.loads((path / 'summary.json').read_text())
        baseline = json.loads((output / 'baseline' / str(row['episode']) / 'summary.json').read_text())
        force_seconds = baseline['physics_substeps'] * config['baseline_timestep']
        ordinary_seconds = (baseline['inner_steps'] * config['ordinary_control_dt_ms'] / 1000
                            * config['control_steps_per_policy'])
        row['checks'].update(
            force_clock_preserved=bool(records) and all(
                math.isclose(r['actual_seconds'], r['expected_seconds'], rel_tol=0, abs_tol=1e-7)
                for r in records) and math.isclose(sum(r['actual_seconds'] for r in records),
                                                   force_seconds, rel_tol=0, abs_tol=1e-7),
            full_clock_preserved=math.isclose(summary['simulated_seconds'], force_seconds + ordinary_seconds,
                                              rel_tol=0, abs_tol=1e-7),
            no_force_warnings=all(not any(r['warning_deltas']) for r in records))
        row['passed'] = all(row['checks'].values())
        row.update(simulated_seconds=summary['simulated_seconds'],
                   expected_simulated_seconds=force_seconds + ordinary_seconds,
                   candidate_cpu_seconds=summary['cpu_seconds'],
                   baseline_cpu_seconds=baseline['cpu_seconds'])
    report['protocol'] = 'fixed_force_frequency_duration_replay_v1'
    report['all_effect_checks_pass'] = all(row['passed'] for row in report['episodes'])
    report['timing_caveat'] = 'Candidates ran as two groups together (6 workers); reused baseline used 3 workers'
    write_json(output / 'frequency_comparison.json', report)
    return report
