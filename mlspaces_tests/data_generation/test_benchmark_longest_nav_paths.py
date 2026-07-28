"""Regression coverage for V3 target identity during nav-goal sampling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.InteractiveNav import benchmark_longest_nav_paths as nav_paths


def test_sample_nav_goal_returns_the_candidate_used_by_the_goal_sampler(monkeypatch) -> None:
    source_target = SimpleNamespace(name="source_toilet", position=np.array([8.0, 8.0, 0.0]))
    selected_target = SimpleNamespace(name="nearest_toilet", position=np.array([2.0, 2.0, 0.0]))

    monkeypatch.setattr(
        nav_paths,
        "episode_nav_objects",
        lambda _env, _episode: ([source_target, selected_target], [source_target.name, selected_target.name], source_target.name),
    )
    monkeypatch.setattr(nav_paths, "nearest_nav_object", lambda _env, _objects: selected_target)
    monkeypatch.setattr(nav_paths.emi, "normalize_point3d", lambda point: np.asarray(point, dtype=float))

    class _Sampler:
        def __init__(self, *_args, **_kwargs) -> None:
            self.target = None

        def set_target(self, target) -> None:
            self.target = target

        def set_robot_view(self, _robot_view) -> None:
            pass

        def sample(self):
            assert self.target is selected_target
            return np.array([2.5, 2.25, 0.0])

    monkeypatch.setattr(nav_paths, "NavGoalSampler", _Sampler)
    env = SimpleNamespace(current_robot=SimpleNamespace(robot_view=object()))

    goal, source, error, target_name, candidates = nav_paths.sample_nav_goal_for_episode(env, object(), {})

    assert np.allclose(goal, [2.5, 2.25, 0.0])
    assert source == "nav_goal_sampler"
    assert error is None
    assert target_name == "nearest_toilet"
    assert candidates == ["source_toilet", "nearest_toilet"]
