from types import SimpleNamespace

import numpy as np
import pytest

from scripts.InteractiveNav.evaluation import benchmark_runner  # Initialize replay dependencies.
from molmo_spaces.tasks import json_eval_task_sampler as module


@pytest.mark.parametrize("coordinate", [0.0, 10.0, 100.0])
@pytest.mark.parametrize("error,restored", [(0.0009, False), (0.00105, True)])
def test_replay_absolute_position_tolerance(monkeypatch, coordinate, error, restored):
    target = np.array([coordinate, 0., 0.])
    initial = target + [error, 0, 0]
    body = SimpleNamespace(position=initial.copy(), quat=np.array([0., 0., 0., 1.]))
    sampler = object.__new__(module.JsonEvalTaskSampler)
    sampler.episode_spec = SimpleNamespace(
        house_index=1, scene_dataset="test", data_split="val",
        scene_modifications=SimpleNamespace(object_poses={"object": [*target, 0, 0, 0, 1]}),
        robot=SimpleNamespace(init_qpos={}))
    sampler.config = SimpleNamespace(scene_dataset="test", data_split="val")
    monkeypatch.setattr(sampler, "_randomize_colors", lambda env: None)
    monkeypatch.setattr(sampler, "set_joint_values", lambda env: None)
    monkeypatch.setattr(module, "create_mlspaces_body", lambda *a: body)
    monkeypatch.setattr(module.mujoco, "mj_resetData", lambda *a: None)
    monkeypatch.setattr(module.mujoco, "mj_forward", lambda *a: None)
    sampler.randomize_scene(SimpleNamespace(current_model=None, current_data=None, robots=[]), None)
    np.testing.assert_array_equal(body.position, target if restored else initial)
