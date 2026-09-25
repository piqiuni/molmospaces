from types import SimpleNamespace

import numpy as np

from scripts.InteractiveNav.evaluation import benchmark_runner as br
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler


def test_replay_pose_and_controls_are_ready_before_settling(monkeypatch):
    events = []
    groups = {name: SimpleNamespace(joint_pos=None) for name in ("base", "head")}
    view = SimpleNamespace(base=SimpleNamespace(pose=None), get_move_group=groups.__getitem__)
    robot = SimpleNamespace(
        robot_view=view,
        controllers={"head": SimpleNamespace(reset=lambda: events.append("reset"))},
        set_stationary=lambda: events.append("stationary"),
        compute_control=lambda: events.append("control"),
    )
    def forward(*args):
        events.append("forward")
    def factory(data):
        assert events == ["forward"]  # Robot controllers require valid world frames.
        events.append("factory")
        return robot
    sampler = object.__new__(br.V3BenchmarkTaskSampler)
    sampler.episode_spec = SimpleNamespace(
        robot=SimpleNamespace(init_qpos={"base": [1, 2, 0], "head": [0, .3]}),
        task={"robot_base_pose": [1, 2, 0, 0, 0, 0, 1]},
    )
    monkeypatch.setattr(sampler, "_create_robot", factory)
    monkeypatch.setattr(br.mujoco, "mj_forward", forward)
    sampler._initialize_before_settle(SimpleNamespace(model=object()))
    np.testing.assert_allclose(groups["head"].joint_pos, [0, .3])
    np.testing.assert_allclose(view.base.pose[:3, 3], [1, 2, 0])
    assert events == ["forward", "factory", "forward", "reset", "stationary", "control"]


def test_other_samplers_keep_noop_initialization():
    data = SimpleNamespace(qpos=np.array([1., 2.]))
    BaseMujocoTaskSampler._initialize_before_settle(None, data)
    np.testing.assert_array_equal(data.qpos, [1., 2.])
