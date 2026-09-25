from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from molmo_spaces.robots.robot_views.abstract import HoloJointsRobotBaseGroup
from scripts.InteractiveNav import force_interaction_bridge as bridge
from scripts.InteractiveNav import force_interaction_runtime as runtime


class JointGroup:
    def __init__(self, data, joint, actuator):
        self.data, self.joint, self.actuator = data, joint, actuator

    @property
    def joint_pos(self):
        return self.data.qpos[[self.joint]]

    @joint_pos.setter
    def joint_pos(self, value):
        self.data.qpos[[self.joint]] = value

    @property
    def joint_vel(self):
        return self.data.qvel[[self.joint]]

    @joint_vel.setter
    def joint_vel(self, value):
        self.data.qvel[[self.joint]] = value

    @property
    def ctrl(self):
        return self.data.ctrl[[self.actuator]]

    @ctrl.setter
    def ctrl(self, value):
        self.data.ctrl[[self.actuator]] = value

    @property
    def noop_ctrl(self):
        return self.joint_pos


def make_env(yaw=0.3):
    names = ('head', 'left_arm', 'right_arm', 'left_gripper', 'right_gripper')
    bodies = ''.join(
        f'<body name="{name}" pos="0 0 {0.2 + i * 0.1}">'
        f'<joint name="{name}" axis="0 1 0" damping="1"/>'
        '<geom size=".03" mass=".1"/></body>' for i, name in enumerate(names))
    actuators = ''.join(f'<position joint="{name}" kp="10"/>' for name in names)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0" timestep=".002"/>
      <worldbody><site name="world"/><body name="robot_base">
        <joint name="x" type="slide" axis="1 0 0"/>
        <joint name="y" type="slide" axis="0 1 0"/>
        <joint name="yaw" axis="0 0 1"/>
        <site name="base"/><geom size=".08" mass="1"/>{bodies}</body>
        <body name="target" pos=".4 0 0"><joint name="hinge" axis="0 0 1"
          range="0 90" damping="1"/><geom type="box" size=".2 .03 .1" pos=".2 0 0"/>
        </body></worldbody><actuator><position joint="x" kp="10"/>
        <position joint="y" kp="10"/><position joint="yaw" kp="10"/>{actuators}
      </actuator></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[2] = yaw
    data.qpos[3:8] = np.linspace(.05, .2, 5)
    data.ctrl[:] = data.qpos[:8]
    mujoco.mj_forward(model, data)
    base = HoloJointsRobotBaseGroup(data, model.site('world').id, model.site('base').id,
                                   [0, 1, 2], [0, 1, 2], model.body('robot_base').id)
    groups = {name: JointGroup(data, i + 3, i + 3) for i, name in enumerate(names)}
    reads, cameras = [], []
    def get_group(name):
        reads.append(name)
        return groups[name]
    view = SimpleNamespace(base=base, get_move_group=get_group)
    env = SimpleNamespace(current_model=model, current_data=data,
                          current_robot=SimpleNamespace(robot_view=view),
                          camera_manager=SimpleNamespace(registry=SimpleNamespace(
                              update_all_cameras=lambda _env: cameras.append(data.xpos.copy()))))
    return env, reads, cameras


@pytest.mark.parametrize('yaw', [0.3, np.pi - .01, -np.pi + .01])
def test_lock_cache_preserves_full_force_trajectory_and_camera_callbacks(yaw):
    outcomes = []
    for cached in (False, True):
        env, reads, cameras = make_env(yaw)
        snapshot = bridge._capture_robot_lock(env)
        assert snapshot is not None
        if not cached:
            snapshot.pop('_view_cache')
        reads.clear()
        trajectory = []
        def lock():
            bridge._apply_robot_lock(env, snapshot)
            d = env.current_data
            trajectory.append(np.concatenate([d.qpos, d.qvel, d.ctrl, d.qacc]).copy())
        result = runtime.drive_joint_group_to_targets(
            env.current_model, env.current_data, {'hinge': .35},
            config=runtime.ForceDriveConfig(max_physics_substeps=80), robot_lock_callback=lock)
        assert len(trajectory) == len(cameras) == 2 * result['physics_substeps']
        assert len(reads) == (0 if cached else 4 * len(trajectory))
        outcomes.append((result, np.asarray(trajectory), np.asarray(cameras)))
    assert outcomes[0][0] == outcomes[1][0]
    np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])
    np.testing.assert_array_equal(outcomes[0][2], outcomes[1][2])


@pytest.mark.parametrize('replace_part', ['model_data', 'view'])
def test_lock_view_cache_does_not_survive_environment_replacement(replace_part):
    old, _, old_cameras = make_env()
    snapshot = bridge._capture_robot_lock(old)
    fresh, reads, cameras = make_env(.8)
    if replace_part == 'view':
        fresh.current_model = old.current_model
        fresh.current_data = old.current_data
    before = old.current_data.qpos.copy()
    bridge._apply_robot_lock(fresh, snapshot)
    assert len(reads) == 4
    assert len(cameras) == 1 and not old_cameras
    np.testing.assert_array_equal(old.current_data.qpos, before)


def test_snapshot_values_are_not_mutated_by_repeated_locks():
    env, _, _ = make_env()
    snapshot = bridge._capture_robot_lock(env)
    expected = {name: tuple(value.copy() for value in values)
                for name, values in snapshot['groups'].items()}
    for _ in range(3):
        env.current_data.qpos[3] += .1
        head = env.current_data.qpos[3]
        bridge._apply_robot_lock(env, snapshot)
        assert env.current_data.qpos[3] == head
    for name, values in expected.items():
        for actual, original in zip(snapshot['groups'][name], values):
            np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize('contact', [False, True])
@pytest.mark.parametrize('mode', ['position', 'geometry'])
def test_position_forward_preserves_physics_and_camera_trajectory(contact, mode):
    outcomes = []
    for enabled in (False, True):
        env, _, cameras = make_env()
        if contact:
            env.current_data.qpos[0] = env.current_data.ctrl[0] = .39
            mujoco.mj_forward(env.current_model, env.current_data)
        snapshot = bridge._capture_robot_lock(env)
        snapshot['_position_forward'] = enabled
        snapshot['_geometry_forward'] = enabled and mode == 'geometry'
        trajectory, results = [], []
        def lock():
            bridge._apply_robot_lock(env, snapshot)
            d = env.current_data
            trajectory.append(np.concatenate([d.qpos, d.qvel, d.ctrl, [d.ncon]]).copy())
        for target in (.35, 0., .6):
            results.append(runtime.drive_joint_group_to_targets(
                env.current_model, env.current_data, {'hinge': target},
                runtime.ForceDriveConfig(max_physics_substeps=100, coalesce_robot_lock=True),
                robot_lock_callback=lock))
        outcomes.append((results, np.asarray(trajectory), np.asarray(cameras), env.current_data.qacc.copy()))
    assert outcomes[0][0] == outcomes[1][0]
    for a, b in zip(outcomes[0][1:], outcomes[1][1:]):
        np.testing.assert_array_equal(a, b)


def test_batched_cameras_preserve_force_states_and_final_camera():
    from contextlib import nullcontext
    from scripts.InteractiveNav.fixed_interaction_compute import BatchForceCameras
    outcomes = []
    original_drive = runtime.drive_joint_group_to_targets
    for batch in (False, True):
        env, _, cameras = make_env()
        snapshot = bridge._capture_robot_lock(env)
        snapshot['_position_forward'] = True
        states = []
        def lock():
            bridge._apply_robot_lock(env, snapshot)
            states.append(env.current_data.qpos.copy())
        with BatchForceCameras(runtime, env) if batch else nullcontext():
            result = runtime.drive_joint_group_to_targets(
                env.current_model, env.current_data, {'hinge': .35},
                runtime.ForceDriveConfig(max_physics_substeps=80, coalesce_robot_lock=True),
                robot_lock_callback=lock)
        assert runtime.drive_joint_group_to_targets is original_drive
        assert len(cameras) == (1 if batch else result['physics_substeps'] + 1)
        outcomes.append((np.array(states), cameras[-1], env.current_data.qvel.copy()))
    for a, b in zip(*outcomes):
        np.testing.assert_array_equal(a, b)


def test_batched_cameras_restore_callbacks_and_flush_on_failure(monkeypatch):
    from scripts.InteractiveNav.fixed_interaction_compute import BatchForceCameras
    env, _, cameras = make_env()
    registry = env.camera_manager.registry
    original_update = registry.update_all_cameras
    def fail():
        registry.update_all_cameras(env)
        raise RuntimeError('test failure')
    monkeypatch.setattr(runtime, 'drive_joint_group_to_targets', fail)
    with pytest.raises(RuntimeError, match='test failure'):
        with BatchForceCameras(runtime, env):
            runtime.drive_joint_group_to_targets()
    assert runtime.drive_joint_group_to_targets is fail
    assert registry.update_all_cameras is original_update
    assert len(cameras) == 1
