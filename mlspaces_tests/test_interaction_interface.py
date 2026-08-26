from types import SimpleNamespace

import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import RBY1Config
from molmo_spaces.env.data_views import MlSpacesArticulationObject
from molmo_spaces.env.interaction_interface import SimulatorInteractionInterface
from molmo_spaces.policy.interactive_nav_sim_policy import (
    InteractionDemoCommand,
    InteractiveNavSimulatorPolicy,
)


class FakeArticulation(MlSpacesArticulationObject):
    def __init__(self, name, joint_names, joint_types, joint_ranges, positions):
        self._name = name
        self.joint_names = list(joint_names)
        self._joint_types = list(joint_types)
        self._joint_ranges = list(joint_ranges)
        self._positions = list(positions)

    @property
    def njoints(self):
        return len(self.joint_names)

    def get_joint_type(self, index):
        return self._joint_types[index]

    def get_joint_range(self, index):
        return self._joint_ranges[index]

    def get_joint_position(self, index):
        return self._positions[index]

    def set_joint_position(self, index, position):
        self._positions[index] = float(position)


class FakeDoor(FakeArticulation):
    def get_hinge_joint_index(self):
        return 1


class FakeObjectManager:
    def __init__(self, objects, door_names=()):
        self.objects = {obj.name: obj for obj in objects}
        self.door_names = set(door_names)
        self.invalidations = 0

    def find_door_names(self):
        return sorted(self.door_names)

    def list_top_level_objects(self):
        return [obj for name, obj in self.objects.items() if name not in self.door_names]

    def get_object_by_name(self, name):
        return self.objects[name]

    def object_metadata(self, name):
        category = "Cabinet" if "cabinet" in name else "Door"
        return {"category": category}

    def invalidate_data_cache(self):
        self.invalidations += 1


class FakeRobotView:
    def get_noop_ctrl_dict(self):
        return {"base": np.zeros(3, dtype=np.float32)}


def _fake_env(objects, door_names=()):
    manager = FakeObjectManager(objects, door_names)
    env = SimpleNamespace(
        current_batch_index=0,
        object_managers=[manager],
        mj_datas=[object()],
        current_model=object(),
        current_robot=SimpleNamespace(robot_view=FakeRobotView()),
    )
    return env, manager


def test_rby1_navigation_range_covers_large_procthor_scenes() -> None:
    assert RBY1Config().holo_base_position_limit_m == 100.0


def test_scan_lists_container_joints_and_normalized_state() -> None:
    cabinet = FakeArticulation(
        "kitchen_cabinet",
        ["left_door", "top_drawer"],
        [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE],
        [(-1.0, 1.0), (0.0, 0.4)],
        [0.0, 0.2],
    )
    env, _ = _fake_env([cabinet])
    interface = SimulatorInteractionInterface(env, forward=lambda *_: None)

    catalog = interface.scan()

    assert len(catalog) == 1
    assert catalog[0].kind == "container"
    assert [joint.name for joint in catalog[0].joints] == ["left_door", "top_drawer"]
    assert np.isclose(catalog[0].joints[1].open_fraction, 0.5)


def test_open_one_drawer_does_not_move_sibling_joint() -> None:
    cabinet = FakeArticulation(
        "kitchen_cabinet",
        ["left_drawer", "right_drawer"],
        [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_SLIDE],
        [(0.0, 0.4), (0.0, 0.6)],
        [0.0, 0.0],
    )
    env, manager = _fake_env([cabinet])
    forward_calls = []
    interface = SimulatorInteractionInterface(
        env, forward=lambda *args: forward_calls.append(args)
    )

    result = interface.set_joint_open_fraction("kitchen_cabinet", 1, 0.5)

    assert cabinet._positions == [0.0, 0.3]
    assert np.isclose(result.joints[1].open_fraction, 0.5)
    assert len(forward_calls) == 1
    assert manager.invalidations == 1


def test_open_door_uses_hinge_and_leaves_handle_untouched() -> None:
    door = FakeDoor(
        "hall_door",
        ["handle", "hinge"],
        [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_HINGE],
        [(-0.5, 0.5), (0.0, 1.5)],
        [0.0, 0.0],
    )
    env, _ = _fake_env([door], door_names=[door.name])
    interface = SimulatorInteractionInterface(
        env,
        forward=lambda *_: None,
        door_factory=lambda *_: door,
    )

    result = interface.open_door("hall_door")

    assert door._positions == [0.0, 1.5]
    assert result.kind == "door"


def test_demo_policy_scans_then_runs_explicit_single_joint_command() -> None:
    cabinet = FakeArticulation(
        "kitchen_cabinet",
        ["drawer"],
        [mujoco.mjtJoint.mjJNT_SLIDE],
        [(0.0, 0.5)],
        [0.0],
    )
    env, _ = _fake_env([cabinet])
    config = SimpleNamespace(policy_config=SimpleNamespace(force_enable_depth=False))
    task = SimpleNamespace(env=env)
    policy = InteractiveNavSimulatorPolicy(
        config,
        task,
        commands=(InteractionDemoCommand.open_joint("kitchen_cabinet", 0),),
        interface_factory=lambda current_env: SimulatorInteractionInterface(
            current_env, forward=lambda *_: None
        ),
    )

    policy.reset()
    action = policy.get_action({})

    assert len(policy.catalog) == 1
    assert cabinet._positions == [0.5]
    assert action["done"] is True
    assert policy.get_info()["available_interfaces"]
