#!/usr/bin/env python3
"""Run the interactive-navigation interface against a real MuJoCo scene.

This is intentionally a lightweight smoke test: it loads one local scene,
attaches RBY1, finds collision-free poses near a door and a container, and
then exercises scan, door opening, and per-joint container actuation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import RBY1MOpenCloseConfig
from molmo_spaces.env.interaction_interface import (
    InteractionObject,
    SimulatorInteractionInterface,
)
from molmo_spaces.env.object_manager import ObjectManager
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView


class SmokeTestEnv:
    """Small subset of BaseMujocoEnv needed by ObjectManager and the interface."""

    def __init__(self, model, data, scene_path: Path, metadata: dict, robot_config):
        self.config = SimpleNamespace(robot_config=robot_config)
        self.current_model = model
        self.current_model_path = scene_path
        self.current_scene_metadata = metadata
        self.current_batch_index = 0
        self.mj_datas = [data]
        self.object_managers = [ObjectManager(self, 0)]


def _load_metadata(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_scene(scene_path: Path, metadata_path: Path | None):
    spec = mujoco.MjSpec.from_file(str(scene_path))
    robot_config = RBY1MOpenCloseConfig()
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=robot_config.robot_namespace,
        pos=[0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        randomize_textures=False,
    )
    robot_config.robot_cls.apply_control_overrides(spec, robot_config)
    model = spec.compile()
    data = mujoco.MjData(model)
    robot_view = RBY1RobotView(
        data,
        robot_config.robot_namespace,
        holo_base=robot_config.use_holo_base,
    )
    for group_name, joint_pos in robot_config.init_qpos.items():
        if group_name == "base":
            continue
        robot_view.get_move_group(group_name).joint_pos = joint_pos
    mujoco.mj_forward(model, data)
    env = SmokeTestEnv(
        model,
        data,
        scene_path,
        _load_metadata(metadata_path),
        robot_config,
    )
    return env, robot_view


def _pose_facing(position_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    yaw = math.atan2(
        target_xy[1] - position_xy[1], target_xy[0] - position_xy[0]
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cosine, -sine, 0.0, position_xy[0]],
            [sine, cosine, 0.0, position_xy[1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _robot_has_collision(env: SmokeTestEnv, robot_view, pose: np.ndarray) -> bool:
    original_pose = robot_view.base.pose.copy()
    robot_view.base.pose = pose
    mujoco.mj_forward(env.current_model, env.mj_datas[0])
    try:
        model, data = env.current_model, env.mj_datas[0]
        namespace = env.config.robot_config.robot_namespace
        for contact in data.contact[: data.ncon]:
            if contact.dist > 0:
                continue
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            root1 = model.body_rootid[body1]
            root2 = model.body_rootid[body2]
            name1 = model.body(root1).name
            name2 = model.body(root2).name
            robot1 = name1.startswith(namespace)
            robot2 = name2.startswith(namespace)
            if not (robot1 or robot2) or (robot1 and robot2):
                continue
            other_body = body2 if robot1 else body1
            other_name = model.body(other_body).name.lower()
            if "floor" not in other_name:
                return True
        return False
    finally:
        robot_view.base.pose = original_pose
        mujoco.mj_forward(env.current_model, env.mj_datas[0])


def _joint_target_xy(env: SmokeTestEnv, joint_name: str) -> np.ndarray:
    joint_id = env.current_model.joint(joint_name).id
    body_id = env.current_model.jnt_bodyid[joint_id]
    return env.mj_datas[0].xpos[body_id, :2].copy()


def _find_interaction_pose(
    env: SmokeTestEnv,
    robot_view,
    interface: SimulatorInteractionInterface,
    obj: InteractionObject,
    joint_index: int,
) -> np.ndarray:
    joint = next(item for item in obj.joints if item.index == joint_index)
    interface.set_joint_open_fraction(obj.name, joint.index, 0.0)
    target_xy = _joint_target_xy(env, joint.name)
    for radius in (0.75, 0.9, 1.05, 1.2, 1.35):
        for angle in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
            position = target_xy + radius * np.array(
                [math.cos(angle), math.sin(angle)]
            )
            pose = _pose_facing(position, target_xy)
            if _robot_has_collision(env, robot_view, pose):
                continue
            interface.set_joint_open_fraction(obj.name, joint.index, 1.0)
            opened_collision = _robot_has_collision(env, robot_view, pose)
            interface.set_joint_open_fraction(obj.name, joint.index, 0.0)
            if not opened_collision:
                return pose
    raise RuntimeError(
        f"No collision-free interaction pose found for {obj.name} joint {joint.index}"
    )


def _pose_record(pose: np.ndarray) -> dict:
    return {
        "x": float(pose[0, 3]),
        "y": float(pose[1, 3]),
        "yaw": float(math.atan2(pose[1, 0], pose[0, 0])),
    }


def _exercise_joint(
    env: SmokeTestEnv,
    robot_view,
    interface: SimulatorInteractionInterface,
    obj: InteractionObject,
    joint_index: int,
    *,
    use_door_api: bool,
) -> dict:
    pose = _find_interaction_pose(env, robot_view, interface, obj, joint_index)
    robot_view.base.pose = pose
    mujoco.mj_forward(env.current_model, env.mj_datas[0])
    if _robot_has_collision(env, robot_view, pose):
        raise AssertionError("Selected interaction pose is in collision")

    initial = interface.scan()
    initial_obj = next(item for item in initial if item.name == obj.name)
    sibling_positions = {
        joint.index: joint.position
        for joint in initial_obj.joints
        if joint.index != joint_index
    }
    observed = []
    for fraction in (0.0, 0.5, 1.0, 0.0):
        if use_door_api:
            state = interface.open_door(obj.name, fraction)
        else:
            state = interface.set_joint_open_fraction(obj.name, joint_index, fraction)
        actual = next(
            joint.open_fraction for joint in state.joints if joint.index == joint_index
        )
        if not math.isclose(actual, fraction, abs_tol=1e-6):
            raise AssertionError(f"Requested {fraction}, observed {actual}")
        for sibling in state.joints:
            if sibling.index in sibling_positions and not math.isclose(
                sibling.position, sibling_positions[sibling.index], abs_tol=1e-9
            ):
                raise AssertionError(
                    f"Actuating joint {joint_index} moved sibling {sibling.index}"
                )
        observed.append(actual)
    return {
        "object": obj.name,
        "kind": obj.kind,
        "joint_index": joint_index,
        "joint_name": next(
            item.name for item in obj.joints if item.index == joint_index
        ),
        "robot_pose": _pose_record(pose),
        "fractions": observed,
        "api": "open_door" if use_door_api else "set_joint_open_fraction",
    }


def run(scene_path: Path, metadata_path: Path | None) -> dict:
    env, robot_view = _load_scene(scene_path, metadata_path)
    interface = SimulatorInteractionInterface(env)
    objects = interface.scan()
    doors = [obj for obj in objects if obj.kind == "door"]
    containers = [obj for obj in objects if obj.kind == "container"]
    if not doors:
        raise RuntimeError("Scene exposes no door through the interaction interface")
    if not containers:
        raise RuntimeError("Scene exposes no container through the interaction interface")

    actions = [
        _exercise_joint(env, robot_view, interface, doors[0], 0, use_door_api=True)
    ]
    container = max(containers, key=lambda item: len(item.joints))
    actions.extend(
        _exercise_joint(
            env,
            robot_view,
            interface,
            container,
            joint.index,
            use_door_api=False,
        )
        for joint in container.joints
    )
    return {
        "scene": str(scene_path),
        "scan": [
            {
                "name": obj.name,
                "category": obj.category,
                "kind": obj.kind,
                "joint_count": len(obj.joints),
            }
            for obj in objects
        ],
        "actions": actions,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Keep the lexical scene path: local scene mirrors commonly use a symlinked
    # XML so MuJoCo's relative mesh paths resolve through the mirror layout.
    result = run(
        args.scene_xml.absolute(),
        args.metadata.resolve() if args.metadata else None,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
