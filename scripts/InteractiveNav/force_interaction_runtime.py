from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import mujoco
import numpy as np

from molmo_spaces.env.data_views import Door


@dataclass(frozen=True)
class ForceDriveConfig:
    max_physics_substeps: int = 1500
    stable_substeps: int = 8
    position_tolerance: float = 0.01
    velocity_tolerance: float = 0.01
    hinge_kp: float = 90.0
    hinge_kd: float = 12.0
    hinge_max_effort: float = 45.0
    slide_kp: float = 600.0
    slide_kd: float = 80.0
    slide_max_effort: float = 160.0
    open_fraction_threshold: float = 0.67
    assume_success: bool = True


class ForceInteractionError(RuntimeError):
    pass


def joint_closed_open_values(joint_range) -> tuple[float, float]:
    lower, upper = (float(joint_range[0]), float(joint_range[1]))
    if lower <= 0.0 <= upper:
        closed = 0.0
    else:
        closed = lower if abs(lower) <= abs(upper) else upper
    opened = lower if abs(lower - closed) >= abs(upper - closed) else upper
    return closed, opened


def joint_open_fraction(value: float, joint_range) -> float:
    closed, opened = joint_closed_open_values(joint_range)
    span = abs(opened - closed)
    if span <= 1e-8:
        return 0.0
    return float(np.clip(abs(float(value) - closed) / span, 0.0, 1.0))


def collect_door_root_groups(env) -> dict[str, dict[str, Any]]:
    model = env.current_model
    data = env.current_data
    object_manager = env.object_managers[env.current_batch_index]
    groups: dict[int, dict[str, Any]] = {}
    for door_name in object_manager.find_door_names():
        try:
            door = Door(door_name, data)
            hinge_index = int(door.get_hinge_joint_index())
            joint_name = str(door.joint_names[hinge_index])
            joint_range = [float(value) for value in door.get_joint_range(hinge_index)]
        except (KeyError, ValueError):
            continue
        body_id = int(model.body(door_name).id)
        root_id = int(model.body_rootid[body_id])
        group = groups.setdefault(
            root_id,
            {
                "root_body_id": root_id,
                "root_body_name": str(model.body(root_id).name or f"door_root_{root_id}"),
                "leaves": [],
            },
        )
        group["leaves"].append(
            {
                "leaf_body_name": str(door_name),
                "hinge_joint_index": hinge_index,
                "hinge_joint_name": joint_name,
                "joint_range": joint_range,
            }
        )
    return {
        str(group["root_body_name"]): group
        for group in sorted(groups.values(), key=lambda item: int(item["root_body_id"]))
    }


def set_all_door_roots_closed(env) -> list[dict[str, Any]]:
    groups = collect_door_root_groups(env)
    transitions = []
    for root_name, group in groups.items():
        leaf_transitions = []
        for leaf in group["leaves"]:
            door = Door(leaf["leaf_body_name"], env.current_data)
            hinge_index = int(leaf["hinge_joint_index"])
            before = float(door.get_joint_position(hinge_index))
            closed, _opened = joint_closed_open_values(leaf["joint_range"])
            door.set_joint_position(hinge_index, closed)
            leaf_transitions.append(
                {
                    **leaf,
                    "before": before,
                    "target": closed,
                }
            )
        transitions.append(
            {
                "root_body_name": root_name,
                "state": "closed",
                "leaf_transitions": leaf_transitions,
            }
        )
    mujoco.mj_forward(env.current_model, env.current_data)
    return transitions


def drive_joint_group_to_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_targets: Mapping[str, float],
    config: ForceDriveConfig | None = None,
) -> dict[str, Any]:
    config = config or ForceDriveConfig()
    specs = []
    for joint_name, target_value in joint_targets.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Force drive only supports hinge/slide joints, got type={joint_type}")
        joint_range = [float(value) for value in model.jnt_range[joint_id]]
        lower, upper = min(joint_range), max(joint_range)
        specs.append(
            {
                "joint_name": str(joint_name),
                "joint_id": int(joint_id),
                "joint_type": joint_type,
                "joint_range": joint_range,
                "qpos_addr": int(model.jnt_qposadr[joint_id]),
                "dof_addr": int(model.jnt_dofadr[joint_id]),
                "body_id": int(model.jnt_bodyid[joint_id]),
                "local_axis": np.asarray(model.jnt_axis[joint_id], dtype=float),
                "target_value": float(np.clip(float(target_value), lower, upper)),
            }
        )
    if not specs:
        raise ValueError("Force drive requires at least one joint target")

    stable_substeps = 0
    completed_substeps = 0
    reached = False
    try:
        for substep in range(max(1, int(config.max_physics_substeps))):
            data.xfrc_applied[:, :] = 0.0
            all_stable = True
            for spec in specs:
                current = float(data.qpos[spec["qpos_addr"]])
                velocity = float(data.qvel[spec["dof_addr"]])
                error = float(spec["target_value"] - current)
                is_slide = spec["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE
                kp = config.slide_kp if is_slide else config.hinge_kp
                kd = config.slide_kd if is_slide else config.hinge_kd
                max_effort = config.slide_max_effort if is_slide else config.hinge_max_effort
                effort = float(np.clip(kp * error - kd * velocity, -max_effort, max_effort))
                world_axis = data.xmat[spec["body_id"]].reshape(3, 3) @ spec["local_axis"]
                if is_slide:
                    data.xfrc_applied[spec["body_id"], :3] += world_axis * effort
                else:
                    data.xfrc_applied[spec["body_id"], 3:] += world_axis * effort
                all_stable = all_stable and (
                    abs(error) <= config.position_tolerance
                    and abs(velocity) <= config.velocity_tolerance
                )
            mujoco.mj_step(model, data)
            completed_substeps = substep + 1
            if all_stable:
                stable_substeps += 1
                if stable_substeps >= max(1, int(config.stable_substeps)):
                    reached = True
                    break
            else:
                stable_substeps = 0
    finally:
        data.xfrc_applied[:, :] = 0.0
        for spec in specs:
            data.qvel[spec["dof_addr"]] = 0.0
        mujoco.mj_forward(model, data)

    joints = []
    for spec in specs:
        final_value = float(data.qpos[spec["qpos_addr"]])
        final_error = float(spec["target_value"] - final_value)
        joints.append(
            {
                "joint_name": spec["joint_name"],
                "joint_type": (
                    "slide" if spec["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE else "hinge"
                ),
                "joint_range": list(spec["joint_range"]),
                "target_value": float(spec["target_value"]),
                "final_value": final_value,
                "final_error": final_error,
                "open_fraction": joint_open_fraction(final_value, spec["joint_range"]),
                "reached_target": abs(final_error) <= config.position_tolerance,
            }
        )
    reached = reached or all(joint["reached_target"] for joint in joints)
    return {
        "method": "xfrc_applied_group_pd",
        "success": bool(reached),
        "physics_substeps": int(completed_substeps),
        "stable_substeps": int(stable_substeps),
        "joints": joints,
        "config": asdict(config),
    }


def open_door_root_with_force(
    env,
    root_body_name: str,
    config: ForceDriveConfig | None = None,
) -> dict[str, Any]:
    config = config or ForceDriveConfig()
    groups = collect_door_root_groups(env)
    group = groups.get(str(root_body_name))
    if group is None:
        raise ValueError(
            f"Door root not found: {root_body_name}; available={sorted(groups)}"
        )
    pre_joint_infos = []
    targets = {}
    for leaf in group["leaves"]:
        joint_name = str(leaf["hinge_joint_name"])
        joint_id = mujoco.mj_name2id(
            env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        qpos_addr = int(env.current_model.jnt_qposadr[joint_id])
        before = float(env.current_data.qpos[qpos_addr])
        _closed, opened = joint_closed_open_values(leaf["joint_range"])
        targets[joint_name] = opened
        pre_joint_infos.append(
            {
                "joint_name": joint_name,
                "joint_type": "hinge",
                "joint_range": list(leaf["joint_range"]),
                "joint_value": before,
                "open_fraction": joint_open_fraction(before, leaf["joint_range"]),
            }
        )

    drive = drive_joint_group_to_targets(
        env.current_model,
        env.current_data,
        targets,
        config=config,
    )
    joint_infos = [
        {
            "joint_name": joint["joint_name"],
            "joint_type": joint["joint_type"],
            "joint_range": list(joint["joint_range"]),
            "joint_value": float(joint["final_value"]),
            "open_fraction": float(joint["open_fraction"]),
        }
        for joint in drive["joints"]
    ]
    open_success = bool(joint_infos) and all(
        float(joint["open_fraction"]) >= float(config.open_fraction_threshold)
        for joint in joint_infos
    )
    if config.assume_success and not open_success:
        fractions = [round(float(joint["open_fraction"]), 4) for joint in joint_infos]
        raise ForceInteractionError(
            f"Force interaction did not reach the configured open threshold for "
            f"{root_body_name}: fractions={fractions} threshold={config.open_fraction_threshold}"
        )
    return {
        "root_body_name": str(root_body_name),
        "action": "open",
        "method": "force_pd",
        "success": bool(open_success),
        "pre_state": (
            "closed"
            if pre_joint_infos and max(info["open_fraction"] for info in pre_joint_infos) <= 0.10
            else "ajar"
        ),
        "post_state": "open" if open_success else "ajar",
        "pre_joint_infos": pre_joint_infos,
        "joint_infos": joint_infos,
        "physics_substeps": int(drive["physics_substeps"]),
        "drive": drive,
    }


def yaw_from_pose_matrix(pose) -> float:
    return math.atan2(float(pose[1, 0]), float(pose[0, 0]))
