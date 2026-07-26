"""Runtime asset compatibility checks for a frozen V3 benchmark."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import mujoco

from scripts.InteractiveNav import container_scene_probe as probe


def _scene_args(episode: dict[str, Any], output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scene_dataset=str(episode["scene_dataset"]),
        data_split=str(episode["data_split"]),
        robot=str(episode["robot"]["robot_name"]),
        variant="base",
        seed=0,
        output_dir=output_dir,
    )


def _runtime_names_from_scene_xml(episode: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Compile only the scene XML for a cheap compatibility preflight.

    The evaluator must later create the full robot/task scene through
    :class:`JsonEvalTaskSampler`.  Loading that full scene a second time merely
    to inspect body names briefly leaves two large MuJoCo contexts alive and
    has caused EGL/driver-side OOM kills.  The benchmark-critical names all
    belong to the static house XML, so a temporary ``MjModel`` is sufficient.
    """

    scene_args = _scene_args(episode, Path("."))
    cfg = probe.build_scene_config(scene_args)
    cfg.task_sampler_config.house_inds = [int(episode["house_index"])]
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    try:
        sampler._increment_task_and_reset_house(
            force_advance_scene=False, house_index=int(episode["house_index"])
        )
        scene_path = probe.prepare_writable_scene_path(
            Path(sampler._current_house_scene_path(variant="base"))
        )
    finally:
        # No scene has been installed here, but closing keeps this helper safe
        # if sampler construction changes in a future MolmoSpaces version.
        sampler.close()

    model = mujoco.MjModel.from_xml_path(scene_path)
    try:
        bodies = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
            for index in range(model.nbody)
        }
        joints = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        }
        bodies.discard(None)
        joints.discard(None)
        return bodies, joints
    finally:
        # The Python MuJoCo binding releases native allocations when the model
        # loses its final reference.  Deleting it before JSON replay prevents
        # two complete models from overlapping in one evaluator process.
        del model


def compatible_episode_payload(episode: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop only stale, non-critical free-body pose entries.

    The source navigation benchmark was generated with an earlier asset release.
    In current ProcTHOR installations a small number of non-task objects may be
    renamed or absent.  Articulations, selected target and all interaction
    objects remain strict compatibility requirements; silently accepting a
    mismatch in any of them would invalidate evaluation.
    """

    del output_dir  # retained for API compatibility and trace layout symmetry
    bodies, joints = _runtime_names_from_scene_xml(episode)

    payload = copy.deepcopy(episode)
    modifications = payload.setdefault("scene_modifications", {})
    poses = dict(modifications.get("object_poses", {}))
    missing_poses = sorted(name for name in poses if name not in bodies)
    nav = payload["interactive_nav"]
    critical_bodies = {str(nav["target"]["selected_instance"])} | {
        str(row["object_name"]) for row in nav.get("interactions", [])
    }
    missing_critical_bodies = sorted(critical_bodies - bodies)
    if missing_critical_bodies:
        raise RuntimeError(
            "Runtime scene is missing task-critical bodies: " + ", ".join(missing_critical_bodies)
        )
    articulation_names = {
        str(row["joint_name"]) for row in modifications.get("articulation_states", [])
    }
    interaction_joints = {str(row["joint_name"]) for row in nav.get("interactions", [])}
    missing_joints = sorted((articulation_names | interaction_joints) - joints)
    if missing_joints:
        raise RuntimeError(
            "Runtime scene is missing recorded articulation joints: " + ", ".join(missing_joints)
        )
    if missing_poses:
        modifications["object_poses"] = {
            name: pose for name, pose in poses.items() if name in bodies
        }
    return payload, {
        "runtime_body_count": len(bodies),
        "runtime_joint_count": len(joints),
        "dropped_noncritical_object_pose_count": len(missing_poses),
        "dropped_noncritical_object_pose_names": missing_poses,
    }
