"""Standalone, reproducible evaluator for the frozen InteractiveNav V3 benchmark.

This implementation intentionally does not modify ``molmo_spaces/evaluation``.
The upstream evaluator remains useful for ordinary NavToObj policies, while this
runner owns V3's interaction action protocol, force execution, terminal
semantics, asset-drift handling, and reporting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

# Must be set before importing modules that can allocate a renderer.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, RBY1Config
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import interactive_nav_v3

from .benchmark_metrics import (
    SUCCESS_OPEN_FRACTION,
    joint_open_fraction,
    path_length_bin,
    reference_path_length_m,
    score_interactions,
    spl,
    summarise_results,
    target_metrics,
    oracle_terminal_goal_consistency,
)
from .benchmark_policies import (
    BenchmarkPolicy,
    NoOpPolicy,
    ScriptedOraclePolicy,
    build_factory_policy,
    build_ros_bridge_policy,
)
from .benchmark_types import EpisodeResult, InteractionAttempt, PolicyAction, PolicyObservation, PublicEpisode


PROTOCOL_VERSION = "interactive_nav_v3_benchmark_eval_v2"


@dataclass
class BenchmarkEvaluationConfig:
    benchmark: Path
    output_dir: Path
    policy: Literal["noop", "scripted_oracle", "ros_bridge", "factory"] = "noop"
    policy_factory: str | None = None
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    workers: int = 1
    max_steps: int = 500
    episode_indices: list[int] | None = None
    max_episodes: int | None = None
    resume: bool = False
    record_video: bool = False
    video_fps: float = 5.0
    camera_names: list[str] = field(default_factory=lambda: ["head_camera"])
    image_resolution: tuple[int, int] | None = (640, 480)
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 10.0
    sim_dt_ms: float = 10.0
    force_duration_seconds: float = 2.0
    force_collection_hz: float = 5.0
    force_target_fraction: float = 1.0
    force_max_internal_steps: int = 1500
    interaction_max_distance_m: float = 1.75
    require_interaction_visible: bool = True
    require_runtime_goal_consistency: bool = True
    allow_internal_object_names: bool = False
    oracle_navigation_mode: Literal["direct_pose", "task_action"] = "direct_pose"
    ros_observation_topic: str = "/molmo_spaces/head_camera/image"
    ros_action_topic: str = "/molmo_spaces/action"
    ros_action_timeout_s: float = 5.0
    ros_cmd_vel_linear_gain: float = 3.0
    ros_require_move_base_active: bool = True
    ros_map_warmup_skip_frames: int = 0

    def validate(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.policy == "factory" and not self.policy_factory:
            raise ValueError("--policy-factory is required with --policy factory")
        if self.policy != "factory" and self.policy_factory:
            raise ValueError("--policy-factory is only valid with --policy factory")
        if self.policy == "ros_bridge" and self.workers != 1:
            raise ValueError("ros_bridge requires --workers 1 because a ROS master is stateful")
        if self.policy == "ros_bridge" and "head_camera" not in self.camera_names:
            raise ValueError("ros_bridge requires head_camera for RGB/depth and pose publication")
        if not self.camera_names:
            raise ValueError("camera_names must not be empty")
        if self.image_resolution is not None and min(self.image_resolution) <= 0:
            raise ValueError("image_resolution must be positive")
        for name, value in (
            ("policy_dt_ms", self.policy_dt_ms),
            ("ctrl_dt_ms", self.ctrl_dt_ms),
            ("sim_dt_ms", self.sim_dt_ms),
            ("force_duration_seconds", self.force_duration_seconds),
            ("force_collection_hz", self.force_collection_hz),
            ("video_fps", self.video_fps),
            ("interaction_max_distance_m", self.interaction_max_distance_m),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.force_target_fraction <= 1.0:
            raise ValueError("force_target_fraction must be in (0, 1]")
        if self.force_max_internal_steps < 1:
            raise ValueError("force_max_internal_steps must be >= 1")
        if not math.isclose(self.policy_dt_ms / self.ctrl_dt_ms, round(self.policy_dt_ms / self.ctrl_dt_ms)):
            raise ValueError("policy_dt_ms must be an integer multiple of ctrl_dt_ms")
        if not math.isclose(self.ctrl_dt_ms / self.sim_dt_ms, round(self.ctrl_dt_ms / self.sim_dt_ms)):
            raise ValueError("ctrl_dt_ms must be an integer multiple of sim_dt_ms")


class BenchmarkReplayConfig(NavToObjBaseConfig):
    """Evaluator-local RBY1 config; never registered in the upstream evaluator."""

    robot_config: RBY1Config = RBY1Config()
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 10.0
    sim_dt_ms: float = 10.0
    task_horizon: int = 500
    output_dir: Path = Path("interactive_nav_benchmark_eval")
    use_passive_viewer: bool = False
    record_videos: bool = False

    @property
    def tag(self) -> str:
        return "interactive_nav_v3_benchmark_eval"


class V3BenchmarkTaskSampler(JsonEvalTaskSampler):
    """Json sampler with strict critical-name checks and safe asset drift repair."""

    runtime_compatibility: dict[str, Any]

    def __init__(self, exp_config: Any, episode_spec: EpisodeSpec, interactive_nav: dict[str, Any]) -> None:
        self._interactive_nav = interactive_nav
        self.runtime_compatibility = {}
        super().__init__(exp_config, episode_spec)

    def randomize_scene(self, env: Any, robot_view: Any) -> None:
        model = env.current_model
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
        nav = self._interactive_nav
        critical_bodies = {str(nav.get("target", {}).get("selected_instance", ""))}
        critical_bodies.discard("")
        critical_bodies.update(str(row["object_name"]) for row in nav.get("interactions", []))
        missing_bodies = sorted(critical_bodies - bodies)
        if missing_bodies:
            raise RuntimeError("Runtime scene is missing task-critical bodies: " + ", ".join(missing_bodies))
        articulation_names = {
            str(row.joint_name) for row in self.episode_spec.scene_modifications.articulation_states
        }
        interaction_joints = {str(row["joint_name"]) for row in nav.get("interactions", [])}
        missing_joints = sorted((articulation_names | interaction_joints) - joints)
        if missing_joints:
            raise RuntimeError("Runtime scene is missing recorded articulation joints: " + ", ".join(missing_joints))
        poses = dict(self.episode_spec.scene_modifications.object_poses)
        missing_poses = sorted(name for name in poses if name not in bodies)
        if missing_poses:
            self.episode_spec.scene_modifications.object_poses = {
                name: pose for name, pose in poses.items() if name in bodies
            }
        self.runtime_compatibility = {
            "runtime_body_count": len(bodies),
            "runtime_joint_count": len(joints),
            "dropped_noncritical_object_pose_count": len(missing_poses),
            "dropped_noncritical_object_pose_names": missing_poses,
        }
        super().randomize_scene(env, robot_view)


@dataclass(frozen=True)
class RuntimeJoint:
    object_name: str
    object_category: str | None
    domain: Literal["channel", "container"]
    joint_name: str
    joint_index: int
    body_id: int
    aabb_center: np.ndarray
    aabb_size: np.ndarray


class InteractionCatalog:
    """Live articulated-object catalog used only inside the evaluator."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self._joints: list[RuntimeJoint] = []
        ctx = SimpleNamespace(env=env)
        for record in probe.collect_door_records(ctx):
            self._joints.append(
                RuntimeJoint(
                    object_name=str(record["name"]),
                    object_category=str(record.get("category") or "Door"),
                    domain="channel",
                    joint_name=str(record["hinge_joint_name"]),
                    joint_index=int(record["hinge_joint_index"]),
                    body_id=int(record["body_id"]),
                    aabb_center=np.asarray(record["aabb_center"], dtype=float),
                    aabb_size=np.asarray(record["aabb_size"], dtype=float),
                )
            )
        _records, containers = probe.collect_scene_records(ctx)
        for record in containers:
            for joint in record.get("joints", []):
                self._joints.append(
                    RuntimeJoint(
                        object_name=str(record["name"]),
                        object_category=None if record.get("category") is None else str(record["category"]),
                        domain="container",
                        joint_name=str(joint["joint_name"]),
                        joint_index=int(joint["joint_index"]),
                        body_id=int(record["body_id"]),
                        aabb_center=np.asarray(record["aabb_center"], dtype=float),
                        aabb_size=np.asarray(record["aabb_size"], dtype=float),
                    )
                )

    @property
    def joints(self) -> list[RuntimeJoint]:
        return list(self._joints)

    def by_name(self, object_name: str, joint_index: int | None) -> RuntimeJoint | None:
        matches = [item for item in self._joints if item.object_name == object_name]
        if joint_index is not None:
            matches = [item for item in matches if item.joint_index == int(joint_index)]
        return matches[0] if len(matches) == 1 else None

    def by_pixel(
        self,
        *,
        camera_name: str,
        pixel_xy: tuple[int, int] | None,
        normalized_pixel_xy: tuple[float, float] | None,
        joint_index: int | None,
    ) -> tuple[RuntimeJoint | None, dict[str, Any]]:
        if camera_name not in self.env.camera_manager.registry:
            return None, {"reason": "unknown_camera"}
        try:
            segmentation = np.asarray(self.env.render_segmentation_frame(camera_name))
        except Exception as exc:
            return None, {"reason": "segmentation_render_failed", "error": f"{type(exc).__name__}: {exc}"}
        if segmentation.ndim < 3 or segmentation.shape[-1] < 2:
            return None, {"reason": "invalid_segmentation_shape", "shape": list(segmentation.shape)}
        height, width = segmentation.shape[:2]
        if normalized_pixel_xy is not None:
            u = int(round(float(normalized_pixel_xy[0]) * (width - 1)))
            v = int(round(float(normalized_pixel_xy[1]) * (height - 1)))
        elif pixel_xy is not None:
            u, v = int(pixel_xy[0]), int(pixel_xy[1])
        else:
            return None, {"reason": "missing_pixel_selector"}
        if not (0 <= u < width and 0 <= v < height):
            return None, {"reason": "pixel_out_of_bounds", "pixel_xy": [u, v], "image_size": [width, height]}
        geom_id, object_type = [int(value) for value in segmentation[v, u, :2]]
        if object_type != int(mujoco.mjtObj.mjOBJ_GEOM) or not (0 <= geom_id < self.env.current_model.ngeom):
            return None, {"reason": "pixel_not_on_geometry", "pixel_xy": [u, v], "geom_id": geom_id, "object_type": object_type}
        body_id = int(self.env.current_model.geom_bodyid[geom_id])
        ancestors: list[int] = []
        current = body_id
        while current >= 0:
            ancestors.append(current)
            if current == 0:
                break
            parent = int(self.env.current_model.body_parentid[current])
            if parent == current:
                break
            current = parent
        candidates = [item for item in self._joints if item.body_id in ancestors]
        if joint_index is not None:
            candidates = [item for item in candidates if item.joint_index == int(joint_index)]
        # The closest ancestor is the most specific visible articulated body.
        candidates.sort(key=lambda item: ancestors.index(item.body_id))
        if not candidates:
            return None, {"reason": "pixel_has_no_articulated_ancestor", "pixel_xy": [u, v], "body_id": body_id}
        if len(candidates) > 1 and joint_index is None:
            return None, {
                "reason": "ambiguous_joint_selector",
                "pixel_xy": [u, v],
                "candidate_joints": [item.joint_index for item in candidates],
            }
        return candidates[0], {"pixel_xy": [u, v], "body_id": body_id, "segmentation_image_size": [width, height]}


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe_json(payload), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_episodes(benchmark: Path) -> tuple[Path, list[dict[str, Any]]]:
    benchmark_file = benchmark / "benchmark.json" if benchmark.is_dir() else benchmark
    payload = json.loads(benchmark_file.read_text())
    if isinstance(payload, dict):
        payload = payload.get("episodes", [])
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of episodes in {benchmark_file}")
    return benchmark_file, payload


def _selected_indices(config: BenchmarkEvaluationConfig, episodes: list[dict[str, Any]]) -> list[int]:
    indices = list(range(len(episodes))) if config.episode_indices is None else list(config.episode_indices)
    if config.max_episodes is not None:
        indices = indices[: int(config.max_episodes)]
    for index in indices:
        if not 0 <= int(index) < len(episodes):
            raise IndexError(f"episode index {index} outside [0,{len(episodes)})")
    return [int(index) for index in indices]


def _build_replay_config(config: BenchmarkEvaluationConfig, output_dir: Path) -> BenchmarkReplayConfig:
    replay = BenchmarkReplayConfig()
    replay.output_dir = output_dir
    replay.task_horizon = int(config.max_steps)
    replay.policy_dt_ms = float(config.policy_dt_ms)
    replay.ctrl_dt_ms = float(config.ctrl_dt_ms)
    replay.sim_dt_ms = float(config.sim_dt_ms)
    replay.num_workers = 1
    replay.num_threads = 1
    replay.profile = False
    replay.datagen_profiler = False
    replay.filter_for_successful_trajectories = False
    replay.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)
    return replay


def _is_current_ros_policy(config: BenchmarkEvaluationConfig) -> bool:
    """Whether the policy consumes the repository's RGB/depth ROS bridge."""

    return bool(
        config.policy == "ros_bridge"
        or (
            config.policy == "factory"
            and config.policy_factory
            and "ros_navigation_factory" in config.policy_factory
        )
    )


def _public_episode(episode: dict[str, Any], camera_names: list[str], resolution: tuple[int, int]) -> PublicEpisode:
    return PublicEpisode(
        house_index=int(episode["house_index"]),
        scene_dataset=str(episode["scene_dataset"]),
        data_split=str(episode["data_split"]),
        instruction=str(episode.get("language", {}).get("task_description", "")),
        task_type=str(episode.get("task", {}).get("task_type", "nav_to_obj")),
        camera_names=list(camera_names),
        image_resolution=tuple(int(value) for value in resolution),
    )


def _build_policy(config: BenchmarkEvaluationConfig, public: PublicEpisode) -> BenchmarkPolicy:
    if config.policy == "noop":
        return NoOpPolicy()
    if config.policy == "scripted_oracle":
        return ScriptedOraclePolicy()
    if config.policy == "factory":
        assert config.policy_factory is not None
        return build_factory_policy(config.policy_factory, public_episode=public, kwargs=config.policy_kwargs)
    if config.policy == "ros_bridge":
        return build_ros_bridge_policy(
            policy_dt_ms=config.policy_dt_ms,
            observation_topic=config.ros_observation_topic,
            action_topic=config.ros_action_topic,
            action_timeout_s=config.ros_action_timeout_s,
            cmd_vel_linear_gain=config.ros_cmd_vel_linear_gain,
            require_move_base_active=config.ros_require_move_base_active,
            map_warmup_skip_frames=config.ros_map_warmup_skip_frames,
        )
    raise ValueError(f"Unsupported policy: {config.policy}")


def _discard_task_rollout_cache(task: Any) -> None:
    # Upstream tasks retain every observation for collection.  Benchmark eval
    # writes a compact trace, so keeping image tensors leaks memory per episode.
    for name in ("observation_cache", "reward_cache", "terminal_cache", "truncated_cache", "success_cache", "action_cache"):
        cache = getattr(task, name, None)
        if isinstance(cache, list):
            cache.clear()


def _base_pose_xyyaw(task: Any) -> np.ndarray:
    pose = np.asarray(task.env.current_robot.robot_view.base.pose, dtype=float)
    return np.asarray([pose[0, 3], pose[1, 3], math.atan2(pose[1, 0], pose[0, 0])], dtype=float)


def _wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def _capture_head_frame(task: Any, frames: list[np.ndarray], enabled: bool) -> None:
    if not enabled:
        return
    try:
        frame = np.asarray(task.env.render_rgb_frame("head_camera"), dtype=np.uint8)
    except Exception:
        return
    frames.append(frame.copy())


def _robot_lock_snapshot(
    env: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    robot_view = env.current_robot.robot_view
    base = robot_view.base
    base_pose = np.asarray(base.pose, dtype=float).copy()
    base_ctrl = np.asarray(base.ctrl, dtype=float).copy()
    base_hold_target = np.asarray(
        [
            float(base_pose[0, 3]),
            float(base_pose[1, 3]),
            math.atan2(float(base_pose[1, 0]), float(base_pose[0, 0])),
        ],
        dtype=float,
    )
    groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("left_arm", "right_arm", "left_gripper", "right_gripper", "torso", "head"):
        if name in robot_view.move_group_ids():
            group = robot_view.get_move_group(name)
            groups[name] = (
                np.asarray(group.joint_pos, dtype=float).copy(),
                np.zeros_like(np.asarray(group.joint_vel, dtype=float)),
                np.asarray(group.noop_ctrl, dtype=float).copy(),
            )
    return base_pose, base_ctrl, base_hold_target, groups


def _apply_robot_lock(
    env: Any,
    snapshot: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ],
) -> None:
    base_pose, base_ctrl, base_hold_target, groups = snapshot
    robot_view = env.current_robot.robot_view
    base = robot_view.base
    base.pose = base_pose.copy()
    base.joint_vel = np.zeros_like(base.joint_vel)
    try:
        # Match the collection force executor: hold the holonomic base at the
        # current world x/y/yaw target instead of restoring a stale/no-op
        # actuator command.  The latter lets the base controller fight the
        # external articulation force and can prevent a door from opening.
        base.ctrl = (
            base_hold_target.copy()
            if np.asarray(base.ctrl).shape == base_hold_target.shape
            else base_ctrl.copy()
        )
    except (AttributeError, ValueError):
        pass
    for name, (qpos, qvel, ctrl) in groups.items():
        group = robot_view.get_move_group(name)
        group.joint_pos = qpos.copy()
        group.joint_vel = qvel.copy()
        try:
            group.ctrl = ctrl.copy()
        except (AttributeError, ValueError):
            pass
    mujoco.mj_forward(env.current_model, env.current_data)
    env.camera_manager.registry.update_all_cameras(env)


def _execute_locked_force(
    env: Any,
    runtime_joint: RuntimeJoint,
    config: BenchmarkEvaluationConfig,
    *,
    frame_callback: callable | None = None,
) -> tuple[bool, dict[str, Any], float, float, float]:
    """Execute the collection-equivalent 2s force ramp while locking the robot."""

    before = joint_open_fraction(
        env,
        {"joint_name": runtime_joint.joint_name},
    )
    joint_id = mujoco.mj_name2id(env.current_model, mujoco.mjtObj.mjOBJ_JOINT, runtime_joint.joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {runtime_joint.joint_name}")
    lower, upper = [float(value) for value in env.current_model.jnt_range[joint_id]]
    closed, opened = probe.joint_closed_open_values([lower, upper])
    target = closed + float(config.force_target_fraction) * (opened - closed)
    controller = probe.ForceJointController(env, runtime_joint.joint_name, target)
    sim_dt = float(env.current_model.opt.timestep)
    sample_count = max(1, int(round(float(config.force_duration_seconds) * float(config.force_collection_hz))))
    internal_per_sample = max(1, int(round((1.0 / float(config.force_collection_hz)) / sim_dt)))
    internal_total = sample_count * internal_per_sample
    if internal_total > int(config.force_max_internal_steps):
        raise ValueError(
            f"force schedule requires {internal_total} internal steps, exceeding force_max_internal_steps="
            f"{config.force_max_internal_steps}"
        )
    snapshot = _robot_lock_snapshot(env)
    max_drift = 0.0
    effort_values: list[float] = []
    joint_values: list[float] = []
    direction_signs: list[float] = []
    try:
        for sample_index in range(sample_count):
            progress = (sample_index + 1) / sample_count
            controller.target_value = closed + float(config.force_target_fraction) * progress * (opened - closed)
            for _ in range(internal_per_sample):
                _apply_robot_lock(env, snapshot)
                row = controller.step()
                effort_values.append(float(row["effort"]))
                joint_values.append(float(row["joint_value"]))
                direction_signs.append(float(row["direction_sign"]))
                current_pose = env.current_robot.robot_view.base.pose
                max_drift = max(max_drift, float(np.linalg.norm(current_pose[:2, 3] - snapshot[0][:2, 3])))
                _apply_robot_lock(env, snapshot)
            if frame_callback is not None:
                frame_callback()
    finally:
        controller.finish()
        _apply_robot_lock(env, snapshot)
    after = joint_open_fraction(env, {"joint_name": runtime_joint.joint_name})
    metadata = controller.result()
    metadata.update(
        {
            "force_duration_seconds": float(config.force_duration_seconds),
            "force_collection_hz": float(config.force_collection_hz),
            "dataset_steps": sample_count,
            "internal_steps": internal_total,
            "simulated_seconds": internal_total * sim_dt,
            "base_lock_enabled": True,
            "upper_body_lock_enabled": True,
            "base_max_xy_drift_m": max_drift,
            "mean_effort": None if not effort_values else float(np.mean(effort_values)),
            "joint_value_min": None if not joint_values else float(np.min(joint_values)),
            "joint_value_max": None if not joint_values else float(np.max(joint_values)),
            "force_direction_readback_adaptive": bool(controller.use_generalized_force),
            "observed_direction_signs": sorted(set(direction_signs)),
        }
    )
    return bool(after >= SUCCESS_OPEN_FRACTION), metadata, before, after, internal_total * sim_dt


def _runtime_joint_from_v3(interaction: dict[str, Any]) -> RuntimeJoint:
    return RuntimeJoint(
        object_name=str(interaction["object_name"]),
        object_category=None if interaction.get("object_category") is None else str(interaction["object_category"]),
        domain="channel" if str(interaction["type"]).startswith("channel_") else "container",
        joint_name=str(interaction["joint_name"]),
        joint_index=int(interaction["joint_index"]),
        body_id=-1,
        aabb_center=np.zeros(3, dtype=float),
        aabb_size=np.zeros(3, dtype=float),
    )


def _aabb_distance_xy(robot_xy: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
    if not np.any(size):
        return 0.0
    delta = np.maximum(np.abs(np.asarray(robot_xy) - np.asarray(center)[:2]) - np.asarray(size)[:2] / 2.0, 0.0)
    return float(np.linalg.norm(delta))


def _resolve_interaction(
    *,
    env: Any,
    episode: dict[str, Any],
    catalog: InteractionCatalog,
    action: PolicyAction,
    uses_oracle_gt: bool,
    allow_internal_object_names: bool,
) -> tuple[RuntimeJoint | None, str | None, str, dict[str, Any]]:
    """Resolve a policy request without exposing V3 identifiers to ordinary policies."""

    if action.operation != "open":
        return None, None, "invalid", {"reason": f"unsupported_operation:{action.operation}"}
    nav_rows = list(episode["interactive_nav"].get("interactions", []))
    by_id = {str(row["interaction_id"]): row for row in nav_rows}
    oracle_id = action.metadata.get("oracle_interaction_id")
    details: dict[str, Any] = {}
    if oracle_id is not None:
        if not uses_oracle_gt:
            return None, None, "invalid", {"reason": "oracle_interaction_id_from_nonoracle"}
        interaction = by_id.get(str(oracle_id))
        if interaction is None:
            return None, None, "invalid", {"reason": "unknown_oracle_interaction_id"}
        return _runtime_joint_from_v3(interaction), str(oracle_id), "required_valid", {"resolver": "oracle_id"}
    candidate: RuntimeJoint | None
    if action.pixel_xy is not None or action.normalized_pixel_xy is not None:
        candidate, details = catalog.by_pixel(
            camera_name=action.camera_name,
            pixel_xy=action.pixel_xy,
            normalized_pixel_xy=action.normalized_pixel_xy,
            joint_index=action.joint_index,
        )
        if candidate is None:
            return None, None, "invalid", details
        details["resolver"] = "pixel"
    elif action.object_name is not None and allow_internal_object_names:
        candidate = catalog.by_name(str(action.object_name), action.joint_index)
        if candidate is None:
            return None, None, "invalid", {"reason": "unknown_or_ambiguous_internal_object_name"}
        details = {"resolver": "internal_object_name_debug"}
    else:
        return None, None, "invalid", {"reason": "nonoracle_interact_requires_pixel_selector"}
    matching = [
        row
        for row in nav_rows
        if str(row["object_name"]) == candidate.object_name and int(row["joint_index"]) == candidate.joint_index
    ]
    if len(matching) == 1:
        return candidate, str(matching[0]["interaction_id"]), "required_valid", details
    return candidate, None, "extra_valid", details


def _check_interaction_access(
    env: Any,
    runtime_joint: RuntimeJoint,
    *,
    camera_name: str,
    max_distance_m: float,
    require_visible: bool,
) -> tuple[bool, dict[str, Any]]:
    robot_xy = np.asarray(env.current_robot.robot_view.base.pose[:2, 3], dtype=float)
    distance = _aabb_distance_xy(robot_xy, runtime_joint.aabb_center, runtime_joint.aabb_size)
    visible = None
    if require_visible:
        try:
            visible = float(env.check_visibility(camera_name, runtime_joint.object_name))
        except Exception as exc:
            return False, {"reason": "visibility_check_failed", "error": f"{type(exc).__name__}: {exc}", "distance_m": distance}
    if distance > max_distance_m:
        return False, {"reason": "interaction_too_far", "distance_m": distance, "max_distance_m": max_distance_m, "visibility": visible}
    if require_visible and (visible is None or visible <= 0.0):
        return False, {"reason": "interaction_not_visible", "distance_m": distance, "visibility": visible}
    return True, {"distance_m": distance, "visibility": visible}


def _apply_view(task: Any, action: PolicyAction) -> dict[str, Any]:
    if action.head_qpos is None and action.torso_qpos is None:
        raise ValueError("view action must specify head_qpos and/or torso_qpos")
    result: dict[str, Any] = {}
    if action.head_qpos is not None:
        result["head_qpos"] = None if not action.head_qpos else probe.set_head_joint_position(task.env, action.head_qpos).astype(float).tolist()
    if action.torso_qpos is not None:
        result["torso_qpos"] = None if not action.torso_qpos else probe.set_torso_joint_position(task.env, action.torso_qpos).astype(float).tolist()
    mujoco.mj_forward(task.env.current_model, task.env.current_data)
    task.env.camera_manager.registry.update_all_cameras(task.env)
    return result


def _execute_oracle_waypoint(task: Any, action: PolicyAction, config: BenchmarkEvaluationConfig) -> tuple[Any, float, bool, dict[str, Any]]:
    metadata = action.metadata
    target = np.asarray(metadata["goal_point"], dtype=float)
    target_yaw = float(metadata.get("goal_yaw", 0.0))
    before = _base_pose_xyyaw(task)
    if config.oracle_navigation_mode == "task_action":
        observation, reward, terminated, truncated, infos = task.step({"base": np.asarray([target[0], target[1], target_yaw], dtype=float)})
        del reward
        terminated_flag = bool(np.asarray(terminated).reshape(-1)[0])
        truncated_flag = bool(np.asarray(truncated).reshape(-1)[0])
        details: dict[str, Any] = {"mode": "task_action", "terminated": terminated_flag, "truncated": truncated_flag, "infos": _safe_json(infos)}
    else:
        robot = task.env.current_robot
        pose = robot.robot_view.base.pose.copy()
        pose[0, 3], pose[1, 3] = float(target[0]), float(target[1])
        c, s = math.cos(target_yaw), math.sin(target_yaw)
        pose[:2, :2] = np.asarray([[c, -s], [s, c]], dtype=float)
        robot.update_control({})
        robot.robot_view.base.pose = pose
        robot.robot_view.base.joint_vel = np.zeros_like(robot.robot_view.base.joint_vel)
        mujoco.mj_forward(task.env.current_model, task.env.current_data)
        task.env.camera_manager.registry.update_all_cameras(task.env)
        observation = task.get_observations()
        details = {"mode": "direct_pose"}
    after = _base_pose_xyyaw(task)
    distance = float(np.linalg.norm(after[:2] - target[:2]))
    yaw_error = abs(_wrap_angle(float(after[2]) - target_yaw))
    reached = distance <= float(metadata.get("position_tolerance_m", 0.25)) and yaw_error <= float(metadata.get("yaw_tolerance_rad", 0.35))
    details.update({"goal_point": target.tolist(), "goal_yaw": target_yaw, "distance_error_m": distance, "yaw_error_rad": yaw_error, "reached": reached})
    return observation, float(np.linalg.norm(after[:2] - before[:2])), reached, details


def _task_base_step(task: Any, action: PolicyAction) -> tuple[Any, float, bool, dict[str, Any]]:
    if action.base_action is None:
        raise ValueError("base action is missing base_action")
    before = _base_pose_xyyaw(task)
    observation, reward, terminated, truncated, infos = task.step(action.base_action)
    after = _base_pose_xyyaw(task)
    return observation, float(np.linalg.norm(after[:2] - before[:2])), bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0]), {
        "reward": float(np.asarray(reward).reshape(-1)[0]),
        "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
        "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
        "infos": _safe_json(infos),
        "base_pose_xyyaw": after.tolist(),
    }


def _recipe(episode: dict[str, Any]) -> str | None:
    nav = episode["interactive_nav"]
    return nav.get("legacy_case_type") or nav.get("generation_validation", {}).get("legacy_case_type")


def _episode_result_base(episode_index: int, episode: dict[str, Any], policy_name: str, uses_oracle_gt: bool) -> dict[str, Any]:
    nav = episode["interactive_nav"]
    return {
        "episode_index": int(episode_index),
        "case_id": str(nav["case_id"]),
        "house_index": int(episode["house_index"]),
        "domains": list(nav["interaction_domains"]),
        "recipe": _recipe(episode),
        "interaction_types": sorted({str(row["type"]) for row in nav.get("interactions", [])}),
        "path_length_bin": path_length_bin(reference_path_length_m(episode)),
        "interaction_requirement": str(nav["interaction_requirement"]),
        "policy_name": policy_name,
        "uses_oracle_gt": uses_oracle_gt,
    }


def _run_signature(config: BenchmarkEvaluationConfig, benchmark_sha256: str, indices: list[int]) -> tuple[str, dict[str, Any]]:
    config_payload = asdict(config)
    for key in ("benchmark", "output_dir", "resume", "workers"):
        config_payload.pop(key, None)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_sha256": benchmark_sha256,
        "episode_indices": indices,
        "evaluation_config": _safe_json(config_payload),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _load_completed_trace(trace_path: Path, signature: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(trace_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "complete" or payload.get("run_signature") != signature:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def evaluate_episode(
    config: BenchmarkEvaluationConfig,
    episode_index: int,
    episode: dict[str, Any],
    *,
    run_signature: str,
) -> dict[str, Any]:
    """Evaluate one frozen V3 episode in exactly one MuJoCo context."""

    config.output_dir = Path(config.output_dir)
    nav = episode["interactive_nav"]
    case_id = str(nav["case_id"])
    episode_dir = config.output_dir / "episodes" / f"{episode_index:04d}_{case_id[:96]}"
    trace_path = episode_dir / "episode_result.json"
    if config.resume and trace_path.is_file():
        existing = _load_completed_trace(trace_path, run_signature)
        if existing is not None:
            return existing

    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    task = None
    sampler = None
    policy: BenchmarkPolicy | None = None
    scene_install_module: Any | None = None
    scene_install_original: Any | None = None
    attempts: list[dict[str, Any]] = []
    navigation_steps = 0
    view_actions = 0
    nav_path_length = 0.0
    nav_sim_seconds = 0.0
    interaction_sim_seconds = 0.0
    terminal_reason = "max_steps"
    runtime_goal_consistency: dict[str, Any] | None = None
    scoring_eligible = True
    runtime_goal_blocked = False
    started = time.monotonic()
    base_data = _episode_result_base(episode_index, episode, config.policy, config.policy == "scripted_oracle")
    try:
        interactive_nav_v3.validate_interactive_nav_v3_episode(episode, expected_domains=list(nav["interaction_domains"]))
        spec = EpisodeSpec.model_validate(episode)
        original_cameras = [camera.name for camera in spec.cameras]
        spec.cameras = [camera for camera in spec.cameras if camera.name in set(config.camera_names)]
        missing_cameras = sorted(set(config.camera_names) - {camera.name for camera in spec.cameras})
        if missing_cameras:
            raise ValueError(f"Requested unavailable camera(s): {missing_cameras}; available={original_cameras}")
        if config.image_resolution is not None:
            spec.img_resolution = tuple(int(item) for item in config.image_resolution)
        # The frozen light benchmark stores head RGB only.  The current ROS
        # navigation stack consumes an RGB-D point cloud, so enable the paired
        # live depth sensor locally without mutating the benchmark JSON.
        if _is_current_ros_policy(config):
            for camera in spec.cameras:
                if camera.name == "head_camera":
                    camera.record_depth = True
        replay = _build_replay_config(config, episode_dir)
        sampler = V3BenchmarkTaskSampler(replay, spec, nav)
        # Collection and this evaluator use a writable scene mirror.  Do not let
        # the upstream sampler attempt to mutate the shared resource cache.
        import molmo_spaces.tasks.task_sampler as task_sampler_module

        scene_install_module = task_sampler_module
        scene_install_original = task_sampler_module.install_scene_with_objects_and_grasps_from_path
        task_sampler_module.install_scene_with_objects_and_grasps_from_path = lambda *args, **kwargs: None
        # ``_get_dataset_index_map`` is a process-global cache.  Install the
        # writable mirror only while constructing this task, then restore the
        # cached source path so a later episode in the same house can replay.
        variants = sampler._get_dataset_index_map()[spec.data_split][spec.house_index]
        source_scene_path = variants["base"]
        variants["base"] = probe.prepare_writable_scene_path(Path(source_scene_path))
        try:
            task = sampler.sample_task(house_index=spec.house_index, variant="base")
        finally:
            variants["base"] = source_scene_path
        if task is None:
            raise RuntimeError("JsonEvalTaskSampler returned no task")
        observation, _info = task.reset()
        catalog = InteractionCatalog(task.env)
        runtime_goal_consistency = oracle_terminal_goal_consistency(task, episode)
        scoring_eligible = bool(runtime_goal_consistency["consistent"])
        runtime_goal_blocked = bool(
            config.require_runtime_goal_consistency
            and runtime_goal_consistency["checked"]
            and not runtime_goal_consistency["consistent"]
        )
        public = _public_episode(episode, [camera.name for camera in spec.cameras], tuple(spec.img_resolution))
        if not runtime_goal_blocked:
            policy = _build_policy(config, public)
            policy.reset(public)
            if isinstance(policy, ScriptedOraclePolicy):
                policy.reset_oracle(list(nav["oracle_plan"]["steps"]))
            base_data["policy_name"] = str(getattr(policy, "name", config.policy))
            base_data["uses_oracle_gt"] = bool(getattr(policy, "uses_oracle_gt", False))
        trace.append(
            {
                "runtime": {
                    "runtime_compatibility": sampler.runtime_compatibility,
                    "oracle_terminal_goal_consistency": runtime_goal_consistency,
                    "runtime_cameras": [camera.name for camera in spec.cameras],
                    "runtime_depth_cameras": [camera.name for camera in spec.cameras if camera.record_depth],
                    "runtime_image_resolution": list(spec.img_resolution),
                    "benchmark_image_resolution": list(episode.get("img_resolution", [])),
                    "interaction_catalog_joint_count": len(catalog.joints),
                }
            }
        )
        _capture_head_frame(task, frames, config.record_video)
        previous_action: dict[str, Any] | None = None

        if runtime_goal_blocked:
            terminal_reason = "runtime_goal_inconsistent"
        for decision_index in range(0 if runtime_goal_blocked else int(config.max_steps)):
            nav_ok, target_distance, target_visibility = target_metrics(task, episode)
            terminal_score = score_interactions(task.env, episode, attempts)
            requirement = str(nav["interaction_requirement"])
            if requirement == "unnecessary" and nav_ok and terminal_score.non_interaction_success:
                terminal_reason = "nav_success_no_interaction"
                break
            if requirement != "unnecessary" and nav_ok and terminal_score.required_interaction_success and terminal_score.sequence_success:
                terminal_reason = "interactive_nav_success"
                break
            policy_observation = PolicyObservation(
                observation=observation,
                instruction=public.instruction,
                step_index=decision_index,
                elapsed_seconds=time.monotonic() - started,
                previous_action=previous_action,
            )
            action = policy.act(policy_observation)
            event: dict[str, Any] = {"decision_step": decision_index, "action": action.to_dict()}
            if action.kind == "stop":
                terminal_reason = str(action.metadata.get("reason", "policy_stop"))
                trace.append(event)
                break
            if action.kind == "base":
                if action.metadata.get("oracle_waypoint"):
                    observation, increment, reached, details = _execute_oracle_waypoint(task, action, config)
                    notify = getattr(policy, "notify_action_result", None)
                    if callable(notify):
                        notify(action, reached=reached)
                    event["oracle_navigation"] = details
                else:
                    observation, increment, terminated, details = _task_base_step(task, action)
                    event["base"] = details
                    if terminated:
                        terminal_reason = "native_task_terminated"
                nav_path_length += increment
                nav_sim_seconds += float(config.policy_dt_ms) / 1000.0
                navigation_steps += 1
                _capture_head_frame(task, frames, config.record_video)
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                trace.append(event)
                if terminal_reason == "native_task_terminated":
                    break
                continue
            if action.kind == "view":
                view_actions += 1
                event["view"] = _apply_view(task, action)
                observation = task.get_observations()
                _capture_head_frame(task, frames, config.record_video)
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                trace.append(event)
                continue
            if action.kind == "observe":
                event["observe"] = {"refreshed": True}
                observation = task.get_observations()
                _capture_head_frame(task, frames, config.record_video)
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                trace.append(event)
                continue
            if action.kind != "interact":
                raise ValueError(f"Unsupported action kind: {action.kind}")

            runtime_joint, interaction_id, classification, resolver_meta = _resolve_interaction(
                env=task.env,
                episode=episode,
                catalog=catalog,
                action=action,
                uses_oracle_gt=bool(getattr(policy, "uses_oracle_gt", False)),
                allow_internal_object_names=config.allow_internal_object_names,
            )
            prereq_satisfied: bool | None = None
            before = after = None
            success = False
            executor_meta: dict[str, Any] = {}
            simulated_seconds = 0.0
            if runtime_joint is None:
                classification = "invalid"
            else:
                if interaction_id is not None:
                    interaction = next(row for row in nav.get("interactions", []) if str(row["interaction_id"]) == interaction_id)
                    prior_success = {
                        str(item["resolved_interaction_id"])
                        for item in attempts
                        if item.get("classification") == "required_valid" and bool(item.get("success")) and item.get("resolved_interaction_id")
                    }
                    prerequisites = {str(item["interaction_id"]) for item in interaction.get("prerequisites", [])}
                    prereq_satisfied = prerequisites.issubset(prior_success)
                    if interaction_id in prior_success:
                        classification = "extra_valid"
                access_ok, access_meta = _check_interaction_access(
                    task.env,
                    runtime_joint,
                    camera_name=action.camera_name,
                    max_distance_m=config.interaction_max_distance_m,
                    require_visible=config.require_interaction_visible,
                )
                resolver_meta["access"] = access_meta
                # Oracle references are grounded in a frozen plan.  They still
                # need a live body/joint, but their planned approach is trusted
                # when renderer occlusion differs slightly across asset builds.
                if bool(getattr(policy, "uses_oracle_gt", False)) and action.metadata.get("oracle_interaction_id"):
                    access_ok = True
                    resolver_meta["access"]["oracle_access_override"] = True
                if access_ok:
                    success, executor_meta, before, after, simulated_seconds = _execute_locked_force(
                        task.env,
                        runtime_joint,
                        config,
                        frame_callback=lambda: _capture_head_frame(task, frames, config.record_video),
                    )
                    observation = task.get_observations()
                else:
                    classification = "invalid"
            attempt = InteractionAttempt(
                requested=action.to_dict(),
                classification=classification,  # type: ignore[arg-type]
                resolved_object_name=None if runtime_joint is None else runtime_joint.object_name,
                resolved_joint_name=None if runtime_joint is None else runtime_joint.joint_name,
                resolved_joint_index=None if runtime_joint is None else runtime_joint.joint_index,
                resolved_interaction_id=interaction_id,
                success=bool(success),
                joint_fraction_before=before,
                joint_fraction_after=after,
                prerequisite_satisfied=prereq_satisfied,
                executor=None if runtime_joint is None else "force_locked",
                simulated_seconds=float(simulated_seconds),
                metadata={"resolver": resolver_meta, "executor": executor_meta},
            ).to_dict()
            attempts.append(attempt)
            interaction_sim_seconds += float(simulated_seconds)
            event["interaction"] = attempt
            _capture_head_frame(task, frames, config.record_video)
            _discard_task_rollout_cache(task)
            previous_action = action.to_dict()
            trace.append(event)
        else:
            if not runtime_goal_blocked:
                terminal_reason = "max_steps"

        nav_ok, target_distance, target_visibility = target_metrics(task, episode)
        terminal_score = score_interactions(task.env, episode, attempts)
        requirement = str(nav["interaction_requirement"])
        overall_success = bool(
            nav_ok
            and terminal_score.required_interaction_success
            and terminal_score.sequence_success
            and (terminal_score.non_interaction_success if requirement == "unnecessary" else True)
        )
        reference = reference_path_length_m(episode)
        correct_count = terminal_score.correct_action_count
        extra_count = sum(row.get("classification") == "extra_valid" for row in attempts)
        invalid_count = sum(row.get("classification") == "invalid" for row in attempts)
        video_path = None
        if config.record_video and frames:
            destination = episode_dir / "head_camera.mp4"
            probe.save_frames_to_mp4(frames, str(destination), fps=float(config.video_fps))
            video_path = str(destination)
        result = EpisodeResult(
            **base_data,
            status="complete",
            success=overall_success,
            nav_success=nav_ok,
            required_interaction_success=terminal_score.required_interaction_success,
            sequence_success=terminal_score.sequence_success,
            non_interaction_success=terminal_score.non_interaction_success,
            terminal_reason=terminal_reason,
            step_count=sum("decision_step" in item for item in trace),
            navigation_step_count=navigation_steps,
            view_action_count=view_actions,
            interaction_action_count=len(attempts),
            correct_interaction_action_count=correct_count,
            extra_interaction_action_count=int(extra_count),
            invalid_interaction_action_count=int(invalid_count),
            navigation_path_length_m=nav_path_length,
            reference_path_length_m=reference,
            spl=spl(overall_success, reference, nav_path_length),
            navigation_simulated_seconds=nav_sim_seconds,
            interaction_simulated_seconds=interaction_sim_seconds,
            total_simulated_seconds=nav_sim_seconds + interaction_sim_seconds + view_actions * float(config.policy_dt_ms) / 1000.0,
            elapsed_seconds=time.monotonic() - started,
            target_distance_m=target_distance,
            target_visibility_fraction=target_visibility,
            interaction_attempts=attempts,
            trace_path=str(trace_path),
            video_path=video_path,
            scoring_eligible=scoring_eligible,
            runtime_goal_consistency=runtime_goal_consistency,
        ).to_dict()
        trace.append({"terminal": {"interaction_score": terminal_score.to_dict(), "success": overall_success}})
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = EpisodeResult(
            **base_data,
            status="exception",
            success=False,
            nav_success=False,
            required_interaction_success=False,
            sequence_success=False,
            non_interaction_success=None,
            terminal_reason="exception",
            step_count=sum("decision_step" in item for item in trace),
            navigation_step_count=navigation_steps,
            view_action_count=view_actions,
            interaction_action_count=len(attempts),
            correct_interaction_action_count=0,
            extra_interaction_action_count=sum(row.get("classification") == "extra_valid" for row in attempts),
            invalid_interaction_action_count=sum(row.get("classification") == "invalid" for row in attempts),
            navigation_path_length_m=nav_path_length,
            reference_path_length_m=reference_path_length_m(episode),
            spl=0.0 if reference_path_length_m(episode) is not None else None,
            navigation_simulated_seconds=nav_sim_seconds,
            interaction_simulated_seconds=interaction_sim_seconds,
            total_simulated_seconds=nav_sim_seconds + interaction_sim_seconds,
            elapsed_seconds=time.monotonic() - started,
            target_distance_m=None,
            target_visibility_fraction=None,
            interaction_attempts=attempts,
            trace_path=str(trace_path),
            error=error,
            scoring_eligible=False,
            runtime_goal_consistency=runtime_goal_consistency,
        ).to_dict()
        trace.append({"exception": error, "traceback": traceback.format_exc()})
        status = "exception"
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception:
                pass
        if sampler is not None:
            try:
                sampler.close()
            except Exception:
                pass
        if scene_install_module is not None and scene_install_original is not None:
            scene_install_module.install_scene_with_objects_and_grasps_from_path = scene_install_original
    _atomic_json(trace_path, {"status": status, "run_signature": run_signature, "result": result, "trace": trace})
    return result


def _worker(payload: tuple[dict[str, Any], int, dict[str, Any], str]) -> dict[str, Any]:
    raw_config, index, episode, signature = payload
    raw_config["benchmark"] = Path(raw_config["benchmark"])
    raw_config["output_dir"] = Path(raw_config["output_dir"])
    raw_config["image_resolution"] = None if raw_config["image_resolution"] is None else tuple(raw_config["image_resolution"])
    return evaluate_episode(BenchmarkEvaluationConfig(**raw_config), index, episode, run_signature=signature)


def _write_reports(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    _atomic_json(output_dir / "results.json", rows)
    _atomic_json(output_dir / "summary.json", summary)
    if rows:
        columns = sorted({key for row in rows for key in row if key not in {"interaction_attempts"}})
        csv_path = output_dir / "results.csv"
        temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(_safe_json(row.get(key)), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in columns})
        temporary.replace(csv_path)
    lines = [
        "# InteractiveNav V3 benchmark evaluation",
        "",
        f"- Episodes replayed: {len(rows)}",
        f"- Scoring-eligible: {summary.get('scoring_eligible_episode_count', len(rows))}",
        f"- Runtime-ineligible: {summary.get('runtime_ineligible_episode_count', 0)}",
        "",
        "## Formal score groups",
        "",
    ]
    if not summary.get("groups"):
        lines.append("No scoring-eligible episodes were available for formal metrics.")
    for name, values in summary.get("groups", {}).items():
        lines.append(
            f"- `{name}`: n={values['episode_count']}, SR={values['success_rate']}, "
            f"NavSR={values['nav_success_rate']}, ISR={values['required_interaction_success_rate']}, "
            f"IP={values['interaction_precision']}, SPL={values['mean_spl']}"
        )
    report = output_dir / "report.md"
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(report)


def run_evaluation(config: BenchmarkEvaluationConfig) -> dict[str, Any]:
    """Run independent V3 evaluation with spawn-safe MuJoCo multiprocessing."""

    config.validate()
    config.benchmark = Path(config.benchmark).resolve()
    config.output_dir = Path(config.output_dir).resolve()
    benchmark_file, episodes = _load_episodes(config.benchmark)
    indices = _selected_indices(config, episodes)
    benchmark_hash = _sha256(benchmark_file)
    signature, manifest_payload = _run_signature(config, benchmark_hash, indices)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_signature = existing.get("run_signature")
        if config.resume:
            if existing_signature != signature:
                raise ValueError("--resume refused: existing output has a different benchmark/evaluation signature")
        else:
            raise FileExistsError(f"Output directory already contains run_manifest.json: {config.output_dir}; use --resume or a new directory")
    else:
        _atomic_json(manifest_path, {"run_signature": signature, **manifest_payload, "created_unix_s": time.time()})
    raw_config = asdict(config)
    raw_config["benchmark"] = str(config.benchmark)
    raw_config["output_dir"] = str(config.output_dir)
    payloads = [(raw_config, index, episodes[index], signature) for index in indices]
    rows: list[dict[str, Any]] = []
    if config.workers == 1:
        for payload in payloads:
            rows.append(_worker(payload))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [pool.submit(_worker, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: int(row["episode_index"]))
    summary = summarise_results(rows)
    summary.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "run_signature": signature,
            "benchmark": str(benchmark_file),
            "benchmark_sha256": benchmark_hash,
            "policy": config.policy,
            "workers": config.workers,
            "max_steps": config.max_steps,
            "episode_indices": indices,
            "result_count": len(rows),
            "complete_count": sum(row.get("status") == "complete" for row in rows),
            "exception_count": sum(row.get("status") == "exception" for row in rows),
        }
    )
    _write_reports(config.output_dir, rows, summary)
    return {"results": rows, "summary": summary}


def _parse_policy_kwargs(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--policy-kwargs-json must decode to an object")
    return parsed


def parse_args() -> BenchmarkEvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=["noop", "scripted_oracle", "ros_bridge", "factory"], default="noop")
    parser.add_argument("--policy-factory")
    parser.add_argument("--policy-kwargs-json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--camera-names", nargs="+", default=["head_camera"])
    parser.add_argument("--image-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=[640, 480])
    parser.add_argument("--policy-dt-ms", type=float, default=200.0)
    parser.add_argument("--ctrl-dt-ms", type=float, default=10.0)
    parser.add_argument("--sim-dt-ms", type=float, default=10.0)
    parser.add_argument("--force-duration-seconds", type=float, default=2.0)
    parser.add_argument("--force-collection-hz", type=float, default=5.0)
    parser.add_argument("--force-target-fraction", type=float, default=1.0)
    parser.add_argument("--force-max-internal-steps", type=int, default=1500)
    parser.add_argument("--interaction-max-distance-m", type=float, default=1.75)
    parser.add_argument(
        "--require-runtime-goal-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refuse formal scoring when a frozen terminal oracle waypoint does not reach "
            "the live selected target. Use --no-require-runtime-goal-consistency only for "
            "integration debugging; such rows remain scoring-ineligible."
        ),
    )
    parser.add_argument("--allow-internal-object-names", action="store_true", help="debug-only non-oracle name selector")
    parser.add_argument("--oracle-navigation-mode", choices=["direct_pose", "task_action"], default="direct_pose")
    parser.add_argument("--ros-observation-topic", default="/molmo_spaces/head_camera/image")
    parser.add_argument("--ros-action-topic", default="/molmo_spaces/action")
    parser.add_argument(
        "--ros-action-timeout-s",
        type=float,
        default=5.0,
        help="Maximum wait for a fresh ROS action per decision; 0 waits indefinitely.",
    )
    parser.add_argument("--ros-cmd-vel-linear-gain", type=float, default=3.0)
    parser.add_argument("--ros-require-move-base-active", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ros-map-warmup-skip-frames", type=int, default=0)
    args = parser.parse_args()
    return BenchmarkEvaluationConfig(
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        policy=args.policy,
        policy_factory=args.policy_factory,
        policy_kwargs=_parse_policy_kwargs(args.policy_kwargs_json),
        workers=args.workers,
        max_steps=args.max_steps,
        episode_indices=args.episode_indices,
        max_episodes=args.max_episodes,
        resume=args.resume,
        record_video=args.record_video,
        video_fps=args.video_fps,
        camera_names=list(args.camera_names),
        image_resolution=tuple(args.image_resolution),
        policy_dt_ms=args.policy_dt_ms,
        ctrl_dt_ms=args.ctrl_dt_ms,
        sim_dt_ms=args.sim_dt_ms,
        force_duration_seconds=args.force_duration_seconds,
        force_collection_hz=args.force_collection_hz,
        force_target_fraction=args.force_target_fraction,
        force_max_internal_steps=args.force_max_internal_steps,
        interaction_max_distance_m=args.interaction_max_distance_m,
        require_runtime_goal_consistency=args.require_runtime_goal_consistency,
        allow_internal_object_names=args.allow_internal_object_names,
        oracle_navigation_mode=args.oracle_navigation_mode,
        ros_observation_topic=args.ros_observation_topic,
        ros_action_topic=args.ros_action_topic,
        ros_action_timeout_s=args.ros_action_timeout_s,
        ros_cmd_vel_linear_gain=args.ros_cmd_vel_linear_gain,
        ros_require_move_base_active=args.ros_require_move_base_active,
        ros_map_warmup_skip_frames=args.ros_map_warmup_skip_frames,
    )


def main() -> int:
    config = parse_args()
    output = run_evaluation(config)
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
