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
from .public_goal import build_public_target_context as build_language_target_context
from .restricted_gt_perception import RestrictedGTPerceptionPublisher
from .ros_object_goal_adapter import (
    RosObjectGoalEvaluatorAdapter,
    build_public_target_context as build_ros_target_context,
)
from .trusted_interaction_skill import (
    JointOpenResult,
    ObjectInteractionRequest,
    OpaqueObjectRegistry,
    OpenPostconditionSpec,
    TrustedInteractionSkill,
)


PROTOCOL_VERSION = "interactive_nav_v3_benchmark_eval_v6"

# Established ROS exploration posture.  The asymmetric shoulder roll tucks the
# arms beside the torso so they neither move after the first navigation action
# nor enter the head-camera depth map.
ROS_NAVIGATION_ARM_QPOS: dict[str, tuple[float, ...]] = {
    "left_arm": (0.28, 0.0, -0.45, -0.64, 0.39, -0.26, -0.04),
    "right_arm": (0.28, 0.0, 0.45, -0.64, 0.39, -0.26, -0.04),
}


@dataclass
class BenchmarkEvaluationConfig:
    benchmark: Path
    output_dir: Path
    policy: Literal["noop", "scripted_oracle", "ros_bridge", "ros_object_goal_rule", "factory"] = "noop"
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
    ros_target_topic: str = "/semantic_decision/target"
    ros_restricted_gt_topic: str = "/semantic_mapping/gt_observations"
    ros_interaction_command_topic: str = "/semantic_decision/interaction_command"
    ros_interaction_result_topic: str = "/semantic_mapping/interaction_result"
    restricted_gt_min_visible_pixels: int = 16
    quality_gate_only: bool = False
    runtime_joint_position_tolerance: float = 0.02
    runtime_joint_fraction_tolerance: float = 0.05
    runtime_base_position_tolerance_m: float = 0.05
    runtime_base_yaw_tolerance_rad: float = 0.05
    progress_every: int = 10

    def validate(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.policy == "factory" and not self.policy_factory:
            raise ValueError("--policy-factory is required with --policy factory")
        if self.policy != "factory" and self.policy_factory:
            raise ValueError("--policy-factory is only valid with --policy factory")
        if self.policy in {"ros_bridge", "ros_object_goal_rule"} and self.workers != 1:
            raise ValueError(f"{self.policy} requires --workers 1 because a ROS master is stateful")
        if self.policy in {"ros_bridge", "ros_object_goal_rule"} and "head_camera" not in self.camera_names:
            raise ValueError(f"{self.policy} requires head_camera for RGB/depth and pose publication")
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
        if self.restricted_gt_min_visible_pixels < 1:
            raise ValueError("restricted_gt_min_visible_pixels must be >= 1")
        if self.progress_every < 1:
            raise ValueError("progress_every must be >= 1")
        for name, value in (
            ("runtime_joint_position_tolerance", self.runtime_joint_position_tolerance),
            ("runtime_joint_fraction_tolerance", self.runtime_joint_fraction_tolerance),
            ("runtime_base_position_tolerance_m", self.runtime_base_position_tolerance_m),
            ("runtime_base_yaw_tolerance_rad", self.runtime_base_yaw_tolerance_rad),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
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

    def set_joint_values(self, env: Any) -> None:
        # JsonEvalTaskSampler treats every articulated ``pickup_obj_name`` as a
        # manipulation target and requires a grasp file.  NavToObj reuses that
        # field only to identify the navigation target; its authoritative joint
        # state is restored later from ``scene_modifications``.
        if self.episode_spec.task.get("task_type") == "nav_to_obj":
            return
        super().set_joint_values(env)

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


@dataclass
class RestrictedRosObjectGoalRuntime:
    """Evaluator-private state for one restricted-GT ROS episode."""

    perception: RestrictedGTPerceptionPublisher
    adapter: RosObjectGoalEvaluatorAdapter
    skill: TrustedInteractionSkill
    opaque_to_source_name: dict[str, str]
    opaque_to_joints: dict[str, tuple[RuntimeJoint, ...]]


def _build_restricted_ros_object_goal_runtime(
    *,
    task: Any,
    catalog: InteractionCatalog,
    episode: dict[str, Any],
    public: PublicEpisode,
    config: BenchmarkEvaluationConfig,
    frame_callback: callable | None = None,
) -> RestrictedRosObjectGoalRuntime:
    """Create evaluator-owned restricted perception and sealed force skills.

    The external ROS graph sees only the public target language and canonical
    restricted-GT observations.  The source-name/joint mapping lives entirely
    in this function's returned private runtime object.
    """

    perception = RestrictedGTPerceptionPublisher(
        camera_name="head_camera",
        min_visible_pixels=int(config.restricted_gt_min_visible_pixels),
        step_interval=1,
        frame_id="world",
    )
    private_registry = OpaqueObjectRegistry()
    opaque_to_source_name: dict[str, str] = {}
    opaque_to_joints: dict[str, tuple[RuntimeJoint, ...]] = {}
    joints_by_object: dict[str, list[RuntimeJoint]] = {}
    for joint in catalog.joints:
        joints_by_object.setdefault(joint.object_name, []).append(joint)
    for source_name in sorted(joints_by_object):
        opaque_id = perception.registry.public_id_for(source_name)
        joints = tuple(sorted(joints_by_object[source_name], key=lambda item: item.joint_index))
        private_registry.register(
            opaque_id,
            joints=joints,
            object_ref=source_name,
            open_postcondition=OpenPostconditionSpec(
                success_fraction=float(SUCCESS_OPEN_FRACTION),
                minimum_open_joints=1,
            ),
        )
        opaque_to_source_name[opaque_id] = source_name
        opaque_to_joints[opaque_id] = joints

    def execute_open_joint(joint: RuntimeJoint) -> JointOpenResult:
        succeeded, metadata, before, after, simulated_seconds = _execute_locked_force(
            task.env,
            joint,
            config,
            frame_callback=frame_callback,
        )
        return JointOpenResult(
            executor_succeeded=bool(succeeded),
            open_fraction_before=before,
            open_fraction_after=after,
            simulated_seconds=float(simulated_seconds),
            metadata=metadata,
        )

    skill = TrustedInteractionSkill(private_registry, execute_open_joint)
    adapter = RosObjectGoalEvaluatorAdapter(
        target_topic=config.ros_target_topic,
        gt_observations_topic=config.ros_restricted_gt_topic,
        interaction_command_topic=config.ros_interaction_command_topic,
        interaction_result_topic=config.ros_interaction_result_topic,
    )
    language_target = build_language_target_context(
        episode.get("language") if isinstance(episode.get("language"), dict) else {},
        public.instruction,
    )
    target_name = str(language_target.get("target_name") or public.instruction or "object")
    adapter.reset(
        episode_id=perception.episode_id,
        target_context=build_ros_target_context(
            episode_id=perception.episode_id,
            target_name=target_name,
            instruction=public.instruction,
            object_labels=list(language_target.get("object_labels") or [target_name]),
            enabled=bool(language_target.get("enabled", True)),
            min_visible_pixels=int(config.restricted_gt_min_visible_pixels),
            min_visible_fraction=0.0,
            min_consecutive_observations=1,
            completion_requires_visibility=True,
            require_current_visibility=False,
        ),
        # This mapping is evaluator-private.  The adapter uses its keys only to
        # validate opaque IDs from ROS interaction commands.
        private_instances={opaque_id: opaque_id for opaque_id in opaque_to_source_name},
    )
    initial_frame = perception.build(task, force=True)
    if initial_frame is not None:
        adapter.publish_restricted_gt_frame(initial_frame, capture_step=0)
    return RestrictedRosObjectGoalRuntime(
        perception=perception,
        adapter=adapter,
        skill=skill,
        opaque_to_source_name=opaque_to_source_name,
        opaque_to_joints=opaque_to_joints,
    )


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
    if _is_current_ros_policy(config):
        for group_name, qpos in ROS_NAVIGATION_ARM_QPOS.items():
            replay.robot_config.init_qpos[group_name] = np.asarray(qpos, dtype=float)
            replay.robot_config.init_qpos_noise_range[group_name] = np.zeros(len(qpos), dtype=float)
    return replay


def _apply_ros_navigation_arm_posture(spec: EpisodeSpec) -> None:
    """Use the deterministic ROS navigation posture without mutating benchmark JSON."""

    for group_name, qpos in ROS_NAVIGATION_ARM_QPOS.items():
        if group_name in spec.robot.init_qpos:
            spec.robot.init_qpos[group_name] = list(qpos)


def _is_current_ros_policy(config: BenchmarkEvaluationConfig) -> bool:
    """Whether the policy consumes the repository's RGB/depth ROS bridge."""

    return bool(
        config.policy in {"ros_bridge", "ros_object_goal_rule"}
        or (
            config.policy == "factory"
            and config.policy_factory
            and "ros_navigation_factory" in config.policy_factory
        )
    )


def _is_ros_object_goal_rule(config: BenchmarkEvaluationConfig) -> bool:
    """Whether this rollout owns restricted-GT and object-skill ROS topics."""

    return config.policy == "ros_object_goal_rule"


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
    if config.policy in {"ros_bridge", "ros_object_goal_rule"}:
        return build_ros_bridge_policy(
            policy_dt_ms=config.policy_dt_ms,
            observation_topic=config.ros_observation_topic,
            action_topic=config.ros_action_topic,
            action_timeout_s=config.ros_action_timeout_s,
            cmd_vel_linear_gain=config.ros_cmd_vel_linear_gain,
            require_move_base_active=config.ros_require_move_base_active,
            map_warmup_skip_frames=config.ros_map_warmup_skip_frames,
            name=config.policy,
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


def _expected_runtime_base_xyyaw(episode: dict[str, Any]) -> np.ndarray:
    """Return the task-authoritative start pose in planar coordinates."""

    pose = np.asarray(episode.get("task", {}).get("robot_base_pose", []), dtype=float)
    if pose.shape == (7,):
        x, y, _z, qw, qx, qy, qz = [float(value) for value in pose]
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return np.asarray([x, y, yaw], dtype=float)
    pose = np.asarray(
        episode.get("robot", {}).get("init_qpos", {}).get("base", []),
        dtype=float,
    )
    return pose


def _runtime_consistency_checks(
    task: Any,
    catalog: InteractionCatalog,
    episode: dict[str, Any],
    sampler: V3BenchmarkTaskSampler,
    config: BenchmarkEvaluationConfig,
) -> dict[str, Any]:
    """Validate that a frozen episode can be scored in the live scene.

    These checks intentionally stop before any policy action.  They cover the
    runtime identities and initial state that formal metrics depend on, while
    leaving navigation and manipulation performance to the evaluated policy.
    """

    nav = episode["interactive_nav"]
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    goal = oracle_terminal_goal_consistency(task, episode)
    checks["oracle_terminal_goal"] = goal
    if goal.get("checked") and not goal.get("consistent"):
        reasons.append("terminal_goal_target_mismatch")

    expected_base = _expected_runtime_base_xyyaw(episode)
    actual_base = _base_pose_xyyaw(task)
    if expected_base.shape == (3,):
        base_position_error = float(np.linalg.norm(actual_base[:2] - expected_base[:2]))
        base_yaw_error = abs(_wrap_angle(float(actual_base[2] - expected_base[2])))
        base_passed = bool(
            base_position_error <= float(config.runtime_base_position_tolerance_m)
            and base_yaw_error <= float(config.runtime_base_yaw_tolerance_rad)
        )
    else:
        base_position_error = None
        base_yaw_error = None
        base_passed = False
    checks["robot_start_pose"] = {
        "passed": base_passed,
        "expected_xyyaw": expected_base.tolist(),
        "actual_xyyaw": actual_base.tolist(),
        "position_error_m": base_position_error,
        "yaw_error_rad": base_yaw_error,
        "position_tolerance_m": float(config.runtime_base_position_tolerance_m),
        "yaw_tolerance_rad": float(config.runtime_base_yaw_tolerance_rad),
    }
    if not base_passed:
        reasons.append("robot_start_pose_mismatch")

    articulation_rows: list[dict[str, Any]] = []
    for expected in episode.get("scene_modifications", {}).get("articulation_states", []):
        joint_name = str(expected["joint_name"])
        joint_id = mujoco.mj_name2id(task.env.current_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            articulation_rows.append({
                "joint_name": joint_name,
                "passed": False,
                "reason": "joint_not_found",
            })
            continue
        actual_position = float(probe.joint_value_by_name(task.env, joint_name))
        expected_position = float(expected["position"])
        actual_fraction = joint_open_fraction(task.env, {"joint_name": joint_name})
        expected_fraction = float(expected["open_fraction"])
        position_error = abs(actual_position - expected_position)
        fraction_error = abs(actual_fraction - expected_fraction)
        articulation_rows.append({
            "object_name": str(expected.get("object_name", "")),
            "joint_name": joint_name,
            "expected_position": expected_position,
            "actual_position": actual_position,
            "position_error": position_error,
            "expected_open_fraction": expected_fraction,
            "actual_open_fraction": actual_fraction,
            "open_fraction_error": fraction_error,
            "passed": bool(
                position_error <= float(config.runtime_joint_position_tolerance)
                and fraction_error <= float(config.runtime_joint_fraction_tolerance)
            ),
        })
    articulation_passed = bool(articulation_rows) and all(row["passed"] for row in articulation_rows)
    checks["articulation_state_readback"] = {
        "passed": articulation_passed,
        "joint_count": len(articulation_rows),
        "position_tolerance": float(config.runtime_joint_position_tolerance),
        "open_fraction_tolerance": float(config.runtime_joint_fraction_tolerance),
        "failed_joints": [row for row in articulation_rows if not row["passed"]],
    }
    if not articulation_passed:
        reasons.append("articulation_state_mismatch")

    resolution_rows: list[dict[str, Any]] = []
    for interaction in nav.get("interactions", []):
        object_name = str(interaction["object_name"])
        joint_index = int(interaction["joint_index"])
        expected_joint_name = str(interaction["joint_name"])
        resolved = catalog.by_name(object_name, joint_index)
        passed = bool(resolved is not None and resolved.joint_name == expected_joint_name)
        resolution_rows.append({
            "interaction_id": str(interaction["interaction_id"]),
            "object_name": object_name,
            "joint_index": joint_index,
            "expected_joint_name": expected_joint_name,
            "resolved_joint_name": None if resolved is None else resolved.joint_name,
            "passed": passed,
        })
    resolution_passed = all(row["passed"] for row in resolution_rows)
    checks["interaction_resolution"] = {
        "passed": resolution_passed,
        "interaction_count": len(resolution_rows),
        "failed_interactions": [row for row in resolution_rows if not row["passed"]],
    }
    if not resolution_passed:
        reasons.append("interaction_resolution_mismatch")

    expected_visible = nav.get("initial_state", {}).get("target_visible")
    if isinstance(expected_visible, bool):
        criteria = nav.get("success_criteria", {})
        camera_name = str(criteria.get("visibility", {}).get("camera_name", "head_camera"))
        target_name = str(nav["target"]["selected_instance"])
        actual_visibility = float(task.env.check_visibility(camera_name, target_name))
        actual_visible = actual_visibility > 0.0
        visibility_passed = actual_visible is expected_visible
        checks["initial_target_visibility"] = {
            "applicable": True,
            "passed": visibility_passed,
            "camera_name": camera_name,
            "target_name": target_name,
            "expected_visible": expected_visible,
            "actual_visible": actual_visible,
            "actual_visibility_fraction": actual_visibility,
        }
        if not visibility_passed:
            reasons.append("initial_target_visibility_mismatch")
    else:
        checks["initial_target_visibility"] = {"applicable": False, "passed": True}

    checks["scene_runtime_compatibility"] = {
        "passed": True,
        **sampler.runtime_compatibility,
    }
    return {
        "schema_version": "interactive_nav_v3_runtime_consistency_v1",
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "checks": checks,
    }


def _redact_runtime_consistency_for_restricted_policy(
    runtime_consistency: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Project evaluator-private replay checks into a safe public summary.

    The checks are necessary for formal scoring, but their raw target names,
    joint names, joint values and poses must not become an accidental side
    channel in a restricted-GT policy's result directory.
    """

    if runtime_consistency is None:
        return None, None
    checks = runtime_consistency.get("checks") if isinstance(runtime_consistency, dict) else {}
    checks = checks if isinstance(checks, dict) else {}
    goal = checks.get("oracle_terminal_goal") if isinstance(checks.get("oracle_terminal_goal"), dict) else {}
    public_goal = {
        "checked": bool(goal.get("checked", False)),
        "consistent": bool(goal.get("consistent", False)),
    }
    public_checks: dict[str, Any] = {"oracle_terminal_goal": public_goal}
    for name, source in checks.items():
        if name == "oracle_terminal_goal" or not isinstance(source, dict):
            continue
        summary: dict[str, Any] = {"passed": bool(source.get("passed", False))}
        for key in ("applicable", "joint_count", "interaction_count"):
            if key in source:
                summary[key] = source[key]
        public_checks[name] = summary
    return public_goal, {
        "schema_version": "interactive_nav_v3_runtime_consistency_public_v1",
        "eligible": bool(runtime_consistency.get("eligible", False)),
        "exclusion_reasons": list(runtime_consistency.get("exclusion_reasons", [])),
        "checks": public_checks,
    }


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


def _resolved_interaction_ids(attempt: dict[str, Any]) -> list[str]:
    """Return all private V3 IDs credited by one evaluator-side attempt."""

    values = attempt.get("resolved_interaction_ids")
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values if value is not None]
    value = attempt.get("resolved_interaction_id")
    return [] if value is None else [str(value)]


def _successful_object_skill_interaction_ids(
    *,
    episode: dict[str, Any],
    source_name: str,
    joints: tuple[RuntimeJoint, ...],
    joint_results: tuple[JointOpenResult, ...],
) -> list[str]:
    """Privately map sealed object-skill postconditions to frozen V3 rows.

    The navigation method receives just one opaque ``open(instance_id)`` result.
    V3's evaluator still needs to know which recorded articulation postconditions
    are actually satisfied.  This mapping intentionally remains in the runner
    and considers only final fractions, never controller substeps.
    """

    rows = list(episode["interactive_nav"].get("interactions", []))
    resolved: list[str] = []
    for joint, result in zip(joints, joint_results, strict=False):
        if result.open_fraction_after is None or float(result.open_fraction_after) < SUCCESS_OPEN_FRACTION:
            continue
        for row in rows:
            if (
                str(row.get("object_name")) == source_name
                and int(row.get("joint_index", -1)) == int(joint.joint_index)
            ):
                resolved.append(str(row["interaction_id"]))
    return list(dict.fromkeys(resolved))


def _publish_restricted_ros_frame(
    runtime: RestrictedRosObjectGoalRuntime,
    task: Any,
    *,
    decision_index: int,
) -> bool:
    """Render and publish one canonical public perception frame for ROS."""

    payload = runtime.perception.build(task, step_index=int(decision_index), force=True)
    if payload is None:
        return False
    runtime.adapter.publish_restricted_gt_frame(payload, capture_step=int(decision_index))
    return True


def _consume_pending_ros_object_goal_interaction(
    *,
    task: Any,
    runtime: RestrictedRosObjectGoalRuntime,
    episode: dict[str, Any],
    private_attempts: list[dict[str, Any]],
    config: BenchmarkEvaluationConfig,
    decision_index: int,
    frames: list[np.ndarray],
) -> dict[str, Any] | None:
    """Execute at most one queued opaque object request inside the evaluator.

    The ROS navigation graph can only submit an opaque ID and ``open``.  This
    function resolves that ID, checks live approach access, and invokes the
    evaluator-owned force skill.  Its return value intentionally has separate
    private scoring and public trace projections.
    """

    request = runtime.adapter.pop_next_interaction_request()
    if request is None:
        return None
    source_name = runtime.opaque_to_source_name.get(request.instance_id)
    joints = runtime.opaque_to_joints.get(request.instance_id, ())
    public_request = ObjectInteractionRequest(
        request_id=request.command_id,
        instance_id=request.instance_id,
        operation="open",
    )
    access_ok = False
    access_meta: dict[str, Any] = {"reason": "unknown_opaque_instance"}
    if source_name is not None and joints:
        # An object-level skill intentionally checks approach access once.  The
        # lower force policy may then operate the object's private joint set.
        access_ok, access_meta = _check_interaction_access(
            task.env,
            joints[0],
            camera_name="head_camera",
            max_distance_m=config.interaction_max_distance_m,
            require_visible=config.require_interaction_visible,
        )

    joint_results: tuple[JointOpenResult, ...] = ()
    public_result: dict[str, Any]
    successful_ids: list[str] = []
    if access_ok:
        event = runtime.skill.execute_private(public_request)
        joint_results = event.joint_results
        successful_ids = _successful_object_skill_interaction_ids(
            episode=episode,
            source_name=str(source_name),
            joints=joints,
            joint_results=joint_results,
        )
        completion = runtime.adapter.complete_interaction(
            request.command_id,
            success=event.public_result.completed,
        )
        public_result = event.public_result.to_public_dict()
        simulated_seconds = float(sum(result.simulated_seconds for result in joint_results))
        skill_completed = bool(event.public_result.completed)
        postcondition = None if event.postcondition is None else event.postcondition.value
    else:
        completion = runtime.adapter.complete_interaction(request.command_id, success=False)
        public_result = {
            "request_id": request.command_id,
            "instance_id": request.instance_id,
            "operation": "open",
            "status": "failed",
        }
        simulated_seconds = 0.0
        skill_completed = False
        postcondition = None

    matching_ids = [
        str(row["interaction_id"])
        for row in episode["interactive_nav"].get("interactions", [])
        if source_name is not None and str(row.get("object_name")) == source_name
    ]
    classification = "required_valid" if matching_ids and access_ok else "extra_valid" if access_ok else "invalid"
    prior_success = {
        interaction_id
        for attempt in private_attempts
        if attempt.get("classification") == "required_valid" and bool(attempt.get("success"))
        for interaction_id in _resolved_interaction_ids(attempt)
    }
    prerequisite_ids = {
        str(prerequisite["interaction_id"])
        for row in episode["interactive_nav"].get("interactions", [])
        if str(row.get("interaction_id")) in successful_ids
        for prerequisite in row.get("prerequisites", [])
    }
    prerequisite_satisfied: bool | None = None
    if matching_ids:
        prerequisite_satisfied = prerequisite_ids.issubset(prior_success)
    private_attempt = InteractionAttempt(
        requested=public_request.to_public_dict(),
        classification=classification,  # type: ignore[arg-type]
        resolved_object_name=source_name,
        resolved_joint_name=None,
        resolved_joint_index=None,
        resolved_interaction_id=successful_ids[0] if successful_ids else None,
        resolved_interaction_ids=successful_ids,
        success=skill_completed,
        joint_fraction_before=None,
        joint_fraction_after=None,
        prerequisite_satisfied=prerequisite_satisfied,
        executor="trusted_object_force" if source_name is not None else None,
        simulated_seconds=simulated_seconds,
        metadata={
            # This private trace is retained only for formal V3 scoring and is
            # never handed to the ROS method.  It records no controller steps.
            "access": access_meta,
            "postcondition": postcondition,
            "joint_result_count": len(joint_results),
        },
    ).to_dict()
    public_attempt = {
        **public_result,
        "decision_step": int(decision_index),
        "simulated_seconds": simulated_seconds,
        "result_status": str(completion.get("status") or "FAILED"),
    }
    observation = task.get_observations()
    _capture_head_frame(task, frames, config.record_video)
    _publish_restricted_ros_frame(runtime, task, decision_index=decision_index)
    _discard_task_rollout_cache(task)
    return {
        "private_attempt": private_attempt,
        "public_attempt": public_attempt,
        "observation": observation,
        "simulated_seconds": simulated_seconds,
    }


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


def _private_episode_visualization_context(
    *,
    task: Any,
    catalog: InteractionCatalog,
    episode: dict[str, Any],
) -> dict[str, Any]:
    """Capture post-evaluation-only GT geometry without exposing it to a policy.

    This is deliberately a sidecar, not an ``EpisodeResult`` field: restricted
    ROS policies must never receive target or internal articulated-object names
    through their public V3 trace while an episode is being evaluated.
    """

    nav = episode["interactive_nav"]
    target_name = str(nav["target"]["selected_instance"])
    objects = task.env.object_managers[task.env.current_batch_index]
    target = objects.get_object_by_name(target_name)
    interactions: list[dict[str, Any]] = []
    for row in nav.get("interactions", []):
        runtime_joint = catalog.by_name(str(row["object_name"]), int(row["joint_index"]))
        entry: dict[str, Any] = {
            "interaction_id": str(row["interaction_id"]),
            "type": str(row.get("type", "interaction")),
        }
        if runtime_joint is not None:
            entry.update(
                {
                    "xy": runtime_joint.aabb_center[:2].astype(float).tolist(),
                    "aabb_center": runtime_joint.aabb_center.astype(float).tolist(),
                    "aabb_size": runtime_joint.aabb_size.astype(float).tolist(),
                }
            )
        interactions.append(entry)
    return {
        "schema_version": "interactive_nav_v3_evaluator_private_visualization_v1",
        "evaluator_private": True,
        "case_id": str(nav["case_id"]),
        "target": {
            "xy": np.asarray(target.position[:2], dtype=float).tolist(),
        },
        "gt_interactions": interactions,
        "actual_interactions": [],
    }


def _private_actual_interaction_markers(
    attempts: list[dict[str, Any]],
    catalog: InteractionCatalog,
) -> list[dict[str, Any]]:
    """Project resolved evaluator attempts to object locations for the sidecar."""

    markers: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str | None]] = set()
    for attempt in attempts:
        object_name = attempt.get("resolved_object_name")
        if object_name is None:
            continue
        joint_index = attempt.get("resolved_joint_index")
        joints = [joint for joint in catalog.joints if joint.object_name == str(object_name)]
        if joint_index is not None:
            joints = [joint for joint in joints if joint.joint_index == int(joint_index)]
        for joint in joints:
            request = attempt.get("requested")
            request = request if isinstance(request, dict) else {}
            request_id = request.get("request_id")
            key = (joint.object_name, joint.joint_index, None if request_id is None else str(request_id))
            if key in seen:
                continue
            seen.add(key)
            markers.append(
                {
                    "request_id": request_id,
                    "xy": joint.aabb_center[:2].astype(float).tolist(),
                    "aabb_center": joint.aabb_center.astype(float).tolist(),
                    "aabb_size": joint.aabb_size.astype(float).tolist(),
                    "success": bool(attempt.get("success")),
                    "classification": attempt.get("classification"),
                }
            )
    return markers


def _protocol_implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("benchmark_metrics.py"),
        Path(__file__).with_name("benchmark_policies.py"),
        Path(__file__).with_name("benchmark_types.py"),
        Path(__file__).with_name("public_goal.py"),
        Path(__file__).with_name("restricted_gt_perception.py"),
        Path(__file__).with_name("ros_object_goal_adapter.py"),
        Path(__file__).with_name("trusted_interaction_skill.py"),
        Path(interactive_nav_v3.__file__),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_signature(config: BenchmarkEvaluationConfig, benchmark_sha256: str, indices: list[int]) -> tuple[str, dict[str, Any]]:
    config_payload = asdict(config)
    for key in ("benchmark", "output_dir", "resume", "workers"):
        config_payload.pop(key, None)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_implementation_sha256": _protocol_implementation_sha256(),
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
    restricted_public_mode = _is_ros_object_goal_rule(config)
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
    public_attempts: list[dict[str, Any]] = []
    restricted_ros_runtime: RestrictedRosObjectGoalRuntime | None = None
    navigation_steps = 0
    view_actions = 0
    nav_path_length = 0.0
    nav_sim_seconds = 0.0
    interaction_sim_seconds = 0.0
    terminal_reason = "max_steps"
    runtime_goal_consistency: dict[str, Any] | None = None
    runtime_consistency: dict[str, Any] | None = None
    scoring_exclusion_reasons: list[str] = []
    scoring_eligible = True
    runtime_goal_blocked = False
    catalog: InteractionCatalog | None = None
    private_visualization: dict[str, Any] | None = None
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
            # The stored V3 default was selected for manipulation and puts both
            # arms in the navigation camera.  This evaluator-only reset posture
            # is the established ROS navigation configuration; the frozen JSON
            # on disk remains unchanged.
            _apply_ros_navigation_arm_posture(spec)
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
        private_visualization = _private_episode_visualization_context(
            task=task,
            catalog=catalog,
            episode=episode,
        )
        runtime_consistency = _runtime_consistency_checks(task, catalog, episode, sampler, config)
        runtime_goal_consistency = runtime_consistency["checks"]["oracle_terminal_goal"]
        scoring_exclusion_reasons = list(runtime_consistency["exclusion_reasons"])
        scoring_eligible = bool(runtime_consistency["eligible"])
        runtime_goal_blocked = bool(
            config.require_runtime_goal_consistency
            and not scoring_eligible
        )
        public = _public_episode(episode, [camera.name for camera in spec.cameras], tuple(spec.img_resolution))
        if not runtime_goal_blocked and not config.quality_gate_only:
            policy = _build_policy(config, public)
            policy.reset(public)
            if isinstance(policy, ScriptedOraclePolicy):
                policy.reset_oracle(list(nav["oracle_plan"]["steps"]))
            if _is_ros_object_goal_rule(config):
                restricted_ros_runtime = _build_restricted_ros_object_goal_runtime(
                    task=task,
                    catalog=catalog,
                    episode=episode,
                    public=public,
                    config=config,
                    frame_callback=lambda: _capture_head_frame(task, frames, config.record_video),
                )
            base_data["policy_name"] = str(getattr(policy, "name", config.policy))
            base_data["uses_oracle_gt"] = bool(getattr(policy, "uses_oracle_gt", False))
        public_runtime_goal_consistency = runtime_goal_consistency
        public_runtime_consistency = runtime_consistency
        if restricted_public_mode:
            public_runtime_goal_consistency, public_runtime_consistency = _redact_runtime_consistency_for_restricted_policy(
                runtime_consistency
            )
        public_runtime_compatibility = sampler.runtime_compatibility
        if restricted_public_mode:
            public_runtime_compatibility = {
                key: value
                for key, value in sampler.runtime_compatibility.items()
                if key != "dropped_noncritical_object_pose_names"
            }
        runtime_trace: dict[str, Any] = {
            "runtime_compatibility": public_runtime_compatibility,
            "oracle_terminal_goal_consistency": public_runtime_goal_consistency,
            "runtime_consistency": public_runtime_consistency,
            "runtime_cameras": [camera.name for camera in spec.cameras],
            "runtime_depth_cameras": [camera.name for camera in spec.cameras if camera.record_depth],
            "runtime_image_resolution": list(spec.img_resolution),
            "benchmark_image_resolution": list(episode.get("img_resolution", [])),
            "interaction_catalog_joint_count": len(catalog.joints),
        }
        if _is_current_ros_policy(config):
            runtime_trace["effective_navigation_arm_qpos"] = {
                name: list(spec.robot.init_qpos[name])
                for name in ROS_NAVIGATION_ARM_QPOS
                if name in spec.robot.init_qpos
            }
        if restricted_public_mode:
            runtime_trace["restricted_gt_protocol"] = {
                "enabled": True,
                "camera_name": "head_camera",
                "minimum_visible_pixels": int(config.restricted_gt_min_visible_pixels),
                "interaction_endpoint": "opaque_object_open",
            }
        trace.append(
            {
                "runtime": runtime_trace
            }
        )
        _capture_head_frame(task, frames, config.record_video)
        previous_action: dict[str, Any] | None = None

        if runtime_goal_blocked:
            terminal_reason = "runtime_consistency_ineligible"
        elif config.quality_gate_only:
            terminal_reason = "quality_gate_complete"
        for decision_index in range(
            0 if runtime_goal_blocked or config.quality_gate_only else int(config.max_steps)
        ):
            nav_ok, target_distance, target_visibility = target_metrics(task, episode)
            terminal_score = score_interactions(task.env, episode, attempts)
            # The task endpoint is deliberately independent from the hidden V3
            # interaction recipe.  Formal interaction-conditioned success is
            # evaluated below from private postconditions, but a method that has
            # reached and sees the actual target finishes its rollout now.
            if nav_ok:
                terminal_reason = "target_found"
                break
            if restricted_ros_runtime is not None:
                consumed = _consume_pending_ros_object_goal_interaction(
                    task=task,
                    runtime=restricted_ros_runtime,
                    episode=episode,
                    private_attempts=attempts,
                    config=config,
                    decision_index=decision_index,
                    frames=frames,
                )
                if consumed is not None:
                    attempts.append(consumed["private_attempt"])
                    public_attempts.append(consumed["public_attempt"])
                    interaction_sim_seconds += float(consumed["simulated_seconds"])
                    observation = consumed["observation"]
                    previous_action = {
                        "kind": "interact",
                        **consumed["public_attempt"],
                    }
                    trace.append(
                        {
                            "decision_step": decision_index,
                            "interaction": consumed["public_attempt"],
                        }
                    )
                    continue
            policy_observation = PolicyObservation(
                observation=observation,
                instruction=public.instruction,
                step_index=decision_index,
                elapsed_seconds=time.monotonic() - started,
                previous_action=previous_action,
            )
            action = policy.act(policy_observation)
            event: dict[str, Any] = {"decision_step": decision_index, "action": action.to_dict()}
            if restricted_ros_runtime is not None:
                consumed = _consume_pending_ros_object_goal_interaction(
                    task=task,
                    runtime=restricted_ros_runtime,
                    episode=episode,
                    private_attempts=attempts,
                    config=config,
                    decision_index=decision_index,
                    frames=frames,
                )
                if consumed is not None:
                    attempts.append(consumed["private_attempt"])
                    public_attempts.append(consumed["public_attempt"])
                    interaction_sim_seconds += float(consumed["simulated_seconds"])
                    observation = consumed["observation"]
                    previous_action = {
                        "kind": "interact",
                        **consumed["public_attempt"],
                    }
                    trace.append(
                        {
                            "decision_step": decision_index,
                            "interaction": consumed["public_attempt"],
                        }
                    )
                    continue
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
                if restricted_ros_runtime is not None:
                    _publish_restricted_ros_frame(restricted_ros_runtime, task, decision_index=decision_index)
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
                if restricted_ros_runtime is not None:
                    _publish_restricted_ros_frame(restricted_ros_runtime, task, decision_index=decision_index)
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                trace.append(event)
                continue
            if action.kind == "observe":
                event["observe"] = {"refreshed": True}
                observation = task.get_observations()
                _capture_head_frame(task, frames, config.record_video)
                if restricted_ros_runtime is not None:
                    _publish_restricted_ros_frame(restricted_ros_runtime, task, decision_index=decision_index)
                _discard_task_rollout_cache(task)
                previous_action = action.to_dict()
                trace.append(event)
                continue
            if action.kind != "interact":
                raise ValueError(f"Unsupported action kind: {action.kind}")
            if restricted_ros_runtime is not None:
                raise ValueError(
                    "ros_object_goal_rule must submit opaque interaction commands on "
                    f"{config.ros_interaction_command_topic}, not a direct PolicyAction"
                )

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
            public_attempts.append(attempt)
            interaction_sim_seconds += float(simulated_seconds)
            event["interaction"] = attempt
            _capture_head_frame(task, frames, config.record_video)
            _discard_task_rollout_cache(task)
            previous_action = action.to_dict()
            trace.append(event)
        else:
            if not runtime_goal_blocked and not config.quality_gate_only:
                terminal_reason = "max_steps"

        nav_ok, target_distance, target_visibility = target_metrics(task, episode)
        terminal_score = score_interactions(task.env, episode, attempts)
        requirement = str(nav["interaction_requirement"])
        task_success = bool(nav_ok)
        interaction_conditioned_success = bool(
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
            # Preserve ``success`` as the historical formal V3 score while
            # exposing the visible task endpoint independently.
            success=interaction_conditioned_success,
            task_success=task_success,
            interaction_conditioned_success=interaction_conditioned_success,
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
            spl=spl(interaction_conditioned_success, reference, nav_path_length),
            navigation_simulated_seconds=nav_sim_seconds,
            interaction_simulated_seconds=interaction_sim_seconds,
            total_simulated_seconds=nav_sim_seconds + interaction_sim_seconds + view_actions * float(config.policy_dt_ms) / 1000.0,
            elapsed_seconds=time.monotonic() - started,
            target_distance_m=target_distance,
            target_visibility_fraction=target_visibility,
            interaction_attempts=public_attempts if restricted_public_mode else attempts,
            trace_path=str(trace_path),
            video_path=video_path,
            scoring_eligible=scoring_eligible,
            scoring_exclusion_reasons=scoring_exclusion_reasons,
            runtime_goal_consistency=public_runtime_goal_consistency,
            runtime_consistency=public_runtime_consistency,
        ).to_dict()
        terminal_trace: dict[str, Any] = {
            "task_success": task_success,
            "interaction_conditioned_success": interaction_conditioned_success,
            "success": interaction_conditioned_success,
        }
        if not restricted_public_mode:
            terminal_trace["interaction_score"] = terminal_score.to_dict()
        else:
            # The complete V3 score remains evaluator-private.  Do not write
            # frozen interaction IDs/fractions into a trace that may be shared
            # with the method being evaluated.
            terminal_trace["interaction_score"] = {
                "required_interaction_success": terminal_score.required_interaction_success,
                "sequence_success": terminal_score.sequence_success,
                "non_interaction_success": terminal_score.non_interaction_success,
            }
        trace.append({"terminal": terminal_trace})
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        public_runtime_goal_consistency = runtime_goal_consistency
        public_runtime_consistency = runtime_consistency
        if restricted_public_mode:
            public_runtime_goal_consistency, public_runtime_consistency = _redact_runtime_consistency_for_restricted_policy(
                runtime_consistency
            )
            error = "restricted evaluator exception"
        result = EpisodeResult(
            **base_data,
            status="exception",
            success=False,
            task_success=False,
            interaction_conditioned_success=False,
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
            interaction_attempts=public_attempts if restricted_public_mode else attempts,
            trace_path=str(trace_path),
            error=error,
            scoring_eligible=False,
            scoring_exclusion_reasons=(
                scoring_exclusion_reasons
                if scoring_exclusion_reasons
                else ["runtime_exception"]
            ),
            runtime_goal_consistency=public_runtime_goal_consistency,
            runtime_consistency=public_runtime_consistency,
        ).to_dict()
        if restricted_public_mode:
            trace.append({"exception": error})
        else:
            trace.append({"exception": error, "traceback": traceback.format_exc()})
        status = "exception"
    finally:
        if restricted_ros_runtime is not None:
            try:
                restricted_ros_runtime.adapter.close()
            except Exception:
                pass
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
    if private_visualization is not None and catalog is not None:
        private_visualization["actual_interactions"] = _private_actual_interaction_markers(attempts, catalog)
        _atomic_json(episode_dir / "episode_visualization.json", private_visualization)
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
    scoring_path = output_dir / "scoring_manifest.jsonl"
    scoring_tmp = scoring_path.with_name(f".{scoring_path.name}.{os.getpid()}.tmp")
    with scoring_tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_safe_json({
                "schema_version": "interactive_nav_v3_scoring_eligibility_v1",
                "episode_index": row.get("episode_index"),
                "case_id": row.get("case_id"),
                "house_index": row.get("house_index"),
                "domains": row.get("domains", []),
                "interaction_requirement": row.get("interaction_requirement"),
                "status": row.get("status"),
                "scoring_eligible": bool(row.get("scoring_eligible", False)),
                "exclusion_reasons": row.get("scoring_exclusion_reasons", []),
                "runtime_consistency": row.get("runtime_consistency"),
            }), ensure_ascii=False) + "\n")
    scoring_tmp.replace(scoring_path)
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
            f"TaskSR={values['task_success_rate']}, ICS={values['interaction_conditioned_success_rate']}, "
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
    progress_started = time.monotonic()

    def record_progress(*, force: bool = False) -> None:
        completed = len(rows)
        if not force and completed % int(config.progress_every) != 0:
            return
        elapsed = max(time.monotonic() - progress_started, 1e-9)
        rate = completed / elapsed
        remaining = len(payloads) - completed
        eta_seconds = None if rate <= 0 else remaining / rate
        payload = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "processed": completed,
            "total": len(payloads),
            "progress": 0.0 if not payloads else completed / len(payloads),
            "eligible": sum(bool(row.get("scoring_eligible")) for row in rows),
            "ineligible": sum(not bool(row.get("scoring_eligible")) for row in rows),
            "exceptions": sum(row.get("status") == "exception" for row in rows),
            "rate_episodes_per_second": rate,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
        }
        _atomic_json(config.output_dir / "progress.json", payload)
        print(
            "[quality-gate-progress] "
            f"time={payload['time']} processed={completed}/{len(payloads)} "
            f"eligible={payload['eligible']} ineligible={payload['ineligible']} "
            f"exceptions={payload['exceptions']} rate={rate:.3f}/s "
            f"eta_s={'unknown' if eta_seconds is None else int(eta_seconds)}",
            flush=True,
        )

    if config.workers == 1:
        for payload in payloads:
            rows.append(_worker(payload))
            record_progress()
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [pool.submit(_worker, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                record_progress()
    record_progress(force=True)
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
    parser.add_argument(
        "--policy",
        choices=["noop", "scripted_oracle", "ros_bridge", "ros_object_goal_rule", "factory"],
        default="noop",
    )
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
        "--require-interaction-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the evaluator-private interaction target to be visible before sealed force execution.",
    )
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
    parser.add_argument("--ros-target-topic", default="/semantic_decision/target")
    parser.add_argument("--ros-restricted-gt-topic", default="/semantic_mapping/gt_observations")
    parser.add_argument("--ros-interaction-command-topic", default="/semantic_decision/interaction_command")
    parser.add_argument("--ros-interaction-result-topic", default="/semantic_mapping/interaction_result")
    parser.add_argument("--restricted-gt-min-visible-pixels", type=int, default=16)
    parser.add_argument(
        "--quality-gate-only",
        action="store_true",
        help="Load and validate runtime consistency without executing policy actions.",
    )
    parser.add_argument("--runtime-joint-position-tolerance", type=float, default=0.02)
    parser.add_argument("--runtime-joint-fraction-tolerance", type=float, default=0.05)
    parser.add_argument("--runtime-base-position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--runtime-base-yaw-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=10)
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
        require_interaction_visible=args.require_interaction_visible,
        require_runtime_goal_consistency=args.require_runtime_goal_consistency,
        allow_internal_object_names=args.allow_internal_object_names,
        oracle_navigation_mode=args.oracle_navigation_mode,
        ros_observation_topic=args.ros_observation_topic,
        ros_action_topic=args.ros_action_topic,
        ros_action_timeout_s=args.ros_action_timeout_s,
        ros_cmd_vel_linear_gain=args.ros_cmd_vel_linear_gain,
        ros_require_move_base_active=args.ros_require_move_base_active,
        ros_map_warmup_skip_frames=args.ros_map_warmup_skip_frames,
        ros_target_topic=args.ros_target_topic,
        ros_restricted_gt_topic=args.ros_restricted_gt_topic,
        ros_interaction_command_topic=args.ros_interaction_command_topic,
        ros_interaction_result_topic=args.ros_interaction_result_topic,
        restricted_gt_min_visible_pixels=args.restricted_gt_min_visible_pixels,
        quality_gate_only=args.quality_gate_only,
        runtime_joint_position_tolerance=args.runtime_joint_position_tolerance,
        runtime_joint_fraction_tolerance=args.runtime_joint_fraction_tolerance,
        runtime_base_position_tolerance_m=args.runtime_base_position_tolerance_m,
        runtime_base_yaw_tolerance_rad=args.runtime_base_yaw_tolerance_rad,
        progress_every=args.progress_every,
    )


def main() -> int:
    config = parse_args()
    output = run_evaluation(config)
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
