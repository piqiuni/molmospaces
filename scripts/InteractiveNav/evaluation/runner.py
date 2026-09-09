"""Independent InteractiveNav V3 evaluator.

Unlike ``molmo_spaces.evaluation.eval_main``, this runner evaluates navigation
*and* interaction: it replays V3 articulation state, executes the configured
force controller, enforces prerequisite order, and writes per-episode traces.
It intentionally does not modify the upstream MolmoSpaces evaluation package.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Keep standalone evaluation consistent with the headless collection pipeline.
# This must precede importing MuJoCo/renderer-owning project modules.
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
from scripts.InteractiveNav.collection.interaction_executors import (
    InteractionExecutionRequest,
    build_interaction_executor,
)

from .metrics import (
    interaction_terminal_metrics,
    joint_open_fraction,
    summarise_results,
    target_metrics,
)
from .policies import MolmoSpacesPolicyAdapter, ScriptedOraclePolicy, build_builtin_policy
from .types import EpisodeResult, InteractionRecord, PolicyAction, PolicyObservation


@dataclass
class EvaluationConfig:
    benchmark: Path
    output_dir: Path
    policy: str = "noop"
    workers: int = 1
    max_steps: int = 500
    episode_indices: list[int] | None = None
    max_episodes: int | None = None
    resume: bool = False
    force_max_steps: int = 1000
    force_target_fraction: float = 1.0
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 10.0
    sim_dt_ms: float = 10.0
    record_video: bool = False
    video_fps: float = 5.0
    ros_action_timeout_s: float = 0.2
    ros_observation_topic: str = "/molmo_spaces/head_camera/image"
    ros_action_topic: str = "/molmo_spaces/action"
    camera_names: list[str] = field(default_factory=lambda: ["head_camera"])
    image_resolution: tuple[int, int] | None = (320, 240)

    def validate(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.policy_dt_ms <= 0 or self.ctrl_dt_ms <= 0 or self.sim_dt_ms <= 0:
            raise ValueError("all timing values must be positive")
        if not math.isclose(self.policy_dt_ms / self.ctrl_dt_ms, round(self.policy_dt_ms / self.ctrl_dt_ms)):
            raise ValueError("policy_dt_ms must be an integer multiple of ctrl_dt_ms")
        if not math.isclose(self.ctrl_dt_ms / self.sim_dt_ms, round(self.ctrl_dt_ms / self.sim_dt_ms)):
            raise ValueError("ctrl_dt_ms must be an integer multiple of sim_dt_ms")
        if self.policy == "ros_bridge" and self.workers != 1:
            raise ValueError("ros_bridge evaluation requires --workers 1; a ROS master is stateful")
        if not self.camera_names:
            raise ValueError("camera_names must contain at least one camera")
        if self.image_resolution is not None and (
            len(self.image_resolution) != 2 or min(self.image_resolution) <= 0
        ):
            raise ValueError("image_resolution must be a positive (width, height) pair or None")


class InteractiveNavRBY1ReplayConfig(NavToObjBaseConfig):
    """Local replay config; it exists only for the standalone evaluator."""

    robot_config: RBY1Config = RBY1Config()
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 10.0
    sim_dt_ms: float = 10.0
    task_horizon: int = 500
    output_dir: Path = Path("interactive_nav_v3_eval_output")
    use_passive_viewer: bool = False
    record_videos: bool = False

    @property
    def tag(self) -> str:
        return "interactive_nav_v3_eval"


class V3JsonEvalTaskSampler(JsonEvalTaskSampler):
    """JSON sampler with V3 asset-drift checks in its *single* live context.

    The current Objaverse installation can omit a few irrelevant free objects
    present when the benchmark was authored.  The upstream sampler correctly
    rejects those object poses, but constructing a separate scene merely to
    discover them doubles the MuJoCo/EGL peak allocation.  Inspect the actual
    task model immediately before upstream pose replay instead.
    """

    runtime_compatibility: dict[str, Any]

    def __init__(self, exp_config: Any, episode_spec: EpisodeSpec, interactive_nav: dict[str, Any]) -> None:
        self._interactive_nav = interactive_nav
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
        missing_critical_bodies = sorted(critical_bodies - bodies)
        if missing_critical_bodies:
            raise RuntimeError(
                "Runtime scene is missing task-critical bodies: "
                + ", ".join(missing_critical_bodies)
            )
        articulation_names = {
            str(row.joint_name) for row in self.episode_spec.scene_modifications.articulation_states
        }
        interaction_joints = {str(row["joint_name"]) for row in nav.get("interactions", [])}
        missing_joints = sorted((articulation_names | interaction_joints) - joints)
        if missing_joints:
            raise RuntimeError(
                "Runtime scene is missing recorded articulation joints: " + ", ".join(missing_joints)
            )
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


def _load_episodes(benchmark: Path) -> list[dict[str, Any]]:
    benchmark_file = benchmark / "benchmark.json" if benchmark.is_dir() else benchmark
    payload = json.loads(benchmark_file.read_text())
    if not isinstance(payload, list):
        payload = payload.get("episodes", [])
    if not isinstance(payload, list):
        raise ValueError(f"Expected episode list in {benchmark_file}")
    return payload


def _episode_indices(config: EvaluationConfig, episodes: list[dict[str, Any]]) -> list[int]:
    if config.episode_indices is not None:
        indices = list(config.episode_indices)
    else:
        indices = list(range(len(episodes)))
    if config.max_episodes is not None:
        indices = indices[: int(config.max_episodes)]
    for index in indices:
        if not 0 <= index < len(episodes):
            raise IndexError(f"Episode index {index} outside [0, {len(episodes)})")
    return indices


def _build_replay_config(config: EvaluationConfig, output_dir: Path) -> InteractiveNavRBY1ReplayConfig:
    replay = InteractiveNavRBY1ReplayConfig()
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
    replay.robot_config.init_qpos["head"] = np.zeros(2, dtype=float)
    replay.robot_config.init_qpos_noise_range["head"] = np.zeros(2, dtype=float)
    return replay


def _public_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Return policy-visible episode context without V3 supervision."""

    return {
        "house_index": int(episode["house_index"]),
        "scene_dataset": str(episode["scene_dataset"]),
        "data_split": str(episode["data_split"]),
        "language": {
            "task_description": str(episode.get("language", {}).get("task_description", "")),
        },
        "task_type": str(episode.get("task", {}).get("task_type", "nav_to_obj")),
    }


def _make_policy(config: EvaluationConfig, replay_config: Any, task: Any):
    if config.policy in {"noop", "scripted_oracle"}:
        return build_builtin_policy(config.policy)
    if config.policy == "ros_bridge":
        from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy

        policy = RosBridgePolicy(
            config=replay_config,
            task=task,
            observation_topic=config.ros_observation_topic,
            action_topic=config.ros_action_topic,
            action_timeout_s=float(config.ros_action_timeout_s),
        )
        return MolmoSpacesPolicyAdapter(policy, name="ros_bridge")
    raise ValueError(f"Unsupported evaluator policy={config.policy!r}")


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _step_path_increment(task: Any, previous_xy: np.ndarray) -> tuple[np.ndarray, float]:
    current_xy = np.asarray(task.env.current_robot.robot_view.base.pose[:2, 3], dtype=float).copy()
    return current_xy, float(np.linalg.norm(current_xy - previous_xy))


def _base_pose_xyyaw(task: Any) -> np.ndarray:
    """Return the live planar base pose used for oracle waypoint completion."""

    pose = np.asarray(task.env.current_robot.robot_view.base.pose, dtype=float)
    yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
    return np.asarray([pose[0, 3], pose[1, 3], yaw], dtype=float)


def _discard_task_rollout_cache(task: Any) -> None:
    """Release upstream trajectory tensors after the evaluator consumed a step.

    ``BaseMujocoTask.step`` retains every RGB/depth observation for collection.
    V3 benchmark evaluation writes a compact JSON trace instead, so retaining
    hundreds of image tensors per episode only increases per-worker memory and
    can make parallel EGL evaluation unstable.  This runs after the first-step
    consistency check has already completed.
    """

    for name in (
        "observation_cache",
        "reward_cache",
        "terminal_cache",
        "truncated_cache",
        "success_cache",
        "action_cache",
    ):
        cache = getattr(task, name, None)
        if isinstance(cache, list):
            cache.clear()


def _oracle_base_action(action: PolicyAction) -> dict[str, Any] | None:
    if action.base_action is not None:
        return action.base_action
    point = action.metadata.get("oracle_goal_point")
    if point is None:
        return None
    return {
        "base": np.asarray(
            [float(point[0]), float(point[1]), float(action.metadata.get("oracle_goal_yaw", 0.0))],
            dtype=float,
        )
    }


def _resolve_interaction(
    episode: dict[str, Any], action: PolicyAction, *, uses_oracle_gt: bool
) -> tuple[dict[str, Any] | None, str | None]:
    interactions = list(episode["interactive_nav"].get("interactions", []))
    oracle_id = action.metadata.get("oracle_interaction_id")
    if oracle_id is not None:
        if not uses_oracle_gt:
            raise PermissionError("Only an explicitly oracle policy may emit oracle_interaction_id")
        for row in interactions:
            if row["interaction_id"] == oracle_id:
                return row, None
        return None, f"unknown_oracle_interaction_id:{oracle_id}"
    matches = [
        row
        for row in interactions
        if row["object_name"] == action.object_name
        and (action.joint_index is None or int(row["joint_index"]) == int(action.joint_index))
    ]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "requested_object_or_joint_not_in_episode"
    return None, "ambiguous_object_joint_request"


def _execute_force_interaction(
    env: Any, interaction: dict[str, Any], config: EvaluationConfig
) -> tuple[bool, dict[str, Any], float, float]:
    before = joint_open_fraction(env, interaction)
    domain = "channel" if str(interaction["type"]).startswith("channel_") else "container"
    request = InteractionExecutionRequest(
        domain=domain,
        object_name=str(interaction["object_name"]),
        joint_name=str(interaction["joint_name"]),
        joint_index=int(interaction["joint_index"]),
        target_fraction=float(config.force_target_fraction),
        max_steps=int(config.force_max_steps),
        interaction_id=str(interaction["interaction_id"]),
    )
    # Preserve the base throughout the external-force phase.  This follows the
    # current full-collection force semantics and avoids collision impulses
    # accidentally moving a navigation trajectory.
    robot_base = env.current_robot.robot_view.base
    base_pose = robot_base.pose.copy()
    executor = build_interaction_executor("force")
    result = executor.execute(env, request)
    robot_base.pose = base_pose
    robot_base.joint_vel = np.zeros_like(robot_base.joint_vel)
    mujoco.mj_forward(env.current_model, env.current_data)
    after = joint_open_fraction(env, interaction)
    success = bool(result.success and after >= 0.8)
    return success, _safe_json(result.metadata), before, after


def _write_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(payload), indent=2, ensure_ascii=False) + "\n")


def evaluate_episode(
    config: EvaluationConfig, episode_index: int, episode: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate exactly one V3 episode in an isolated process/context."""

    config.output_dir = Path(config.output_dir)
    config.benchmark = Path(config.benchmark)
    nav = episode["interactive_nav"]
    case_id = str(nav["case_id"])
    episode_dir = config.output_dir / "episodes" / f"{episode_index:04d}_{case_id[:96]}"
    trace_path = episode_dir / "episode_result.json"
    if config.resume and trace_path.is_file():
        existing = json.loads(trace_path.read_text())
        if existing.get("status") == "complete" and "result" in existing:
            return existing["result"]

    interactive_nav_v3.validate_interactive_nav_v3_episode(
        episode, expected_domains=list(nav["interaction_domains"])
    )
    replay_config = _build_replay_config(config, episode_dir)
    spec = EpisodeSpec.model_validate(episode)
    # Evaluation policies currently consume first-person head observations.
    # Replaying every benchmark recording camera (wrist/follower) is needlessly
    # expensive and can exhaust renderer memory when several MuJoCo workers
    # start at once.  This is a runtime observation selection only: the frozen
    # benchmark JSON remains untouched and callers may explicitly request more
    # cameras through ``--camera-names``.
    requested_cameras = set(config.camera_names)
    original_camera_names = [camera.name for camera in spec.cameras]
    recorded_resolution = tuple(spec.img_resolution)
    spec.cameras = [camera for camera in spec.cameras if camera.name in requested_cameras]
    missing_cameras = sorted(requested_cameras - set(camera.name for camera in spec.cameras))
    if missing_cameras:
        raise ValueError(
            f"episode {episode_index} does not define requested camera(s): {missing_cameras}; "
            f"available={original_camera_names}"
        )
    if config.image_resolution is not None:
        spec.img_resolution = tuple(int(value) for value in config.image_resolution)
    sampler = V3JsonEvalTaskSampler(replay_config, spec, nav)
    task = None
    policy = None
    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    interaction_records: list[InteractionRecord] = []
    executed_ids: list[str] = []
    navigation_steps = 0
    interaction_action_count = 0
    wrong_interaction_count = 0
    nav_path_length = 0.0
    error: str | None = None
    terminal_reason = "max_steps"
    start_time = time.monotonic()

    try:
        # JsonEvalTaskSampler resolves the logical assets/ scene path.  The
        # current resource installation keeps the actual XML under the cache
        # mirror, so use the same writable-path resolver as collection/full
        # rollout before sampling.  This modifies only the sampler-local map.
        # ``update_scene`` normally re-installs resources based on a path
        # relative to ``assets/scenes``.  The writable scene mirror is outside
        # that root by design (it protects the shared resource cache from
        # MuJoCo writes).  The resource was already resolved above, so keep
        # this process-local no-op used by the existing collection loader.
        import molmo_spaces.tasks.task_sampler as task_sampler_module

        task_sampler_module.install_scene_with_objects_and_grasps_from_path = (
            lambda *args, **kwargs: {}
        )
        dataset_map = sampler._get_dataset_index_map()
        variants = dataset_map[spec.data_split][spec.house_index]
        variants["base"] = probe.prepare_writable_scene_path(Path(variants["base"]))
        task = sampler.sample_task(house_index=spec.house_index, variant="base")
        if task is None:
            raise RuntimeError("JsonEvalTaskSampler returned no task")
        trace.append(
            {
                "runtime_compatibility": sampler.runtime_compatibility,
                "runtime_observation_cameras": [camera.name for camera in spec.cameras],
                "runtime_image_resolution": list(spec.img_resolution),
                "benchmark_image_resolution": list(recorded_resolution),
            }
        )
        observation, _info = task.reset()
        # The standard sampler restored all recorded state during reset.  Check
        # it here before giving the policy its first observation.
        initial_nav_success, initial_distance, initial_visibility = target_metrics(task, episode)
        policy = _make_policy(config, replay_config, task)
        public = _public_episode(episode)
        if getattr(policy, "uses_oracle_gt", False):
            public["_oracle_steps"] = list(nav["oracle_plan"]["steps"])
        policy.reset(public)
        previous_xy = np.asarray(task.env.current_robot.robot_view.base.pose[:2, 3], dtype=float).copy()
        previous_action: dict[str, Any] | None = None

        for step_index in range(int(config.max_steps)):
            nav_success, distance, visibility = target_metrics(task, episode)
            # Native NavToObj success is only a subcondition for V3.  In a
            # required-interaction episode it must not short-circuit the
            # policy before it performs the recorded environmental change.
            if nav_success and str(nav["interaction_requirement"]) == "unnecessary":
                terminal_reason = "nav_success"
                break
            policy_obs = PolicyObservation(
                observation=observation,
                instruction=public["language"]["task_description"],
                step_index=step_index,
                elapsed_seconds=time.monotonic() - start_time,
                previous_action=previous_action,
            )
            action = policy.act(policy_obs)
            action_dict = action.to_dict()
            event: dict[str, Any] = {"step": step_index, "action": action_dict}

            if action.kind == "stop":
                terminal_reason = str(action.metadata.get("reason", "policy_stop"))
                trace.append(event)
                break
            if action.kind == "base":
                base_action = _oracle_base_action(action)
                if base_action is None:
                    raise ValueError("base action is missing base_action and oracle goal")
                observation, reward, terminated, truncated, infos = task.step(base_action)
                previous_xy, increment = _step_path_increment(task, previous_xy)
                nav_path_length += increment
                navigation_steps += 1
                notify = getattr(policy, "notify_action_result", None)
                if callable(notify):
                    notify(action, base_pose=_base_pose_xyyaw(task))
                event.update(
                    {
                        "reward": float(np.asarray(reward).reshape(-1)[0]),
                        "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
                        "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
                        "infos": _safe_json(infos),
                        "path_increment_m": increment,
                        "base_pose_xyyaw": _base_pose_xyyaw(task).tolist(),
                    }
                )
                if config.record_video and isinstance(observation, list) and observation and "head_camera" in observation[0]:
                    frames.append(np.asarray(observation[0]["head_camera"], dtype=np.uint8).copy())
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    terminal_reason = "task_terminated"
                    trace.append(event)
                    _discard_task_rollout_cache(task)
                    break
                _discard_task_rollout_cache(task)
            elif action.kind == "interact":
                interaction_action_count += 1
                interaction, resolve_error = _resolve_interaction(
                    episode, action, uses_oracle_gt=bool(getattr(policy, "uses_oracle_gt", False))
                )
                if interaction is None:
                    wrong_interaction_count += 1
                    record = InteractionRecord(
                        requested_object_name=action.object_name,
                        requested_joint_index=action.joint_index,
                        resolved_interaction_id=None,
                        success=False,
                        joint_fraction_before=None,
                        joint_fraction_after=None,
                        executor="force",
                        metadata={"resolve_error": resolve_error},
                    )
                else:
                    success, metadata, before, after = _execute_force_interaction(task.env, interaction, config)
                    if not success:
                        wrong_interaction_count += 1
                    else:
                        executed_ids.append(str(interaction["interaction_id"]))
                    observation = task.get_observations()
                    record = InteractionRecord(
                        requested_object_name=action.object_name,
                        requested_joint_index=action.joint_index,
                        resolved_interaction_id=str(interaction["interaction_id"]),
                        success=success,
                        joint_fraction_before=before,
                        joint_fraction_after=after,
                        executor="force",
                        metadata=metadata,
                    )
                interaction_records.append(record)
                event["interaction"] = record.to_dict()
            else:
                raise ValueError(f"Unsupported policy action kind: {action.kind}")
            previous_action = action_dict
            trace.append(event)
        else:
            terminal_reason = "max_steps"

        nav_success, distance, visibility = target_metrics(task, episode)
        required_ok, sequence_ok, non_interaction_ok, fractions = interaction_terminal_metrics(
            task.env, episode, executed_ids
        )
        requirement = str(nav["interaction_requirement"])
        success = bool(nav_success and required_ok and sequence_ok and non_interaction_ok)
        video_path = None
        if config.record_video and frames:
            video = episode_dir / "head_camera.mp4"
            probe.save_frames_to_mp4(frames, str(video), fps=float(config.video_fps))
            video_path = str(video)
        result = EpisodeResult(
            episode_index=episode_index,
            case_id=case_id,
            house_index=int(episode["house_index"]),
            domains=list(nav["interaction_domains"]),
            interaction_requirement=requirement,
            policy_name=str(getattr(policy, "name", type(policy).__name__)),
            uses_oracle_gt=bool(getattr(policy, "uses_oracle_gt", False)),
            success=success,
            nav_success=nav_success,
            required_interaction_success=required_ok,
            sequence_success=sequence_ok,
            non_interaction_success=non_interaction_ok if requirement == "unnecessary" else None,
            terminal_reason=terminal_reason,
            step_count=len(trace),
            navigation_step_count=navigation_steps,
            interaction_action_count=interaction_action_count,
            wrong_interaction_count=wrong_interaction_count,
            navigation_path_length_m=nav_path_length,
            elapsed_seconds=time.monotonic() - start_time,
            target_distance_m=distance,
            target_visibility_fraction=visibility,
            interaction_records=[record.to_dict() for record in interaction_records],
            trace_path=str(trace_path),
            video_path=video_path,
        ).to_dict()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = EpisodeResult(
            episode_index=episode_index,
            case_id=case_id,
            house_index=int(episode["house_index"]),
            domains=list(nav.get("interaction_domains", [])),
            interaction_requirement=str(nav.get("interaction_requirement", "unknown")),
            policy_name=config.policy,
            uses_oracle_gt=config.policy == "scripted_oracle",
            success=False, nav_success=False, required_interaction_success=False, sequence_success=False,
            non_interaction_success=None, terminal_reason="exception", step_count=len(trace),
            navigation_step_count=navigation_steps, interaction_action_count=interaction_action_count,
            wrong_interaction_count=wrong_interaction_count, navigation_path_length_m=nav_path_length,
            elapsed_seconds=time.monotonic() - start_time, target_distance_m=None,
            target_visibility_fraction=None, interaction_records=[r.to_dict() for r in interaction_records],
            trace_path=str(trace_path), error=error,
        ).to_dict()
        trace.append({"exception": error, "traceback": traceback.format_exc()})
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

    _write_trace(trace_path, {"status": "complete", "result": result, "trace": trace})
    return result


def _worker(payload: tuple[dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
    config_data, index, episode = payload
    config = EvaluationConfig(**config_data)
    return evaluate_episode(config, index, episode)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Run a V3 benchmark with one isolated MuJoCo context per episode."""

    config.validate()
    config.benchmark = Path(config.benchmark).resolve()
    config.output_dir = Path(config.output_dir).resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = _load_episodes(config.benchmark)
    indices = _episode_indices(config, episodes)
    config_data = asdict(config)
    config_data["benchmark"] = str(config.benchmark)
    config_data["output_dir"] = str(config.output_dir)
    rows: list[dict[str, Any]] = []
    payloads = [(config_data, index, episodes[index]) for index in indices]
    if config.workers == 1:
        for payload in payloads:
            rows.append(_worker(payload))
    else:
        # Process workers rather than Python threads: MuJoCo objects are not
        # thread-safe and each worker must own its renderer/context.
        # MuJoCo/EGL is not fork-safe once imported in the coordinator process.
        # Use a clean interpreter for every worker, rather than inheriting GL
        # handles or renderer thread state through Linux's default ``fork``.
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
            "benchmark": str(config.benchmark),
            "policy": config.policy,
            "workers": config.workers,
            "max_steps": config.max_steps,
            "episode_indices": indices,
            "result_count": len(rows),
        }
    )
    (config.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"results": rows, "summary": summary}


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=["noop", "scripted_oracle", "ros_bridge"], default="noop")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-max-steps", type=int, default=1000)
    parser.add_argument("--force-target-fraction", type=float, default=1.0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--ros-action-timeout-s", type=float, default=0.2)
    parser.add_argument("--ros-observation-topic", default="/molmo_spaces/head_camera/image")
    parser.add_argument("--ros-action-topic", default="/molmo_spaces/action")
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=["head_camera"],
        help="Benchmark cameras exposed to the policy (default: head_camera only).",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[320, 240],
        help="Runtime policy image resolution; use --image-resolution 640 480 to replay the recorded size.",
    )
    args = parser.parse_args()
    return EvaluationConfig(
        benchmark=args.benchmark, output_dir=args.output_dir, policy=args.policy, workers=args.workers,
        max_steps=args.max_steps, episode_indices=args.episode_indices, max_episodes=args.max_episodes,
        resume=args.resume, force_max_steps=args.force_max_steps,
        force_target_fraction=args.force_target_fraction, record_video=args.record_video,
        video_fps=args.video_fps, ros_action_timeout_s=args.ros_action_timeout_s,
        ros_observation_topic=args.ros_observation_topic, ros_action_topic=args.ros_action_topic,
        camera_names=list(args.camera_names),
        image_resolution=tuple(args.image_resolution),
    )


def main() -> int:
    config = parse_args()
    output = run_evaluation(config)
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
