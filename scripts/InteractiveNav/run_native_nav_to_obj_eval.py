#!/usr/bin/env python3
"""Run the current interactive navigation stack on the native MolmoSpaces benchmark.

The benchmark/task construction and success accounting are owned by
``molmo_spaces.evaluation.eval_main``.  This file only supplies the ROS bridge
policy, the native JSON task-sampler hook, and the rollout/debug hooks needed by
the interactive navigation system.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MLSPACES_CACHE_DIR", str(Path.home() / ".cache/molmo-spaces-resources"))
os.environ.setdefault("MLSPACES_ASSETS_DIR", str(_REPO_ROOT / "assets"))

# ROS setup files may prepend another editable MolmoSpaces checkout.  The
# native benchmark run must use the code in this worktree, including the
# runner injection in eval_main.py.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import RBY1Config
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.evaluation.eval_main import run_evaluation
from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner
from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

from force_interaction_bridge import AtomicForceInteractionController
from force_interaction_runtime import ForceDriveConfig
from ros_completion_monitor import CompletionMonitorConfig, RosCompletionMonitor
from run_nav_ros_sim import NavRosRolloutRunner
from scripts.InteractiveNav.evaluation.benchmark_runner import ROS_NAVIGATION_ARM_QPOS


log = logging.getLogger(__name__)


DEFAULT_BENCHMARK_DIR = Path(
    os.environ["MLSPACES_ASSETS_DIR"]
)
DEFAULT_BENCHMARK_DIR = DEFAULT_BENCHMARK_DIR / (
    "benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _target_metadata(task) -> dict[str, Any]:
    task_config = task.config.task_config
    pickup_obj_name = str(getattr(task_config, "pickup_obj_name", ""))
    candidates = [
        str(name)
        for name in (getattr(task_config, "pickup_obj_candidates", None) or [])
        if str(name)
    ]
    if not candidates and pickup_obj_name:
        candidates = [pickup_obj_name]

    object_manager = task.env.object_managers[task.env.current_batch_index]
    category = ""
    natural_name = ""
    try:
        category = str(object_manager.category_from_name(pickup_obj_name) or "")
    except Exception:
        pass
    try:
        natural_name = str(object_manager.fallback_expression(pickup_obj_name) or "")
    except Exception:
        natural_name = pickup_obj_name.replace("_", " ").strip()
    if not natural_name:
        natural_name = category or pickup_obj_name.replace("_", " ").strip()
    if not category:
        category = natural_name

    # Keep the ROS-side early-exit criterion aligned with the benchmark task.
    # The executor receives this value through the target context and still
    # leaves the task itself as the final success authority.
    try:
        success_distance_threshold_m = float(
            getattr(task_config, "succ_pos_threshold", 1.5)
        )
    except (TypeError, ValueError):
        success_distance_threshold_m = 1.5
    if not math.isfinite(success_distance_threshold_m) or success_distance_threshold_m <= 0.0:
        success_distance_threshold_m = 1.5

    labels = sorted(
        {
            value.casefold()
            for value in (category, natural_name)
            if str(value).strip()
        }
    )
    selection_mode = str(getattr(task_config, "selection_mode", "any_candidate"))
    target_context = {
        "enabled": True,
        "selection_mode": selection_mode,
        "target_name": natural_name,
        "object_labels": labels,
        # NavTask defines success over this benchmark-provided candidate set.
        # Keep semantic target selection on the same instances instead of
        # accepting another object of the same category elsewhere in the house.
        "candidate_instances": candidates,
        "standoff_m": 1.0,
        # False means the ordinary nav-to-object goal does not require an
        # interaction, but it intentionally leaves all interaction candidates
        # enabled.  A hidden target can therefore still trigger module 2.
        "require_interaction": False,
        "completion_requires_visibility": True,
        "require_current_visibility": False,
        "target_require_same_room": False,
        "target_allow_connected_room": True,
        "target_min_visible_pixels": 16,
        # Mirror NavToObjTask's planar distance threshold.  The executor must
        # also observe the selected target before it can signal completion.
        "success_distance_threshold_m": success_distance_threshold_m,
    }
    return {
        "mode": "native_molmospaces_nav_to_obj",
        "selection_mode": selection_mode,
        "pickup_obj_name": pickup_obj_name,
        "pickup_obj_category": category,
        "target_name": natural_name,
        "object_labels": labels,
        "candidate_instances": candidates,
        "target_context": target_context,
        "interaction_available": True,
    }


def _resolve_distance_adaptive_horizon_steps(
    base_horizon_steps: int,
    initial_goal_distance_m: float,
    *,
    minimum_steps: int,
    fixed_overhead_steps: int,
    steps_per_meter: float,
) -> int:
    """Return a conservative per-episode screening horizon.

    ``initial_goal_distance_m`` is a straight-line lower bound, not a
    traversable-path estimate.  The result is therefore clamped to the
    original benchmark horizon and used only when the opt-in screening mode is
    enabled.
    """

    if base_horizon_steps <= 0 or not math.isfinite(initial_goal_distance_m):
        return base_horizon_steps
    lower_bound = min(base_horizon_steps, max(1, int(minimum_steps)))
    proposed_steps = math.ceil(
        float(fixed_overhead_steps) + float(steps_per_meter) * initial_goal_distance_m
    )
    return min(base_horizon_steps, max(lower_bound, proposed_steps))


def _apply_native_navigation_arm_posture(episode_spec) -> dict[str, list[float]]:
    """Tuck RBY1 arms for navigation without modifying the benchmark JSON."""

    applied: dict[str, list[float]] = {}
    for group_name, qpos in ROS_NAVIGATION_ARM_QPOS.items():
        if group_name not in episode_spec.robot.init_qpos:
            continue
        posture = list(qpos)
        episode_spec.robot.init_qpos[group_name] = posture
        applied[group_name] = posture
    return applied


class NativeRosBridgePolicy(RosBridgePolicy):
    """ROS bridge configured for a native JSON ``NavToObjTask`` episode."""

    def __init__(self, config, task=None) -> None:
        # eval_main constructs the preloaded policy with ``task_type`` as the
        # second argument.  The rollout hook assigns the real task immediately
        # before reset(), so do not retain that string as a task object.
        if not hasattr(task, "env"):
            task = None
        debug_root = Path(config.native_debug_dir).expanduser().resolve()
        debug_root.mkdir(parents=True, exist_ok=True)
        self.native_debug_root = debug_root
        self.native_episode_index = 0
        self.current_native_episode_dir: Path | None = None
        self.native_target_metadata: dict[str, Any] = {}
        self.native_episode_active = False
        self.native_episode_generation = 0
        self.native_dynamic_horizon_enabled = bool(
            config.native_dynamic_horizon_enabled
        )
        self.native_dynamic_horizon_min_steps = int(
            config.native_dynamic_horizon_min_steps
        )
        self.native_dynamic_horizon_base_steps = int(
            config.native_dynamic_horizon_base_steps
        )
        self.native_dynamic_horizon_steps_per_meter = float(
            config.native_dynamic_horizon_steps_per_meter
        )
        self.native_horizon_metadata: dict[str, Any] = {}

        super().__init__(
            config,
            task,
            observation_topic=config.native_observation_topic,
            action_topic=config.native_action_topic,
            pointcloud_topic=config.native_pointcloud_topic,
            camera_info_topic=config.native_camera_info_topic,
            depth_topic=config.native_depth_topic,
            action_timeout_s=config.native_action_timeout_s,
            blocking_observation_republish_period_s=0.25,
            blocking_republish_pointcloud=False,
            queue_size=1,
            observation_queue_size=0,
            publish_pointcloud=True,
            publish_camera_info=True,
            depth_camera_name="head_camera",
            pointcloud_frame_id="tf_frame_lidar",
            optical_frame_id="head_camera_optical_frame",
            pointcloud_stride=2,
            pointcloud_self_filter_radius_m=0.32,
            odom_topic="/odom",
            publish_odom=True,
            publish_odom_twist=True,
            map_frame_id="tf_frame_map",
            odom_frame_id="tf_frame_odom",
            base_frame_id="tf_frame_base_link",
            allow_static_lidar_tf_fallback=False,
            cmd_vel_topic="/cmd_vel_stamped",
            cmd_vel_timeout_s=0.5,
            require_fresh_cmd_vel=True,
            require_move_base_active_for_cmd_vel=False,
            map_warmup_skip_frames=config.native_map_warmup_skip_frames,
            immediate_noop_after_publish=False,
            timing_log_every_n_frames=30,
            extra_image_topic="",
            extra_image_camera_name="",
            publish_realtime_gt=True,
            realtime_gt_topic="/semantic_mapping/gt_observations",
            realtime_gt_camera_name="head_camera",
            realtime_gt_min_visible_pixels=16,
            # Keep the bridge gate aligned with the target-candidate gate.
            # Thin objects can have a reliable pixel count but occupy only a
            # tiny portion of a loose projected box.
            realtime_gt_min_visible_fraction=0.0,
            realtime_gt_required_consecutive_observations=2,
            realtime_gt_step_interval=3,
            realtime_gt_max_distance_m=4.0,
            # Keep the simulator frame manifest at the run root so the
            # existing offline six-panel video builder can consume it.
            step_frame_dir=str(debug_root.parent / "sim_step_frames"),
            step_frame_queue_size=4,
            step_sync_topic="/molmo_spaces/step_sync",
        )

        self.scene_timeout_s = float(config.native_scene_timeout_s)
        self.max_consecutive_action_timeouts = int(
            config.native_max_consecutive_action_timeouts
        )
        self.retain_task_history = False
        self.sim_timing_log_every_n_steps = int(config.native_sim_timing_log_every_n_steps)
        self.step_log_every_n_steps = int(config.native_step_log_every_n_steps)
        self.debug_snapshot_path = str(debug_root / "native_first_frame.png")
        self.debug_snapshot_camera_name = "head_camera"
        self.debug_snapshot_saved = False
        # Semantic goal-status is the only ROS-side lifecycle signal that can
        # end a native rollout.  It is deliberately separate from execution
        # state/individual behavior feedback; NavToObjTask still judges the
        # official benchmark success after the loop exits.
        self.completion_monitor = RosCompletionMonitor(
            CompletionMonitorConfig(
                mode="semantic",
                semantic_target_requires_distance_and_visibility=True,
            ),
            semantic_topic="/semantic_decision/goal_status",
            output_path=debug_root / "completion_status.json",
        )

        self._native_target_publisher = self._rospy.Publisher(
            "/semantic_decision/target",
            self._String,
            queue_size=1,
            latch=True,
        )
        self.force_interaction_controller = AtomicForceInteractionController(
            command_topic="/semantic_decision/interaction_command",
            result_topic="/semantic_mapping/interaction_result",
            feedback_topic="/semantic_decision/interaction_action_feedback",
            output_path=debug_root / "force_interaction_events.json",
            force_config=ForceDriveConfig(
                max_physics_substeps=3000,
                open_fraction_threshold=0.67,
                assume_success=False,
            ),
            # Preserve the native benchmark articulation state.  The controller
            # still listens for and executes explicit interaction commands.
            close_all_doors_on_prepare=False,
            close_all_containers_on_prepare=False,
            interaction_execution_mode=config.native_interaction_execution_mode,
            interaction_transition_steps=config.native_interaction_transition_steps,
            drawer_execution_mode=config.native_interaction_execution_mode,
            drawer_transition_steps=config.native_interaction_transition_steps,
            drawer_observation_steps=1,
        )

    def _configure_episode_horizon(self) -> None:
        """Record the initial GT distance and optionally shorten this episode."""

        raw_horizon = getattr(self.task, "_native_base_task_horizon_steps", None)
        if raw_horizon is None:
            raw_horizon = getattr(self.task, "_task_horizon", None)
        try:
            base_horizon_steps = int(raw_horizon)
        except (TypeError, ValueError, OverflowError):
            self.native_horizon_metadata = {
                "dynamic_horizon_enabled": self.native_dynamic_horizon_enabled,
                "base_task_horizon_steps": raw_horizon,
                "effective_task_horizon_steps": raw_horizon,
                "initial_goal_distance_error": "task_horizon_not_finite",
            }
            return

        # A rollout framework may reset its policy more than once for the same
        # task.  Preserve the original benchmark cap so a second reset does
        # not treat an already shortened horizon as its new baseline.
        setattr(self.task, "_native_base_task_horizon_steps", base_horizon_steps)

        horizon_metadata: dict[str, Any] = {
            "dynamic_horizon_enabled": self.native_dynamic_horizon_enabled,
            "base_task_horizon_steps": base_horizon_steps,
            "effective_task_horizon_steps": base_horizon_steps,
        }
        try:
            initial_goal_distance_m = float(self.task.calculate_distance(0))
        except Exception as exc:
            horizon_metadata["initial_goal_distance_error"] = str(exc)
            self.native_horizon_metadata = horizon_metadata
            return
        if not math.isfinite(initial_goal_distance_m):
            horizon_metadata["initial_goal_distance_error"] = "distance_not_finite"
            self.native_horizon_metadata = horizon_metadata
            return

        horizon_metadata["initial_goal_distance_m"] = initial_goal_distance_m
        if self.native_dynamic_horizon_enabled:
            effective_horizon_steps = _resolve_distance_adaptive_horizon_steps(
                base_horizon_steps,
                initial_goal_distance_m,
                minimum_steps=self.native_dynamic_horizon_min_steps,
                fixed_overhead_steps=self.native_dynamic_horizon_base_steps,
                steps_per_meter=self.native_dynamic_horizon_steps_per_meter,
            )
            # ``BaseMujocoTask`` reads this value in ``is_done`` every step.
            # Do not modify the shared experiment config: only this episode is
            # distance-adaptive, and the original benchmark limit remains an
            # upper bound.
            self.task._task_horizon = effective_horizon_steps
            horizon_metadata["effective_task_horizon_steps"] = effective_horizon_steps
            horizon_metadata["formula"] = {
                "minimum_steps": self.native_dynamic_horizon_min_steps,
                "fixed_overhead_steps": self.native_dynamic_horizon_base_steps,
                "steps_per_meter": self.native_dynamic_horizon_steps_per_meter,
            }
            log.info(
                "Distance-adaptive screening horizon: %.3fm -> %d/%d steps",
                initial_goal_distance_m,
                effective_horizon_steps,
                base_horizon_steps,
            )
        self.native_horizon_metadata = horizon_metadata

    def reset(self):
        super().reset()
        self.native_episode_index += 1
        self.native_episode_generation += 1
        self.native_episode_active = True
        self.debug_snapshot_saved = False
        self.current_native_episode_dir = self.native_debug_root / (
            f"episode_{self.native_episode_index:04d}"
        )
        self.current_native_episode_dir.mkdir(parents=True, exist_ok=True)
        self.force_interaction_controller.output_path = (
            self.current_native_episode_dir / "force_interaction_events.json"
        )

        self._configure_episode_horizon()
        self.native_target_metadata = _target_metadata(self.task)
        self.native_target_metadata["horizon"] = dict(self.native_horizon_metadata)
        target_context = dict(self.native_target_metadata["target_context"])
        target_context["episode_id"] = f"native_nav_to_obj_{self.native_episode_index:04d}"
        target_context["episode_active"] = True
        target_context["episode_generation"] = self.native_episode_generation
        self.native_target_metadata["target_context"] = target_context
        self.completion_monitor.output_path = (
            self.current_native_episode_dir / "completion_status.json"
        )
        self.completion_monitor.configure_semantic_episode(
            episode_id=str(target_context["episode_id"]),
            episode_generation=int(target_context["episode_generation"]),
        )
        _write_json(
            self.current_native_episode_dir / "target_selection.json",
            self.native_target_metadata,
        )
        _write_json(
            self.native_debug_root / "target_selection.json",
            self.native_target_metadata,
        )
        self._native_target_publisher.publish(
            self._String(
                data=json.dumps(target_context, ensure_ascii=False, separators=(",", ":"))
            )
        )

    def finish_episode(
        self,
        step_index: int | None = None,
        *,
        reason: str = "native_rollout_finished",
    ) -> None:
        """Stop downstream semantic actions before evaluator finalization."""
        if not self.native_episode_active:
            return
        self.native_episode_active = False
        target_context = dict(self.native_target_metadata.get("target_context") or {})
        target_context.update(
            {
                "enabled": False,
                "episode_active": False,
                "episode_generation": self.native_episode_generation,
                "finish_reason": str(reason),
            }
        )
        if step_index is not None:
            target_context["finish_step"] = int(step_index)
        self._native_target_publisher.publish(
            self._String(
                data=json.dumps(target_context, ensure_ascii=False, separators=(",", ":"))
            )
        )
        self._move_base_cancel_pub.publish(self._GoalID())
        # Do not reset the explorer after the terminal episode transition.  A
        # reset starts a fresh initial-spin timer, which can race ROS shutdown
        # and keep producing frontier work after the benchmark is finished.
        # The target lifecycle transition above and the move_base cancel are
        # sufficient to stop downstream semantic/execution nodes.

    def close(self):
        try:
            self.finish_episode(reason="policy_close")
            self.force_interaction_controller.close()
        finally:
            try:
                self._native_target_publisher.unregister()
            except Exception:
                pass
            self.completion_monitor.close()
            super().close()


class NativeRosBridgePolicyConfig(BasePolicyConfig):
    policy_cls: type = NativeRosBridgePolicy
    policy_type: str = "ros_bridge"


class NativeNavToObjEvalConfig(JsonBenchmarkEvalConfig):
    """Official JSON benchmark config with the current ROS interaction stack."""

    robot_config: RBY1Config = RBY1Config()
    policy_config: NativeRosBridgePolicyConfig = NativeRosBridgePolicyConfig()

    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 10.0
    sim_dt_ms: float = 10.0
    # Preserve the first physical state that satisfies NavToObjTask's official
    # distance-and-visibility criterion before a subsequent ROS action can
    # occlude the target again.
    end_on_success: bool = True
    record_videos: bool = True

    native_debug_dir: Path = Path("native_nav_debug")
    native_observation_topic: str = "/molmo_spaces/head_camera/image"
    native_action_topic: str = "/molmo_spaces/action"
    native_pointcloud_topic: str = "/registered_scan"
    native_camera_info_topic: str = "/molmo_spaces/head_camera/camera_info"
    native_depth_topic: str = "/molmo_spaces/head_camera/depth"
    native_action_timeout_s: float = 0.5
    native_scene_timeout_s: float = 1200.0
    native_max_consecutive_action_timeouts: int = 0
    native_map_warmup_skip_frames: int = 3
    native_sim_timing_log_every_n_steps: int = 50
    native_step_log_every_n_steps: int = 50
    native_interaction_execution_mode: str = "smooth"
    native_interaction_transition_steps: int = 5
    native_filter_missing_scene_objects: bool = False
    native_dynamic_horizon_enabled: bool = False
    native_dynamic_horizon_min_steps: int = 360
    native_dynamic_horizon_base_steps: int = 240
    native_dynamic_horizon_steps_per_meter: float = 45.0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        env_debug_dir = os.environ.get("NATIVE_NAV_DEBUG_DIR")
        if env_debug_dir:
            self.native_debug_dir = Path(env_debug_dir).expanduser().resolve()
        record_videos = os.environ.get("NATIVE_NAV_RECORD_VIDEOS")
        if record_videos:
            self.record_videos = record_videos.strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        action_timeout_s = os.environ.get("NATIVE_NAV_ACTION_TIMEOUT_S")
        if action_timeout_s:
            try:
                parsed_action_timeout_s = float(action_timeout_s)
            except ValueError as exc:
                raise ValueError(
                    "NATIVE_NAV_ACTION_TIMEOUT_S must be a positive number"
                ) from exc
            if (
                not math.isfinite(parsed_action_timeout_s)
                or parsed_action_timeout_s <= 0.0
            ):
                raise ValueError("NATIVE_NAV_ACTION_TIMEOUT_S must be a positive number")
            self.native_action_timeout_s = parsed_action_timeout_s
        max_consecutive_action_timeouts = os.environ.get(
            "NATIVE_NAV_MAX_CONSECUTIVE_ACTION_TIMEOUTS"
        )
        if max_consecutive_action_timeouts:
            try:
                self.native_max_consecutive_action_timeouts = int(
                    max_consecutive_action_timeouts
                )
            except ValueError as exc:
                raise ValueError(
                    "NATIVE_NAV_MAX_CONSECUTIVE_ACTION_TIMEOUTS must be a non-negative integer"
                ) from exc
            if self.native_max_consecutive_action_timeouts < 0:
                raise ValueError(
                    "NATIVE_NAV_MAX_CONSECUTIVE_ACTION_TIMEOUTS must be a non-negative integer"
                )
        dynamic_horizon = os.environ.get("NATIVE_NAV_DYNAMIC_HORIZON")
        if dynamic_horizon:
            normalized_dynamic_horizon = dynamic_horizon.strip().lower()
            if normalized_dynamic_horizon in {"1", "true", "yes", "on"}:
                self.native_dynamic_horizon_enabled = True
            elif normalized_dynamic_horizon in {"0", "false", "no", "off"}:
                self.native_dynamic_horizon_enabled = False
            else:
                raise ValueError(
                    "NATIVE_NAV_DYNAMIC_HORIZON must be a boolean value"
                )
        dynamic_min_steps = os.environ.get("NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS")
        if dynamic_min_steps:
            try:
                self.native_dynamic_horizon_min_steps = int(dynamic_min_steps)
            except ValueError as exc:
                raise ValueError(
                    "NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS must be a positive integer"
                ) from exc
            if self.native_dynamic_horizon_min_steps <= 0:
                raise ValueError(
                    "NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS must be a positive integer"
                )
        dynamic_base_steps = os.environ.get("NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS")
        if dynamic_base_steps:
            try:
                self.native_dynamic_horizon_base_steps = int(dynamic_base_steps)
            except ValueError as exc:
                raise ValueError(
                    "NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS must be a non-negative integer"
                ) from exc
            if self.native_dynamic_horizon_base_steps < 0:
                raise ValueError(
                    "NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS must be a non-negative integer"
                )
        dynamic_steps_per_meter = os.environ.get(
            "NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER"
        )
        if dynamic_steps_per_meter:
            try:
                self.native_dynamic_horizon_steps_per_meter = float(
                    dynamic_steps_per_meter
                )
            except ValueError as exc:
                raise ValueError(
                    "NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER must be non-negative"
                ) from exc
            if (
                not math.isfinite(self.native_dynamic_horizon_steps_per_meter)
                or self.native_dynamic_horizon_steps_per_meter < 0.0
            ):
                raise ValueError(
                    "NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER must be non-negative"
                )
        if os.environ.get("NATIVE_NAV_FILTER_MISSING_SCENE_OBJECTS", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.native_filter_missing_scene_objects = True

    @property
    def tag(self) -> str:
        return "native_nav_to_obj_ros_bridge"


class NativeNavToObjJsonEvalRunner(JsonEvalRunner):
    """Native JSON runner with depth enabled for ROS mapping/debug only."""

    @staticmethod
    def get_episode_task_sampler(
        exp_config,
        episode_spec,
        shared_task_sampler,
        datagen_profiler,
    ) -> JsonEvalTaskSampler:
        # The benchmark's RGB/camera poses remain authoritative.  The native
        # ROS mapper additionally needs head depth, so only this sensor flag is
        # enabled at runtime; it does not change the task or initial state.
        runtime_spec = episode_spec.model_copy(deep=True)
        for camera in runtime_spec.cameras:
            if str(getattr(camera, "name", "")) == "head_camera":
                camera.record_depth = True
        applied_arm_posture = _apply_native_navigation_arm_posture(runtime_spec)
        if applied_arm_posture:
            log.info(
                "Applied ROS navigation arm posture for native nav_to_obj replay: %s",
                applied_arm_posture,
            )
        sampler_cls = JsonEvalTaskSampler
        if getattr(exp_config, "native_filter_missing_scene_objects", False):
            sampler_cls = NativeCompatibleJsonEvalTaskSampler
        sampler = sampler_cls(exp_config, runtime_spec)
        if datagen_profiler is not None:
            sampler.set_datagen_profiler(datagen_profiler)
        return sampler

    @staticmethod
    def run_single_rollout(*args, **kwargs):
        policy = kwargs.get("policy")
        task = kwargs.get("task")
        if policy is None and len(args) >= 3:
            task = args[1]
            policy = args[2]
        try:
            success = NavRosRolloutRunner.run_single_rollout(*args, **kwargs)
        except Exception as exc:
            if policy is not None and getattr(policy, "current_native_episode_dir", None):
                failure_result = {
                    "exception": type(exc).__name__,
                    "error": str(exc),
                }
                if getattr(policy, "native_horizon_metadata", None):
                    failure_result["horizon"] = dict(policy.native_horizon_metadata)
                _write_json(
                    policy.current_native_episode_dir / "official_nav_to_obj_result.json",
                    failure_result,
                )
            raise
        finally:
            if policy is not None and hasattr(policy, "finish_episode"):
                policy.finish_episode(reason="rollout_runner_returned")

        result = {
            "official_task": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "official_success": bool(success),
        }
        if task is not None:
            try:
                result["task_description"] = task.get_task_description()
            except Exception:
                pass
            try:
                result["distance_m"] = float(task.calculate_distance(0))
            except Exception as exc:
                result["distance_error"] = str(exc)
            try:
                result["head_camera_visible"] = bool(task.check_object_visible(0))
            except Exception as exc:
                result["visibility_error"] = str(exc)
        if policy is not None and getattr(policy, "current_native_episode_dir", None):
            result["debug_dir"] = str(policy.current_native_episode_dir)
            if getattr(policy, "native_horizon_metadata", None):
                result["horizon"] = dict(policy.native_horizon_metadata)
            _write_json(
                policy.current_native_episode_dir / "official_nav_to_obj_result.json",
                result,
            )
            _write_json(policy.native_debug_root / "official_nav_to_obj_result.json", result)
        return bool(success)


class NativeCompatibleJsonEvalTaskSampler(JsonEvalTaskSampler):
    """Replay helper for a benchmark/resource-version mismatch.

    The native JSON sampler remains the default.  When explicitly enabled, this
    subclass removes only benchmark pose/joint entries whose bodies/joints are
    absent from the locally installed scene.  It does not synthesize objects or
    alter the target; the report is written so this run cannot be mistaken for an
    exact benchmark replay.
    """

    def randomize_scene(self, env, robot_view) -> None:
        model = env.current_model
        available_bodies = {
            str(model.body(body_id).name) for body_id in range(model.nbody)
        }
        available_joints = {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
            for joint_id in range(model.njnt)
        }

        scene_modifications = self.episode_spec.scene_modifications
        original_object_poses = dict(scene_modifications.object_poses)
        original_articulation_states = list(scene_modifications.articulation_states)
        missing_object_poses = sorted(
            body_name for body_name in original_object_poses if body_name not in available_bodies
        )
        missing_articulation_states = [
            state
            for state in original_articulation_states
            if state.joint_name not in available_joints
        ]

        if missing_object_poses or missing_articulation_states:
            runtime_spec = self.episode_spec.model_copy(deep=True)
            runtime_spec.scene_modifications.object_poses = {
                body_name: pose
                for body_name, pose in original_object_poses.items()
                if body_name in available_bodies
            }
            runtime_spec.scene_modifications.articulation_states = [
                state
                for state in original_articulation_states
                if state.joint_name in available_joints
            ]
            self.episode_spec = runtime_spec

        report = {
            "mode": "compatibility_filter_missing_scene_objects",
            "exact_native_replay": False,
            "house_index": int(self.episode_spec.house_index),
            "missing_object_pose_bodies": missing_object_poses,
            "missing_articulation_states": [
                {
                    "object_name": str(state.object_name),
                    "joint_name": str(state.joint_name),
                }
                for state in missing_articulation_states
            ],
            "original_object_pose_count": len(original_object_poses),
            "replayed_object_pose_count": len(self.episode_spec.scene_modifications.object_poses),
            "original_articulation_state_count": len(original_articulation_states),
            "replayed_articulation_state_count": len(
                self.episode_spec.scene_modifications.articulation_states
            ),
        }
        self.compatibility_report = report
        debug_root = getattr(self.config, "native_debug_dir", None)
        if debug_root is not None:
            traj_key = str(getattr(self.episode_spec.source, "traj_key", "episode"))
            safe_traj_key = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in traj_key
            )
            _write_json(
                Path(debug_root) / "benchmark_compatibility" / (
                    f"house_{self.episode_spec.house_index}_{safe_traj_key}.json"
                ),
                report,
            )
        if missing_object_poses or missing_articulation_states:
            log.warning(
                "Native benchmark compatibility mode filtered %d missing object poses "
                "and %d missing articulation states for house %s.",
                len(missing_object_poses),
                len(missing_articulation_states),
                self.episode_spec.house_index,
            )

        super().randomize_scene(env, robot_view)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current interactive navigation algorithm on native nav_to_obj."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-root", type=Path, default=Path("native_nav_to_obj_eval"))
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--task-horizon-steps", type=int, default=None)
    parser.add_argument(
        "--filter-missing-scene-objects",
        action="store_true",
        help="Compatibility mode for benchmark/resource-version mismatches; not an exact benchmark replay.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    debug_dir = args.debug_dir or Path(
        os.environ.get("NATIVE_NAV_DEBUG_DIR", str(args.output_root / "debug"))
    )
    debug_dir = debug_dir.expanduser().resolve()
    os.environ["NATIVE_NAV_DEBUG_DIR"] = str(debug_dir)
    if args.filter_missing_scene_objects:
        os.environ["NATIVE_NAV_FILTER_MISSING_SCENE_OBJECTS"] = "1"
    debug_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    log.info("Native benchmark directory: %s", args.benchmark_dir)
    log.info("Native eval output root: %s", args.output_root)
    log.info("Interactive debug directory: %s", debug_dir)

    results = run_evaluation(
        eval_config_cls=NativeNavToObjEvalConfig,
        benchmark_dir=args.benchmark_dir,
        task_horizon_steps=args.task_horizon_steps,
        output_dir=args.output_root,
        num_workers=1,
        use_wandb=False,
        episode_idx=args.episode_idx,
        runner_cls=NativeNavToObjJsonEvalRunner,
    )
    summary = {
        "benchmark_dir": str(args.benchmark_dir.resolve()),
        "official_eval_output_dir": str(results.output_dir),
        "success_count": int(results.success_count),
        "total_count": int(results.total_count),
        "success_rate": float(results.success_rate),
        "debug_dir": str(debug_dir),
        "native_filter_missing_scene_objects": bool(
            getattr(results.exp_config, "native_filter_missing_scene_objects", False)
        ),
        "native_dynamic_horizon": {
            "enabled": bool(
                getattr(results.exp_config, "native_dynamic_horizon_enabled", False)
            ),
            "minimum_steps": int(
                getattr(results.exp_config, "native_dynamic_horizon_min_steps", 0)
            ),
            "fixed_overhead_steps": int(
                getattr(results.exp_config, "native_dynamic_horizon_base_steps", 0)
            ),
            "steps_per_meter": float(
                getattr(
                    results.exp_config,
                    "native_dynamic_horizon_steps_per_meter",
                    0.0,
                )
            ),
        },
    }
    _write_json(results.output_dir / "native_eval_summary.json", summary)
    _write_json(debug_dir / "native_eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
