import argparse
import datetime
import logging
from pathlib import Path
import struct
import time
import zlib

import mujoco
import numpy as np

from robot_conversion_patches import patch_droid_config_for_rum

from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.camera_configs import (
    FixedExocentricCameraConfig,
    FrankaDroidCameraSystem,
    RBY1GoProD455CameraSystem,
    RobotMountedCameraConfig,
)
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig, FrankaRobotConfig, RBY1Config
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.policy.learned_policy.left_arm_keyboard_debug_policy import (
    LeftArmKeyboardDebugPolicy,
)
from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.profiler_utils import Profiler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SceneExecutionTimeout(RuntimeError):
    pass


def resolve_action_timeout_s(action_timeout_s: float, scene_timeout_s: float) -> float:
    if action_timeout_s > 0.0:
        return action_timeout_s
    if scene_timeout_s > 0.0:
        return min(5.0, scene_timeout_s)
    return action_timeout_s


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_rgb_png(path: Path, frame) -> None:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = np.nan_to_num(arr, nan=0.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected image-like array, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr[:, :, :3])
    height, width = arr.shape[:2]
    rows = [b"\x00" + arr[y].tobytes() for y in range(height)]
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n"
    data += _png_chunk(b"IHDR", header)
    data += _png_chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
    data += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def extract_observation_image(observation, camera_name: str):
    if isinstance(observation, list) and observation:
        observation = observation[0]
    if not isinstance(observation, dict):
        return None
    frame = observation.get(camera_name)
    if frame is not None:
        return frame
    return None


def maybe_save_debug_snapshot(policy, observation) -> None:
    snapshot_path = getattr(policy, "debug_snapshot_path", "")
    if not snapshot_path or getattr(policy, "debug_snapshot_saved", False):
        return
    camera_name = getattr(policy, "debug_snapshot_camera_name", "debug_front_camera")
    frame = extract_observation_image(observation, camera_name)
    if frame is None:
        return
    path = Path(snapshot_path).expanduser().resolve()
    write_rgb_png(path, frame)
    path.with_suffix(".txt").write_text(f"camera={camera_name}\nshape={np.asarray(frame).shape}\n")
    policy.debug_snapshot_saved = True
    log.info("Saved debug camera snapshot: %s", path)


def configure_run_file_logging(output_dir: Path) -> Path:
    """Attach a file logger under the current run output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_nav_ros_sim.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return log_path

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root_logger.addHandler(file_handler)
    return log_path


def ensure_head_camera_exists(camera_system) -> None:
    """
    Ensure nav task's expected camera name (`head_camera`) exists.

    NavToObjTask currently checks visibility using `head_camera`. Some robot camera systems
    (e.g. FrankaDroidCameraSystem) do not define that name, so we alias an exocentric camera.
    """
    names = [cam.name for cam in camera_system.cameras]
    if "head_camera" in names:
        return

    for preferred in ("exo_camera_1", "exo_camera", "wrist_camera"):
        for cam in camera_system.cameras:
            if cam.name == preferred:
                log.warning(
                    "Camera '%s' aliased to 'head_camera' for nav visibility check.", preferred
                )
                cam.name = "head_camera"
                return

    # Fallback: alias the first configured camera.
    if camera_system.cameras:
        first_name = camera_system.cameras[0].name
        log.warning("Camera '%s' aliased to 'head_camera' as fallback.", first_name)
        camera_system.cameras[0].name = "head_camera"


def enable_depth_for_camera(camera_system, camera_name: str) -> bool:
    """Enable depth output for a specific camera if present."""
    for cam in camera_system.cameras:
        if cam.name == camera_name:
            cam.record_depth = True
            return True
    return False


def disable_camera_randomization(camera_system) -> None:
    """Disable camera pose/FOV noise for deterministic ROS mapping/debug runs."""
    for cam in camera_system.cameras:
        if hasattr(cam, "fov_noise_degrees"):
            cam.fov_noise_degrees = None
        if hasattr(cam, "pos_noise_range"):
            cam.pos_noise_range = None
        if hasattr(cam, "orientation_noise_degrees"):
            cam.orientation_noise_degrees = None


def parse_qpos_csv(value: str | None, expected_len: int = 7):
    if not value:
        return None
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated joint values, got {len(values)}: {value!r}")
    import numpy as np

    return np.asarray(values, dtype="float32")


def parse_float_csv(value: str | None, expected_len: int):
    if not value:
        return None
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(values)}: {value!r}")
    return values


def yaw_to_quat_wxyz(yaw: float) -> list[float]:
    return [float(np.cos(yaw * 0.5)), 0.0, 0.0, float(np.sin(yaw * 0.5))]


def lookat_forward_up(camera_pos: list[float], target_pos: list[float]) -> tuple[list[float], list[float]]:
    forward = np.asarray(target_pos, dtype=float) - np.asarray(camera_pos, dtype=float)
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=float)
    else:
        right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / max(np.linalg.norm(up), 1e-6)
    return forward.tolist(), up.tolist()


class NavRosRolloutRunner(ParallelRolloutRunner):
    @staticmethod
    def patch_config(frozen_config, data=None, exp_config=None):
        if "FrankaRobotConfig" in str(type(frozen_config.robot_config)):
            if "rum" in str(type(exp_config.robot_config)).lower():
                return patch_droid_config_for_rum(frozen_config, data)
        return frozen_config

    @staticmethod
    def run_single_rollout(
        episode_seed: int,
        task: BaseMujocoTask,
        policy,
        profiler: Profiler | None = None,
        viewer=None,
        shutdown_event=None,
        datagen_profiler: Profiler | None = None,
        end_on_success: bool = False,
    ):
        log.info("Starting task.reset() ...")
        if hasattr(policy, "prepare_episode_reset"):
            policy.prepare_episode_reset()
        observation, _info = task.reset()
        log.info("task.reset() completed.")
        # print(f"Observation: {observation}", flush=True)
        if viewer is not None:
            viewer.sync()

        policy.task = task
        policy.reset()
        maybe_save_debug_snapshot(policy, observation)

        try:
            task.env.current_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            log.warning("Sleep flag not set.")

        success = False
        step_idx = 0
        episode_started_mono = time.monotonic()
        consecutive_action_timeouts = 0
        scene_timeout_s = max(0.0, float(getattr(policy, "scene_timeout_s", 0.0)))
        max_consecutive_action_timeouts = max(
            0, int(getattr(policy, "max_consecutive_action_timeouts", 0))
        )
        timing_count = 0
        timing_sums_ms = {
            "policy": 0.0,
            "task": 0.0,
            "physics": 0.0,
            "sensors": 0.0,
            "loop": 0.0,
        }
        timing_log_every = max(0, int(getattr(policy, "sim_timing_log_every_n_steps", 20)))
        step_log_every = max(0, int(getattr(policy, "step_log_every_n_steps", 10)))
        while not task.is_done():
            if shutdown_event is not None and shutdown_event.is_set():
                return False
            elapsed_s = time.monotonic() - episode_started_mono
            if scene_timeout_s > 0.0 and elapsed_s >= scene_timeout_s:
                raise SceneExecutionTimeout(
                    f"Scene exceeded wall timeout of {scene_timeout_s:.1f}s "
                    f"after {step_idx} completed steps"
                )

            maybe_save_debug_snapshot(policy, observation)
            loop_t0 = time.perf_counter()
            policy_t0 = time.perf_counter()
            action_cmd = policy.get_action(observation)
            policy_ms = (time.perf_counter() - policy_t0) * 1000.0
            if getattr(policy, "last_action_timed_out", False):
                consecutive_action_timeouts += 1
            else:
                consecutive_action_timeouts = 0
            elapsed_s = time.monotonic() - episode_started_mono
            if scene_timeout_s > 0.0 and elapsed_s >= scene_timeout_s:
                raise SceneExecutionTimeout(
                    f"Scene exceeded wall timeout of {scene_timeout_s:.1f}s "
                    f"while waiting for action after {step_idx} completed steps"
                )
            if (
                max_consecutive_action_timeouts > 0
                and consecutive_action_timeouts >= max_consecutive_action_timeouts
            ):
                raise SceneExecutionTimeout(
                    f"Scene produced no fresh navigation action for "
                    f"{consecutive_action_timeouts} consecutive waits "
                    f"after {step_idx} completed steps"
                )
            if action_cmd is None:
                break

            if step_log_every > 0 and (step_idx < 5 or step_idx % step_log_every == 0):
                print(f"Step: {step_idx} Action command[base]: {action_cmd['base']}", flush=True)

            task_t0 = time.perf_counter()
            observation, reward, terminal, truncated, infos = task.step(action_cmd)
            task_ms = (time.perf_counter() - task_t0) * 1000.0
            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            task_timing = getattr(task, "last_step_timing_ms", {})
            timing_count += 1
            timing_sums_ms["policy"] += policy_ms
            timing_sums_ms["task"] += task_ms
            timing_sums_ms["physics"] += float(task_timing.get("physics_step", 0.0))
            timing_sums_ms["sensors"] += float(task_timing.get("sensor_polling", 0.0))
            timing_sums_ms["loop"] += loop_ms
            if timing_log_every > 0 and timing_count >= timing_log_every:
                avg = {key: value / timing_count for key, value in timing_sums_ms.items()}
                print(
                    "SimLoop timing over "
                    f"{timing_count} steps: policy={avg['policy']:.2f}ms, "
                    f"task={avg['task']:.2f}ms "
                    f"(physics={avg['physics']:.2f}ms sensors={avg['sensors']:.2f}ms), "
                    f"loop={avg['loop']:.2f}ms, "
                    f"simulated_dt={float(task.config.policy_dt_ms) / 1000.0:.3f}s",
                    flush=True,
                )
                timing_count = 0
                for key in timing_sums_ms:
                    timing_sums_ms[key] = 0.0
            step_idx += 1
            # print(f"Observation: {observation}", flush=True)
            # print(f"Reward: {reward}", flush=True)
            # print(f"Terminal: {terminal}", flush=True)
            # print(f"Truncated: {truncated}", flush=True)
            # print(f"Infos: {infos}", flush=True)
            if end_on_success and "success" in infos[0] and infos[0]["success"]:
                success = True
                break

            if viewer is not None:
                viewer.sync()

        try:
            task.env.current_model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            log.warning("Sleep flag not reset.")

        success = task.judge_success() if hasattr(task, "judge_success") else success
        return success


def build_nav_config(args) -> NavToObjBaseConfig:
    control_steps_per_policy = args.policy_dt_ms / args.ctrl_dt_ms
    if args.ctrl_dt_ms <= 0.0 or not np.isclose(
        control_steps_per_policy, round(control_steps_per_policy)
    ):
        raise ValueError(
            "policy_dt_ms must be an integer multiple of ctrl_dt_ms "
            f"(got {args.policy_dt_ms} and {args.ctrl_dt_ms})"
        )
    sim_steps_per_control = args.ctrl_dt_ms / args.sim_dt_ms
    if args.sim_dt_ms <= 0.0 or not np.isclose(
        sim_steps_per_control, round(sim_steps_per_control)
    ):
        raise ValueError(
            "ctrl_dt_ms must be an integer multiple of sim_dt_ms "
            f"(got {args.ctrl_dt_ms} and {args.sim_dt_ms})"
        )

    cfg = NavToObjBaseConfig()
    cfg.seed = args.seed
    cfg.task_type = "nav_to_obj"
    cfg.scene_dataset = args.scene_dataset
    cfg.data_split = args.data_split
    cfg.task_horizon = args.task_horizon
    cfg.num_workers = 1
    cfg.use_passive_viewer = args.viewer
    cfg.ctrl_dt_ms = args.ctrl_dt_ms
    cfg.sim_dt_ms = args.sim_dt_ms

    cfg.task_sampler_config.samples_per_house = args.samples_per_house
    # nav_ros_sim is usually used for live ROS debugging; default to a single attempt.
    cfg.task_sampler_config.max_total_attempts_multiplier = args.max_scene_attempts
    if args.policy_mode == "left_arm_debug":
        # Keep the same scene running for multiple debug episodes.
        cfg.task_sampler_config.samples_per_house = max(args.samples_per_house, args.debug_loop_episodes)
    cfg.task_sampler_config.house_inds = args.house_inds or [args.house_ind]
    cfg.task_sampler_config.randomize_lighting = args.randomize_scene
    cfg.task_sampler_config.randomize_textures = args.randomize_scene
    cfg.task_sampler_config.randomize_dynamics = args.randomize_scene
    # Keep task sensors enabled: freeze_task_config() requires observation["qpos"].
    # Disabling sensors causes reset-time KeyError('qpos') in the datagen pipeline.
    cfg.task_config.use_sensors = True
    if args.disable_task_sensors:
        log.warning(
            "--disable_task_sensors is ignored for nav_ros_sim because task freezing "
            "requires sensor observation field 'qpos'."
        )

    if args.exploration_only:
        cfg.task_sampler_config.pickup_types = []
    elif args.target_types:
        cfg.task_sampler_config.pickup_types = [s.strip() for s in args.target_types.split(",")]

    if args.robot == "droid":
        cfg.robot_config = FrankaRobotConfig()
        cfg.camera_config = FrankaDroidCameraSystem()
        cfg.camera_config.img_resolution = (320, 240)
        ensure_head_camera_exists(cfg.camera_config)
    elif args.robot == "rby1":
        cfg.robot_config = RBY1Config()
        # Keep the navigation sensor head level from the first simulation step.
        # The sampler synchronizes the actuator target with this qpos below.
        cfg.robot_config.init_qpos["head"] = np.zeros(2, dtype=float)
        cfg.robot_config.init_qpos_noise_range["head"] = np.zeros(2, dtype=float)
        arm_qpos = parse_qpos_csv(args.initial_arm_qpos)
        left_arm_qpos = parse_qpos_csv(args.initial_left_arm_qpos)
        right_arm_qpos = parse_qpos_csv(args.initial_right_arm_qpos)
        if left_arm_qpos is None:
            left_arm_qpos = arm_qpos
        if right_arm_qpos is None:
            right_arm_qpos = arm_qpos
        if left_arm_qpos is not None:
            cfg.robot_config.init_qpos["left_arm"] = left_arm_qpos.copy()
            cfg.robot_config.init_qpos_noise_range["left_arm"] = left_arm_qpos * 0.0
        if right_arm_qpos is not None:
            cfg.robot_config.init_qpos["right_arm"] = right_arm_qpos.copy()
            cfg.robot_config.init_qpos_noise_range["right_arm"] = right_arm_qpos * 0.0
        fixed_pose_xyyaw = parse_float_csv(args.fixed_robot_xyyaw, expected_len=3)
        if fixed_pose_xyyaw is not None:
            x, y, yaw = fixed_pose_xyyaw
            cfg.task_config.robot_base_pose = [x, y, 0.1, *yaw_to_quat_wxyz(yaw)]
        cfg.camera_config = RBY1GoProD455CameraSystem()
        if not args.include_wrist_cameras:
            cfg.camera_config.cameras = [
                camera for camera in cfg.camera_config.cameras if camera.name == "head_camera"
            ]
        if args.publish_debug_front_camera:
            fixed_camera_pos = parse_float_csv(args.fixed_debug_camera_pos, expected_len=3)
            fixed_camera_target = parse_float_csv(args.fixed_debug_camera_target, expected_len=3)
            if fixed_camera_pos is not None and fixed_camera_target is not None:
                forward, up = lookat_forward_up(fixed_camera_pos, fixed_camera_target)
                cfg.camera_config.cameras.append(
                    FixedExocentricCameraConfig(
                        name=args.debug_front_camera_name,
                        pos=fixed_camera_pos,
                        forward=forward,
                        up=up,
                        fov=75.0,
                        skip_erosion=True,
                    )
                )
            else:
                debug_camera_offset = parse_float_csv(args.debug_front_camera_offset, expected_len=3)
                debug_camera_lookat_offset = parse_float_csv(args.debug_front_camera_lookat_offset, expected_len=3)
                cfg.camera_config.cameras.append(
                    RobotMountedCameraConfig(
                        name=args.debug_front_camera_name,
                        reference_body_names=["robot_0/base", "base"],
                        camera_offset=debug_camera_offset or [-1.4, 0.0, 1.35],
                        lookat_offset=debug_camera_lookat_offset or [0.0, 0.0, 0.35],
                        camera_quaternion=[0.5, 0.5, -0.5, -0.5],
                        up_axis="z",
                        fov=75.0,
                        skip_erosion=True,
                    )
                )
        if not args.randomize_camera:
            disable_camera_randomization(cfg.camera_config)
        ensure_head_camera_exists(cfg.camera_config)
        if enable_depth_for_camera(cfg.camera_config, "head_camera"):
            log.info("Enabled depth for head_camera in this run.")
        else:
            log.warning("head_camera not found; cannot enable head depth for this run.")
    elif args.robot == "rum":
        cfg.robot_config = FloatingRUMRobotConfig()
        # RUM camera config may come from base config; still enforce camera naming compatibility.
        if cfg.camera_config is not None:
            ensure_head_camera_exists(cfg.camera_config)
    else:
        raise ValueError(f"Unsupported robot: {args.robot}")

    # Keep planner config available for fallback/debugging.
    cfg.policy_config = AStarNavToObjPolicyConfig()
    cfg.policy_dt_ms = args.policy_dt_ms
    # For ROS runtime/debugging, stop after the first sampled episode regardless of success.
    cfg.filter_for_successful_trajectories = False
    return cfg


def build_output_dir(run_name_prefix: str) -> Path:
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name_prefix:
        run_name = f"{run_name_prefix}_{run_name}"
    return ASSETS_DIR / "datagen" / "nav_to_obj_ros_sim_v1" / run_name


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value_lower = str(value).strip().lower()
    if value_lower in {"true", "1", "yes", "y"}:
        return True
    if value_lower in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nav-to-object simulation with ROS bridge policy."
    )
    parser.add_argument("--viewer", action="store_true", help="Enable passive viewer")
    parser.add_argument("--robot", type=str, default="droid", choices=["droid", "rby1", "rum"])
    parser.add_argument("--scene_dataset", type=str, default="procthor-10k")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--house_ind", type=int, default=0)
    parser.add_argument(
        "--house_inds",
        type=lambda value: [int(item.strip()) for item in value.split(",") if item.strip()],
        default=None,
        help="Comma-separated houses processed sequentially in one simulator/ROS process.",
    )
    parser.add_argument("--samples_per_house", type=int, default=1)
    parser.add_argument("--target_types", type=str, default=None)
    parser.add_argument(
        "--exploration_only",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Do not filter the scene by a requested target type; use any navigable object only to initialize the scene.",
    )
    parser.add_argument("--task_horizon", type=int, default=300)
    parser.add_argument("--policy_dt_ms", type=float, default=200.0)
    parser.add_argument(
        "--ctrl_dt_ms",
        type=float,
        default=10.0,
        help="Low-level control update period in milliseconds.",
    )
    parser.add_argument("--sim_dt_ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--randomize_scene", action="store_true")
    parser.add_argument(
        "--randomize_camera",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Keep camera pose/FOV randomization enabled. Disabled by default for ROS mapping debug.",
    )
    parser.add_argument("--run_name_prefix", type=str, default="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Explicit run output directory; overrides the timestamped default.",
    )
    parser.add_argument(
        "--disable_task_sensors",
        action="store_true",
        default=True,
        help="Disable task sensor suite to avoid heavy reset-time rendering in ROS bridge mode.",
    )
    parser.add_argument(
        "--enable_task_sensors",
        dest="disable_task_sensors",
        action="store_false",
        help="Enable task sensor suite (may be significantly slower at reset).",
    )

    parser.add_argument("--observation_topic", type=str, default="/molmo_spaces/head_camera/image")
    parser.add_argument("--depth_topic", type=str, default="/molmo_spaces/head_camera/depth")
    parser.add_argument("--action_topic", type=str, default="/molmo_spaces/action")
    parser.add_argument("--pointcloud_topic", type=str, default="/registered_scan")
    parser.add_argument("--camera_info_topic", type=str, default="/molmo_spaces/head_camera/camera_info")
    parser.add_argument("--extra_image_topic", type=str, default="/molmo_spaces/debug_front_camera/image")
    parser.add_argument("--debug_front_camera_name", type=str, default="debug_front_camera")
    parser.add_argument("--publish_debug_front_camera", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--include_wrist_cameras",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Render wrist RGB/depth streams; disabled by default because ROS exploration does not use them.",
    )
    parser.add_argument("--debug_snapshot_path", type=str, default="")
    parser.add_argument("--debug_snapshot_camera_name", type=str, default="debug_front_camera")
    parser.add_argument("--fixed_robot_xyyaw", type=str, default="")
    parser.add_argument("--fixed_debug_camera_pos", type=str, default="")
    parser.add_argument("--fixed_debug_camera_target", type=str, default="")
    parser.add_argument("--debug_front_camera_offset", type=str, default="-1.4,0.0,1.35")
    parser.add_argument("--debug_front_camera_lookat_offset", type=str, default="0.0,0.0,0.35")
    parser.add_argument("--depth_camera_name", type=str, default="head_camera")
    parser.add_argument("--pointcloud_frame_id", type=str, default="tf_frame_lidar")
    parser.add_argument("--pointcloud_stride", type=int, default=2)
    parser.add_argument(
        "--pointcloud_roll_correction_deg",
        type=float,
        default=0.0,
        help="Roll correction (deg) around robot forward axis for pointcloud leveling.",
    )
    parser.add_argument(
        "--lidar_calib_x_m",
        type=float,
        default=0.0,
        help="Extra base->lidar TF calibration x offset in lidar frame (meters).",
    )
    parser.add_argument(
        "--lidar_calib_y_m",
        type=float,
        default=0.0,
        help="Extra base->lidar TF calibration y offset in lidar frame (meters).",
    )
    parser.add_argument(
        "--lidar_calib_z_m",
        type=float,
        default=0.0,
        help="Extra base->lidar TF calibration z offset in lidar frame (meters).",
    )
    parser.add_argument(
        "--lidar_calib_roll_deg",
        type=float,
        default=0.0,
        help="Extra base->lidar TF roll calibration in lidar frame (degrees).",
    )
    parser.add_argument(
        "--lidar_calib_pitch_deg",
        type=float,
        default=0.0,
        help="Extra base->lidar TF pitch calibration in lidar frame (degrees).",
    )
    parser.add_argument(
        "--lidar_calib_yaw_deg",
        type=float,
        default=0.0,
        help="Extra base->lidar TF yaw calibration in lidar frame (degrees).",
    )
    parser.add_argument(
        "--allow_static_lidar_tf_fallback",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Allow fixed base->lidar TF fallback when live camera sensor_param is missing.",
    )
    parser.add_argument("--depth_fov_deg", type=float, default=90.0)
    parser.add_argument("--depth_min_m", type=float, default=0.1)
    parser.add_argument("--depth_max_m", type=float, default=30.0)
    parser.add_argument(
        "--action_timeout_s",
        type=float,
        default=0.0,
        help="ROS action wait timeout; <=0 blocks until a fresh action is available.",
    )
    parser.add_argument(
        "--scene_timeout_s",
        type=float,
        default=0.0,
        help="Maximum wall-clock seconds per scene attempt; 0 disables scene timeout.",
    )
    parser.add_argument(
        "--max_scene_attempts",
        type=int,
        default=1,
        help="Maximum attempts after scene execution errors or timeouts.",
    )
    parser.add_argument(
        "--max_consecutive_action_timeouts",
        type=int,
        default=12,
        help="Abort an attempt after this many consecutive action wait timeouts; 0 disables.",
    )
    parser.add_argument(
        "--blocking_observation_republish_period_s",
        type=float,
        default=0.25,
        help="Republish the current observation while blocking so ROS can initialize and plan.",
    )
    parser.add_argument(
        "--blocking_republish_pointcloud",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Also republish the same mapping cloud while blocked; disabled to avoid duplicate OCC integration.",
    )
    parser.add_argument(
        "--map_warmup_skip_frames",
        type=int,
        default=0,
        help="Observation frames skipped before mapping publish; keep 0 for blocking startup.",
    )
    parser.add_argument(
        "--require_fresh_cmd_vel",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Only execute cmd_vel received after the current observation was published.",
    )
    parser.add_argument(
        "--require_move_base_active_for_cmd_vel",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Do not advance simulation on startup/no-goal zero velocity commands.",
    )
    parser.add_argument(
        "--cmd_vel_linear_gain",
        type=float,
        default=3.0,
        help="Linear velocity gain applied to incoming /cmd_vel_stamped before stepping base.",
    )
    parser.add_argument(
        "--initial_arm_qpos",
        type=str,
        default="0.28,0.0,0.0,-0.64,0.39,-0.26,-0.04",
        help="Comma-separated 7-DoF RBY1 left/right arm initial and hold qpos for ROS navigation runs.",
    )
    parser.add_argument("--initial_left_arm_qpos", type=str, default="")
    parser.add_argument("--initial_right_arm_qpos", type=str, default="")
    parser.add_argument(
        "--immediate_noop_after_publish",
        action="store_true",
        help="Publish observations then immediately return noop action (no ROS action wait).",
    )
    parser.add_argument(
        "--timing_log_every_n_frames",
        type=int,
        default=20,
        help="Log averaged per-frame ROS bridge timing every N frames (0 disables).",
    )
    parser.add_argument(
        "--sim_timing_log_every_n_steps",
        type=int,
        default=20,
        help="Log policy wait, physics, sensor, and full task timing every N simulation steps.",
    )
    parser.add_argument(
        "--step_log_every_n_steps",
        type=int,
        default=10,
        help="Print action details every N steps after the first five (0 disables).",
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="ros_bridge",
        choices=["ros_bridge", "left_arm_debug"],
        help="Policy mode: ROS bridge or keyboard left-arm debug.",
    )
    parser.add_argument(
        "--left_arm_joint_delta",
        type=float,
        default=0.05,
        help="Joint delta for left-arm keyboard debug policy.",
    )
    parser.add_argument(
        "--debug_loop_episodes",
        type=int,
        default=1,
        help="For left_arm_debug mode: minimum episodes to run on the same house/scene.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_scene_attempts < 1:
        raise ValueError("--max_scene_attempts must be at least 1")
    if args.scene_timeout_s < 0.0:
        raise ValueError("--scene_timeout_s cannot be negative")
    if args.max_consecutive_action_timeouts < 0:
        raise ValueError("--max_consecutive_action_timeouts cannot be negative")
    exp_config = build_nav_config(args)
    exp_config.output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else build_output_dir(args.run_name_prefix)
    )
    exp_config.save_config()
    log_path = configure_run_file_logging(exp_config.output_dir)
    log.info("Run log file: %s", log_path)
    log.info(
        "Starting nav ROS sim: dataset=%s split=%s house_ind=%s target_types=%s "
        "policy_dt_ms=%.1f ctrl_dt_ms=%.1f sim_dt_ms=%.1f action_timeout_s=%.3f cameras=%s",
        args.scene_dataset,
        args.data_split,
        args.house_ind,
        args.target_types or "<any>",
        args.policy_dt_ms,
        args.ctrl_dt_ms,
        args.sim_dt_ms,
        args.action_timeout_s,
        ",".join(camera.name for camera in exp_config.camera_config.cameras),
    )

    if args.policy_mode == "left_arm_debug":
        policy = LeftArmKeyboardDebugPolicy(
            config=exp_config,
            task=None,
            joint_delta=args.left_arm_joint_delta,
        )
    else:
        effective_action_timeout_s = resolve_action_timeout_s(
            args.action_timeout_s, args.scene_timeout_s
        )
        policy = RosBridgePolicy(
            config=exp_config,
            task=None,
            observation_topic=args.observation_topic,
            action_topic=args.action_topic,
            pointcloud_topic=args.pointcloud_topic,
            camera_info_topic=args.camera_info_topic,
            depth_topic=args.depth_topic,
            action_timeout_s=effective_action_timeout_s,
            blocking_observation_republish_period_s=args.blocking_observation_republish_period_s,
            blocking_republish_pointcloud=args.blocking_republish_pointcloud,
            depth_camera_name=args.depth_camera_name,
            pointcloud_frame_id=args.pointcloud_frame_id,
            pointcloud_stride=args.pointcloud_stride,
            pointcloud_roll_correction_deg=args.pointcloud_roll_correction_deg,
            lidar_calib_x_m=args.lidar_calib_x_m,
            lidar_calib_y_m=args.lidar_calib_y_m,
            lidar_calib_z_m=args.lidar_calib_z_m,
            lidar_calib_roll_deg=args.lidar_calib_roll_deg,
            lidar_calib_pitch_deg=args.lidar_calib_pitch_deg,
            lidar_calib_yaw_deg=args.lidar_calib_yaw_deg,
            allow_static_lidar_tf_fallback=args.allow_static_lidar_tf_fallback,
            depth_fov_deg=args.depth_fov_deg,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            cmd_vel_control_dt_s=args.policy_dt_ms / 1000.0,
            cmd_vel_linear_gain=args.cmd_vel_linear_gain,
            require_fresh_cmd_vel=args.require_fresh_cmd_vel,
            require_move_base_active_for_cmd_vel=args.require_move_base_active_for_cmd_vel,
            map_warmup_skip_frames=args.map_warmup_skip_frames,
            immediate_noop_after_publish=args.immediate_noop_after_publish,
            timing_log_every_n_frames=args.timing_log_every_n_frames,
            extra_image_topic=args.extra_image_topic,
            extra_image_camera_name=args.debug_front_camera_name,
        )
        arm_qpos = parse_qpos_csv(args.initial_arm_qpos)
        left_arm_qpos = parse_qpos_csv(args.initial_left_arm_qpos)
        right_arm_qpos = parse_qpos_csv(args.initial_right_arm_qpos)
        if left_arm_qpos is None:
            left_arm_qpos = arm_qpos
        if right_arm_qpos is None:
            right_arm_qpos = arm_qpos
        if left_arm_qpos is not None:
            policy.default_left_arm_qpos = left_arm_qpos.copy()
        if right_arm_qpos is not None:
            policy.default_right_arm_qpos = right_arm_qpos.copy()
        policy.scene_timeout_s = args.scene_timeout_s
        policy.max_consecutive_action_timeouts = args.max_consecutive_action_timeouts
    policy.sim_timing_log_every_n_steps = args.sim_timing_log_every_n_steps
    policy.step_log_every_n_steps = args.step_log_every_n_steps
    policy.debug_snapshot_path = args.debug_snapshot_path
    policy.debug_snapshot_camera_name = args.debug_snapshot_camera_name
    policy.debug_snapshot_saved = False
    print("Creating runner ...")
    runner = NavRosRolloutRunner(exp_config)
    print("Starting runner.run() ...")
    try:
        print("Running runner.run() ...")
        runner.run(preloaded_policy=policy)
    finally:
        print("Closing policy ...")
        if hasattr(policy, "close"):
            policy.close()


if __name__ == "__main__":
    main()
