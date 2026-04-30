import argparse
import datetime
import logging
from pathlib import Path
import time

import mujoco

from robot_conversion_patches import patch_droid_config_for_rum

from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    RBY1GoProD455CameraSystem,
)
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig, FrankaRobotConfig, RBY1Config
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.profiler_utils import Profiler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        observation, _info = task.reset()
        log.info("task.reset() completed.")
        print(f"Observation: {observation}", flush=True)
        if viewer is not None:
            viewer.sync()

        policy.task = task
        policy.reset()

        try:
            task.env.current_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            log.warning("Sleep flag not set.")

        success = False
        while not task.is_done():
            if shutdown_event is not None and shutdown_event.is_set():
                return False

            action_cmd = policy.get_action(observation)
            print(f"Action command: {action_cmd}", flush=True)
            if action_cmd is None:
                break

            observation, reward, terminal, truncated, infos = task.step(action_cmd)
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
    cfg = NavToObjBaseConfig()
    cfg.seed = args.seed
    cfg.task_type = "nav_to_obj"
    cfg.scene_dataset = args.scene_dataset
    cfg.data_split = args.data_split
    cfg.task_horizon = args.task_horizon
    cfg.num_workers = 1
    cfg.use_passive_viewer = args.viewer

    cfg.task_sampler_config.samples_per_house = args.samples_per_house
    cfg.task_sampler_config.house_inds = [args.house_ind]
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

    if args.target_types:
        cfg.task_sampler_config.pickup_types = [s.strip() for s in args.target_types.split(",")]

    if args.robot == "droid":
        cfg.robot_config = FrankaRobotConfig()
        cfg.camera_config = FrankaDroidCameraSystem()
        cfg.camera_config.img_resolution = (320, 240)
        ensure_head_camera_exists(cfg.camera_config)
    elif args.robot == "rby1":
        cfg.robot_config = RBY1Config()
        cfg.camera_config = RBY1GoProD455CameraSystem()
        ensure_head_camera_exists(cfg.camera_config)
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
    return cfg


def build_output_dir(run_name_prefix: str) -> Path:
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name_prefix:
        run_name = f"{run_name_prefix}_{run_name}"
    return ASSETS_DIR / "datagen" / "nav_to_obj_ros_sim_v1" / run_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nav-to-object simulation with ROS bridge policy."
    )
    parser.add_argument("--viewer", action="store_true", help="Enable passive viewer")
    parser.add_argument("--robot", type=str, default="droid", choices=["droid", "rby1", "rum"])
    parser.add_argument("--scene_dataset", type=str, default="procthor-10k")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--house_ind", type=int, default=0)
    parser.add_argument("--samples_per_house", type=int, default=1)
    parser.add_argument("--target_types", type=str, default=None)
    parser.add_argument("--task_horizon", type=int, default=300)
    parser.add_argument("--policy_dt_ms", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--randomize_scene", action="store_true")
    parser.add_argument("--run_name_prefix", type=str, default="")
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

    parser.add_argument("--observation_topic", type=str, default="/molmo_spaces/observation")
    parser.add_argument("--action_topic", type=str, default="/molmo_spaces/action")
    parser.add_argument("--action_timeout_s", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_config = build_nav_config(args)
    exp_config.output_dir = build_output_dir(args.run_name_prefix)
    exp_config.save_config()

    policy = RosBridgePolicy(
        config=exp_config,
        task=None,
        observation_topic=args.observation_topic,
        action_topic=args.action_topic,
        action_timeout_s=args.action_timeout_s,
    )
    print("Creating runner ...")
    runner = NavRosRolloutRunner(exp_config)
    print("Starting runner.run() ...")
    try:
        print("Running runner.run() ...")
        runner.run(preloaded_policy=policy)
    finally:
        print("Closing policy ...")
        policy.close()


if __name__ == "__main__":
    main()
