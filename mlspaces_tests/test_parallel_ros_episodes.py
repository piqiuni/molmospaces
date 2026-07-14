import argparse
from pathlib import Path
import tempfile
import unittest

from scripts.InteractiveNav.run_parallel_ros_episodes import Worker, split_round_robin


class ParallelRosEpisodesTest(unittest.TestCase):
    def test_round_robin_shards_are_deterministic(self):
        self.assertEqual(split_round_robin([4, 7, 10, 12, 15, 18], 3), [[4, 12], [7, 15], [10, 18]])

    def test_worker_builds_isolated_environment_and_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                base_master_port=11411,
                output_dir=Path(temp_dir),
                gpu_ids=["1"],
                scene_dataset="procthor-10k",
                data_split="train",
                robot="rby1",
                target_types="Chair",
                task_horizon=50,
                scene_timeout_s=600.0,
                max_scene_attempts=2,
                max_consecutive_action_timeouts=12,
                samples_per_house=1,
                exploration_only=True,
                start_explore_py=True,
                start_semantic_mapping=False,
                resource_interval_s=2.0,
                master_timeout_s=30.0,
                shutdown_grace_s=15.0,
                worker_timeout_s=0.0,
                setup_file=Path("/tmp/setup.zsh"),
                ros_hostname="127.0.0.1",
                sim_extra_args="--step_log_every_n_steps 0",
                record_debug=True,
                recorder_script=Path("/tmp/record_explore_debug.py"),
                recorder_extra_args="--first-person-video-fps 10",
                recorder_shutdown_grace_s=120.0,
            )
            worker = Worker(1, [7, 15], args)
            env = worker.environment()
            command = worker.launch_command()

            self.assertEqual(env["ROS_MASTER_URI"], "http://127.0.0.1:11412")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(env["ROS_HOME"], "/tmp/molmospaces_ros/worker_001")
            self.assertIn("house_inds:=7,15", command)
            self.assertIn("scene_timeout_s:=600.0", command)
            self.assertIn("max_scene_attempts:=2", command)
            self.assertIn(f"output_dir:={worker.worker_dir / 'sim'}", command)
            self.assertIn(
                "sim_extra_args:=--samples_per_house 1 --step_log_every_n_steps 0",
                command,
            )
            self.assertIn("publish_debug_front_camera:=true", command)
            recorder_command = worker.recorder_command()
            self.assertIn("--first-person-video-with-map", recorder_command)
            self.assertIn("--overlay-contact-sheet-columns", recorder_command)
            self.assertEqual(recorder_command[-2:], ["--first-person-video-fps", "10"])


if __name__ == "__main__":
    unittest.main()
