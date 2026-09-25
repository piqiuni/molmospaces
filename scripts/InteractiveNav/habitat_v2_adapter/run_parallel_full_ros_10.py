#!/usr/bin/env python3
"""Run ten ObjectNav-v2 scenes with isolated full ROS stacks and shared models."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen

import yaml

from aggregate_parallel_results import aggregate
from run_parallel_yoloe_10 import SCENES


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ldl/outputs/habitat_objectnav_v2_m2/parallel10_full_ros_yoloe"),
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-episode-seconds", type=int, default=500)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--ros-master-port-base", type=int, default=13600)
    parser.add_argument("--bridge-port-base", type=int, default=12300)
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="repeatable focused scene selection; defaults to the canonical ten-scene slice",
    )
    parser.add_argument(
        "--record-original-ros-video",
        action="store_true",
        help="bind one original ROS six-panel recorder to each isolated worker",
    )
    return parser.parse_args()


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_health(port: int, timeout_s: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last_error = "not started"
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
            if payload.get("ready") and payload.get("module3_enabled") is False:
                return payload
            last_error = f"invalid health payload: {payload}"
        except Exception as exc:  # startup polling records the final error below
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"full ROS bridge {port} did not become ready: {last_error}")


def main() -> int:
    args = _args()
    scenes = tuple(args.scene_id) if args.scene_id else tuple(SCENES)
    adapter_dir = Path(__file__).resolve().parent
    interactive_dir = adapter_dir.parent
    project_root = interactive_dir.parent.parent
    source_profile = interactive_dir / "configs/habitat_objectnav_v2/full_ros_graph_m2_navigation_only.yaml"
    base_profile = yaml.safe_load(source_profile.read_text(encoding="utf-8"))
    evaluator = adapter_dir / "evaluate.py"
    start_stack = adapter_dir / "start_ros_full_stack.sh"
    habitat_lab = Path("/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab")
    habitat_python = Path("/home/ldl/conda_envs/habitat-challenge-2023/bin/python")

    ports = [
        port
        for index in range(len(scenes))
        for port in (
            args.ros_master_port_base + index,
            args.bridge_port_base + index,
            args.bridge_port_base + index + 10,
        )
    ]
    occupied = [port for port in ports if not _port_is_free(port)]
    if occupied:
        raise RuntimeError(f"parallel full-ROS ports already in use: {occupied}")

    run_stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_root = args.output_root / run_stamp
    run_root.mkdir(parents=True, exist_ok=False)
    worker_root = run_root / "workers"
    worker_root.mkdir()
    runtime_root = Path("/home/ldl/tmp/habitat-objectnav-full-ros-parallel10") / run_stamp
    cache_root = Path("/home/ldl/.cache/habitat-objectnav-full-ros-parallel10") / run_stamp
    runtime_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "workers": len(scenes),
        "scenes": list(scenes),
        "max_steps": args.max_steps,
        "max_episode_seconds": args.max_episode_seconds,
        "gpu_id": args.gpu_id,
        "shared_m2_endpoint": base_profile["modules"]["module2"]["endpoint"],
        "shared_yoloe_gateway": None,
        "dedicated_yolo_per_worker": True,
        "yoloe_replicas_per_worker": 1,
        "isolated_ros_state_per_worker": True,
        "module3_enabled": False,
        "original_ros_six_panel_video": bool(args.record_original_ros_video),
    }
    (run_root / "parallel_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    stacks: dict[str, tuple[subprocess.Popen[str], object]] = {}
    evaluators: dict[str, tuple[subprocess.Popen[str], object]] = {}
    exit_codes: dict[str, int] = {}
    try:
        for index, scene in enumerate(scenes):
            worker_dir = worker_root / scene
            worker_dir.mkdir()
            master_port = args.ros_master_port_base + index
            bridge_port = args.bridge_port_base + index
            worker_runtime = runtime_root / scene
            worker_cache = cache_root / scene
            worker_runtime.mkdir(parents=True, exist_ok=True)
            worker_cache.mkdir(parents=True, exist_ok=True)

            profile = deepcopy(base_profile)
            profile["profile"] = f"{base_profile['profile']}-worker-{index:02d}"
            profile["modules"]["module1"]["sidecar"]["endpoint"] = (
                f"http://127.0.0.1:{bridge_port}"
            )
            profile_path = worker_dir / "adapter_config.yaml"
            profile_path.write_text(
                yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )

            stack_env = os.environ.copy()
            stack_env.update(
                {
                    "ROS_MASTER_PORT": str(master_port),
                    "FULL_STACK_BRIDGE_PORT": str(bridge_port),
                    "FULL_STACK_YOLO_PORT": str(bridge_port + 10),
                    "HABITAT_FULL_STACK_RUNTIME_ROOT": str(worker_runtime / "ros"),
                }
            )
            if args.record_original_ros_video:
                stack_env["HABITAT_FULL_STACK_RECORDER_OUTPUT_DIR"] = str(
                    worker_dir / "original_ros_recorder"
                )
            stack_log = (worker_dir / "ros_stack.log").open("w", encoding="utf-8")
            stack = subprocess.Popen(
                ["bash", str(start_stack)],
                cwd=str(project_root),
                env=stack_env,
                stdout=stack_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stacks[scene] = (stack, stack_log)

        for index, scene in enumerate(scenes):
            health = _wait_health(args.bridge_port_base + index)
            print(f"ROS ready {scene}: {health}", flush=True)

        for index, scene in enumerate(scenes):
            worker_dir = worker_root / scene
            worker_runtime = runtime_root / scene
            worker_cache = cache_root / scene
            env = os.environ.copy()
            env.update(
                {
                    "EGL_PLATFORM": "surfaceless",
                    "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
                    "TMPDIR": str(worker_runtime / "habitat-tmp"),
                    "XDG_CACHE_HOME": str(worker_cache),
                    "PYTHONPYCACHEPREFIX": str(worker_cache / "pycache"),
                    "PYTHONPATH": f"{interactive_dir}:{habitat_lab}",
                }
            )
            Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            command = [
                str(habitat_python),
                str(evaluator),
                "--adapter-config", str(worker_dir / "adapter_config.yaml"),
                "--output-dir", str(worker_dir),
                "--scene-id", scene,
                "--scene-count", "1",
                "--episodes-per-scene", "1",
                "--max-steps", str(args.max_steps),
                "--max-episode-seconds", str(args.max_episode_seconds),
                "--gpu-id", "0",
                "--public-trace",
                "--posthoc-step-metrics",
                "--posthoc-topdown-map",
                "--allow-no-mllm-success",
            ]
            eval_log = (worker_dir / "worker.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                env=env,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            evaluators[scene] = (process, eval_log)
            print(f"started {scene}: pid={process.pid}", flush=True)

        while len(exit_codes) < len(evaluators):
            for scene, (process, _) in evaluators.items():
                if scene not in exit_codes and process.poll() is not None:
                    exit_codes[scene] = int(process.returncode)
                    print(f"finished {scene}: exit={process.returncode}", flush=True)
            if len(exit_codes) < len(evaluators):
                time.sleep(2.0)
    except KeyboardInterrupt:
        for process, _ in evaluators.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        raise
    finally:
        for process, _ in stacks.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process, stream in evaluators.values():
            if process.poll() is None:
                process.wait(timeout=20)
            stream.close()
        for process, stream in stacks.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            stream.close()

    (run_root / "worker_exit_codes.json").write_text(
        json.dumps(exit_codes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = aggregate(run_root)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
