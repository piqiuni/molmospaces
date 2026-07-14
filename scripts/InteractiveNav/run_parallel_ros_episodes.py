#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETUP = REPO_ROOT / "Interactive-Nav-SG-nav" / "devel" / "setup.zsh"
DEFAULT_RECORDER = REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "explore_py_pkg" / "scripts" / "record_explore_debug.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated MolmoSpaces ROS workers on separate ROS masters.")
    parser.add_argument("--house-inds", nargs="+", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-master-port", type=int, default=11411)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs="*", default=[])
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--target-types", default="Chair")
    parser.add_argument("--task-horizon", type=int, default=500)
    parser.add_argument("--scene-timeout-s", type=float, default=0.0)
    parser.add_argument("--max-scene-attempts", type=int, default=1)
    parser.add_argument("--max-consecutive-action-timeouts", type=int, default=12)
    parser.add_argument("--samples-per-house", type=int, default=1)
    parser.add_argument("--exploration-only", action="store_true")
    parser.add_argument("--start-explore-py", action="store_true")
    parser.add_argument("--start-semantic-mapping", action="store_true")
    parser.add_argument("--resource-interval-s", type=float, default=2.0)
    parser.add_argument("--master-timeout-s", type=float, default=30.0)
    parser.add_argument("--shutdown-grace-s", type=float, default=15.0)
    parser.add_argument("--worker-timeout-s", type=float, default=0.0)
    parser.add_argument("--setup-file", type=Path, default=DEFAULT_SETUP)
    parser.add_argument("--ros-hostname", default="127.0.0.1")
    parser.add_argument("--sim-extra-args", default="")
    parser.add_argument("--record-debug", action="store_true")
    parser.add_argument("--recorder-script", type=Path, default=DEFAULT_RECORDER)
    parser.add_argument("--recorder-extra-args", default="")
    parser.add_argument("--recorder-shutdown-grace-s", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_round_robin(items: list[int], workers: int) -> list[list[int]]:
    return [items[index::workers] for index in range(workers)]


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_host_sample() -> dict[str, Any]:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0])
    load1, load5, load15 = os.getloadavg()
    return {
        "timestamp": utc_now(),
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "mem_total_mb": meminfo["MemTotal"] / 1024.0,
        "mem_available_mb": meminfo["MemAvailable"] / 1024.0,
        "swap_used_mb": (meminfo["SwapTotal"] - meminfo["SwapFree"]) / 1024.0,
    }


def read_gpu_samples() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue
        rows.append({
            "index": int(row[0]),
            "name": row[1],
            "utilization_gpu_pct": float(row[2]),
            "memory_used_mb": float(row[3]),
            "memory_total_mb": float(row[4]),
            "power_draw_w": None if row[5] == "[N/A]" else float(row[5]),
        })
    return rows


def pids_for_master(master_uri: str) -> list[int]:
    marker = f"ROS_MASTER_URI={master_uri}".encode()
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if marker in (entry / "environ").read_bytes().split(b"\0"):
                pids.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return pids


def process_usage(pids: list[int]) -> dict[str, Any]:
    rss_kb = 0
    cpu_ticks = 0
    live_pids = []
    for pid in pids:
        try:
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
            stat = Path(f"/proc/{pid}/stat").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        live_pids.append(pid)
        for line in status:
            if line.startswith("VmRSS:"):
                rss_kb += int(line.split()[1])
                break
        cpu_ticks += int(stat[13]) + int(stat[14])
    return {"process_count": len(live_pids), "rss_mb": rss_kb / 1024.0, "cpu_ticks": cpu_ticks}


def terminate_group(process: subprocess.Popen | None, grace_s: float) -> None:
    if process is None or process.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGINT, grace_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


class Worker:
    def __init__(self, worker_id: int, houses: list[int], args: argparse.Namespace):
        self.worker_id = worker_id
        self.houses = houses
        self.args = args
        self.port = args.base_master_port + worker_id
        self.master_uri = f"http://127.0.0.1:{self.port}"
        self.worker_dir = args.output_dir / f"worker_{worker_id:03d}"
        self.ros_home = Path("/tmp/molmospaces_ros") / f"worker_{worker_id:03d}"
        self.gpu_id = args.gpu_ids[worker_id % len(args.gpu_ids)] if args.gpu_ids else None
        self.roscore: subprocess.Popen | None = None
        self.roslaunch: subprocess.Popen | None = None
        self.recorder: subprocess.Popen | None = None
        self.status: dict[str, Any] = {}
        self.stop_monitor = threading.Event()
        self.monitor_thread: threading.Thread | None = None

    def log_event(self, message: str) -> None:
        with (self.worker_dir / "worker.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_now()} {message}\n")

    def write_episode_events(self, status: str, **extra: Any) -> None:
        path = self.worker_dir / "episodes.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            for house in self.houses:
                event = {
                    "worker_id": self.worker_id,
                    "episode_id": house,
                    "house_id": house,
                    "ros_master_uri": self.master_uri,
                    "status": status,
                    "timestamp": utc_now(),
                    "output_dir": str(self.worker_dir / "sim" / f"house_{house}"),
                    **extra,
                }
                stream.write(json.dumps(event) + "\n")

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "ROS_MASTER_URI": self.master_uri,
            "ROS_HOME": str(self.ros_home),
            "ROS_LOG_DIR": str(self.ros_home / "log"),
            "ROS_HOSTNAME": self.args.ros_hostname,
            "PYTHONUNBUFFERED": "1",
        })
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        return env

    def launch_command(self) -> list[str]:
        house_csv = ",".join(str(house) for house in self.houses)
        command = [
            "roslaunch", "nav_pkg", "molmospaces_nav_system.launch",
            f"robot:={self.args.robot}",
            f"scene_dataset:={self.args.scene_dataset}",
            f"data_split:={self.args.data_split}",
            f"house_ind:={self.houses[0]}",
            f"house_inds:={house_csv}",
            f"target_types:={self.args.target_types}",
            f"task_horizon:={self.args.task_horizon}",
            f"scene_timeout_s:={self.args.scene_timeout_s}",
            f"max_scene_attempts:={self.args.max_scene_attempts}",
            f"max_consecutive_action_timeouts:={self.args.max_consecutive_action_timeouts}",
            f"exploration_only:={'true' if self.args.exploration_only else 'false'}",
            f"start_explore_py:={'true' if self.args.start_explore_py else 'false'}",
            f"start_semantic_mapping:={'true' if self.args.start_semantic_mapping else 'false'}",
            f"publish_debug_front_camera:={'true' if self.args.record_debug else 'false'}",
            f"output_dir:={self.worker_dir / 'sim'}",
        ]
        sim_args = [f"--samples_per_house {self.args.samples_per_house}"]
        if self.args.sim_extra_args:
            sim_args.append(self.args.sim_extra_args)
        command.append(f"sim_extra_args:={' '.join(sim_args)}")
        return command

    def recorder_command(self) -> list[str]:
        command = [
            sys.executable,
            str(self.args.recorder_script),
            "--output-dir",
            str(self.worker_dir / "debug"),
            "--stall-snapshot-sec",
            "30",
            "--stall-snapshot-distance-m",
            "0.15",
            "--stall-snapshot-cooldown-sec",
            "45",
            "--external-image-topic",
            "/molmo_spaces/debug_front_camera/image",
            "--external-video",
            "--first-person-video-with-map",
            "--first-person-video-fps",
            "15",
            "--overlay-contact-sheet-columns",
            "4",
            "--runtime-video-encode",
            "--async-artifact-writes",
        ]
        if self.args.recorder_extra_args:
            command.extend(shlex.split(self.args.recorder_extra_args))
        return command

    def shell_command(self, command: list[str]) -> list[str]:
        quoted = shlex.join(command)
        setup = shlex.quote(str(self.args.setup_file))
        return ["/bin/zsh", "-lc", f"source {setup} && exec {quoted}"]

    def wait_for_master(self) -> None:
        deadline = time.monotonic() + self.args.master_timeout_s
        command = self.shell_command(["rosparam", "list"])
        while time.monotonic() < deadline:
            if self.roscore is not None and self.roscore.poll() is not None:
                raise RuntimeError(f"roscore exited with code {self.roscore.returncode}")
            result = subprocess.run(command, env=self.environment(), capture_output=True, timeout=5)
            if result.returncode == 0:
                return
            time.sleep(0.25)
        raise TimeoutError(f"ROS master did not become ready: {self.master_uri}")

    def monitor(self) -> None:
        path = self.worker_dir / "resources.jsonl"
        previous_ticks = None
        previous_time = None
        with path.open("a", encoding="utf-8") as stream:
            while not self.stop_monitor.wait(self.args.resource_interval_s):
                sample = read_host_sample()
                usage = process_usage(pids_for_master(self.master_uri))
                now = time.monotonic()
                cpu_pct = None
                if previous_ticks is not None and previous_time is not None:
                    elapsed = now - previous_time
                    tick_delta = usage["cpu_ticks"] - previous_ticks
                    if tick_delta >= 0:
                        cpu_pct = 100.0 * tick_delta / (os.sysconf("SC_CLK_TCK") * elapsed)
                previous_ticks = usage["cpu_ticks"]
                previous_time = now
                sample.update({
                    "worker_id": self.worker_id,
                    "ros_master_uri": self.master_uri,
                    "worker_cpu_pct": cpu_pct,
                    "worker_rss_mb": usage["rss_mb"],
                    "worker_process_count": usage["process_count"],
                    "gpus": read_gpu_samples(),
                })
                stream.write(json.dumps(sample) + "\n")
                stream.flush()

    def run(self) -> None:
        started = time.monotonic()
        self.status = {
            "worker_id": self.worker_id,
            "houses": self.houses,
            "ros_master_uri": self.master_uri,
            "ros_home": str(self.ros_home),
            "gpu_id": self.gpu_id,
            "output_dir": str(self.worker_dir),
            "status": "running",
            "start_time": utc_now(),
        }
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        (self.ros_home / "log").mkdir(parents=True, exist_ok=True)
        self.log_event(f"starting houses={self.houses} master={self.master_uri} gpu={self.gpu_id}")
        self.write_episode_events("running", start_time=self.status["start_time"])
        env = self.environment()
        try:
            with (self.worker_dir / "roscore.log").open("w", encoding="utf-8") as roscore_log:
                self.roscore = subprocess.Popen(
                    self.shell_command(["roscore", "-p", str(self.port)]),
                    env=env, stdout=roscore_log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                self.wait_for_master()
                self.log_event("roscore ready")
                self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
                self.monitor_thread.start()
                with (self.worker_dir / "roslaunch.log").open("w", encoding="utf-8") as launch_log:
                    self.roslaunch = subprocess.Popen(
                        self.shell_command(self.launch_command()),
                        env=env, stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    if self.args.record_debug:
                        recorder_log = (self.worker_dir / "recorder.log").open(
                            "w", encoding="utf-8"
                        )
                        self.recorder = subprocess.Popen(
                            self.shell_command(self.recorder_command()),
                            env=env,
                            stdout=recorder_log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        recorder_log.close()
                        self.log_event(f"debug recorder started pid={self.recorder.pid}")
                    try:
                        exit_code = self.roslaunch.wait(
                            timeout=None if self.args.worker_timeout_s <= 0 else self.args.worker_timeout_s
                        )
                    except subprocess.TimeoutExpired:
                        self.status["status"] = "timeout"
                        exit_code = None
                    self.status["exit_code"] = exit_code
                    if self.status["status"] == "running":
                        self.status["status"] = "success" if exit_code == 0 else "failed"
        except KeyboardInterrupt:
            self.status["status"] = "interrupted"
        except Exception as exc:
            self.status["status"] = "failed"
            self.status["error"] = repr(exc)
        finally:
            self.stop_monitor.set()
            terminate_group(self.recorder, self.args.recorder_shutdown_grace_s)
            terminate_group(self.roslaunch, self.args.shutdown_grace_s)
            terminate_group(self.roscore, self.args.shutdown_grace_s)
            if self.monitor_thread is not None:
                self.monitor_thread.join(timeout=5)
            self.status["end_time"] = utc_now()
            self.status["elapsed_sec"] = time.monotonic() - started
            self.status["recorder_exit_code"] = (
                self.recorder.returncode if self.recorder is not None else None
            )
            self.write_episode_events(
                self.status["status"],
                end_time=self.status["end_time"],
                elapsed_sec=self.status["elapsed_sec"],
                exit_code=self.status.get("exit_code"),
                error=self.status.get("error"),
            )
            self.log_event(
                f"finished status={self.status['status']} exit_code={self.status.get('exit_code')} "
                f"elapsed_sec={self.status['elapsed_sec']:.3f}"
            )
            (self.worker_dir / "status.json").write_text(json.dumps(self.status, indent=2) + "\n")


def prepare_output(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        elif not args.resume:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def summarize_resources(worker: Worker) -> dict[str, Any]:
    path = worker.worker_dir / "resources.jsonl"
    if not path.exists():
        return {"samples": 0}
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cpu = [sample["worker_cpu_pct"] for sample in samples if sample.get("worker_cpu_pct") is not None]
    rss = [sample["worker_rss_mb"] for sample in samples]
    gpu_used = [gpu["memory_used_mb"] for sample in samples for gpu in sample.get("gpus", [])]
    gpu_util = [gpu["utilization_gpu_pct"] for sample in samples for gpu in sample.get("gpus", [])]
    return {
        "samples": len(samples),
        "avg_worker_cpu_pct": sum(cpu) / len(cpu) if cpu else None,
        "peak_worker_cpu_pct": max(cpu) if cpu else None,
        "avg_worker_rss_mb": sum(rss) / len(rss) if rss else None,
        "peak_worker_rss_mb": max(rss) if rss else None,
        "peak_gpu_memory_used_mb_global": max(gpu_used) if gpu_used else None,
        "avg_gpu_utilization_pct_global": sum(gpu_util) / len(gpu_util) if gpu_util else None,
    }


def main() -> int:
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if args.num_workers > len(args.house_inds):
        args.num_workers = len(args.house_inds)
    if not args.setup_file.exists():
        raise FileNotFoundError(args.setup_file)
    if args.record_debug and not args.recorder_script.exists():
        raise FileNotFoundError(args.recorder_script)
    prepare_output(args)
    shards = split_round_robin(args.house_inds, args.num_workers)
    workers = [Worker(index, houses, args) for index, houses in enumerate(shards)]
    plan = {
        "created_at": utc_now(),
        "workers": [{
            "worker_id": worker.worker_id,
            "houses": worker.houses,
            "ros_master_uri": worker.master_uri,
            "ros_home": str(worker.ros_home),
            "gpu_id": worker.gpu_id,
            "output_dir": str(worker.worker_dir),
            "command": worker.launch_command(),
            "recorder_command": worker.recorder_command() if args.record_debug else None,
            "rviz_command": f"ROS_MASTER_URI={worker.master_uri} ROS_HOME={worker.ros_home} rviz",
        } for worker in workers],
    }
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    occupied = [worker.port for worker in workers if not port_available(worker.port)]
    if occupied:
        raise RuntimeError(f"ROS master ports are occupied: {occupied}")
    threads = [threading.Thread(target=worker.run, name=f"worker-{worker.worker_id}") for worker in workers]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        for worker in workers:
            worker.stop_monitor.set()
            terminate_group(worker.recorder, args.recorder_shutdown_grace_s)
            terminate_group(worker.roslaunch, args.shutdown_grace_s)
            terminate_group(worker.roscore, args.shutdown_grace_s)
        for thread in threads:
            thread.join(timeout=args.shutdown_grace_s + 6)
    summary = {
        "completed_at": utc_now(),
        "status": "success" if all(worker.status.get("status") == "success" for worker in workers) else "failed",
        "workers": [{**worker.status, "resources": summarize_resources(worker)} for worker in workers],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    signal.signal(signal.SIGTERM, previous_sigterm)
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
