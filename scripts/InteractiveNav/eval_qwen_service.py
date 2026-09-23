"""Batch-owned single vLLM service over all visible GPUs; no automatic restarts."""
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


DEFAULT_MAX_NUM_SEQS_PER_REPLICA = 16


def visible_devices(environment):
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [v.strip() for v in visible.split(",") if v.strip()]
        if not devices or any(v in ("-1", "none", "void") for v in devices):
            raise RuntimeError("No visible GPUs for Qwen")
        if all(device.isdigit() for device in devices):
            return devices
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        mapping = dict((uuid.strip(), index.strip()) for index, uuid in
                       (line.split(",", 1) for line in inventory.splitlines()))
        resolved = []
        for device in devices:
            if device.isdigit():
                resolved.append(device)
                continue
            matches = [index for uuid, index in mapping.items() if uuid.startswith(device)]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot resolve visible GPU {device!r} to one NVML index")
            resolved.append(matches[0])
        return resolved
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
    devices = result.split()
    if not devices:
        raise RuntimeError("No GPUs found for Qwen")
    return devices


def endpoint_for_port(port=8000):
    return f"http://127.0.0.1:{int(port)}/v1"


class QwenService:
    def __init__(self, output, environment, port=8000, target_concurrency=None):
        self.output = Path(output) / "qwen-service"
        self.environment = dict(environment)
        self.port = int(port)
        self.endpoint = endpoint_for_port(self.port)
        self.target_concurrency = int(target_concurrency) if target_concurrency is not None else None
        if self.target_concurrency is not None and self.target_concurrency < 1:
            raise ValueError("target_concurrency must be positive")
        self.devices = []
        self.tensor_parallel_size = None
        self.data_parallel_size = None
        self.max_num_seqs = None
        self.processes = []

    def check(self):
        for label, process in self.processes:
            if process.poll() is not None:
                raise RuntimeError(f"Qwen {label} exited ({process.returncode}); see {self.output}. No restart performed.")

    def _spawn(self, label, command, environment):
        with (self.output / f"{label}.launcher.log").open("ab") as log:
            process = subprocess.Popen(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append((label, process))

    def _ready(self, ports, stop):
        pending = set(ports)
        deadline = time.monotonic() + 600
        while pending:
            self.check()
            if stop.is_set():
                raise InterruptedError("Qwen startup interrupted")
            for port in list(pending):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                        if json.load(response).get("data"):
                            pending.remove(port)
                except (OSError, ValueError):
                    pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"Qwen readiness timed out: {sorted(pending)}")
            if pending:
                stop.wait(1)

    def start(self, stop):
        devices = visible_devices(self.environment)
        port = self.port
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
        self.output.mkdir(parents=True, exist_ok=True)
        env = dict(self.environment)
        env.pop("INTERACTIVE_NAV_EVAL_RUN_ID", None)
        env["QWEN36_LOG_DIR"] = str(self.output)
        env["QWEN36_RUNTIME_DIR"] = str(self.output / "runtime")
        env["QWEN36_GPU_IDS"] = ",".join(devices)
        env["QWEN36_PORT"] = str(port)
        root = Path(env.get("QWEN_ROOT", "/home/ldl/qwen36-fp8"))
        launcher = Path(env.get("QWEN_SINGLE_SCRIPT", str(root / "serve_qwen36_fp8_remote.zsh")))
        if not launcher.is_file():
            raise FileNotFoundError(f"Qwen single-instance launcher not found: {launcher}")
        tensor_parallel = int(env.get("QWEN36_TP_SIZE", "1"))
        if tensor_parallel < 1 or len(devices) % tensor_parallel:
            raise RuntimeError(
                f"QWEN36_TP_SIZE={tensor_parallel} is incompatible with {len(devices)} visible GPUs"
            )
        # Always cover every visible GPU. This deliberately overwrites a
        # stale DP value inherited from an older two-card service.
        data_parallel = len(devices) // tensor_parallel
        requested_max_num_seqs = env.get("QWEN36_MAX_NUM_SEQS")
        if requested_max_num_seqs is None:
            target_total_concurrency = max(
                DEFAULT_MAX_NUM_SEQS_PER_REPLICA * data_parallel,
                self.target_concurrency or 0,
            )
            max_num_seqs = max(1, math.ceil(target_total_concurrency / data_parallel))
        else:
            max_num_seqs = int(requested_max_num_seqs)
            if max_num_seqs < 1:
                raise ValueError("QWEN36_MAX_NUM_SEQS must be positive")
        env["QWEN36_TP_SIZE"] = str(tensor_parallel)
        env["QWEN36_DP_SIZE"] = str(data_parallel)
        env["QWEN36_MAX_NUM_SEQS"] = str(max_num_seqs)
        # One API process lets the vLLM internal DP scheduler see all ranks.
        env["QWEN36_API_SERVER_COUNT"] = "1"
        self.devices = devices
        self.tensor_parallel_size = tensor_parallel
        self.data_parallel_size = data_parallel
        self.max_num_seqs = max_num_seqs
        self._ready_ports = [port]
        self._spawn("vllm", ["bash", str(launcher)], env)
        self._ready([port], stop)
        (self.output / "deployment.json").write_text(json.dumps({
            "devices": devices,
            "port": port,
            "endpoint": self.endpoint,
            "endpoint_mode": "single_vllm_internal_scheduler",
            "tensor_parallel_size": tensor_parallel,
            "data_parallel_size": data_parallel,
            "max_num_seqs_per_replica": max_num_seqs,
            "total_max_num_seqs": max_num_seqs * data_parallel,
            "api_server_count": 1,
            "restart_policy": "never",
        }, indent=2))

    def close(self):
        for _, process in reversed(self.processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _, process in reversed(self.processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)


class QwenLBService(QwenService):
    """One single-GPU vLLM replica per visible GPU behind a TCP LB."""

    def __init__(self, output, environment, *, backend_ports=(8000, 8001), lb_port=8010,
                 target_concurrency=None):
        super().__init__(output, environment, port=lb_port, target_concurrency=target_concurrency)
        self.backend_ports = tuple(int(port) for port in backend_ports)

    def start(self, stop):
        devices = visible_devices(self.environment)
        if len(devices) < 2 or len(set(devices)) != len(devices):
            raise RuntimeError(f"Qwen LB requires at least two distinct GPUs, found {devices}")
        if len(self.backend_ports) != len(devices):
            raise RuntimeError(
                f"Qwen LB requires one backend port per GPU: {len(self.backend_ports)} ports for {len(devices)} GPUs"
            )
        if len(set((*self.backend_ports, self.port))) != len(devices) + 1:
            raise RuntimeError("Qwen LB and backend ports must be distinct")
        for port in (*self.backend_ports, self.port):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))

        root = Path(self.environment.get("QWEN_ROOT", "/home/ldl/qwen36-fp8"))
        launcher = Path(self.environment.get("QWEN_SINGLE_SCRIPT", str(root / "serve_qwen36_fp8_remote.zsh")))
        lb_script = Path(self.environment.get("QWEN_LB_SCRIPT", str(root / "bench/lb.py")))
        python_bin = root / "venv/bin/python"
        for path in (launcher, lb_script, python_bin):
            if not path.is_file():
                raise FileNotFoundError(path)
        if int(self.environment.get("QWEN36_TP_SIZE", "1")) != 1:
            raise RuntimeError("Qwen LB requires TP=1 on each GPU")

        self.output.mkdir(parents=True, exist_ok=True)
        max_num_seqs = int(self.environment.get("QWEN36_MAX_NUM_SEQS", "16"))
        if max_num_seqs < 1:
            raise ValueError("QWEN36_MAX_NUM_SEQS must be positive")
        self.devices = devices
        self.tensor_parallel_size = 1
        self.data_parallel_size = len(devices)
        self.max_num_seqs = max_num_seqs
        for device, port in zip(devices, self.backend_ports):
            env = dict(self.environment)
            env.pop("INTERACTIVE_NAV_EVAL_RUN_ID", None)
            env.update({
                "QWEN36_GPU_IDS": device, "QWEN36_PORT": str(port),
                "QWEN36_TP_SIZE": "1", "QWEN36_DP_SIZE": "1",
                "QWEN36_API_SERVER_COUNT": "1", "QWEN36_MAX_NUM_SEQS": str(max_num_seqs),
                "QWEN36_LOG_DIR": str(self.output),
                "QWEN36_RUNTIME_DIR": str(self.output / f"runtime-gpu{device}"),
            })
            self._spawn(f"gpu{device}", ["bash", str(launcher)], env)
        self._ready(self.backend_ports, stop)
        self._spawn("lb", [str(python_bin), str(lb_script),
                           "--listen-port", str(self.port),
                           "--backends", ",".join(map(str, self.backend_ports))], self.environment)
        self._ready([self.port], stop)
        (self.output / "deployment.json").write_text(json.dumps({
            "devices": devices,
            "backend_ports": list(self.backend_ports),
            "port": self.port,
            "endpoint": self.endpoint,
            "endpoint_mode": (
                "two_single_gpu_replicas_least_inflight_lb" if len(devices) == 2
                else "single_gpu_replicas_least_inflight_lb"
            ),
            "tensor_parallel_size": 1,
            "data_parallel_size_per_replica": 1,
            "max_num_seqs_per_replica": max_num_seqs,
            "total_max_num_seqs": len(devices) * max_num_seqs,
            "api_server_count_per_replica": 1,
            "restart_policy": "never",
        }, indent=2))
