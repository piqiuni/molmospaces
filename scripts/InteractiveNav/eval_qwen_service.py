"""Batch-owned single vLLM service over all visible GPUs; no automatic restarts."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


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
    def __init__(self, output, environment, port=8000):
        self.output = Path(output) / "qwen-service"
        self.environment = dict(environment)
        self.port = int(port)
        self.endpoint = endpoint_for_port(self.port)
        self.devices = []
        self.tensor_parallel_size = None
        self.data_parallel_size = None
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
        env["QWEN36_TP_SIZE"] = str(tensor_parallel)
        env["QWEN36_DP_SIZE"] = str(data_parallel)
        # One API process lets the vLLM internal DP scheduler see all ranks.
        env["QWEN36_API_SERVER_COUNT"] = "1"
        env.setdefault("QWEN36_MAX_MODEL_LEN", "16384")
        env.setdefault("QWEN36_MAX_NUM_SEQS", "16")
        self.devices = devices
        self.tensor_parallel_size = tensor_parallel
        self.data_parallel_size = data_parallel
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
            "api_server_count": 1,
            "max_model_len": int(env["QWEN36_MAX_MODEL_LEN"]),
            "max_num_seqs": int(env["QWEN36_MAX_NUM_SEQS"]),
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
