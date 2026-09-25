#!/usr/bin/env python3
"""Run the eight-arm online M2 comparison against the shared Qwen service.

The launcher deliberately never starts or stops Qwen.  It checks the live
service contract and GPU inventory, assigns the 60 MuJoCo workers across all
four visible GPUs, and stops the evaluator groups together if the shared API
disappears or a GPU approaches exhaustion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OLD_ROOT = Path(
    "/home/ldl/outputs/interactive-nav/"
    "m2-online-mixed0-29-60w-8arm-20260923_000602"
)
DEFAULT_MANIFEST = Path(
    "/home/ldl/outputs/interactive-nav/"
    "m2-online-mixed0-29-20260922/manifest.json"
)
BENCHMARK = Path(
    "/home/ldl/molmospaces/scripts/InteractiveNav/output/"
    "interactive_nav_v3_procthor10k_val_release_v1_2/benchmark/benchmark.json"
)
PYTHON = Path("/home/ldl/conda_envs/mlspaces/bin/python")
EVAL = REPO / "scripts/InteractiveNav/run_benchmark_eval.py"
ENDPOINT = "http://127.0.0.1:8000/v1"
INDICES = list(range(2000, 2030))
DEVICES = ["0", "1", "2", "3"]
WORKERS = [8, 8, 8, 8, 7, 7, 7, 7]


def service_state() -> dict:
    state: dict = {}
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000" + path, timeout=3
            ) as response:
                state[path] = {
                    "status": response.status,
                    "body": response.read().decode(errors="replace"),
                }
        except Exception as exc:  # pragma: no cover - live host check
            state[path] = {"error": repr(exc)}
    try:
        payload = json.loads(state.get("/v1/models", {}).get("body", "{}"))
        model = (payload.get("data") or [{}])[0]
    except (TypeError, ValueError):
        model = {}
    state["model_id"] = model.get("id")
    state["max_model_len"] = model.get("max_model_len")
    return state


def gpu_state() -> list[dict]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            result.append(
                {
                    "index": fields[0],
                    "used_mib": int(float(fields[1])),
                    "total_mib": int(float(fields[2])),
                    "util_pct": float(fields[3]),
                }
            )
    return result


def assert_ports_free(start: int, count: int) -> None:
    for port in range(start, start + count):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


def set_env(text: str, key: str, value: object) -> str:
    prefix = key + "="
    lines = text.splitlines()
    output = []
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            if not replaced:
                output.append(prefix + str(value))
                replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(prefix + str(value))
    return "\n".join(output) + "\n"


def preflight() -> tuple[dict, list[dict]]:
    state = service_state()
    if state.get("/health", {}).get("status") != 200:
        raise RuntimeError("Qwen /health is not ready: " + repr(state))
    if state.get("/v1/models", {}).get("status") != 200:
        raise RuntimeError("Qwen /v1/models is not ready: " + repr(state))
    if state.get("model_id") != "qwen3.6-35b-a3b-fp8":
        raise RuntimeError("Unexpected Qwen model: " + repr(state))
    if state.get("max_model_len") != 10240:
        raise RuntimeError("Unexpected max_model_len: " + repr(state))
    command_lines = subprocess.check_output(
        "ps -eo args | grep '[v]llm serve' | grep -- '--port 8000' || true",
        shell=True,
        text=True,
    )
    for token in (
        "--tensor-parallel-size 1",
        "--data-parallel-size 4",
        "--max-model-len 10240",
        "--max-num-seqs 16",
    ):
        if token not in command_lines:
            raise RuntimeError(f"Qwen process is missing {token!r}: {command_lines}")
    gpus = gpu_state()
    if [row["index"] for row in gpus] != DEVICES:
        raise RuntimeError(f"Expected GPUs {DEVICES}, found {gpus}")
    return state, gpus


def build_run(root: Path, old_root: Path, manifest_path: Path, base_port: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    arms = {arm["id"]: arm for arm in manifest["arms"]}
    root.mkdir(parents=True)
    (root / "prompts").mkdir()
    (root / "groups").mkdir()
    records = {}
    common = {
        "SEMANTIC_M2_TIMEOUT_S": "12",
        "SEMANTIC_M2_TIMEOUT_RETRY_COUNT": "1",
        "SEMANTIC_M2_TIMEOUT_RETRY_BACKOFF_S": "1",
        "SEMANTIC_M2_REASONING_EFFORT": "off",
        "SEMANTIC_M2_MAX_TOKENS": "1536",
        "SEMANTIC_M2_TEMPERATURE": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MLSPACES_CACHE_DIR": "/home/ldl/molmo-spaces-resources",
        "MLSPACES_ASSETS_DIR": "/home/ldl/molmospaces/assets",
        "NLTK_DATA": "/home/ldl/nltk_data",
    }
    for group_index, group_id in enumerate(f"G{i}" for i in range(8)):
        arm = arms[group_id]
        group = root / "groups" / group_id
        group.mkdir()
        prompt_src = old_root / "prompts" / f"{group_id}.txt"
        prompt_dst = root / "prompts" / f"{group_id}.txt"
        if not prompt_src.is_file():
            raise FileNotFoundError(prompt_src)
        shutil.copy2(prompt_src, prompt_dst)
        source_env = old_root / "groups" / group_id / "semantic_model.env"
        text = source_env.read_text() if source_env.is_file() else ""
        text = text.replace(str(old_root), str(root))
        values = dict(common)
        values.update(arm.get("environment", {}))
        values.update(
            {
                "SEMANTIC_MODEL_MODE": "http",
                "SEMANTIC_MODEL_ENDPOINT": ENDPOINT,
                "SEMANTIC_MODEL_NAME": "qwen3.6-35b-a3b-fp8",
                "SEMANTIC_MODEL_PROTOCOL": "openai_chat",
                "SEMANTIC_MODEL_API_KEY_ENV": "SEMANTIC_MODEL_LOCAL_API_KEY",
                "SEMANTIC_MODEL_LOCAL_API_KEY": "local",
                "SEMANTIC_MODEL_REASONING_EFFORT": "off",
                "SEMANTIC_M2_ENDPOINT": ENDPOINT,
                "SEMANTIC_M2_MODEL_NAME": "qwen3.6-35b-a3b-fp8",
                "SEMANTIC_M2_PROMPT_FILE": str(prompt_dst),
                "SEMANTIC_M2_MODE": "http",
                "SEMANTIC_M2_PROTOCOL": "openai_chat",
                "SEMANTIC_M2_API_KEY_ENV": "INTERACTIVE_NAV_QWEN_UNUSED_API_KEY",
            }
        )
        for key, value in values.items():
            text = set_env(text, key, value)
        env_path = group / "semantic_model.env"
        env_path.write_text(text)
        if group_index < 4:
            egl = list(DEVICES)
        else:
            shift = group_index - 4
            egl = DEVICES[shift:] + DEVICES[:shift]
        workers = WORKERS[group_index]
        config = {
            "output_root": str(group / "eval"),
            "benchmark": str(BENCHMARK),
            "episode_indices": list(INDICES),
            "workers": workers,
            "max_steps": 2000,
            "min_steps": 200,
            "step_budget_mode": "dynamic",
            "ros_observation_turn_multiplier": 1.0,
            "ros_command_starvation_timeout_s": 300,
            "semantic_attribute_request_timeout_s": 15,
            "scene_timeout_s": 10800,
            "base_master_port": base_port + sum(WORKERS[:group_index]),
            "conda_env": "/home/ldl/conda_envs/mlspaces",
            "python_bin": str(PYTHON),
            "semantic_model_env_file": str(env_path),
            "model_endpoints": [ENDPOINT],
            "mujoco_egl_devices": egl,
            "check_mujoco_gpu_inventory": True,
            "required_mujoco_gpu_count": 4,
            "progress_interval_s": 10,
            "recording": False,
            "retry_rounds": 0,
            "resume": False,
        }
        config_path = group / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        records[group_id] = {
            "label": arm.get("label"),
            "workers": workers,
            "egl": egl,
            "base_master_port": config["base_master_port"],
            "config": str(config_path),
            "env": str(env_path),
            "prompt": str(prompt_dst),
            "output": str(group / "eval"),
        }
    return records


def progress(records: dict) -> dict:
    reported = []
    running = queued = 0
    for record in records.values():
        output = Path(record["output"])
        for index in INDICES:
            task = output / f"episode_{index:04d}"
            try:
                row = json.loads((task / "batch_task_summary.json").read_text())
            except (OSError, ValueError):
                row = {}
            is_reported = (
                row.get("worker_result_reported") is True
                or str(row.get("worker_result_reported", "")).lower() == "true"
                or row.get("runner_exit_code") is not None
            )
            if is_reported:
                reported.append(row)
            elif list(task.glob("attempt_*")):
                running += 1
            else:
                queued += 1
    done = [row for row in reported if row.get("completed") is True]

    def rate(key):
        values = [row.get(key) for row in done if isinstance(row.get(key), bool)]
        return None if not values else sum(values) / len(values)

    def mean(key):
        values = [row.get(key) for row in done if isinstance(row.get(key), (int, float))]
        return None if not values else sum(values) / len(values)

    return {
        "reported": len(reported),
        "completed": len(done),
        "failed": len(reported) - len(done),
        "running": running,
        "queued": queued,
        "task_sr": rate("task_success"),
        "nav_sr": rate("nav_success"),
        "success": rate("success"),
        "spl": mean("spl"),
        "steps": mean("step_count"),
    }


def stop_groups(processes: dict[str, subprocess.Popen], reason: str) -> None:
    print(f"[guard] stopping evaluator groups: {reason}", flush=True)
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(p.poll() is None for p in processes.values()):
        time.sleep(0.5)
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def run(args: argparse.Namespace) -> int:
    state, gpus = preflight()
    old_root = args.old_root.expanduser().resolve()
    if not args.manifest.is_file() or not old_root.is_dir() or not BENCHMARK.is_file():
        raise FileNotFoundError("missing frozen manifest, prompt root, or benchmark")
    total_workers = sum(WORKERS)
    assert_ports_free(args.base_master_port, total_workers)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.output_root.expanduser().resolve() / f"m2-online-mixed0-29-60w-8arm-4gpu-{stamp}"
    if root.exists():
        raise FileExistsError(root)
    records = build_run(root, old_root, args.manifest, args.base_master_port)
    (root / "preflight.json").write_text(
        json.dumps({"service": state, "gpu_inventory": gpus, "devices": DEVICES}, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "experiment_spec.json").write_text(
        json.dumps(
            {
                "experiment_id": root.name,
                "service_contract": {"endpoint": ENDPOINT, "tp": 1, "dp": 4, "max_model_len": 10240, "max_num_seqs": 16},
                "benchmark": str(BENCHMARK),
                "episode_indices": INDICES,
                "workers_total": total_workers,
                "workers_by_arm": dict(zip(records, WORKERS)),
                "egl_by_arm": {key: value["egl"] for key, value in records.items()},
                "arms": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    processes = {}
    logs = {}
    for group_id, record in records.items():
        command = [str(PYTHON), "-u", str(EVAL), "--config", record["config"], "--output-dir", record["output"]]
        log = (Path(record["output"]).parent / "launcher.log").open("wb")
        logs[group_id] = log
        process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[group_id] = process
        print(f"[launch] {group_id} pid={process.pid} workers={record['workers']} egl={record['egl']} ports={record['base_master_port']}-{record['base_master_port'] + record['workers'] - 1}", flush=True)
    (root / "launched.json").write_text(
        json.dumps({key: {"pid": process.pid, **records[key]} for key, process in processes.items()}, ensure_ascii=False, indent=2) + "\n"
    )
    started = time.monotonic()
    last_print = 0.0
    health_failures = 0
    guard_reason = None
    try:
        while True:
            now = time.monotonic()
            if now - last_print >= 15:
                s = progress(records)
                current_gpus = gpu_state()
                print(
                    f"[overall {(now - started) / 60:.1f}min] completed {s['completed']}/240 | reported {s['reported']} | running {s['running']} | queued {s['queued']} | failures {s['failed']} | TaskSR={s['task_sr']} NavSR={s['nav_sr']} SPL={s['spl']} Steps={s['steps']} | GPU={[row['used_mib'] for row in current_gpus]}",
                    flush=True,
                )
                current_health = service_state()
                if current_health.get("/health", {}).get("status") != 200 or current_health.get("/v1/models", {}).get("status") != 200:
                    health_failures += 1
                    print(f"[health] failure {health_failures}/2: {current_health}", flush=True)
                else:
                    health_failures = 0
                if any(row["used_mib"] >= 80000 for row in current_gpus):
                    guard_reason = "GPU memory guard reached 80000 MiB"
                last_print = now
            if health_failures >= 2:
                guard_reason = "Qwen HTTP health lost twice"
            if guard_reason:
                stop_groups(processes, guard_reason)
                break
            if not any(process.poll() is None for process in processes.values()):
                break
            time.sleep(5)
    except KeyboardInterrupt:
        guard_reason = "manual interrupt"
        stop_groups(processes, guard_reason)
    finally:
        for log in logs.values():
            log.close()
    final = {
        "finished_at": dt.datetime.now().astimezone().isoformat(),
        "guard_reason": guard_reason,
        "summary": progress(records),
        "process_codes": {key: process.returncode for key, process in processes.items()},
        "qwen_after": service_state(),
        "gpu_after": gpu_state(),
    }
    (root / "final_monitor.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print("[final] " + json.dumps(final, ensure_ascii=False), flush=True)
    return 0 if guard_reason is None and final["summary"]["completed"] == 240 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=Path("/home/ldl/outputs/interactive-nav"))
    parser.add_argument("--base-master-port", type=int, default=21000)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
