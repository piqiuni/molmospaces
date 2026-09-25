"""Fixed-scene, multi-arm online evaluation lanes; no implicit episode retries."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import urllib.request


SCHEMA = "interactive_nav_m2_online_experiment_v1"
SIX_ARM_SCHEMA = "interactive_nav_m2_online_experiment_v2"
MODEL_NAME = "qwen3.6-35b-a3b-fp8"
ARM_DESIGN = {
    "G0": ("historical_compat_v1", "legacy", "B", 8),
    "G1": ("public_facts_v2", "legacy", "B", 30),
    "G2": ("public_facts_v2", "all_actions_12_frontiers", "B", 30),
    **{f"G{i}": ("public_facts_v2", "all_actions_12_frontiers", prompt, 30)
       for i, prompt in enumerate(("P1", "P2", "P3", "P4", "P6"), 3)},
}
SIX_ARM_DESIGN = {
    "G0": ("historical_compat_v1", "legacy", "B", 8),
    "G1": ("public_facts_v2", "legacy", "B", 8),
    "G2": ("public_facts_v2", "legacy", "B", 30),
    "G3": ("public_facts_v2", "all_actions_12_frontiers", "B", 30),
    "G4": ("public_facts_v2", "all_actions_12_frontiers", "P1", 30),
    "G5": ("public_facts_v2", "all_actions_12_frontiers", "P6", 30),
}
SIX_ARM_LANES = {0: {"G0", "G1", "G2"}, 1: {"G3", "G4", "G5"}}
THREAD_KEYS = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}
DATA_PATH_KEYS = {"MLSPACES_CACHE_DIR", "MLSPACES_ASSETS_DIR", "NLTK_DATA"}
MODEL_SUFFIXES = {"MODE", "COMMAND", "ENDPOINT", "API_KEY_ENV", "NAME", "PROTOCOL", "TIMEOUT_S",
                  "TEMPERATURE", "MAX_TOKENS", "REASONING_EFFORT", "IMAGE_DETAIL"}
M2_SUFFIXES = (MODEL_SUFFIXES - {"NAME"}) | {
    "MODEL_NAME", "TIMEOUT_RETRY_COUNT", "TIMEOUT_RETRY_BACKOFF_S", "CONTEXT_PROFILE", "PROMPT_FILE",
    "RECENT_DECISION_LIMIT", "CANDIDATE_POOL_MODE", "CANDIDATE_TOP_K", "CANDIDATE_MAX_FRONTIER_CANDIDATES",
}
ALLOWED_ENV = THREAD_KEYS | DATA_PATH_KEYS | {"SEMANTIC_MODEL_" + key for key in MODEL_SUFFIXES} | {
    "SEMANTIC_M2_" + key for key in M2_SUFFIXES}
ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=")


def _imports():
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import run_benchmark_eval as baseline
    import run_interactive_nav_v3_ros_eval_batch as batch
    return baseline, batch


def _environment(values):
    if not isinstance(values, dict) or set(values) - ALLOWED_ENV:
        raise ValueError("Only explicitly allowed experiment/model environment keys may be overridden")
    result = {key: str(value) for key, value in values.items()}
    if any("\n" in value or "\x00" in value for value in result.values()):
        raise ValueError("Environment values must be single-line")
    for key in DATA_PATH_KEYS & result.keys():
        _output_path(result[key])
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files():
    repo = Path(__file__).resolve().parents[2]
    scripts = repo / "scripts/InteractiveNav"
    decision = repo / "Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts"
    return [Path(__file__), scripts / "run_benchmark_eval.py", scripts / "run_interactive_nav_v3_ros_eval_batch.py",
            scripts / "run_interactive_nav_v3_ros_eval_test.zsh", scripts / "configs/evaluation/benchmark_eval.conf",
            decision / "semantic_rule_decision_node.py", decision / "semantic_candidate_node.py",
            decision.parent / "config/default.yaml", scripts / "configs/semantic_decision/object_goal_v3_full_mllm.yaml",
            repo / "Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts/semantic_mllm_py_pkg/client.py",
            *[decision / "semantic_decision_py_pkg" / name for name in (
                "model_policy.py", "env_config.py", "candidate_curator.py", "behavior_candidates.py",
                "public_robot_context.py", "room_context.py", "frontier_context.py")],
            scripts / "evaluation/goal_status.py", scripts / "evaluation/goal_equivalence.py",
            scripts / "evaluation/scene_distractor_filter.py",
            decision / "semantic_decision_py_pkg/mission_completion.py"]


def prepare_manifest(template_path: Path, output_dir: Path) -> dict:
    """Freeze shared prompts, configuration, source hashes and plans; no services or dotenv reads."""
    output_dir = Path(_output_path(output_dir))
    if (output_dir / "manifest.json").exists() or (output_dir / "prompts").exists():
        raise FileExistsError("Use a fresh experiment manifest/prompt destination")
    manifest = normalize_manifest(json.loads(template_path.read_text()), template_path.parent)
    if any(not Path(lane["output_dir"]).is_relative_to(output_dir) for lane in manifest["lanes"]):
        raise ValueError("All lane outputs must belong to the frozen experiment root")
    if not isinstance(manifest.get("launcher_config"), dict):
        manifest["launcher_config"] = json.loads(Path(manifest["launcher_config"]).read_text())
    benchmark = Path(manifest.get("benchmark", manifest["launcher_config"]["benchmark"]))
    actual = _sha(benchmark)
    if manifest.get("benchmark_sha256") and manifest["benchmark_sha256"] != actual:
        raise ValueError("Benchmark hash differs from the template")
    manifest["benchmark"], manifest["benchmark_sha256"] = str(benchmark), actual
    pending = []
    for arm in manifest["arms"]:
        source = Path(({**manifest["common_environment"], **arm["environment"]})["SEMANTIC_M2_PROMPT_FILE"])
        content = source.read_bytes()
        if not content.decode("utf-8").strip():
            raise ValueError(f"Empty frozen prompt for {arm['id']}")
        raw_hash = hashlib.sha256(content).hexdigest()
        if arm.get("prompt_sha256") and arm["prompt_sha256"] != raw_hash:
            raise ValueError(f"Prompt differs from template: {arm['id']}")
        destination = output_dir / "prompts" / f"{arm['id']}.txt"
        arm.update(prompt_source=str(source), prompt_sha256=raw_hash,
                   prompt_strip_sha256=hashlib.sha256(content.decode("utf-8").strip().encode()).hexdigest())
        arm["environment"]["SEMANTIC_M2_PROMPT_FILE"] = str(destination)
        pending.append((destination, content))
    manifest["source_sha256"] = {str(path): _sha(path) for path in _source_files()}
    manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    manifest["frozen_at"] = time.time()
    manifest["template_sha256"] = _sha(template_path)
    manifest["experiment_root"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts").mkdir()
    for destination, content in pending:
        destination.write_bytes(content)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def verify_frozen_manifest(manifest):
    if manifest.get("benchmark_sha256") and _sha(manifest["benchmark"]) != manifest["benchmark_sha256"]:
        raise ValueError("Frozen benchmark hash mismatch")
    for path, expected in manifest.get("source_sha256", {}).items():
        if _sha(path) != expected:
            raise ValueError(f"Frozen source hash mismatch: {path}")
    for arm in manifest["arms"]:
        source = arm_environment(manifest, manifest["lanes"][0], arm)["SEMANTIC_M2_PROMPT_FILE"]
        if arm.get("prompt_sha256") and _sha(source) != arm["prompt_sha256"]:
            raise ValueError(f"Frozen prompt hash mismatch: {arm['id']}")


def _output_path(value):
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(Path("/home/ldl")) or path.resolve() == Path("/home/ldl"):
        raise ValueError("Output directories must be explicit subdirectories of /home/ldl")
    return str(path.resolve())


def normalize_manifest(document: dict, manifest_dir: Path | None = None) -> dict:
    """Validate and expand the frozen 8-arm legacy or 6-arm two-task design."""
    result = deepcopy(document)
    schema = result.get("schema_version")
    if schema not in {SCHEMA, SIX_ARM_SCHEMA} or not result.get("experiment_id"):
        raise ValueError("Invalid experiment manifest schema/identity")
    six_arm = schema == SIX_ARM_SCHEMA
    arm_design = SIX_ARM_DESIGN if six_arm else ARM_DESIGN
    lane_ids = set(SIX_ARM_LANES) if six_arm else {0, 1, 2}
    workers = 20 if six_arm else 15
    if result.get("mixed_indices") != list(range(30)) or int(result.get("episode_start", 2000)) != 2000:
        raise ValueError("This experiment must contain Mixed 0..29, global indices 2000..2029")
    for key, expected in (("worker_count", workers), ("max_steps", 2000), ("min_steps", 200),
                          ("step_budget_mode", "dynamic"), ("recording", False)):
        if result.get(key, expected) != expected:
            raise ValueError(f"Frozen design requires {key}={expected}")
        result[key] = expected
    result["episode_start"] = 2000
    result["seed"] = int(result.get("seed", 20260922))
    result["common_environment"] = _environment(result.get("common_environment", {}))
    result["worker_start_interval_s"] = float(result.get("worker_start_interval_s", 2))
    result["infra_failure_pause_threshold"] = int(result.get("infra_failure_pause_threshold", 3))
    if result["worker_start_interval_s"] < 0 or result["infra_failure_pause_threshold"] < 1:
        raise ValueError("Invalid launch interval or infrastructure stop threshold")
    arms = result.get("arms", [])
    if len(arms) != len(arm_design) or {arm.get("id") for arm in arms} != set(arm_design):
        raise ValueError(f"Expected exactly {sorted(arm_design)}")
    for arm in arms:
        arm["environment"] = _environment(arm.get("environment", {}))
        merged = {**result["common_environment"], **arm["environment"]}
        profile, pool, prompt, history = arm_design[arm["id"]]
        required = {"SEMANTIC_M2_CONTEXT_PROFILE": profile, "SEMANTIC_M2_CANDIDATE_POOL_MODE": pool,
                    "SEMANTIC_M2_RECENT_DECISION_LIMIT": str(history), "SEMANTIC_M2_CANDIDATE_TOP_K": "8",
                    "SEMANTIC_M2_CANDIDATE_MAX_FRONTIER_CANDIDATES": "12"}
        for key, value in required.items():
            if merged.get(key) != value:
                raise ValueError(f"Arm {arm['id']} violates frozen design: {key}")
        if arm.get("context", profile) != profile or arm.get("pool", pool) != pool or str(arm.get("prompt", prompt)).upper() != prompt:
            raise ValueError(f"Arm {arm['id']} metadata contradicts its design")
        path = Path(merged.get("SEMANTIC_M2_PROMPT_FILE", ""))
        if not path.is_absolute():
            raise ValueError("Every arm needs an explicit absolute frozen prompt file")
        arm.update(context=profile, pool=pool, prompt=prompt)
    lanes = result.get("lanes", [])
    if len(lanes) != len(lane_ids) or {lane.get("id") for lane in lanes} != lane_ids:
        raise ValueError(f"Expected exactly lanes {sorted(lane_ids)}")
    for lane in lanes:
        lane["output_dir"] = _output_path(lane["output_dir"])
        parsed = urlparse(lane["model_endpoint"])
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.username or parsed.password or not parsed.port or parsed.path != "/v1":
            raise ValueError("Each lane must use a loopback HTTP Qwen /v1 endpoint without credentials")
        if not six_arm:
            lane["egl_device_id"] = str(lane["egl_device_id"])
        if (not six_arm and not lane["egl_device_id"].isdigit()) or not 1024 <= int(lane["base_master_port"]) <= 65515:
            raise ValueError("Invalid EGL ID or ROS port range")
        lane["environment"] = _environment(lane.get("environment", {}))
    for i, left in enumerate(lanes):
        for right in lanes[i + 1:]:
            a, b = Path(left["output_dir"]), Path(right["output_dir"])
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("Lane output directories overlap")
    local = {lane["id"]: lane for lane in lanes}
    if not six_arm and abs(int(local[0]["base_master_port"]) - int(local[1]["base_master_port"])) < workers:
        raise ValueError("Local ROS port ranges overlap")
    if not six_arm and local[0]["egl_device_id"] == local[1]["egl_device_id"]:
        raise ValueError("Local simulation lanes must have distinct EGL devices")
    jobs = []
    for lane in sorted(lanes, key=lambda item: item["id"]):
        selected_arms = SIX_ARM_LANES[lane["id"]] if six_arm else set(arm_design)
        lane_jobs = [{"job_id": f"{arm['id']}-mixed{index:04d}", "arm": arm["id"], "mixed_index": index,
                      "episode_index": 2000 + index, "lane_id": lane["id"],
                      "output_dir": str(Path(lane["output_dir"]) / arm["id"] / f"episode_{2000 + index:04d}")}
                     for index in range(30) if six_arm or index % 3 == lane["id"]
                     for arm in sorted(arms, key=lambda item: item["id"]) if arm["id"] in selected_arms]
        random.Random(result["seed"] + lane["id"]).shuffle(lane_jobs)
        jobs.extend(lane_jobs)
    supplied = result.get("planned_jobs")
    if supplied is not None and {job["job_id"]: job for job in supplied} != {job["job_id"]: job for job in jobs}:
        raise ValueError("Supplied planned jobs differ from the fixed design")
    result["planned_jobs"] = jobs
    result["retry_policy"] = "none; preserve attempt_001 and publish explicit incomplete/infrastructure failure queue"
    if isinstance(result.get("launcher_config"), str) and manifest_dir is not None:
        config = Path(result["launcher_config"])
        result["launcher_config"] = str(config if config.is_absolute() else (manifest_dir / config).resolve())
    return result


def arm_environment(manifest, lane, arm):
    values = {**manifest["common_environment"], **lane.get("environment", {}), **arm["environment"]}
    values.update({key: "1" for key in THREAD_KEYS})
    for prefix in ("SEMANTIC_MODEL_", "SEMANTIC_M2_"):
        values.update({prefix + "MODE": "http", prefix + "PROTOCOL": "openai_chat",
                       prefix + "ENDPOINT": lane["model_endpoint"], prefix + "COMMAND": "",
                       prefix + "API_KEY_ENV": "INTERACTIVE_NAV_QWEN_UNUSED_API_KEY"})
    values.update(SEMANTIC_MODEL_NAME=MODEL_NAME, SEMANTIC_M2_MODEL_NAME=MODEL_NAME)
    return values


def derived_env_text(source: str, overrides: dict[str, str]) -> str:
    """Remove every previous assignment, including duplicates, before appending."""
    lines = [line for line in source.splitlines() if not ((match := ENV_LINE.match(line)) and match[1] in overrides)]
    lines += ["", "# Frozen online experiment overrides; one assignment per overridden key."]
    lines += [f"{key}={shlex.quote(str(value))}" for key, value in sorted(overrides.items())]
    return "\n".join(lines) + "\n"


def wrapper_text(snapshot: Path, overrides: dict[str, str]) -> str:
    exports = "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in sorted(overrides.items()))
    return f'#!/usr/bin/env bash\nset -euo pipefail\n{exports}\nexec bash {shlex.quote(str(snapshot))} "$@"\n'


def classify_incomplete(row):
    if row.get("completed") is True:
        return None
    error = str(row.get("error") or "")
    lower = error.lower()
    markers = ("no space left on device", "cannot initialize a egl", "cuda out of memory",
               "modulenotfounderror", "no module named", "filenotfounderror", "connection refused")
    clear = any(marker in lower for marker in markers) or (row.get("runner_exit_code") == 127 and bool(error))
    return {"classification": "confirmed_infrastructure" if clear else "incomplete_unclassified",
            "reason": error or f"incomplete exit={row.get('runner_exit_code')} status={row.get('episode_status')}",
            "retry_authorized": False}


def validate_model_inventory(document, *, minimum_context=16384):
    found = next((item for item in document.get("data", []) if item.get("id") == MODEL_NAME), None)
    if found is None:
        raise RuntimeError("qwen_model_identity_mismatch")
    limit = found.get("max_model_len")
    if limit is not None and int(limit) < minimum_context:
        raise RuntimeError(f"qwen_context_window_below_{minimum_context}")
    return {"healthy": True, "model": MODEL_NAME, "reported_max_model_len": limit, "checked_at": time.time()}


def check_qwen(endpoint, *, minimum_context=16384):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(endpoint.rstrip("/") + "/models", timeout=3) as response:
        return validate_model_inventory(json.load(response), minimum_context=minimum_context)


def lane_status(manifest, lane, jobs, results, active, state, started, *, error=None):
    completed = sum(row.get("completed") is True for row in results.values())
    per_arm = {}
    for arm in manifest["arms"]:
        selected = [job for job in jobs if job["arm"] == arm["id"]]
        rows = [results[job["job_id"]] for job in selected if job["job_id"] in results]
        per_arm[arm["id"]] = {"planned": len(selected), "reported": len(rows),
                              "completed": sum(row.get("completed") is True for row in rows),
                              "failed": sum(row.get("completed") is not True for row in rows),
                              "success": sum(row.get("completed") is True and bool(row.get("success")) for row in rows)}
    failures = [{**job, **classify_incomplete(results[job["job_id"]])} for job in jobs
                if job["job_id"] in results and classify_incomplete(results[job["job_id"]])]
    return {"schema_version": SCHEMA, "experiment_id": manifest["experiment_id"], "lane_id": lane["id"],
            "state": state, "updated_at": time.time(), "elapsed_sec": time.monotonic() - started,
            "planned": len(jobs), "reported": len(results), "active": list(active.values()),
            "queued": len(jobs) - len(results) - len(active), "completed": completed,
            "failed": len(results) - completed, "per_arm": per_arm, "failure_queue": failures,
            "error": error, "auto_retry_enabled": False}


def _config(manifest, lane, fallback):
    source = manifest.get("launcher_config", fallback)
    config = deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text())
    config.update(manifest.get("config_overrides", {}))
    config.update(lane.get("config_overrides", {}))
    six_arm = manifest["schema_version"] == SIX_ARM_SCHEMA
    config.update(benchmark=manifest.get("benchmark", config["benchmark"]), workers=manifest["worker_count"],
                  episode_indices=[2000 + i for i in range(30) if six_arm or i % 3 == lane["id"]],
                  max_steps=2000, min_steps=200, step_budget_mode="dynamic", recording=False, resume=False,
                  base_master_port=int(lane["base_master_port"]), model_endpoints=[lane["model_endpoint"]],
                  mujoco_egl_devices=["0", "1", "2", "3"] if six_arm else [lane["egl_device_id"]])
    if six_arm:
        config.update(check_mujoco_gpu_inventory=True, required_mujoco_gpu_count=4)
    config["ros_command_starvation_timeout_s"] = max(300, float(config["ros_command_starvation_timeout_s"]))
    if lane.get("semantic_model_env_file"):
        config["semantic_model_env_file"] = lane["semantic_model_env_file"]
    return config


def _native_args(config, output, runner, baseline, batch):
    command, _ = baseline.build_command(config, output)
    previous = sys.argv
    try:
        sys.argv = [command[2], *command[3:], "--runner", str(runner)]
        args = batch.parse_args()
    finally:
        sys.argv = previous
    batch.validate_args(args)
    return args


def run_lane(manifest_path: Path, lane_id: int, *, config_path=None, dry_run=False, start_qwen=False) -> int:
    baseline, batch = _imports()
    manifest = normalize_manifest(json.loads(manifest_path.read_text()), manifest_path.parent)
    verify_frozen_manifest(manifest)
    lane = next((item for item in manifest["lanes"] if item["id"] == lane_id), None)
    if lane is None:
        raise ValueError("Unknown experiment lane")
    jobs = [job for job in manifest["planned_jobs"] if job["lane_id"] == lane_id]
    config = _config(manifest, lane, config_path or baseline.DEFAULT_CONFIG)
    if manifest["schema_version"] == SIX_ARM_SCHEMA and not dry_run:
        baseline.check_mujoco_gpu_inventory(config)
    output = Path(lane["output_dir"])
    if start_qwen and manifest["schema_version"] == SCHEMA and lane_id != 2:
        raise ValueError("Local lanes must share the existing Qwen service; --start-qwen is remote-only")
    if dry_run:
        print(json.dumps({"experiment_id": manifest["experiment_id"], "lane_id": lane_id, "workers": config["workers"],
                          "start_qwen": start_qwen, "config": config, "jobs": jobs,
                          "arm_overrides": {arm["id"]: arm_environment(manifest, lane, arm) for arm in manifest["arms"]},
                          "retry_policy": manifest["retry_policy"]}, ensure_ascii=False, indent=2))
        return 0
    for port in range(config["base_master_port"], config["base_master_port"] + config["workers"]):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    output.mkdir(parents=True, exist_ok=False)
    source_runner = baseline.REPO / "scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh"
    snapshot = output / "runner_snapshot.sh"
    snapshot.write_bytes(source_runner.read_bytes())
    source_env = Path(config["semantic_model_env_file"])
    if not source_env.is_absolute():
        source_env = baseline.REPO / source_env
    source_text = source_env.read_text()
    args_by_arm = {}
    for arm in manifest["arms"]:
        directory = output / arm["id"] / "experiment"
        directory.mkdir(parents=True)
        overrides = arm_environment(manifest, lane, arm)
        prompt_source = Path(overrides["SEMANTIC_M2_PROMPT_FILE"])
        prompt_bytes = prompt_source.read_bytes()
        if arm.get("prompt_sha256") and hashlib.sha256(prompt_bytes).hexdigest() != arm["prompt_sha256"]:
            raise ValueError(f"Frozen prompt hash differs: {arm['id']}")
        prompt_copy = directory / "prompt.txt"
        prompt_copy.write_bytes(prompt_bytes)
        overrides["SEMANTIC_M2_PROMPT_FILE"] = str(prompt_copy)
        env_file = directory / "semantic_model.env"
        env_file.write_text(derived_env_text(source_text, overrides))
        env_file.chmod(0o600)
        wrapper = directory / "runner.sh"
        wrapper.write_text(wrapper_text(snapshot, overrides))
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        arm_config = {**config, "semantic_model_env_file": str(env_file)}
        args_by_arm[arm["id"]] = _native_args(arm_config, output / arm["id"], wrapper, baseline, batch)
        batch.atomic_json(directory / "provenance.json", {"arm": arm["id"], "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                          "override_environment": overrides, "runner_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()})
    for suffix in ("tmp", "cache"):
        (output / suffix).mkdir()
    os.environ.update({baseline.OWNER_KEY: str(output), "MIN_STEPS": "200", "PYTHONDONTWRITEBYTECODE": "1",
                       "INTERACTIVE_NAV_SCRIPT_DIR": str(source_runner.parent), "TMPDIR": str(output / "tmp"),
                       "XDG_CACHE_HOME": str(output / "cache"), **{key: "1" for key in THREAD_KEYS}})
    batch.atomic_json(output / "experiment_manifest.json", manifest)
    stop, lock, launch_lock = threading.Event(), threading.Lock(), threading.Lock()
    results, active, errors = {}, {}, []
    pending = queue.Queue()
    for ordinal, job in enumerate(jobs):
        pending.put((ordinal, job))
    started = time.monotonic()
    next_start = started
    consecutive_incomplete = 0
    interrupted = False
    telemetry = batch.BatchResourceTelemetry(output, 5.0)
    service_health = {"healthy": None, "consecutive_failures": 0}
    service = baseline.QwenService(output, os.environ, port=urlparse(lane["model_endpoint"]).port,
                                   target_concurrency=config["workers"]) if start_qwen else None
    minimum_context = 10240 if manifest["schema_version"] == SIX_ARM_SCHEMA else 16384
    def request_stop(*_):
        nonlocal interrupted
        interrupted = True
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    def publish(state):
        with lock:
            status = lane_status(manifest, lane, jobs, results, active, state, started, error="; ".join(errors) or None)
            status["service_health"] = dict(service_health)
        batch.atomic_json(output / "lane_status.json", status)
        batch.atomic_json(output / "failure_queue.json", status["failure_queue"])
        return status
    def worker(worker_id):
        nonlocal next_start, consecutive_incomplete
        while not stop.is_set():
            with launch_lock:
                if stop.wait(max(0, next_start - time.monotonic())):
                    return
                try:
                    ordinal, job = pending.get_nowait()
                except queue.Empty:
                    return
                next_start = time.monotonic() + manifest["worker_start_interval_s"]
            plan = batch.EpisodePlan(ordinal, worker_id, job["episode_index"], config["base_master_port"] + worker_id, Path(job["output_dir"]))
            with lock:
                active[worker_id] = {**job, "worker_id": worker_id, "started_at": time.time()}
            try:
                row = batch.run_episode(plan, args_by_arm[job["arm"]], telemetry)
            except Exception as exc:
                row = batch.failed_plan_result(plan, args_by_arm[job["arm"]], exc)
            row.update(experiment_id=manifest["experiment_id"], experiment_arm=job["arm"], lane_id=lane_id,
                       job_id=job["job_id"], mixed_index=job["mixed_index"])
            batch.atomic_json(Path(job["output_dir"]) / "batch_task_summary.json", row)
            failure = classify_incomplete(row)
            with lock:
                results[job["job_id"]] = row
                active.pop(worker_id, None)
                consecutive_incomplete = consecutive_incomplete + 1 if failure else 0
                if consecutive_incomplete >= manifest["infra_failure_pause_threshold"]:
                    errors.append("consecutive_incomplete_threshold; conservative queue pause, not an infrastructure classification; no retries")
                    stop.set()
            pending.task_done()
    publish("starting")
    telemetry.start()
    try:
        if service:
            startup_errors = []
            def start_service():
                try:
                    service.start(stop)
                except BaseException as exc:
                    startup_errors.append(exc)
            starter = threading.Thread(target=start_service, name="owned-qwen-startup")
            starter.start()
            while starter.is_alive():
                publish("starting_qwen")
                starter.join(timeout=10)
            if startup_errors:
                raise startup_errors[0]
        service_health.update(check_qwen(lane["model_endpoint"], minimum_context=minimum_context), consecutive_failures=0)
        next_health_check = time.monotonic() + 30
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            futures = [executor.submit(worker, worker_id) for worker_id in range(config["workers"])]
            while not all(future.done() for future in futures):
                status = publish("running")
                print(json.dumps({key: status[key] for key in ("lane_id", "planned", "reported", "completed", "failed", "queued", "per_arm")}), flush=True)
                if service:
                    try:
                        service.check()
                    except RuntimeError as exc:
                        errors.append(str(exc)); stop.set()
                if time.monotonic() >= next_health_check:
                    try:
                        service_health.update(check_qwen(lane["model_endpoint"], minimum_context=minimum_context), consecutive_failures=0)
                    except (OSError, ValueError, RuntimeError):
                        service_health.update(healthy=False, checked_at=time.time(),
                                              consecutive_failures=service_health["consecutive_failures"] + 1)
                        if service_health["consecutive_failures"] >= 3:
                            errors.append("qwen_health_failure_threshold; queue dispatch paused; shared service never restarted")
                            stop.set()
                    next_health_check = time.monotonic() + 30
                if any(future.done() and future.exception() is not None for future in futures):
                    errors.append("worker_exception; queue dispatch paused")
                    stop.set()
                if stop.is_set():
                    baseline.cleanup(str(output))
                if stop.is_set():
                    time.sleep(0.2)
                else:
                    stop.wait(min(10, float(config.get("progress_interval_s", 10))))
            for future in futures:
                future.result()
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        stop.set()
    finally:
        baseline.cleanup(str(output))
        resource = telemetry.stop()
        if service:
            service.close()
    state = "stopped" if interrupted else "failed" if errors or len(results) != len(jobs) or any(row.get("completed") is not True for row in results.values()) else "completed"
    status = publish(state)
    batch.atomic_json(output / "lane_result.json", {"status": status, "results": results, "resource_summary": resource,
                      "pending_jobs": [job for job in jobs if job["job_id"] not in results]})
    return 130 if interrupted else int(state != "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--normalize-manifest", type=Path)
    action.add_argument("--prepare-experiment", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.prepare_experiment:
        if args.output_dir is None:
            parser.error("--prepare-experiment requires --output-dir")
        manifest = prepare_manifest(args.prepare_experiment, args.output_dir)
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "planned_jobs": len(manifest["planned_jobs"]), "git_head": manifest["git_head"]}))
    else:
        print(json.dumps(normalize_manifest(json.loads(args.normalize_manifest.read_text()), args.normalize_manifest.parent), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
