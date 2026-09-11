#!/usr/bin/env python3
"""Run isolated single-episode InteractiveNav V3 ROS evaluations in parallel.

The V3 ROS runner owns one ROS master, simulator, recorder, evaluator, and
offline visual artefact set.  It must therefore be invoked once per episode;
this wrapper only schedules those independent invocations.  A task directory
keeps every attempt, so ``--resume`` can retry an incomplete episode without
overwriting its previous ROS/evaluator evidence.

Example (outputs and caches should live on the large /home/ldl volume)::

    /home/ldl/conda_envs/mlspaces/bin/python \
      scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py \
      --benchmark scripts/InteractiveNav/output/.../benchmark/mixed.json \
      --output-dir /home/ldl/outputs/interactive-nav/v3_mixed_10x \
      --episode-indices 0 1 2 3 4 5 6 7 8 9 \
      --workers 10 --base-master-port 12600 --max-steps 1500 \
      --model-endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
      --no-recording --scene-timeout-s 7200 --allow-failures

``--no-recording`` (also available as ``--fast-eval``) retains the canonical
episode-result JSON and batch telemetry but starts no recorder, writes no
per-step frames, and does not wait for a recorder acknowledgement.  This is
the intended mode for the large V3 success-rate runs.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "scripts" / "InteractiveNav" / "run_interactive_nav_v3_ros_eval_test.zsh"
SUMMARY_SCHEMA_VERSION = "interactive_nav_v3_ros_batch_v2"
PLANNED_INVOCATION_SCHEMA_VERSION = "interactive_nav_v3_ros_planned_invocation_v1"
PLANNED_INVOCATION_FILENAME = "planned_invocation.json"

# The single-episode runner is intentionally configured through environment
# variables.  Keep the defaults mirrored here so a resumed batch can prove it
# is reusing the same launch contract without serialising output-local paths.
_RUNNER_ENV_DEFAULTS: dict[str, str] = {
    "METHOD": "full_mllm_object_goal",
    "POLICY": "ros_object_goal_rule",
    "MIN_STEPS": "300",
    "DYNAMIC_PATH_FREE_M": "3.0",
    "DYNAMIC_STEPS_PER_PATH_M": "25.0",
    "DYNAMIC_CHANNEL_INTERACTION_STEPS": "150",
    "DYNAMIC_CONTAINER_INTERACTION_STEPS": "200",
    "DYNAMIC_CONTAINER_JOINT_STEPS": "40",
    "DYNAMIC_STEP_QUANTUM": "50",
    "VIDEO_FPS": "5",
    "RECORD_HEAD_CAMERA": "false",
    "SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS": "384",
    "ROS_ACTION_TIMEOUT_S": "0.2",
    "ROS_STEP_READY_BARRIER_ENABLED": "true",
    "ROS_STEP_READY_TOPIC": "/semantic_decision/step_ready",
    "ROS_STEP_READY_WARMUP_SKIP_FRAMES": "0",
    "ROS_STEP_READY_TIMEOUT_S": "2.0",
    "ROS_STEP_READY_BOOTSTRAP_TIMEOUT_S": "10.0",
    "SEMANTIC_DECISION_OVERRIDE": str(
        REPO_ROOT / "scripts" / "InteractiveNav" / "configs" / "semantic_decision" / "object_goal_v3_full_mllm.yaml"
    ),
    "SEMANTIC_MAPPING_OVERRIDE": str(
        REPO_ROOT / "scripts" / "InteractiveNav" / "configs" / "semantic_decision" / "full_mllm_mapping.yaml"
    ),
    "EXPLORE_PY_CONFIG_OVERRIDE": str(
        REPO_ROOT / "scripts" / "InteractiveNav" / "configs" / "semantic_decision" / "semantic_controlled_explore.yaml"
    ),
    "NAV_CONFIG_OVERRIDE": str(
        REPO_ROOT / "scripts" / "InteractiveNav" / "configs" / "semantic_decision" / "semantic_interaction_nav.yaml"
    ),
    "ROS_SETUP": str(REPO_ROOT / "Interactive-Nav-SG-nav" / "devel" / "setup.bash"),
}
_RUNNER_FILE_ENV_KEYS = (
    "SEMANTIC_DECISION_OVERRIDE",
    "SEMANTIC_MAPPING_OVERRIDE",
    "EXPLORE_PY_CONFIG_OVERRIDE",
    "NAV_CONFIG_OVERRIDE",
    "ROS_SETUP",
)

# ``benchmark_runner`` records an equivalent evaluator implementation digest in
# its own manifest.  The batch wrapper cannot import it cheaply or safely before
# ROS/conda setup, so retain the same file-level evidence here.  This prevents a
# --resume from silently carrying an old evaluator protocol across code changes.
_V3_EVALUATOR_PROTOCOL_FILES = (
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluate_interactive_nav_v3.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "benchmark_runner.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "benchmark_metrics.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "benchmark_policies.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "benchmark_interaction_adapter.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "benchmark_interaction_executor.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "benchmark_types.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "public_goal.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "restricted_gt_perception.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "ros_object_goal_adapter.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "ros_navigation_stall.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "goal_status.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "ros_policy_termination.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "evaluation" / "trusted_interaction_skill.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "force_interaction_runtime.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "force_interaction_bridge.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "container_scene_probe.py",
    REPO_ROOT / "scripts" / "InteractiveNav" / "interactive_nav_v3.py",
)

# The evaluator launches a real ROS navigation stack.  Its public interaction
# and completion contract therefore also depends on these runtime seams.  Keep
# this list explicit (rather than hashing the whole nested repository) so a
# resume is invalidated by a relevant ROS change without making every batch
# startup scan unrelated assets/tests.
_ROS_RUNTIME_PROTOCOL_FILES = (
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "nav_pkg" / "launch" / "molmospaces_nav_system.launch",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "nav_pkg" / "launch" / "nav.launch",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "nav_pkg" / "scripts" / "run_nav_ros_sim.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_decision_py_pkg" / "launch" / "semantic_decision.launch",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_decision_py_pkg" / "scripts" / "semantic_behavior_executor.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_decision_py_pkg" / "scripts" / "semantic_rule_decision_node.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_decision_py_pkg" / "scripts" / "semantic_decision_py_pkg" / "behavior_execution.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_decision_py_pkg" / "scripts" / "semantic_decision_py_pkg" / "post_interaction_traversal.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_decision_py_pkg" / "scripts" / "semantic_decision_py_pkg" / "rule_policy.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_decision_py_pkg" / "scripts" / "semantic_decision_py_pkg" / "model_policy.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_mapping_py_pkg" / "launch" / "semantic_mapping_py.launch",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_mapping_py_pkg" / "scripts" / "interaction_attribute_inference_node.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_mapping_py_pkg" / "scripts" / "semantic_mapping_py_pkg" / "interaction_result_contract.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_mapping_py_pkg" / "scripts" / "semantic_mapping_py_pkg" / "interaction_graph_store.py",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "explore_py_pkg" / "launch" / "explore_py.launch",
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "explore_py_pkg" / "scripts" / "explore_py_pkg" / "frontier_core.py",
)

# These fields are evaluator-owned facts.  Keep the batch wrapper's normal and
# recovery summaries aligned so an artifact recovered after an interrupted
# wrapper is indistinguishable from a normally collected evaluator outcome
# except for the explicit recovery provenance below.
EPISODE_RESULT_SUMMARY_FIELDS = (
    "case_id",
    "house_index",
    "domains",
    "recipe",
    "interaction_types",
    "interaction_requirement",
    "success",
    "task_success",
    "interaction_conditioned_success",
    "nav_success",
    "required_interaction_success",
    "sequence_success",
    "terminal_reason",
    "step_count",
    "navigation_step_count",
    "view_action_count",
    "interaction_action_count",
    "correct_interaction_action_count",
    "invalid_interaction_action_count",
    "navigation_path_length_m",
    "reference_path_length_m",
    "spl",
    "elapsed_seconds",
    "target_distance_m",
    "target_visibility_fraction",
    "scoring_eligible",
    "early_stop",
    "goal_definition_relaxed_success",
    "goal_definition_relaxed_instance_id",
    "goal_definition_relaxed_reason",
    "timing_summary",
)


@dataclass(frozen=True)
class EpisodePlan:
    """One independently runnable V3 episode task."""

    ordinal: int
    worker_id: int
    episode_index: int
    master_port: int
    task_dir: Path

    @property
    def ros_master_uri(self) -> str:
        return f"http://127.0.0.1:{self.master_port}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated V3 ROS object-goal benchmark episodes in a worker pool."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Explicit frozen V3 benchmark JSON (for example channel.json, container.json, or mixed.json).",
    )
    parser.add_argument("--episode-indices", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-master-port", type=int, default=12600)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--step-budget-mode",
        choices=("dynamic", "fixed"),
        default="dynamic",
        help=(
            "Forward the V3 evaluator budget mode to every isolated runner. "
            "Use fixed when --max-steps is a required per-episode applied-step budget."
        ),
    )
    parser.add_argument(
        "--semantic-attribute-request-timeout-s",
        type=float,
        default=30.0,
        help="M1 request timeout forwarded to each V3 runner; independent of M2/M3.",
    )
    parser.add_argument(
        "--ros-command-starvation-timeout-s",
        type=float,
        default=60.0,
        help="Continuous no-fresh-command wall time before the V3 evaluator stops.",
    )
    parser.add_argument(
        "--ros-observation-turn-multiplier",
        type=float,
        default=1.5,
        help="Maximum policy-observation turns relative to each applied-step budget.",
    )
    parser.add_argument(
        "--scene-timeout-s",
        type=float,
        default=7200.0,
        help="Wall-clock cap for one full runner lifecycle, including recorder drain/rendering.",
    )
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--runner-shell",
        choices=("bash", "zsh"),
        default="bash",
        help="Shell used to launch --runner; the maintained runner is Bash-compatible.",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Conda environment name or absolute prefix passed to the single-episode runner.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=None,
        help="MolmoSpaces Python executable passed to the single-episode runner.",
    )
    parser.add_argument(
        "--model-endpoints",
        nargs="+",
        metavar="URL",
        help=(
            "Local MLLM endpoint URLs. Logical workers are assigned URLs round-robin; "
            "each episode receives a derived dotenv with its assigned endpoint."
        ),
    )
    parser.add_argument(
        "--semantic-model-env-file",
        type=Path,
        help=(
            "Base semantic-model dotenv. With --model-endpoints it is copied into "
            "each attempt and only SEMANTIC_MODEL_ENDPOINT is overridden."
        ),
    )
    parser.add_argument(
        "--mujoco-egl-devices",
        nargs="+",
        metavar="DEVICE_ID",
        help="Optional EGL GPU IDs assigned to logical workers round-robin.",
    )
    parser.add_argument(
        "--no-recording",
        "--fast-eval",
        dest="fast_eval",
        action="store_true",
        help=(
            "Run evaluator-only mode: no recorder, no frame queue/ack barrier, no MP4/topdown requirement."
        ),
    )
    parser.add_argument(
        "--resource-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample batch-wide host CPU/RAM and GPU utilization/memory (default: enabled).",
    )
    parser.add_argument(
        "--resource-sample-interval-s",
        type=float,
        default=2.0,
        help="Resource telemetry sampling period in seconds.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or ``None`` when the file is not readable."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _stable_json_hash(payload: Any) -> str:
    """Return a deterministic digest for one JSON-serialisable protocol record."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_runner_setting(key: str) -> str:
    """Return one output-independent V3 runner setting after environment defaults."""

    return str(os.environ.get(key, _RUNNER_ENV_DEFAULTS[key]))


def _file_digest_record(path: Path) -> dict[str, str] | None:
    digest = sha256_file(path)
    if digest is None:
        return None
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        relative = path.resolve()
    return {"path": str(relative), "sha256": digest}


def _effective_runtime_contract(args: argparse.Namespace) -> tuple[str, str | None]:
    """Resolve the interpreter values the shell runner will actually use.

    The wrapper activates ``CONDA_ENV`` before choosing ``PYTHON_BIN``.  A
    resume signature must therefore include inherited values and the wrapper's
    absolute default, rather than only explicit CLI flags.
    """

    requested_conda = getattr(args, "conda_env", None)
    conda_value = (
        requested_conda
        or infer_conda_prefix()
        or os.environ.get("CONDA_ENV")
        or "/home/ldl/conda_envs/mlspaces"
    )
    conda_path = Path(str(conda_value)).expanduser()
    runtime_conda = str(conda_path.resolve()) if conda_path.is_absolute() else str(conda_path)

    requested_python = getattr(args, "python_bin", None)
    python_value = requested_python or os.environ.get("PYTHON_BIN")
    if python_value is None and conda_path.is_absolute():
        candidate = conda_path / "bin" / "python"
        if candidate.is_file():
            python_value = str(candidate)
    runtime_python = None
    if python_value:
        python_path = Path(str(python_value)).expanduser()
        runtime_python = str(python_path.resolve()) if python_path.is_absolute() else str(python_path)
    return runtime_conda, runtime_python


def _protocol_files_sha256(files: tuple[Path, ...]) -> str | None:
    """Digest one explicit implementation contract without importing ROS."""

    records: list[dict[str, str]] = []
    for path in files:
        record = _file_digest_record(path)
        if record is None:
            return None
        records.append(record)
    return _stable_json_hash(records)


def _v3_evaluator_protocol_sha256() -> str | None:
    """Digest evaluator-only implementation files for the invocation contract."""

    return _protocol_files_sha256(_V3_EVALUATOR_PROTOCOL_FILES)


def _ros_runtime_protocol_sha256() -> str | None:
    """Digest the explicit ROS runtime seam launched by the V3 runner."""

    return _protocol_files_sha256(_ROS_RUNTIME_PROTOCOL_FILES)


def planned_invocation_payload(
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Build the complete output-independent identity for one planned episode.

    This is the single seam for batch resume and artifact recovery.  It contains
    every evaluator or launch input that can change formal V3 semantics or the
    outer batch completion contract, but deliberately excludes output paths,
    worker/port assignment, timestamps, telemetry, and other scheduling-only
    details.
    """

    benchmark_hash = sha256_file(Path(args.benchmark))
    runner_record = _file_digest_record(Path(args.runner))
    evaluator_digest = _v3_evaluator_protocol_sha256()
    ros_runtime_digest = _ros_runtime_protocol_sha256()
    if (
        not benchmark_hash
        or runner_record is None
        or not evaluator_digest
        or not ros_runtime_digest
    ):
        return None
    runner_settings = {
        key: _resolved_runner_setting(key)
        for key in sorted(_RUNNER_ENV_DEFAULTS)
    }
    file_settings: dict[str, dict[str, str]] = {}
    for key in _RUNNER_FILE_ENV_KEYS:
        record = _file_digest_record(Path(runner_settings[key]))
        if record is None:
            return None
        file_settings[key] = record
    semantic_model_env: dict[str, str] | None = None
    semantic_model_env_file = getattr(args, "semantic_model_env_file", None)
    if semantic_model_env_file is None:
        inherited_model_env = os.environ.get("SEMANTIC_MODEL_ENV_FILE")
        semantic_model_env_file = (
            Path(inherited_model_env)
            if inherited_model_env
            else REPO_ROOT / ".env"
        )
    if semantic_model_env_file is not None:
        record = _file_digest_record(Path(semantic_model_env_file))
        if record is None:
            return None
        semantic_model_env = record
    runtime_conda, runtime_python = _effective_runtime_contract(args)
    return {
        "schema_version": PLANNED_INVOCATION_SCHEMA_VERSION,
        "episode_index": int(plan.episode_index),
        "benchmark_sha256": benchmark_hash,
        "runner": runner_record,
        "evaluator_protocol_sha256": evaluator_digest,
        "ros_runtime_protocol_sha256": ros_runtime_digest,
        "runner_shell": str(args.runner_shell),
        "fast_eval": bool(args.fast_eval),
        "scene_timeout_s": float(args.scene_timeout_s),
        "max_steps": int(args.max_steps),
        "step_budget_mode": str(args.step_budget_mode),
        "semantic_attribute_request_timeout_s": float(
            args.semantic_attribute_request_timeout_s
        ),
        "ros_command_starvation_timeout_s": float(
            args.ros_command_starvation_timeout_s
        ),
        "ros_observation_turn_multiplier": float(
            args.ros_observation_turn_multiplier
        ),
        "model_endpoint": assigned_model_endpoint(plan.worker_id, args),
        "semantic_model_env": semantic_model_env,
        "runtime_python": runtime_python,
        "runtime_conda": runtime_conda,
        "runner_settings": runner_settings,
        "runner_file_settings": file_settings,
    }


def planned_invocation_signature(
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]] | None:
    """Return one exact planned-invocation signature and its audit payload."""

    payload = planned_invocation_payload(plan, args)
    if payload is None:
        return None
    return _stable_json_hash(payload), payload


def planned_invocation_record(
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    signed = planned_invocation_signature(plan, args)
    if signed is None:
        return None
    signature, payload = signed
    return {
        "planned_invocation_signature": signature,
        "planned_invocation": payload,
    }


def _has_matching_planned_invocation(
    record: Mapping[str, Any],
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> bool:
    """Validate a persisted summary/attempt against today's exact plan."""

    expected = planned_invocation_record(plan, args)
    if expected is None:
        return False
    return (
        record.get("planned_invocation_signature")
        == expected["planned_invocation_signature"]
        and record.get("planned_invocation") == expected["planned_invocation"]
    )


def _write_planned_invocation(
    attempt_dir: Path,
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Persist the planned signature before the subprocess can create artifacts."""

    record = planned_invocation_record(plan, args)
    if record is None:
        raise RuntimeError("unable to hash planned V3 invocation inputs")
    atomic_json(attempt_dir / PLANNED_INVOCATION_FILENAME, record)
    return record


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def assigned_model_endpoint(worker_id: int, args: argparse.Namespace) -> str | None:
    endpoints = args.model_endpoints
    if not endpoints:
        return None
    return str(endpoints[worker_id % len(endpoints)])


def assigned_mujoco_egl_device(worker_id: int, args: argparse.Namespace) -> str | None:
    devices = args.mujoco_egl_devices
    if not devices:
        return None
    return str(devices[worker_id % len(devices)])


def derive_episode_model_env_file(
    attempt_dir: Path,
    source_env_file: Path,
    endpoint: str,
) -> Path:
    """Copy the local model config and append an episode-specific endpoint.

    ``python-dotenv`` loads this file with ``override=True`` in the semantic
    stack, so the final setting wins without replacing the model name, timeout,
    or any local-only configuration in the source file.
    """

    source_text = source_env_file.read_text(encoding="utf-8")
    if source_text and not source_text.endswith("\n"):
        source_text += "\n"
    derived_path = attempt_dir / "semantic_model.env"
    derived_text = (
        source_text
        + "\n# Generated by run_interactive_nav_v3_ros_eval_batch.py; do not edit mid-run.\n"
        + f"SEMANTIC_MODEL_ENDPOINT={shlex.quote(endpoint)}\n"
    )
    temporary = derived_path.with_name(f".{derived_path.name}.{os.getpid()}.tmp")
    temporary.write_text(derived_text, encoding="utf-8")
    temporary.replace(derived_path)
    return derived_path


def host_memory_usage() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        fields = raw_value.strip().split()
        if not fields:
            continue
        try:
            values[key] = float(fields[0]) / 1024.0
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    result: dict[str, float] = {}
    if total is not None and available is not None:
        result["host_mem_used_mb"] = total - available
        result["host_mem_available_mb"] = available
    if swap_total is not None and swap_free is not None:
        result["host_swap_used_mb"] = swap_total - swap_free
    return result


def host_cpu_counters() -> tuple[float, float] | None:
    """Return total and idle Linux CPU jiffies, or ``None`` when unavailable."""

    try:
        first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not first or first[0] != "cpu":
        return None
    try:
        counters = [float(value) for value in first[1:]]
    except ValueError:
        return None
    if not counters:
        return None
    idle = counters[3] if len(counters) > 3 else 0.0
    iowait = counters[4] if len(counters) > 4 else 0.0
    return sum(counters), idle + iowait


def gpu_usage() -> list[dict[str, float | int]]:
    """Read GPU utilization without requiring a Python CUDA dependency."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, float | int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "memory_used_mb": float(fields[1]),
                    "memory_total_mb": float(fields[2]),
                    "utilization_gpu_percent": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return rows


class BatchResourceTelemetry:
    """One lightweight sampler for the whole batch, not one ``nvidia-smi`` per worker."""

    fieldnames = (
        "wall_time",
        "elapsed_sec",
        "active_episode_count",
        "active_episode_indices",
        "host_cpu_percent",
        "host_mem_used_mb",
        "host_mem_available_mb",
        "host_swap_used_mb",
        "gpu_metrics_json",
    )

    def __init__(self, output_dir: Path, interval_s: float) -> None:
        self.path = output_dir / "resource_telemetry.csv"
        self.interval_s = max(0.25, float(interval_s))
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active: dict[int, int] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="v3-batch-resource-monitor", daemon=True)
        self._thread.start()

    def mark_started(self, plan: EpisodePlan) -> None:
        with self._lock:
            self._active[plan.ordinal] = plan.episode_index

    def mark_finished(self, plan: EpisodePlan) -> None:
        with self._lock:
            self._active.pop(plan.ordinal, None)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s + 1.0))
        return summarize_resource_telemetry(self.path)

    def _run(self) -> None:
        previous_cpu = host_cpu_counters()
        # ``--resume`` may start the wrapper after previous episodes have
        # already produced useful peak telemetry.  Append rather than truncate
        # that evidence; the summary naturally aggregates all samples.
        append = self.path.is_file() and self.path.stat().st_size > 0
        with self.path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not append:
                writer.writeheader()
            while True:
                current_cpu = host_cpu_counters()
                host_cpu_percent: float | None = None
                if previous_cpu is not None and current_cpu is not None:
                    total_delta = current_cpu[0] - previous_cpu[0]
                    idle_delta = current_cpu[1] - previous_cpu[1]
                    if total_delta > 0.0:
                        host_cpu_percent = 100.0 * max(0.0, min(1.0, 1.0 - idle_delta / total_delta))
                previous_cpu = current_cpu
                with self._lock:
                    active_indexes = sorted(self._active.values())
                row: dict[str, Any] = {
                    "wall_time": time.time(),
                    "elapsed_sec": time.monotonic() - self._started,
                    "active_episode_count": len(active_indexes),
                    "active_episode_indices": json.dumps(active_indexes),
                    "host_cpu_percent": host_cpu_percent,
                    "gpu_metrics_json": json.dumps(gpu_usage(), separators=(",", ":")),
                    **host_memory_usage(),
                }
                writer.writerow(row)
                handle.flush()
                if self._stop.wait(self.interval_s):
                    break


def summarize_resource_telemetry(path: Path) -> dict[str, Any]:
    """Return compact peak metrics for the batch summary while retaining raw CSV."""

    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError:
        rows = []
    if not rows:
        return {"resource_telemetry_path": str(path), "resource_sample_count": 0}

    def numeric_values(key: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            raw = row.get(key)
            if raw in {None, ""}:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        return values

    gpu_peaks: dict[int, dict[str, float | int]] = {}
    for row in rows:
        try:
            metrics = json.loads(row.get("gpu_metrics_json") or "[]")
        except ValueError:
            metrics = []
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            try:
                index = int(metric["index"])
                memory_used = float(metric["memory_used_mb"])
                memory_total = float(metric["memory_total_mb"])
                utilization = float(metric["utilization_gpu_percent"])
            except (KeyError, TypeError, ValueError):
                continue
            peak = gpu_peaks.setdefault(
                index,
                {
                    "index": index,
                    "peak_memory_used_mb": 0.0,
                    "memory_total_mb": memory_total,
                    "peak_utilization_gpu_percent": 0.0,
                },
            )
            peak["peak_memory_used_mb"] = max(float(peak["peak_memory_used_mb"]), memory_used)
            peak["memory_total_mb"] = memory_total
            peak["peak_utilization_gpu_percent"] = max(
                float(peak["peak_utilization_gpu_percent"]), utilization
            )

    cpu = numeric_values("host_cpu_percent")
    memory_used = numeric_values("host_mem_used_mb")
    memory_available = numeric_values("host_mem_available_mb")
    swap_used = numeric_values("host_swap_used_mb")
    active = numeric_values("active_episode_count")
    return {
        "resource_telemetry_path": str(path),
        "resource_sample_count": len(rows),
        "peak_active_episode_count": max(active) if active else None,
        "mean_host_cpu_percent": sum(cpu) / len(cpu) if cpu else None,
        "peak_host_cpu_percent": max(cpu) if cpu else None,
        "peak_host_mem_used_mb": max(memory_used) if memory_used else None,
        "min_host_mem_available_mb": min(memory_available) if memory_available else None,
        "peak_host_swap_used_mb": max(swap_used) if swap_used else None,
        "gpu_peaks": [gpu_peaks[index] for index in sorted(gpu_peaks)],
    }


def terminate_group(process: subprocess.Popen[bytes], grace_s: float = 30.0) -> None:
    """Stop the runner and all children without leaving a ROS master behind."""

    if process.poll() is not None:
        return
    for signum, wait_s in (
        (signal.SIGINT, grace_s),
        (signal.SIGTERM, 10.0),
        (signal.SIGKILL, 2.0),
    ):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def infer_conda_prefix() -> str | None:
    """Infer an activated prefix only when this wrapper runs inside a conda env."""

    active = os.environ.get("CONDA_PREFIX")
    if active and (Path(active) / "bin" / "python").is_file():
        return active
    executable = Path(sys.executable).resolve()
    candidate = executable.parent.parent
    if executable.parent.name == "bin" and (candidate / "conda-meta").is_dir():
        return str(candidate)
    return None


def artifact_paths(attempt_dir: Path, episode_index: int) -> dict[str, Path | None]:
    episode_results = sorted(
        (attempt_dir / "eval" / "episodes").glob(
            f"{episode_index:04d}_*/episode_result.json"
        )
    )
    result_path = episode_results[0] if len(episode_results) == 1 else None
    episode_dir = result_path.parent if result_path is not None else None
    return {
        "episode_result": result_path,
        # The maintained V3 runner records raw artifacts under ``debug/`` and
        # builds the six-panel MP4 offline into the attempt-level videos dir.
        "six_panel_video": attempt_dir / "videos" / "overview_6panel.mp4",
        "topdown": None if episode_dir is None else episode_dir / "episode_topdown.png",
    }


def publish_file(source: Path, destination: Path) -> None:
    """Expose a final artifact directly below the episode directory.

    The attempt tree remains the authoritative evidence location.  The shallow
    path is a hard-link alias when possible, so publishing a large MP4 does not
    double disk usage; ``copy2`` is the portable fallback.
    """

    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_episode_artifacts(
    task_dir: Path,
    paths: Mapping[str, Path | None],
    *,
    recording_required: bool,
) -> dict[str, str | None]:
    """Publish image/video aliases at ``output_dir/episode_xxxx/``."""

    published: dict[str, str | None] = {
        "six_panel_video": None,
        "topdown": None,
        "topdown_metadata": None,
        "errors": None,
    }
    errors: list[str] = []
    specs = (
        ("six_panel_video", paths.get("six_panel_video"), task_dir / "overview_6panel.mp4"),
        ("topdown", paths.get("topdown"), task_dir / "episode_topdown.png"),
        (
            "topdown_metadata",
            None if paths.get("topdown") is None else paths["topdown"].with_suffix(".json"),
            task_dir / "episode_topdown.json",
        ),
    )
    for key, source, destination in specs:
        if source is None or not Path(source).is_file():
            continue
        try:
            publish_file(Path(source), destination)
        except OSError as exc:
            errors.append(f"{source} -> {destination}: {type(exc).__name__}: {exc}")
            continue
        published[key] = str(destination)
    if errors:
        published["errors"] = json.dumps(errors, ensure_ascii=False)
    if recording_required and published["six_panel_video"] is None:
        errors.append("recording artifact was not published")
    return published


def load_episode_result(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {}, None
    document = read_json(path)
    result = document.get("result")
    if isinstance(result, dict):
        status = document.get("status")
        return result, str(status) if status is not None else None
    status = document.get("status")
    return document, str(status) if status is not None else None


def video_is_valid(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def episode_result_is_complete(path: Path | None) -> bool:
    result, document_status = load_episode_result(path)
    return document_status == "complete" and result.get("status") == "complete"


def existing_completed_summary(
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Return a completed summary only when it matches this exact V3 plan."""

    task_dir = plan.task_dir
    summary = read_json(task_dir / "batch_task_summary.json")
    if not summary.get("completed"):
        return None
    if not _has_matching_planned_invocation(summary, plan, args):
        return None
    # Do not let a recording-mode resume silently satisfy a fast-eval request,
    # or vice versa: their evidence contracts are intentionally different.
    if bool(summary.get("fast_eval", False)) != bool(args.fast_eval):
        return None
    episode_result = summary.get("episode_result_path")
    if not isinstance(episode_result, str) or not episode_result_is_complete(Path(episode_result)):
        return None
    if not args.fast_eval:
        video = summary.get("six_panel_video")
        if not isinstance(video, str) or not video_is_valid(Path(video)):
            return None
    return summary


def next_attempt_dir(task_dir: Path) -> Path:
    existing_numbers: list[int] = []
    for candidate in task_dir.glob("attempt_*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("attempt_")
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    return task_dir / f"attempt_{max(existing_numbers, default=0) + 1:03d}"


def plan_to_dict(plan: EpisodePlan, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ordinal": plan.ordinal,
        "worker_id": plan.worker_id,
        "episode_index": plan.episode_index,
        "ros_master_uri": plan.ros_master_uri,
        "output_dir": str(plan.task_dir),
        "model_endpoint": assigned_model_endpoint(plan.worker_id, args),
        "mujoco_egl_device": assigned_mujoco_egl_device(plan.worker_id, args),
    }


def append_task_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def encode_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def make_environment(args: argparse.Namespace, plan: EpisodePlan, attempt_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    cache_root = args.output_dir / "_runtime_cache"
    temporary_root = attempt_dir / "tmp"
    ros_home = attempt_dir / "ros_home"
    ros_log_dir = ros_home / "log"
    scene_mirror = attempt_dir / "scene_mirror"
    hf_cache = cache_root / "hf"
    torch_cache = cache_root / "torch"
    cuda_cache = cache_root / "cuda"
    for path in (
        cache_root,
        temporary_root,
        attempt_dir / "mplconfig",
        ros_home,
        ros_log_dir,
        scene_mirror,
        hf_cache,
        torch_cache,
        cuda_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "MAX_STEPS": str(args.max_steps),
            "STEP_BUDGET_MODE": str(args.step_budget_mode),
            "SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S": str(
                args.semantic_attribute_request_timeout_s
            ),
            "ROS_COMMAND_STARVATION_TIMEOUT_S": str(
                args.ros_command_starvation_timeout_s
            ),
            "ROS_OBSERVATION_TURN_MULTIPLIER": str(
                args.ros_observation_turn_multiplier
            ),
            "BENCHMARK": str(args.benchmark),
            "ROS_MASTER_URI": plan.ros_master_uri,
            "MPLCONFIGDIR": str(attempt_dir / "mplconfig"),
            "TMPDIR": str(temporary_root),
            "XDG_CACHE_HOME": str(cache_root),
            # Keep all per-episode ROS logs and transient compiler/model caches
            # off the nearly full root volume and prevent concurrent workers
            # from sharing one ROS_HOME.
            "ROS_HOME": str(ros_home),
            "ROS_LOG_DIR": str(ros_log_dir),
            "INTERACTIVE_NAV_SCENE_MIRROR": str(scene_mirror),
            "HF_HOME": str(hf_cache),
            "TORCH_HOME": str(torch_cache),
            "CUDA_CACHE_PATH": str(cuda_cache),
            "PYTHONUNBUFFERED": "1",
            "FAST_EVAL": "true" if args.fast_eval else "false",
            "RECORD_HEAD_CAMERA": "false" if args.fast_eval else environment.get("RECORD_HEAD_CAMERA", "false"),
        }
    )
    conda_env, python_bin = _effective_runtime_contract(args)
    if conda_env:
        environment["CONDA_ENV"] = conda_env
    if python_bin:
        environment["PYTHON_BIN"] = python_bin
    model_endpoint = assigned_model_endpoint(plan.worker_id, args)
    if model_endpoint is not None:
        assert args.semantic_model_env_file is not None
        environment["SEMANTIC_MODEL_ENV_FILE"] = str(
            derive_episode_model_env_file(attempt_dir, args.semantic_model_env_file, model_endpoint)
        )
    elif args.semantic_model_env_file is not None:
        environment["SEMANTIC_MODEL_ENV_FILE"] = str(args.semantic_model_env_file)
    mujoco_egl_device = assigned_mujoco_egl_device(plan.worker_id, args)
    if mujoco_egl_device is not None:
        environment["MUJOCO_EGL_DEVICE_ID"] = mujoco_egl_device
    return environment


def base_summary(
    plan: EpisodePlan,
    args: argparse.Namespace,
    *,
    planned_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "episode_index": plan.episode_index,
        "worker_id": plan.worker_id,
        "ros_master_uri": plan.ros_master_uri,
        "output_dir": str(plan.task_dir),
        "runner": str(args.runner),
        "benchmark": str(args.benchmark),
        "scene_timeout_s": float(args.scene_timeout_s),
        "max_steps": args.max_steps,
        "step_budget_mode": str(args.step_budget_mode),
        "semantic_attribute_request_timeout_s": float(
            args.semantic_attribute_request_timeout_s
        ),
        "ros_command_starvation_timeout_s": float(
            args.ros_command_starvation_timeout_s
        ),
        "ros_observation_turn_multiplier": float(
            args.ros_observation_turn_multiplier
        ),
        "fast_eval": bool(args.fast_eval),
        "recording_enabled": not bool(args.fast_eval),
        "model_endpoint": assigned_model_endpoint(plan.worker_id, args),
        "mujoco_egl_device": assigned_mujoco_egl_device(plan.worker_id, args),
    }
    # A normal run writes this record before it launches the subprocess.  Keep
    # that original launch identity in its summary instead of recomputing after
    # a potentially long evaluation window.
    record = dict(planned_record) if planned_record is not None else planned_invocation_record(plan, args)
    if record is not None:
        result.update(record)
    return result


def copy_episode_result_fields(summary: dict[str, Any], episode_result: dict[str, Any]) -> None:
    for key in EPISODE_RESULT_SUMMARY_FIELDS:
        if key in episode_result:
            summary[key] = episode_result[key]


def latest_attempt_dir(task_dir: Path) -> Path | None:
    """Return the only attempt eligible for recovery: the highest numeric one."""

    attempts: list[tuple[int, Path]] = []
    for candidate in task_dir.glob("attempt_*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("attempt_")
        if not suffix.isdigit():
            # A hand-made/unknown attempt name makes recovery provenance
            # ambiguous.  Leave it for an explicit rerun instead.
            return None
        attempts.append((int(suffix), candidate))
    if not attempts:
        return None
    return max(attempts, key=lambda item: item[0])[1]


def matching_current_attempt_artifact(
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any], str, str] | None:
    """Validate a sole current-attempt result before reconstructing a summary.

    A batch summary is a wrapper-side artifact.  If a process dies in the tiny
    interval after the single-episode runner wrote its evaluator result, that
    result is still usable only when it can be tied to the *latest* attempt,
    exact planned index, frozen benchmark and evaluator run signature.  Older
    attempts are intentionally ignored; a newer empty attempt is never allowed
    to fall back to stale evidence.
    """

    attempt_dir = latest_attempt_dir(plan.task_dir)
    if attempt_dir is None:
        return None
    attempt_invocation = read_json(attempt_dir / PLANNED_INVOCATION_FILENAME)
    if not _has_matching_planned_invocation(attempt_invocation, plan, args):
        return None
    task_log = plan.task_dir / "batch_task.log"
    try:
        task_log_text = task_log.read_text(encoding="utf-8")
    except OSError:
        return None
    if (
        f"attempt={attempt_dir.name}" not in task_log_text
        or "start command=" not in task_log_text
        or str(attempt_dir) not in task_log_text
    ):
        return None

    # One isolated wrapper invocation must produce exactly one evaluator trace.
    # Do not guess when a stale or partially copied attempt has multiple files.
    result_paths = sorted((attempt_dir / "eval" / "episodes").glob("*/episode_result.json"))
    if len(result_paths) != 1:
        return None
    result_path = result_paths[0]
    if not result_path.parent.name.startswith(f"{plan.episode_index:04d}_"):
        return None

    document = read_json(result_path)
    episode_result, document_status = load_episode_result(result_path)
    if document_status != "complete" or episode_result.get("status") != "complete":
        return None
    if episode_result.get("episode_index") != plan.episode_index:
        return None
    run_signature = document.get("run_signature")
    if not isinstance(run_signature, str) or not run_signature:
        return None

    run_manifest = read_json(attempt_dir / "eval" / "run_manifest.json")
    if run_manifest.get("run_signature") != run_signature:
        return None
    if run_manifest.get("episode_indices") != [plan.episode_index]:
        return None
    evaluation_config = run_manifest.get("evaluation_config")
    if not isinstance(evaluation_config, dict):
        return None
    if evaluation_config.get("episode_indices") != [plan.episode_index]:
        return None
    if evaluation_config.get("max_steps") != args.max_steps:
        return None
    if evaluation_config.get("step_budget_mode") != args.step_budget_mode:
        return None
    if evaluation_config.get("ros_command_starvation_timeout_s") != float(
        args.ros_command_starvation_timeout_s
    ):
        return None
    if evaluation_config.get("ros_observation_turn_multiplier") != float(
        args.ros_observation_turn_multiplier
    ):
        return None
    benchmark_hash = sha256_file(args.benchmark)
    if not benchmark_hash or run_manifest.get("benchmark_sha256") != benchmark_hash:
        return None

    paths = artifact_paths(attempt_dir, plan.episode_index)
    if paths["episode_result"] != result_path:
        return None
    if not args.fast_eval and not video_is_valid(paths["six_panel_video"]):
        return None
    return attempt_dir, result_path, episode_result, document_status, run_signature


def recover_missing_task_summary(plan: EpisodePlan, args: argparse.Namespace) -> dict[str, Any] | None:
    """Atomically reconstruct a missing wrapper summary from validated evidence.

    This is deliberately narrower than evaluator resume: it never reuses an
    old attempt, guesses between multiple traces, or trusts a result whose run
    signature/configuration is not exactly the planned V3 invocation.
    """

    task_summary = plan.task_dir / "batch_task_summary.json"
    if task_summary.exists():
        return None
    evidence = matching_current_attempt_artifact(plan, args)
    if evidence is None:
        return None
    attempt_dir, result_path, episode_result, document_status, run_signature = evidence
    paths = artifact_paths(attempt_dir, plan.episode_index)
    six_panel = paths["six_panel_video"]
    topdown = paths["topdown"]
    published = publish_episode_artifacts(
        plan.task_dir,
        paths,
        recording_required=not args.fast_eval,
    )
    recovered_at = utc_now()
    result = base_summary(
        plan,
        args,
        planned_record=read_json(attempt_dir / PLANNED_INVOCATION_FILENAME),
    )
    result.update(
        {
            "attempt_dir": str(attempt_dir),
            "attempt": attempt_dir.name,
            "runner_log": str(attempt_dir / "runner.log"),
            "started_at": None,
            "finished_at": recovered_at,
            "elapsed_sec": None,
            "timed_out": False,
            # The shell's exit status was lost with the wrapper process.  Do
            # not fabricate it; completion rests on the signed evaluator
            # artifact validated above.
            "runner_exit_code": None,
            "exit_code": None,
            "completed": True,
            "resumed": False,
            "episode_result_path": str(result_path),
            "episode_document_status": document_status,
            "episode_status": episode_result.get("status"),
            "run_signature": run_signature,
            "artifact_validation_mode": (
                "episode_result_only_fast_eval" if args.fast_eval else "episode_result_and_six_panel_video"
            ),
            "six_panel_video": (
                None
                if args.fast_eval
                else published.get("six_panel_video") or str(six_panel)
            ),
            "six_panel_video_bytes": (
                0 if args.fast_eval or not video_is_valid(six_panel) else six_panel.stat().st_size
            ),
            "topdown": (
                None
                if args.fast_eval or topdown is None
                else published.get("topdown") or str(topdown)
            ),
            "topdown_exists": (
                False
                if args.fast_eval
                else topdown is not None and topdown.is_file() and topdown.stat().st_size > 0
            ),
            "error": episode_result.get("error"),
            "recovered_batch_task_summary": True,
            "recovery_source": "complete_current_attempt_episode_result",
            "recovery_reason": "missing_batch_task_summary",
            "recovered_at": recovered_at,
        }
    )
    copy_episode_result_fields(result, episode_result)
    atomic_json(attempt_dir / "wrapper_summary.json", result)
    atomic_json(task_summary, result)
    append_task_log(
        plan.task_dir / "batch_task.log",
        f"[{recovered_at}] attempt={attempt_dir.name} recovered_missing_batch_task_summary "
        f"run_signature={run_signature}",
    )
    return result


def recover_unreported_task_summaries(
    plans: list[EpisodePlan],
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> None:
    """Add only formally validated wrapper recoveries to the live result list."""

    reported_indices = {
        row.get("episode_index")
        for row in results
        if isinstance(row.get("episode_index"), int)
    }
    for plan in plans:
        if plan.episode_index in reported_indices:
            continue
        recovered = recover_missing_task_summary(plan, args)
        if recovered is not None:
            results.append(recovered)
            reported_indices.add(plan.episode_index)


def run_episode(
    plan: EpisodePlan,
    args: argparse.Namespace,
    telemetry: BatchResourceTelemetry | None = None,
) -> dict[str, Any]:
    """Run one episode and leave a self-contained summary in its task directory."""

    task_dir = plan.task_dir
    if args.dry_run:
        result = base_summary(plan, args)
        result.update(
            {
                "command": [args.runner_shell, str(args.runner), str(task_dir / "attempt_001"), str(plan.episode_index)],
                "completed": False,
                "dry_run": True,
                "resumed": False,
                "fast_eval": bool(args.fast_eval),
                "model_endpoint": assigned_model_endpoint(plan.worker_id, args),
                "mujoco_egl_device": assigned_mujoco_egl_device(plan.worker_id, args),
            }
        )
        return result

    recovered = recover_missing_task_summary(plan, args) if args.resume else None
    if recovered is not None:
        return recovered

    previous = existing_completed_summary(plan, args) if args.resume else None
    if previous is not None:
        result = dict(previous)
        result.update({"worker_id": plan.worker_id, "resumed": True, "resume_skipped": True})
        return result

    if task_dir.exists() and not args.resume and (task_dir / "batch_task_summary.json").exists():
        result = base_summary(plan, args)
        result.update(
            {
                "completed": False,
                "runner_exit_code": 2,
                "exit_code": 2,
                "error": "existing task output; use --resume or select a new --output-dir",
                "resumed": False,
            }
        )
        return result

    task_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = next_attempt_dir(task_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    planned_record = _write_planned_invocation(attempt_dir, plan, args)
    environment = make_environment(args, plan, attempt_dir)
    command = [args.runner_shell, str(args.runner), str(attempt_dir), str(plan.episode_index)]
    task_log = task_dir / "batch_task.log"
    runner_log = attempt_dir / "runner.log"
    started_wall_time = utc_now()
    started = time.monotonic()
    append_task_log(
        task_log,
        f"[{started_wall_time}] attempt={attempt_dir.name} "
        f"planned_invocation_signature={planned_record['planned_invocation_signature']} "
        f"start command={shlex.join(command)}",
    )

    timed_out = False
    exception_text: str | None = None
    if telemetry is not None:
        telemetry.mark_started(plan)
    try:
        with runner_log.open("wb") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                runner_exit_code = process.wait(timeout=args.scene_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_group(process)
                runner_exit_code = 124
    except (OSError, subprocess.SubprocessError) as exc:
        runner_exit_code = 127
        exception_text = f"{type(exc).__name__}: {exc}"
    finally:
        if telemetry is not None:
            telemetry.mark_finished(plan)

    elapsed_sec = time.monotonic() - started
    paths = artifact_paths(attempt_dir, plan.episode_index)
    episode_document = read_json(paths["episode_result"]) if paths["episode_result"] else {}
    episode_result, document_status = load_episode_result(paths["episode_result"])
    six_panel = paths["six_panel_video"]
    topdown = paths["topdown"]
    published = publish_episode_artifacts(
        plan.task_dir,
        paths,
        recording_required=not args.fast_eval,
    )
    episode_result_complete = (
        document_status == "complete" and episode_result.get("status") == "complete"
    )
    completed = bool(
        runner_exit_code == 0
        and episode_result_complete
        and (args.fast_eval or video_is_valid(six_panel))
    )
    result = base_summary(plan, args, planned_record=planned_record)
    result.update(
        {
            "attempt_dir": str(attempt_dir),
            "attempt": attempt_dir.name,
            "runner_log": str(runner_log),
            "command": command,
            "started_at": started_wall_time,
            "finished_at": utc_now(),
            "elapsed_sec": elapsed_sec,
            "timed_out": timed_out,
            "runner_exit_code": runner_exit_code,
            "exit_code": runner_exit_code,
            "completed": completed,
            "resumed": False,
            "episode_result_path": None if paths["episode_result"] is None else str(paths["episode_result"]),
            "episode_document_status": document_status,
            "episode_status": episode_result.get("status"),
            "run_signature": episode_document.get("run_signature"),
            "artifact_validation_mode": (
                "episode_result_only_fast_eval" if args.fast_eval else "episode_result_and_six_panel_video"
            ),
            "six_panel_video": (
                None
                if args.fast_eval
                else published.get("six_panel_video") or str(six_panel)
            ),
            "six_panel_video_bytes": (
                0 if args.fast_eval or not video_is_valid(six_panel) else six_panel.stat().st_size
            ),
            "topdown": (
                None
                if args.fast_eval or topdown is None
                else published.get("topdown") or str(topdown)
            ),
            "topdown_exists": (
                False
                if args.fast_eval
                else bool(
                    published.get("topdown")
                    or (topdown is not None and topdown.is_file() and topdown.stat().st_size > 0)
                )
            ),
            "artifact_publish_errors": published.get("errors"),
            "semantic_model_env_file": environment.get("SEMANTIC_MODEL_ENV_FILE"),
            "error": exception_text or episode_result.get("error"),
        }
    )
    copy_episode_result_fields(result, episode_result)
    atomic_json(attempt_dir / "wrapper_summary.json", result)
    atomic_json(task_dir / "batch_task_summary.json", result)
    append_task_log(
        task_log,
        f"[{result['finished_at']}] attempt={attempt_dir.name} exit_code={runner_exit_code} "
        f"completed={completed} elapsed_sec={elapsed_sec:.1f}",
    )
    return result


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def missing_plan_summary(plan: EpisodePlan, args: argparse.Namespace) -> dict[str, Any]:
    """Represent a planned episode whose worker never reported a wrapper result."""

    result = base_summary(plan, args)
    result.update(
        {
            "completed": False,
            "report_status": "missing",
            "worker_result_reported": False,
            "failure_kind": "missing_episode_report",
            "incomplete_reason": "planned_episode_not_reported",
            "episode_status": "not_reported",
            "terminal_reason": "missing_episode_report",
            "scoring_eligible": False,
            # Keep every score-like outcome explicitly false: no evaluator
            # result is evidence of neither task success nor navigation success.
            "success": False,
            "task_success": False,
            "nav_success": False,
            "required_interaction_success": False,
            "sequence_success": False,
            "error": "no worker result was reported for this planned episode",
        }
    )
    return result


def write_summary(
    args: argparse.Namespace,
    plans: list[EpisodePlan],
    results: list[dict[str, Any]],
    resource_summary: dict[str, Any] | None = None,
    *,
    recover_missing_task_summaries: bool = False,
) -> dict[str, Any]:
    if recover_missing_task_summaries:
        # A worker can be interrupted after the evaluator atomically commits
        # its result but before this wrapper writes batch_task_summary.json.
        # Do this only after the worker pool has joined: during an intermediate
        # progress write another shell may still be validating/tearing down.
        recover_unreported_task_summaries(plans, args, results)
    ordered_results = sorted(results, key=lambda row: int(row.get("episode_index", -1)))
    planned_indices = {plan.episode_index for plan in plans}
    reported_planned_indices = {
        int(row["episode_index"])
        for row in ordered_results
        if isinstance(row.get("episode_index"), int)
        and int(row["episode_index"]) in planned_indices
    }
    missing_rows = [
        missing_plan_summary(plan, args)
        for plan in plans
        if plan.episode_index not in reported_planned_indices
    ]
    episode_rows = sorted(
        [
            {
                **row,
                "report_status": row.get("report_status", "reported"),
                "worker_result_reported": True,
            }
            for row in ordered_results
        ]
        + missing_rows,
        key=lambda row: int(row.get("episode_index", -1)),
    )
    completed = [row for row in ordered_results if bool(row.get("completed"))]
    failures = [row for row in ordered_results if not bool(row.get("completed"))]
    aggregate = {
        "planned_episode_count": len(plans),
        "reported_episode_count": len(ordered_results),
        "reported_planned_episode_count": len(reported_planned_indices),
        "missing_episode_count": len(missing_rows),
        "planned_minus_reported_episode_count": len(missing_rows),
        "completed_episode_count": len(completed),
        "reported_failed_or_incomplete_episode_count": len(failures),
        "failed_or_incomplete_episode_count": len(failures) + len(missing_rows),
        "six_panel_video_count": sum(video_is_valid(Path(str(row["six_panel_video"]))) for row in ordered_results if row.get("six_panel_video")),
        "formal_success_count": sum(bool(row.get("success")) for row in completed),
        "task_success_count": sum(bool(row.get("task_success")) for row in completed),
        "nav_success_count": sum(bool(row.get("nav_success")) for row in completed),
        "goal_definition_relaxed_success_count": sum(
            bool(row.get("goal_definition_relaxed_success")) for row in completed
        ),
        "mean_runner_elapsed_sec": numeric_mean(completed, "elapsed_sec"),
        "mean_evaluator_elapsed_sec": numeric_mean(completed, "elapsed_seconds"),
        "mean_step_count": numeric_mean(completed, "step_count"),
        "mean_spl": numeric_mean(completed, "spl"),
    }
    if resource_summary is not None:
        aggregate["resource_telemetry"] = resource_summary
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "config": {
            "output_dir": str(args.output_dir),
            "benchmark": str(args.benchmark),
            "runner": str(args.runner),
            "runner_shell": args.runner_shell,
            "workers": args.workers,
            "base_master_port": args.base_master_port,
            "max_steps": args.max_steps,
            "step_budget_mode": str(args.step_budget_mode),
            "semantic_attribute_request_timeout_s": float(
                args.semantic_attribute_request_timeout_s
            ),
            "ros_command_starvation_timeout_s": float(
                args.ros_command_starvation_timeout_s
            ),
            "ros_observation_turn_multiplier": float(
                args.ros_observation_turn_multiplier
            ),
            "scene_timeout_s": args.scene_timeout_s,
            "fast_eval": bool(args.fast_eval),
            "recording_enabled": not bool(args.fast_eval),
            "model_endpoints": args.model_endpoints,
            "mujoco_egl_devices": args.mujoco_egl_devices,
            "semantic_model_env_file": (
                None if args.semantic_model_env_file is None else str(args.semantic_model_env_file)
            ),
            "resource_telemetry": bool(args.resource_telemetry),
            "resource_sample_interval_s": args.resource_sample_interval_s,
            "resume": args.resume,
        },
        "plans": [plan_to_dict(plan, args) for plan in plans],
        "aggregate": aggregate,
        "episodes": episode_rows,
        "missing_episode_reports": missing_rows,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "aggregate_metrics.json", aggregate)
    fields = [
        "episode_index",
        "worker_id",
        "model_endpoint",
        "mujoco_egl_device",
        "semantic_model_env_file",
        "fast_eval",
        "artifact_validation_mode",
        "completed",
        "report_status",
        "worker_result_reported",
        "failure_kind",
        "incomplete_reason",
        "runner_exit_code",
        "elapsed_sec",
        "episode_status",
        "case_id",
        "house_index",
        "domains",
        "interaction_types",
        "interaction_requirement",
        "success",
        "task_success",
        "nav_success",
        "required_interaction_success",
        "goal_definition_relaxed_success",
        "terminal_reason",
        "step_count",
        "interaction_action_count",
        "spl",
        "six_panel_video",
        "topdown",
        "runner_log",
        "output_dir",
        "error",
    ]
    temporary = args.output_dir / f".summary.csv.{os.getpid()}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({key: encode_csv_value(row.get(key)) for key in fields})
    temporary.replace(args.output_dir / "summary.csv")
    return aggregate


def _gallery_slug(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    safe = "".join(
        char if (char.isalnum() or char in {"-", "_"}) else "-"
        for char in text
    ).strip("-_")
    return (safe or fallback)[:120]


def _relative_symlink_atomic(source: Path, destination: Path) -> None:
    """Create/replace a relative symlink without touching the source artifact."""

    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(os.path.relpath(source, destination.parent))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _render_batch_contact_sheet(
    output_path: Path,
    images: list[tuple[str, Path]],
    *,
    tile_width: int = 420,
    tile_height: int = 330,
    columns: int = 4,
) -> tuple[Path | None, list[str]]:
    """Render a bounded contact sheet for the shallow episode top-down images."""

    errors: list[str] = []
    if not images:
        return None, errors
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return None, [f"Pillow unavailable: {type(exc).__name__}: {exc}"]

    loaded: list[tuple[str, Any]] = []
    for label, path in images:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        loaded.append((label, image))
    if not loaded:
        return None, errors

    label_height = 34
    rows = (len(loaded) + max(1, columns) - 1) // max(1, columns)
    sheet = Image.new(
        "RGB",
        (tile_width * max(1, columns), tile_height * rows),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        for index, (label, image) in enumerate(loaded):
            thumbnail = ImageOps.contain(
                image,
                (tile_width - 12, tile_height - label_height - 12),
                method=Image.Resampling.LANCZOS,
            )
            column = index % max(1, columns)
            row = index // max(1, columns)
            x0 = column * tile_width
            y0 = row * tile_height
            image_x = x0 + (tile_width - thumbnail.width) // 2
            image_y = y0 + label_height + (tile_height - label_height - thumbnail.height) // 2
            sheet.paste(thumbnail, (image_x, image_y))
            draw.rectangle(
                (x0, y0, x0 + tile_width - 1, y0 + label_height - 1),
                fill=(255, 255, 255),
            )
            draw.text((x0 + 8, y0 + 8), label[:58], fill=(20, 20, 20), font=font)
            thumbnail.close()
    finally:
        for _label, image in loaded:
            image.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        sheet.save(temporary, format="PNG", optimize=True)
        temporary.replace(output_path)
    finally:
        sheet.close()
        temporary.unlink(missing_ok=True)
    return output_path, errors


def publish_batch_galleries(
    args: argparse.Namespace,
    plans: list[EpisodePlan],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish stable batch-level image/video aliases after all workers join.

    The attempt tree and the per-episode shallow aliases remain authoritative.
    Gallery entries are additional links/copies for browsing and are never used
    as resume inputs, so a missing optional artifact cannot invalidate a score.
    """

    output_dir = Path(args.output_dir)
    recording_enabled = not bool(args.fast_eval)
    result_by_index = {
        int(row["episode_index"]): row
        for row in rows
        if isinstance(row.get("episode_index"), int)
    }
    topdown_dir = output_dir / "topdown_gallery"
    video_dir = output_dir / "video_gallery"
    errors: list[str] = []
    topdown_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    contact_images: list[tuple[str, Path]] = []

    for plan in sorted(plans, key=lambda item: item.ordinal):
        row = result_by_index.get(plan.episode_index, {})
        case_id = str(row.get("case_id") or f"episode_{plan.episode_index:04d}")
        stem = f"{plan.episode_index:04d}_{_gallery_slug(case_id, fallback='episode')}"

        if not recording_enabled:
            continue

        source_png = Path(str(row["topdown"])) if row.get("topdown") else None
        source_json = (
            None if source_png is None else source_png.with_suffix(".json")
        )
        source_video = (
            Path(str(row["six_panel_video"]))
            if row.get("six_panel_video")
            else None
        )

        if source_png is not None and source_png.is_file():
            gallery_png = topdown_dir / f"{stem}_topdown.png"
            try:
                publish_file(source_png, gallery_png)
                gallery_json = None
                if source_json is not None and source_json.is_file():
                    gallery_json = topdown_dir / f"{stem}_topdown.json"
                    publish_file(source_json, gallery_json)
                metadata = read_json(source_json) if source_json and source_json.is_file() else {}
                coverage = metadata.get("coverage", {}) if isinstance(metadata, dict) else {}
                topdown_rows.append(
                    {
                        "episode_index": plan.episode_index,
                        "case_id": case_id,
                        "source_png": str(source_png),
                        "gallery_png": str(gallery_png),
                        "source_json": None if source_json is None else str(source_json),
                        "gallery_json": None if gallery_json is None else str(gallery_json),
                        "coverage": coverage.get("exploration_coverage_ratio"),
                        "mapped_free": coverage.get("mapped_free_coverage_ratio"),
                        "false_occupied": coverage.get("mapped_occupied_on_gt_free_ratio"),
                    }
                )
                coverage_value = coverage.get("exploration_coverage_ratio")
                coverage_text = (
                    f"cov {100.0 * float(coverage_value):.1f}%"
                    if isinstance(coverage_value, (int, float))
                    else "cov n/a"
                )
                contact_images.append(
                    (f"{plan.episode_index:04d} {coverage_text}", gallery_png)
                )
            except (OSError, ValueError, TypeError) as exc:
                errors.append(
                    f"topdown episode {plan.episode_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
        elif bool(row.get("completed")):
            errors.append(f"topdown episode {plan.episode_index}: missing PNG")

        if source_video is not None and source_video.is_file():
            link_path = video_dir / f"{stem}_overview_6panel.mp4"
            try:
                _relative_symlink_atomic(source_video, link_path)
                video_rows.append(
                    {
                        "episode_index": plan.episode_index,
                        "case_id": case_id,
                        "source_video": str(source_video),
                        "link": str(link_path),
                        "relative_target": os.path.relpath(source_video.resolve(), link_path.parent),
                        "bytes": source_video.stat().st_size,
                    }
                )
            except OSError as exc:
                errors.append(
                    f"video episode {plan.episode_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
        elif bool(row.get("completed")):
            errors.append(
                f"video episode {plan.episode_index}: missing overview_6panel.mp4"
            )

    contact_sheet = None
    if recording_enabled:
        topdown_dir.mkdir(parents=True, exist_ok=True)
        contact_sheet, contact_errors = _render_batch_contact_sheet(
            output_dir / "contact_sheet_all.png",
            contact_images,
        )
        errors.extend(f"contact sheet: {error}" for error in contact_errors)
        video_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            topdown_dir / "index.json",
            {
                "schema_version": "interactive_nav_topdown_gallery_v2",
                "source_batch": str(output_dir),
                "count": len(topdown_rows),
                "contact_sheet": None if contact_sheet is None else "../contact_sheet_all.png",
                "rows": topdown_rows,
                "errors": errors,
            },
        )
        atomic_json(
            video_dir / "index.json",
            {
                "schema_version": "interactive_nav_video_gallery_v2",
                "source_batch": str(output_dir),
                "count": len(video_rows),
                "symlink_count": len(video_rows),
                "total_source_bytes": sum(int(row["bytes"]) for row in video_rows),
                "rows": video_rows,
                "errors": errors,
            },
        )

    summary = {
        "topdown_gallery": str(topdown_dir) if recording_enabled else None,
        "topdown_count": len(topdown_rows),
        "video_gallery": str(video_dir) if recording_enabled else None,
        "video_count": len(video_rows),
        "contact_sheet": None if contact_sheet is None else str(contact_sheet),
        "error_count": len(errors),
        "errors": errors,
    }
    atomic_json(output_dir / "artifact_gallery_summary.json", summary)
    return summary


def validate_args(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.benchmark = args.benchmark.expanduser().resolve()
    args.runner = args.runner.expanduser().resolve()
    if args.python_bin is not None:
        args.python_bin = args.python_bin.expanduser().resolve()
    if args.semantic_model_env_file is not None:
        args.semantic_model_env_file = args.semantic_model_env_file.expanduser().resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.scene_timeout_s <= 0:
        raise ValueError("--scene-timeout-s must be positive")
    if args.semantic_attribute_request_timeout_s <= 0.0:
        raise ValueError("--semantic-attribute-request-timeout-s must be positive")
    if args.ros_command_starvation_timeout_s < 0.0:
        raise ValueError("--ros-command-starvation-timeout-s must be non-negative")
    if args.ros_observation_turn_multiplier < 1.0:
        raise ValueError("--ros-observation-turn-multiplier must be at least 1")
    if args.resource_sample_interval_s <= 0.0:
        raise ValueError("--resource-sample-interval-s must be positive")
    if not 1 <= args.base_master_port <= 65535:
        raise ValueError("--base-master-port must be in [1, 65535]")
    if any(index < 0 for index in args.episode_indices):
        raise ValueError("--episode-indices must be non-negative")
    if len(set(args.episode_indices)) != len(args.episode_indices):
        raise ValueError("--episode-indices must not contain duplicates")
    if args.base_master_port + args.workers - 1 > 65535:
        raise ValueError("base master port range exceeds 65535")
    if not args.benchmark.is_file():
        raise FileNotFoundError(args.benchmark)
    if not args.runner.is_file():
        raise FileNotFoundError(args.runner)
    if args.python_bin is not None and not args.python_bin.is_file():
        raise FileNotFoundError(args.python_bin)
    if args.model_endpoints:
        args.model_endpoints = [endpoint.strip() for endpoint in args.model_endpoints]
        if any(not endpoint for endpoint in args.model_endpoints):
            raise ValueError("--model-endpoints must not contain an empty URL")
    if args.mujoco_egl_devices:
        args.mujoco_egl_devices = [device.strip() for device in args.mujoco_egl_devices]
        if any(not device.isdigit() for device in args.mujoco_egl_devices):
            raise ValueError("--mujoco-egl-devices values must be non-negative integer IDs")
    if args.semantic_model_env_file is None and args.model_endpoints:
        inherited = os.environ.get("SEMANTIC_MODEL_ENV_FILE")
        args.semantic_model_env_file = (
            Path(inherited).expanduser().resolve()
            if inherited
            else (REPO_ROOT / ".env").resolve()
        )
    if args.semantic_model_env_file is not None and not args.semantic_model_env_file.is_file():
        raise FileNotFoundError(args.semantic_model_env_file)


def failed_plan_result(plan: EpisodePlan, args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    """Persist one unexpected wrapper failure without stopping the other workers."""

    result = base_summary(plan, args)
    result.update(
        {
            "completed": False,
            "runner_exit_code": 127,
            "exit_code": 127,
            "error": f"wrapper exception: {type(exc).__name__}: {exc}",
            "resumed": False,
        }
    )
    plan.task_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(plan.task_dir / "batch_task_summary.json", result)
    return result


def run_worker(
    worker_id: int,
    plans: list[EpisodePlan],
    args: argparse.Namespace,
    telemetry: BatchResourceTelemetry | None,
) -> list[dict[str, Any]]:
    """Run a worker's plans serially so its ROS master port is never shared."""

    del worker_id  # The worker identity is already frozen into every plan.
    results: list[dict[str, Any]] = []
    for plan in plans:
        try:
            results.append(run_episode(plan, args, telemetry))
        except Exception as exc:  # Preserve the rest of an overnight shard.
            results.append(failed_plan_result(plan, args, exc))
    return results


def main() -> int:
    args = parse_args()
    validate_args(args)
    plans = [
        EpisodePlan(
            ordinal=ordinal,
            worker_id=ordinal % args.workers,
            episode_index=episode_index,
            # A logical worker owns this port and executes its shard serially.
            # Assigning one port per *episode* is unsafe once a thread pool
            # starts later tasks out of order.
            master_port=args.base_master_port + (ordinal % args.workers),
            task_dir=args.output_dir / f"episode_{episode_index:04d}",
        )
        for ordinal, episode_index in enumerate(args.episode_indices)
    ]
    if args.dry_run:
        rows = [run_episode(plan, args) for plan in plans]
        print(json.dumps({"plans": rows}, ensure_ascii=False, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "batch_manifest.json",
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "created_at": utc_now(),
            "config": {
                "runner": str(args.runner),
                "runner_shell": args.runner_shell,
                "benchmark": str(args.benchmark),
                "workers": args.workers,
                "base_master_port": args.base_master_port,
                "max_steps": args.max_steps,
                "step_budget_mode": str(args.step_budget_mode),
                "semantic_attribute_request_timeout_s": float(
                    args.semantic_attribute_request_timeout_s
                ),
                "ros_command_starvation_timeout_s": float(
                    args.ros_command_starvation_timeout_s
                ),
                "ros_observation_turn_multiplier": float(
                    args.ros_observation_turn_multiplier
                ),
                "scene_timeout_s": args.scene_timeout_s,
                "fast_eval": bool(args.fast_eval),
                "recording_enabled": not bool(args.fast_eval),
                "model_endpoints": args.model_endpoints,
                "mujoco_egl_devices": args.mujoco_egl_devices,
                "semantic_model_env_file": (
                    None if args.semantic_model_env_file is None else str(args.semantic_model_env_file)
                ),
                "resource_telemetry": bool(args.resource_telemetry),
                "resource_sample_interval_s": args.resource_sample_interval_s,
            },
            "plans": [plan_to_dict(plan, args) for plan in plans],
        },
    )

    results: list[dict[str, Any]] = []
    shards = [
        [plan for plan in plans if plan.worker_id == worker_id]
        for worker_id in range(args.workers)
    ]
    telemetry = (
        BatchResourceTelemetry(args.output_dir, args.resource_sample_interval_s)
        if args.resource_telemetry
        else None
    )
    if telemetry is not None:
        telemetry.start()
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(plans))) as executor:
            futures = {
                executor.submit(run_worker, worker_id, shard, args, telemetry): worker_id
                for worker_id, shard in enumerate(shards)
                if shard
            }
            for future in as_completed(futures):
                worker_results = future.result()
                results.extend(worker_results)
                write_summary(args, plans, results)
    finally:
        resource_summary = telemetry.stop() if telemetry is not None else None

    aggregate = write_summary(
        args,
        plans,
        results,
        resource_summary,
        recover_missing_task_summaries=True,
    )
    gallery_summary = publish_batch_galleries(args, plans, results)
    aggregate["artifact_gallery"] = gallery_summary
    final_summary = read_json(args.output_dir / "summary.json")
    if isinstance(final_summary, dict):
        final_summary["aggregate"] = aggregate
        atomic_json(args.output_dir / "summary.json", final_summary)
    atomic_json(args.output_dir / "aggregate_metrics.json", aggregate)
    failure_count = int(aggregate["failed_or_incomplete_episode_count"])
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "episode_count": len(results),
                "failure_count": failure_count,
                "summary": str(args.output_dir / "summary.json"),
                "resource_telemetry": (
                    None if resource_summary is None else resource_summary.get("resource_telemetry_path")
                ),
                "artifact_gallery": gallery_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if args.allow_failures or failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
