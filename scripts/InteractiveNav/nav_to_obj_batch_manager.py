#!/usr/bin/env python3
"""Lease-based batch scheduler for isolated native ``nav_to_obj`` replays.

The native evaluator intentionally evaluates one JSON benchmark episode against
one stateful ROS master.  This utility leaves that evaluator untouched and
coordinates many *independent* invocations of its existing shell launcher.

The SQLite ledger is the source of truth.  It uses WAL plus ``BEGIN IMMEDIATE``
transactions so another ``worker`` process can be started later against the
same ``--run-root`` without selecting an already-running or terminal episode.
Each claim has a renewable lease; a dead worker's claim becomes eligible for a
later worker after it expires.  Every physical attempt receives its own output
directory, so retries cannot overwrite traces or evaluator outputs.

Typical use (on the host that owns the ROS workers)::

    python scripts/InteractiveNav/nav_to_obj_batch_manager.py init \
      --benchmark-dir "$BENCHMARK_DIR" --run-root outputs/nav_batch
    python scripts/InteractiveNav/nav_to_obj_batch_manager.py run \
      --run-root outputs/nav_batch --workers 2 --worker-slot-start 0

Add capacity later without restarting existing workers::

    python scripts/InteractiveNav/nav_to_obj_batch_manager.py worker \
      --run-root outputs/nav_batch --worker-id gpu0-extra --worker-slot 2

The default command is ``run_native_nav_to_obj_eval.zsh <attempt-output-dir>``.
It receives ``EPISODE_IDX``, ``BENCHMARK_DIR``, ``ROS_HOUSE_IND``,
``ROS_TARGET_TYPES`` and a unique ``ROS_MASTER_URI`` via its environment.
``--command-template`` can replace that command; it is split with :mod:`shlex`
(not a shell) and supports the documented placeholders in ``init --help``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAUNCHER = REPO_ROOT / "scripts" / "InteractiveNav" / "run_native_nav_to_obj_eval.zsh"
RUN_CONFIG_FILENAME = "run_config.json"
LEDGER_FILENAME = "episode_ledger.sqlite3"
MANIFEST_FILENAME = "episode_manifest.json"
SCHEMA_VERSION = "native_nav_to_obj_batch_manager_v2"
TERMINAL_STATUSES = {"completed", "failed", "exhausted"}


@dataclass(frozen=True)
class BenchmarkEpisode:
    """The small, scheduler-relevant subset of one benchmark episode."""

    episode_idx: int
    house_index: int
    source_traj_key: str | None
    pickup_obj_name: str
    target_type: str


@dataclass(frozen=True)
class Claim:
    """An atomically assigned episode lease."""

    benchmark_sha256: str
    episode_idx: int
    house_index: int
    source_traj_key: str | None
    pickup_obj_name: str
    target_type: str
    attempt: int
    worker_id: str
    claim_token: str
    claimed_at: float
    lease_expires_at: float


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so status readers never observe a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _benchmark_json_payload(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(
            f"{path} must contain a JSON list, matching "
            "molmo_spaces.evaluation.benchmark_schema.load_all_episodes()."
        )
    if not all(isinstance(episode, dict) for episode in payload):
        raise ValueError(f"{path} contains a non-object episode entry")
    return payload


def _episode_record(episode: dict[str, Any], episode_idx: int, source: Path) -> BenchmarkEpisode:
    try:
        house_index = int(episode["house_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Episode {episode_idx} in {source} has no valid house_index") from exc
    source_info = episode.get("source")
    traj_key = None
    if isinstance(source_info, dict) and source_info.get("traj_key") is not None:
        traj_key = str(source_info["traj_key"])
    task = episode.get("task")
    if not isinstance(task, dict) or not str(task.get("pickup_obj_name", "")).strip():
        raise ValueError(
            f"Episode {episode_idx} in {source} has no task.pickup_obj_name; "
            "native ROS launch cannot safely choose ROS_TARGET_TYPES."
        )
    pickup_obj_name = str(task["pickup_obj_name"])
    target_type = ""
    for key in ("pickup_obj_category", "target_category", "object_category", "target_type"):
        if str(task.get(key, "")).strip():
            target_type = str(task[key]).strip()
            break
    if not target_type:
        # Native ProcTHOR object names use the semantic category as their first
        # underscore-delimited component (e.g. ``laptop_...`` and ``vase_...``).
        # This is intentionally not derived from a free-form language phrase.
        target_type = pickup_obj_name.split("_", 1)[0].strip()
    if not target_type:
        raise ValueError(f"Episode {episode_idx} in {source} has no usable target type")
    return BenchmarkEpisode(
        episode_idx=episode_idx,
        house_index=house_index,
        source_traj_key=traj_key,
        pickup_obj_name=pickup_obj_name,
        target_type=target_type,
    )


def load_benchmark_manifest(benchmark_dir: Path) -> tuple[list[BenchmarkEpisode], str, list[str]]:
    """Load native-evaluator episode ordering and a content fingerprint.

    The ordering deliberately mirrors ``load_all_episodes``: a top-level
    ``benchmark.json`` takes precedence; otherwise legacy ``house_*`` folders
    and their ``episode_*.json`` files are traversed in sorted order.
    """

    benchmark_dir = benchmark_dir.expanduser().resolve()
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark directory does not exist: {benchmark_dir}")

    fingerprint = hashlib.sha256()
    records: list[BenchmarkEpisode] = []
    sources: list[str] = []
    benchmark_file = benchmark_dir / "benchmark.json"
    if benchmark_file.is_file():
        raw = benchmark_file.read_bytes()
        fingerprint.update(b"benchmark.json\0")
        fingerprint.update(raw)
        for episode_idx, episode in enumerate(_benchmark_json_payload(benchmark_file)):
            records.append(_episode_record(episode, episode_idx, benchmark_file))
        sources.append(str(benchmark_file))
    else:
        for house_dir in sorted(benchmark_dir.glob("house_*")):
            if not house_dir.is_dir():
                continue
            for episode_file in sorted(house_dir.glob("episode_*.json")):
                relative_path = episode_file.relative_to(benchmark_dir).as_posix()
                fingerprint.update(relative_path.encode("utf-8"))
                fingerprint.update(b"\0")
                fingerprint.update(episode_file.read_bytes())
                episode = _read_json(episode_file)
                if not isinstance(episode, dict):
                    raise ValueError(f"{episode_file} must contain one JSON object")
                records.append(_episode_record(episode, len(records), episode_file))
                sources.append(str(episode_file))

    if not records:
        raise ValueError(
            f"No episodes found in {benchmark_dir}; expected benchmark.json or house_*/episode_*.json."
        )
    return records, fingerprint.hexdigest(), sources


def _sha256_file(path: Path) -> str | None:
    """Return a stable source hash without failing custom command-template runs."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git_head(repo_root: Path) -> str | None:
    """Best-effort provenance only; source hashes remain the enforcement mechanism."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _runtime_provenance(launcher: Path) -> dict[str, Any]:
    """Capture the evaluator entrypoints that define a native replay's behavior."""

    native_evaluator = REPO_ROOT / "scripts" / "InteractiveNav" / "run_native_nav_to_obj_eval.py"
    return {
        "git_head": _git_head(REPO_ROOT),
        "launcher_path": str(launcher),
        "launcher_sha256": _sha256_file(launcher),
        "native_evaluator_path": str(native_evaluator),
        "native_evaluator_sha256": _sha256_file(native_evaluator),
    }


def _validate_ros_master_uri(ros_master_uri: str) -> int:
    """Reject malformed/invalid ROS master URIs before a worker can claim work."""

    try:
        parsed = urlparse(ros_master_uri)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid --ros-master-uri {ros_master_uri!r}: {exc}") from exc
    if parsed.scheme != "http" or not parsed.hostname or port is None:
        raise ValueError(
            "ROS master URI must be an http://HOST:PORT URI, "
            f"got {ros_master_uri!r}"
        )
    if not 1 <= port <= 65535:
        raise ValueError(f"ROS master port must be in [1, 65535], got {port}")
    return port


def _run_config_signature(run_config: dict[str, Any]) -> str:
    unsigned = dict(run_config)
    unsigned.pop("config_signature", None)
    return _sha256_text(_canonical_json(unsigned))


def _parse_worker_environment(entries: list[str]) -> dict[str, str]:
    """Parse repeatable ``KEY=VALUE`` host-path overrides without persisting secrets."""

    environment: dict[str, str] = {}
    sensitive_fragments = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY")
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--worker-env must use KEY=VALUE, got {entry!r}")
        key, value = entry.split("=", 1)
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid --worker-env key: {key!r}")
        if any(fragment in key.upper() for fragment in sensitive_fragments):
            raise ValueError(
                f"Refusing to persist sensitive-looking variable {key!r}; "
                "put it in --semantic-model-env-file instead."
            )
        environment[key] = value
    return environment


def _normalize_cuda_visible_devices(value: str | None) -> str | None:
    """Validate one runtime CUDA binding while preserving CUDA's native syntax."""

    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("--cuda-visible-devices must not be empty")
    if "\x00" in normalized:
        raise ValueError("--cuda-visible-devices must not contain a NUL byte")
    return normalized


def _parse_cuda_visible_devices_list(
    value: str | None, *, workers: int
) -> list[str | None]:
    """Map a comma-separated GPU list onto the local workers spawned by ``run``."""

    if value is None:
        return [None] * workers
    bindings = [_normalize_cuda_visible_devices(item) for item in value.split(",")]
    if any(binding is None for binding in bindings):  # Defensive for future parser changes.
        raise AssertionError("non-empty CUDA binding list unexpectedly contains None")
    if len(bindings) != workers:
        raise ValueError(
            "--cuda-visible-devices-list must contain exactly one comma-separated binding "
            f"per --workers value (got {len(bindings)} bindings for {workers} workers)"
        )
    return bindings


def _parse_episode_indices(entries: list[str] | None, *, total_count: int) -> list[int] | None:
    """Accept whitespace- or comma-separated global indices and reject ambiguous subsets."""

    if entries is None:
        return None
    parsed: list[int] = []
    for entry in entries:
        for part in entry.split(","):
            if not part:
                raise ValueError("--episode-indices cannot contain an empty comma-separated item")
            try:
                episode_idx = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid --episode-indices value: {part!r}") from exc
            if not 0 <= episode_idx < total_count:
                raise ValueError(
                    f"Episode index {episode_idx} is outside [0, {total_count}) for this benchmark"
                )
            if episode_idx in parsed:
                raise ValueError(f"Duplicate episode index in --episode-indices: {episode_idx}")
            parsed.append(episode_idx)
    if not parsed:
        raise ValueError("--episode-indices must select at least one episode")
    return parsed


def build_run_config(
    *,
    benchmark_dir: Path,
    benchmark_fingerprint: str,
    benchmark_total_episode_count: int,
    episode_count: int,
    selected_episode_indices: list[int],
    seed: int,
    launcher: Path,
    command_template: str | None,
    task_horizon_steps: int | None,
    filter_missing_scene_objects: bool,
    semantic_model_env_file: Path | None,
    base_ros_master_port: int,
    ros_hostname: str,
    lease_seconds: float,
    heartbeat_interval_seconds: float,
    max_attempts_per_episode: int,
    episode_timeout_seconds: float | None = None,
    worker_slot_lease_seconds: float | None = None,
    worker_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create the persisted, non-secret execution contract for a batch run."""

    if not 1 <= base_ros_master_port <= 65535:
        raise ValueError("base_ros_master_port must be in [1, 65535]")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if heartbeat_interval_seconds <= 0 or heartbeat_interval_seconds >= lease_seconds:
        raise ValueError("heartbeat_interval_seconds must be positive and shorter than lease_seconds")
    if max_attempts_per_episode < 1:
        raise ValueError("max_attempts_per_episode must be >= 1")
    if task_horizon_steps is not None and task_horizon_steps < 1:
        raise ValueError("task_horizon_steps must be >= 1")
    if episode_timeout_seconds is not None and episode_timeout_seconds <= 0:
        raise ValueError("episode_timeout_seconds must be positive when provided")
    resolved_worker_slot_lease_seconds = (
        max(60.0, heartbeat_interval_seconds * 3.0)
        if worker_slot_lease_seconds is None
        else float(worker_slot_lease_seconds)
    )
    if resolved_worker_slot_lease_seconds <= heartbeat_interval_seconds:
        raise ValueError(
            "worker_slot_lease_seconds must be longer than heartbeat_interval_seconds"
        )

    run_config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_dir": str(benchmark_dir.expanduser().resolve()),
        "benchmark_sha256": benchmark_fingerprint,
        "benchmark_total_episode_count": int(benchmark_total_episode_count),
        "episode_count": int(episode_count),
        "selected_episode_indices": list(selected_episode_indices),
        "random_seed": int(seed),
        "launcher": str(launcher.expanduser().resolve()),
        "command_template": command_template,
        "task_horizon_steps": task_horizon_steps,
        "filter_missing_scene_objects": bool(filter_missing_scene_objects),
        # This is only a path.  Do not copy model credentials into the run metadata.
        "semantic_model_env_file": (
            None if semantic_model_env_file is None else str(semantic_model_env_file.expanduser().resolve())
        ),
        "base_ros_master_port": int(base_ros_master_port),
        "ros_hostname": str(ros_hostname),
        "lease_seconds": float(lease_seconds),
        "heartbeat_interval_seconds": float(heartbeat_interval_seconds),
        "worker_slot_lease_seconds": resolved_worker_slot_lease_seconds,
        "max_attempts_per_episode": int(max_attempts_per_episode),
        "episode_timeout_seconds": episode_timeout_seconds,
        # Host-specific paths are allowed here so independently started workers
        # inherit the same runtime. Credentials must stay in the referenced
        # semantic-model env file, never in this persistent JSON ledger.
        "worker_environment": dict(worker_environment or {}),
        "repo_root": str(REPO_ROOT),
        "runtime_provenance": _runtime_provenance(launcher.expanduser().resolve()),
        "native_evaluator_contract": {
            "entrypoint": "scripts/InteractiveNav/run_native_nav_to_obj_eval.zsh",
            "official_task": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "execution_model": "one external evaluator invocation per leased episode and ROS master",
            "runtime_adaptations": [
                "head_camera depth is enabled for ROS mapping/debug",
                "RBY1 navigation arm posture is applied only to the runtime replay spec",
                "native evaluator may record an asset-compatibility warning when explicitly enabled",
                "official success is read from native_eval_summary.json, not inferred by this scheduler",
            ],
        },
    }
    run_config["config_signature"] = _run_config_signature(run_config)
    return run_config


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=60.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


class EpisodeLedger:
    """SQLite-backed atomic claim ledger shared by independently started workers."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _transaction(self) -> sqlite3.Connection:
        connection = _connect(self.database_path)
        connection.execute("BEGIN IMMEDIATE")
        return connection

    @staticmethod
    def _commit(connection: sqlite3.Connection) -> None:
        connection.execute("COMMIT")
        connection.close()

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("ROLLBACK")
        finally:
            connection.close()

    def initialize(self, run_config: dict[str, Any], episodes: list[BenchmarkEpisode]) -> None:
        """Create or verify an immutable run ledger and insert its episode rows."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = _connect(self.database_path)
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_idx INTEGER PRIMARY KEY,
                    benchmark_sha256 TEXT NOT NULL,
                    house_index INTEGER NOT NULL,
                    source_traj_key TEXT,
                    pickup_obj_name TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    selection_rank INTEGER NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN (
                        'pending', 'running', 'completed', 'failed', 'exhausted'
                    )),
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    claimed_by TEXT,
                    claim_token TEXT,
                    claimed_at REAL,
                    heartbeat_at REAL,
                    lease_expires_at REAL,
                    completed_at REAL,
                    return_code INTEGER,
                    official_success INTEGER,
                    output_dir TEXT,
                    result_summary_path TEXT,
                    error_message TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS episodes_benchmark_index_idx
                    ON episodes(benchmark_sha256, episode_idx);

                CREATE INDEX IF NOT EXISTS episodes_claimable_idx
                    ON episodes(status, selection_rank);
                CREATE INDEX IF NOT EXISTS episodes_lease_idx
                    ON episodes(status, lease_expires_at);

                CREATE TABLE IF NOT EXISTS attempts (
                    claim_token TEXT PRIMARY KEY,
                    benchmark_sha256 TEXT NOT NULL,
                    episode_idx INTEGER NOT NULL REFERENCES episodes(episode_idx),
                    attempt_number INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'running', 'completed', 'failed', 'reclaimed'
                    )),
                    claimed_at REAL NOT NULL,
                    heartbeat_at REAL,
                    lease_expires_at REAL,
                    finished_at REAL,
                    return_code INTEGER,
                    official_success INTEGER,
                    output_dir TEXT,
                    result_summary_path TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS worker_slots (
                    ros_master_uri TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    session_token TEXT NOT NULL,
                    claimed_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    lease_expires_at REAL NOT NULL
                );
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            existing = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            expected_signature = str(run_config["config_signature"])
            if existing:
                if existing.get("config_signature") != expected_signature:
                    raise ValueError(
                        "Existing ledger has a different run configuration or benchmark. "
                        "Use a new --run-root; do not mix methods/results in one ledger."
                    )
                row_count = int(connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
                if row_count != len(episodes):
                    raise ValueError(
                        f"Existing ledger has {row_count} episodes but this benchmark has {len(episodes)}."
                    )
                connection.execute("COMMIT")
                return

            now = time.time()
            randomized_indices = [episode.episode_idx for episode in episodes]
            random.Random(int(run_config["random_seed"])).shuffle(randomized_indices)
            ranks = {episode_idx: rank for rank, episode_idx in enumerate(randomized_indices)}
            connection.executemany(
                """
                INSERT INTO episodes(
                    episode_idx, benchmark_sha256, house_index, source_traj_key, selection_rank,
                    pickup_obj_name, target_type, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                [
                    (
                        episode.episode_idx,
                        str(run_config["benchmark_sha256"]),
                        episode.house_index,
                        episode.source_traj_key,
                        ranks[episode.episode_idx],
                        episode.pickup_obj_name,
                        episode.target_type,
                        now,
                    )
                    for episode in episodes
                ],
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [
                    ("config_signature", expected_signature),
                    ("run_config", _canonical_json(run_config)),
                    ("created_at", str(now)),
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def verify_config(self, run_config: dict[str, Any]) -> None:
        connection = _connect(self.database_path)
        try:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'config_signature'"
            ).fetchone()
            if row is None:
                raise ValueError(f"Ledger is not initialized: {self.database_path}")
            if str(row["value"]) != str(run_config["config_signature"]):
                raise ValueError("Ledger configuration does not match run_config.json")
        finally:
            connection.close()

    def verify_benchmark_manifest(
        self,
        run_config: dict[str, Any],
        all_episodes: list[BenchmarkEpisode],
        benchmark_fingerprint: str,
    ) -> None:
        """Ensure exported ROS context still names the episode the evaluator will load."""

        expected_fingerprint = str(run_config["benchmark_sha256"])
        if benchmark_fingerprint != expected_fingerprint:
            raise ValueError(
                "Benchmark content changed after init: expected SHA256 "
                f"{expected_fingerprint}, found {benchmark_fingerprint}. "
                "Do not reuse this run root with a changed benchmark."
            )
        expected_total = int(run_config["benchmark_total_episode_count"])
        if len(all_episodes) != expected_total:
            raise ValueError(
                "Benchmark episode count changed after init: expected "
                f"{expected_total}, found {len(all_episodes)}."
            )
        selected_indices = [int(value) for value in run_config["selected_episode_indices"]]
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("run_config.json has duplicate selected_episode_indices")
        if any(index < 0 or index >= len(all_episodes) for index in selected_indices):
            raise ValueError("run_config.json selects an episode outside the current benchmark")

        connection = _connect(self.database_path)
        try:
            rows = list(
                connection.execute(
                    """
                    SELECT benchmark_sha256, episode_idx, house_index, source_traj_key,
                           pickup_obj_name, target_type
                    FROM episodes ORDER BY episode_idx ASC
                    """
                )
            )
        finally:
            connection.close()
        if len(rows) != len(selected_indices):
            raise ValueError(
                "Ledger selection no longer matches run_config.json; choose a new --run-root."
            )
        rows_by_index = {int(row["episode_idx"]): row for row in rows}
        if set(rows_by_index) != set(selected_indices):
            raise ValueError(
                "Ledger episode indices no longer match run_config.json; choose a new --run-root."
            )
        for episode_idx in selected_indices:
            expected = all_episodes[episode_idx]
            row = rows_by_index[episode_idx]
            actual = (
                str(row["benchmark_sha256"]),
                int(row["house_index"]),
                None if row["source_traj_key"] is None else str(row["source_traj_key"]),
                str(row["pickup_obj_name"]),
                str(row["target_type"]),
            )
            wanted = (
                expected_fingerprint,
                expected.house_index,
                expected.source_traj_key,
                expected.pickup_obj_name,
                expected.target_type,
            )
            if actual != wanted:
                raise ValueError(
                    "Benchmark episode metadata changed after init for global index "
                    f"{episode_idx}; refusing to export stale ROS_HOUSE_IND/ROS_TARGET_TYPES."
                )

    @staticmethod
    def _reclaim_expired_in_transaction(
        connection: sqlite3.Connection,
        *,
        now: float,
        max_attempts: int,
    ) -> list[int]:
        expired = list(
            connection.execute(
                """
                SELECT episode_idx, claim_token, attempt_count
                FROM episodes
                WHERE status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (now,),
            )
        )
        for row in expired:
            episode_idx = int(row["episode_idx"])
            claim_token = str(row["claim_token"])
            attempts = int(row["attempt_count"])
            status = "pending" if attempts < max_attempts else "exhausted"
            message = (
                "lease expired; claim made eligible for retry"
                if status == "pending"
                else "lease expired after maximum attempt count"
            )
            connection.execute(
                """
                UPDATE attempts
                SET status = 'reclaimed', finished_at = ?, error_message = ?
                WHERE claim_token = ? AND status = 'running'
                """,
                (now, message, claim_token),
            )
            connection.execute(
                """
                UPDATE episodes
                SET status = ?, claimed_by = NULL, claim_token = NULL, claimed_at = NULL,
                    heartbeat_at = NULL, lease_expires_at = NULL, error_message = ?, updated_at = ?
                WHERE episode_idx = ? AND status = 'running' AND claim_token = ?
                """,
                (status, message, now, episode_idx, claim_token),
            )
        return [int(row["episode_idx"]) for row in expired]

    def reclaim_expired(self, *, now: float | None = None, max_attempts: int) -> list[int]:
        connection = self._transaction()
        try:
            reclaimed = self._reclaim_expired_in_transaction(
                connection,
                now=time.time() if now is None else now,
                max_attempts=max_attempts,
            )
            self._commit(connection)
            return reclaimed
        except Exception:
            self._rollback(connection)
            raise

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        max_attempts: int,
        now: float | None = None,
    ) -> Claim | None:
        """Atomically reclaim stale work and claim one randomized pending episode."""

        claim_time = time.time() if now is None else now
        connection = self._transaction()
        try:
            self._reclaim_expired_in_transaction(
                connection,
                now=claim_time,
                max_attempts=max_attempts,
            )
            row = connection.execute(
                """
                SELECT benchmark_sha256, episode_idx, house_index, source_traj_key, pickup_obj_name,
                       target_type, attempt_count
                FROM episodes
                WHERE status = 'pending' AND attempt_count < ?
                ORDER BY selection_rank ASC
                LIMIT 1
                """,
                (max_attempts,),
            ).fetchone()
            if row is None:
                self._commit(connection)
                return None

            claim_token = uuid.uuid4().hex
            attempt = int(row["attempt_count"]) + 1
            lease_expires_at = claim_time + lease_seconds
            updated = connection.execute(
                """
                UPDATE episodes
                SET status = 'running', attempt_count = ?, claimed_by = ?, claim_token = ?,
                    claimed_at = ?, heartbeat_at = ?, lease_expires_at = ?,
                    error_message = NULL, updated_at = ?
                WHERE episode_idx = ? AND status = 'pending'
                """,
                (
                    attempt,
                    worker_id,
                    claim_token,
                    claim_time,
                    claim_time,
                    lease_expires_at,
                    claim_time,
                    int(row["episode_idx"]),
                ),
            ).rowcount
            if updated != 1:  # Defensive: BEGIN IMMEDIATE makes this unreachable.
                raise RuntimeError("Atomic episode claim unexpectedly lost ownership")
            connection.execute(
                """
                INSERT INTO attempts(
                    claim_token, benchmark_sha256, episode_idx, attempt_number, worker_id, status,
                    claimed_at, heartbeat_at, lease_expires_at
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    claim_token,
                    str(row["benchmark_sha256"]),
                    int(row["episode_idx"]),
                    attempt,
                    worker_id,
                    claim_time,
                    claim_time,
                    lease_expires_at,
                ),
            )
            self._commit(connection)
            return Claim(
                benchmark_sha256=str(row["benchmark_sha256"]),
                episode_idx=int(row["episode_idx"]),
                house_index=int(row["house_index"]),
                source_traj_key=(
                    None if row["source_traj_key"] is None else str(row["source_traj_key"])
                ),
                pickup_obj_name=str(row["pickup_obj_name"]),
                target_type=str(row["target_type"]),
                attempt=attempt,
                worker_id=worker_id,
                claim_token=claim_token,
                claimed_at=claim_time,
                lease_expires_at=lease_expires_at,
            )
        except Exception:
            self._rollback(connection)
            raise

    def set_output_dir(self, claim: Claim, output_dir: Path) -> bool:
        connection = self._transaction()
        try:
            updated = connection.execute(
                """
                UPDATE episodes SET output_dir = ?, updated_at = ?
                WHERE episode_idx = ? AND status = 'running' AND claim_token = ?
                """,
                (str(output_dir), time.time(), claim.episode_idx, claim.claim_token),
            ).rowcount
            if updated:
                connection.execute(
                    "UPDATE attempts SET output_dir = ? WHERE claim_token = ?",
                    (str(output_dir), claim.claim_token),
                )
            self._commit(connection)
            return bool(updated)
        except Exception:
            self._rollback(connection)
            raise

    def heartbeat(self, claim: Claim, *, lease_seconds: float) -> bool:
        heartbeat_time = time.time()
        lease_expires_at = heartbeat_time + lease_seconds
        connection = self._transaction()
        try:
            updated = connection.execute(
                """
                UPDATE episodes
                SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE episode_idx = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    heartbeat_time,
                    lease_expires_at,
                    heartbeat_time,
                    claim.episode_idx,
                    claim.claim_token,
                ),
            ).rowcount
            if updated:
                connection.execute(
                    """
                    UPDATE attempts SET heartbeat_at = ?, lease_expires_at = ?
                    WHERE claim_token = ? AND status = 'running'
                    """,
                    (heartbeat_time, lease_expires_at, claim.claim_token),
                )
            self._commit(connection)
            return bool(updated)
        except Exception:
            self._rollback(connection)
            raise

    def reserve_worker_slot(
        self,
        *,
        worker_id: str,
        session_token: str,
        ros_master_uri: str,
        lease_seconds: float,
    ) -> None:
        """Reserve one ROS URI so dynamically added workers cannot share a master."""

        now = time.time()
        connection = self._transaction()
        try:
            connection.execute(
                "DELETE FROM worker_slots WHERE lease_expires_at < ?",
                (now,),
            )
            existing = connection.execute(
                "SELECT worker_id, session_token FROM worker_slots WHERE ros_master_uri = ?",
                (ros_master_uri,),
            ).fetchone()
            if existing is not None:
                raise RuntimeError(
                    f"ROS master URI already reserved by live worker {existing['worker_id']!r}: "
                    f"{ros_master_uri}. Choose a different --worker-slot or --ros-master-uri."
                )
            connection.execute(
                """
                INSERT INTO worker_slots(
                    ros_master_uri, worker_id, session_token, claimed_at, heartbeat_at, lease_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ros_master_uri, worker_id, session_token, now, now, now + lease_seconds),
            )
            self._commit(connection)
        except Exception:
            self._rollback(connection)
            raise

    def heartbeat_worker_slot(
        self,
        *,
        session_token: str,
        ros_master_uri: str,
        lease_seconds: float,
    ) -> bool:
        now = time.time()
        connection = self._transaction()
        try:
            updated = connection.execute(
                """
                UPDATE worker_slots
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE ros_master_uri = ? AND session_token = ?
                """,
                (now, now + lease_seconds, ros_master_uri, session_token),
            ).rowcount
            self._commit(connection)
            return bool(updated)
        except Exception:
            self._rollback(connection)
            raise

    def release_worker_slot(self, *, session_token: str, ros_master_uri: str) -> None:
        connection = self._transaction()
        try:
            connection.execute(
                "DELETE FROM worker_slots WHERE ros_master_uri = ? AND session_token = ?",
                (ros_master_uri, session_token),
            )
            self._commit(connection)
        except Exception:
            self._rollback(connection)
            raise

    def finish(
        self,
        claim: Claim,
        *,
        status: str,
        return_code: int | None,
        official_success: bool | None,
        result_summary_path: Path | None,
        error_message: str | None,
    ) -> bool:
        """Commit one result only if this worker still owns its original claim."""

        if status not in {"completed", "failed"}:
            raise ValueError(f"Unsupported final status: {status}")
        finished_at = time.time()
        connection = self._transaction()
        try:
            updated = connection.execute(
                """
                UPDATE episodes
                SET status = ?, claimed_by = NULL, claim_token = NULL, claimed_at = NULL,
                    heartbeat_at = NULL, lease_expires_at = NULL, completed_at = ?,
                    return_code = ?, official_success = ?, result_summary_path = ?,
                    error_message = ?, updated_at = ?
                WHERE episode_idx = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    status,
                    finished_at,
                    return_code,
                    None if official_success is None else int(official_success),
                    None if result_summary_path is None else str(result_summary_path),
                    error_message,
                    finished_at,
                    claim.episode_idx,
                    claim.claim_token,
                ),
            ).rowcount
            if updated:
                connection.execute(
                    """
                    UPDATE attempts
                    SET status = ?, finished_at = ?, return_code = ?, official_success = ?,
                        result_summary_path = ?, error_message = ?
                    WHERE claim_token = ? AND status = 'running'
                    """,
                    (
                        status,
                        finished_at,
                        return_code,
                        None if official_success is None else int(official_success),
                        None if result_summary_path is None else str(result_summary_path),
                        error_message,
                        claim.claim_token,
                    ),
                )
            self._commit(connection)
            return bool(updated)
        except Exception:
            self._rollback(connection)
            raise

    def retry_failed(self, *, max_attempts: int) -> int:
        """Explicitly requeue terminal evaluator failures, never completed episodes."""

        connection = self._transaction()
        try:
            now = time.time()
            updated = connection.execute(
                """
                UPDATE episodes
                SET status = 'pending', return_code = NULL, official_success = NULL,
                    error_message = 'explicit retry requested', updated_at = ?
                WHERE status = 'failed' AND attempt_count < ?
                """,
                (now, max_attempts),
            ).rowcount
            self._commit(connection)
            return int(updated)
        except Exception:
            self._rollback(connection)
            raise

    def plan(self, *, count: int, max_attempts: int) -> list[dict[str, Any]]:
        connection = _connect(self.database_path)
        try:
            rows = connection.execute(
                """
                SELECT benchmark_sha256, episode_idx, house_index, source_traj_key, pickup_obj_name,
                       target_type, attempt_count, selection_rank
                FROM episodes
                WHERE status = 'pending' AND attempt_count < ?
                ORDER BY selection_rank ASC
                LIMIT ?
                """,
                (max_attempts, count),
            )
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def status(self) -> dict[str, Any]:
        connection = _connect(self.database_path)
        try:
            now = time.time()
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM metadata WHERE key IN ('config_signature', 'run_config')"
                )
            }
            stored_config = json.loads(metadata["run_config"]) if "run_config" in metadata else {}
            statuses = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM episodes GROUP BY status"
                )
            }
            running = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT benchmark_sha256, episode_idx, house_index, pickup_obj_name, target_type,
                           claimed_by, attempt_count, claimed_at,
                           heartbeat_at, lease_expires_at, output_dir
                    FROM episodes WHERE status = 'running' ORDER BY lease_expires_at ASC
                    """
                )
            ]
            worker_slots = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT ros_master_uri, worker_id, claimed_at, heartbeat_at, lease_expires_at
                    FROM worker_slots ORDER BY ros_master_uri ASC
                    """
                )
            ]
            live_worker_slots = [
                row
                for row in worker_slots
                if float(row["lease_expires_at"]) >= now
            ]
            return {
                "database": str(self.database_path),
                "config_signature": metadata.get("config_signature"),
                "benchmark_sha256": stored_config.get("benchmark_sha256"),
                "now": now,
                "counts": {status: statuses.get(status, 0) for status in sorted(TERMINAL_STATUSES | {"pending", "running"})},
                "selected_episode_count": int(stored_config.get("episode_count", sum(statuses.values()))),
                "terminal_episode_count": sum(statuses.get(status, 0) for status in TERMINAL_STATUSES),
                "leased_episode_count": statuses.get("running", 0),
                "pending_episode_count": statuses.get("pending", 0),
                "pending_never_started": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM episodes WHERE status = 'pending' AND attempt_count = 0"
                    ).fetchone()[0]
                ),
                "stale_running": sum(
                    1
                    for row in running
                    if row["lease_expires_at"] is not None and float(row["lease_expires_at"]) < now
                ),
                "running": running,
                "active_worker_slot_count": len(live_worker_slots),
                "stale_worker_slot_count": len(worker_slots) - len(live_worker_slots),
                "worker_slots": worker_slots,
                "official_successes": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM episodes WHERE status = 'completed' AND official_success = 1"
                    ).fetchone()[0]
                ),
            }
        finally:
            connection.close()


class LeaseHeartbeat:
    """Refresh a claim's lease while its evaluator subprocess is alive."""

    def __init__(
        self,
        *,
        ledger: EpisodeLedger,
        claim: Claim,
        lease_seconds: float,
        interval_seconds: float,
    ) -> None:
        self._ledger = ledger
        self._claim = claim
        self._lease_seconds = lease_seconds
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.error_message: str | None = None
        self._started = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=max(5.0, self._interval_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                still_owned = self._ledger.heartbeat(
                    self._claim,
                    lease_seconds=self._lease_seconds,
                )
            except Exception as exc:  # A failed heartbeat risks duplicate execution.
                self.error_message = f"heartbeat error: {type(exc).__name__}: {exc}"
                self.lost_event.set()
                return
            if not still_owned:
                self.error_message = "heartbeat rejected because this claim no longer owns the episode"
                self.lost_event.set()
                return


class WorkerSlotHeartbeat:
    """Keep a worker's ROS-master reservation live independently of episode claims."""

    def __init__(
        self,
        *,
        ledger: EpisodeLedger,
        worker_id: str,
        session_token: str,
        ros_master_uri: str,
        lease_seconds: float,
        interval_seconds: float,
    ) -> None:
        self._ledger = ledger
        self._worker_id = worker_id
        self._session_token = session_token
        self._ros_master_uri = ros_master_uri
        self._lease_seconds = lease_seconds
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.error_message: str | None = None
        self._started = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=max(5.0, self._interval_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                still_owned = self._ledger.heartbeat_worker_slot(
                    session_token=self._session_token,
                    ros_master_uri=self._ros_master_uri,
                    lease_seconds=self._lease_seconds,
                )
            except Exception as exc:
                self.error_message = f"worker-slot heartbeat error: {type(exc).__name__}: {exc}"
                self.lost_event.set()
                return
            if not still_owned:
                self.error_message = "worker-slot heartbeat rejected; ROS master reservation was lost"
                self.lost_event.set()
                return


def _terminate_process_group(process: subprocess.Popen[bytes], *, grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def _format_command(
    run_config: dict[str, Any],
    *,
    claim: Claim,
    output_dir: Path,
    ros_master_uri: str,
) -> list[str]:
    template = run_config.get("command_template")
    if template is None:
        return [str(run_config["launcher"]), str(output_dir)]
    values = {
        "launcher": str(run_config["launcher"]),
        "repo_root": str(run_config["repo_root"]),
        "episode_idx": claim.episode_idx,
        "benchmark_dir": run_config["benchmark_dir"],
        "output_dir": str(output_dir),
        "worker_id": claim.worker_id,
        "attempt": claim.attempt,
        "ros_master_uri": ros_master_uri,
        "house_index": claim.house_index,
        "target_type": claim.target_type,
        "pickup_obj_name": claim.pickup_obj_name,
    }
    try:
        command = shlex.split(str(template).format(**values))
    except KeyError as exc:
        raise ValueError(f"Unknown --command-template placeholder: {exc.args[0]}") from exc
    if not command:
        raise ValueError("--command-template produced an empty command")
    return command


def _attempt_directory(run_root: Path, claim: Claim) -> Path:
    return (
        run_root
        / "episodes"
        / f"episode_{claim.episode_idx:08d}"
        / f"attempt_{claim.attempt:03d}_{claim.claim_token[:12]}"
    )


def _find_native_summary(attempt_dir: Path) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """Find the native evaluator's authoritative one-episode summary."""

    parsed: list[tuple[Path, dict[str, Any]]] = []
    parse_errors: list[str] = []
    for path in attempt_dir.rglob("native_eval_summary.json"):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(payload, dict):
            parse_errors.append(f"{path}: summary is not a JSON object")
            continue
        try:
            if int(payload.get("total_count", -1)) != 1:
                parse_errors.append(f"{path}: total_count is not 1")
                continue
            int(payload["success_count"])
            official_dir = Path(str(payload["official_eval_output_dir"])).resolve()
        except (KeyError, TypeError, ValueError) as exc:
            parse_errors.append(f"{path}: invalid native summary ({exc})")
            continue
        parsed.append((path, {**payload, "official_eval_output_dir": str(official_dir)}))

    if not parsed:
        detail = "; ".join(parse_errors) if parse_errors else "file was not produced"
        return None, None, f"No valid native_eval_summary.json: {detail}"
    # The evaluator writes one copy under the official eval output and one under
    # debug.  Prefer the former even if it has the same timestamp as the debug copy.
    parsed.sort(
        key=lambda item: (
            Path(item[1]["official_eval_output_dir"]).resolve() == item[0].parent.resolve(),
            item[0].stat().st_mtime,
        ),
        reverse=True,
    )
    return parsed[0][0], parsed[0][1], None


def _run_claim(
    *,
    run_root: Path,
    run_config: dict[str, Any],
    ledger: EpisodeLedger,
    claim: Claim,
    ros_master_uri: str,
    worker_slot_heartbeat: WorkerSlotHeartbeat | None = None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    attempt_dir = _attempt_directory(run_root, claim)
    output_dir = attempt_dir / "output"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    if not ledger.set_output_dir(claim, attempt_dir):
        raise RuntimeError("Claim was lost before its attempt directory could be registered")

    command = _format_command(
        run_config,
        claim=claim,
        output_dir=output_dir,
        ros_master_uri=ros_master_uri,
    )
    environment = os.environ.copy()
    worker_environment = {
        str(key): str(value) for key, value in run_config["worker_environment"].items()
    }
    inherited_cuda_visible_devices = environment.get("CUDA_VISIBLE_DEVICES")
    environment.update(worker_environment)
    # Do not allow an interactive shell's leftovers to silently override the
    # immutable batch contract or make independent attempts share temp/debug
    # directories. Runtime knobs that remain configurable belong in
    # --worker-env and are persisted in run_config.json.
    for key in (
        "EPISODE_IDX",
        "BENCHMARK_DIR",
        "ROS_MASTER_URI",
        "ROS_HOUSE_IND",
        "ROS_TARGET_TYPES",
        "ROS_TASK_HORIZON",
        "TASK_HORIZON_STEPS",
        "FILTER_MISSING_SCENE_OBJECTS",
        "NATIVE_NAV_FILTER_MISSING_SCENE_OBJECTS",
        "SEMANTIC_MODEL_ENV_FILE",
        "SEMANTIC_DECISION_ENV_FILE",
        "DEBUG_DIR",
        "NATIVE_NAV_DEBUG_DIR",
        "TMPDIR",
        "TMP",
        "TEMP",
        "ROS_HOME",
        "ROS_LOG_DIR",
        "ROS_IP",
        "ROS_HOSTNAME",
    ):
        environment.pop(key, None)
    attempt_tmp_dir = attempt_dir / "tmp"
    environment.update(
        {
            "EPISODE_IDX": str(claim.episode_idx),
            "BENCHMARK_DIR": str(run_config["benchmark_dir"]),
            "ROS_MASTER_URI": ros_master_uri,
            "ROS_HOUSE_IND": str(claim.house_index),
            "ROS_TARGET_TYPES": claim.target_type,
            "FILTER_MISSING_SCENE_OBJECTS": str(
                bool(run_config.get("filter_missing_scene_objects"))
            ).lower(),
            "DEBUG_DIR": str(attempt_dir / "debug"),
            "TMPDIR": str(attempt_tmp_dir),
            "TMP": str(attempt_tmp_dir),
            "TEMP": str(attempt_tmp_dir),
            "ROS_IP": "127.0.0.1",
            "ROS_HOSTNAME": str(run_config["ros_hostname"]),
            "NATIVE_NAV_BATCH_RUN_ROOT": str(run_root),
            "NATIVE_NAV_BATCH_WORKER_ID": claim.worker_id,
            "NATIVE_NAV_BATCH_ATTEMPT": str(claim.attempt),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if run_config.get("task_horizon_steps") is not None:
        resolved_horizon = str(run_config["task_horizon_steps"])
        environment["TASK_HORIZON_STEPS"] = resolved_horizon
        environment["ROS_TASK_HORIZON"] = resolved_horizon
    semantic_model_env_file = run_config.get("semantic_model_env_file")
    if semantic_model_env_file:
        environment["SEMANTIC_MODEL_ENV_FILE"] = str(semantic_model_env_file)
    if cuda_visible_devices is not None:
        # This per-invocation binding deliberately wins over a persisted
        # --worker-env value without mutating run_config.json.
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        cuda_visible_devices_source = "worker_cli"
    elif "CUDA_VISIBLE_DEVICES" in worker_environment:
        cuda_visible_devices_source = "worker_env"
    elif inherited_cuda_visible_devices is not None:
        cuda_visible_devices_source = "inherited_environment"
    else:
        cuda_visible_devices_source = "unset"
    effective_cuda_visible_devices = environment.get("CUDA_VISIBLE_DEVICES")

    claim_payload = {
        "claim": asdict(claim),
        "command": command,
        "environment_overrides": {
            key: environment[key]
            for key in (
                "EPISODE_IDX",
                "BENCHMARK_DIR",
                "ROS_MASTER_URI",
                "ROS_HOUSE_IND",
                "ROS_TARGET_TYPES",
                "ROS_TASK_HORIZON",
                "TASK_HORIZON_STEPS",
                "FILTER_MISSING_SCENE_OBJECTS",
                "SEMANTIC_MODEL_ENV_FILE",
                "DEBUG_DIR",
                "TMPDIR",
                "ROS_IP",
                "ROS_HOSTNAME",
                "CUDA_VISIBLE_DEVICES",
            )
            if key in environment
        },
        "runtime_provenance": {
            "cuda_visible_devices": effective_cuda_visible_devices,
            "cuda_visible_devices_source": cuda_visible_devices_source,
        },
        "started_at": time.time(),
    }
    _write_json(attempt_dir / "claim.json", claim_payload)

    start_time = time.monotonic()
    episode_timeout_seconds = run_config.get("episode_timeout_seconds")
    deadline = (
        None
        if episode_timeout_seconds is None
        else start_time + float(episode_timeout_seconds)
    )
    return_code: int | None = None
    launch_error: str | None = None
    process: subprocess.Popen[bytes] | None = None
    heartbeat = LeaseHeartbeat(
        ledger=ledger,
        claim=claim,
        lease_seconds=float(run_config["lease_seconds"]),
        interval_seconds=float(run_config["heartbeat_interval_seconds"]),
    )
    try:
        with (attempt_dir / "stdout.log").open("wb") as stdout, (
            attempt_dir / "stderr.log"
        ).open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=str(run_config["repo_root"]),
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            heartbeat.start()
            while process.poll() is None:
                if (
                    worker_slot_heartbeat is not None
                    and worker_slot_heartbeat.lost_event.is_set()
                ):
                    launch_error = (
                        worker_slot_heartbeat.error_message
                        or "lost worker-slot ROS master reservation"
                    )
                    _terminate_process_group(process)
                    break
                if heartbeat.lost_event.wait(0.25):
                    launch_error = heartbeat.error_message or "lost episode lease"
                    _terminate_process_group(process)
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    launch_error = (
                        "batch manager timeout after "
                        f"{float(episode_timeout_seconds):.1f} seconds"
                    )
                    _terminate_process_group(process)
                    break
            return_code = process.wait()
    except KeyboardInterrupt:
        launch_error = "worker interrupted by KeyboardInterrupt"
        if process is not None:
            _terminate_process_group(process)
        raise
    except Exception as exc:
        launch_error = f"launch error: {type(exc).__name__}: {exc}"
    finally:
        heartbeat.stop()

    elapsed_seconds = time.monotonic() - start_time
    summary_path, summary, summary_error = _find_native_summary(attempt_dir)
    official_success: bool | None = None
    status = "failed"
    error_message = launch_error
    failure_kind: str | None = None
    if return_code == 0 and launch_error is None and summary is not None:
        official_success = int(summary["success_count"]) > 0
        status = "completed"
    else:
        if launch_error is not None and "timeout" in launch_error:
            failure_kind = "timeout"
        elif launch_error is not None:
            failure_kind = "launcher_exception"
        elif return_code != 0:
            failure_kind = "launcher_nonzero_exit"
        else:
            failure_kind = "missing_or_invalid_native_summary"
        if error_message is None:
            error_message = summary_error or f"launcher returned non-zero exit code {return_code}"

    result = {
        "claim": asdict(claim),
        "status": status,
        "return_code": return_code,
        "official_success": official_success,
        "summary_path": None if summary_path is None else str(summary_path),
        "summary": summary,
        "error_message": error_message,
        "failure_kind": failure_kind,
        "cuda_visible_devices": effective_cuda_visible_devices,
        "cuda_visible_devices_source": cuda_visible_devices_source,
        "elapsed_seconds": elapsed_seconds,
        "finished_at": time.time(),
    }
    _write_json(attempt_dir / "batch_result.json", result)
    ownership_retained = ledger.finish(
        claim,
        status=status,
        return_code=return_code,
        official_success=official_success,
        result_summary_path=summary_path,
        error_message=error_message,
    )
    result["ledger_committed"] = ownership_retained
    _write_json(attempt_dir / "batch_result.json", result)
    return result


def initialize_run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    all_episodes, benchmark_fingerprint, sources = load_benchmark_manifest(args.benchmark_dir)
    selected_indices = _parse_episode_indices(
        args.episode_indices,
        total_count=len(all_episodes),
    )
    if selected_indices is None:
        episodes = all_episodes
        selected_indices = [episode.episode_idx for episode in episodes]
    else:
        episodes_by_index = {episode.episode_idx: episode for episode in all_episodes}
        episodes = [episodes_by_index[episode_idx] for episode_idx in selected_indices]
    launcher = args.launcher.expanduser().resolve()
    if args.command_template is None and not launcher.is_file():
        raise FileNotFoundError(f"Native launcher does not exist: {launcher}")
    if args.semantic_model_env_file is not None and not args.semantic_model_env_file.is_file():
        raise FileNotFoundError(
            f"Semantic model environment file does not exist: {args.semantic_model_env_file}"
        )
    run_config = build_run_config(
        benchmark_dir=args.benchmark_dir,
        benchmark_fingerprint=benchmark_fingerprint,
        benchmark_total_episode_count=len(all_episodes),
        episode_count=len(episodes),
        selected_episode_indices=selected_indices,
        seed=args.seed,
        launcher=launcher,
        command_template=args.command_template,
        task_horizon_steps=args.task_horizon_steps,
        filter_missing_scene_objects=args.filter_missing_scene_objects,
        semantic_model_env_file=args.semantic_model_env_file,
        base_ros_master_port=args.base_ros_master_port,
        ros_hostname=args.ros_hostname,
        lease_seconds=args.lease_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        max_attempts_per_episode=args.max_attempts_per_episode,
        episode_timeout_seconds=args.episode_timeout_seconds,
        worker_slot_lease_seconds=args.worker_slot_lease_seconds,
        worker_environment=_parse_worker_environment(args.worker_env),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / RUN_CONFIG_FILENAME
    if config_path.exists():
        existing_config = _read_json(config_path)
        if existing_config.get("config_signature") != run_config["config_signature"]:
            raise ValueError(
                f"{config_path} already describes a different run. Choose a new --run-root."
            )
    else:
        _write_json(config_path, run_config)
    _write_json(
        run_root / MANIFEST_FILENAME,
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark_dir": run_config["benchmark_dir"],
            "benchmark_sha256": benchmark_fingerprint,
            "episode_count": len(episodes),
            "benchmark_total_episode_count": len(all_episodes),
            "selected_episode_indices": selected_indices,
            "sources": sources,
            "episodes": [
                {
                    "benchmark_sha256": benchmark_fingerprint,
                    "global_episode_idx": episode.episode_idx,
                    **asdict(episode),
                }
                for episode in episodes
            ],
        },
    )
    ledger = EpisodeLedger(run_root / LEDGER_FILENAME)
    ledger.initialize(run_config, episodes)
    return {
        "run_root": str(run_root),
        "ledger": str(ledger.database_path),
        "config_signature": run_config["config_signature"],
        "episode_count": len(episodes),
        "benchmark_sha256": benchmark_fingerprint,
    }


def _load_run_config(run_root: Path) -> tuple[Path, dict[str, Any], EpisodeLedger]:
    run_root = run_root.expanduser().resolve()
    config_path = run_root / RUN_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Run is not initialized; missing {config_path}")
    run_config = _read_json(config_path)
    if run_config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported run config schema: {run_config.get('schema_version')!r}")
    if _run_config_signature(run_config) != run_config.get("config_signature"):
        raise ValueError(f"Run config checksum mismatch: {config_path}")
    ledger = EpisodeLedger(run_root / LEDGER_FILENAME)
    ledger.verify_config(run_config)
    return run_root, run_config, ledger


def _verify_runtime_provenance(run_config: dict[str, Any]) -> None:
    """Refuse evaluator entrypoint drift before assigning any new work."""

    provenance = run_config.get("runtime_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("run_config.json is missing runtime provenance; reinitialize the batch run")
    for label, path_key, hash_key in (
        ("native launcher", "launcher_path", "launcher_sha256"),
        ("native evaluator", "native_evaluator_path", "native_evaluator_sha256"),
    ):
        expected_hash = provenance.get(hash_key)
        path_value = provenance.get(path_key)
        if expected_hash is None:
            # A custom command template may intentionally not have a local
            # launcher file. Its command remains persisted in run_config.
            continue
        if not isinstance(path_value, str):
            raise ValueError(f"run_config.json has invalid {label} provenance")
        actual_hash = _sha256_file(Path(path_value))
        if actual_hash != expected_hash:
            raise ValueError(
                f"{label.capitalize()} changed after init: {path_value}. "
                "Use a new --run-root so results do not mix evaluator revisions."
            )


def verify_execution_contract(run_config: dict[str, Any], ledger: EpisodeLedger) -> None:
    """Validate the benchmark and native entrypoints immediately before execution."""

    all_episodes, benchmark_fingerprint, _sources = load_benchmark_manifest(
        Path(str(run_config["benchmark_dir"]))
    )
    ledger.verify_benchmark_manifest(run_config, all_episodes, benchmark_fingerprint)
    _verify_runtime_provenance(run_config)


def worker_main(args: argparse.Namespace) -> int:
    run_root, run_config, ledger = _load_run_config(args.run_root)
    verify_execution_contract(run_config, ledger)
    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    if args.worker_slot < 0:
        raise ValueError("worker_slot must be >= 0")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max_episodes must be >= 1 when provided")
    cuda_visible_devices = _normalize_cuda_visible_devices(
        getattr(args, "cuda_visible_devices", None)
    )
    if args.ros_master_uri is not None:
        ros_master_uri = str(args.ros_master_uri)
    else:
        master_port = int(run_config["base_ros_master_port"]) + int(args.worker_slot)
        if not 1 <= master_port <= 65535:
            raise ValueError(
                "base_ros_master_port + worker_slot must be in [1, 65535], got "
                f"{master_port}"
            )
        ros_master_uri = f"http://{run_config['ros_hostname']}:{master_port}"
    _validate_ros_master_uri(ros_master_uri)
    max_attempts = int(run_config["max_attempts_per_episode"])
    session_token = uuid.uuid4().hex
    completed_by_worker = 0
    worker_exit_code = 0
    worker_slot_error: str | None = None
    release_error: str | None = None
    slot_reserved = False
    slot_heartbeat: WorkerSlotHeartbeat | None = None
    slot_heartbeat_started = False
    try:
        # A ROS URI is a process-global namespace.  Claim it separately from
        # an episode so a later `worker` command cannot accidentally drive the
        # same ROS master even before either worker obtains a benchmark item.
        ledger.reserve_worker_slot(
            worker_id=worker_id,
            session_token=session_token,
            ros_master_uri=ros_master_uri,
            lease_seconds=float(run_config["worker_slot_lease_seconds"]),
        )
        slot_reserved = True
        slot_heartbeat = WorkerSlotHeartbeat(
            ledger=ledger,
            worker_id=worker_id,
            session_token=session_token,
            ros_master_uri=ros_master_uri,
            lease_seconds=float(run_config["worker_slot_lease_seconds"]),
            interval_seconds=float(run_config["heartbeat_interval_seconds"]),
        )
        slot_heartbeat.start()
        slot_heartbeat_started = True
        print(
            _canonical_json(
                {
                    "event": "worker_slot_reserved",
                    "worker_id": worker_id,
                    "worker_slot": args.worker_slot,
                    "ros_master_uri": ros_master_uri,
                    "cuda_visible_devices": cuda_visible_devices,
                }
            ),
            flush=True,
        )
        if args.retry_failed:
            retried = ledger.retry_failed(max_attempts=max_attempts)
            print(
                _canonical_json(
                    {"event": "retry_failed", "worker_id": worker_id, "requeued_count": retried}
                ),
                flush=True,
            )

        while args.max_episodes is None or completed_by_worker < args.max_episodes:
            if slot_heartbeat.lost_event.is_set():
                worker_slot_error = slot_heartbeat.error_message or "lost worker-slot reservation"
                worker_exit_code = 1
                print(
                    _canonical_json(
                        {
                            "event": "worker_slot_lost",
                            "worker_id": worker_id,
                            "ros_master_uri": ros_master_uri,
                            "error_message": worker_slot_error,
                        }
                    ),
                    flush=True,
                )
                break
            claim = ledger.claim_next(
                worker_id=worker_id,
                lease_seconds=float(run_config["lease_seconds"]),
                max_attempts=max_attempts,
            )
            if claim is None:
                break
            print(
                _canonical_json(
                    {
                        "event": "claimed",
                        "worker_id": worker_id,
                        "worker_slot": args.worker_slot,
                        "ros_master_uri": ros_master_uri,
                        "episode_idx": claim.episode_idx,
                        "house_index": claim.house_index,
                        "target_type": claim.target_type,
                        "cuda_visible_devices": cuda_visible_devices,
                        "attempt": claim.attempt,
                    }
                ),
                flush=True,
            )
            try:
                result = _run_claim(
                    run_root=run_root,
                    run_config=run_config,
                    ledger=ledger,
                    claim=claim,
                    ros_master_uri=ros_master_uri,
                    worker_slot_heartbeat=slot_heartbeat,
                    cuda_visible_devices=cuda_visible_devices,
                )
            except KeyboardInterrupt:
                ledger.finish(
                    claim,
                    status="failed",
                    return_code=None,
                    official_success=None,
                    result_summary_path=None,
                    error_message="worker interrupted by KeyboardInterrupt",
                )
                raise
            except Exception as exc:
                error_message = f"batch manager error: {type(exc).__name__}: {exc}"
                ledger.finish(
                    claim,
                    status="failed",
                    return_code=None,
                    official_success=None,
                    result_summary_path=None,
                    error_message=error_message,
                )
                result = {
                    "event": "manager_error",
                    "episode_idx": claim.episode_idx,
                    "attempt": claim.attempt,
                    "error_message": error_message,
                }
            print(_canonical_json(result), flush=True)
            completed_by_worker += 1
            if slot_heartbeat.lost_event.is_set():
                worker_slot_error = slot_heartbeat.error_message or "lost worker-slot reservation"
                worker_exit_code = 1
                break
    finally:
        if slot_heartbeat is not None and slot_heartbeat_started:
            slot_heartbeat.stop()
        if slot_reserved:
            try:
                ledger.release_worker_slot(
                    session_token=session_token,
                    ros_master_uri=ros_master_uri,
                )
            except Exception as exc:
                release_error = f"worker-slot release error: {type(exc).__name__}: {exc}"
                worker_exit_code = 1
        try:
            status = ledger.status()
        except Exception as exc:
            status = {"status_error": f"{type(exc).__name__}: {exc}"}
            worker_exit_code = 1
        print(
            _canonical_json(
                {
                    "event": "worker_finished",
                    "worker_id": worker_id,
                    "worker_slot": args.worker_slot,
                    "ros_master_uri": ros_master_uri,
                    "cuda_visible_devices": cuda_visible_devices,
                    "claimed_count": completed_by_worker,
                    "worker_slot_error": worker_slot_error,
                    "release_error": release_error,
                    "status": status,
                }
            ),
            flush=True,
        )
    return worker_exit_code


def run_workers(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.worker_slot_start < 0:
        raise ValueError("worker_slot_start must be >= 0")
    if args.max_episodes_per_worker is not None and args.max_episodes_per_worker < 1:
        raise ValueError("max_episodes_per_worker must be >= 1 when provided")
    cuda_bindings = _parse_cuda_visible_devices_list(
        getattr(args, "cuda_visible_devices_list", None), workers=args.workers
    )
    run_root, run_config, ledger = _load_run_config(args.run_root)
    verify_execution_contract(run_config, ledger)
    highest_port = (
        int(run_config["base_ros_master_port"])
        + int(args.worker_slot_start)
        + int(args.workers)
        - 1
    )
    if highest_port > 65535:
        raise ValueError(
            "base_ros_master_port + requested worker slots exceeds 65535 "
            f"(highest requested port: {highest_port})"
        )
    script_path = Path(__file__).resolve()
    worker_id_prefix = args.worker_id_prefix or f"{socket.gethostname()}-batch-{os.getpid()}"
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for worker_offset in range(args.workers):
            worker_slot = args.worker_slot_start + worker_offset
            cuda_visible_devices = cuda_bindings[worker_offset]
            command = [
                sys.executable,
                str(script_path),
                "worker",
                "--run-root",
                str(run_root),
                "--worker-id",
                f"{worker_id_prefix}-{worker_slot}",
                "--worker-slot",
                str(worker_slot),
            ]
            if args.max_episodes_per_worker is not None:
                command.extend(["--max-episodes", str(args.max_episodes_per_worker)])
            if cuda_visible_devices is not None:
                command.extend(["--cuda-visible-devices", cuda_visible_devices])
            if args.retry_failed:
                command.append("--retry-failed")
            processes.append(subprocess.Popen(command, start_new_session=True))
        return_code = 0
        for process in processes:
            return_code = max(return_code, process.wait())
        return return_code
    except BaseException:
        for process in processes:
            _terminate_process_group(process)
        raise


def _add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument(
        "--command-template",
        default=None,
        help=(
            "Optional shlex-split command replacing the launcher. Supported placeholders: "
            "{launcher}, {repo_root}, {episode_idx}, {benchmark_dir}, {output_dir}, {worker_id}, "
            "{attempt}, {ros_master_uri}, {house_index}, {target_type}, {pickup_obj_name}."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--episode-indices",
        nargs="+",
        default=None,
        help="Optional global benchmark indices for a bounded smoke subset; accepts spaces and commas.",
    )
    parser.add_argument("--task-horizon-steps", type=int, default=None)
    parser.add_argument("--filter-missing-scene-objects", action="store_true")
    parser.add_argument(
        "--semantic-model-env-file",
        type=Path,
        default=None,
        help="Path forwarded as SEMANTIC_MODEL_ENV_FILE; its contents are never copied into the ledger.",
    )
    parser.add_argument("--base-ros-master-port", type=int, default=11601)
    parser.add_argument("--ros-hostname", default="127.0.0.1")
    parser.add_argument("--lease-seconds", type=float, default=7200.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--worker-slot-lease-seconds",
        type=float,
        default=None,
        help=(
            "TTL for a live ROS-master slot reservation. Defaults to max(60, three heartbeat "
            "intervals), independently of the longer per-episode lease."
        ),
    )
    parser.add_argument("--max-attempts-per-episode", type=int, default=2)
    parser.add_argument(
        "--episode-timeout-seconds",
        type=float,
        default=None,
        help="Optional outer timeout per launcher invocation; a timeout is recorded as a failed attempt.",
    )
    parser.add_argument(
        "--worker-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Persist a non-secret host-path environment override for every worker (repeatable).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create or verify an immutable run ledger.")
    _add_init_arguments(init_parser)

    worker_parser = subparsers.add_parser(
        "worker",
        help="Claim and run episodes until the shared ledger has no eligible work.",
    )
    worker_parser.add_argument("--run-root", type=Path, required=True)
    worker_parser.add_argument("--worker-id", default=None)
    worker_parser.add_argument("--worker-slot", type=int, default=0)
    worker_parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help=(
            "Runtime CUDA_VISIBLE_DEVICES override for this worker; it takes precedence over "
            "the persisted --worker-env value and is recorded per attempt."
        ),
    )
    worker_parser.add_argument(
        "--ros-master-uri",
        default=None,
        help="Override the URI derived from base port + worker slot.",
    )
    worker_parser.add_argument("--max-episodes", type=int, default=None)
    worker_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="At startup only, requeue failed (not completed) episodes below their attempt cap.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Spawn a local set of independent worker processes; more workers may join later.",
    )
    run_parser.add_argument("--run-root", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, required=True)
    run_parser.add_argument("--worker-slot-start", type=int, default=0)
    run_parser.add_argument("--worker-id-prefix", default=None)
    run_parser.add_argument("--max-episodes-per-worker", type=int, default=None)
    run_parser.add_argument(
        "--cuda-visible-devices-list",
        default=None,
        help=(
            "Comma-separated CUDA_VISIBLE_DEVICES bindings, one per spawned worker in slot order "
            "(for example: 0,1)."
        ),
    )
    run_parser.add_argument("--retry-failed", action="store_true")

    status_parser = subparsers.add_parser("status", help="Print ledger counts and live leases as JSON.")
    status_parser.add_argument("--run-root", type=Path, required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Show the next randomized pending episodes without claiming them.",
    )
    plan_parser.add_argument("--run-root", type=Path, required=True)
    plan_parser.add_argument("--count", type=int, default=20)

    reclaim_parser = subparsers.add_parser(
        "reclaim",
        help="Move expired leases back to pending/exhausted without starting a worker.",
    )
    reclaim_parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        print(json.dumps(initialize_run(args), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        return worker_main(args)
    if args.command == "run":
        return run_workers(args)
    if args.command == "status":
        _run_root, _run_config, ledger = _load_run_config(args.run_root)
        print(json.dumps(ledger.status(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "plan":
        _run_root, run_config, ledger = _load_run_config(args.run_root)
        if args.count < 1:
            raise ValueError("count must be >= 1")
        print(
            json.dumps(
                ledger.plan(count=args.count, max_attempts=int(run_config["max_attempts_per_episode"])),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "reclaim":
        _run_root, run_config, ledger = _load_run_config(args.run_root)
        reclaimed = ledger.reclaim_expired(
            max_attempts=int(run_config["max_attempts_per_episode"])
        )
        print(json.dumps({"reclaimed_episode_indices": reclaimed}, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
