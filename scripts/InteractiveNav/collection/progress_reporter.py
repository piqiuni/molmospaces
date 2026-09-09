"""File-based progress reporting for long-running interactive-nav collection.

The reporter deliberately reads only atomic partial artifacts and log files. It
does not communicate with or modify worker processes, so it is safe to run with
one or many workers.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _list_len(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        payload = _read_json(path, [])
        if isinstance(payload, list):
            total += len(payload)
    return total


def _preferred_shard_artifacts(
    batch: Path,
    *,
    shard_pattern: str,
    final_name: str,
    partial_name: str,
) -> list[Path]:
    paths = []
    for shard in sorted(batch.glob(shard_pattern)):
        final_path = shard / final_name
        partial_path = shard / partial_name
        if final_path.exists():
            paths.append(final_path)
        elif partial_path.exists():
            paths.append(partial_path)
    return paths


def _batch_number(path: Path) -> int:
    match = re.search(r"batch_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _active_batch(root: Path, pattern: str) -> Path | None:
    batches = sorted(root.glob(pattern), key=_batch_number)
    completed_numbers = [
        _batch_number(batch)
        for batch in batches
        if (batch / "batch_meta.json").exists()
    ]
    latest_completed = max(completed_numbers, default=-1)
    # Interrupted historical batches may intentionally remain without metadata.
    # Once a later batch is finalized, those older partial artifacts are resume
    # inputs rather than the currently active wave.
    active = [
        batch
        for batch in batches
        if not (batch / "batch_meta.json").exists()
        and _batch_number(batch) > latest_completed
    ]
    return active[-1] if active else None


def _eta_text(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def _progress_line(
    *,
    stage: str,
    batch: str,
    processed: int,
    total: int,
    valid: int,
    started_at: float,
    now: float,
    global_text: str = "",
    worker_loads: list[int] | None = None,
) -> str:
    elapsed = max(now - started_at, 0.0)
    rate = processed / elapsed if processed > 0 and elapsed > 0 else 0.0
    remaining = max(total - processed, 0)
    eta = remaining / rate if rate > 0 else None
    percent = 100.0 * processed / total if total > 0 else 0.0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S%z", time.localtime(now))
    worker_text = ""
    if worker_loads:
        worker_text = (
            f" workers={len(worker_loads)}"
            f" worker_loads={','.join(str(value) for value in worker_loads)}"
        )
    return (
        f"[collection-progress] time={timestamp} stage={stage} batch={batch} "
        f"batch_progress={processed}/{total} ({percent:.1f}%) batch_valid={valid}"
        f"{worker_text} "
        f"rate={rate:.3f}/s elapsed={_eta_text(elapsed)} eta={_eta_text(eta)}"
        f"{global_text}"
    )


def _deduplicated_episode_count(paths: list[Path]) -> int:
    case_ids: set[str] = set()
    fallback_count = 0
    for path in paths:
        payload = _read_json(path, [])
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            rows = payload["episodes"]
        elif isinstance(payload, dict) and "interactive_nav" in payload:
            rows = [payload]
        else:
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            case_id = (
                row.get("interactive_nav", {}).get("case_id")
                if isinstance(row.get("interactive_nav"), dict)
                else row.get("case_id")
            )
            if case_id:
                case_ids.add(str(case_id))
            else:
                fallback_count += 1
    return len(case_ids) + fallback_count


def _global_start_time(root: Path, now: float) -> float:
    candidates = [
        path.stat().st_mtime
        for pattern in (
            "raw/channel_batches/batch_*",
            "raw/container_batches/batch_*",
            "raw/mixed_shards_batch_*",
        )
        for path in root.glob(pattern)
        if path.exists()
    ]
    return min(candidates) if candidates else now


def _domain_counts(root: Path) -> dict[str, int]:
    raw = root / "raw"
    paths = {
        "channel": [
            raw / "channel" / "benchmark.json",
            *raw.glob("channel_batches/batch_*/shard_*/output/benchmark.json"),
            *raw.glob("channel_batches/batch_*/shard_*/output/samples/*/sample.json"),
        ],
        "container": [
            raw / "container" / "benchmark.json",
            *raw.glob("container_batches/batch_*/benchmark.json"),
            *raw.glob(
                "container_batches/batch_*/shards/shard_*/benchmark/benchmark.json"
            ),
            *raw.glob(
                "container_batches/batch_*/shards/shard_*/benchmark/benchmark.partial.json"
            ),
        ],
        "mixed": [
            raw / "mixed" / "benchmark.json",
            *raw.glob("mixed_shards_batch_*/shard_*/benchmark.json"),
            *raw.glob("mixed_shards_batch_*/shard_*/benchmark.partial.json"),
        ],
    }
    return {
        domain: _deduplicated_episode_count(domain_paths)
        for domain, domain_paths in paths.items()
    }


def _global_snapshot(
    root: Path,
    *,
    global_target: int,
    domain_targets: dict[str, int],
    now: float,
) -> str:
    domain_counts = _domain_counts(root)
    raw_valid = sum(domain_counts.values())
    checkpoint_target = sum(domain_targets.values())
    checkpoint_progress = sum(
        min(domain_counts.get(domain, 0), target)
        for domain, target in domain_targets.items()
    )
    checkpoint_percent = (
        100.0 * checkpoint_progress / checkpoint_target
        if checkpoint_target > 0
        else 0.0
    )
    domain_text = ",".join(
        f"{domain}:{domain_counts.get(domain, 0)}/{target}"
        for domain, target in domain_targets.items()
    )
    balanced_path = root / "balanced" / "benchmark.json"
    balanced = _deduplicated_episode_count([balanced_path]) if balanced_path.exists() else 0
    started_at = _global_start_time(root, now)
    elapsed = max(now - started_at, 0.0)
    rate = raw_valid / elapsed if raw_valid > 0 and elapsed > 0 else 0.0
    eta = max(global_target - raw_valid, 0) / rate if rate > 0 else None
    return (
        f" domains={domain_text}"
        f" checkpoint_progress={checkpoint_progress}/{checkpoint_target}"
        f" ({checkpoint_percent:.1f}%)"
        f" checkpoint_remaining={max(checkpoint_target - checkpoint_progress, 0)}"
        f" checkpoint_raw_valid={raw_valid}"
        f" global_raw_valid={raw_valid}/{global_target}"
        f" balanced_materialized={balanced}/{global_target}"
        f" global_balanced={balanced}/{global_target}"
        f" global_rate={rate:.3f}/s"
        f" global_elapsed={_eta_text(elapsed)}"
        f" global_eta={_eta_text(eta)}"
    )


def _container_snapshot(root: Path, now: float, global_text: str) -> str | None:
    batch = _active_batch(root, "raw/container_batches/batch_*")
    if batch is None:
        return None
    plans = sorted(batch.glob("shards/shard_*/collection_plan.json"))
    worker_loads: list[int] = []
    started_at = batch.stat().st_mtime
    for plan in plans:
        payload = _read_json(plan, {})
        selection = payload.get("selection", {}) if isinstance(payload, dict) else {}
        worker_loads.append(int(selection.get("selected_house_count", 0)))
        started_at = min(started_at, plan.stat().st_mtime)
    total = sum(worker_loads)
    processed = _list_len(
        _preferred_shard_artifacts(
            batch,
            shard_pattern="shards/shard_*",
            final_name="benchmark/house_catalog.json",
            partial_name="benchmark/house_catalog.partial.json",
        )
    )
    valid = _list_len(
        _preferred_shard_artifacts(
            batch,
            shard_pattern="shards/shard_*",
            final_name="benchmark/benchmark.json",
            partial_name="benchmark/benchmark.partial.json",
        )
    )
    return _progress_line(
        stage="container",
        batch=batch.name,
        processed=processed,
        total=total,
        valid=valid,
        started_at=started_at,
        now=now,
        global_text=global_text,
        worker_loads=worker_loads,
    )


def _channel_snapshot(root: Path, now: float, global_text: str) -> str | None:
    batch = _active_batch(root, "raw/channel_batches/batch_*")
    if batch is None:
        return None
    input_paths = sorted(batch.glob("shard_*/input/benchmark.json"))
    worker_loads = [
        len(payload) if isinstance(payload := _read_json(path, []), list) else 0
        for path in input_paths
    ]
    total = sum(worker_loads)
    processed = 0
    started_at = batch.stat().st_mtime
    for log_path in sorted(batch.glob("shard_*/run.log")):
        try:
            text = log_path.read_text(errors="ignore")
        except OSError:
            continue
        processed += len(re.findall(r"\[(?:ok|fail)\] ep=\d+", text))
        started_at = min(started_at, log_path.stat().st_mtime)
    valid = len(list(batch.glob("shard_*/output/samples/*/sample.json")))
    return _progress_line(
        stage="channel",
        batch=batch.name,
        processed=processed,
        total=total,
        valid=valid,
        started_at=started_at,
        now=now,
        global_text=global_text,
        worker_loads=worker_loads,
    )


def _mixed_snapshot(root: Path, now: float, global_text: str) -> str | None:
    batch = _active_batch(root, "raw/mixed_shards_batch_*")
    if batch is None:
        return None
    plans = sorted(batch.glob("shard_*/candidate_plan.json"))
    worker_loads = [
        len(payload.get("candidates", []))
        if isinstance(payload := _read_json(path, {}), dict)
        else 0
        for path in plans
    ]
    total = sum(worker_loads)
    valid_rows = _preferred_shard_artifacts(
        batch,
        shard_pattern="shard_*",
        final_name="valid.json",
        partial_name="valid.partial.json",
    )
    rejected_rows = _preferred_shard_artifacts(
        batch,
        shard_pattern="shard_*",
        final_name="rejected.json",
        partial_name="rejected.partial.json",
    )
    processed = _list_len([*valid_rows, *rejected_rows])
    valid = _list_len(
        _preferred_shard_artifacts(
            batch,
            shard_pattern="shard_*",
            final_name="benchmark.json",
            partial_name="benchmark.partial.json",
        )
    )
    started_at = min(
        [batch.stat().st_mtime, *[path.stat().st_mtime for path in plans]]
    )
    return _progress_line(
        stage="mixed",
        batch=batch.name,
        processed=processed,
        total=total,
        valid=valid,
        started_at=started_at,
        now=now,
        global_text=global_text,
        worker_loads=worker_loads,
    )


def snapshot(
    root: Path,
    *,
    global_target: int = 3000,
    domain_targets: dict[str, int] | None = None,
) -> str:
    now = time.time()
    if domain_targets is None:
        per_domain = global_target // 3
        domain_targets = {
            "channel": per_domain,
            "container": per_domain,
            "mixed": global_target - 2 * per_domain,
        }
    global_text = _global_snapshot(
        root,
        global_target=global_target,
        domain_targets=domain_targets,
        now=now,
    )
    for probe in (_container_snapshot, _channel_snapshot, _mixed_snapshot):
        line = probe(root, now, global_text)
        if line is not None:
            return line
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S%z", time.localtime(now))
    return (
        f"[collection-progress] time={timestamp} stage=idle "
        f"no active batch artifacts{global_text}"
    )


class ProgressReporter:
    def __init__(
        self,
        root: Path,
        *,
        interval_seconds: float = 60.0,
        sink: Callable[[str], None] | None = None,
        global_target: int = 3000,
        domain_targets: dict[str, int] | None = None,
    ) -> None:
        self.root = root
        self.interval_seconds = interval_seconds
        self.global_target = global_target
        self.domain_targets = domain_targets
        self.sink = sink or (lambda message: print(message, flush=True))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="interactive-nav-progress",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sink(
                    snapshot(
                        self.root,
                        global_target=self.global_target,
                        domain_targets=self.domain_targets,
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostics must not stop collection.
                self.sink(f"[collection-progress] reporter_error={exc!r}")
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
