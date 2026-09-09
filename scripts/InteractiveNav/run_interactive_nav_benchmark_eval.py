#!/usr/bin/env python3
"""Run a non-ROS InteractiveNav V3 benchmark over the three frozen domains.

This is a thin scheduler around :mod:`evaluation.benchmark_runner`.  It is
intended for ordinary/external policies (``--policy factory``), and deliberately
does not start a ROS master or pass a live task to the policy.  The policy sees
only the public ``PublicEpisode``/``PolicyObservation`` contract implemented by
``benchmark_policies.ExternalPolicyAdapter``.  A factory may return an object
with ``reset(episode_dict)``, ``act(policy_observation)`` (or ``get_action``),
and optional ``close()``; actions may be ``PolicyAction`` instances or the
normalised dictionaries documented in ``benchmark_policies.normalize_policy_action``.

The three domain files are scheduled in a deterministic channel/container/mixed
round-robin order.  ``--workers`` (also accepted as ``--threads``) is the
number of isolated Python processes, rather than shared Python threads: each
process owns its MuJoCo/EGL context.  Progress is shown with tqdm and mirrored
as append-only human-readable records in ``progress.log`` plus an atomic
``progress.json`` snapshot.

Example (a small smoke run)::

    /home/ldl/conda_envs/mlspaces/bin/python \
      scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py \
      --output-dir /home/ldl/outputs/interactive-nav/basic_smoke \
      --episodes-per-domain 10 --workers 10 \
      --policy factory --policy-factory my_policy:build_policy

The default benchmark root is the frozen release bundled with this repository.
It is stored as deterministic ``.json.gz`` archives and decoded transparently.
Pass explicit ``--*-benchmark`` paths when evaluating another release.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import os
import sys
import time
from typing import Any, Iterable, Mapping

# These must be set before benchmark_runner imports MuJoCo/renderer-owning
# modules.  They also make accidental default GL backend selection less likely
# in a clean spawn interpreter.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:  # tqdm is a normal project dependency, but keep the CLI importable offline.
    from tqdm import tqdm
except Exception:  # pragma: no cover - exercised only in minimal environments.
    tqdm = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ``benchmark_runner`` uses one small, ROS-originated graph-rules module even
# for non-ROS replay.  Add its source package directly so the basic evaluator
# does not require sourcing a ROS workspace.
_SEMANTIC_SCRIPTS = REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "semantic_mapping_py_pkg" / "scripts"
if _SEMANTIC_SCRIPTS.is_dir() and str(_SEMANTIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_SCRIPTS))
from scripts.InteractiveNav.evaluation.benchmark_io import load_benchmark_episodes


DEFAULT_BENCHMARK_ROOT = (
    REPO_ROOT
    / "scripts"
    / "InteractiveNav"
    / "benchmarks"
    / "interactive_nav_v3_procthor10k_val_release_v1_1"
)
DOMAIN_NAMES = ("channel", "container", "mixed")


@dataclass(frozen=True)
class DomainEpisode:
    """One selected episode, retaining the source-domain provenance."""

    domain: str
    source_path: str
    source_index: int
    episode: dict[str, Any]


@dataclass(frozen=True)
class ScheduledEpisode:
    """A worker payload description independent of process-local objects."""

    item: DomainEpisode
    output_dir: str
    run_signature: str
    config: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    """Load either the release's top-level list or ``{"episodes": [...]}``."""

    _benchmark_file, episodes = load_benchmark_episodes(path)
    return episodes


def _default_domain_path(root: Path, domain: str) -> Path:
    for suffix in (".json", ".json.gz"):
        candidate = root / f"{domain}{suffix}"
        if candidate.is_file():
            return candidate
    return root / f"{domain}.json.gz"


def _domain_path_args(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.benchmark_root).expanduser()
    values = {
        "channel": Path(args.channel_benchmark).expanduser()
        if args.channel_benchmark
        else _default_domain_path(root, "channel"),
        "container": Path(args.container_benchmark).expanduser()
        if args.container_benchmark
        else _default_domain_path(root, "container"),
        "mixed": Path(args.mixed_benchmark).expanduser()
        if args.mixed_benchmark
        else _default_domain_path(root, "mixed"),
    }
    for domain, path in values.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {domain} benchmark: {path}. Pass --{domain}-benchmark or --benchmark-root."
            )
    return {domain: path.resolve() for domain, path in values.items()}


def _balanced_counts(total: int, capacities: Mapping[str, int]) -> dict[str, int]:
    """Allocate ``total`` slots with domain counts differing by at most one.

    A domain can be exhausted early (for a small custom benchmark); remaining
    slots are assigned to the currently least represented available domain.  The
    deterministic tie order is channel, container, mixed.
    """

    if total < 0:
        raise ValueError("total episode count must be non-negative")
    counts = {domain: 0 for domain in DOMAIN_NAMES}
    for _ in range(total):
        available = [
            domain
            for domain in DOMAIN_NAMES
            if counts[domain] < int(capacities.get(domain, 0))
        ]
        if not available:
            break
        domain = min(available, key=lambda value: (counts[value], DOMAIN_NAMES.index(value)))
        counts[domain] += 1
    return counts


def select_domain_episodes(
    domain_episodes: Mapping[str, list[dict[str, Any]]],
    *,
    episodes_per_domain: int | None = None,
    max_episodes: int | None = None,
) -> list[DomainEpisode]:
    """Select deterministic, approximately 1:1:1 domain episodes.

    ``episodes_per_domain`` takes precedence over ``max_episodes``.  Within a
    domain, the frozen local order is preserved.  The returned list is globally
    round-robin ordered so a bounded process queue naturally receives all three
    domains from the beginning of a run.
    """

    if episodes_per_domain is not None and episodes_per_domain < 1:
        raise ValueError("episodes_per_domain must be >= 1")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be >= 1")
    capacities = {domain: len(domain_episodes.get(domain, [])) for domain in DOMAIN_NAMES}
    if episodes_per_domain is not None:
        counts = {
            domain: min(int(episodes_per_domain), capacities[domain]) for domain in DOMAIN_NAMES
        }
    elif max_episodes is not None:
        counts = _balanced_counts(int(max_episodes), capacities)
    else:
        counts = capacities

    # Round-robin by domain, not by source index.  This gives each bounded
    # worker queue an even mixture even when episodes have different durations.
    selected: list[DomainEpisode] = []
    cursors = {domain: 0 for domain in DOMAIN_NAMES}
    while any(cursors[domain] < counts[domain] for domain in DOMAIN_NAMES):
        for domain in DOMAIN_NAMES:
            cursor = cursors[domain]
            if cursor >= counts[domain]:
                continue
            episode = domain_episodes[domain][cursor]
            source_path = str(getattr(episode, "_source_path", ""))
            # ``_source_path`` is only used by internal callers that already
            # loaded multiple files.  The CLI fills the path explicitly below.
            selected.append(
                DomainEpisode(
                    domain=domain,
                    source_path=source_path,
                    source_index=cursor,
                    episode=dict(episode),
                )
            )
            cursors[domain] += 1
    return selected


def _attach_source_paths(
    selected: Iterable[DomainEpisode], paths: Mapping[str, Path]
) -> list[DomainEpisode]:
    return [
        DomainEpisode(
            domain=item.domain,
            source_path=str(paths[item.domain].resolve()),
            source_index=item.source_index,
            episode=item.episode,
        )
        for item in selected
    ]


def _parse_policy_kwargs(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--policy-kwargs-json must decode to a JSON object")
    return parsed


def _normalise_config_for_worker(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized["benchmark"] = Path(normalized["benchmark"])
    normalized["output_dir"] = Path(normalized["output_dir"])
    image_resolution = normalized.get("image_resolution")
    normalized["image_resolution"] = (
        None if image_resolution is None else tuple(image_resolution)
    )
    return normalized


def _run_episode_worker(payload: ScheduledEpisode) -> dict[str, Any]:
    """Spawn-safe worker; imports the evaluator only inside the child."""

    # Scene preparation creates symlinks below this directory.  A shared mirror
    # races when independently spawned MuJoCo workers prepare their first scene
    # at the same time, so each process owns one stable mirror for the run.
    run_root = Path(payload.output_dir).parent
    worker_mirror = run_root / "_runtime_cache" / "scene_mirror_workers" / str(os.getpid())
    worker_mirror.mkdir(parents=True, exist_ok=True)
    os.environ["INTERACTIVE_NAV_SCENE_MIRROR"] = str(worker_mirror)
    os.environ["INTERACTIVE_NAV_DEFAULT_SCENE_MIRROR"] = str(worker_mirror)

    # Importing here keeps the coordinator useful for manifest/selection checks
    # on machines without a configured simulator, while spawn workers still get
    # the canonical evaluator implementation.
    try:
        from scripts.InteractiveNav.evaluation.benchmark_runner import (
            BenchmarkEvaluationConfig,
            evaluate_episode,
        )
        config = BenchmarkEvaluationConfig(**_normalise_config_for_worker(payload.config))
        result = evaluate_episode(
            config,
            int(payload.item.source_index),
            dict(payload.item.episode),
            run_signature=payload.run_signature,
        )
        # Keep source provenance outside the evaluator's public policy trace.  Do
        # this inside the guarded section as well: a malformed external runner
        # result (e.g. ``None`` or a non-mapping) must become one scored
        # exception row rather than aborting ``Pool.imap_unordered``.
        result = dict(result)
    except Exception as exc:
        # A broken scene/policy must become a scored exception row instead of
        # terminating an otherwise useful mixed-domain batch.
        return _exception_result(payload.item, exc)
    # The evaluator's per-episode JSON remains protocol-compatible; these fields
    # are wrapper-owned and appear in aggregate results only.
    result["benchmark_domain"] = payload.item.domain
    result["source_episode_index"] = int(payload.item.source_index)
    result["source_benchmark"] = payload.item.source_path
    result["episode_key"] = f"{payload.item.domain}:{int(payload.item.source_index):04d}"
    return result


def _exception_result(item: DomainEpisode, exc: BaseException) -> dict[str, Any]:
    nav_payload = item.episode.get("interactive_nav", {})
    nav = nav_payload if isinstance(nav_payload, dict) else {}
    try:
        house_index = int(item.episode.get("house_index", -1))
    except (TypeError, ValueError):
        house_index = -1
    return {
        "episode_index": int(item.source_index),
        "case_id": str(nav.get("case_id", f"{item.domain}_{item.source_index}")),
        "house_index": house_index,
        "domains": list(nav.get("interaction_domains", [item.domain])),
        "recipe": nav.get("legacy_case_type"),
        "interaction_types": [],
        "path_length_bin": None,
        "interaction_requirement": str(nav.get("interaction_requirement", "unknown")),
        "policy_name": "exception",
        "uses_oracle_gt": False,
        "status": "exception",
        "success": False,
        "task_success": False,
        "interaction_conditioned_success": False,
        "nav_success": False,
        "required_interaction_success": False,
        "sequence_success": False,
        "non_interaction_success": None,
        "terminal_reason": "exception",
        "step_count": 0,
        "navigation_step_count": 0,
        "view_action_count": 0,
        "interaction_action_count": 0,
        "correct_interaction_action_count": 0,
        "extra_interaction_action_count": 0,
        "invalid_interaction_action_count": 0,
        "navigation_path_length_m": 0.0,
        "reference_path_length_m": None,
        "spl": 0.0,
        "navigation_simulated_seconds": 0.0,
        "interaction_simulated_seconds": 0.0,
        "total_simulated_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "target_distance_m": None,
        "target_visibility_fraction": None,
        "interaction_attempts": [],
        "scoring_eligible": False,
        "scoring_exclusion_reasons": ["runtime_exception"],
        "error": f"{type(exc).__name__}: {exc}",
        "benchmark_domain": item.domain,
        "source_episode_index": int(item.source_index),
        "source_benchmark": item.source_path,
        "episode_key": f"{item.domain}:{int(item.source_index):04d}",
        "topdown_path": None,
        "topdown_metadata_path": None,
        "topdown_exists": False,
        "topdown_error": None,
    }


def _render_basic_topdowns(rows: list[dict[str, Any]], *, enabled: bool) -> dict[str, int]:
    """Render basic-eval reports after scoring, serially.

    Static scene reconstruction is considerably more expensive than writing a
    trace and is not safe to run concurrently with every simulator worker.  A
    post-pass keeps benchmark timing/SR independent of visualisation while
    ensuring each completed episode gets the same start/GT/trajectory report.
    """

    counts = {"requested": 0, "created": 0, "missing": 0, "failed": 0}
    if not enabled:
        return counts
    try:
        from scripts.InteractiveNav.evaluation.episode_topdown import render_episode_topdown
    except Exception as exc:  # pragma: no cover - dependency-specific fallback
        for row in rows:
            row["topdown_exists"] = False
            row["topdown_error"] = f"renderer import: {type(exc).__name__}: {exc}"
            counts["failed"] += 1
        return counts

    for row in rows:
        counts["requested"] += 1
        trace_value = row.get("trace_path") or row.get("episode_result_path")
        benchmark_value = row.get("source_benchmark")
        if not trace_value or not benchmark_value:
            row["topdown_exists"] = False
            row["topdown_error"] = "missing trace_path or source_benchmark"
            counts["missing"] += 1
            continue
        trace_path = Path(str(trace_value))
        output_path = trace_path.with_name("episode_topdown.png")
        row["topdown_path"] = str(output_path)
        row["topdown_metadata_path"] = str(output_path.with_suffix(".json"))
        if output_path.is_file() and output_path.stat().st_size > 0:
            row["topdown_exists"] = True
            row["topdown_error"] = None
            counts["created"] += 1
            continue
        try:
            render_episode_topdown(
                episode_result_path=trace_path,
                benchmark_path=Path(str(benchmark_value)),
                debug_dir=trace_path.parent / "debug",
                output_path=output_path,
                private_context_path=trace_path.with_name("episode_visualization.json"),
            )
            row["topdown_exists"] = output_path.is_file() and output_path.stat().st_size > 0
            row["topdown_error"] = None if row["topdown_exists"] else "renderer returned without output"
            counts["created" if row["topdown_exists"] else "failed"] += 1
        except Exception as exc:  # preserve the scored result and report the artifact error
            row["topdown_exists"] = False
            row["topdown_error"] = f"{type(exc).__name__}: {exc}"
            counts["failed"] += 1
    return counts


def _episode_sr(rows: list[dict[str, Any]]) -> float | None:
    """Formal wrapper SR from ``result.success`` on scoring-eligible rows.

    ``scoring_eligible`` is the denominator contract: an eligible row with a
    malformed/missing ``success`` field is still a failed planned episode,
    rather than silently disappearing from the denominator.
    """

    eligible = [row for row in rows if bool(row.get("scoring_eligible", True))]
    if not eligible:
        return None
    return float(sum(bool(row.get("success", False)) for row in eligible) / len(eligible))


def _episode_nav_sr(rows: list[dict[str, Any]]) -> float | None:
    eligible = [row for row in rows if bool(row.get("scoring_eligible", True))]
    if not eligible:
        return None
    return float(
        sum(bool(row.get("nav_success", row.get("task_success", False))) for row in eligible)
        / len(eligible)
    )


def _planned_success_rate(rows: list[dict[str, Any]], planned_count: int | None = None) -> float | None:
    """Return success/plan, counting exceptions or missing rows as failures."""

    denominator = len(rows) if planned_count is None else int(planned_count)
    if denominator <= 0:
        return None
    successes = sum(bool(row.get("success", False)) for row in rows)
    return float(successes / denominator)


def _domain_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.InteractiveNav.evaluation.benchmark_metrics import summarise_results

    summary = summarise_results(rows)
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_NAMES:
        domain_rows = [row for row in rows if row.get("benchmark_domain") == domain]
        if not domain_rows:
            continue
        by_domain[domain] = {
            "episode_count": len(domain_rows),
            "sr": _episode_sr(domain_rows),
            "planned_sr": _planned_success_rate(domain_rows),
            "nav_sr": _episode_nav_sr(domain_rows),
            "task_sr": _rate(domain_rows, "task_success", "nav_success"),
            "interaction_conditioned_sr": _rate(
                domain_rows, "interaction_conditioned_success", "success"
            ),
            "exception_count": sum(row.get("status") == "exception" for row in domain_rows),
            "scoring_eligible_count": sum(
                bool(row.get("scoring_eligible", True)) for row in domain_rows
            ),
        }
    summary["domain_summary"] = by_domain
    summary["domain_counts"] = {
        domain: sum(1 for row in rows if row.get("benchmark_domain") == domain)
        for domain in DOMAIN_NAMES
    }
    return summary


def _rate(rows: list[dict[str, Any]], key: str, fallback: str) -> float | None:
    values = [row.get(key, row.get(fallback)) for row in rows]
    values = [value for value in values if value is not None]
    return None if not values else float(sum(bool(value) for value in values) / len(values))


def _write_summary_csv(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write a compact domain/overall SR table for spreadsheet inspection."""

    fields = (
        "group",
        "episode_count",
        "scoring_eligible_count",
        "sr",
        "planned_sr",
        "nav_sr",
        "task_sr",
        "interaction_conditioned_sr",
        "exception_count",
        "topdown_exists_count",
        "topdown_path",
        "topdown_exists",
        "topdown_error",
    )
    records: list[dict[str, Any]] = []
    overall = {
        "group": "overall",
        "episode_count": len(rows),
        "scoring_eligible_count": sum(bool(row.get("scoring_eligible", True)) for row in rows),
        "sr": _episode_sr(rows),
        "planned_sr": _planned_success_rate(rows),
        "nav_sr": _episode_nav_sr(rows),
        "task_sr": _rate(rows, "task_success", "nav_success"),
        "interaction_conditioned_sr": _rate(rows, "interaction_conditioned_success", "success"),
        "exception_count": sum(row.get("status") == "exception" for row in rows),
        "topdown_exists_count": sum(bool(row.get("topdown_exists")) for row in rows),
        "topdown_path": None,
        "topdown_exists": None,
        "topdown_error": None,
    }
    records.append(overall)
    for domain, values in summary.get("domain_summary", {}).items():
        records.append(
            {
                "group": domain,
                "episode_count": values.get("episode_count"),
                "scoring_eligible_count": values.get("scoring_eligible_count"),
                "sr": values.get("sr"),
                "planned_sr": values.get("planned_sr"),
                "nav_sr": values.get("nav_sr"),
                "task_sr": values.get("task_sr"),
                "interaction_conditioned_sr": values.get("interaction_conditioned_sr"),
                "exception_count": values.get("exception_count"),
                "topdown_exists_count": sum(
                    bool(row.get("topdown_exists"))
                    for row in rows
                    if row.get("benchmark_domain") == domain
                ),
                "topdown_path": None,
                "topdown_exists": None,
                "topdown_error": None,
            }
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


class _Progress:
    """Tqdm terminal bar plus append-only progress log/snapshot."""

    def __init__(self, output_dir: Path, total: int) -> None:
        self.output_dir = output_dir
        self.total = int(total)
        self.started = time.monotonic()
        self.rows: list[dict[str, Any]] = []
        self.log_path = output_dir / "progress.log"
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        self.bar = (
            tqdm(total=self.total, desc="interactive-nav-eval", unit="ep", dynamic_ncols=True)
            if tqdm is not None
            else None
        )

    def update(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        completed = len(self.rows)
        elapsed = max(time.monotonic() - self.started, 1e-9)
        sr = _episode_sr(self.rows)
        domain_counts = {
            domain: sum(item.get("benchmark_domain") == domain for item in self.rows)
            for domain in DOMAIN_NAMES
        }
        domain_sr = {
            domain: _episode_sr(
                [item for item in self.rows if item.get("benchmark_domain") == domain]
            )
            for domain in DOMAIN_NAMES
        }
        snapshot = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed": completed,
            "total": self.total,
            "fraction": 0.0 if not self.total else completed / self.total,
            "sr": sr,
            "planned_sr": _planned_success_rate(self.rows, self.total),
            "nav_sr": _episode_nav_sr(self.rows),
            "scoring_eligible": sum(bool(item.get("scoring_eligible", True)) for item in self.rows),
            "task_sr": _rate(self.rows, "task_success", "nav_success"),
            "interaction_conditioned_sr": _rate(
                self.rows, "interaction_conditioned_success", "success"
            ),
            "domain_counts": domain_counts,
            "domain_sr": domain_sr,
            "exception_count": sum(item.get("status") == "exception" for item in self.rows),
            "elapsed_seconds": elapsed,
            "episodes_per_second": completed / elapsed,
        }
        _atomic_json(self.output_dir / "progress.json", snapshot)
        # Keep a grep-friendly human line in addition to the machine snapshot;
        # this is intentionally append-only so a killed worker pool leaves a
        # useful progress trail.
        self.log_handle.write(
            "[tqdm] "
            f"{completed}/{self.total} ({snapshot['fraction']:.1%}) "
            f"SR={sr if sr is not None else '-'} "
            f"PlannedSR={snapshot['planned_sr'] if snapshot['planned_sr'] is not None else '-'} "
            f"NavSR={snapshot['nav_sr'] if snapshot['nav_sr'] is not None else '-'} "
            f"eligible={snapshot['scoring_eligible']}/{self.total} "
            f"channel={domain_sr['channel'] if domain_sr['channel'] is not None else '-'} "
            f"container={domain_sr['container'] if domain_sr['container'] is not None else '-'} "
            f"mixed={domain_sr['mixed'] if domain_sr['mixed'] is not None else '-'}\n"
        )
        self.log_handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
        self.log_handle.flush()
        if self.bar is not None:
            postfix = {
                "SR": "-" if sr is None else f"{sr:.3f}",
                "ch": "-" if domain_sr["channel"] is None else f"{domain_sr['channel']:.2f}",
                "ct": "-" if domain_sr["container"] is None else f"{domain_sr['container']:.2f}",
                "mx": "-" if domain_sr["mixed"] is None else f"{domain_sr['mixed']:.2f}",
            }
            self.bar.set_postfix(postfix, refresh=False)
            self.bar.update(1)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        self.log_handle.close()


def _make_config(args: argparse.Namespace, source: Path, output_dir: Path) -> dict[str, Any]:
    # Import lazily: ``--help`` and selection validation should work without
    # initializing MuJoCo, while child workers use the canonical dataclass.
    from scripts.InteractiveNav.evaluation.benchmark_runner import BenchmarkEvaluationConfig

    config = BenchmarkEvaluationConfig(
        benchmark=source,
        output_dir=output_dir,
        policy=args.policy,
        policy_factory=args.policy_factory,
        policy_kwargs=_parse_policy_kwargs(args.policy_kwargs_json),
        # evaluate_episode owns one simulator; outer workers provide isolation.
        workers=1,
        max_steps=int(args.max_steps),
        step_budget_mode=str(args.step_budget_mode),
        min_steps=int(args.min_steps),
        resume=bool(args.resume),
        record_video=bool(args.record_video),
        video_fps=float(args.video_fps),
        camera_names=list(args.camera_names),
        image_resolution=None
        if args.image_resolution is None
        else tuple(int(value) for value in args.image_resolution),
        progress_every=max(1, int(args.progress_every)),
    )
    config.validate()
    return asdict(config) | {
        "benchmark": str(source.resolve()),
        "output_dir": str(output_dir.resolve()),
    }


def _build_schedule(args: argparse.Namespace) -> tuple[list[ScheduledEpisode], dict[str, Any]]:
    paths = _domain_path_args(args)
    loaded = {domain: _read_episodes(path) for domain, path in paths.items()}
    selected = _attach_source_paths(
        select_domain_episodes(
            loaded,
            episodes_per_domain=args.episodes_per_domain,
            max_episodes=args.max_episodes,
        ),
        paths,
    )
    if not selected:
        raise ValueError("No benchmark episodes selected")

    output_dir = Path(args.output_dir).resolve()
    domain_output_dirs = {domain: output_dir / domain for domain in DOMAIN_NAMES}
    schedules: list[ScheduledEpisode] = []
    manifest_domains: dict[str, Any] = {}
    # Signatures are domain-specific and include the selected local indices;
    # this keeps --resume safe when one domain's source or selection changes.
    from scripts.InteractiveNav.evaluation import benchmark_runner

    for domain in DOMAIN_NAMES:
        domain_items = [item for item in selected if item.domain == domain]
        if not domain_items:
            continue
        source = paths[domain]
        domain_config = _make_config(args, source, domain_output_dirs[domain])
        config_obj = benchmark_runner.BenchmarkEvaluationConfig(
            **_normalise_config_for_worker(domain_config)
        )
        signature, _ = benchmark_runner._run_signature(
            config_obj,
            _sha256(source),
            [item.source_index for item in domain_items],
        )
        manifest_domains[domain] = {
            "benchmark": str(source),
            "benchmark_sha256": _sha256(source),
            "episode_indices": [item.source_index for item in domain_items],
            "count": len(domain_items),
            "run_signature": signature,
            "output_dir": str(domain_output_dirs[domain]),
        }
        for item in domain_items:
            schedules.append(
                ScheduledEpisode(
                    item=item,
                    output_dir=str(domain_output_dirs[domain]),
                    run_signature=signature,
                    config=domain_config,
                )
            )

    # ``select_domain_episodes`` already creates a round-robin sequence; retain
    # that order after per-domain signature construction.
    order = {(item.domain, item.source_index): index for index, item in enumerate(selected)}
    schedules.sort(key=lambda item: order[(item.item.domain, item.item.source_index)])
    worker_domain_counts = {
        str(worker): {domain: 0 for domain in DOMAIN_NAMES}
        for worker in range(max(1, int(args.workers)))
    }
    for ordinal, item in enumerate(selected):
        worker = str(ordinal % max(1, int(args.workers)))
        worker_domain_counts[worker][item.domain] += 1
    manifest = {
        "schema_version": "interactive_nav_v3_mixed_eval_v1",

        "created_at": datetime.now(timezone.utc).isoformat(),
        "workers": int(args.workers),
        "policy": args.policy,
        "policy_factory": args.policy_factory,
        "episodes_per_domain": args.episodes_per_domain,
        "max_episodes": args.max_episodes,
        "selected_episode_count": len(schedules),
        "domain_counts": {domain: sum(item.domain == domain for item in selected) for domain in DOMAIN_NAMES},
        "planned_worker_domain_counts": worker_domain_counts,
        "domains": manifest_domains,
        "record_video": bool(args.record_video),
        "render_topdown": bool(args.render_topdown),
    }
    return schedules, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help=f"Frozen benchmark directory (default: {DEFAULT_BENCHMARK_ROOT}).",
    )
    parser.add_argument("--channel-benchmark", type=Path)
    parser.add_argument("--container-benchmark", type=Path)
    parser.add_argument("--mixed-benchmark", type=Path)
    parser.add_argument("--workers", "--threads", dest="workers", type=int, default=1)
    parser.add_argument(
        "--episodes-per-domain",
        type=int,
        help="Select this many episodes from each domain (takes precedence over --max-episodes).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Total selected episodes, allocated as evenly as possible across domains.",
    )
    parser.add_argument(
        "--policy",
        choices=("noop", "scripted_oracle", "factory"),
        default="noop",
        help="Non-ROS policy adapter. Use factory for an external policy module.",
    )
    parser.add_argument("--policy-factory", help="External factory as module.path:callable (required for factory).")
    parser.add_argument("--policy-kwargs-json", help="JSON object forwarded to the external policy factory.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--step-budget-mode", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--min-steps", type=int, default=300)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--render-topdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write episode_topdown.png/.json after scoring (default: enabled).",
    )
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--camera-names", nargs="+", default=["head_camera"])
    parser.add_argument("--image-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=[640, 480])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write the schedule manifest without starting MuJoCo workers.",
    )
    parser.add_argument("--progress-every", type=int, default=1, help="Retained for config/signature compatibility; progress always updates per episode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers/--threads must be >= 1")
    if args.episodes_per_domain is not None and args.episodes_per_domain < 1:
        parser.error("--episodes-per-domain must be >= 1")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be >= 1")
    if args.policy == "factory" and not args.policy_factory:
        parser.error("--policy-factory is required with --policy factory")
    if args.policy != "factory" and args.policy_factory:
        parser.error("--policy-factory is only valid with --policy factory")
    if args.record_video and args.workers > 1:
        # Video remains supported in parallel; this note makes the potentially
        # high disk/renderer cost explicit without changing user intent.
        print("[benchmark-eval] recording enabled for every episode; expect higher disk and EGL use", file=sys.stderr)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep temporary files and common ML/renderer caches on the large volume
    # alongside the run, never in /tmp or /root/.cache by accident.
    runtime_tmp = output_dir / "_runtime_tmp"
    runtime_cache = output_dir / "_runtime_cache"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    runtime_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    os.environ["XDG_CACHE_HOME"] = str(runtime_cache / "xdg")
    os.environ["HF_HOME"] = str(runtime_cache / "hf")
    os.environ["TORCH_HOME"] = str(runtime_cache / "torch")
    os.environ["CUDA_CACHE_PATH"] = str(runtime_cache / "cuda")
    os.environ["NLTK_DATA"] = str(runtime_cache / "nltk")
    os.environ["MPLCONFIGDIR"] = str(runtime_cache / "mpl")
    os.environ["INTERACTIVE_NAV_SCENE_MIRROR"] = str(runtime_cache / "scene_mirror")
    os.environ["INTERACTIVE_NAV_DEFAULT_SCENE_MIRROR"] = str(runtime_cache / "scene_mirror")
    for cache_path in (
        runtime_cache / "xdg",
        runtime_cache / "hf",
        runtime_cache / "torch",
        runtime_cache / "cuda",
        runtime_cache / "nltk",
        runtime_cache / "mpl",
        runtime_cache / "scene_mirror",
    ):
        cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    schedules, manifest = _build_schedule(args)
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already contains {manifest_path}; use --resume or a new directory"
        )
    if manifest_path.exists() and args.resume:
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot resume: invalid {manifest_path}") from exc
        # Worker count may change on resume, but benchmark/policy/selection
        # identity must remain exact so prior traces cannot be mixed silently.
        for key in (
            "schema_version",
            "policy",
            "policy_factory",
            "episodes_per_domain",
            "max_episodes",
            "domain_counts",
            "domains",
        ):
            if previous_manifest.get(key) != manifest.get(key):
                raise ValueError(
                    f"--resume refused: manifest field {key!r} differs from the current plan"
                )
    else:
        _atomic_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    progress = _Progress(output_dir, len(schedules))
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        if args.workers == 1:
            for payload in schedules:
                try:
                    row = _run_episode_worker(payload)
                except Exception as exc:  # keep the batch moving and score failure.
                    row = _exception_result(payload.item, exc)
                rows.append(row)
                progress.update(row)
        else:
            with multiprocessing.get_context("spawn").Pool(processes=args.workers) as pool:
                # imap_unordered preserves bounded queueing and emits results as
                # soon as each isolated simulator finishes.
                for row in pool.imap_unordered(_run_episode_worker, schedules, chunksize=1):
                    # Worker results carry source provenance, so completion order
                    # does not affect the final deterministic sort.
                    rows.append(row)
                    progress.update(row)
    finally:
        progress.close()

    rows.sort(key=lambda row: (DOMAIN_NAMES.index(str(row.get("benchmark_domain", "mixed"))), int(row.get("source_episode_index", row.get("episode_index", 0)))))
    topdown_counts = _render_basic_topdowns(rows, enabled=bool(args.render_topdown))
    summary = _domain_summary(rows)
    summary.update(
        {
            "schema_version": "interactive_nav_v3_mixed_eval_v1",
            "workers": int(args.workers),
            "elapsed_seconds": time.monotonic() - started,
            "result_count": len(rows),
            "planned_episode_count": len(schedules),
            "completed_episode_count": len(rows),
            "scoring_eligible_episode_count": sum(
                bool(row.get("scoring_eligible", True)) for row in rows
            ),
            "exception_count": sum(row.get("status") == "exception" for row in rows),
            # Count ordinary policy failures as well as runtime exceptions and
            # missing worker rows; SR=0 must not be reported as "no failures".
            "failed_or_incomplete_episode_count": sum(
                not bool(row.get("success", False)) for row in rows
            ) + max(0, len(schedules) - len(rows)),
            "sr": _episode_sr(rows),
            "planned_sr": _planned_success_rate(rows, len(schedules)),
            "nav_sr": _episode_nav_sr(rows),
            "policy": args.policy,
            "policy_factory": args.policy_factory,
            "benchmark_domains": manifest["domains"],
            "topdown": topdown_counts,
            "topdown_artifact_count": sum(bool(row.get("topdown_exists")) for row in rows),
            "topdown_failure_count": sum(
                bool(args.render_topdown) and not bool(row.get("topdown_exists")) for row in rows
            ),
        }
    )
    # Keep both the canonical evaluator aggregate and explicit wrapper metadata.
    _atomic_json(output_dir / "results.json", rows)
    _atomic_json(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
