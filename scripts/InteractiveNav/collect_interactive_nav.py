#!/usr/bin/env python3
"""Unified InteractiveNav v3 collection entry point.

The command deliberately keeps scene discovery/seeding separate from the existing
domain-specific MuJoCo collectors.  This makes a train-scene run reproducible while
preserving the validated door/container/mixed generation code.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from scripts.InteractiveNav import interactive_nav_v3 as v3
from scripts.InteractiveNav.collection.config import CollectionConfig, load_collection_config
from scripts.InteractiveNav.collection.scene_source import build_scene_manifest
from scripts.InteractiveNav.collection.seed_builder import (
    build_house_seed_episodes,
    load_template_episode,
    write_nav_benchmark_source,
)
from scripts.InteractiveNav.collection.full_rollout_recorder import validate_full_rollout
from scripts.InteractiveNav.collection.progress_reporter import ProgressReporter, _domain_counts
from scripts.InteractiveNav.select_container_interaction_candidates import (
    build_dynamic_collection_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_fingerprint(
    config: CollectionConfig,
    *,
    seed_benchmark: Path,
    rough_paths: Iterable[Path],
) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    runtime = payload.get("runtime", {})
    # Worker count may change for a performance comparison; it must not change
    # the selected tasks or their deterministic seed.  Likewise the light
    # scheduler only changes how the same deterministic domain queues are
    # executed, so a sequential run can safely resume with domain_parallel.
    runtime.pop("workers", None)
    runtime.pop("light_scheduler", None)
    runtime.pop("domain_wave_items_per_worker", None)
    runtime.pop("resume", None)
    runtime.pop("save_images", None)
    runtime.pop("save_plots", None)
    payload["runtime"] = runtime
    # This controls final-materialization filtering only. It does not alter
    # deterministic raw collection cases, so existing raw checkpoints remain
    # safely resumable when toggled.
    payload.get("balance", {}).pop("enforce_max_samples_per_house", None)
    resolved_seed_benchmark = seed_benchmark.resolve()
    resolved_rough_paths = [path.resolve() for path in sorted(rough_paths)]
    return {
        "schema_version": "interactive_nav_collection_fingerprint_v1",
        "config": payload,
        # Config paths may be relative while legacy outputs used absolute
        # paths.  The resolved path identifies the same immutable source in
        # both cases and avoids a false resume mismatch.
        "seed_benchmark": str(resolved_seed_benchmark),
        "seed_benchmark_sha256": _sha256_file(resolved_seed_benchmark),
        "rough_catalogs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in resolved_rough_paths
            if path.exists()
        ],
    }


def _ensure_collection_fingerprint(
    config: CollectionConfig,
    *,
    root: Path,
    seed_benchmark: Path,
    rough_paths: Iterable[Path],
) -> None:
    path = root / "collection.fingerprint.json"
    current = _collection_fingerprint(
        config, seed_benchmark=seed_benchmark, rough_paths=rough_paths
    )
    if path.exists():
        if read_json(path) != current:
            raise RuntimeError(
                f"Collection fingerprint mismatch; refusing to reuse {root}. "
                "Use a new output root for changed source/config/catalogs."
            )
        return
    artifact_paths = [
        root / name
        for name in ("balanced", "full", "channel", "container", "mixed", "mixed_shards")
    ]
    raw_path = root / "raw"
    raw_artifacts = False
    if raw_path.exists():
        raw_artifacts = any(
            (raw_path / name).exists()
            for name in ("channel", "container", "mixed", "channel_shards")
        ) or any(raw_path.glob("mixed_shards*"))
    existing_artifacts = any(path.exists() for path in artifact_paths) or (
        raw_artifacts
    )
    if existing_artifacts:
        raise RuntimeError(
            f"Existing collection output lacks a fingerprint: {root}. "
            "Refusing silent resume; move it aside or start a new output root."
        )
    write_json(path, current)


def load_episodes(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    return list(episodes)


def output_root(config: CollectionConfig) -> Path:
    root = config.output.root
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_benchmark_path(config: CollectionConfig) -> Path:
    if config.source.kind == "nav_benchmark":
        return output_root(config) / "source" / "benchmark.json"
    return output_root(config) / "seeds" / "benchmark.json"


def save_resolved_config(config: CollectionConfig) -> None:
    import yaml

    path = output_root(config) / "config.resolved.yaml"
    payload = config.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def run_command(command: list[str], *, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def split_even(values: list[int], workers: int) -> list[list[int]]:
    if not values:
        return []
    shards = [[] for _ in range(min(max(1, workers), len(values)))]
    for index, value in enumerate(values):
        shards[index % len(shards)].append(value)
    return shards


def _split_round_robin(values: list[Any], workers: int) -> list[list[Any]]:
    if not values:
        return []
    shards = [[] for _ in range(min(max(1, workers), len(values)))]
    for index, value in enumerate(values):
        shards[index % len(shards)].append(value)
    return shards


def _stable_house_order(
    house_indices: Iterable[int], *, seed: int, domain: str
) -> list[int]:
    return sorted(
        {int(value) for value in house_indices},
        key=lambda house_index: (
            hashlib.sha256(
                f"{int(seed)}::{domain}::{house_index}".encode()
            ).hexdigest(),
            house_index,
        ),
    )


def _checkpoint_target_counts(
    config: CollectionConfig, checkpoint_per_domain: int | None
) -> dict[str, int]:
    final_targets = config.balance.target_counts()
    if checkpoint_per_domain is None:
        return final_targets
    if checkpoint_per_domain < 1:
        raise ValueError("checkpoint_per_domain must be positive")
    if any(checkpoint_per_domain > value for value in final_targets.values()):
        raise ValueError(
            "checkpoint_per_domain cannot exceed the configured final domain target"
        )
    return {domain: checkpoint_per_domain for domain in final_targets}


def _checkpoint_house_caps(
    config: CollectionConfig,
    *,
    targets: dict[str, int],
    seed_episodes: list[dict[str, Any]],
    container_rough: Path,
    mixed_rough: Path,
) -> dict[str, int]:
    source_houses = {int(episode["house_index"]) for episode in seed_episodes}
    container_payload = read_json(container_rough)
    container_houses = {
        int(row["house_index"])
        for row in container_payload.get("houses", [])
        if int(row.get("strict_pair_count", 0)) > 0
        and int(row["house_index"]) in source_houses
    }
    mixed_payload = read_json(mixed_rough)
    mixed_houses = {
        int(row["house_index"])
        for row in mixed_payload.get("candidates", [])
        if row.get("rough_candidate_type") == "mixed_required_verified"
        and row.get("mixed_required_verified") is True
        and int(row["house_index"]) in source_houses
    }
    eligible_counts = {
        "channel": len(source_houses),
        "container": len(container_houses),
        "mixed": len(mixed_houses),
    }
    return {
        domain: min(
            config.balance.max_samples_per_house[domain],
            max(1, math.ceil(targets[domain] / max(eligible_counts[domain], 1))),
        )
        for domain in targets
    }


def _domain_worker_allocation(
    total_workers: int,
    active_domains: Iterable[str] = ("channel", "container", "mixed"),
    *,
    current_counts: dict[str, int] | None = None,
    targets: dict[str, int] | None = None,
) -> dict[str, int]:
    """Allocate a global simulator budget to the domains still needing data."""
    active = [domain for domain in ("channel", "container", "mixed") if domain in set(active_domains)]
    if not active:
        return {}
    if total_workers < len(active):
        raise ValueError(
            "domain_parallel needs at least one total worker for every active domain"
        )
    allocation = {domain: 1 for domain in active}
    counts = current_counts or {}
    target_counts = targets or {}
    # Allocate released slots to the largest remaining normalized deficit.
    # Ties intentionally favor Mixed, then Container, because they are the
    # expensive domains and should receive handoff capacity first.
    priority = {"mixed": 0, "container": 1, "channel": 2}
    for _ in range(total_workers - len(active)):
        chosen = max(
            active,
            key=lambda domain: (
                (max(target_counts.get(domain, 0) - counts.get(domain, 0), 0)
                 / max(target_counts.get(domain, 1), 1)),
                -priority[domain],
            ),
        )
        allocation[chosen] += 1
    return allocation


def _channel_selected_count(
    raw_root: Path, *, target: int, per_house: int
) -> int:
    """Count Channel samples with the same recipe/house-cap policy as collection."""
    sample_paths = list(
        (raw_root / "channel_shards").glob("shard_*/output/samples/*/sample.json")
    ) + list(
        raw_root.glob("channel_batches/batch_*/shard_*/output/samples/*/sample.json")
    )
    samples = []
    benchmark_path = raw_root / "channel" / "benchmark.json"
    if benchmark_path.exists():
        samples.extend(load_episodes(benchmark_path))
    for sample_path in sorted(sample_paths):
        try:
            sample = read_json(sample_path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(sample, dict) and "interactive_nav" in sample:
            samples.append(sample)
    allowed_recipes = {
        "single_path_door_closed",
        "distractor_doors_closed",
        "mixed_critical_and_distractor_closed",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in _deduplicate(samples):
        case_type = _case_type(episode)
        if case_type in allowed_recipes:
            grouped[case_type].append(episode)
    quota = {
        "single_path_door_closed": round(target * 0.60),
        "distractor_doors_closed": round(target * 0.20),
    }
    quota["mixed_critical_and_distractor_closed"] = target - sum(quota.values())
    return len(
        _select_channel_with_soft_recipe_quotas(
            grouped, quota, limit=target, per_house=per_house
        )
    )


def _container_selected_count(
    raw_root: Path, *, target: int, per_house: int
) -> int:
    episodes: list[dict[str, Any]] = []
    paths = [
        raw_root / "container" / "benchmark.json",
        *sorted(raw_root.glob("container_batches/batch_*/benchmark.json")),
        *sorted(
            raw_root.glob(
                "container_batches/batch_*/shards/shard_*/benchmark/benchmark.partial.json"
            )
        ),
        *sorted(
            (raw_root / "container" / "shards").glob(
                "shard_*/benchmark/benchmark.partial.json"
            )
        ),
    ]
    for path in paths:
        if path.exists():
            episodes.extend(load_episodes(path))
    required = [
        episode
        for episode in _deduplicate(episodes)
        if episode.get("interactive_nav", {}).get("interaction_requirement")
        == "required"
    ]
    return len(_round_robin_by_house(required, target, per_house))


def _mixed_selected_count(
    config: CollectionConfig,
    raw_root: Path,
    *,
    target: int,
    per_house: int,
) -> int:
    episodes: list[dict[str, Any]] = []
    for path in [
        raw_root / "mixed" / "benchmark.json",
        raw_root / "mixed" / "benchmark.partial.json",
        *sorted(raw_root.glob("mixed_shards*/shard_*/benchmark.json")),
        *sorted(raw_root.glob("mixed_shards*/shard_*/benchmark.partial.json")),
    ]:
        if path.exists():
            episodes.extend(load_episodes(path))
    required = [
        episode
        for episode in _deduplicate(episodes)
        if episode.get("interactive_nav", {}).get("interaction_domains")
        == ["channel", "container"]
        and episode.get("interactive_nav", {}).get("interaction_requirement")
        == "required"
    ]
    return len(
        _select_balanced_domain_episodes(
            required,
            domain="mixed",
            limit=target,
            per_house=per_house,
            balance_target_categories=config.balance.balance_target_categories == "equal",
            balance_path_lengths=config.balance.balance_path_lengths == "equal",
            path_length_bins_m=config.balance.path_length_bins_m,
            relax_house_cap_if_needed=config.balance.relax_house_cap_if_needed,
        )
    )


def _domain_selected_counts(
    config: CollectionConfig,
    raw_root: Path,
    *,
    targets: dict[str, int],
    per_house_limits: dict[str, int],
) -> dict[str, int]:
    return {
        "channel": _channel_selected_count(
            raw_root,
            target=targets["channel"],
            per_house=per_house_limits["channel"],
        ),
        "container": _container_selected_count(
            raw_root,
            target=targets["container"],
            per_house=per_house_limits["container"],
        ),
        "mixed": _mixed_selected_count(
            config,
            raw_root,
            target=targets["mixed"],
            per_house=per_house_limits["mixed"],
        ),
    }


def _episode_source_index(episode: dict[str, Any], local_index: int) -> int:
    return int(
        episode.get("seed_generation", {}).get("source_episode_index", local_index)
    )


def _mixed_candidate_plan_entries(
    rough_paths: list[Path],
    seed_episodes: list[dict[str, Any]],
    *,
    source_variants_per_pair: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    available_sources = {
        _episode_source_index(episode, local_index)
        for local_index, episode in enumerate(seed_episodes)
    }
    available_houses = {int(episode["house_index"]) for episode in seed_episodes}
    candidates_by_case: dict[str, dict[str, Any]] = {}
    for rough_path in rough_paths:
        for candidate in read_json(rough_path).get("candidates", []):
            if candidate.get("rough_candidate_type") != "mixed_required_verified":
                continue
            if candidate.get("mixed_required_verified") is not True:
                continue
            if int(candidate["house_index"]) not in available_houses:
                continue
            candidates_by_case.setdefault(str(candidate["case_id"]), candidate)

    entries: list[dict[str, Any]] = []
    for case_id, candidate in candidates_by_case.items():
        options = []
        seen_sources: set[int] = set()
        for option in candidate.get("path_options", []):
            source_index = int(option["source_episode_index"])
            if source_index not in available_sources or source_index in seen_sources:
                continue
            seen_sources.add(source_index)
            options.append(
                {
                    "source_episode_index": source_index,
                    "path_length_m": float(option.get("all_open_path_length_m", 0.0)),
                }
            )
            if len(options) >= source_variants_per_pair:
                break
        if not options:
            for source_index in candidate.get("source_episode_indices", []):
                source_index = int(source_index)
                if source_index in available_sources and source_index not in seen_sources:
                    seen_sources.add(source_index)
                    options.append(
                        {
                            "source_episode_index": source_index,
                            "path_length_m": float(
                                candidate.get("all_open_path_length_m", 0.0)
                            ),
                        }
                    )
                    if len(options) >= source_variants_per_pair:
                        break
        for option in options:
            source_index = int(option["source_episode_index"])
            entries.append(
                {
                    "candidate_key": f"{case_id}::src{source_index}",
                    "case_id": case_id,
                    "house_index": int(candidate["house_index"]),
                    "source_episode_index": source_index,
                    "target_category": str(
                        candidate.get("target_category", "unknown")
                    ).lower(),
                    "path_length_m": float(option["path_length_m"]),
                    "estimated_total_interaction_count": int(
                        candidate.get("estimated_total_interaction_count", 0)
                    ),
                }
            )

    entries.sort(
        key=lambda row: (
            row["path_length_m"],
            row["estimated_total_interaction_count"],
            row["case_id"],
            row["source_episode_index"],
        )
    )
    by_house: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_house[int(entry["house_index"])].append(entry)
    ordered: list[dict[str, Any]] = []
    house_order = _stable_house_order(by_house, seed=seed, domain="mixed")
    while by_house:
        for house_index in house_order:
            if house_index not in by_house:
                continue
            ordered.append(by_house[house_index].pop(0))
            if not by_house[house_index]:
                del by_house[house_index]
    return ordered


def _write_or_validate_house_plan(
    config: CollectionConfig,
    *,
    seed_benchmark: Path,
    seed_episodes: list[dict[str, Any]],
    container_rough: Path,
    mixed_rough: Path,
) -> Path:
    source_houses = {int(episode["house_index"]) for episode in seed_episodes}
    container_houses = {
        int(row["house_index"])
        for row in read_json(container_rough).get("houses", [])
        if int(row.get("strict_pair_count", 0)) > 0
        and int(row["house_index"]) in source_houses
    }
    mixed_houses = {
        int(row["house_index"])
        for row in read_json(mixed_rough).get("candidates", [])
        if row.get("rough_candidate_type") == "mixed_required_verified"
        and row.get("mixed_required_verified") is True
        and int(row["house_index"]) in source_houses
    }
    eligible = {
        "channel": source_houses,
        "container": container_houses,
        "mixed": mixed_houses,
    }
    payload = {
        "schema_version": "interactive_nav_house_plan_v1",
        "seed": config.runtime.seed,
        "source_benchmark": str(seed_benchmark),
        "source_benchmark_sha256": _sha256_file(seed_benchmark),
        "rough_catalogs": {
            "container": {
                "path": str(container_rough),
                "sha256": _sha256_file(container_rough),
            },
            "mixed": {
                "path": str(mixed_rough),
                "sha256": _sha256_file(mixed_rough),
            },
        },
        "domains": {
            domain: {
                "eligible_house_count": len(houses),
                "house_order": _stable_house_order(
                    houses, seed=config.runtime.seed, domain=domain
                ),
            }
            for domain, houses in eligible.items()
        },
    }
    path = output_root(config) / "house_plan.json"
    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError(
                f"House plan mismatch; refusing nondeterministic resume from {path}"
            )
    else:
        write_json(path, payload)
    return path


def write_manifest(config: CollectionConfig) -> Path:
    manifest = build_scene_manifest(config.source)
    path = output_root(config) / "scene_manifest.json"
    write_json(path, manifest)
    return path


def _seed_worker_command(config_path: Path, house_indices: list[int], output_path: Path) -> list[str]:
    return [
        PYTHON,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--stage",
        "seed-worker",
        "--worker_houses",
        ",".join(str(index) for index in house_indices),
        "--worker_output",
        str(output_path),
    ]


def _seed_generation_config(config: CollectionConfig) -> dict[str, Any]:
    return {
        "scene_dataset": config.source.scene_dataset,
        "data_split": config.source.data_split,
        "variant": config.source.variant,
        "seeds_per_house": config.source.seeds_per_house,
        "seed_candidate_pool": config.source.seed_candidate_pool,
        "preferred_object_categories": config.source.preferred_object_categories,
        "preferred_object_names": config.source.preferred_object_names,
        "min_start_goal_distance_m": config.source.min_start_goal_distance_m,
        "max_start_goal_distance_m": config.source.max_start_goal_distance_m,
        "prefer_longest_start_goal": config.source.prefer_longest_start_goal,
        "runtime_seed": config.runtime.seed,
    }


def run_seed_worker(config: CollectionConfig, house_indices: list[int], output_path: Path) -> int:
    manifest = read_json(output_root(config) / "scene_manifest.json")
    available = {int(row["house_index"]): row for row in manifest["houses"]}
    template = load_template_episode()
    generation_config = _seed_generation_config(config)
    existing = read_json(output_path) if output_path.exists() else {}
    if existing.get("generation_config") != generation_config:
        existing = {}
    episodes: list[dict[str, Any]] = list(existing.get("episodes", []))
    failures: list[dict[str, Any]] = list(existing.get("failures", []))
    completed = {int(value) for value in existing.get("completed_house_indices", [])}
    for house_index in house_indices:
        if house_index in completed:
            continue
        if house_index not in available:
            failures.append({"house_index": house_index, "reason": "missing_manifest_entry"})
            continue
        try:
            generated, failed = build_house_seed_episodes(
                house_index=house_index,
                scene_dataset=config.source.scene_dataset,
                data_split=config.source.data_split,
                variant=config.source.variant,
                robot="rby1",
                seed=config.runtime.seed,
                seeds_per_house=config.source.seeds_per_house,
                candidate_pool=config.source.seed_candidate_pool,
                template=template,
                preferred_object_categories=config.source.preferred_object_categories,
                preferred_object_names=config.source.preferred_object_names,
                min_start_goal_distance_m=config.source.min_start_goal_distance_m,
                max_start_goal_distance_m=config.source.max_start_goal_distance_m,
                prefer_longest_start_goal=config.source.prefer_longest_start_goal,
            )
            episodes.extend(generated)
            failures.extend(failed)
        except Exception as exc:
            failures.append(
                {"house_index": house_index, "reason": "seed_generation_failed", "error": repr(exc)}
            )
        completed.add(house_index)
        write_json(
            output_path,
            {
                "schema_version": "interactive_nav_train_seed_shard_v1",
                "house_indices": house_indices,
                "generation_config": generation_config,
                "completed_house_indices": sorted(completed),
                "episodes": episodes,
                "failures": failures,
            },
        )
    return 0 if episodes else 1


def build_seed_benchmark(config: CollectionConfig) -> Path:
    if config.source.kind == "nav_benchmark":
        return write_nav_benchmark_source(config.source, output_root(config))
    manifest_path = output_root(config) / "scene_manifest.json"
    # Re-resolve from the current config so a changed house range cannot
    # silently reuse a stale scene selection.
    write_manifest(config)
    manifest = read_json(manifest_path)
    houses = [int(row["house_index"]) for row in manifest["houses"]]
    seed_root = output_root(config) / "seeds"
    seed_root.mkdir(parents=True, exist_ok=True)
    shards = split_even(houses, config.runtime.workers)
    config_path = output_root(config) / "config.resolved.yaml"
    shard_paths = [seed_root / "shards" / f"shard_{i:03d}.json" for i in range(len(shards))]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MUJOCO_GL"] = config.runtime.mujoco_gl
    env.setdefault("MPLCONFIGDIR", str(seed_root / "matplotlib-cache"))
    results: list[dict[str, Any]] = []
    generation_config = _seed_generation_config(config)
    with ThreadPoolExecutor(max_workers=len(shards) or 1) as executor:
        futures = {}
        for index, shard in enumerate(shards):
            path = shard_paths[index]
            if config.runtime.resume and path.exists():
                payload = read_json(path)
                completed = {int(value) for value in payload.get("completed_house_indices", [])}
                if (
                    payload.get("generation_config") == generation_config
                    and completed.issuperset(shard)
                ):
                    results.append({"shard": index, "path": str(path), "returncode": 0, "resumed": True})
                    continue
            futures[executor.submit(
                run_command,
                _seed_worker_command(config_path, shard, path),
                log_path=seed_root / "logs" / f"shard_{index:03d}.log",
                env=env,
            )] = (index, path)
        for future in as_completed(futures):
            index, path = futures[future]
            results.append({"shard": index, "path": str(path), "returncode": future.result(), "resumed": False})
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected_houses = set(houses)
    for path in shard_paths:
        if not path.exists():
            continue
        payload = read_json(path)
        episodes.extend(
            episode
            for episode in payload.get("episodes", [])
            if int(episode["house_index"]) in selected_houses
        )
        failures.extend(
            failure
            for failure in payload.get("failures", [])
            if int(failure["house_index"]) in selected_houses
        )
    episodes.sort(key=lambda episode: (int(episode["house_index"]), int(episode.get("seed", 0))))
    write_json(seed_root / "benchmark.json", episodes)
    write_json(
        seed_root / "summary.json",
        {
            "schema_version": "interactive_nav_train_seed_summary_v1",
            "scene_manifest": str(manifest_path),
            "house_count": len(houses),
            "seed_episode_count": len(episodes),
            "seed_house_count": len({int(episode["house_index"]) for episode in episodes}),
            "failure_count": len(failures),
            "failures": failures,
            "shards": results,
        },
    )
    if not episodes:
        raise RuntimeError("Seed generation produced no episodes")
    return seed_root / "benchmark.json"


def run_container_fine_collection(
    config: CollectionConfig,
    *,
    raw_root: Path,
    seed_benchmark: Path,
    rough_catalog: Path,
    env: dict[str, str],
    target: int,
    per_house: int,
    max_houses_per_wave: int | None = None,
    allow_partial: bool = False,
) -> Path:
    """Run the Container queue independently of Channel and Mixed collectors."""
    container_benchmark = raw_root / "container" / "benchmark.json"
    resumed_container: list[dict[str, Any]] = []
    if config.runtime.resume:
        if container_benchmark.exists():
            resumed_container.extend(load_episodes(container_benchmark))
        for batch_benchmark in sorted(
            (raw_root / "container_batches").glob("batch_*/benchmark.json")
        ):
            resumed_container.extend(load_episodes(batch_benchmark))
        for batch_partial in sorted(
            (raw_root / "container_batches").glob(
                "batch_*/shards/shard_*/benchmark/benchmark.partial.json"
            )
        ):
            resumed_container.extend(load_episodes(batch_partial))
        for partial in sorted(
            (raw_root / "container" / "shards").glob(
                "shard_*/benchmark/benchmark.partial.json"
            )
        ):
            resumed_container.extend(load_episodes(partial))
    resumed_container = _deduplicate(resumed_container)

    def selected_count() -> int:
        required = [
            episode
            for episode in resumed_container
            if episode.get("interactive_nav", {}).get("interaction_requirement")
            == "required"
        ]
        return len(_round_robin_by_house(required, target, per_house))

    if selected_count() >= target:
        write_json(container_benchmark, resumed_container)
        return container_benchmark

    plan = read_json(output_root(config) / "house_plan.json")["domains"]["container"][
        "house_order"
    ]
    batch_root = raw_root / "container_batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    existing_batches = sorted(batch_root.glob("batch_*"))
    scheduler_attempted_slots: set[tuple[int, int]] = set()
    for batch_dir in existing_batches:
        meta = batch_dir / "batch_meta.json"
        if meta.exists():
            payload = read_json(meta)
            if payload.get("scheduler_wave") is True:
                scheduler_attempted_slots.update(
                    (int(house_index), int(slot_offset))
                    for house_index, slot_offset in payload.get("slot_offsets", {}).items()
                )
    required_by_house = Counter(
        int(episode["house_index"])
        for episode in resumed_container
        if episode.get("interactive_nav", {}).get("interaction_requirement")
        == "required"
    )
    # `per_house` is a dynamic cap, not the number requested in each wave. We
    # take exactly one next slot per house: finish pass 0 over all houses, then
    # pass 1, and so on. This gives broad house coverage before increasing the
    # per-house count, while still allowing later houses to replace failures.
    selected_houses = [
        int(value)
        for value in plan
        if required_by_house[int(value)] < per_house
        and (int(value), required_by_house[int(value)])
        not in scheduler_attempted_slots
    ]
    plan_position = {int(house_index): index for index, house_index in enumerate(plan)}
    selected_houses.sort(
        key=lambda house_index: (required_by_house[house_index], plan_position[house_index])
    )
    scheduler_status = raw_root / "container" / "scheduler_status.json"

    def write_scheduler_status(*, capacity_exhausted: bool) -> None:
        write_json(
            scheduler_status,
            {
                "target": target,
                "selected_count": selected_count(),
                "per_house_cap": per_house,
                "remaining_house_slot_count": len(selected_houses),
                "capacity_exhausted": capacity_exhausted,
            },
        )

    if not selected_houses:
        write_json(container_benchmark, resumed_container)
        write_scheduler_status(capacity_exhausted=True)
        if allow_partial:
            return container_benchmark
        raise RuntimeError("Container fine collection did not reach checkpoint target")
    batch_index = len(existing_batches)
    needed = max(target - selected_count(), 1)
    while needed > 0 and selected_houses:
        # This is a global attempt wave. The nested collector only receives
        # this domain's assigned share of the global worker budget.
        batch_limit = max_houses_per_wave or max(needed, 32)
        batch_houses = selected_houses[: min(batch_limit, max(needed, 1))]
        selected_houses = selected_houses[len(batch_houses) :]
        batch_dir = batch_root / f"batch_{batch_index:03d}"
        batch_index += 1
        slot_offsets = {
            int(house_index): int(required_by_house[int(house_index)])
            for house_index in batch_houses
        }
        # Build a deterministic candidate list through the requested cap, then
        # expose only the next uncollected slot. The builder still receives all
        # later candidates as fallbacks for that slot.
        manifest = build_dynamic_collection_plan(
            rough_catalog,
            max_samples=len(batch_houses) * per_house,
            samples_per_house=per_house,
            house_indices=batch_houses,
            seed=config.runtime.seed,
        )
        selected_manifest_houses = []
        for house in manifest.get("houses", []):
            house_index = int(house["house_index"])
            offset = slot_offsets[house_index]
            candidates = list(house.get("candidates", []))[offset:]
            if not candidates:
                continue
            selected_manifest_houses.append(
                {
                    **house,
                    "target_sample_count": 1,
                    "candidates": candidates,
                }
            )
        manifest["houses"] = selected_manifest_houses
        manifest.setdefault("selection", {})["requested_sample_count"] = len(
            selected_manifest_houses
        )
        manifest["selection"]["dynamic_slot_offsets"] = {
            str(house_index): slot_offsets[house_index]
            for house_index in batch_houses
        }
        manifest_path = batch_dir / "collection_plan.json"
        write_json(manifest_path, manifest)
        command = [
            PYTHON,
            "scripts/InteractiveNav/collect_container_fine_parallel.py",
            "--benchmark_dir",
            str(seed_benchmark),
            "--candidate_manifest",
            str(manifest_path),
            "--output_dir",
            str(batch_dir),
            "--seed",
            str(config.runtime.seed),
            "--workers",
            str(config.runtime.workers),
            "--mujoco_gl",
            config.runtime.mujoco_gl,
            "--no-save_images",
            "--no-save_plots",
        ]
        run_command(command, log_path=batch_dir / "run.log", env=env)
        write_json(
            batch_dir / "batch_meta.json",
            {
                "batch_index": batch_index - 1,
                "pass": per_house,
                "house_indices": batch_houses,
                "scheduler_wave": max_houses_per_wave is not None,
                "slot_offsets": {
                    str(house_index): slot_offsets[house_index]
                    for house_index in batch_houses
                },
            },
        )
        if (batch_dir / "benchmark.json").exists():
            resumed_container.extend(load_episodes(batch_dir / "benchmark.json"))
            resumed_container = _deduplicate(resumed_container)
        needed = target - selected_count()
        if max_houses_per_wave is not None:
            break
    write_json(container_benchmark, resumed_container)
    # A slot accepted in this wave may unlock the next per-house slot. The next
    # invocation recomputes that state from the newly written benchmark, so only
    # the no-selectable-slot branch above may declare capacity exhaustion.
    write_scheduler_status(capacity_exhausted=False)
    if selected_count() < target:
        if allow_partial:
            return container_benchmark
        raise RuntimeError("Container fine collection did not reach checkpoint target")
    return container_benchmark


def run_light_collectors(
    config: CollectionConfig,
    seed_benchmark: Path,
    *,
    targets: dict[str, int] | None = None,
    per_house_limits: dict[str, int] | None = None,
) -> dict[str, Path]:
    root = output_root(config) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    seed_episodes = load_episodes(seed_benchmark)
    common_env = os.environ.copy()
    common_env["PYTHONPATH"] = f"{REPO_ROOT}:{common_env.get('PYTHONPATH', '')}"
    common_env["MUJOCO_GL"] = config.runtime.mujoco_gl
    common_env["PYTHONHASHSEED"] = str(config.runtime.seed)
    common_env["INTERACTIVE_NAV_COLLECTION_SEED"] = str(config.runtime.seed)
    common_env.setdefault("MPLCONFIGDIR", str(root / "matplotlib-cache"))
    targets = targets or config.balance.target_counts()
    rough = config.rough.container_catalog or (
        root / "container_rough" / "rough_catalog.json"
    )
    if not rough.exists():
        if not config.rough.generate_if_missing:
            raise FileNotFoundError(
                f"Container fine collection requires a precomputed rough catalog: {rough}"
            )
        container_rough_command = [
            PYTHON,
            "scripts/InteractiveNav/collect_container_rough_catalog_parallel.py",
            "--benchmark_dir", str(seed_benchmark),
            "--output_dir", str(rough.parent),
            "--workers", str(config.runtime.workers),
            "--mujoco_gl", config.runtime.mujoco_gl,
        ]
        run_command(
            container_rough_command,
            log_path=root / "container_rough.log",
            env=common_env,
        )
    if not rough.exists():
        raise RuntimeError(f"container rough catalog was not produced: {rough}")
    mixed_rough = config.rough.mixed_catalog or (
        root / "mixed_rough" / "mixed_rough_catalog.json"
    )
    resumed_mixed_candidates: list[dict[str, Any]] = []
    resumed_mixed_houses: list[dict[str, Any]] = []
    resumed_mixed_failures: list[dict[str, Any]] = []
    if config.runtime.resume and not mixed_rough.exists():
        for shard_dir in sorted((root / "mixed_rough" / "shards").glob("shard_*")):
            candidates_path = shard_dir / "candidates.partial.json"
            houses_path = shard_dir / "houses.partial.json"
            failures_path = shard_dir / "failures.partial.json"
            if candidates_path.exists():
                resumed_mixed_candidates.extend(read_json(candidates_path))
            if houses_path.exists():
                resumed_mixed_houses.extend(read_json(houses_path))
            if failures_path.exists():
                resumed_mixed_failures.extend(read_json(failures_path))
        resumed_mixed_candidates = _deduplicate_rough_candidates(
            resumed_mixed_candidates
        )
        required_mixed_candidates = [
            candidate
            for candidate in resumed_mixed_candidates
            if candidate.get("rough_candidate_type") == "mixed_required_verified"
            and candidate.get("mixed_required_verified") is True
        ]
        resumable_mixed_selection = _round_robin_rough_by_house(
            required_mixed_candidates,
            targets["mixed"],
            config.balance.max_samples_per_house["mixed"],
        )
        if len(resumable_mixed_selection) >= targets["mixed"]:
            write_json(
                mixed_rough,
                {
                    "schema_version": "mixed_rough_catalog_v1",
                    "source_container_rough_catalog": str(rough),
                    "benchmark_dir": str(seed_benchmark),
                    "input_scope": {
                        "selection_scope": "partial_until_balanced_capacity",
                        "strict_pair_semantics_preserved": True,
                    },
                    "summary": {
                        "partial_coverage": True,
                        "candidate_count": len(resumed_mixed_candidates),
                        "required_candidate_count": len(required_mixed_candidates),
                        "resumable_selected_count": len(resumable_mixed_selection),
                    },
                    "houses": resumed_mixed_houses,
                    "candidates": resumed_mixed_candidates,
                    "failures": resumed_mixed_failures,
                    "shards": [],
                },
            )
    if not mixed_rough.exists():
        if not config.rough.generate_if_missing:
            raise FileNotFoundError(
                f"Mixed fine collection requires a precomputed rough catalog: {mixed_rough}"
            )
        mixed_rough_command = [
            PYTHON,
            "scripts/InteractiveNav/collect_mixed_rough_catalog.py",
            "--container_rough_catalog", str(rough),
            "--benchmark_dir", str(seed_benchmark),
            "--output_dir", str(mixed_rough.parent),
            "--workers", str(config.runtime.workers),
            "--mujoco_gl", config.runtime.mujoco_gl,
        ]
        run_command(mixed_rough_command, log_path=root / "mixed_rough.log", env=common_env)
    if not mixed_rough.exists():
        raise RuntimeError(f"mixed rough catalog was not produced: {mixed_rough}")
    per_house_limits = per_house_limits or _checkpoint_house_caps(
        config,
        targets=targets,
        seed_episodes=seed_episodes,
        container_rough=rough,
        mixed_rough=mixed_rough,
    )
    _ensure_collection_fingerprint(
        config,
        root=output_root(config),
        seed_benchmark=seed_benchmark,
        rough_paths=[rough, mixed_rough],
    )
    _write_or_validate_house_plan(
        config,
        seed_benchmark=seed_benchmark,
        seed_episodes=seed_episodes,
        container_rough=rough,
        mixed_rough=mixed_rough,
    )
    if config.runtime.light_scheduler == "manifest_parallel":
        from scripts.InteractiveNav.collection.manifest_scheduler import (
            run_manifest_light_collection,
        )

        return run_manifest_light_collection(
            config,
            root=output_root(config),
            seed_benchmark=seed_benchmark,
            container_rough=rough,
            mixed_rough=mixed_rough,
            targets=targets,
        )
    if config.runtime.light_scheduler == "house_batch_parallel":
        from scripts.InteractiveNav.collection.manifest_scheduler import (
            run_house_batch_light_collection,
        )

        return run_house_batch_light_collection(
            config,
            root=output_root(config),
            seed_benchmark=seed_benchmark,
            container_rough=rough,
            mixed_rough=mixed_rough,
            targets=targets,
        )
    if config.runtime.light_scheduler == "domain_parallel":
        # One global scheduler owns the total simulator budget.  It runs short,
        # deterministic waves and recomputes deficits after every wave, so a
        # completed Channel/Container queue immediately hands its slots to
        # Mixed rather than leaving simulators idle.
        domain_envs = {
            domain: {
                **common_env,
                "MPLCONFIGDIR": str(root / f"matplotlib-cache-{domain}"),
            }
            for domain in ("channel", "container", "mixed")
        }
        raw_paths = {
            "channel": root / "channel" / "benchmark.json",
            "container": root / "container" / "benchmark.json",
            "mixed": root / "mixed" / "benchmark.json",
        }
        rounds: list[dict[str, Any]] = []
        stalled_rounds = {domain: 0 for domain in targets}
        while True:
            before_counts = _domain_selected_counts(
                config,
                root,
                targets=targets,
                per_house_limits=per_house_limits,
            )
            active_domains = [
                domain
                for domain, target in targets.items()
                if before_counts.get(domain, 0) < target
            ]
            if not active_domains:
                break
            worker_allocation = _domain_worker_allocation(
                config.runtime.workers,
                active_domains,
                current_counts=before_counts,
                targets=targets,
            )
            domain_configs = {
                domain: config.model_copy(
                    update={
                        "runtime": config.runtime.model_copy(
                            update={"workers": worker_allocation[domain]}
                        )
                    }
                )
                for domain in active_domains
            }
            wave_items = {
                domain: (
                    config.runtime.domain_wave_items_per_worker
                    * worker_allocation[domain]
                )
                for domain in active_domains
            }
            with ThreadPoolExecutor(max_workers=len(active_domains)) as executor:
                futures = {}
                for domain in active_domains:
                    domain_config = domain_configs[domain]
                    if domain == "container":
                        future = executor.submit(
                            run_container_fine_collection,
                            domain_config,
                            raw_root=root,
                            seed_benchmark=seed_benchmark,
                            rough_catalog=rough,
                            env=domain_envs[domain],
                            target=targets[domain],
                            per_house=per_house_limits[domain],
                            max_houses_per_wave=wave_items[domain],
                            allow_partial=True,
                        )
                    elif domain == "channel":
                        future = executor.submit(
                            run_door_parallel,
                            domain_config,
                            seed_episodes,
                            root,
                            domain_envs[domain],
                            target=targets[domain],
                            per_house=per_house_limits[domain],
                            max_houses_per_wave=wave_items[domain],
                            allow_partial=True,
                        )
                    else:
                        future = executor.submit(
                            run_mixed_fine_parallel,
                            domain_config,
                            mixed_rough=mixed_rough,
                            seed_benchmark=seed_benchmark,
                            raw_root=root,
                            env=domain_envs[domain],
                            target=targets[domain],
                            per_house=per_house_limits[domain],
                            max_candidates_per_wave=wave_items[domain],
                            allow_partial=True,
                        )
                    futures[future] = domain
                for future in as_completed(futures):
                    raw_paths[futures[future]] = future.result()
            after_counts = _domain_selected_counts(
                config,
                root,
                targets=targets,
                per_house_limits=per_house_limits,
            )
            for domain in active_domains:
                if after_counts.get(domain, 0) > before_counts.get(domain, 0):
                    stalled_rounds[domain] = 0
                else:
                    stalled_rounds[domain] += 1
            rounds.append(
                {
                    "round": len(rounds),
                    "before_counts": before_counts,
                    "after_counts": after_counts,
                    "active_domains": active_domains,
                    "worker_allocation": worker_allocation,
                    "wave_items": wave_items,
                }
            )
            write_json(
                root / "domain_scheduler.ledger.json",
                {
                    "schema_version": "interactive_nav_domain_parallel_ledger_v2",
                    "scheduler": "deficit_driven_domain_waves",
                    "workers_total": config.runtime.workers,
                    "maximum_simulation_count": config.runtime.workers,
                    "targets": targets,
                    "raw_paths": {domain: str(path) for domain, path in raw_paths.items()},
                "selected_episode_counts": after_counts,
                "raw_episode_counts": _domain_counts(output_root(config)),
                    "rounds": rounds,
                },
            )
            container_status_path = root / "container" / "scheduler_status.json"
            container_capacity_exhausted = False
            if container_status_path.exists():
                container_capacity_exhausted = bool(
                    read_json(container_status_path).get("capacity_exhausted", False)
                )
            stalled = [
                domain
                for domain in active_domains
                if (
                    domain == "container"
                    and container_capacity_exhausted
                    and after_counts.get(domain, 0) < targets[domain]
                )
                or (
                    domain != "container" and stalled_rounds[domain] >= 12
                )
            ]
            if stalled:
                raise RuntimeError(
                    "Collection capacity stalled for "
                    + ", ".join(
                        f"{domain} ({after_counts.get(domain, 0)}/{targets[domain]})"
                        for domain in stalled
                    )
                )
        write_json(
            root / "domain_scheduler.ledger.json",
            {
                "schema_version": "interactive_nav_domain_parallel_ledger_v2",
                "scheduler": "deficit_driven_domain_waves",
                "workers_total": config.runtime.workers,
                "maximum_simulation_count": config.runtime.workers,
                "targets": targets,
                "raw_paths": {domain: str(path) for domain, path in raw_paths.items()},
                "selected_episode_counts": _domain_selected_counts(
                    config,
                    root,
                    targets=targets,
                    per_house_limits=per_house_limits,
                ),
                "raw_episode_counts": _domain_counts(output_root(config)),
                "rounds": rounds,
                "completed": True,
            },
        )
        return raw_paths
    rough_payload = read_json(rough)
    source_houses = sorted({int(episode["house_index"]) for episode in seed_episodes})
    source_house_set = set(source_houses)
    rough_house_count = len(
        {
            int(house["house_index"])
            for house in rough_payload.get("houses", [])
            if int(house.get("strict_pair_count", 0)) > 0
            and int(house["house_index"]) in source_house_set
        }
    )
    container_benchmark = root / "container" / "benchmark.json"
    resumed_container: list[dict[str, Any]] = []
    if config.runtime.resume:
        if container_benchmark.exists():
            resumed_container.extend(load_episodes(container_benchmark))
        for batch_benchmark in sorted(
            (root / "container_batches").glob("batch_*/benchmark.json")
        ):
            resumed_container.extend(load_episodes(batch_benchmark))
        # A worker may be interrupted after writing only the atomic partial
        # artifacts. Keep those valid episodes for the next checkpoint instead
        # of silently rescanning the same houses from scratch.
        for batch_partial in sorted(
            (root / "container_batches").glob(
                "batch_*/shards/shard_*/benchmark/benchmark.partial.json"
            )
        ):
            resumed_container.extend(load_episodes(batch_partial))
        for partial in sorted(
            (root / "container" / "shards").glob(
                "shard_*/benchmark/benchmark.partial.json"
            )
        ):
            resumed_container.extend(load_episodes(partial))
    resumed_container = _deduplicate(resumed_container)
    required_container = [
        episode
        for episode in resumed_container
        if episode.get("interactive_nav", {}).get("interaction_requirement") == "required"
    ]
    resumable_container_selection = _round_robin_by_house(
        required_container,
        targets["container"],
        config.balance.max_samples_per_house["container"],
    )
    if len(resumable_container_selection) >= targets["container"]:
        write_json(container_benchmark, resumed_container)
        write_json(
            root / "container" / "resume_summary.json",
            {
                "schema_version": "container_fine_resume_summary_v1",
                "raw_episode_count": len(resumed_container),
                "required_episode_count": len(required_container),
                "resumable_selected_count": len(resumable_container_selection),
                "resumed_from_incremental_samples": True,
            },
        )
    else:
        plan = read_json(output_root(config) / "house_plan.json")["domains"]["container"]["house_order"]
        batch_root = root / "container_batches"
        batch_root.mkdir(parents=True, exist_ok=True)
        existing_batches = sorted(batch_root.glob("batch_*"))
        completed_houses: set[int] = set()
        for batch_dir in existing_batches:
            meta = batch_dir / "batch_meta.json"
            if meta.exists():
                completed_houses.update(int(value) for value in read_json(meta).get("house_indices", []))
            for house_catalog in batch_dir.glob(
                "shards/shard_*/benchmark/house_catalog.partial.json"
            ):
                try:
                    completed_houses.update(
                        int(row["house_index"])
                        for row in read_json(house_catalog)
                    )
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue
        selected_houses = [int(value) for value in plan if int(value) not in completed_houses]
        batch_index = len(existing_batches)
        needed = max(targets["container"] - len(resumable_container_selection), 1)
        while needed > 0 and selected_houses:
            batch_houses = selected_houses[: max(needed, 32)]
            selected_houses = selected_houses[len(batch_houses):]
            batch_dir = batch_root / f"batch_{batch_index:03d}"
            batch_index += 1
            command = [
                PYTHON,
                "scripts/InteractiveNav/collect_container_fine_parallel.py",
                "--benchmark_dir", str(seed_benchmark),
                "--rough_catalog", str(rough),
                "--output_dir", str(batch_dir),
                "--max_samples", str(len(batch_houses)),
                "--samples_per_house", str(per_house_limits["container"]),
                "--house_indices", ",".join(str(value) for value in batch_houses),
                "--seed", str(config.runtime.seed),
                "--workers", str(config.runtime.workers),
                "--mujoco_gl", config.runtime.mujoco_gl,
                "--no-save_images", "--no-save_plots",
            ]
            run_command(command, log_path=batch_dir / "run.log", env=common_env)
            write_json(batch_dir / "batch_meta.json", {
                "batch_index": batch_index - 1,
                "pass": per_house_limits["container"],
                "house_indices": batch_houses,
            })
            if (batch_dir / "benchmark.json").exists():
                resumed_container.extend(load_episodes(batch_dir / "benchmark.json"))
                resumed_container = _deduplicate(resumed_container)
            required_container = [
                episode for episode in resumed_container
                if episode.get("interactive_nav", {}).get("interaction_requirement") == "required"
            ]
            needed = targets["container"] - len(_round_robin_by_house(
                required_container, targets["container"], per_house_limits["container"]
            ))
        write_json(container_benchmark, resumed_container)
        if len(_round_robin_by_house(required_container, targets["container"], per_house_limits["container"])) < targets["container"]:
            raise RuntimeError("Container fine collection did not reach checkpoint target")
    if not container_benchmark.exists():
        raise RuntimeError(f"container benchmark was not produced: {container_benchmark}")
    run_door_parallel(
        config, seed_episodes, root, common_env,
        target=targets["channel"], per_house=per_house_limits["channel"]
    )
    run_mixed_fine_parallel(
        config,
        mixed_rough=mixed_rough,
        seed_benchmark=seed_benchmark,
        raw_root=root,
        env=common_env,
        target=targets["mixed"],
        per_house=per_house_limits["mixed"],
    )
    return {
        "channel": root / "channel" / "benchmark.json",
        "container": root / "container" / "benchmark.json",
        "mixed": root / "mixed" / "benchmark.json",
    }


def run_mixed_fine_parallel(
    config: CollectionConfig,
    *,
    mixed_rough: Path,
    seed_benchmark: Path,
    raw_root: Path,
    env: dict[str, str],
    target: int | None = None,
    per_house: int | None = None,
    max_candidates_per_wave: int | None = None,
    allow_partial: bool = False,
) -> Path:
    target = target if target is not None else config.balance.target_counts()["mixed"]
    per_house = per_house if per_house is not None else config.balance.max_samples_per_house["mixed"]
    mixed_root = raw_root / "mixed"

    rough_paths = [mixed_rough]
    for additional_root in sorted(raw_root.glob("mixed_rough_additional_*")):
        catalog_path = additional_root / "mixed_rough_catalog.json"
        if not catalog_path.exists():
            candidates: list[dict[str, Any]] = []
            houses: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            for shard_dir in sorted((additional_root / "shards").glob("shard_*")):
                for filename, destination in [
                    ("candidates.partial.json", candidates),
                    ("houses.partial.json", houses),
                    ("failures.partial.json", failures),
                ]:
                    path = shard_dir / filename
                    if path.exists():
                        destination.extend(read_json(path))
            candidates = _deduplicate_rough_candidates(candidates)
            if candidates:
                write_json(
                    catalog_path,
                    {
                        "schema_version": "mixed_rough_catalog_v1",
                        "source_container_rough_catalog": "incremental_additional_scan",
                        "benchmark_dir": str(seed_benchmark),
                        "input_scope": {
                            "selection_scope": "partial_additional_houses",
                            "strict_pair_semantics_preserved": True,
                        },
                        "summary": {
                            "partial_coverage": True,
                            "candidate_count": len(candidates),
                        },
                        "houses": houses,
                        "candidates": candidates,
                        "failures": failures,
                        "shards": [],
                    },
                )
        if catalog_path.exists():
            rough_paths.append(catalog_path)

    def existing_episodes() -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        for path in [
            mixed_root / "benchmark.json",
            mixed_root / "benchmark.partial.json",
            *sorted(raw_root.glob("mixed_shards*/shard_*/benchmark.json")),
            *sorted(raw_root.glob("mixed_shards*/shard_*/benchmark.partial.json")),
        ]:
            if path.exists():
                episodes.extend(load_episodes(path))
        return _deduplicate(episodes)

    def existing_recorded_keys() -> set[str]:
        keys: set[str] = set()
        paths = [
            *raw_root.glob("mixed_shards*/shard_*/valid*.json"),
            *raw_root.glob("mixed_shards*/shard_*/rejected*.json"),
        ]
        for path in sorted(paths):
            try:
                rows = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            for row in rows if isinstance(rows, list) else []:
                key = row.get("candidate_key")
                if key:
                    keys.add(str(key))
        return keys

    def episode_candidate_key(episode: dict[str, Any]) -> str | None:
        interactive = episode.get("interactive_nav", {})
        explicit = interactive.get("candidate_key")
        if explicit:
            return str(explicit)
        case_id = interactive.get("case_id")
        source_index = interactive.get("parent_benchmark_episode_index")
        if case_id is None:
            return None
        if source_index is None:
            return None
        source_index = int(source_index)
        suffix = f"__src{source_index}"
        base_case_id = str(case_id)
        if base_case_id.endswith(suffix):
            base_case_id = base_case_id[: -len(suffix)]
        return f"{base_case_id}::src{source_index}"

    def balanced_selection(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        required = [
            episode
            for episode in episodes
            if episode.get("interactive_nav", {}).get("interaction_domains")
            == ["channel", "container"]
            and episode.get("interactive_nav", {}).get("interaction_requirement")
            == "required"
        ]
        return _select_balanced_domain_episodes(
            required,
            domain="mixed",
            limit=target,
            per_house=per_house,
            balance_target_categories=config.balance.balance_target_categories
            == "equal",
            balance_path_lengths=config.balance.balance_path_lengths == "equal",
            path_length_bins_m=config.balance.path_length_bins_m,
            relax_house_cap_if_needed=config.balance.relax_house_cap_if_needed,
        )

    resumed = existing_episodes()
    if config.runtime.resume and len(balanced_selection(resumed)) >= target:
        write_json(mixed_root / "benchmark.json", resumed)
        return mixed_root / "benchmark.json"

    source_episodes = load_episodes(seed_benchmark)
    plan_entries = _mixed_candidate_plan_entries(
        rough_paths,
        source_episodes,
        source_variants_per_pair=config.collection.domains.mixed.source_variants_per_pair,
        seed=config.runtime.seed,
    )
    processed_keys = {
        key for key in (episode_candidate_key(episode) for episode in resumed) if key
    }
    processed_keys.update(existing_recorded_keys())
    all_results: list[dict[str, Any]] = []
    batch_index = len(sorted(raw_root.glob("mixed_shards_batch_*")))

    while True:
        episodes = existing_episodes()
        selection = balanced_selection(episodes)
        if len(selection) >= target:
            write_json(mixed_root / "benchmark.json", episodes)
            write_json(
                mixed_root / "parallel_summary.json",
                {
                    "schema_version": "mixed_fine_parallel_summary_v2",
                    "raw_episode_count": len(episodes),
                    "resumable_selected_count": len(selection),
                    "worker_count": config.runtime.workers,
                    "candidate_plan_count": len(plan_entries),
                    "processed_candidate_count": len(processed_keys),
                    "batches": all_results,
                },
            )
            return mixed_root / "benchmark.json"

        remaining = [
            entry for entry in plan_entries
            if entry["candidate_key"] not in processed_keys
        ]
        if not remaining:
            if allow_partial:
                write_json(mixed_root / "benchmark.json", episodes)
                return mixed_root / "benchmark.json"
            raise RuntimeError(
                f"Mixed fine collection capacity is insufficient: target={target} "
                f"selected={len(selection)} raw={len(episodes)}"
            )
        needed = max(target - len(selection), 1)
        # The attempt wave is global, not per shard. This prevents N workers
        # from multiplying the requested oversampling by N.
        # Candidate identity must not depend on worker count. A fixed minimum
        # keeps small jobs parallelizable while preserving the same wave for
        # 1/2/4-worker runs.
        wave_size = min(
            len(remaining),
            max(1, min(needed, max_candidates_per_wave or needed)),
        )
        wave = remaining[:wave_size]
        shard_entries = _split_round_robin(wave, config.runtime.workers)
        shard_root = raw_root / f"mixed_shards_batch_{batch_index:03d}"
        batch_index += 1
        jobs = []
        for index, entries in enumerate(shard_entries):
            output_dir = shard_root / f"shard_{index:03d}"
            plan_path = output_dir / "candidate_plan.json"
            write_json(
                plan_path,
                {
                    "schema_version": "mixed_candidate_plan_v1",
                    "candidates": entries,
                },
            )
            house_indices = sorted({int(row["house_index"]) for row in entries})
            command = [
                PYTHON,
                "scripts/InteractiveNav/build_mixed_interaction_benchmark.py",
                "--mixed_rough_catalog", str(mixed_rough),
                "--benchmark_dir", str(seed_benchmark),
                "--output_dir", str(output_dir),
                "--candidate_plan", str(plan_path),
                "--max_samples", str(len(entries)),
                "--rough_candidate_types", "mixed_required_verified",
                "--max_samples_per_house", str(max(per_house, len(entries))),
                "--source_variants_per_pair",
                str(config.collection.domains.mixed.source_variants_per_pair),
                "--house_indices", ",".join(str(value) for value in house_indices),
                "--variant", config.source.variant,
                "--seed", str(config.runtime.seed),
                "--candidate_timeout_seconds", str(config.runtime.candidate_timeout_seconds),
                "--no-save_images",
                "--no-save_plots",
            ]
            container_benchmark = raw_root / "container" / "benchmark.json"
            if container_benchmark.exists():
                command.extend(["--container_benchmark", str(container_benchmark)])
            for additional_path in rough_paths[1:]:
                command.extend(["--additional_mixed_rough_catalog", str(additional_path)])
            jobs.append((index, command, output_dir, house_indices))
        results = []
        with ThreadPoolExecutor(max_workers=len(jobs) or 1) as executor:
            futures = {
                executor.submit(
                    run_command,
                    command,
                    log_path=output_dir / "run.log",
                    env=env,
                ): (index, output_dir, house_indices)
                for index, command, output_dir, house_indices in jobs
            }
            for future in as_completed(futures):
                index, output_dir, house_indices = futures[future]
                results.append(
                    {
                        "shard": index,
                        "returncode": future.result(),
                        "output_dir": str(output_dir),
                        "house_indices": house_indices,
                    }
                )
        all_results.append(
            {
                "batch": batch_index - 1,
                "wave_size": wave_size,
                "shards": sorted(results, key=lambda value: value["shard"]),
            }
        )
        before_processed = len(processed_keys)
        processed_keys.update(
            key
            for key in (
                episode_candidate_key(episode)
                for episode in existing_episodes()
            )
            if key
        )
        processed_keys.update(existing_recorded_keys())
        if len(processed_keys) == before_processed:
            if allow_partial:
                write_json(mixed_root / "benchmark.json", existing_episodes())
                return mixed_root / "benchmark.json"
            raise RuntimeError(
                "Mixed fine collection stalled: no candidate was recorded in the latest wave"
            )
        if max_candidates_per_wave is not None:
            write_json(mixed_root / "benchmark.json", existing_episodes())
            return mixed_root / "benchmark.json"


def run_door_parallel(
    config: CollectionConfig,
    seed_episodes: list[dict[str, Any]],
    raw_root: Path,
    env: dict[str, str],
    target: int | None = None,
    per_house: int | None = None,
    max_houses_per_wave: int | None = None,
    allow_partial: bool = False,
) -> Path:
    resumed_samples = []
    sample_paths = list(
        (raw_root / "channel_shards").glob("shard_*/output/samples/*/sample.json")
    ) + list((raw_root / "channel_batches").glob("batch_*/shard_*/output/samples/*/sample.json"))
    for sample_path in sorted(sample_paths):
        try:
            resumed_samples.append(read_json(sample_path))
        except (OSError, json.JSONDecodeError):
            continue
    resumed_samples = _deduplicate(resumed_samples)
    allowed_recipes = {
        "single_path_door_closed",
        "distractor_doors_closed",
        "mixed_critical_and_distractor_closed",
    }
    resumable = [
        episode for episode in resumed_samples if _case_type(episode) in allowed_recipes
    ]
    target = target if target is not None else config.balance.target_counts()["channel"]
    per_house = per_house if per_house is not None else config.balance.max_samples_per_house["channel"]
    quota = {
        "single_path_door_closed": round(target * 0.60),
        "distractor_doors_closed": round(target * 0.20),
    }
    quota["mixed_critical_and_distractor_closed"] = target - sum(quota.values())
    grouped_resumable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in resumable:
        grouped_resumable[_case_type(episode)].append(episode)
    resume_selection = _select_channel_with_soft_recipe_quotas(
        grouped_resumable,
        quota,
        limit=target,
        per_house=per_house,
    )
    if config.runtime.resume and len(resume_selection) >= target:
        channel_root = raw_root / "channel"
        write_json(channel_root / "benchmark.json", resumed_samples)
        write_json(channel_root / "failures.json", [])
        write_json(
            channel_root / "summary.json",
            {
                "schema_version": "door_interaction_parallel_collection_summary_v1",
                "input_episode_count": len(seed_episodes),
                "generated_episode_count": len(resumed_samples),
                "failure_count": 0,
                "worker_count": 0,
                "resumed_from_incremental_samples": True,
                "resumable_valid_count": len(resumable),
                "resumable_selected_count": len(resume_selection),
                "shards": [],
            },
        )
        return channel_root / "benchmark.json"

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in seed_episodes:
        grouped[int(episode["house_index"])].append(episode)
    try:
        house_order = read_json(output_root(config) / "house_plan.json")["domains"]["channel"]["house_order"]
    except (FileNotFoundError, KeyError):
        house_order = _stable_house_order(grouped, seed=config.runtime.seed, domain="channel")
    batch_root = raw_root / "channel_batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    completed_houses: set[int] = set()
    for meta in batch_root.glob("batch_*/batch_meta.json"):
        completed_houses.update(int(value) for value in read_json(meta).get("house_indices", []))
    pending_houses = [int(value) for value in house_order if int(value) in grouped and int(value) not in completed_houses]
    results = []
    failures: list[dict[str, Any]] = []
    benchmark = list(resumed_samples)
    batch_index = len(sorted(batch_root.glob("batch_*")))
    while len(resume_selection) < target and pending_houses:
        batch_limit = max_houses_per_wave or max(32, target - len(resume_selection))
        batch_houses = pending_houses[: min(batch_limit, max(target - len(resume_selection), 1))]
        pending_houses = pending_houses[len(batch_houses):]
        batch_dir = batch_root / f"batch_{batch_index:03d}"
        batch_index += 1
        shard_jobs = []
        for shard_index, shard_houses in enumerate(
            _split_round_robin(batch_houses, config.runtime.workers)
        ):
            benchmark_dir = batch_dir / f"shard_{shard_index:03d}" / "input"
            output_dir = batch_dir / f"shard_{shard_index:03d}" / "output"
            shard_episodes = [episode for house in shard_houses for episode in grouped[house]]
            write_json(benchmark_dir / "benchmark.json", shard_episodes)
            command = [
                PYTHON, "scripts/InteractiveNav/build_door_interaction_benchmark.py",
                "--benchmark_dir", str(benchmark_dir), "--output_dir", str(output_dir),
                "--mode", "build", "--input_mode", "original", "--variant", config.source.variant,
                "--max_episodes", str(len(shard_episodes)),
                "--preserve_source_episode_indices",
                "--num_distractor_samples_per_episode", "1", "--num_mixed_samples_per_critical_door", "1",
                "--distractor_k_min", str(config.collection.domains.channel.distractor_k_min),
                "--distractor_k_max", str(config.collection.domains.channel.distractor_k_max),
            ]
            shard_env = {
                **env,
                "INTERACTIVE_NAV_SCENE_MIRROR": str(
                    batch_dir / f"shard_{shard_index:03d}" / "scene_assets"
                ),
            }
            shard_jobs.append((shard_index, command, output_dir, shard_houses, shard_env))
        with ThreadPoolExecutor(max_workers=len(shard_jobs) or 1) as executor:
            futures = {
                executor.submit(
                    run_command, command,
                    log_path=batch_dir / f"shard_{index:03d}" / "run.log",
                    env=shard_env,
                ): (index, output_dir, shard_houses, shard_env)
                for index, command, output_dir, shard_houses, shard_env in shard_jobs
            }
            for future in as_completed(futures):
                index, output_dir, shard_houses, _shard_env = futures[future]
                returncode = future.result()
                results.append({"batch": batch_index - 1, "shard": index, "house_indices": shard_houses, "returncode": returncode, "output_dir": str(output_dir)})
                if (output_dir / "benchmark.json").exists():
                    benchmark.extend(load_episodes(output_dir / "benchmark.json"))
                if (output_dir / "failures.json").exists():
                    failures.extend(read_json(output_dir / "failures.json"))
                for path in sorted((output_dir / "samples").glob("*/sample.json")):
                    try:
                        benchmark.append(read_json(path))
                    except (OSError, json.JSONDecodeError):
                        pass
        write_json(batch_dir / "batch_meta.json", {"batch_index": batch_index - 1, "pass": per_house, "house_indices": batch_houses})
        benchmark = _deduplicate(benchmark)
        grouped_resumable: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for episode in benchmark:
            if _case_type(episode) in allowed_recipes:
                grouped_resumable[_case_type(episode)].append(episode)
        resume_selection = _select_channel_with_soft_recipe_quotas(grouped_resumable, quota, limit=target, per_house=per_house)
        if max_houses_per_wave is not None:
            break
    benchmark = _deduplicate(benchmark)
    if len(resume_selection) < target and not allow_partial:
        raise RuntimeError(
            f"Channel collection capacity is insufficient: target={target} "
            f"selected={len(resume_selection)} raw={len(benchmark)}"
        )
    channel_root = raw_root / "channel"
    write_json(channel_root / "benchmark.json", benchmark)
    write_json(channel_root / "failures.json", failures)
    write_json(
        channel_root / "summary.json",
        {
            "schema_version": "door_interaction_parallel_collection_summary_v1",
            "input_episode_count": len(seed_episodes),
            "generated_episode_count": len(benchmark),
            "failure_count": len(failures),
            "worker_count": config.runtime.workers,
            "batches": results,
        },
    )
    return channel_root / "benchmark.json"


def _case_type(episode: dict[str, Any]) -> str:
    return str(
        episode.get("interactive_nav", {}).get("legacy_case_type")
        or episode.get("interactive_nav", {}).get("case_type")
        or ""
    )


def _deduplicate(episodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for episode in episodes:
        case_id = str(episode.get("interactive_nav", {}).get("case_id", ""))
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        result.append(episode)
    return result


def _deduplicate_rough_candidates(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for candidate in candidates:
        case_id = str(candidate.get("case_id", ""))
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        result.append(candidate)
    return result


def _round_robin_rough_by_house(
    candidates: list[dict[str, Any]], limit: int, per_house: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[int(candidate["house_index"])].append(candidate)
    selected: list[dict[str, Any]] = []
    for rank in range(per_house):
        for house_index in sorted(grouped):
            values = sorted(grouped[house_index], key=lambda value: str(value["case_id"]))
            if rank < len(values):
                selected.append(values[rank])
                if len(selected) >= limit:
                    return selected
    return selected


def _round_robin_by_house(episodes: list[dict[str, Any]], limit: int, per_house: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[int(episode["house_index"])].append(episode)
    for values in grouped.values():
        values.sort(key=lambda episode: str(episode["interactive_nav"]["case_id"]))
    selected: list[dict[str, Any]] = []
    for house_index in sorted(grouped):
        selected.extend(grouped[house_index][:per_house])
    return selected[:limit]


def _select_channel_with_soft_recipe_quotas(
    episodes_by_recipe: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    limit: int,
    per_house: int,
) -> list[dict[str, Any]]:
    """Prefer the requested recipe mix, then fill from any valid channel recipe."""
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    house_counts: Counter[int] = Counter()

    def add_candidates(candidates: Iterable[dict[str, Any]], count: int) -> None:
        if count <= 0:
            return
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for episode in candidates:
            grouped[int(episode["house_index"])].append(episode)
        for values in grouped.values():
            values.sort(key=lambda value: str(value["interactive_nav"]["case_id"]))
        added = 0
        cursors: Counter[int] = Counter()
        while added < count and len(selected) < limit:
            progressed = False
            for house_index in sorted(grouped):
                values = grouped[house_index]
                while cursors[house_index] < len(values):
                    episode = values[cursors[house_index]]
                    cursors[house_index] += 1
                    case_id = str(episode["interactive_nav"]["case_id"])
                    if case_id in selected_ids:
                        continue
                    if house_counts[house_index] >= per_house:
                        break
                    selected.append(episode)
                    selected_ids.add(case_id)
                    house_counts[house_index] += 1
                    added += 1
                    progressed = True
                    break
                if added >= count or len(selected) >= limit:
                    return
            if not progressed:
                return

    for recipe, quota in quotas.items():
        add_candidates(episodes_by_recipe.get(recipe, []), quota)
    if len(selected) < limit:
        add_candidates(
            (
                episode
                for recipe in quotas
                for episode in episodes_by_recipe.get(recipe, [])
            ),
            limit - len(selected),
        )
    return selected[:limit]


def _episode_target_category(episode: dict[str, Any]) -> str:
    target = episode.get("interactive_nav", {}).get("target", {})
    return str(
        target.get("category")
        or episode.get("interactive_nav", {}).get("target_category")
        or "unknown"
    ).lower()


def _episode_container_interaction_type(episode: dict[str, Any]) -> str:
    interactions = episode.get("interactive_nav", {}).get("interactions", [])
    for interaction in interactions:
        interaction_type = str(interaction.get("type", ""))
        if interaction_type.startswith("container_"):
            return interaction_type
    return "unknown"


def _episode_path_length_m(episode: dict[str, Any], domain: str) -> float | None:
    navigation = episode.get("interactive_nav", {}).get(
        "generation_validation", {}
    ).get("navigation_validation", {})
    if domain == "container":
        value = navigation.get("path_length_m")
    elif domain == "mixed":
        approach = navigation.get("approach_path_length_m")
        restored = navigation.get("oracle_restored_path_length_m")
        value = (
            float(approach) + float(restored)
            if approach is not None and restored is not None
            else navigation.get("all_open_path_length_m")
        )
    else:
        value = navigation.get("all_open_path_length_m")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _path_length_bin(length_m: float | None, bins: list[float]) -> str:
    if length_m is None:
        return "unknown"
    for lower, upper in zip(bins, bins[1:]):
        if lower <= length_m < upper:
            return f"[{lower:g},{upper:g})"
    return f"[{bins[-1]:g},inf)"


def _select_balanced_domain_episodes(
    episodes: list[dict[str, Any]],
    *,
    domain: str,
    limit: int,
    per_house: int,
    balance_target_categories: bool,
    balance_path_lengths: bool,
    path_length_bins_m: list[float],
    relax_house_cap_if_needed: bool = False,
) -> list[dict[str, Any]]:
    """Best-effort deterministic balance over category, path bin, and house."""
    candidates = _deduplicate(episodes)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    interaction_type_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    house_counts: Counter[int] = Counter()
    effective_per_house = per_house
    while len(selected) < limit:
        available = [
            episode
            for episode in candidates
            if str(episode.get("interactive_nav", {}).get("case_id", ""))
            not in selected_ids
            and house_counts[int(episode["house_index"])] < effective_per_house
        ]
        if not available:
            if relax_house_cap_if_needed:
                remaining = [
                    episode
                    for episode in candidates
                    if str(episode.get("interactive_nav", {}).get("case_id", ""))
                    not in selected_ids
                ]
                if remaining:
                    effective_per_house += 1
                    continue
            break

        def score(episode: dict[str, Any]) -> tuple[Any, ...]:
            category = _episode_target_category(episode)
            path_bin = _path_length_bin(
                _episode_path_length_m(episode, domain), path_length_bins_m
            )
            return (
                category_counts[category] if balance_target_categories else 0,
                path_counts[path_bin] if balance_path_lengths else 0,
                interaction_type_counts[_episode_container_interaction_type(episode)],
                house_counts[int(episode["house_index"])],
                category,
                path_bin,
                str(episode["interactive_nav"]["case_id"]),
            )

        chosen = min(available, key=score)
        selected.append(chosen)
        selected_ids.add(str(chosen["interactive_nav"]["case_id"]))
        category_counts[_episode_target_category(chosen)] += 1
        interaction_type_counts[_episode_container_interaction_type(chosen)] += 1
        path_counts[
            _path_length_bin(_episode_path_length_m(chosen, domain), path_length_bins_m)
        ] += 1
        house_counts[int(chosen["house_index"])] += 1
    return selected


def _balance_distribution(
    episodes: list[dict[str, Any]], domain: str, bins: list[float]
) -> dict[str, Any]:
    categories = Counter(_episode_target_category(episode) for episode in episodes)
    paths = Counter(
        _path_length_bin(_episode_path_length_m(episode, domain), bins)
        for episode in episodes
    )
    lengths = [
        value
        for episode in episodes
        if (value := _episode_path_length_m(episode, domain)) is not None
    ]
    return {
        "target_category_counts": dict(sorted(categories.items())),
        "container_interaction_type_counts": dict(
            sorted(
                Counter(
                    _episode_container_interaction_type(episode) for episode in episodes
                ).items()
            )
        ),
        "path_length_bin_counts": dict(sorted(paths.items())),
        "path_length_m": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
    }


def balance_benchmark(
    config: CollectionConfig,
    raw_paths: dict[str, Path],
    *,
    targets: dict[str, int] | None = None,
    per_house_limits: dict[str, int] | None = None,
) -> Path:
    targets = targets or config.balance.target_counts()
    per_house_limits = per_house_limits or dict(config.balance.max_samples_per_house)
    # Collection may use a finite per-house slot cap to obtain broad coverage.
    # That scheduling device must not silently become a final-data exclusion
    # rule when the protocol requests unrestricted balance selection.
    selection_house_limits = (
        per_house_limits
        if config.balance.enforce_max_samples_per_house
        else targets
    )
    channel = _deduplicate(load_episodes(raw_paths["channel"]))
    channel_allowed = {
        "single_path_door_closed",
        "distractor_doors_closed",
        "mixed_critical_and_distractor_closed",
    }
    channel = [episode for episode in channel if _case_type(episode) in channel_allowed]
    container = _deduplicate(load_episodes(raw_paths["container"]))
    container = [
        episode
        for episode in container
        if episode.get("interactive_nav", {}).get("interaction_requirement") == "required"
    ]
    mixed = _deduplicate(load_episodes(raw_paths["mixed"]))
    mixed = [
        episode
        for episode in mixed
        if episode.get("interactive_nav", {}).get("interaction_domains")
        == ["channel", "container"]
        and episode.get("interactive_nav", {}).get("interaction_requirement") == "required"
    ]
    channel_by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in channel:
        channel_by_recipe[_case_type(episode)].append(episode)
    channel_quota = {
        "single_path_door_closed": round(targets["channel"] * 0.60),
        "distractor_doors_closed": round(targets["channel"] * 0.20),
    }
    channel_quota["mixed_critical_and_distractor_closed"] = (
        targets["channel"] - sum(channel_quota.values())
    )
    selected_channel = _select_channel_with_soft_recipe_quotas(
        channel_by_recipe,
        channel_quota,
        limit=targets["channel"],
        per_house=selection_house_limits["channel"],
    )
    category_balance = config.balance.balance_target_categories == "equal"
    path_balance = config.balance.balance_path_lengths == "equal"
    selected_container = _select_balanced_domain_episodes(
        container,
        domain="container",
        limit=targets["container"],
        per_house=selection_house_limits["container"],
        balance_target_categories=category_balance,
        balance_path_lengths=path_balance,
        path_length_bins_m=config.balance.path_length_bins_m,
        relax_house_cap_if_needed=(
            config.balance.relax_house_cap_if_needed
            or not config.balance.enforce_max_samples_per_house
        ),
    )
    selected_mixed = _select_balanced_domain_episodes(
        mixed,
        domain="mixed",
        limit=targets["mixed"],
        per_house=selection_house_limits["mixed"],
        balance_target_categories=category_balance,
        balance_path_lengths=path_balance,
        path_length_bins_m=config.balance.path_length_bins_m,
        relax_house_cap_if_needed=(
            config.balance.relax_house_cap_if_needed
            or not config.balance.enforce_max_samples_per_house
        ),
    )
    selected = {
        "channel": selected_channel,
        "container": selected_container,
        "mixed": selected_mixed,
    }
    deficits = {
        domain: targets[domain] - len(values) for domain, values in selected.items()
    }
    if any(value > 0 for value in deficits.values()):
        raise RuntimeError(
            f"Could not satisfy balanced quotas. targets={targets} selected="
            f"{ {key: len(value) for key, value in selected.items()} } deficits={deficits}"
        )
    balanced = selected_channel[: targets["channel"]] + selected_container[: targets["container"]] + selected_mixed[: targets["mixed"]]
    balanced_root = output_root(config) / "balanced"
    write_json(balanced_root / "channel.json", selected_channel[: targets["channel"]])
    write_json(balanced_root / "container.json", selected_container[: targets["container"]])
    write_json(balanced_root / "mixed.json", selected_mixed[: targets["mixed"]])
    write_json(balanced_root / "benchmark.json", balanced)
    write_json(
        balanced_root / "summary.json",
        {
            "schema_version": "interactive_nav_v3_balanced_collection_summary_v1",
            "target_counts": targets,
            "actual_counts": {domain: len(values) for domain, values in selected.items()},
            "recipe_counts": {
                "channel": dict(Counter(_case_type(episode) for episode in selected_channel)),
                "container": dict(Counter(
                    str(episode.get("interactive_nav", {}).get("interaction_domains"))
                    for episode in selected_container
                )),
                "mixed": dict(Counter(_case_type(episode) for episode in selected_mixed)),
            },
            "unique_houses": {
                domain: len({int(episode["house_index"]) for episode in values})
                for domain, values in selected.items()
            },
            "distribution": {
                "container": _balance_distribution(
                    selected_container, "container", config.balance.path_length_bins_m
                ),
                "mixed": _balance_distribution(
                    selected_mixed, "mixed", config.balance.path_length_bins_m
                ),
            },
            "balance_policy": {
                "enforce_three_way_balance": config.balance.enforce_three_way_balance,
                "balance_target_categories": config.balance.balance_target_categories,
                "balance_path_lengths": config.balance.balance_path_lengths,
                "relax_house_cap_if_needed": config.balance.relax_house_cap_if_needed,
                "enforce_max_samples_per_house": (
                    config.balance.enforce_max_samples_per_house
                ),
                "path_length_bins_m": config.balance.path_length_bins_m,
                "raw_oversample_factors": config.balance.raw_oversample_factors,
                "effective_max_samples_per_house": per_house_limits,
            },
            "excluded": {
                "open_gt_control": True,
                "synthetic_wrong_action_rollout": True,
            },
        },
    )
    return balanced_root / "benchmark.json"


def audit_benchmark(config: CollectionConfig, benchmark_path: Path) -> Path:
    episodes = load_episodes(benchmark_path)
    errors: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    requirement_counts: Counter[str] = Counter()
    recipe_counts: Counter[str] = Counter()
    interaction_type_counts: Counter[str] = Counter()
    control_mode_counts: Counter[str] = Counter()
    target_category_counts: Counter[str] = Counter()
    house_counts: Counter[int] = Counter()
    top_level_key_counts: Counter[str] = Counter()
    interactive_key_counts: Counter[str] = Counter()
    placeholder_paths: list[str] = []

    def inspect_placeholders(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                inspect_placeholders(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for item_index, item in enumerate(value):
                inspect_placeholders(item, f"{path}[{item_index}]")
        elif isinstance(value, str) and value.strip().lower() in {
            "unknown",
            "placeholder",
            "todo",
            "not-sampled",
        }:
            placeholder_paths.append(path)
    for index, episode in enumerate(episodes):
        domains = episode.get("interactive_nav", {}).get("interaction_domains", [])
        expected = "mixed" if domains == ["channel", "container"] else domains[0] if len(domains) == 1 else "unknown"
        try:
            validated = v3.validate_interactive_nav_v3_episode(episode, expected_domains=domains)
            valid.append(validated)
            domain_counts[expected] += 1
            nav = validated["interactive_nav"]
            requirement_counts[str(nav["interaction_requirement"])] += 1
            recipe_counts[str(nav.get("legacy_case_type") or "canonical")] += 1
            target_category_counts[str(nav.get("target", {}).get("category", "missing"))] += 1
            house_counts[int(validated["house_index"])] += 1
            top_level_key_counts.update(validated.keys())
            interactive_key_counts.update(nav.keys())
            for interaction in nav.get("interactions", []):
                interaction_type_counts[str(interaction.get("type", "missing"))] += 1
            for plan in nav.get("oracle_plans", []):
                for step in plan.get("steps", []):
                    if step.get("type") == "open_joint":
                        control_mode_counts[str(step.get("control_mode", "missing"))] += 1
            inspect_placeholders(validated, f"episodes[{index}]")
            if validated.get("interactive_nav", {}).get("legacy_case_type") == "all_open_control":
                raise ValueError("open_gt_control/all_open_control is excluded")
        except Exception as exc:
            errors.append({"index": index, "case_id": episode.get("interactive_nav", {}).get("case_id"), "error": repr(exc)})
    balanced_root = output_root(config) / "balanced"
    audit = balanced_root / "audit.json"
    key_coverage = {
        key: count / len(valid) if valid else 0.0
        for key, count in sorted(top_level_key_counts.items())
    }
    interactive_key_coverage = {
        key: count / len(valid) if valid else 0.0
        for key, count in sorted(interactive_key_counts.items())
    }
    write_json(
        audit,
        {
            "schema_version": "interactive_nav_v3_collection_audit_v1",
            "input": str(benchmark_path),
            "episode_count": len(episodes),
            "valid_count": len(valid),
            "error_count": len(errors),
            "domain_counts": dict(domain_counts),
            "interaction_requirement_counts": dict(requirement_counts),
            "recipe_counts": dict(recipe_counts),
            "interaction_type_counts": dict(interaction_type_counts),
            "open_joint_control_mode_counts": dict(control_mode_counts),
            "target_category_counts": dict(target_category_counts),
            "per_house_counts": {str(key): value for key, value in sorted(house_counts.items())},
            "unique_case_ids": len({episode.get("interactive_nav", {}).get("case_id") for episode in valid}),
            "unique_houses": len({int(episode["house_index"]) for episode in valid}),
            "top_level_key_coverage": key_coverage,
            "interactive_nav_key_coverage": interactive_key_coverage,
            "placeholder_value_count": len(placeholder_paths),
            "placeholder_paths": placeholder_paths[:100],
            "errors": errors,
        },
    )
    report_lines = [
        "# InteractiveNav v3 Collection Structure Report",
        "",
        f"- Episodes: {len(episodes)}",
        f"- V3 valid: {len(valid)}",
        f"- Validation errors: {len(errors)}",
        f"- Unique houses: {len(house_counts)}",
        f"- Placeholder values: {len(placeholder_paths)}",
        "",
        "## Domain distribution",
        "",
        *[f"- {key}: {value}" for key, value in sorted(domain_counts.items())],
        "",
        "## Interaction requirements",
        "",
        *[f"- {key}: {value}" for key, value in sorted(requirement_counts.items())],
        "",
        "## Recipes",
        "",
        *[f"- {key}: {value}" for key, value in sorted(recipe_counts.items())],
        "",
        "## Interaction types",
        "",
        *[f"- {key}: {value}" for key, value in sorted(interaction_type_counts.items())],
        "",
        "## Oracle open control modes",
        "",
        *[f"- {key}: {value}" for key, value in sorted(control_mode_counts.items())],
    ]
    (balanced_root / "structure_report.md").write_text("\n".join(report_lines) + "\n")
    if errors or placeholder_paths:
        raise RuntimeError(f"V3 audit found {len(errors)} invalid episodes; see {audit}")
    return audit


def run_full_collectors(config: CollectionConfig) -> Path:
    if (
        "mixed" in config.full.domains
        and config.policy.channel.executor != config.policy.container.executor
    ):
        raise ValueError(
            "The mixed full runner currently requires the same channel/container executor"
        )
    benchmark = output_root(config) / "balanced" / "benchmark.json"
    episodes = load_episodes(benchmark)
    expected_domains = {
        "channel": ["channel"],
        "container": ["container"],
        "mixed": ["channel", "container"],
    }
    candidate_pools: dict[str, list[dict[str, Any]]] = {}
    for domain in config.full.domains:
        interaction_prefix = {
            "channel": "channel_",
            "container": "container_",
            "mixed": None,
        }[domain]
        candidates = [
            episode
            for episode in episodes
            if episode.get("interactive_nav", {}).get("interaction_domains")
            == expected_domains[domain]
            and (
                interaction_prefix is None
                or any(
                    str(interaction.get("type", "")).startswith(interaction_prefix)
                    for interaction in episode.get("interactive_nav", {}).get(
                        "interactions", []
                    )
                )
            )
        ]
        if config.full.selection_strategy == "shortest_validated_path":
            def validated_path_cost(episode: dict[str, Any]) -> float:
                validation = episode.get("interactive_nav", {}).get(
                    "generation_validation", {}
                ).get("navigation_validation", {})
                keys = {
                    "channel": ["all_open_path_length_m"],
                    "container": ["path_length_m"],
                    "mixed": ["approach_path_length_m", "oracle_restored_path_length_m"],
                }[domain]
                values = [validation.get(key) for key in keys]
                if any(value is None for value in values):
                    return float("inf")
                return float(sum(float(value) for value in values))

            candidates.sort(
                key=lambda episode: (
                    validated_path_cost(episode),
                    int(episode["house_index"]),
                    str(episode["interactive_nav"]["case_id"]),
                )
            )
        if len(candidates) < config.full.max_episodes:
            raise RuntimeError(
                f"Requested {config.full.max_episodes} full {domain} episodes, "
                f"found {len(candidates)}"
            )
        candidate_pools[domain] = candidates[
            : config.full.max_candidate_attempts_per_domain
        ]
    full_root = output_root(config) / "full"
    runs = []
    common_env = os.environ.copy()
    common_env["PYTHONPATH"] = f"{REPO_ROOT}:{common_env.get('PYTHONPATH', '')}"
    common_env["MUJOCO_GL"] = config.runtime.mujoco_gl
    common_env.setdefault("MPLCONFIGDIR", str(full_root / "matplotlib-cache"))
    valid_counts = {domain: 0 for domain in config.full.domains}
    for domain in config.full.domains:
        for candidate_rank, episode in enumerate(candidate_pools[domain]):
            if valid_counts[domain] >= config.full.max_episodes:
                break
            case_id = str(episode["interactive_nav"]["case_id"])
            executor = (
                config.policy.channel.executor
                if domain in {"channel", "mixed"}
                else config.policy.container.executor
            )
            runner = (
                "scripts/InteractiveNav/record_mixed_rby1_rollout.py"
                if domain == "mixed"
                else "scripts/InteractiveNav/record_interactive_nav_rby1_rollout.py"
            )
            command = [
                PYTHON,
                runner,
                str(benchmark),
                "--case_id",
                case_id,
                "--output_dir",
                str(full_root / "runs"),
                "--variant",
                config.source.variant,
                "--data_split",
                config.source.data_split,
                "--seed",
                str(config.runtime.seed),
                "--interaction_executor",
                executor,
                "--max_steps",
                str(config.full.max_steps),
                "--max_base_adjustment_steps",
                str(config.full.max_base_adjustment_steps),
                "--collection_hz",
                str(config.full.collection_hz),
                "--navigation_speed_mps",
                str(config.full.navigation_speed_mps),
                "--required_open_fraction",
                str(config.full.required_open_fraction),
                "--img_width",
                str(config.full.image_width),
                "--img_height",
                str(config.full.image_height),
                "--force_fallback_max_steps",
                str(max(config.policy.channel.max_steps, config.policy.container.max_steps)),
                "--force_fallback_target_fraction",
                str(
                    min(
                        config.policy.channel.target_fraction,
                        config.policy.container.target_fraction,
                    )
                    if domain == "mixed"
                    else (
                        config.policy.channel.target_fraction
                        if domain == "channel"
                        else config.policy.container.target_fraction
                    )
                ),
                "--door_force_target_fraction",
                str(config.policy.channel.target_fraction),
                "--container_force_target_fraction",
                str(config.policy.container.target_fraction),
                "--door_force_max_steps",
                str(config.policy.channel.max_steps),
                "--container_force_max_steps",
                str(config.policy.container.max_steps),
                "--door_force_duration_seconds",
                str(config.policy.channel.duration_seconds),
                "--container_force_duration_seconds",
                str(config.policy.container.duration_seconds),
            ]
            if config.full.lock_base_during_force:
                command.append("--lock_base_during_force")
            if domain != "mixed":
                command.extend(["--domain", domain])
            log_path = full_root / "logs" / f"{domain}__{case_id}.log"
            returncode = run_command(command, log_path=log_path, env=common_env)
            run_dir_candidates = sorted(
                (full_root / "runs").glob(f"*{case_id[:48]}*")
            )
            run_dir = run_dir_candidates[-1] if run_dir_candidates else None
            trajectory = run_dir / "trajectory.h5" if run_dir is not None else None
            audit = None
            if trajectory is not None and trajectory.exists():
                audit = validate_full_rollout(trajectory)
            required_segments = {
                "channel": {
                    "nav_to_door",
                    "force_open_door",
                    "nav_to_target",
                    "terminal_observation",
                },
                "container": {
                    "nav_to_container",
                    "force_open_container",
                    "terminal_observation",
                },
                "mixed": {
                    "nav_to_door",
                    "force_open_door",
                    "nav_to_container",
                    "force_open_container",
                    "terminal_observation",
                },
            }[domain]
            observed_segments = set((audit or {}).get("segment_counts", {}))
            force_step_count = int(
                (audit or {}).get("action_type_counts", {}).get("force_joint", 0)
            )
            training_eligible = bool(
                returncode == 0
                and audit is not None
                and audit.get("success") is True
                and required_segments.issubset(observed_segments)
                and int(audit.get("terminal_step_count", 0)) >= 1
                and (executor != "force" or force_step_count >= 1)
            )
            if training_eligible:
                valid_counts[domain] += 1
            runs.append(
                {
                    "domain": domain,
                    "candidate_rank": candidate_rank,
                    "case_id": case_id,
                    "returncode": returncode,
                    "run_dir": str(run_dir) if run_dir is not None else None,
                    "trajectory": str(trajectory) if trajectory is not None else None,
                    "audit": audit,
                    "training_eligible": training_eligible,
                    "log": str(log_path),
                }
            )
    summary = full_root / "summary.json"
    write_json(
        summary,
        {
            "schema_version": "interactive_nav_full_collection_summary_v1",
            "benchmark": str(benchmark),
            "requested_episode_count": len(config.full.domains)
            * config.full.max_episodes,
            "requested_episodes_per_domain": config.full.max_episodes,
            "max_candidate_attempts_per_domain": (
                config.full.max_candidate_attempts_per_domain
            ),
            "attempted_trajectory_count": len(runs),
            "valid_trajectory_count": sum(row["training_eligible"] for row in runs),
            "valid_trajectory_count_by_domain": valid_counts,
            "runs": runs,
        },
    )
    if any(
        valid_counts[domain] < config.full.max_episodes
        for domain in config.full.domains
    ):
        raise RuntimeError(f"Full rollout collection failed; see {summary}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified InteractiveNav v3 collector")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "manifest",
            "seeds",
            "seed-worker",
            "light",
            "mixed",
            "balance",
            "channel-rebuild",
            "audit",
            "full",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--worker_houses")
    parser.add_argument("--worker_output", type=Path)
    parser.add_argument(
        "--checkpoint_per_domain",
        type=int,
        help="This run's cumulative target per domain; excluded from the collection fingerprint.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_collection_config(args.config)
    save_resolved_config(config)
    targets = _checkpoint_target_counts(config, args.checkpoint_per_domain)
    if args.stage in {"light", "mixed", "all"} and config.collection.mode == "light":
        progress_reporter = ProgressReporter(
            output_root(config),
            global_target=config.balance.total_samples,
            domain_targets=targets,
        )
        progress_reporter.start()
        atexit.register(progress_reporter.stop)
    if args.stage == "full":
        if config.collection.mode != "full":
            raise ValueError("The full stage requires collection.mode=full")
        print(run_full_collectors(config))
        return 0
    if config.collection.mode == "full" and args.stage in {"light", "all"}:
        raise ValueError("Use --stage full when collection.mode=full")
    if args.stage == "manifest":
        if config.source.kind == "nav_benchmark":
            benchmark = write_nav_benchmark_source(config.source, output_root(config))
            print(benchmark.with_name("benchmark_manifest.json"))
        else:
            print(write_manifest(config))
        return 0
    if args.stage == "channel-rebuild":
        from scripts.InteractiveNav.collection.manifest_scheduler import (
            rebuild_channel_from_house_cache,
        )

        print(
            rebuild_channel_from_house_cache(
                root=output_root(config),
                target=targets["channel"],
                seed=config.runtime.seed,
            )
        )
        return 0
    if args.stage == "seed-worker":
        if not args.worker_houses or args.worker_output is None:
            raise ValueError("seed-worker requires --worker_houses and --worker_output")
        houses = [int(value) for value in args.worker_houses.split(",") if value]
        return run_seed_worker(config, houses, args.worker_output)
    if args.stage == "mixed":
        seed_benchmark = build_seed_benchmark(config)
        raw_root = output_root(config) / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)
        container_rough = config.rough.container_catalog or (
            raw_root / "container_rough" / "rough_catalog.json"
        )
        mixed_rough = config.rough.mixed_catalog or (
            raw_root / "mixed_rough" / "mixed_rough_catalog.json"
        )
        missing = [
            path for path in (container_rough, mixed_rough) if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "The mixed-only stage requires precomputed rough catalogs: "
                + ", ".join(str(path) for path in missing)
            )
        _ensure_collection_fingerprint(
            config,
            root=output_root(config),
            seed_benchmark=seed_benchmark,
            rough_paths=[container_rough, mixed_rough],
        )
        seed_episodes = load_episodes(seed_benchmark)
        _write_or_validate_house_plan(
            config,
            seed_benchmark=seed_benchmark,
            seed_episodes=seed_episodes,
            container_rough=container_rough,
            mixed_rough=mixed_rough,
        )
        per_house_limits = _checkpoint_house_caps(
            config,
            targets=targets,
            seed_episodes=seed_episodes,
            container_rough=container_rough,
            mixed_rough=mixed_rough,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
        env["MUJOCO_GL"] = config.runtime.mujoco_gl
        env["PYTHONHASHSEED"] = str(config.runtime.seed)
        env["INTERACTIVE_NAV_COLLECTION_SEED"] = str(config.runtime.seed)
        env.setdefault("MPLCONFIGDIR", str(raw_root / "matplotlib-cache"))
        print(
            run_mixed_fine_parallel(
                config,
                mixed_rough=mixed_rough,
                seed_benchmark=seed_benchmark,
                raw_root=raw_root,
                env=env,
                target=targets["mixed"],
                per_house=per_house_limits["mixed"],
            )
        )
        return 0
    if args.stage in {"seeds", "all"}:
        seed_benchmark = build_seed_benchmark(config)
    else:
        seed_benchmark = source_benchmark_path(config)
    if args.stage == "seeds":
        print(seed_benchmark)
        return 0
    if args.stage in {"light", "all"}:
        raw_paths = run_light_collectors(config, seed_benchmark, targets=targets)
    else:
        root = output_root(config) / "raw"
        raw_paths = {
            "channel": root / "channel" / "benchmark.json",
            "container": root / "container" / "benchmark.json",
            "mixed": root / "mixed" / "benchmark.json",
        }
    if args.stage in {"balance", "all"}:
        per_house_limits = _checkpoint_house_caps(
            config,
            targets=targets,
            seed_episodes=load_episodes(seed_benchmark),
            container_rough=config.rough.container_catalog or (output_root(config) / "raw" / "container_rough" / "rough_catalog.json"),
            mixed_rough=config.rough.mixed_catalog or (output_root(config) / "raw" / "mixed_rough" / "mixed_rough_catalog.json"),
        )
        benchmark = balance_benchmark(
            config, raw_paths, targets=targets, per_house_limits=per_house_limits
        )
    else:
        benchmark = output_root(config) / "balanced" / "benchmark.json"
    if args.stage in {"audit", "all"}:
        audit = audit_benchmark(config, benchmark)
        print(json.dumps({"benchmark": str(benchmark), "audit": str(audit)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
