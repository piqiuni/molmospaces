#!/usr/bin/env python3
"""Unified InteractiveNav v3 collection entry point.

The command deliberately keeps scene discovery/seeding separate from the existing
domain-specific MuJoCo collectors.  This makes a train-scene run reproducible while
preserving the validated door/container/mixed generation code.
"""

from __future__ import annotations

import argparse
import copy
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


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def run_seed_worker(config: CollectionConfig, house_indices: list[int], output_path: Path) -> int:
    manifest = read_json(output_root(config) / "scene_manifest.json")
    available = {int(row["house_index"]): row for row in manifest["houses"]}
    template = load_template_episode()
    existing = read_json(output_path) if output_path.exists() else {}
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
    with ThreadPoolExecutor(max_workers=len(shards) or 1) as executor:
        futures = {}
        for index, shard in enumerate(shards):
            path = shard_paths[index]
            if config.runtime.resume and path.exists():
                payload = read_json(path)
                completed = {int(value) for value in payload.get("completed_house_indices", [])}
                if completed.issuperset(shard):
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


def run_light_collectors(config: CollectionConfig, seed_benchmark: Path) -> dict[str, Path]:
    root = output_root(config) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    seed_episodes = load_episodes(seed_benchmark)
    common_env = os.environ.copy()
    common_env["PYTHONPATH"] = f"{REPO_ROOT}:{common_env.get('PYTHONPATH', '')}"
    common_env["MUJOCO_GL"] = config.runtime.mujoco_gl
    common_env.setdefault("MPLCONFIGDIR", str(root / "matplotlib-cache"))
    targets = config.balance.target_counts()
    run_door_parallel(config, seed_episodes, root, common_env)
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
    container_fine_command = [
        PYTHON,
        "scripts/InteractiveNav/collect_container_fine_parallel.py",
        "--benchmark_dir", str(seed_benchmark),
        "--rough_catalog", str(rough),
        "--output_dir", str(root / "container"),
        "--max_samples", str(max(targets["container"] * 3, 100)),
        "--samples_per_house", "2",
        "--workers", str(config.runtime.workers),
        "--mujoco_gl", config.runtime.mujoco_gl,
        "--no-save_images",
        "--no-save_plots",
    ]
    container_benchmark = root / "container" / "benchmark.json"
    resumed_container: list[dict[str, Any]] = []
    if config.runtime.resume:
        if container_benchmark.exists():
            resumed_container.extend(load_episodes(container_benchmark))
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
        run_command(container_fine_command, log_path=root / "container.log", env=common_env)
    if not container_benchmark.exists():
        raise RuntimeError(f"container benchmark was not produced: {container_benchmark}")
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
    run_mixed_fine_parallel(
        config,
        mixed_rough=mixed_rough,
        seed_benchmark=seed_benchmark,
        raw_root=root,
        env=common_env,
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
) -> Path:
    target = config.balance.target_counts()["mixed"]
    per_house = config.balance.max_samples_per_house["mixed"]
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

    existing_batch_roots = sorted(raw_root.glob("mixed_shards*"))
    shard_root = raw_root / f"mixed_shards_batch_{len(existing_batch_roots):03d}"

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

    resumed = existing_episodes()
    resumed_required = [
        episode
        for episode in resumed
        if episode.get("interactive_nav", {}).get("interaction_domains")
        == ["channel", "container"]
        and episode.get("interactive_nav", {}).get("interaction_requirement")
        == "required"
    ]
    resumed_selection = _round_robin_by_house(
        resumed_required, target, per_house
    )
    if config.runtime.resume and len(resumed_selection) >= target:
        write_json(mixed_root / "benchmark.json", resumed)
        return mixed_root / "benchmark.json"

    candidate_houses_by_path: dict[Path, set[int]] = {}
    for rough_path in rough_paths:
        rough_payload = read_json(rough_path)
        candidate_houses_by_path[rough_path] = {
            int(candidate["house_index"])
            for candidate in rough_payload.get("candidates", [])
            if candidate.get("rough_candidate_type") == "mixed_required_verified"
            and candidate.get("mixed_required_verified") is True
        }
    preferred_paths = rough_paths[1:] if len(rough_paths) > 1 and resumed else rough_paths
    candidate_houses = sorted(
        set().union(*(candidate_houses_by_path[path] for path in preferred_paths))
    )
    house_shards = split_even(candidate_houses, config.runtime.workers)
    jobs = []
    for index, house_indices in enumerate(house_shards):
        output_dir = shard_root / f"shard_{index:03d}"
        command = [
            PYTHON,
            "scripts/InteractiveNav/build_mixed_interaction_benchmark.py",
            "--mixed_rough_catalog", str(mixed_rough),
            "--benchmark_dir", str(seed_benchmark),
            "--output_dir", str(output_dir),
            "--max_samples", str(max(len(house_indices) * per_house, 1)),
            "--rough_candidate_types", "mixed_required_verified",
            "--max_samples_per_house", str(per_house),
            "--source_variants_per_pair", "1",
            "--house_indices", ",".join(str(value) for value in house_indices),
            "--variant", config.source.variant,
            "--no-save_images",
            "--no-save_plots",
        ]
        for additional_path in rough_paths[1:]:
            command.extend(["--additional_mixed_rough_catalog", str(additional_path)])
        jobs.append((index, command, output_dir, house_indices))
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs) or 1) as executor:
        futures = {
            executor.submit(
                run_command,
                command,
                log_path=shard_root / f"shard_{index:03d}" / "run.log",
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
    episodes = existing_episodes()
    required = [
        episode
        for episode in episodes
        if episode.get("interactive_nav", {}).get("interaction_domains")
        == ["channel", "container"]
        and episode.get("interactive_nav", {}).get("interaction_requirement")
        == "required"
    ]
    selection = _round_robin_by_house(required, target, per_house)
    write_json(mixed_root / "benchmark.json", episodes)
    write_json(
        mixed_root / "parallel_summary.json",
        {
            "schema_version": "mixed_fine_parallel_summary_v1",
            "raw_episode_count": len(episodes),
            "required_episode_count": len(required),
            "resumable_selected_count": len(selection),
            "worker_count": len(jobs),
            "shards": sorted(results, key=lambda value: value["shard"]),
        },
    )
    if len(selection) < target:
        raise RuntimeError(
            f"Mixed fine collection capacity is insufficient: target={target} "
            f"selected={len(selection)} raw={len(episodes)}"
        )
    return mixed_root / "benchmark.json"


def run_door_parallel(
    config: CollectionConfig,
    seed_episodes: list[dict[str, Any]],
    raw_root: Path,
    env: dict[str, str],
) -> Path:
    resumed_samples = []
    for sample_path in sorted(
        (raw_root / "channel_shards").glob("shard_*/output/samples/*/sample.json")
    ):
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
    target = config.balance.target_counts()["channel"]
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
        per_house=config.balance.max_samples_per_house["channel"],
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
    house_shards = split_even(sorted(grouped), config.runtime.workers)
    shard_root = raw_root / "channel_shards"
    jobs = []
    for index, house_indices in enumerate(house_shards):
        benchmark_dir = shard_root / f"shard_{index:03d}" / "input"
        output_dir = shard_root / f"shard_{index:03d}" / "output"
        shard_episodes = [episode for house in house_indices for episode in grouped[house]]
        prior_log = shard_root / f"shard_{index:03d}" / "run.log"
        completed_indices: set[int] = set()
        if config.runtime.resume and prior_log.exists():
            import re

            for match in re.finditer(r"\[(?:ok|fail)\] ep=(\d+)", prior_log.read_text(errors="ignore")):
                completed_indices.add(int(match.group(1)))
        if completed_indices:
            shard_episodes = [
                episode
                for episode_index, episode in enumerate(shard_episodes)
                if episode_index not in completed_indices
            ]
        if not shard_episodes:
            continue
        write_json(benchmark_dir / "benchmark.json", shard_episodes)
        command = [
            PYTHON,
            "scripts/InteractiveNav/build_door_interaction_benchmark.py",
            "--benchmark_dir", str(benchmark_dir),
            "--output_dir", str(output_dir),
            "--mode", "build",
            "--input_mode", "original",
            "--variant", config.source.variant,
            "--max_episodes", str(len(shard_episodes)),
            "--num_distractor_samples_per_episode", "1",
            "--num_mixed_samples_per_critical_door", "1",
            "--distractor_k_min", str(config.collection.domains.channel.distractor_k_min),
            "--distractor_k_max", str(config.collection.domains.channel.distractor_k_max),
        ]
        jobs.append((index, command, output_dir, len(shard_episodes)))
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs) or 1) as executor:
        futures = {
            executor.submit(
                run_command,
                command,
                log_path=shard_root / f"shard_{index:03d}" / "run.log",
                env=env,
            ): (index, output_dir, episode_count)
            for index, command, output_dir, episode_count in jobs
        }
        for future in as_completed(futures):
            index, output_dir, episode_count = futures[future]
            results.append(
                {
                    "shard": index,
                    "returncode": future.result(),
                    "episode_count": episode_count,
                    "output_dir": str(output_dir),
                }
            )
    benchmark: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda row: row["shard"]):
        directory = Path(result["output_dir"])
        if (directory / "benchmark.json").exists():
            benchmark.extend(load_episodes(directory / "benchmark.json"))
        if (directory / "failures.json").exists():
            failures.extend(read_json(directory / "failures.json"))
    for sample_path in sorted(shard_root.glob("shard_*/output/samples/*/sample.json")):
        try:
            benchmark.append(read_json(sample_path))
        except (OSError, json.JSONDecodeError):
            continue
    benchmark = _deduplicate(benchmark)
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
            "worker_count": len(jobs),
            "shards": results,
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


def balance_benchmark(config: CollectionConfig, raw_paths: dict[str, Path]) -> Path:
    targets = config.balance.target_counts()
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
        per_house=config.balance.max_samples_per_house["channel"],
    )
    selected_container = _round_robin_by_house(
        container,
        targets["container"],
        config.balance.max_samples_per_house["container"],
    )
    selected_mixed = _round_robin_by_house(
        mixed,
        targets["mixed"],
        config.balance.max_samples_per_house["mixed"],
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
    selected: list[tuple[str, dict[str, Any]]] = []
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
        candidates = candidates[: config.full.max_episodes]
        if len(candidates) < config.full.max_episodes:
            raise RuntimeError(
                f"Requested {config.full.max_episodes} full {domain} episodes, "
                f"found {len(candidates)}"
            )
        selected.extend((domain, episode) for episode in candidates)
    full_root = output_root(config) / "full"
    runs = []
    common_env = os.environ.copy()
    common_env["PYTHONPATH"] = f"{REPO_ROOT}:{common_env.get('PYTHONPATH', '')}"
    common_env["MUJOCO_GL"] = config.runtime.mujoco_gl
    common_env.setdefault("MPLCONFIGDIR", str(full_root / "matplotlib-cache"))
    for domain, episode in selected:
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
            "--video_fps",
            str(config.full.video_fps),
            "--required_open_fraction",
            str(config.full.required_open_fraction),
            "--img_width",
            str(config.full.image_width),
            "--img_height",
            str(config.full.image_height),
            "--force_fallback_max_steps",
            str(max(config.policy.channel.max_steps, config.policy.container.max_steps)),
        ]
        if domain != "mixed":
            command.extend(["--domain", domain])
        log_path = full_root / "logs" / f"{domain}__{case_id}.log"
        returncode = run_command(command, log_path=log_path, env=common_env)
        candidates = sorted((full_root / "runs").glob(f"*{case_id[:48]}*"))
        run_dir = candidates[-1] if candidates else None
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
        force_step_count = int((audit or {}).get("action_type_counts", {}).get("force_joint", 0))
        training_eligible = bool(
            returncode == 0
            and audit is not None
            and audit.get("success") is True
            and required_segments.issubset(observed_segments)
            and int(audit.get("terminal_step_count", 0)) >= 1
            and (executor != "force" or force_step_count >= 1)
        )
        runs.append(
            {
                "domain": domain,
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
            "requested_episode_count": len(selected),
            "requested_episodes_per_domain": config.full.max_episodes,
            "valid_trajectory_count": sum(row["training_eligible"] for row in runs),
            "runs": runs,
        },
    )
    if any(not row["training_eligible"] for row in runs):
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
            "balance",
            "audit",
            "full",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--worker_houses")
    parser.add_argument("--worker_output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_collection_config(args.config)
    save_resolved_config(config)
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
    if args.stage == "seed-worker":
        if not args.worker_houses or args.worker_output is None:
            raise ValueError("seed-worker requires --worker_houses and --worker_output")
        houses = [int(value) for value in args.worker_houses.split(",") if value]
        return run_seed_worker(config, houses, args.worker_output)
    if args.stage in {"seeds", "all"}:
        seed_benchmark = build_seed_benchmark(config)
    else:
        seed_benchmark = source_benchmark_path(config)
    if args.stage == "seeds":
        print(seed_benchmark)
        return 0
    if args.stage in {"light", "all"}:
        raw_paths = run_light_collectors(config, seed_benchmark)
    else:
        root = output_root(config) / "raw"
        raw_paths = {
            "channel": root / "channel" / "benchmark.json",
            "container": root / "container" / "benchmark.json",
            "mixed": root / "mixed" / "benchmark.json",
        }
    if args.stage in {"balance", "all"}:
        benchmark = balance_benchmark(config, raw_paths)
    else:
        benchmark = output_root(config) / "balanced" / "benchmark.json"
    if args.stage in {"audit", "all"}:
        audit = audit_benchmark(config, benchmark)
        print(json.dumps({"benchmark": str(benchmark), "audit": str(audit)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
