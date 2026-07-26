"""Deterministic task-manifest scheduler for light InteractiveNav collection.

The manifest owns sample identity and ordering.  Workers only execute the next
reserved task, which makes resume and worker-count changes independent of data
selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from scripts.InteractiveNav.collection.config import CollectionConfig
from scripts.InteractiveNav.select_container_interaction_candidates import (
    build_dynamic_collection_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
CHANNEL_RECIPES = (
    "single_path_door_closed",
    "distractor_doors_closed",
    "mixed_critical_and_distractor_closed",
)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _case_id(episode: dict[str, Any]) -> str:
    return str(episode.get("interactive_nav", {}).get("case_id", ""))


def _deduplicate(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for episode in episodes:
        case_id = _case_id(episode)
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        result.append(episode)
    return result


def _source_index(episode: dict[str, Any], fallback: int) -> int:
    return int(episode.get("seed_generation", {}).get("source_episode_index", fallback))


def _stable_order(values: list[int], *, seed: int, domain: str) -> list[int]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}:{domain}:{value}".encode()).hexdigest(),
            value,
        ),
    )


def _recipe_for_position(position: int, total: int) -> str:
    # Deterministic 60/20/20 allocation. Each Channel round uses the same
    # proportions, while the central state records failures explicitly.
    value = (position + 0.5) / max(total, 1)
    if value < 0.60:
        return CHANNEL_RECIPES[0]
    if value < 0.80:
        return CHANNEL_RECIPES[1]
    return CHANNEL_RECIPES[2]


def _path_bin(length_m: float) -> str:
    for lower, upper in zip((0.0, 3.0, 5.0, 8.0, 12.0, 20.0), (3.0, 5.0, 8.0, 12.0, 20.0, float("inf"))):
        if lower <= length_m < upper:
            return f"[{lower:g},{upper:g})"
    return "unknown"


def _round_robin_strata(tasks: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Interleave static strata without depending on worker completion order."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[tuple(str(task.get(key, "unknown")) for key in keys)].append(task)
    for values in groups.values():
        values.sort(key=lambda task: str(task["task_id"]))
    ordered_groups = sorted(groups)
    result = []
    while True:
        progressed = False
        for key in ordered_groups:
            values = groups[key]
            if values:
                result.append(values.pop(0))
                progressed = True
        if not progressed:
            return result


def _channel_tasks(seed_episodes: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_house: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for local_index, episode in enumerate(seed_episodes):
        by_house[int(episode["house_index"])].append((_source_index(episode, local_index), episode))
    houses = _stable_order(sorted(by_house), seed=seed, domain="channel")
    for values in by_house.values():
        values.sort(key=lambda value: value[0])
    tasks = []
    max_rounds = max((len(values) for values in by_house.values()), default=0)
    for round_index in range(max_rounds):
        round_houses = [house for house in houses if round_index < len(by_house[house])]
        for position, house_index in enumerate(round_houses):
            source_index, episode = by_house[house_index][round_index]
            task_id = f"channel:r{round_index}:h{house_index}:s{source_index}"
            tasks.append(
                {
                    "task_id": task_id,
                    "domain": "channel",
                    "round": round_index,
                    "house_index": house_index,
                    "source_episode_index": source_index,
                    "episode_index_offset": len(tasks) + 1,
                    "recipe": _recipe_for_position(position, len(round_houses)),
                    "episode": episode,
                }
            )
    return tasks


def _container_tasks(
    rough_catalog: Path,
    *,
    house_indices: list[int],
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    # This cap constructs reserve slots for the static rough manifest only. It
    # is not a final-data house cap.
    per_house_slots = max(1, (target + max(len(house_indices), 1) - 1) // max(len(house_indices), 1))
    plan = build_dynamic_collection_plan(
        rough_catalog,
        max_samples=len(house_indices) * per_house_slots,
        samples_per_house=per_house_slots,
        house_indices=house_indices,
        seed=seed,
    )
    tasks = []
    for house in plan.get("houses", []):
        for slot_index, candidate in enumerate(house.get("candidates", [])[:per_house_slots]):
            tasks.append(
                {
                    "task_id": (
                        f"container:h{int(house['house_index'])}:slot{slot_index}:"
                        f"{candidate.get('container_name')}:{candidate.get('object_name')}"
                    ),
                    "domain": "container",
                    "house_index": int(house["house_index"]),
                    "slot": slot_index,
                    "target_category": str(candidate.get("object_category", "unknown")).lower(),
                    "container_type": str(candidate.get("container_category", "unknown")).lower(),
                    "path_bin": str(candidate.get("selected_start_distance_bin", "unknown")),
                    "candidate": candidate,
                }
            )
    return _round_robin_strata(tasks, ("target_category", "container_type", "path_bin"))


def _mixed_tasks(
    mixed_rough: Path,
    *,
    source_episode_indices: set[int],
    source_houses: set[int],
    variants_per_pair: int,
) -> list[dict[str, Any]]:
    payload = _read_json(mixed_rough, {})
    tasks = []
    for candidate in payload.get("candidates", []):
        if (
            candidate.get("rough_candidate_type") != "mixed_required_verified"
            or candidate.get("mixed_required_verified") is not True
            or int(candidate.get("house_index", -1)) not in source_houses
        ):
            continue
        case_id = str(candidate["case_id"])
        options = []
        for option in candidate.get("path_options", []):
            source_index = int(option["source_episode_index"])
            if source_index in source_episode_indices:
                options.append((source_index, float(option.get("all_open_path_length_m", 0.0))))
            if len(options) >= variants_per_pair:
                break
        if not options:
            for source_index in candidate.get("source_episode_indices", []):
                source_index = int(source_index)
                if source_index in source_episode_indices:
                    options.append((source_index, float(candidate.get("all_open_path_length_m", 0.0))))
                if len(options) >= variants_per_pair:
                    break
        for source_index, path_length_m in options:
            tasks.append(
                {
                    "task_id": f"mixed:{case_id}:src{source_index}",
                    "domain": "mixed",
                    "case_id": case_id,
                    "candidate_key": f"{case_id}::src{source_index}",
                    "house_index": int(candidate["house_index"]),
                    "source_episode_index": source_index,
                    "target_category": str(candidate.get("target_category", "unknown")).lower(),
                    "container_type": str(candidate.get("container_category", "unknown")).lower(),
                    "path_length_m": path_length_m,
                    "path_bin": _path_bin(path_length_m),
                }
            )
    return _round_robin_strata(tasks, ("target_category", "container_type", "path_bin"))


def build_or_load_manifests(
    config: CollectionConfig,
    *,
    root: Path,
    seed_benchmark: Path,
    container_rough: Path,
    mixed_rough: Path,
    targets: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    manifest_root = root / "raw" / "task_manifests"
    paths = {domain: manifest_root / f"{domain}.json" for domain in targets}
    if all(path.exists() for path in paths.values()):
        return {domain: _read_json(path, {}).get("tasks", []) for domain, path in paths.items()}
    seed_episodes = _read_json(seed_benchmark, [])
    source_houses = sorted({int(episode["house_index"]) for episode in seed_episodes})
    source_indices = {_source_index(episode, index) for index, episode in enumerate(seed_episodes)}
    manifests = {
        "channel": _channel_tasks(seed_episodes, seed=config.runtime.seed),
        "container": _container_tasks(
            container_rough,
            house_indices=source_houses,
            target=targets["container"],
            seed=config.runtime.seed,
        ),
        "mixed": _mixed_tasks(
            mixed_rough,
            source_episode_indices=source_indices,
            source_houses=set(source_houses),
            variants_per_pair=config.collection.domains.mixed.source_variants_per_pair,
        ),
    }
    for domain, tasks in manifests.items():
        _write_json(
            paths[domain],
            {
                "schema_version": "interactive_nav_manifest_v2",
                "domain": domain,
                "seed": config.runtime.seed,
                "target": targets[domain],
                "task_count": len(tasks),
                "tasks": tasks,
            },
        )
    return manifests


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path, [])
    if isinstance(payload, dict):
        payload = payload.get("episodes", [])
    return payload if isinstance(payload, list) else []


def _run(command: list[str], *, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        return subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode


def _formal_episode(domain: str, episode: dict[str, Any], recipe: str | None = None) -> bool:
    interactive = episode.get("interactive_nav", {})
    if interactive.get("schema_version") != "interactive_nav_v3":
        return False
    if domain == "channel":
        case_type = interactive.get("legacy_case_type")
        return (
            case_type in CHANNEL_RECIPES
            if recipe is None
            else case_type == recipe
        )
    if domain == "container":
        return interactive.get("interaction_requirement") == "required"
    return (
        interactive.get("interaction_domains") == ["channel", "container"]
        and interactive.get("interaction_requirement") == "required"
    )


def _execute_task(
    task: dict[str, Any],
    *,
    task_root: Path,
    seed_benchmark: Path,
    container_rough: Path,
    mixed_rough: Path,
    config: CollectionConfig,
) -> dict[str, Any]:
    domain = task["domain"]
    run_dir = task_root / task["task_id"].replace(":", "__").replace("/", "_")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MUJOCO_GL"] = config.runtime.mujoco_gl
    env["MPLCONFIGDIR"] = str(run_dir / "matplotlib-cache")
    if domain == "channel":
        input_dir = run_dir / "input"
        _write_json(input_dir / "benchmark.json", [task["episode"]])
        output_dir = run_dir / "output"
        returncode = _run(
            [
                PYTHON,
                "scripts/InteractiveNav/build_door_interaction_benchmark.py",
                "--benchmark_dir", str(input_dir),
                "--output_dir", str(output_dir),
                "--mode", "build", "--input_mode", "original", "--variant", config.source.variant,
                "--max_episodes", "1",
                "--episode_index_offset", str(task["episode_index_offset"]),
                "--case_type_filter", str(task["recipe"]),
                "--max_output_samples_per_house", "1",
                "--num_distractor_samples_per_episode", "1",
                "--num_mixed_samples_per_critical_door", "1",
                "--distractor_k_min", str(config.collection.domains.channel.distractor_k_min),
                "--distractor_k_max", str(config.collection.domains.channel.distractor_k_max),
            ],
            log_path=run_dir / "run.log", env=env,
        )
        episodes = _load_episodes(output_dir / "benchmark.json")
        episodes = [episode for episode in episodes if _formal_episode("channel", episode, task["recipe"])]
    elif domain == "container":
        manifest = {
            "schema_version": "container_interaction_collection_plan_v2",
            "selection": {"mode": "manifest", "requested_sample_count": 1},
            "houses": [{"house_index": task["house_index"], "target_sample_count": 1, "candidates": [task["candidate"]]}],
        }
        manifest_path = run_dir / "candidate_manifest.json"
        _write_json(manifest_path, manifest)
        output_dir = run_dir / "output"
        returncode = _run(
            [
                PYTHON,
                "scripts/InteractiveNav/collect_container_fine_parallel.py",
                "--benchmark_dir", str(seed_benchmark),
                "--candidate_manifest", str(manifest_path),
                "--output_dir", str(output_dir), "--workers", "1",
                "--seed", str(config.runtime.seed), "--mujoco_gl", config.runtime.mujoco_gl,
                "--no-save_images", "--no-save_plots",
            ],
            log_path=run_dir / "run.log", env=env,
        )
        episodes = _load_episodes(output_dir / "benchmark.json")
        episodes = [episode for episode in episodes if _formal_episode("container", episode)]
    else:
        output_dir = run_dir / "output"
        plan_path = run_dir / "candidate_plan.json"
        _write_json(
            plan_path,
            {
                "schema_version": "mixed_candidate_plan_v1",
                # The mixed builder expands rough candidates itself; its plan
                # only needs the stable key including the selected source.
                "candidates": [{"candidate_key": task["candidate_key"]}],
            },
        )
        returncode = _run(
            [
                PYTHON,
                "scripts/InteractiveNav/build_mixed_interaction_benchmark.py",
                "--mixed_rough_catalog", str(mixed_rough),
                "--benchmark_dir", str(seed_benchmark),
                "--output_dir", str(output_dir),
                "--candidate_plan", str(plan_path), "--max_samples", "1",
                "--rough_candidate_types", "mixed_required_verified",
                "--max_samples_per_house", "1", "--variant", config.source.variant,
                "--source_variants_per_pair",
                str(config.collection.domains.mixed.source_variants_per_pair),
                "--seed", str(config.runtime.seed),
                "--candidate_timeout_seconds", str(config.runtime.candidate_timeout_seconds),
                "--no-save_images", "--no-save_plots",
            ],
            log_path=run_dir / "run.log", env=env,
        )
        episodes = _load_episodes(output_dir / "benchmark.json")
        episodes = [episode for episode in episodes if _formal_episode("mixed", episode)]
    return {
        "task_id": task["task_id"], "domain": domain, "returncode": returncode,
        "episodes": episodes[:1], "run_dir": str(run_dir),
    }


def _load_domain_episodes(root: Path, domain: str) -> list[dict[str, Any]]:
    return _deduplicate(_load_episodes(root / "raw" / domain / "benchmark.json"))


def _write_domain_episodes(root: Path, domain: str, episodes: list[dict[str, Any]]) -> None:
    _write_json(root / "raw" / domain / "benchmark.json", _deduplicate(episodes))


def _desired_allocations(active: list[str], counts: dict[str, int], targets: dict[str, int], workers: int) -> dict[str, int]:
    allocation = {domain: 1 for domain in active}
    priority = {"mixed": 0, "container": 1, "channel": 2}
    for _ in range(max(0, workers - len(active))):
        domain = max(
            active,
            key=lambda value: ((targets[value] - counts[value]) / max(targets[value], 1), -priority[value]),
        )
        allocation[domain] += 1
    return allocation


def _reset_empty_mixed_plan_tasks(
    completed: dict[str, dict[str, Any]],
) -> list[str]:
    """Requeue tasks completed by the pre-v3 source-variant wiring bug.

    The faulty invocation did not expand the rough candidate to its selected
    source episode.  Its plan key could therefore never match and the builder
    recorded both zero generated episodes and zero collection houses.  This is
    distinct from a real validation rejection, which opens one house and has a
    rejection record.
    """
    requeue = []
    for task_id, result in completed.items():
        if result.get("domain") != "mixed":
            continue
        run_dir = result.get("run_dir")
        if not run_dir:
            continue
        summary = _read_json(Path(run_dir) / "output" / "summary.json", {})
        if (
            summary.get("generated_episode_count") == 0
            and summary.get("collection_house_count") == 0
            and summary.get("rejected_candidate_count") == 0
        ):
            requeue.append(task_id)
    for task_id in requeue:
        completed.pop(task_id, None)
    return requeue


def run_manifest_light_collection(
    config: CollectionConfig,
    *,
    root: Path,
    seed_benchmark: Path,
    container_rough: Path,
    mixed_rough: Path,
    targets: dict[str, int],
) -> dict[str, Path]:
    manifests = build_or_load_manifests(
        config, root=root, seed_benchmark=seed_benchmark, container_rough=container_rough,
        mixed_rough=mixed_rough, targets=targets,
    )
    state_path = root / "raw" / "task_manifests" / "state.json"
    state = _read_json(state_path, {"completed": {}, "history": []})
    completed: dict[str, dict[str, Any]] = state.get("completed", {})
    requeued_mixed = _reset_empty_mixed_plan_tasks(completed)
    if requeued_mixed:
        state["completed"] = completed
        state.setdefault("migrations", []).append(
            {
                "name": "retry_mixed_source_variant_plan_v1",
                "time": time.time(),
                "requeued_task_count": len(requeued_mixed),
            }
        )
        _write_json(state_path, state)
        print(
            "[manifest-migration] requeued "
            f"{len(requeued_mixed)} empty mixed candidate-plan tasks",
            flush=True,
        )
    inflight: set[str] = set()
    domain_episodes = {domain: _load_domain_episodes(root, domain) for domain in targets}
    task_root = root / "raw" / "task_runs"

    def counts() -> dict[str, int]:
        return {domain: len(domain_episodes[domain]) for domain in targets}

    def next_task(domain: str) -> dict[str, Any] | None:
        tasks = manifests[domain]
        pending = [task for task in tasks if task["task_id"] not in completed and task["task_id"] not in inflight]
        if domain == "channel" and pending:
            first_round = min(int(task["round"]) for task in pending)
            pending = [task for task in pending if int(task["round"]) == first_round]
        return pending[0] if pending else None

    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=config.runtime.workers) as executor:
        while True:
            current = counts()
            active = [
                domain for domain in targets
                if current[domain] < targets[domain] and next_task(domain) is not None
            ]
            allocation = _desired_allocations(active, current, targets, config.runtime.workers) if active else {}
            while len(futures) < config.runtime.workers:
                inflight_by_domain = Counter(task["domain"] for task in futures.values())
                eligible = [domain for domain in active if inflight_by_domain[domain] < allocation[domain]]
                if not eligible:
                    break
                domain = max(eligible, key=lambda value: (targets[value] - current[value], value))
                task = next_task(domain)
                if task is None:
                    break
                inflight.add(task["task_id"])
                future = executor.submit(
                    _execute_task, task, task_root=task_root, seed_benchmark=seed_benchmark,
                    container_rough=container_rough, mixed_rough=mixed_rough, config=config,
                )
                futures[future] = task
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                inflight.discard(task["task_id"])
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - subprocess wrapper guard
                    result = {"task_id": task["task_id"], "domain": task["domain"], "returncode": -1, "episodes": [], "error": repr(exc)}
                domain = task["domain"]
                accepted = []
                if counts()[domain] < targets[domain]:
                    accepted = result.get("episodes", [])[: max(targets[domain] - counts()[domain], 0)]
                    domain_episodes[domain] = _deduplicate(domain_episodes[domain] + accepted)
                    _write_domain_episodes(root, domain, domain_episodes[domain])
                completed[task["task_id"]] = {
                    "domain": domain, "returncode": result.get("returncode"),
                    "accepted_case_ids": [_case_id(episode) for episode in accepted],
                    "run_dir": result.get("run_dir"),
                }
                state["completed"] = completed
                state.setdefault("history", []).append({"time": time.time(), "counts": counts(), "task_id": task["task_id"]})
                _write_json(state_path, state)
                print(f"[manifest-progress] domain={domain} counts={counts()} completed={len(completed)}", flush=True)

    final_counts = counts()
    _write_json(
        root / "raw" / "task_manifests" / "summary.json",
        {
            "schema_version": "interactive_nav_manifest_v2_summary",
            "targets": targets, "counts": final_counts,
            "completed_task_count": len(completed),
            "remaining_task_count": {
                domain: sum(task["task_id"] not in completed for task in tasks)
                for domain, tasks in manifests.items()
            },
        },
    )
    return {domain: root / "raw" / domain / "benchmark.json" for domain in targets}


# House-batch collection ----------------------------------------------------
#
# The original manifest scheduler made every rough candidate an independent
# subprocess.  A failed candidate therefore paid the full MuJoCo house-load
# cost again.  These helpers deliberately keep a whole deterministic candidate
# queue in one task: one process opens one house, validates candidates until the
# per-house quota is reached, then releases the context.


def _house_quota(target: int, house_count: int) -> int:
    if house_count <= 0:
        raise ValueError("Cannot build a house-batch manifest without eligible houses")
    return max(1, math.ceil(target / house_count))


def _channel_house_tasks(
    seed_episodes: list[dict[str, Any]], *, target: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    by_house: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for local_index, episode in enumerate(seed_episodes):
        by_house[int(episode["house_index"])].append(
            (_source_index(episode, local_index), episode)
        )
    houses = _stable_order(sorted(by_house), seed=seed, domain="channel-house")
    quota = _house_quota(target, len(houses))
    offset = 1
    tasks = []
    for house_index in houses:
        values = sorted(by_house[house_index], key=lambda value: value[0])
        tasks.append(
            {
                "task_id": f"channel-house:h{house_index}",
                "domain": "channel",
                "house_index": house_index,
                "house_quota": quota,
                # Keep IDs unique across per-house temporary benchmark files.
                "episode_index_offset": offset,
                "episodes": [episode for _, episode in values],
            }
        )
        offset += len(values)
    return tasks, quota


def _container_house_tasks(
    rough_catalog: Path,
    *,
    source_houses: set[int],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    # Request a quota for every eligible house.  The scheduler stops globally
    # at ``target``; the extra per-house capacity is only a fallback for houses
    # that fail fine validation.
    all_candidates = build_dynamic_collection_plan(
        rough_catalog,
        max_samples=max(1, len(source_houses)),
        samples_per_house=1,
        house_indices=_stable_order(
            sorted(source_houses), seed=seed, domain="container-house-filter"
        ),
        seed=seed,
    )
    eligible_houses = [int(row["house_index"]) for row in all_candidates.get("houses", [])]
    quota = _house_quota(target, len(eligible_houses))
    plan = build_dynamic_collection_plan(
        rough_catalog,
        max_samples=len(eligible_houses) * quota,
        samples_per_house=quota,
        house_indices=eligible_houses,
        seed=seed,
    )
    tasks = [
        {
            "task_id": f"container-house:h{int(house['house_index'])}",
            "domain": "container",
            "house_index": int(house["house_index"]),
            "house_quota": quota,
            "candidates": house.get("candidates", []),
        }
        for house in plan.get("houses", [])
    ]
    return tasks, quota


def _mixed_house_tasks(
    mixed_rough: Path,
    *,
    source_episode_indices: set[int],
    source_houses: set[int],
    variants_per_pair: int,
    target: int,
) -> tuple[list[dict[str, Any]], int]:
    # ``_mixed_tasks`` already gives a deterministic category/path interleave.
    # Preserve that order inside each house while moving every house into one
    # persistent fine-validation task.
    candidate_tasks = _mixed_tasks(
        mixed_rough,
        source_episode_indices=source_episode_indices,
        source_houses=source_houses,
        variants_per_pair=variants_per_pair,
    )
    by_house: dict[int, list[dict[str, Any]]] = defaultdict(list)
    house_order: list[int] = []
    seen_houses: set[int] = set()
    for candidate in candidate_tasks:
        house_index = int(candidate["house_index"])
        by_house[house_index].append(candidate)
        if house_index not in seen_houses:
            seen_houses.add(house_index)
            house_order.append(house_index)
    quota = _house_quota(target, len(house_order))
    tasks = [
        {
            "task_id": f"mixed-house:h{house_index}",
            "domain": "mixed",
            "house_index": house_index,
            "house_quota": quota,
            "candidate_keys": [row["candidate_key"] for row in by_house[house_index]],
        }
        for house_index in house_order
    ]
    return tasks, quota


def build_or_load_house_manifests(
    config: CollectionConfig,
    *,
    root: Path,
    seed_benchmark: Path,
    container_rough: Path,
    mixed_rough: Path,
    targets: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    manifest_root = root / "raw" / "house_manifests"
    paths = {domain: manifest_root / f"{domain}.json" for domain in targets}
    if all(path.exists() for path in paths.values()):
        return {domain: _read_json(path, {}).get("tasks", []) for domain, path in paths.items()}

    seed_episodes = _read_json(seed_benchmark, [])
    source_houses = {int(episode["house_index"]) for episode in seed_episodes}
    source_indices = {
        _source_index(episode, index) for index, episode in enumerate(seed_episodes)
    }
    channel, channel_quota = _channel_house_tasks(
        seed_episodes, target=targets["channel"], seed=config.runtime.seed
    )
    container, container_quota = _container_house_tasks(
        container_rough,
        source_houses=source_houses,
        target=targets["container"],
        seed=config.runtime.seed,
    )
    mixed, mixed_quota = _mixed_house_tasks(
        mixed_rough,
        source_episode_indices=source_indices,
        source_houses=source_houses,
        variants_per_pair=config.collection.domains.mixed.source_variants_per_pair,
        target=targets["mixed"],
    )
    manifests = {"channel": channel, "container": container, "mixed": mixed}
    quotas = {
        "channel": channel_quota,
        "container": container_quota,
        "mixed": mixed_quota,
    }
    for domain, tasks in manifests.items():
        _write_json(
            paths[domain],
            {
                "schema_version": "interactive_nav_house_batch_manifest_v1",
                "domain": domain,
                "seed": config.runtime.seed,
                "target": targets[domain],
                "per_house_quota": quotas[domain],
                "eligible_house_count": len(tasks),
                "task_count": len(tasks),
                "tasks": tasks,
            },
        )
    return manifests


def _formal_house_episode(domain: str, episode: dict[str, Any]) -> bool:
    if not _formal_episode(domain, episode, None):
        return False
    return True


def _execute_house_task(
    task: dict[str, Any],
    *,
    task_root: Path,
    seed_benchmark: Path,
    mixed_rough: Path,
    config: CollectionConfig,
) -> dict[str, Any]:
    domain = task["domain"]
    quota = int(task["house_quota"])
    run_dir = task_root / task["task_id"].replace(":", "__").replace("/", "_")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MUJOCO_GL"] = config.runtime.mujoco_gl
    env["MPLCONFIGDIR"] = str(run_dir / "matplotlib-cache")

    if domain == "channel":
        input_dir = run_dir / "input"
        _write_json(input_dir / "benchmark.json", task["episodes"])
        output_dir = run_dir / "output"
        returncode = _run(
            [
                PYTHON,
                "scripts/InteractiveNav/build_door_interaction_benchmark.py",
                "--benchmark_dir", str(input_dir),
                "--output_dir", str(output_dir),
                "--mode", "build", "--input_mode", "original", "--variant", config.source.variant,
                "--max_episodes", str(len(task["episodes"])),
                "--episode_index_offset", str(task["episode_index_offset"]),
                # Filter before applying the per-house cap.  Otherwise the
                # lexically first all_closed record consumes a slot and is
                # discarded only later by _formal_house_episode.
                "--case_type_filter", ",".join(CHANNEL_RECIPES),
                "--max_output_samples_per_house", str(quota),
                "--num_distractor_samples_per_episode", "1",
                "--num_mixed_samples_per_critical_door", "1",
                "--distractor_k_min", str(config.collection.domains.channel.distractor_k_min),
                "--distractor_k_max", str(config.collection.domains.channel.distractor_k_max),
            ],
            log_path=run_dir / "run.log", env=env,
        )
        episodes = [
            episode for episode in _load_episodes(output_dir / "benchmark.json")
            if _formal_house_episode("channel", episode)
        ]
    elif domain == "container":
        manifest = {
            "schema_version": "container_interaction_collection_plan_v2",
            "selection": {"mode": "house_batch", "requested_sample_count": quota},
            "houses": [
                {
                    "house_index": task["house_index"],
                    "target_sample_count": quota,
                    "candidates": task["candidates"],
                }
            ],
        }
        manifest_path = run_dir / "candidate_manifest.json"
        _write_json(manifest_path, manifest)
        output_dir = run_dir / "output"
        returncode = _run(
            [
                PYTHON,
                "scripts/InteractiveNav/collect_container_fine_parallel.py",
                "--benchmark_dir", str(seed_benchmark),
                "--candidate_manifest", str(manifest_path),
                "--output_dir", str(output_dir), "--workers", "1",
                "--seed", str(config.runtime.seed), "--mujoco_gl", config.runtime.mujoco_gl,
                "--no-save_images", "--no-save_plots",
            ],
            log_path=run_dir / "run.log", env=env,
        )
        episodes = [
            episode for episode in _load_episodes(output_dir / "benchmark.json")
            if _formal_house_episode("container", episode)
        ]
    else:
        output_dir = run_dir / "output"
        _write_json(
            run_dir / "candidate_plan.json",
            {
                "schema_version": "mixed_candidate_plan_v1",
                "candidates": [
                    {"candidate_key": candidate_key}
                    for candidate_key in task["candidate_keys"]
                ],
            },
        )
        returncode = _run(
            [
                PYTHON,
                "scripts/InteractiveNav/build_mixed_interaction_benchmark.py",
                "--mixed_rough_catalog", str(mixed_rough),
                "--benchmark_dir", str(seed_benchmark),
                "--output_dir", str(output_dir),
                "--candidate_plan", str(run_dir / "candidate_plan.json"),
                "--max_samples", str(quota),
                "--rough_candidate_types", "mixed_required_verified",
                "--max_samples_per_house", str(quota), "--variant", config.source.variant,
                "--source_variants_per_pair",
                str(config.collection.domains.mixed.source_variants_per_pair),
                "--seed", str(config.runtime.seed),
                "--candidate_timeout_seconds", str(config.runtime.candidate_timeout_seconds),
                "--no-save_images", "--no-save_plots",
            ],
            log_path=run_dir / "run.log", env=env,
        )
        episodes = [
            episode for episode in _load_episodes(output_dir / "benchmark.json")
            if _formal_house_episode("mixed", episode)
        ]
    return {
        "task_id": task["task_id"],
        "domain": domain,
        "house_index": int(task["house_index"]),
        "returncode": returncode,
        "episodes": episodes[:quota],
        "run_dir": str(run_dir),
    }


def run_house_batch_light_collection(
    config: CollectionConfig,
    *,
    root: Path,
    seed_benchmark: Path,
    container_rough: Path,
    mixed_rough: Path,
    targets: dict[str, int],
) -> dict[str, Path]:
    manifests = build_or_load_house_manifests(
        config,
        root=root,
        seed_benchmark=seed_benchmark,
        container_rough=container_rough,
        mixed_rough=mixed_rough,
        targets=targets,
    )
    state_path = root / "raw" / "house_manifests" / "state.json"
    state = _read_json(state_path, {"completed": {}, "history": []})
    completed: dict[str, dict[str, Any]] = state.get("completed", {})
    domain_episodes = {domain: _load_domain_episodes(root, domain) for domain in targets}
    task_root = root / "raw" / "house_runs"
    inflight: set[str] = set()

    def counts() -> dict[str, int]:
        return {domain: len(domain_episodes[domain]) for domain in targets}

    def house_count(domain: str, house_index: int) -> int:
        return sum(
            int(episode.get("house_index", -1)) == house_index
            for episode in domain_episodes[domain]
        )

    def next_task(domain: str) -> dict[str, Any] | None:
        for task in manifests[domain]:
            task_id = task["task_id"]
            if task_id in completed or task_id in inflight:
                continue
            if house_count(domain, int(task["house_index"])) >= int(task["house_quota"]):
                completed[task_id] = {
                    "domain": domain,
                    "house_index": int(task["house_index"]),
                    "returncode": None,
                    "accepted_case_ids": [],
                    "reason": "quota_already_satisfied_by_resumed_data",
                }
                state["completed"] = completed
                _write_json(state_path, state)
                continue
            return task
        return None

    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=config.runtime.workers) as executor:
        while True:
            current = counts()
            active = [
                domain
                for domain in targets
                if current[domain] < targets[domain] and next_task(domain) is not None
            ]
            allocation = (
                _desired_allocations(active, current, targets, config.runtime.workers)
                if active
                else {}
            )
            while len(futures) < config.runtime.workers:
                inflight_by_domain = Counter(task["domain"] for task in futures.values())
                eligible = [
                    domain
                    for domain in active
                    if inflight_by_domain[domain] < allocation[domain]
                ]
                if not eligible:
                    break
                domain = max(
                    eligible,
                    key=lambda value: (targets[value] - current[value], value),
                )
                task = next_task(domain)
                if task is None:
                    break
                inflight.add(task["task_id"])
                future = executor.submit(
                    _execute_house_task,
                    task,
                    task_root=task_root,
                    seed_benchmark=seed_benchmark,
                    mixed_rough=mixed_rough,
                    config=config,
                )
                futures[future] = task
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                inflight.discard(task["task_id"])
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - subprocess wrapper guard
                    result = {
                        "task_id": task["task_id"],
                        "domain": task["domain"],
                        "house_index": task["house_index"],
                        "returncode": -1,
                        "episodes": [],
                        "error": repr(exc),
                    }
                domain = task["domain"]
                known_case_ids = {
                    _case_id(episode) for episode in domain_episodes[domain]
                }
                remaining_domain = max(targets[domain] - counts()[domain], 0)
                remaining_house = max(
                    int(task["house_quota"])
                    - house_count(domain, int(task["house_index"])),
                    0,
                )
                accepted = []
                for episode in result.get("episodes", []):
                    case_id = _case_id(episode)
                    if (
                        not case_id
                        or case_id in known_case_ids
                        or len(accepted) >= remaining_domain
                        or len(accepted) >= remaining_house
                    ):
                        continue
                    accepted.append(episode)
                    known_case_ids.add(case_id)
                if accepted:
                    domain_episodes[domain] = _deduplicate(
                        domain_episodes[domain] + accepted
                    )
                    _write_domain_episodes(root, domain, domain_episodes[domain])
                completed[task["task_id"]] = {
                    "domain": domain,
                    "house_index": int(task["house_index"]),
                    "returncode": result.get("returncode"),
                    "accepted_case_ids": [_case_id(episode) for episode in accepted],
                    "run_dir": result.get("run_dir"),
                }
                state["completed"] = completed
                state.setdefault("history", []).append(
                    {
                        "time": time.time(),
                        "counts": counts(),
                        "task_id": task["task_id"],
                        "house_index": int(task["house_index"]),
                    }
                )
                _write_json(state_path, state)
                print(
                    "[house-batch-progress] "
                    f"domain={domain} house={task['house_index']} "
                    f"accepted={len(accepted)} counts={counts()} "
                    f"completed={len(completed)}",
                    flush=True,
                )

    final_counts = counts()
    _write_json(
        root / "raw" / "house_manifests" / "summary.json",
        {
            "schema_version": "interactive_nav_house_batch_summary_v1",
            "targets": targets,
            "counts": final_counts,
            "completed_task_count": len(completed),
            "remaining_task_count": {
                domain: sum(task["task_id"] not in completed for task in tasks)
                for domain, tasks in manifests.items()
            },
        },
    )
    return {domain: root / "raw" / domain / "benchmark.json" for domain in targets}


def rebuild_channel_from_house_cache(
    *,
    root: Path,
    target: int,
    seed: int,
) -> Path:
    """Materialize Channel data from already-rendered house-batch case files.

    The first house-batch implementation capped each builder output before it
    removed ``all_closed`` cases.  The individual formal case JSON files still
    exist under ``house_runs``; rebuilding from them avoids reloading 898
    scenes.  Selection first covers every usable house up to its ceiling quota,
    then deterministically backfills from houses with remaining candidates.
    """
    run_root = root / "raw" / "house_runs"
    candidates_by_house: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for scan_index in sorted(run_root.glob("channel-house__*/output/scan_index.jsonl")):
        for line in scan_index.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            for case in row.get("case_summaries", []):
                if case.get("case_type") not in CHANNEL_RECIPES:
                    continue
                sample_path = Path(case.get("sample_path", ""))
                sample = _read_json(sample_path, {})
                if not _formal_episode("channel", sample, None):
                    continue
                case_id = _case_id(sample)
                if case_id:
                    candidates_by_house[int(sample["house_index"])][case_id] = sample
    if not candidates_by_house:
        raise RuntimeError("No cached formal Channel samples were found")

    def rank(sample: dict[str, Any], salt: str) -> tuple[int, str]:
        recipe = str(sample["interactive_nav"].get("legacy_case_type", ""))
        recipe_rank = CHANNEL_RECIPES.index(recipe)
        digest = hashlib.sha256(
            f"{seed}:{salt}:{_case_id(sample)}".encode()
        ).hexdigest()
        return recipe_rank, digest

    houses = _stable_order(
        sorted(candidates_by_house), seed=seed, domain="channel-cache-rebuild"
    )
    quota = _house_quota(target, len(houses))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_per_house: Counter[int] = Counter()
    recipe_counts: Counter[str] = Counter()

    # First pass: retain broad scene coverage while choosing the least-used
    # recipe available in each house.
    for _ in range(quota):
        for house_index in houses:
            if len(selected) >= target:
                break
            options = [
                sample
                for case_id, sample in candidates_by_house[house_index].items()
                if case_id not in selected_ids
            ]
            if not options:
                continue
            sample = min(
                options,
                key=lambda value: (
                    recipe_counts[
                        str(value["interactive_nav"].get("legacy_case_type", ""))
                    ],
                    rank(value, "coverage"),
                ),
            )
            case_id = _case_id(sample)
            selected.append(sample)
            selected_ids.add(case_id)
            selected_per_house[house_index] += 1
            recipe_counts[str(sample["interactive_nav"]["legacy_case_type"])] += 1

    # Some houses have fewer than the ceiling quota.  Backfill deterministically
    # from the remaining verified cases instead of ending below the global goal.
    remaining = [
        sample
        for house_index in houses
        for case_id, sample in candidates_by_house[house_index].items()
        if case_id not in selected_ids
    ]
    while remaining and len(selected) < target:
        sample = min(
            remaining,
            key=lambda value: (
                recipe_counts[
                    str(value["interactive_nav"].get("legacy_case_type", ""))
                ],
                selected_per_house[int(value["house_index"])],
                rank(value, "backfill"),
            ),
        )
        case_id = _case_id(sample)
        selected.append(sample)
        selected_ids.add(case_id)
        selected_per_house[int(sample["house_index"])] += 1
        recipe_counts[str(sample["interactive_nav"]["legacy_case_type"])] += 1
        remaining = [value for value in remaining if _case_id(value) != case_id]

    if len(selected) < target:
        raise RuntimeError(
            f"Cached Channel cases are insufficient: {len(selected)}/{target}"
        )
    selected.sort(key=lambda value: _case_id(value))
    output_path = root / "raw" / "channel" / "benchmark.json"
    _write_json(output_path, selected)
    _write_json(
        root / "raw" / "channel" / "cache_rebuild_summary.json",
        {
            "schema_version": "interactive_nav_channel_cache_rebuild_v1",
            "target": target,
            "selected_count": len(selected),
            "cached_candidate_count": sum(
                len(values) for values in candidates_by_house.values()
            ),
            "eligible_house_count": len(houses),
            "initial_per_house_quota": quota,
            "houses_above_initial_quota": sum(
                count > quota for count in selected_per_house.values()
            ),
            "recipe_counts": dict(recipe_counts),
        },
    )
    return output_path
