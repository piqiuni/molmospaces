#!/usr/bin/env python3
"""Rebuild repaired V3 Channel samples from the original scene generator.

The historical Channel target repair changed object references in frozen JSON,
but deliberately did not rebuild the path or door evidence.  This command
recreates the affected source episodes with the current Channel builder, in
parallel by source episode, and atomically materializes a complete replacement
Channel file.  Unaffected samples are copied byte-for-byte at the JSON-object
level and are never silently discarded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
CHANNEL_CASE_TYPES = {
    "single_path_door_closed",
    "distractor_doors_closed",
    "mixed_critical_and_distractor_closed",
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_episodes(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("episodes", payload.get("benchmark", []))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON episode list: {path}")
    return payload


def case_id(episode: dict[str, Any]) -> str:
    return str(episode.get("interactive_nav", {}).get("case_id", ""))


def target_name(episode: dict[str, Any]) -> str:
    interactive = episode.get("interactive_nav", {})
    return str(
        interactive.get("target", {}).get("selected_instance")
        or episode.get("task", {}).get("pickup_obj_name", "")
    )


def _source_xy(episode: dict[str, Any]) -> tuple[float, float] | None:
    pose = episode.get("task", {}).get("robot_base_pose")
    if not isinstance(pose, list) or len(pose) < 2:
        return None
    return float(pose[0]), float(pose[1])


def resolve_source_episode(
    repair: dict[str, Any], source_episodes: list[dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    """Resolve by physical source pose, not the stale generated parent index.

    ``parent_benchmark_episode_index`` is an output-builder index and is not
    guaranteed to be the index of the source nav benchmark after worker
    sharding.  House plus recorded robot XY uniquely identifies the source in
    the original benchmark and is the stable provenance key used here.
    """
    expected_xy = repair.get("source_robot_xy")
    house_index = int(repair["house_index"])
    matches: list[tuple[int, dict[str, Any], float]] = []
    if isinstance(expected_xy, list) and len(expected_xy) >= 2:
        for index, episode in enumerate(source_episodes):
            if int(episode.get("house_index", -1)) != house_index:
                continue
            xy = _source_xy(episode)
            if xy is None:
                continue
            distance = ((xy[0] - float(expected_xy[0])) ** 2 + (xy[1] - float(expected_xy[1])) ** 2) ** 0.5
            if distance <= 1e-3:
                matches.append((index, episode, distance))
    if len(matches) != 1:
        # A useful fallback for old manifests that recorded the source list
        # index rather than the builder parent index.
        fallback_index = int(repair.get("source_nav_benchmark_index", -1))
        if 0 <= fallback_index < len(source_episodes):
            candidate = source_episodes[fallback_index]
            if int(candidate.get("house_index", -1)) == house_index:
                return fallback_index, candidate
        raise ValueError(
            f"Could not uniquely resolve source for case={repair.get('case_id')}: "
            f"house={house_index} xy={expected_xy} matches={len(matches)}"
        )
    index, episode, _distance = matches[0]
    return index, episode


def _generated_samples(output_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    benchmark_path = output_dir / "benchmark.json"
    if benchmark_path.exists():
        samples.extend(load_episodes(benchmark_path))
    for path in sorted((output_dir / "samples").glob("*/sample.json")):
        try:
            samples.append(read_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    unique: dict[str, dict[str, Any]] = {}
    for sample in samples:
        identifier = case_id(sample)
        if identifier:
            unique[identifier] = sample
    return list(unique.values())


def _validate_rebuilt_sample(sample: dict[str, Any], repair: dict[str, Any]) -> None:
    expected_case = str(repair["case_id"])
    expected_target = str(repair["new_selected_target"])
    actual_case = case_id(sample)
    if actual_case != expected_case:
        raise ValueError(f"case id mismatch: expected={expected_case} actual={actual_case}")
    if target_name(sample) != expected_target:
        raise ValueError(
            f"target mismatch for {expected_case}: expected={expected_target} actual={target_name(sample)}"
        )
    task = sample.get("task", {})
    target = sample.get("interactive_nav", {}).get("target", {})
    success = (
        sample.get("interactive_nav", {})
        .get("generation_validation", {})
        .get("success_evidence", {})
    )
    old_target = str(repair["old_selected_target"])
    fixed_references = [
        task.get("pickup_obj_name"),
        *(task.get("pickup_obj_candidates") or []),
        target.get("selected_instance"),
        *(target.get("instruction_consistent_candidates") or []),
        success.get("target_object_name"),
        *(sample.get("task_relevant_objects") or [])[:1],
    ]
    if old_target in fixed_references:
        raise ValueError(f"old target remains in a fixed target reference: {expected_case}")
    validation = (
        sample.get("interactive_nav", {})
        .get("generation_validation", {})
        .get("target_identity_validation")
    )
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError(f"target identity evidence did not pass for {expected_case}: {validation!r}")
    if validation.get("target_object_name") != expected_target:
        raise ValueError(f"target evidence mismatch for {expected_case}: {validation!r}")
    oracle_steps = sample.get("interactive_nav", {}).get("oracle_plan", {}).get("steps", [])
    observe_steps = [step for step in oracle_steps if step.get("type") == "observe_target"]
    if len(observe_steps) != 1 or observe_steps[0].get("object_name") != expected_target:
        raise ValueError(f"oracle target evidence mismatch for {expected_case}")


def _group_repairs(
    repairs: list[dict[str, Any]], source_episodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for repair in repairs:
        source_index, source_episode = resolve_source_episode(repair, source_episodes)
        parent_index = int(repair["source_parent_episode_index"])
        key = (source_index, parent_index)
        group = grouped.setdefault(
            key,
            {
                "group_id": f"source_{source_index:05d}_parent_{parent_index:05d}",
                "source_index": source_index,
                "parent_index": parent_index,
                "source_episode": source_episode,
                "repairs": [],
            },
        )
        group["repairs"].append(repair)
    return sorted(grouped.values(), key=lambda value: value["group_id"])


def _run_group(
    group: dict[str, Any],
    *,
    output_root: Path,
    variant: str,
    seed: int,
    sampling_seed: int,
    target_goal_tolerance_m: float,
    mujoco_gl: str,
) -> dict[str, Any]:
    group_base = output_root / "runs" / group["group_id"]
    group_root = group_base
    if group_base.exists() and any(group_base.iterdir()):
        attempt_index = 1
        while (group_base / f"attempt_{attempt_index:03d}").exists():
            attempt_index += 1
        group_root = group_base / f"attempt_{attempt_index:03d}"
    input_root = group_root / "input"
    builder_output = group_root / "output"
    write_json(input_root / "benchmark.json", [group["source_episode"]])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MUJOCO_GL"] = mujoco_gl
    env["MPLCONFIGDIR"] = str(group_root / "matplotlib-cache")
    env["INTERACTIVE_NAV_SCENE_MIRROR"] = str(group_root / "scene_assets")
    command = [
        PYTHON,
        "scripts/InteractiveNav/build_door_interaction_benchmark.py",
        "--benchmark_dir",
        str(input_root),
        "--output_dir",
        str(builder_output),
        "--mode",
        "build",
        "--input_mode",
        "original",
        "--variant",
        variant,
        "--robot",
        "rby1",
        "--max_episodes",
        "1",
        "--episode_index_offset",
        str(group["parent_index"]),
        "--seed",
        str(seed),
        "--sampling_seed",
        str(sampling_seed),
        "--target_goal_tolerance_m",
        str(target_goal_tolerance_m),
        "--target_object_override",
        str(group["repairs"][0]["new_selected_target"]),
        "--num_distractor_samples_per_episode",
        "1",
        "--num_mixed_samples_per_critical_door",
        "1",
        "--distractor_k_min",
        "1",
        "--distractor_k_max",
        "2",
    ]
    log_path = group_root / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        return {
            "group_id": group["group_id"],
            "source_index": group["source_index"],
            "parent_index": group["parent_index"],
            "returncode": int(completed.returncode),
            "status": "failed",
            "error": f"builder_returncode_{completed.returncode}",
            "log_path": str(log_path),
            "elapsed_sec": time.perf_counter() - started,
        }
    try:
        generated = {case_id(sample): sample for sample in _generated_samples(builder_output)}
        replacements: dict[str, str] = {}
        for repair in group["repairs"]:
            identifier = str(repair["case_id"])
            sample = generated.get(identifier)
            if sample is None:
                raise ValueError(
                    f"builder did not produce requested case {identifier}; "
                    f"available={sorted(generated)[:10]}"
                )
            _validate_rebuilt_sample(sample, repair)
            destination = group_root / "repaired" / identifier / "sample.json"
            write_json(destination, sample)
            replacements[identifier] = str(destination)
        result = {
            "group_id": group["group_id"],
            "source_index": group["source_index"],
            "parent_index": group["parent_index"],
            "returncode": int(completed.returncode),
            "status": "passed",
            "replacement_paths": replacements,
            "log_path": str(log_path),
            "elapsed_sec": time.perf_counter() - started,
        }
        write_json(group_root / "result.json", result)
        return result
    except Exception as exc:
        result = {
            "group_id": group["group_id"],
            "source_index": group["source_index"],
            "parent_index": group["parent_index"],
            "returncode": int(completed.returncode),
            "status": "failed",
            "error": str(exc),
            "log_path": str(log_path),
            "elapsed_sec": time.perf_counter() - started,
        }
        write_json(group_root / "result.json", result)
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-channel", type=Path, required=True)
    parser.add_argument("--source-benchmark", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variant", choices=["base", "ceiling"], default="base")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--sampling-seed", type=int, default=20260708)
    parser.add_argument("--target-goal-tolerance-m", type=float, default=0.25)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    current_path = args.current_channel.resolve()
    source_path = args.source_benchmark.resolve()
    repair_manifest_path = args.repair_manifest.resolve()
    output_root = args.output_root.resolve()
    current = load_episodes(current_path)
    source = load_episodes(source_path)
    repair_manifest = read_json(repair_manifest_path)
    repairs = repair_manifest.get("repairs")
    if not isinstance(repairs, list) or not repairs:
        raise ValueError("repair manifest has no repairs")
    current_by_case = {case_id(episode): episode for episode in current}
    if len(current_by_case) != len(current):
        raise ValueError("current Channel file contains duplicate or missing case IDs")
    for repair in repairs:
        identifier = str(repair["case_id"])
        if identifier not in current_by_case:
            raise ValueError(f"repair case is absent from current Channel file: {identifier}")
        if target_name(current_by_case[identifier]) != str(repair["new_selected_target"]):
            raise ValueError(
                f"current Channel target does not match repair manifest for {identifier}"
            )
    groups = _group_repairs(repairs, source)
    if args.limit_groups is not None:
        if args.limit_groups < 1:
            raise ValueError("--limit-groups must be positive")
        groups = groups[: args.limit_groups]
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    state = read_json(state_path) if args.resume and state_path.exists() else {
        "schema_version": "interactive_nav_v3_channel_rebuild_state_v1",
        "groups": {},
    }
    completed_groups: dict[str, dict[str, Any]] = state.setdefault("groups", {})
    if args.dry_run:
        print(
            json.dumps(
                {
                    "current_count": len(current),
                    "repair_count": len(repairs),
                    "source_count": len(source),
                    "group_count": len(groups),
                    "group_sizes": {group["group_id"]: len(group["repairs"]) for group in groups},
                    "workers": args.workers,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    pending = [
        group
        for group in groups
        if not (
            args.resume
            and completed_groups.get(group["group_id"], {}).get("status") == "passed"
        )
    ]
    started = time.perf_counter()
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as executor:
        for group in pending:
            futures[
                executor.submit(
                    _run_group,
                    group,
                    output_root=output_root,
                    variant=args.variant,
                    seed=args.seed,
                    sampling_seed=args.sampling_seed,
                    target_goal_tolerance_m=args.target_goal_tolerance_m,
                    mujoco_gl=args.mujoco_gl,
                )
            ] = group
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                group = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - worker wrapper
                    result = {
                        "group_id": group["group_id"],
                        "status": "failed",
                        "error": repr(exc),
                    }
                completed_groups[group["group_id"]] = result
                state["groups"] = completed_groups
                write_json(state_path, state)
                passed = sum(value.get("status") == "passed" for value in completed_groups.values())
                failed = sum(value.get("status") == "failed" for value in completed_groups.values())
                print(
                    f"[channel-rebuild-progress] {passed + failed}/{len(groups)} "
                    f"passed={passed} failed={failed} group={group['group_id']} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    failed_groups = [value for value in completed_groups.values() if value.get("status") != "passed"]
    if failed_groups:
        write_json(
            output_root / "failure_report.json",
            {"schema_version": "interactive_nav_v3_channel_rebuild_failures_v1", "groups": failed_groups},
        )
        raise RuntimeError(f"Channel rebuild failed for {len(failed_groups)} source groups")

    replacements: dict[str, dict[str, Any]] = {}
    for group in groups:
        result = completed_groups[group["group_id"]]
        for identifier, path in result.get("replacement_paths", {}).items():
            replacements[identifier] = read_json(Path(path))
    if set(replacements) != {str(repair["case_id"]) for repair in repairs[: len(groups) if args.limit_groups else len(repairs)]}:
        if args.limit_groups:
            # A limited smoke run must not materialize a misleading partial full
            # benchmark; its per-group artifacts and state are the output.
            write_json(
                output_root / "smoke_summary.json",
                {
                    "schema_version": "interactive_nav_v3_channel_rebuild_smoke_v1",
                    "group_count": len(groups),
                    "replacement_count": len(replacements),
                    "elapsed_sec": time.perf_counter() - started,
                },
            )
            return
        raise RuntimeError("Replacement set does not cover the repair manifest")

    merged: list[dict[str, Any]] = []
    changed = 0
    for episode in current:
        identifier = case_id(episode)
        replacement = replacements.get(identifier)
        if replacement is not None:
            merged.append(replacement)
            changed += 1
        else:
            merged.append(copy.deepcopy(episode))
    if len(merged) != len(current) or len({case_id(episode) for episode in merged}) != len(merged):
        raise RuntimeError("Merged Channel benchmark is not a unique one-to-one episode list")
    unchanged = [episode for episode in current if case_id(episode) not in replacements]
    unchanged_merged = [episode for episode in merged if case_id(episode) not in replacements]
    if json_sha256(unchanged) != json_sha256(unchanged_merged):
        raise RuntimeError("Unrepaired Channel entries changed during merge")
    write_json(output_root / "benchmark.json", merged)
    write_json(output_root / "channel.json", merged)
    report = {
        "schema_version": "interactive_nav_v3_channel_target_rebuild_v1",
        "source_current_channel": str(current_path),
        "source_current_channel_sha256": sha256(current_path),
        "source_benchmark": str(source_path),
        "source_benchmark_sha256": sha256(source_path),
        "source_repair_manifest": str(repair_manifest_path),
        "source_repair_manifest_sha256": sha256(repair_manifest_path),
        "summary": {
            "input_episode_count": len(current),
            "repaired_episode_count": changed,
            "unchanged_episode_count": len(current) - changed,
            "source_group_count": len(groups),
            "worker_count": args.workers,
            "paths_oracle_interactions_rebuilt": True,
            "target_identity_validation_required": True,
        },
        "groups": completed_groups,
        "output_benchmark_sha256": json_sha256(merged),
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json(output_root / "repair_manifest.json", report)
    write_json(
        output_root / "validation_report.json",
        {
            "schema_version": "interactive_nav_v3_channel_target_rebuild_validation_v1",
            "episode_count": len(merged),
            "repaired_count": changed,
            "all_repaired_target_evidence_passed": all(
                episode.get("interactive_nav", {})
                .get("generation_validation", {})
                .get("target_identity_validation", {})
                .get("passed")
                is True
                for episode in merged
                if case_id(episode) in replacements
            ),
            "unchanged_json_sha256": json_sha256(unchanged),
        },
    )
    print(json.dumps(report["summary"] | {"output_root": str(output_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
