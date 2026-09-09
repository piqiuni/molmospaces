from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav.select_container_interaction_candidates import (
    build_dynamic_collection_plan,
)


def load_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def split_houses(houses: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(min(workers, len(houses)))]
    for index, house in enumerate(houses):
        shards[index % len(shards)].append(house)
    return shards


def run_shard(
    index: int,
    houses: list[dict[str, Any]],
    plan: dict[str, Any],
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    shard_dir = args.output_dir / "shards" / f"shard_{index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_plan = {
        **plan,
        "houses": houses,
        "selection": {
            **plan.get("selection", {}),
            "selected_house_count": len(houses),
            "requested_sample_count": sum(
                int(row.get("target_sample_count", len(row.get("slots", []))))
                for row in houses
            ),
        },
    }
    plan_path = shard_dir / "collection_plan.json"
    write_json(plan_path, shard_plan)
    command = [
        sys.executable,
        "scripts/InteractiveNav/build_container_interaction_benchmark.py",
        "--benchmark_dir",
        str(args.benchmark_dir),
        "--candidate_file",
        str(plan_path),
        "--output_dir",
        str(shard_dir / "benchmark"),
        "--seed",
        str(args.seed),
        "--save_images" if args.save_images else "--no-save_images",
        "--save_plots" if args.save_plots else "--no-save_plots",
    ]
    log_path = shard_dir / "run.log"
    started_at = time.time()
    shard_env = env.copy()
    shard_env["MPLCONFIGDIR"] = str(shard_dir / "matplotlib-cache")
    with open(log_path, "w") as log_handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=shard_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary_path = shard_dir / "benchmark" / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {
        "shard_index": index,
        "candidate_house_count": len(houses),
        "requested_sample_count": sum(
            int(row.get("target_sample_count", len(row.get("slots", []))))
            for row in houses
        ),
        "completed_house_count": int(summary.get("complete_collection_house_count", 0)),
        "collection_house_count": int(summary.get("collection_house_count", 0)),
        "generated_episode_count": int(summary.get("generated_episode_count", 0)),
        "returncode": result.returncode,
        "elapsed_sec": time.time() - started_at,
        "output_dir": str(shard_dir),
        "log_path": str(log_path),
    }


def merge_results(
    args: argparse.Namespace,
    plan: dict[str, Any],
    results: list[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    benchmark = []
    valid_pairs = []
    rejected_pairs = []
    house_catalog = []
    fridge_slides = []
    failures = []
    for result in sorted(results, key=lambda row: row["shard_index"]):
        directory = Path(result["output_dir"]) / "benchmark"
        for filename, destination in (
            ("benchmark.json", benchmark),
            ("valid_pairs.json", valid_pairs),
            ("rejected_pairs.json", rejected_pairs),
            ("house_catalog.json", house_catalog),
            ("fridge_slide_compartment_candidates.json", fridge_slides),
            ("failures.json", failures),
        ):
            path = directory / filename
            if path.exists():
                destination.extend(load_json(path))
        if result.get("returncode", 0) != 0:
            failures.append(
                {
                    "shard_index": result["shard_index"],
                    "reason": "shard_process_failed",
                    "returncode": result["returncode"],
                    "log_path": result.get("log_path"),
                }
            )

    collected_houses = sorted({int(row["house_index"]) for row in valid_pairs})
    requested_by_house = {
        int(row["house_index"]): int(
            row.get("target_sample_count", len(row.get("slots", [])))
        )
        for row in plan.get("houses", [])
    }
    collected_counts = {
        house_index: sum(
            int(row["house_index"]) == house_index for row in valid_pairs
        )
        for house_index in requested_by_house
    }
    complete_houses = sorted(
        house_index
        for house_index, requested in requested_by_house.items()
        if collected_counts[house_index] >= requested
    )
    valid_pairs.sort(key=lambda row: (int(row["house_index"]), row["case_id"]))
    invalid_versions = [
        episode.get("interactive_nav", {}).get("schema_version")
        for episode in benchmark
        if episode.get("interactive_nav", {}).get("schema_version")
        != "interactive_nav_v3"
    ]
    if invalid_versions:
        raise ValueError(
            "Fine-collection shards must contain only interactive_nav_v3 episodes; "
            f"found {invalid_versions[:5]}"
        )
    benchmark_by_case = {
        episode["interactive_nav"]["case_id"]: episode for episode in benchmark
    }
    benchmark = [benchmark_by_case[row["case_id"]] for row in valid_pairs]
    collected_plan = {
        **plan,
        "houses": [
            {
                **row,
                "collected_sample_count": collected_counts[int(row["house_index"])],
                "collection_quota_complete": (
                    collected_counts[int(row["house_index"])]
                    >= requested_by_house[int(row["house_index"])]
                ),
            }
            for row in plan.get("houses", [])
        ],
    }
    summary = {
        "schema_version": "container_interaction_parallel_collection_summary_v1",
        "collection_mode": "fixed",
        "selected_house_count": len(requested_by_house),
        "requested_sample_count": sum(requested_by_house.values()),
        "complete_collection_house_count": len(complete_houses),
        "collection_house_count": len(collected_houses),
        "partial_collection_house_count": sum(
            0 < collected_counts[house_index] < requested
            for house_index, requested in requested_by_house.items()
        ),
        "zero_sample_house_count": sum(
            collected_counts[house_index] == 0 for house_index in requested_by_house
        ),
        "generated_episode_count": len(benchmark),
        "valid_pair_count": len(valid_pairs),
        "rejected_attempt_count": len(rejected_pairs),
        "failure_count": len(failures),
        "elapsed_sec": elapsed_sec,
        "worker_count": args.workers,
        "shards": sorted(results, key=lambda row: row["shard_index"]),
    }
    write_json(args.output_dir / "benchmark.json", benchmark)
    write_json(args.output_dir / "valid_pairs.json", valid_pairs)
    write_json(args.output_dir / "rejected_pairs.json", rejected_pairs)
    write_json(args.output_dir / "house_catalog.json", house_catalog)
    write_json(
        args.output_dir / "fridge_slide_compartment_candidates.json", fridge_slides
    )
    write_json(args.output_dir / "failures.json", failures)
    write_json(args.output_dir / "collection_plan.json", collected_plan)
    write_json(args.output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect fixed-house container interactions directly from a rough catalog."
    )
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rough_catalog", type=Path)
    source.add_argument("--candidate_manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--samples_per_house", type=int, default=2)
    parser.add_argument("--target_house_count", type=int)
    parser.add_argument("--house_indices", help="Comma-separated fixed house indices.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mujoco_gl", default="egl")
    parser.add_argument(
        "--save_images", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save_plots", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rough_catalog is not None:
        explicit_houses = (
            [int(value) for value in args.house_indices.split(",")]
            if args.house_indices
            else None
        )
        plan = build_dynamic_collection_plan(
            args.rough_catalog,
            max_samples=args.max_samples,
            samples_per_house=args.samples_per_house,
            target_house_count=args.target_house_count,
            house_indices=explicit_houses,
            seed=args.seed,
        )
    else:
        plan = load_json(args.candidate_manifest)
    shards = split_houses(plan.get("houses", []), max(1, args.workers))
    if not shards:
        raise ValueError("No eligible houses were found in the collection source")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MUJOCO_GL"] = args.mujoco_gl
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-container-fine-parallel")
    started_at = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = {
            executor.submit(run_shard, index, shard, plan, args, env): index
            for index, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    summary = merge_results(args, plan, results, time.time() - started_at)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
