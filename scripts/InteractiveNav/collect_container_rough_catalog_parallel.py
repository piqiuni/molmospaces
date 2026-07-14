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
DEFAULT_BENCHMARK_DIR = Path(
    "/home/user/ldl/molmospaces/assets/benchmarks/molmospaces-bench-v2/"
    "procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "scripts/InteractiveNav/output/container_interaction_all_houses/rough_catalog"
)


def load_episodes(benchmark_dir: Path) -> list[dict[str, Any]]:
    path = benchmark_dir / "benchmark.json" if benchmark_dir.is_dir() else benchmark_dir
    with open(path) as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(episodes, dict):
        episodes = list(episodes.values())
    return list(episodes)


def unique_houses(episodes: list[dict[str, Any]], max_houses: int | None) -> list[int]:
    houses = []
    seen = set()
    for episode in episodes:
        house_index = int(episode["house_index"])
        if house_index in seen:
            continue
        seen.add(house_index)
        houses.append(house_index)
        if max_houses is not None and len(houses) >= max_houses:
            break
    return houses


def balanced_shards(houses: list[int], workers: int) -> list[list[int]]:
    shards = [[] for _ in range(min(workers, len(houses)))]
    for index, house in enumerate(houses):
        shards[index % len(shards)].append(house)
    return [shard for shard in shards if shard]


def run_shard(
    shard_index: int,
    houses: list[int],
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    shard_dir = args.output_dir / "shards" / f"shard_{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/InteractiveNav/build_container_interaction_benchmark.py",
        "--benchmark_dir",
        str(args.benchmark_dir),
        "--output_dir",
        str(shard_dir),
        "--house_indices",
        ",".join(map(str, houses)),
        "--catalog_only",
        "--no-save_images",
        "--no-save_plots",
    ]
    log_path = shard_dir / "run.log"
    started_at = time.time()
    with open(log_path, "w") as log_handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "shard_index": shard_index,
        "houses": houses,
        "output_dir": str(shard_dir),
        "log_path": str(log_path),
        "returncode": result.returncode,
        "elapsed_sec": time.time() - started_at,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def merge_shards(
    args: argparse.Namespace,
    requested_houses: list[int],
    shard_results: list[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    houses = []
    fridge_slides = []
    failures = []
    for result in sorted(shard_results, key=lambda row: row["shard_index"]):
        shard_dir = Path(result["output_dir"])
        house_path = shard_dir / "house_catalog.json"
        slide_path = shard_dir / "fridge_slide_compartment_candidates.json"
        failure_path = shard_dir / "failures.json"
        if house_path.exists():
            houses.extend(json.load(open(house_path)))
        if slide_path.exists():
            fridge_slides.extend(json.load(open(slide_path)))
        if failure_path.exists():
            failures.extend(json.load(open(failure_path)))
        if result["returncode"] != 0 and not failure_path.exists():
            failures.append(
                {
                    "shard_index": result["shard_index"],
                    "houses": result["houses"],
                    "reason": "shard_process_failed",
                    "returncode": result["returncode"],
                    "log_path": result["log_path"],
                }
            )

    order = {house_index: index for index, house_index in enumerate(requested_houses)}
    houses.sort(key=lambda row: order.get(int(row["house_index"]), len(order)))
    completed = {int(row["house_index"]) for row in houses}
    failed_houses = {int(row["house_index"]) for row in failures if "house_index" in row}
    missing_houses = [
        house for house in requested_houses if house not in completed and house not in failed_houses
    ]
    summary = {
        "schema_version": "container_rough_catalog_summary_v1",
        "benchmark_dir": str(args.benchmark_dir),
        "requested_house_count": len(requested_houses),
        "completed_house_count": len(completed),
        "failure_count": len(failures),
        "missing_house_count": len(missing_houses),
        "worker_count": args.workers,
        "elapsed_sec": elapsed_sec,
        "strict_pair_count": sum(int(row["strict_pair_count"]) for row in houses),
        "fridge_count": sum(int(row["num_fridges"]) for row in houses),
        "dresser_count": sum(int(row["num_dressers"]) for row in houses),
        "fridges_with_objects": sum(
            int(row["num_fridges_with_objects"]) for row in houses
        ),
        "dressers_with_objects": sum(
            int(row["num_dressers_with_objects"]) for row in houses
        ),
        "fridge_slide_joint_count": len(fridge_slides),
        "fridge_slide_joints_with_objects": sum(
            bool(row["contained_objects"]) for row in fridge_slides
        ),
        "fridge_slide_object_count": sum(
            len(row["contained_objects"]) for row in fridge_slides
        ),
        "missing_houses": missing_houses,
    }
    payload = {
        "schema_version": "container_rough_catalog_v1",
        "summary": summary,
        "houses": houses,
        "fridge_slide_compartment_candidates": fridge_slides,
        "failures": failures,
        "shards": sorted(shard_results, key=lambda row: row["shard_index"]),
    }
    write_json(args.output_dir / "rough_catalog.json", payload)
    write_json(args.output_dir / "house_catalog.json", houses)
    write_json(args.output_dir / "summary.json", summary)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan unique houses in parallel and merge one rough catalog."
    )
    parser.add_argument("--benchmark_dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_houses", type=int)
    parser.add_argument("--mujoco_gl", default="egl")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    episodes = load_episodes(args.benchmark_dir)
    houses = unique_houses(episodes, args.max_houses)
    shards = balanced_shards(houses, max(1, args.workers))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MUJOCO_GL"] = args.mujoco_gl
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-container-rough-catalog")

    started_at = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = {
            executor.submit(run_shard, index, shard, args, env): index
            for index, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"shard {result['shard_index']} finished: "
                f"returncode={result['returncode']} elapsed={result['elapsed_sec']:.1f}s",
                flush=True,
            )

    payload = merge_shards(args, houses, results, time.time() - started_at)
    print(json.dumps(payload["summary"], indent=2))
    return 0 if not payload["failures"] and not payload["summary"]["missing_houses"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
