from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = Path(
    "/home/user/ldl/molmospaces/assets/benchmarks/molmospaces-bench-v2/"
    "procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "scripts/InteractiveNav/output/container_interaction_all_houses"


def run_stage(name: str, command: list[str], env: dict[str, str]) -> dict[str, object]:
    print(f"[{name}] {' '.join(command)}", flush=True)
    started_at = time.time()
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    elapsed_sec = time.time() - started_at
    print(f"[{name}] completed in {elapsed_sec:.1f}s", flush=True)
    return {"stage": name, "elapsed_sec": elapsed_sec, "command": command}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rough scan and fixed-house fine collection."
    )
    parser.add_argument("--benchmark_dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--samples_per_house", type=int, default=2)
    parser.add_argument("--target_house_count", type=int)
    parser.add_argument("--rough_workers", type=int, default=4)
    parser.add_argument("--fine_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_rough", action="store_true")
    parser.add_argument("--skip_collect", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rough_dir = args.output_root / "rough_catalog"
    benchmark_dir = args.output_root / "benchmark"
    args.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-container-collection")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    python = sys.executable
    stages = []

    if not args.skip_rough:
        stages.append(
            run_stage(
                "rough_catalog",
                [
                    python,
                    "scripts/InteractiveNav/collect_container_rough_catalog_parallel.py",
                    "--benchmark_dir",
                    str(args.benchmark_dir),
                    "--output_dir",
                    str(rough_dir),
                    "--workers",
                    str(args.rough_workers),
                ],
                env,
            )
        )

    if not args.skip_collect:
        fine_script = (
            "scripts/InteractiveNav/collect_container_fine_parallel.py"
            if args.fine_workers > 1
            else "scripts/InteractiveNav/build_container_interaction_benchmark.py"
        )
        collection_command = [
            python,
            fine_script,
            "--benchmark_dir",
            str(args.benchmark_dir),
            "--output_dir",
            str(benchmark_dir),
            "--rough_catalog",
            str(rough_dir / "rough_catalog.json"),
            "--max_samples",
            str(args.max_samples),
            "--samples_per_house",
            str(args.samples_per_house),
            "--seed",
            str(args.seed),
            "--save_images" if args.save_images else "--no-save_images",
            "--save_plots" if args.save_plots else "--no-save_plots",
        ]
        if args.fine_workers > 1:
            collection_command.extend(
                [
                    "--workers",
                    str(args.fine_workers),
                ]
            )
        if args.target_house_count is not None:
            collection_command.extend(
                ["--target_house_count", str(args.target_house_count)]
            )
        stages.append(run_stage("fine_collection", collection_command, env))

    with open(args.output_root / "serial_collection_status.json", "w") as handle:
        json.dump(
            {
                "benchmark_dir": str(args.benchmark_dir),
                "rough_catalog": str(rough_dir / "rough_catalog.json"),
                "benchmark_output": str(benchmark_dir),
                "stages": stages,
            },
            handle,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
