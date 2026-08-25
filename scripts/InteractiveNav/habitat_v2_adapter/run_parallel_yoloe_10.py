#!/usr/bin/env python3
"""Run the deterministic first-10 ObjectNav-v2 scenes as ten worker processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


SCENES = (
    "00800-TEEsavR23oF",
    "00802-wcojb4TFT35",
    "00803-k1cupFYWXJ6",
    "00808-y9hTuugGdiq",
    "00810-CrMo8WxCyVb",
    "00813-svBbv1Pavdk",
    "00814-p53SfW6mjZe",
    "00815-h1zeeAwLh9Z",
    "00820-mL8ThkuaVTM",
    "00821-eF36g7L6Z9M",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ldl/outputs/habitat_objectnav_v2_m2/parallel10_ros_yoloe"),
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-episode-seconds", type=int, default=500)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    adapter_dir = Path(__file__).resolve().parent
    interactive_dir = adapter_dir.parent
    evaluator = adapter_dir / "evaluate.py"
    adapter_config = (
        interactive_dir
        / "configs/habitat_objectnav_v2/module1_ros_yoloe_m2_navigation_only.yaml"
    )
    habitat_lab = Path("/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab")
    python = Path("/home/ldl/conda_envs/habitat-challenge-2023/bin/python")
    run_stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_root = args.output_root / run_stamp
    suffix = 1
    while run_root.exists():
        run_root = args.output_root / f"{run_stamp}-{suffix}"
        suffix += 1
    worker_root = run_root / "workers"
    worker_root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "workers": len(SCENES),
        "scenes": list(SCENES),
        "max_steps": args.max_steps,
        "max_episode_seconds": args.max_episode_seconds,
        "gpu_id": args.gpu_id,
        "adapter_config": str(adapter_config),
        "module3_enabled": False,
    }
    (run_root / "parallel_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    processes: dict[str, tuple[subprocess.Popen[str], object]] = {}
    for scene in SCENES:
        worker_dir = worker_root / scene
        worker_dir.mkdir(parents=True, exist_ok=False)
        cache_key = scene.replace("-", "_")
        task_tmp = Path("/home/ldl/tmp/habitat-objectnav-parallel10") / run_stamp / cache_key
        task_cache = Path("/home/ldl/.cache/habitat-objectnav-parallel10") / run_stamp / cache_key
        task_tmp.mkdir(parents=True, exist_ok=True)
        task_cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "EGL_PLATFORM": "surfaceless",
                "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
                "TMPDIR": str(task_tmp),
                "XDG_CACHE_HOME": str(task_cache),
                "PYTHONPYCACHEPREFIX": str(task_cache / "pycache"),
                "PYTHONPATH": f"{interactive_dir}:{habitat_lab}",
            }
        )
        command = [
            str(python),
            str(evaluator),
            "--adapter-config",
            str(adapter_config),
            "--output-dir",
            str(worker_dir),
            "--scene-id",
            scene,
            "--scene-count",
            "1",
            "--episodes-per-scene",
            "1",
            "--max-steps",
            str(args.max_steps),
            "--max-episode-seconds",
            str(args.max_episode_seconds),
            "--gpu-id",
            "0",
            "--public-trace",
            "--posthoc-step-metrics",
            "--posthoc-topdown-map",
            "--allow-no-mllm-success",
        ]
        log_stream = (worker_dir / "worker.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(Path("/home/ldl/molmospaces-exp-setting")),
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        processes[scene] = (process, log_stream)
        print(f"started {scene}: pid={process.pid}", flush=True)

    exit_codes: dict[str, int] = {}
    try:
        while len(exit_codes) < len(processes):
            for scene, (process, _) in processes.items():
                if scene in exit_codes:
                    continue
                code = process.poll()
                if code is not None:
                    exit_codes[scene] = code
                    print(f"finished {scene}: exit={code}", flush=True)
            if len(exit_codes) < len(processes):
                time.sleep(2.0)
    except KeyboardInterrupt:
        for process, _ in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        raise
    finally:
        for _, log_stream in processes.values():
            log_stream.close()

    (run_root / "worker_exit_codes.json").write_text(
        json.dumps(exit_codes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    from aggregate_parallel_results import aggregate

    summary = aggregate(run_root)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
