"""Run independent initialization-only probes, with bounded concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    indices = args.episode_indices if args.episode_indices is not None else list(range(args.episodes))
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        parser.error("episode indices must be unique and nonnegative")
    args.episodes = len(indices)
    if min(args.workers, args.episodes, args.timeout) <= 0:
        parser.error("workers, episodes and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    environment = dict(os.environ, TMPDIR=str(args.output / "tmp"),
                       XDG_CACHE_HOME="/home/ldl/.cache", HF_HOME="/home/ldl/.cache/huggingface",
                       TORCH_HOME="/home/ldl/.cache/torch", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                       MLSPACES_ASSETS_DIR="/home/ldl/molmospaces/assets",
                       MLSPACES_CACHE_DIR="/home/ldl/molmo-spaces-resources",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    Path(environment["TMPDIR"]).mkdir()
    gpu_ids = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    if not gpu_ids:
        raise RuntimeError("No renderer GPU available")
    (args.output / "launch.json").write_text(json.dumps({
        "pid": os.getpid(), "workers": args.workers, "episodes": args.episodes,
        "timeout_seconds": args.timeout, "started_at": time.time(), "gpu_ids": gpu_ids,
        "scope": "sample_task and reset only; no policy, ROS or Qwen",
        "python": sys.executable, "episode_indices": indices}, indent=2))

    def run(index):
        directory = args.output / f"episode_{index:04d}"
        started = time.monotonic()
        env = dict(environment, MUJOCO_EGL_DEVICE_ID=gpu_ids[index % len(gpu_ids)])
        with (args.output / f"episode_{index:04d}.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, str(repo / "scripts/InteractiveNav/profile_scene_initialization.py"),
                                        "--episode", str(index), "--output", str(directory)],
                                       cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            reason = None
            while process.poll() is None:
                if stop.wait(1) or time.monotonic() - started > args.timeout:
                    reason = "cancelled" if stop.is_set() else "timeout"
                    for sig, grace in ((signal.SIGINT, 10), (signal.SIGKILL, 10)):
                        if process.poll() is not None:
                            break
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            break
                        try:
                            process.wait(timeout=grace)
                        except subprocess.TimeoutExpired:
                            pass
                    break
        row = {"episode": index, "exit_code": process.poll(), "reason": reason,
               "wall_seconds": time.monotonic() - started}
        for name in ("timings", "state_audit"):
            try:
                row[name] = json.loads((directory / f"{name}.json").read_text())
            except (OSError, ValueError):
                row[name] = {}
        row["passed"] = row["exit_code"] == 0 and row["state_audit"].get("passed") is True
        return row

    rows = []
    next_index = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.output / "results.jsonl").open("a") as results:
        active = set()
        while active or (next_index < args.episodes and not stop.is_set()):
            while len(active) < args.workers and next_index < args.episodes and not stop.is_set():
                active.add(pool.submit(run, indices[next_index]))
                next_index += 1
            done, active = wait(active, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                results.write(json.dumps(row) + "\n")
                results.flush()
            status = {"finished": len(rows), "passed": sum(r["passed"] for r in rows),
                      "failed": sum(not r["passed"] for r in rows), "active": len(active),
                      "queued": args.episodes - next_index, "updated_at": time.time()}
            memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            status["mem_available"] = memory["MemAvailable"].strip()
            status["loadavg"] = os.getloadavg()
            (args.output / "status.json").write_text(json.dumps(status, indent=2))
            with (args.output / "resources.jsonl").open("a") as resources:
                resources.write(json.dumps(status) + "\n")
            print(json.dumps(status), flush=True)
    (args.output / "finished.json").write_text(json.dumps({"finished": len(rows), "cancelled": stop.is_set(),
        "passed": sum(r["passed"] for r in rows), "ended_at": time.time()}, indent=2))
    return 0 if len(rows) == args.episodes and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
