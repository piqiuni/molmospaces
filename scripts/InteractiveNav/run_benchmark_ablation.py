#!/usr/bin/env python3
"""Run Full or one module ablation with the existing V3 evaluator and budget."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

import run_benchmark_eval as baseline
from ablations import BASELINE_COMMIT, DESIGN_REVISION, VARIANTS
from ablations.perception import REFRESH_PROFILES
from ablations.launch import artifact_digests, render_artifacts, runner_path, write_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--m1-refresh-profile", choices=REFRESH_PROFILES,
                        help="shared M1 schedule; use the same profile for Full and all ablations")
    parser.add_argument("--config", type=Path, default=baseline.DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--base-master-port", type=int)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--recording", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate and print; no files, ROS or model calls")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    refresh_profile = args.m1_refresh_profile or config.get("m1_refresh_profile", "baseline")
    if refresh_profile not in REFRESH_PROFILES:
        parser.error("invalid m1_refresh_profile in config")
    config["m1_refresh_profile"] = refresh_profile
    for key in ("workers", "max_steps", "base_master_port", "episode_indices", "recording"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    name = datetime.datetime.now().strftime(f"ablation-{args.variant}-%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or Path(config["output_root"]) / name).expanduser().resolve()
    if not output.is_relative_to(Path("/home/ldl")):
        parser.error("outputs and runtime caches must be under /home/ldl")
    command, indices = baseline.build_command(config, output)
    directory = output / "ablation"
    artifacts = render_artifacts(baseline.REPO, directory, args.variant, config["python_bin"], refresh_profile)
    if artifacts:
        command += ["--runner", str(runner_path(baseline.REPO, directory, args.variant, refresh_profile))]
    manifest = {"variant": args.variant, "design_revision": DESIGN_REVISION,
                "m1_refresh_profile": refresh_profile, "m1_refresh_overrides": REFRESH_PROFILES[refresh_profile],
                "reference_baseline_commit": BASELINE_COMMIT,
                "output_dir": str(output), "command": command,
                "artifact_sha256": artifact_digests(artifacts), "config": config}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    for port in range(config["base_master_port"], config["base_master_port"] + config["workers"]):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    output.mkdir(parents=True, exist_ok=False)
    write_artifacts(artifacts)
    manifest["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=baseline.REPO, text=True).strip()
    manifest["git_status"] = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=baseline.REPO, text=True)
    (output / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "launch_config.json").write_text(json.dumps(config, indent=2))
    environment = os.environ.copy()
    for key, suffix in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache")):
        path = output / suffix
        path.mkdir()
        environment[key] = str(path)
    environment.update({baseline.OWNER_KEY: str(output),
                        "MIN_STEPS": str(min(config.get("min_steps", 200), config["max_steps"])),
                        "CONDA_ENV": config["conda_env"], "PYTHON_BIN": config["python_bin"],
                        "NLTK_DATA": "/home/ldl/nltk_data",
                        "HF_HOME": "/home/ldl/.cache/huggingface", "TORCH_HOME": "/home/ldl/.cache/torch"})
    stop, force = threading.Event(), threading.Event()

    def request_stop(*_):
        if stop.is_set():
            force.set()
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    print(f"[ablation] {args.variant} | {len(indices)} episodes | {output}", flush=True)
    started = time.monotonic()
    process = None
    try:
        with (output / "batch.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=baseline.REPO, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                print(baseline.progress(output, indices, started), flush=True)
                if process.poll() is not None or stop.wait(config["progress_interval_s"]):
                    break
    finally:
        baseline.cleanup(str(output), force)
        if process is not None:
            process.wait(timeout=10)
    print(baseline.completion_report(output, indices, time.monotonic() - started,
                                    stopped=stop.is_set(), returncode=process.returncode), flush=True)
    return 130 if stop.is_set() else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
