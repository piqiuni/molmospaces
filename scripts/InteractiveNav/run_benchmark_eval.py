#!/usr/bin/env python3
"""Configuration-driven benchmark launcher with live progress and owned cleanup."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_qwen_service import QwenService, endpoint_for_port, visible_devices

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "scripts/InteractiveNav/configs/evaluation/benchmark_batch.json"
OWNER_KEY = "INTERACTIVE_NAV_EVAL_RUN_ID"


def incomplete_indices(output, indices):
    pending = []
    for index in indices:
        try:
            row = json.loads((output / f"episode_{index:04d}/batch_task_summary.json").read_text())
        except (OSError, ValueError):
            row = {}
        if row.get("completed") is not True:
            pending.append(index)
    return pending


def build_command(config: dict, output: Path) -> tuple[list[str], list[int]]:
    indices = config.get("episode_indices")
    if indices is None:
        episode_stride = int(config.get("episode_stride", 1))
        if episode_stride < 1:
            raise ValueError("episode_stride must be positive")
        indices = [
            i
            for lo, hi in config["episode_ranges"]
            for i in range(lo, hi + 1, episode_stride)
        ]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("episode_ranges must be nonempty and must not overlap")
    if config["workers"] < 1 or config["max_steps"] < 1 or config["progress_interval_s"] <= 0:
        raise ValueError("workers, max_steps and progress_interval_s must be positive")
    command = [config["python_bin"], "-u", str(REPO / "scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py")]
    command += ["--output-dir", str(output), "--episode-indices", *map(str, indices)]
    for key in ("benchmark", "workers", "max_steps", "step_budget_mode", "base_master_port",
                "ros_observation_turn_multiplier", "ros_command_starvation_timeout_s",
                "semantic_attribute_request_timeout_s", "scene_timeout_s", "conda_env", "python_bin"):
        command += ["--" + key.replace("_", "-"), str(config[key])]
    model_env = Path(config["semantic_model_env_file"])
    command += ["--semantic-model-env-file", str(model_env if model_env.is_absolute() else REPO / model_env)]
    for key in ("model_endpoints", "mujoco_egl_devices"):
        command += ["--" + key.replace("_", "-"), *map(str, config[key])]
    if not config.get("recording", True):
        command += ["--no-recording"]
    if config.get("resume", False):
        command += ["--resume"]
    return command + ["--resource-telemetry", "--allow-failures"], indices


def check_mujoco_gpu_inventory(config: dict, environment=None) -> list[str] | None:
    """Validate the EGL assignment against the GPUs present at launch time.

    Shared-Qwen experiments do not pass ``--start-qwen`` and therefore used to
    skip the only GPU discovery performed by this launcher.  A stale static
    device list could consequently start a full comparison with all simulators
    on two cards while Qwen occupied all four.  GPU-bound configs are checked by
    default; ``check_mujoco_gpu_inventory: false`` can be used only for a
    deliberately device-less/externally managed run.  Configs can optionally
    require a minimum card count.  The resolved inventory is persisted in
    ``launch_config.json`` below.
    """
    check = config.get("check_mujoco_gpu_inventory")
    if check is None:
        check = bool(config.get("mujoco_egl_devices"))
    if not check:
        return None
    devices = [str(device) for device in visible_devices(os.environ if environment is None else environment)]
    required = int(config.get("required_mujoco_gpu_count", 0))
    if config.get("auto_assign_mujoco_egl_devices", False):
        requested = list(devices)
        config["mujoco_egl_devices"] = requested
    else:
        requested = [str(device) for device in config.get("mujoco_egl_devices", [])]
    if required < 0:
        raise ValueError("required_mujoco_gpu_count must be non-negative")
    if required and len(devices) < required:
        raise RuntimeError(
            f"Need at least {required} visible GPUs for MuJoCo EGL, found {devices}"
        )
    if not requested:
        raise RuntimeError("GPU inventory check requires mujoco_egl_devices")
    missing = [device for device in requested if device not in devices]
    if missing:
        raise RuntimeError(
            f"Configured MuJoCo EGL GPU IDs {missing} are not available; "
            f"visible GPUs are {devices}"
        )
    if required and len(set(requested)) < required:
        raise RuntimeError(
            f"Configured MuJoCo EGL assignment uses {sorted(set(requested))}, "
            f"but {required} unique GPUs are required"
        )
    config["gpu_preflight"] = {
        "visible_devices": devices,
        "requested_mujoco_egl_devices": requested,
        "required_mujoco_gpu_count": required,
        "worker_count": int(config.get("workers", 0)),
        "workers_per_gpu_if_even": (
            int(config.get("workers", 0)) / len(set(requested))
            if requested else None
        ),
    }
    return devices


def progress(output: Path, indices: list[int], started: float) -> str:
    rows = {}
    try:
        with (output / "summary.csv").open() as stream:
            rows = {int(row["episode_index"]): row for row in csv.DictReader(stream)}
    except (OSError, ValueError, KeyError):
        pass
    completed = failed = running = queued = 0
    completed_rows = []
    for index in indices:
        row = rows.get(index, {})
        task = output / f"episode_{index:04d}"
        # Per-episode summaries are written before a whole worker shard ends.
        try:
            row = json.loads((task / "batch_task_summary.json").read_text())
        except (OSError, ValueError):
            pass
        reported = str(row.get("worker_result_reported", "")).lower() == "true" or "runner_exit_code" in row and row.get("runner_exit_code") not in (None, "")
        if reported:
            ok = str(row.get("completed", "")).lower() == "true"
            if ok:
                completed += 1
                completed_rows.append(row)
            else:
                failed += 1
            continue
        attempts = sorted(task.glob("attempt_*"))
        if not attempts:
            queued += 1
            continue
        running += 1

    def mean(key: str) -> float:
        values = []
        for row in completed_rows:
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return sum(values) / len(values) if values else 0.0

    def rate(key: str) -> float:
        return sum(str(row.get(key, "")).lower() == "true" for row in completed_rows) / len(completed_rows)

    elapsed = time.monotonic() - started
    status = (f"[eval {elapsed / 60:.1f}min] 完成 {completed}/{len(indices)} | "
              f"运行/收尾 {running} | 排队 {queued} | 运行异常 {failed}")
    if not completed_rows:
        return status + "\n  已完成均值：暂无可聚合结果"
    metrics = (
        f"TaskSR={rate('task_success'):.3f} | "
        f"ICS={rate('interaction_conditioned_success'):.3f} | "
        f"NavSR={rate('nav_success'):.3f} | "
        f"ISR={rate('required_interaction_success'):.3f} | "
        f"SPL={mean('spl'):.3f} | "
        f"Steps={mean('step_count'):.1f} | "
        f"Eval={mean('elapsed_seconds'):.1f}s"
    )
    return status + f"\n  已完成均值（n={completed}）：{metrics}"


def owned_processes(owner: str) -> dict[int, str]:
    """Match a unique inherited marker, including reparented ROS processes."""
    result = {}
    marker = f"{OWNER_KEY}={owner}".encode()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if marker not in (path / "environ").read_bytes().split(b"\0"):
                continue
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z":
                result[int(path.name)] = fields[19]
        except (OSError, ValueError, IndexError):
            pass
    return result


def cleanup(owner: str, force: threading.Event | None = None) -> None:
    targets = owned_processes(owner)
    def send(sig: int) -> None:
        for pid, started in owned_processes(owner).items():
            try:
                # Guard against PID reuse between enumeration and signalling.
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[19] == started:
                    os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                pass

    print(f"[eval] 正在清理本轮进程（{len(targets)} 个）；再次 Ctrl+C 将立即强制结束。", flush=True)
    deadline = time.monotonic() + 8
    while owned_processes(owner) and time.monotonic() < deadline:
        if force is not None and force.is_set():
            break
        send(signal.SIGTERM)
        time.sleep(.2)
    # Repeat to catch detached/reparented children born during shutdown.
    deadline = time.monotonic() + 5
    while owned_processes(owner) and time.monotonic() < deadline:
        send(signal.SIGKILL)
        time.sleep(.1)
    remaining = owned_processes(owner)
    if remaining:
        raise RuntimeError(f"本轮进程清理未完成，残留 PID：{sorted(remaining)}")
    print(f"[eval] 已清理本轮子进程（初始 {len(targets)} 个），模型服务未触碰。", flush=True)


def completion_report(output: Path, indices: list[int], elapsed: float, *, stopped: bool,
                      returncode: int) -> str:
    """Present existing scores; never treat missing results as navigation failures."""
    def read(path):
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    summary = read(output / "summary.json")
    rows = {r["episode_index"]: r for r in summary.get("episodes", [])}
    # Interrupted workers may have committed task summaries but not the batch summary.
    for index in indices:
        row = read(output / f"episode_{index:04d}/batch_task_summary.json")
        if row:
            rows[index] = row
    reported = [rows[i] for i in indices if i in rows and
                (rows[i].get("worker_result_reported") is True or
                 rows[i].get("runner_exit_code") is not None)]
    completed = [r for r in reported if r.get("completed") is True]
    missing = len(indices) - len(reported)
    partial = stopped or returncode != 0 or len(completed) != len(indices)
    lines = [f"[eval] {'部分结果' if partial else '全部评测完成'} | batch 退出码 {returncode}",
             f"  计划 {len(indices)} | 已报告 {len(reported)} | 运行完成 {len(completed)} | "
             f"运行异常 {len(reported)-len(completed)} | 未报告 {missing}",
             "  注：运行完成不等于导航成功；以下评分仅统计运行完成且该指标有效的场景。"]

    def rate(label, key):
        values = [r[key] for r in completed if isinstance(r.get(key), bool)]
        return (f"{label} {sum(values)}/{len(values)} ({sum(values)/len(values):.1%})"
                if values else f"{label} N/A")

    def mean(key):
        values = [float(r[key]) for r in completed if isinstance(r.get(key), (int, float))]
        return f"{sum(values)/len(values):.3f} (n={len(values)})" if values else "N/A"

    lines += ["  " + " | ".join(rate(label, key) for label, key in (
        ("正式成功率", "success"), ("任务成功率", "task_success"), ("导航成功率", "nav_success"))),
        f"  平均 SPL {mean('spl')} | 平均步数 {mean('step_count')}",
        f"  平均实际路径 {mean('navigation_path_length_m')} m | GT 路径 {mean('reference_path_length_m')} m",
        "  " + " | ".join(rate(label, key) for label, key in (
            ("必要交互满足", "required_interaction_success"), ("序列成功", "sequence_success"))),
        f"  批次墙钟 {elapsed:.1f} s | 单场 runner 均值 {mean('elapsed_sec')} s | evaluator 均值 {mean('elapsed_seconds')} s"]
    counts = Counter(str(r.get("terminal_reason") or "unknown") for r in reported)
    lines.append("  终止原因：" + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "N/A"))
    total_ms = samples = 0
    for row in completed:
        timing = row.get("timing_summary") or {}
        phase = timing.get("phases", {}).get("step_total", {})
        if timing.get("unit") == "ms" and phase.get("count", 0) > 0:
            total_ms += phase["total"]
            samples += phase["count"]
    if samples and total_ms > 0:
        lines.append(f"  循环加权均值 {total_ms/1000/samples:.3f} s/step（{samples} 个样本）；"
                     f"已完成场景循环数/批次墙钟 {samples/elapsed:.3f} step/s")
    else:
        lines.append("  循环耗时：N/A（没有有效 step_total 计时）")
    aggregate = read(output / "aggregate_metrics.json") or summary.get("aggregate", {})
    resource = aggregate.get("resource_telemetry") or {}
    if resource:
        def number(key, scale=1):
            value = resource.get(key)
            return f"{value/scale:.2f}" if isinstance(value, (int, float)) else "N/A"
        lines.append(f"  主机 CPU 平均/峰值 {number('mean_host_cpu_percent')}%/{number('peak_host_cpu_percent')}% | "
                     f"主机内存峰值 {number('peak_host_mem_used_mb', 1024)} GiB（含其他进程）")
        for gpu in resource.get("gpu_peaks", []):
            lines.append(f"  GPU {gpu['index']}：显存峰值 {gpu['peak_memory_used_mb']/1024:.2f} GiB | "
                         f"利用率峰值 {gpu['peak_utilization_gpu_percent']:.1f}%（含模型服务）")
    else:
        lines.append("  资源汇总：N/A；若中断可检查 resource_telemetry.csv")
    videos = sum(bool(r.get("six_panel_video")) and Path(r["six_panel_video"]).is_file() for r in reported)
    lines += [f"  已存在视频 {videos}/{len(indices)} | 视频目录 {output / 'video_gallery'}",
              f"  逐场结果 {output / 'summary.csv'}",
              f"  原始指标 {output / 'aggregate_metrics.json'}"]
    text = "\n".join(lines)
    (output / "completion_report.txt").write_text(text + "\n")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-indices", type=int, nargs="+", help="场景编号，如 10 11 1010")
    parser.add_argument("--recording", action=argparse.BooleanOptionalAction, default=None,
                        help="完整录制（默认开启）；--no-recording 使用 fast eval")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-qwen", action="store_true", help="Start one multi-GPU Qwen service once for this batch")
    parser.add_argument("--retry-rounds", type=int, default=None, help="Deferred rounds for incomplete episodes (default 1)")
    parser.add_argument("--experiment-manifest", type=Path, help="Frozen multi-arm online experiment manifest")
    parser.add_argument("--experiment-lane", type=int, choices=(0, 1, 2), help="One 15-worker experiment lane")
    args = parser.parse_args()
    if args.experiment_manifest is not None:
        if args.experiment_lane is None:
            parser.error("--experiment-manifest requires --experiment-lane")
        if any(value is not None for value in (args.workers, args.max_steps, args.output_dir, args.episode_indices, args.recording)) or args.retry_rounds not in (None, 0):
            parser.error("Experiment resources/outputs are frozen in the manifest; normal overrides/retries are disabled")
        from run_m2_experiment_lane import run_lane
        return run_lane(args.experiment_manifest, args.experiment_lane, config_path=args.config,
                        dry_run=args.dry_run, start_qwen=args.start_qwen)
    if args.experiment_lane is not None:
        parser.error("--experiment-lane requires --experiment-manifest")
    config = json.loads(args.config.read_text())
    retry_rounds = args.retry_rounds if args.retry_rounds is not None else int(config.get("retry_rounds", 1))
    if retry_rounds < 0:
        parser.error("--retry-rounds must be non-negative")
    start_qwen = args.start_qwen or config.get("start_qwen", False)
    if start_qwen:
        devices = visible_devices(os.environ)
        qwen_port = int(os.environ.get("QWEN36_PORT", "8000"))
        # One API endpoint is backed by one vLLM instance. vLLM's internal
        # DP/TP scheduler owns all visible GPUs; the eval client does no LB.
        config["model_endpoints"] = [endpoint_for_port(qwen_port)]
        if not args.dry_run:
            config["mujoco_egl_devices"] = list(devices)
    for key in ("workers", "max_steps", "episode_indices", "recording"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    check_mujoco_gpu_inventory(config, os.environ)
    name = datetime.datetime.now().strftime("eval-%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or Path(config["output_root"]) / name).resolve()
    command, indices = build_command(config, output)
    if args.expected_episodes is not None:
        if args.expected_episodes < 1 or len(indices) != args.expected_episodes:
            parser.error(f"expected {args.expected_episodes} episodes, selected {len(indices)}")
        config["expected_episodes"] = args.expected_episodes
    if args.dry_run:
        print(json.dumps({"output_dir": str(output), "command": command}, indent=2))
        return 0
    for port in range(config["base_master_port"], config["base_master_port"] + config["workers"]):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    resume = bool(config.get("resume", False))
    output.mkdir(parents=True, exist_ok=resume)
    runner = REPO / "scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh"
    snapshot = output / "runner_snapshot.sh"
    if not (resume and snapshot.exists()):
        snapshot.write_bytes(runner.read_bytes())
    subprocess.run(["bash", "-n", str(snapshot)], check=True)
    command += ["--runner", str(snapshot)]
    environment = os.environ.copy()
    environment["INTERACTIVE_NAV_SCRIPT_DIR"] = str(runner.parent)
    for key, suffix in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache")):
        directory = output / suffix
        directory.mkdir(exist_ok=resume)
        environment[key] = str(directory)
    environment.update({OWNER_KEY: str(output), "MIN_STEPS": str(min(config.get("min_steps", 200), config["max_steps"])),
                        "CONDA_ENV": config["conda_env"], "PYTHON_BIN": config["python_bin"],
                        "HF_HOME": "/home/ldl/.cache/huggingface", "TORCH_HOME": "/home/ldl/.cache/torch"})
    (output / "launch_config.json").write_text(json.dumps(config, indent=2))
    stop = threading.Event()
    force = threading.Event()
    def request_stop(*_):
        if stop.is_set():
            force.set()
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    recording_label = "完整录制" if config.get("recording", True) else "fast eval"
    print(f"[eval] {len(indices)} 场 | {config['workers']} worker | {config['step_budget_mode']} / 上限 {config['max_steps']} 步 | {recording_label}\n"
          f"[eval] 模型：{config['model_endpoints']}\n[eval] 输出：{output}\n"
          "[eval] Ctrl+C 停止并清理本轮任务；详细日志见各场景 attempt 目录。", flush=True)
    started = time.monotonic()
    process = None
    service = QwenService(output, environment, port=int(os.environ.get("QWEN36_PORT", "8000"))) if start_qwen else None
    returncode = 1
    try:
        if service:
            print("[eval] 启动单实例 Qwen 服务；所有可见 GPU 由同一个 vLLM API 内部调度，统一入口 8000。", flush=True)
            service.start(stop)
        with (output / "batch.log").open("ab" if resume else "wb") as log:
            for round_index in range(retry_rounds + 1):
                round_command = list(command)
                if round_index and "--resume" not in round_command:
                    round_command.append("--resume")
                process = subprocess.Popen(round_command, cwd=REPO, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    print(progress(output, indices, started), flush=True)
                    if service:
                        service.check()
                    if process.poll() is not None or stop.wait(config["progress_interval_s"]):
                        break
                if stop.is_set():
                    break
                pending = incomplete_indices(output, indices)
                (output / f"retry_queue_{round_index}.json").write_text(json.dumps(pending))
                returncode = process.returncode or (1 if pending else 0)
                if not pending:
                    break
                print(f"[eval] 第 {round_index + 1} 轮结束，{len(pending)} 个运行异常／未完成场景进入尾部重试队列。", flush=True)
    except (RuntimeError, InterruptedError) as exc:
        print(f"[eval] {exc}", flush=True)
        returncode = 1
    finally:
        try:
            cleanup(str(output), force)
            if process is not None:
                process.wait(timeout=10)
        finally:
            if service:
                service.close()
    print(f"[eval] {'已手动停止' if stop.is_set() else '已结束'}；日志：{output / 'batch.log'}", flush=True)
    print(completion_report(output, indices, time.monotonic() - started,
                            stopped=stop.is_set(), returncode=returncode), flush=True)
    return 130 if stop.is_set() else returncode


if __name__ == "__main__":
    raise SystemExit(main())
