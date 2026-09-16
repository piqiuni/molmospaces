import importlib.util
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import pytest

spec = importlib.util.spec_from_file_location("eval_launcher", Path(__file__).with_name("run_benchmark_eval.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_command_pins_runtime_and_budget(tmp_path):
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    command, indices = launcher.build_command(config, tmp_path)
    assert indices == [10, 11, 12, 1010, 1011, 1012, 2010, 2011, 2012, 2013]
    assert command[command.index("--conda-env") + 1] == config["conda_env"]
    assert command[command.index("--python-bin") + 1] == config["python_bin"]
    assert command[command.index("--ros-observation-turn-multiplier") + 1] == "1"
    assert "--no-recording" not in command
    assert command[command.index("--step-budget-mode") + 1] == "dynamic"
    config["episode_ranges"].append([10, 10])
    with pytest.raises(ValueError, match="overlap"):
        launcher.build_command(config, tmp_path)


def test_progress_does_not_count_placeholder_as_finished(tmp_path):
    (tmp_path / "summary.csv").write_text(
        "episode_index,worker_result_reported,runner_exit_code,completed\n"
        "10,False,,False\n11,False,,False\n12,False,,False\n"
    )
    task = tmp_path / "episode_0010"
    task.mkdir()
    (task / "batch_task_summary.json").write_text(json.dumps({
        "runner_exit_code": 0, "completed": True, "terminal_reason": "max_steps",
    }))
    debug = tmp_path / "episode_0011/attempt_001/debug"
    debug.mkdir(parents=True)
    (debug / "trajectory.csv").write_text("step_id,elapsed_sec\n17,43.5\n")
    message = launcher.progress(tmp_path, [10, 11, 12], time.monotonic())
    assert "完成 1/3" in message
    assert "运行/收尾 1" in message
    assert "排队 1" in message
    assert "已完成均值（n=1）" in message


def test_progress_averages_only_completed_episodes_without_recording(tmp_path):
    for index, completed, steps, success in ((10, True, 70, True), (11, False, 900, False)):
        task = tmp_path / f"episode_{index:04d}"
        task.mkdir()
        (task / "batch_task_summary.json").write_text(json.dumps({
            "runner_exit_code": 0 if completed else 1,
            "completed": completed,
            "step_count": steps,
            "nav_success": success,
        }))
    message = launcher.progress(tmp_path, [10, 11, 12], time.monotonic())
    assert "完成 1/3" in message
    assert "运行异常 1" in message
    assert "排队 1" in message
    assert "NavSR=1.000" in message
    assert "Steps=70.0" in message


def test_cli_overrides(tmp_path):
    result = subprocess.run([
        sys.executable, str(Path(launcher.__file__)), "--dry-run",
        "--output-dir", str(tmp_path / "run"), "--episode-indices", "1010", "1011",
        "--workers", "2", "--max-steps", "50", "--no-recording",
    ], capture_output=True, text=True, check=True)
    command = json.loads(result.stdout)["command"]
    assert command[command.index("--episode-indices") + 1:command.index("--benchmark")] == ["1010", "1011"]
    assert command[command.index("--workers") + 1] == "2"
    assert command[command.index("--max-steps") + 1] == "50"
    assert "--no-recording" in command


def test_completion_report_uses_weighted_timings_and_explicit_denominators(tmp_path):
    rows = [dict(episode_index=i, worker_result_reported=True, completed=True,
                 success=i == 10, spl=.5 if i == 10 else 0,
                 terminal_reason="target_found" if i == 10 else "max_steps",
                 timing_summary={"unit": "ms", "phases": {"step_total": {
                     "count": count, "total": total}}})
            for i, count, total in [(10, 10, 10000), (11, 30, 90000)]]
    (tmp_path / "summary.json").write_text(json.dumps({"episodes": rows}))
    report = launcher.completion_report(tmp_path, [10, 11], 50, stopped=False, returncode=0)
    assert "全部评测完成" in report
    assert "正式成功率 1/2 (50.0%)" in report
    assert "平均 SPL 0.250" in report
    assert "2.500 s/step" in report
    assert "0.800 step/s" in report
    assert (tmp_path / "completion_report.txt").read_text().strip() == report


def test_partial_report_recovers_task_summary_without_scoring_missing_rows(tmp_path):
    task = tmp_path / "episode_0010"
    task.mkdir()
    (task / "batch_task_summary.json").write_text(json.dumps({
        "episode_index": 10, "runner_exit_code": 0, "completed": True, "success": True}))
    report = launcher.completion_report(tmp_path, [10, 11], 50, stopped=True, returncode=-15)
    assert "部分结果" in report
    assert "未报告 1" in report
    assert "正式成功率 1/1 (100.0%)" in report
    assert "循环耗时：N/A" in report


@pytest.mark.parametrize("repeat_signal", [False, True])
def test_ctrl_c_cleans_stubborn_detached_processes(tmp_path, repeat_signal):
    output = tmp_path / "run"
    ready = tmp_path / "ready"
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config.update(workers=1, base_master_port=0)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    # Substitute a tiny process tree for the simulator; no ROS/GPU is needed.
    child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    fake_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        f"time.sleep(.2); Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    harness = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('launcher',{launcher.__file__!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.build_command=lambda config,output: ([sys.executable,'-c',{fake_code!r}],[10]); "
        "sys.exit(m.main())"
    )
    proc = subprocess.Popen([sys.executable, "-c", harness, "--output-dir", str(output),
                             "--config", str(config_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists(), proc.communicate(timeout=2)[0] if proc.poll() is not None else "startup timeout"
        proc.send_signal(signal.SIGINT)
        if repeat_signal:
            time.sleep(.3)
            proc.send_signal(signal.SIGINT)
        stdout, _ = proc.communicate(timeout=20)
        assert proc.returncode == 130, stdout
        assert "已手动停止" in stdout
        assert not launcher.owned_processes(str(output))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        launcher.cleanup(str(output))


def test_cleanup_finds_detached_child_and_preserves_unrelated_process(tmp_path):
    owner = str(tmp_path)
    environment = {**os.environ, launcher.OWNER_KEY: owner}
    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid)"
    parent = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True)
    detached_pid = int(parent.stdout.strip())
    other_env = {k: v for k, v in os.environ.items() if k != launcher.OWNER_KEY}
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=other_env)
    try:
        assert detached_pid in launcher.owned_processes(owner)
        launcher.cleanup(owner)
        assert not launcher.owned_processes(owner)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
        for pid in launcher.owned_processes(owner):
            os.kill(pid, 9)
