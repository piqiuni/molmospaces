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


def test_launcher_defaults_to_ten_second_scene_spacing(tmp_path):
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    command, _ = launcher.build_command(config, tmp_path)
    assert command[command.index("--scene-start-interval-s") + 1] == "10.0"
    config["scene_start_interval_s"] = 0
    command, _ = launcher.build_command(config, tmp_path)
    assert command[command.index("--scene-start-interval-s") + 1] == "0"


def test_retry_queue_excludes_completed_algorithm_failures(tmp_path):
    for index, row in [(0, {"completed": True, "success": False}),
                       (5, {"completed": False, "success": True, "exit_code": 2})]:
        directory = tmp_path / f"episode_{index:04d}"
        directory.mkdir()
        (directory / "batch_task_summary.json").write_text(json.dumps(row))
    assert launcher.incomplete_indices(tmp_path, [0, 5, 10]) == [5, 10]


def test_qwen_respects_visible_gpu_selection(monkeypatch):
    assert launcher.visible_devices({"CUDA_VISIBLE_DEVICES": "3,7"}) == ["3", "7"]
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: "3, GPU-abc\n7, GPU-def\n")
    assert launcher.visible_devices({"CUDA_VISIBLE_DEVICES": "GPU-abc"}) == ["3"]
    with pytest.raises(RuntimeError):
        launcher.visible_devices({"CUDA_VISIBLE_DEVICES": ""})


def test_shared_qwen_gpu_preflight_checks_inventory_and_records_assignment(monkeypatch):
    monkeypatch.setattr(launcher, "visible_devices", lambda environment: ["0", "1", "2", "3"])
    config = {
        "check_mujoco_gpu_inventory": True,
        "required_mujoco_gpu_count": 4,
        "workers": 60,
        "mujoco_egl_devices": ["1", "2", "3", "0"],
    }
    assert launcher.check_mujoco_gpu_inventory(config, {}) == ["0", "1", "2", "3"]
    assert config["gpu_preflight"]["visible_devices"] == ["0", "1", "2", "3"]
    assert config["gpu_preflight"]["workers_per_gpu_if_even"] == 15.0


def test_shared_qwen_gpu_preflight_fails_before_two_card_run(monkeypatch):
    monkeypatch.setattr(launcher, "visible_devices", lambda environment: ["0", "1"])
    config = {
        "check_mujoco_gpu_inventory": True,
        "required_mujoco_gpu_count": 4,
        "workers": 60,
        "mujoco_egl_devices": ["0", "1"],
    }
    with pytest.raises(RuntimeError, match="at least 4 visible GPUs"):
        launcher.check_mujoco_gpu_inventory(config, {})



def test_expected_episode_count_is_checked_before_launch(tmp_path, monkeypatch, capsys):
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config.update(episode_indices=[2000, 2001], check_mujoco_gpu_inventory=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["eval", "--config", str(config_path),
                                     "--expected-episodes", "30", "--dry-run"])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
    assert "expected 30 episodes, selected 2" in capsys.readouterr().err


def test_qwen_failure_does_not_restart(tmp_path):
    class DeadProcess:
        returncode = 1
        def poll(self):
            return 1
    service = launcher.QwenService(tmp_path, {})
    service.processes = [("gpu0", DeadProcess())]
    with pytest.raises(RuntimeError, match="No restart"):
        service.check()


@pytest.mark.parametrize("count", [1, 2, 4])
def test_qwen_starts_single_vllm_instance_for_all_gpus(tmp_path, monkeypatch, count):
    import eval_qwen_service as qwen
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            pass
    monkeypatch.setattr(qwen.socket, "socket", Socket)
    service = qwen.QwenService(tmp_path, {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(count))})
    commands = []
    environments = []
    monkeypatch.setattr(service, "_spawn",
                        lambda label, command, env: (commands.append(command), environments.append(env)))
    monkeypatch.setattr(service, "_ready", lambda *args: None)
    service.start(None)
    assert len(commands) == 1
    assert commands[0][0:2] == ["bash", "/home/ldl/qwen36-fp8/serve_qwen36_fp8_remote.zsh"]
    assert environments[0]["QWEN36_GPU_IDS"] == ",".join(str(i) for i in range(count))
    assert environments[0]["QWEN36_TP_SIZE"] == "1"
    assert environments[0]["QWEN36_DP_SIZE"] == str(count)
    assert environments[0]["QWEN36_API_SERVER_COUNT"] == "1"
    assert environments[0]["QWEN36_MAX_MODEL_LEN"] == "16384"
    assert environments[0]["QWEN36_MAX_NUM_SEQS"] == "16"
    assert service.endpoint == "http://127.0.0.1:8000/v1"
    assert service.tensor_parallel_size == 1
    assert service.data_parallel_size == count
    assert service.max_num_seqs == 16
    assert environments[0]["QWEN36_MAX_NUM_SEQS"] == "16"


@pytest.mark.parametrize(
    ("count", "target", "expected"),
    [(1, 60, 60), (2, 60, 30), (4, 60, 16), (4, 100, 25)],
)
def test_qwen_concurrency_scales_with_gpus_and_eval_workers(
    tmp_path, monkeypatch, count, target, expected
):
    import eval_qwen_service as qwen
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            pass
    monkeypatch.setattr(qwen.socket, "socket", Socket)
    service = qwen.QwenService(
        tmp_path,
        {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(count))},
        target_concurrency=target,
    )
    environments = []
    monkeypatch.setattr(service, "_spawn",
                        lambda label, command, env: environments.append(env))
    monkeypatch.setattr(service, "_ready", lambda *args: None)
    service.start(None)
    assert service.max_num_seqs == expected
    assert environments[0]["QWEN36_MAX_NUM_SEQS"] == str(expected)


def test_qwen_explicit_concurrency_override_wins(tmp_path, monkeypatch):
    import eval_qwen_service as qwen
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            pass
    monkeypatch.setattr(qwen.socket, "socket", Socket)
    service = qwen.QwenService(
        tmp_path,
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "QWEN36_MAX_NUM_SEQS": "23",
        },
        target_concurrency=60,
    )
    environments = []
    monkeypatch.setattr(service, "_spawn",
                        lambda label, command, env: environments.append(env))
    monkeypatch.setattr(service, "_ready", lambda *args: None)
    service.start(None)
    assert service.max_num_seqs == 23
    assert environments[0]["QWEN36_MAX_NUM_SEQS"] == "23"


def test_single_endpoint_is_not_round_robin():
    import eval_qwen_service as qwen
    assert qwen.endpoint_for_port(8000) == "http://127.0.0.1:8000/v1"


def test_two_gpu_qwen_lb_launches_independent_backends(tmp_path, monkeypatch):
    import eval_qwen_service as qwen
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            pass
    monkeypatch.setattr(qwen.socket, "socket", Socket)
    service = qwen.QwenLBService(tmp_path, {"CUDA_VISIBLE_DEVICES": "0,1",
                                            "QWEN36_MAX_NUM_SEQS": "16"})
    spawned = []
    healthy = []
    monkeypatch.setattr(service, "_spawn", lambda label, command, env:
                        spawned.append((label, command, env)))
    monkeypatch.setattr(service, "_ready", lambda ports, stop: healthy.append(ports))
    service.start(None)
    assert [label for label, _, _ in spawned] == ["gpu0", "gpu1", "lb"]
    assert [env["QWEN36_GPU_IDS"] for _, _, env in spawned[:2]] == ["0", "1"]
    assert [env["QWEN36_PORT"] for _, _, env in spawned[:2]] == ["8000", "8001"]
    assert all(env["QWEN36_DP_SIZE"] == "1" for _, _, env in spawned[:2])
    assert spawned[2][1][-2:] == ["--backends", "8000,8001"]
    assert healthy == [(8000, 8001), [8010]]
    assert service.endpoint == "http://127.0.0.1:8010/v1"
    assert json.loads((tmp_path / "qwen-service/deployment.json").read_text())["total_max_num_seqs"] == 32


def test_four_gpu_qwen_lb_routes_four_backends_on_distinct_ports(tmp_path, monkeypatch):
    import eval_qwen_service as qwen

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            pass

    monkeypatch.setattr(qwen.socket, "socket", Socket)
    service = qwen.QwenLBService(
        tmp_path,
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "QWEN36_MAX_NUM_SEQS": "16"},
        backend_ports=(8100, 8101, 8102, 8103),
        lb_port=8000,
        target_concurrency=60,
    )
    spawned = []
    healthy = []
    monkeypatch.setattr(service, "_spawn", lambda label, command, env:
                        spawned.append((label, command, env)))
    monkeypatch.setattr(service, "_ready", lambda ports, stop: healthy.append(ports))

    service.start(None)

    assert [label for label, _, _ in spawned] == ["gpu0", "gpu1", "gpu2", "gpu3", "lb"]
    assert [env["QWEN36_GPU_IDS"] for _, _, env in spawned[:4]] == ["0", "1", "2", "3"]
    assert [env["QWEN36_PORT"] for _, _, env in spawned[:4]] == ["8100", "8101", "8102", "8103"]
    assert all(env["QWEN36_DP_SIZE"] == "1" for _, _, env in spawned[:4])
    assert spawned[4][1][-2:] == ["--backends", "8100,8101,8102,8103"]
    assert healthy == [(8100, 8101, 8102, 8103), [8000]]
    assert service.endpoint == "http://127.0.0.1:8000/v1"
    deployment = json.loads((tmp_path / "qwen-service/deployment.json").read_text())
    assert deployment["devices"] == ["0", "1", "2", "3"]
    assert deployment["total_max_num_seqs"] == 64


def test_qwen_lb_rejects_mismatch_between_gpu_count_and_backend_ports(tmp_path):
    import eval_qwen_service as qwen

    service = qwen.QwenLBService(
        tmp_path, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        backend_ports=(8100, 8101), lb_port=8000,
    )
    with pytest.raises(RuntimeError, match="one backend port per GPU"):
        service.start(None)


def test_launcher_runs_deferred_retry_and_retains_full_selection(tmp_path, monkeypatch):
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config.update(
        workers=1,
        base_master_port=0,
        episode_indices=[0, 5],
        check_mujoco_gpu_inventory=False,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    commands = []
    class Batch:
        returncode = 0
        def __init__(self, command, **kwargs):
            commands.append(command)
            for index in [0, 5]:
                directory = output / f"episode_{index:04d}"
                directory.mkdir(exist_ok=True)
                (directory / "batch_task_summary.json").write_text(json.dumps({
                    "episode_index": index, "completed": index == 0 or len(commands) == 2,
                    "runner_exit_code": 0 if index == 0 or len(commands) == 2 else 4}))
        def poll(self):
            return 0
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(sys, "argv", ["eval", "--config", str(config_path), "--output-dir", str(output)])
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", Batch)
    monkeypatch.setattr(launcher, "cleanup", lambda *a: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *a: None)
    assert launcher.main() == 0
    assert len(commands) == 2
    assert "--resume" not in commands[0] and "--resume" in commands[1]
    assert json.loads((output / "retry_queue_0.json").read_text()) == [5]
    assert json.loads((output / "retry_queue_1.json").read_text()) == []
    assert (output / "runner_snapshot.sh").exists()


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


def test_command_can_resume_matching_incomplete_batch(tmp_path):
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config["resume"] = True
    command, _ = launcher.build_command(config, tmp_path)
    assert "--resume" in command


def test_command_supports_interleaved_stride_offset(tmp_path):
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config.update({
        "episode_ranges": [[0, 29]],
        "episode_stride": 3,
        "episode_stride_offset": 1,
    })
    _, indices = launcher.build_command(config, tmp_path)
    assert indices == [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    config["episode_stride_offset"] = 3
    with pytest.raises(ValueError, match="episode_stride_offset"):
        launcher.build_command(config, tmp_path)


def test_command_uniformly_samples_episode_range_by_stride(tmp_path):
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config["episode_ranges"] = [[0, 19]]
    config["episode_stride"] = 5
    _, indices = launcher.build_command(config, tmp_path)
    assert indices == [0, 5, 10, 15]

    config["episode_stride"] = 0
    with pytest.raises(ValueError, match="episode_stride must be positive"):
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


def test_resume_reuses_existing_output_and_appends_batch_log(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "batch.log").write_text("previous batch\n")
    config = json.loads(launcher.DEFAULT_CONFIG.read_text())
    config.update(workers=1, base_master_port=0, resume=True, retry_rounds=0, progress_interval_s=0.1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        launcher,
        "build_command",
        lambda config, output: ([sys.executable, "-c", "print('resumed batch')"], [10]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(launcher.__file__), "--config", str(config_path), "--output-dir", str(output)],
    )
    # An empty batch exiting zero cannot make the missing episode complete.
    assert launcher.main() == 1
    assert (output / "batch.log").read_text() == "previous batch\nresumed batch\n"
    assert (output / "tmp").is_dir()
    assert (output / "cache").is_dir()


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


def test_shared_qwen_gpu_preflight_checks_inventory_and_records_assignment(monkeypatch):
    monkeypatch.setattr(launcher, "visible_devices", lambda environment: ["0", "1", "2", "3"])
    config = {
        "check_mujoco_gpu_inventory": True,
        "required_mujoco_gpu_count": 4,
        "workers": 60,
        "mujoco_egl_devices": ["1", "2", "3", "0"],
    }
    assert launcher.check_mujoco_gpu_inventory(config, {}) == ["0", "1", "2", "3"]
    assert config["gpu_preflight"]["visible_devices"] == ["0", "1", "2", "3"]
    assert config["gpu_preflight"]["workers_per_gpu_if_even"] == 15.0



def test_shared_qwen_gpu_preflight_fails_before_two_card_run(monkeypatch):
    monkeypatch.setattr(launcher, "visible_devices", lambda environment: ["0", "1"])
    config = {
        "check_mujoco_gpu_inventory": True,
        "required_mujoco_gpu_count": 4,
        "workers": 60,
        "mujoco_egl_devices": ["0", "1"],
    }
    with pytest.raises(RuntimeError, match="at least 4 visible GPUs"):
        launcher.check_mujoco_gpu_inventory(config, {})
