"""Qwen launch defaults and process ownership; never start SSH in tests."""

from pathlib import Path
from types import SimpleNamespace
import io
import json
import os
import subprocess

import pytest

import qwen_ssh_tunnel as tunnel


@pytest.mark.parametrize("args,forward", [
    ([], "127.0.0.1:18080:127.0.0.1:8100"),
    (["--local-port", "19080", "--remote-port", "9000"],
     "127.0.0.1:19080:127.0.0.1:9000"),
])
def test_supervisor_pid_becomes_ssh_process(monkeypatch, args, forward):
    calls = []
    monkeypatch.setattr(tunnel.os, "execvp", lambda executable, argv:
                        calls.append((executable, argv)))
    tunnel.main(args)
    assert len(calls) == 1
    executable, argv = calls[0]
    assert executable == "ssh"
    assert argv[argv.index("-L") + 1] == forward
    assert "ExitOnForwardFailure=yes" in argv
    assert "ServerAliveCountMax=3" in argv


def test_both_physical_launchers_use_the_same_remote_service_default():
    root = Path(__file__).resolve().parents[1]
    for name in ("physical_nav_all.sh", "start_physical_nav.sh"):
        source = (root / name).read_text()
        assert '${PHYSICAL_NAV_QWEN_REMOTE_PORT:-8100}' in source
        assert '${PHYSICAL_NAV_QWEN_REMOTE_PORT:-8000}' not in source


def _args(tmp_path):
    return SimpleNamespace(ssh_port=41051, user="root", host="example.test",
                           local_port=18080, remote_port=8100, remote_bind="127.0.0.1",
                           pid_file=str(tmp_path / "qwen.pid"), log_file=str(tmp_path / "qwen.log"))


def test_owned_forward_recognizes_legacy_port_but_not_other_sessions(tmp_path):
    expected = tunnel.ssh_command(_args(tmp_path))
    old = list(expected)
    old[old.index("-L") + 1] = "127.0.0.1:18080:127.0.0.1:8000"
    assert tunnel.owned_forward(old, expected).endswith(":8000")
    assert tunnel.owned_forward(old + ["echo", "hello"], expected) is None
    other = list(old)
    other[-1] = "root@another.test"
    assert tunnel.owned_forward(other, expected) is None
    other = list(old)
    other[other.index("-L") + 1] = "127.0.0.1:19080:127.0.0.1:8000"
    assert tunnel.owned_forward(other, expected) is None


def test_ensure_reuses_only_healthy_matching_forward(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path)
    expected = tunnel.ssh_command(args)
    monkeypatch.setattr(tunnel, "matching_processes", lambda _: [(321, expected, expected[-2])])
    monkeypatch.setattr(tunnel, "model_endpoint_ready", lambda _: True)
    monkeypatch.setattr(tunnel.os, "kill", lambda *a: pytest.fail("must not kill healthy tunnel"))
    assert tunnel.ensure_tunnel(args) == 321
    output = capsys.readouterr().out
    assert "Qwen health=PASS remote=example.test:8100" in output
    assert "http://127.0.0.1:18080/v1/models" in output
    assert Path(args.pid_file).read_text() == "321\n"
    monkeypatch.setattr(tunnel, "model_endpoint_ready", lambda _: False)
    with pytest.raises(RuntimeError, match="unhealthy"):
        tunnel.ensure_tunnel(args)


def test_cli_reports_health_failure_with_remote_address(tmp_path, monkeypatch, capsys):
    def fail(_args):
        raise RuntimeError("model endpoint unavailable")
    monkeypatch.setattr(tunnel, "ensure_tunnel", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        tunnel.main(["--ensure", "--host", "example.test", "--remote-port", "9100",
                     "--pid-file", str(tmp_path / "pid"),
                     "--log-file", str(tmp_path / "log")])
    assert "Qwen health=FAIL remote=example.test:9100" in capsys.readouterr().out


def test_ensure_refuses_unknown_listener_even_with_recycled_pid_file(tmp_path, monkeypatch):
    args = _args(tmp_path)
    Path(args.pid_file).write_text("9999\n")
    monkeypatch.setattr(tunnel, "matching_processes", lambda _: [])
    monkeypatch.setattr(tunnel, "local_port_open", lambda _: True)
    monkeypatch.setattr(tunnel.os, "kill", lambda *a: pytest.fail("unknown PID must not be killed"))
    with pytest.raises(RuntimeError, match="unrecognized listener"):
        tunnel.ensure_tunnel(args)


def test_ensure_replaces_exact_stale_forward_and_checks_health(tmp_path, monkeypatch):
    args = _args(tmp_path)
    expected = tunnel.ssh_command(args)
    old = list(expected)
    old[-2] = "127.0.0.1:18080:127.0.0.1:8000"
    monkeypatch.setattr(tunnel, "matching_processes", lambda _: [(321, old, old[-2])])
    running = [True]
    monkeypatch.setattr(tunnel, "process_args", lambda _: old if running[0] else [])
    killed = []
    def terminate(pid, sig):
        killed.append((pid, sig))
        running[0] = False
    monkeypatch.setattr(tunnel.os, "kill", terminate)
    monkeypatch.setattr(tunnel, "local_port_open", lambda _: False)
    launched = []
    def launch(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(pid=654, poll=lambda: None)
    monkeypatch.setattr(tunnel.subprocess, "Popen", launch)
    monkeypatch.setattr(tunnel, "model_endpoint_ready", lambda _: True)
    monkeypatch.setattr(tunnel.time, "sleep", lambda _: None)
    assert tunnel.ensure_tunnel(args) == 654
    assert killed == [(321, tunnel.signal.SIGTERM)]
    assert launched[0][0] == expected
    assert launched[0][1]["start_new_session"] is True


@pytest.mark.parametrize("payload,valid", [
    ({"choices": [{"message": {"content": "OK"}}]}, True),
    ({"choices": [{"message": {"content": ""}}]}, False),
    ({"choices": []}, False),
    ({"error": "unavailable"}, False),
])
def test_inference_requires_nonempty_answer(monkeypatch, payload, valid):
    calls = []
    def open_request(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(tunnel.urllib.request, "build_opener",
                        lambda *_: SimpleNamespace(open=open_request))
    if valid:
        tunnel.verify_inference(18080, "test-model", 8)
    else:
        with pytest.raises(RuntimeError, match="inference health failed"):
            tunnel.verify_inference(18080, "test-model", 8)
    request, timeout = calls[0]
    assert request.full_url.endswith("/v1/chat/completions")
    assert json.loads(request.data)["model"] == "test-model"
    assert timeout == 8


def test_inference_timeout_is_fatal(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("test timeout")
    monkeypatch.setattr(tunnel.urllib.request, "build_opener",
                        lambda *_: SimpleNamespace(open=fail))
    with pytest.raises(RuntimeError, match="test timeout"):
        tunnel.verify_inference(18080, "test-model", 8)


def test_all_start_cannot_skip_failed_qwen_check(tmp_path):
    """Run the real shell entry with only its Python health checker stubbed."""
    stub = tmp_path / "python3"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\nexit 1\n')
    stub.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "physical_nav_all.sh"
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               PHYSICAL_NAV_ALL_RUNTIME_DIR=str(tmp_path / "state"),
               PHYSICAL_NAV_ALL_LOG_DIR=str(tmp_path / "logs"),
               PHYSICAL_NAV_START_QWEN_TUNNEL="0")
    result = subprocess.run(["bash", str(script), "start"], env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "--verify-inference" in result.stdout
    assert "remaining services were not started" in result.stderr
