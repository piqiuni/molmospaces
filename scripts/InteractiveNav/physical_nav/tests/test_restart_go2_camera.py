import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "restart_go2_camera.py"
SPEC = importlib.util.spec_from_file_location("restart_go2_camera", SCRIPT)
camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera)


def test_modes_preserve_transport_imu_align_and_unrequested_rate():
    original = ["python3", "/camera.py", "--url", "ws://host:12335",
                "--enable-camera-imu", "--align-to", "depth",
                "--color-width=640", "--color-height", "480", "--color-fps", "10",
                "--depth-fps", "10", "--publish-fps", "10"]
    result = camera.command_for_modes(original, (1280, 720, 30), None, None)
    assert result[:7] == original[:7]
    assert "--color-width=640" not in result
    for key, value in (("--color-width", "1280"), ("--color-height", "720"),
                       ("--color-fps", "30"), ("--depth-fps", "10"),
                       ("--publish-fps", "10")):
        assert result[result.index(key) + 1] == value
    assert original[8] == "--color-height"


@pytest.mark.parametrize("value", ["1280x720", "640x0@30", "640x480@-1", "640x480@nan"])
def test_bad_modes_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        camera.mode(value)


def test_process_identity_excludes_remote_helper_and_embedded_paths():
    assert camera.is_bridge_command(["/usr/bin/python3.8", "-u", "/camera.py"], "/camera.py")
    assert not camera.is_bridge_command(["python3", "-", "--remote", "--bridge-path", "/camera.py"], "/camera.py")
    assert not camera.is_bridge_command(["python3", "-c", "pass", "/camera.py"], "/camera.py")
    assert not camera.is_bridge_command(["bash", "/camera.py"], "/camera.py")


def test_dry_run_never_stops_or_launches(monkeypatch):
    monkeypatch.setattr(camera, "find_bridge", lambda _: (123, ["python3", "/camera.py"]))
    monkeypatch.setattr(camera.os, "kill", lambda *_: pytest.fail("dry run killed a process"))
    monkeypatch.setattr(camera.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("dry run launched a process"))
    args = SimpleNamespace(bridge_path="/camera.py", rgb=(640, 480, 30),
                           depth=None, publish_fps=10, dry_run=True)
    result = camera.remote_restart(args)
    assert result["dry_run"]
    assert result["old_pid"] == 123


@pytest.mark.parametrize("fail_new_mode", [False, True])
def test_restart_replaces_only_exact_camera_process(tmp_path, monkeypatch, fail_new_mode):
    bridge = tmp_path / "camera.py"
    bridge.write_text("import time, sys\n"
                      + ("if '--color-width' in sys.argv: sys.exit(2)\n" if fail_new_mode else "")
                      + "print('D435i started', flush=True)\ntime.sleep(60)\n")
    original = subprocess.Popen([sys.executable, str(bridge), "--align-to", "depth"],
                                stdout=subprocess.DEVNULL)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 stdout=subprocess.DEVNULL)
    children = []
    popen = subprocess.Popen

    def record_child(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(camera.subprocess, "Popen", record_child)
    try:
        args = SimpleNamespace(bridge_path=str(bridge), rgb=(1280, 720, 30),
                               depth=(848, 480, 30), publish_fps=10, dry_run=False,
                               log=str(tmp_path / "camera.log"), timeout=5)
        result = camera.remote_restart(args)
        assert result["old_pid"] == original.pid
        assert result["pid"] == children[-1].pid != original.pid
        assert result["camera_started"] is not fail_new_mode
        if fail_new_mode:
            assert result["previous_mode_relaunched"]
            assert "--color-width" not in result["command"]
        assert original.wait(timeout=2) != 0
        assert unrelated.poll() is None
        assert "--align-to" in camera.read_process(children[-1].pid)
    finally:
        for child in [original, unrelated] + children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_local_entry_uses_only_ssh_and_updates_camera_ownership(tmp_path, monkeypatch):
    calls = []

    def remote(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps({"pid": 456, "camera_started": True}))

    monkeypatch.setattr(camera.subprocess, "run", remote)
    monkeypatch.setenv("PHYSICAL_NAV_ALL_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--rgb", "1280x720@30",
                                     "--depth", "848x480@30", "--publish-fps", "10"])
    camera.main()
    assert len(calls) == 1
    assert calls[0][0][0] == "ssh"
    assert "--rgb 1280x720@30" in calls[0][0][-1]
    assert "--depth 848x480@30" in calls[0][0][-1]
    assert "physical_nav_all.sh" not in calls[0][0][-1]
    assert (tmp_path / "go2_bridge.pid").read_text() == "456\n"
    assert (tmp_path / "go2_bridge.owned").exists()
