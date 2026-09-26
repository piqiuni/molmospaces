import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launch(*args, **overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PHYSICAL_NAV_")}
    env.update(overrides)
    return subprocess.run(
        ["bash", str(ROOT / "run_yolo.sh"), "--dry-run", *args],
        env=env, text=True, capture_output=True, timeout=5,
    )


def option(result, name):
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("gpu,device", [("0", "cuda:0"), ("2", "cuda:2"), ("cpu", "cpu")])
def test_single_gpu_selector(gpu, device):
    result = launch("--gpu", gpu, CUDA_VISIBLE_DEVICES="3")
    assert option(result, "--device") == device
    assert "ignoring inherited CUDA_VISIBLE_DEVICES" in result.stderr


def test_defaults_and_environment_are_shared_with_supervisor():
    result = launch()
    assert option(result, "--device") == "cuda:0"
    assert option(result, "--rate") == "10"
    assert option(result, "--camera-z") == "0.885"
    assert option(result, "--camera-pitch") == "0.157079633"
    result = launch(PHYSICAL_NAV_YOLO_GPU="1", PHYSICAL_NAV_YOLO_RATE="5",
                    PHYSICAL_NAV_CAMERA_Z="1.2", PHYSICAL_NAV_START_WEB="0")
    assert option(result, "--device") == "cuda:1"
    assert option(result, "--rate") == "5"
    assert option(result, "--camera-z") == "1.2"
    assert option(result, "--web-url") == ""


def test_cli_overrides_gpu_environment_and_preserves_extra_arguments():
    result = launch("--gpu", "2", "--model-path", "/tmp/model with spaces.pt",
                    PHYSICAL_NAV_YOLO_GPU="1")
    assert option(result, "--device") == "cuda:2"
    assert shlex.split(result.stdout)[-2:] == ["--model-path", "/tmp/model with spaces.pt"]


@pytest.mark.parametrize("args", [("--gpu",), ("--gpu", "-1"), ("--gpu", "0,1"),
                                  ("--device", "cuda:1"), ("--device=cuda:1",)])
def test_invalid_or_second_device_selector_is_rejected(args):
    assert launch(*args).returncode == 2


@pytest.mark.parametrize("key", ["PHYSICAL_NAV_YOLO_DEVICE", "PHYSICAL_NAV_YOLO_CUDA_VISIBLE_DEVICES"])
def test_legacy_environment_requires_explicit_migration(key):
    result = launch(**{key: "0"})
    assert result.returncode == 2
    assert "use PHYSICAL_NAV_YOLO_GPU" in result.stderr


def test_supervisor_uses_shared_entry_and_fingerprints_it():
    source = (ROOT / "start_physical_nav.sh").read_text()
    start = source.index("  if (( YOLO_REUSED == 0 )); then")
    block = source[start:source.index('    register_process yoloe', start)]
    assert 'bash "${ROOT_DIR}/run_yolo.sh"' in block
    assert "CUDA_VISIBLE_DEVICES=" not in block
    assert 'sha256sum "${ROOT_DIR}/run_yolo.sh"' in source
    all_source = (ROOT / "physical_nav_all.sh").read_text()
    assert 'PHYSICAL_NAV_YOLO_GPU="${PHYSICAL_NAV_YOLO_GPU:-0}"' in all_source
    assert "PHYSICAL_NAV_YOLO_DEVICE=" not in all_source
