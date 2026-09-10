"""D435 startup failures must not leave a RealSense pipeline running."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_readonly_sensor_bridge import D435iSource


class _FakeConfig:
    def enable_stream(self, *_args):
        pass


class _FakePipeline:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.callback_arg_counts = []

    def start(self, _config, *_callbacks):
        self.started += 1
        self.callback_arg_counts.append(len(_callbacks))
        return object()

    def stop(self):
        self.stopped += 1


def test_post_start_initialization_failure_stops_pipeline(monkeypatch):
    pipeline = _FakePipeline()

    def fail_align(_stream):
        raise RuntimeError("align construction failed")

    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=_FakeConfig,
        frame_queue=lambda *_args: object(),
        align=fail_align,
        stream=SimpleNamespace(
            depth="depth", color="color", gyro="gyro", accel="accel",
        ),
        format=SimpleNamespace(
            z16="z16", bgr8="bgr8", motion_xyz32f="motion_xyz32f",
        ),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    with pytest.raises(RuntimeError, match="align construction failed"):
        D435iSource(848, 480, 10, align_to="depth")

    assert pipeline.started == 1
    assert pipeline.stopped == 1
    assert pipeline.callback_arg_counts == [1]


def test_align_color_keeps_native_frameset_queue_path(monkeypatch):
    pipeline = _FakePipeline()

    def fail_align(_stream):
        raise RuntimeError("align construction failed")

    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=_FakeConfig,
        frame_queue=lambda *_args: object(),
        align=fail_align,
        stream=SimpleNamespace(
            depth="depth", color="color", gyro="gyro", accel="accel",
        ),
        format=SimpleNamespace(
            z16="z16", bgr8="bgr8", motion_xyz32f="motion_xyz32f",
        ),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    with pytest.raises(RuntimeError, match="align construction failed"):
        D435iSource(848, 480, 10, align_to="color")

    assert pipeline.started == 1
    assert pipeline.stopped == 1
    assert pipeline.callback_arg_counts == [1]


def test_invalid_alignment_is_rejected_before_loading_realsense(monkeypatch):
    monkeypatch.delitem(sys.modules, "pyrealsense2", raising=False)
    with pytest.raises(ValueError, match="align_to"):
        D435iSource(848, 480, 10, align_to="invalid")
