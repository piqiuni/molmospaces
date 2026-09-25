from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_yoloe_bridge import _warmup_detector


def fixture_model(monkeypatch, kind="cuda", failure=None):
    calls = []
    synchronized = []
    device = SimpleNamespace(type=kind)
    cuda = SimpleNamespace(
        synchronize=lambda value: synchronized.append(value),
        get_device_name=lambda value: "test GPU",
        memory_allocated=lambda value: 1024 ** 2,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    def predict(**kwargs):
        calls.append(kwargs)
        if failure:
            raise failure
        return []

    model = SimpleNamespace(predict=predict, predictor=SimpleNamespace(device=device))
    args = SimpleNamespace(device="cuda:2", imgsz=64, conf=.1, iou=.7, max_det=50)
    return model, args, calls, synchronized


def test_warmup_runs_real_predict_contract_and_synchronizes(monkeypatch, capsys):
    model, args, calls, synchronized = fixture_model(monkeypatch)
    _warmup_detector(model, args)
    assert len(calls) == 2
    assert calls[0]["source"].shape == (64, 64, 3)
    assert calls[0]["source"].dtype == np.uint8
    assert calls[0]["device"] == "cuda:2"
    assert calls[0]["save"] is False
    assert synchronized == [model.predictor.device]
    assert "warmup OK" in capsys.readouterr().out


def test_cuda_failure_is_fatal_not_empty_report(monkeypatch, capsys):
    model, args, calls, synchronized = fixture_model(monkeypatch, failure=RuntimeError("CUDA 101"))
    with pytest.raises(RuntimeError, match="warmup FAILED.*CUDA 101"):
        _warmup_detector(model, args)
    assert len(calls) == 1
    assert not synchronized
    assert "warmup OK" not in capsys.readouterr().out


def test_silent_cpu_fallback_is_rejected(monkeypatch):
    model, args, *_ = fixture_model(monkeypatch, kind="cpu")
    with pytest.raises(RuntimeError, match="predictor is on"):
        _warmup_detector(model, args)


def test_explicit_cpu_is_allowed(monkeypatch):
    model, args, calls, synchronized = fixture_model(monkeypatch, kind="cpu")
    args.device = "cpu"
    _warmup_detector(model, args)
    assert len(calls) == 2
    assert not synchronized
