from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("rospy")
import physical_interaction_policy_node as module
from interaction_policy import HumanAssistActuator, InteractionRequest, InteractionResult


def message(seq):
    return SimpleNamespace(width=8, height=6, step=24, encoding="rgb8", data=bytes(144),
                           header=SimpleNamespace(seq=seq, stamp=SimpleNamespace(to_sec=lambda: float(seq))))


def node():
    value = object.__new__(module.PhysicalInteractionPolicyNode)
    value._lock = threading.RLock()
    value._latest_image_message = None
    value._encoded_image_message = None
    value._encoded_sample = None
    return value


def test_human_assist_speaks_english():
    calls = []
    actuator = HumanAssistActuator(lambda text, wait: calls.append(text) or {"accepted": True})
    actuator.execute(InteractionRequest("test", target_kind="door"), lambda: False)
    assert calls == ["Please open the door in front of me. Thank you."]


def test_policy_deadline_returns_timeout_and_clears_owner(monkeypatch):
    import json
    value = node()
    value._active_command = "command"
    value._cancel = threading.Event()
    value._param = lambda name, default: 2.0 if name == "policy_timeout_s" else default
    results, events = [], []
    value._result_pub = SimpleNamespace(publish=lambda m: results.append(json.loads(m.data)))
    value._event_pub = SimpleNamespace(publish=lambda m: events.append(json.loads(m.data)))
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    def execute(request, cancel):
        assert not cancel()
        now[0] = 103.0
        assert cancel()
        return InteractionResult(request.command_id, True, "SUCCEEDED", request.target_id, request.target_kind)
    monkeypatch.setattr(module, "build_interaction_policy", lambda *a, **kw: SimpleNamespace(execute=execute))
    value._run(InteractionRequest("command", target_kind="door"))
    assert results[-1]["status"] == "TIMEOUT"
    assert results[-1]["success"] is False
    assert results[-1]["detail"]["reason"] == "interaction_policy_timeout"
    assert events[-1]["stage"] == "FINISHED"
    assert value._active_command == ""
    assert value._policy_deadline == 0.0


def test_idle_camera_callbacks_do_not_encode_and_repeated_reads_use_cache(monkeypatch):
    policy = node()
    assert policy._image_provider() is None
    original = module.PILImage.frombytes
    calls = []

    def decode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.PILImage, "frombytes", decode)
    for seq in range(1, 101):
        policy._image_callback(message(seq))
    assert not calls
    sample = policy._image_provider()
    assert sample["seq"] == 100 and sample["stamp"] == 100.0
    assert sample["image_data_url"].startswith("data:image/jpeg;base64,")
    assert policy._image_provider() is sample
    assert len(calls) == 1


def test_new_frame_during_encode_is_not_replaced_by_encoded_predecessor(monkeypatch):
    policy = node()
    first, second = message(1), message(2)
    original = module.PILImage.frombytes

    def decode(*args, **kwargs):
        # A callback is free to run while image conversion is in progress.
        policy._image_callback(second)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.PILImage, "frombytes", decode)
    policy._image_callback(first)
    assert policy._image_provider()["seq"] == 1
    assert policy._latest_image_message is second
    assert policy._encoded_image_message is None
    assert policy._image_provider()["seq"] == 2
    assert policy._encoded_image_message is second
