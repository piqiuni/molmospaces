"""Regression tests for M3's latest-frame freshness guard."""

from __future__ import annotations

import pathlib
import sys
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interaction_policy import InteractionRequest, VisualMLLMVerifier  # noqa: E402
import interaction_policy as policy_module  # noqa: E402


@pytest.fixture
def fake_clock(monkeypatch):
    class Clock:
        now = 100.0

        def advance(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(policy_module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(policy_module.time, "time", lambda: clock.now)
    monkeypatch.setattr(policy_module.time, "sleep", clock.advance)
    return clock


@pytest.mark.parametrize("late_reason", ["cancel", "deadline"])
def test_m3_rejects_success_response_that_returns_after_cancel_or_deadline(fake_clock, late_reason):
    calls, events = [], []
    cancelled = False

    def model(*args, **kwargs):
        nonlocal cancelled
        calls.append(1)
        if len(calls) == 2:
            if late_reason == "cancel":
                cancelled = True
            else:
                fake_clock.advance(10.0)
        fake_clock.advance(0.6)
        return {"state": "open", "confidence": 0.99}

    verifier = VisualMLLMVerifier(
        lambda: {"image_data_url": "data:image/jpeg;base64,AA==", "seq": len(calls)},
        model, events.append, stable_open_s=0.5, timeout_s=2.0,
        sample_period_s=0.2, require_fresh_frames=True,
    )
    result = verifier.verify(InteractionRequest("late"), lambda: cancelled)
    assert len(calls) == 2
    assert not result["success"]
    assert result["status"] == ("CANCELLED" if late_reason == "cancel" else "TIMEOUT")
    assert not any(event["phase"] == "SUCCEEDED" for event in events)


@pytest.mark.parametrize("expected,observed,succeeds", [
    ("open", "open", True), ("open", "ajar", True),
    ("closed", "closed", True), ("closed", "open", False),
    ("closed", "ajar", False), ("open", "closed", False),
])
def test_m3_verifies_requested_postcondition(fake_clock, expected, observed, succeeds):
    def model(*args, **kwargs):
        fake_clock.advance(0.3)
        return {"state": observed, "confidence": 0.95}

    verifier = VisualMLLMVerifier(lambda: "data:image/jpeg;base64,AA==", model,
                                  stable_open_s=0.5, timeout_s=2.0, sample_period_s=0.2)
    result = verifier.verify(InteractionRequest("state", expected_state=expected), lambda: False)
    assert result["success"] is succeeds
    assert result["post_state"] == (observed if succeeds else "unknown")


@pytest.mark.parametrize("confidence", [float("inf"), float("nan"), "bad", 1.5, -0.2])
def test_invalid_m3_confidence_is_never_success(fake_clock, confidence):
    def model(*args, **kwargs):
        fake_clock.advance(0.3)
        return {"state": "open", "confidence": confidence}

    verifier = VisualMLLMVerifier(lambda: "data:image/jpeg;base64,AA==", model,
                                  stable_open_s=0.5, timeout_s=2.0, min_confidence=0.0)
    result = verifier.verify(InteractionRequest("invalid-confidence"), lambda: False)
    assert not result["success"]
    assert result["status"] == "TIMEOUT"


def test_m3_uses_only_post_start_monotonic_capture_stamps(fake_clock):
    calls, events = [], []
    initial = iter([99.0, None, 150.0, 100.05, 100.04])
    sequence = 0

    def provider():
        nonlocal sequence
        sequence += 1
        stamp = next(initial, fake_clock.now)
        return {"image_data_url": "data:image/jpeg;base64,AA==", "seq": sequence, "stamp": stamp}

    def model(*args, **kwargs):
        calls.append(sequence)
        return {"state": "open", "confidence": 0.99}

    verifier = VisualMLLMVerifier(provider, model, events.append,
                                  stable_open_s=0.5, sample_period_s=0.2, timeout_s=3.0,
                                  require_post_start_frames=True, capture_clock=lambda: fake_clock.now)
    result = verifier.verify(InteractionRequest("causal"), lambda: False)
    assert result["success"]
    assert not set(calls).intersection({1, 2, 3, 5})
    reasons = {sample["reason"] for sample in result["evidence"]}
    assert {"capture_stamp_unavailable", "capture_stamp_in_future",
            "capture_not_after_verification_start_or_previous_frame"} <= reasons
    assert result["stable_duration_s"] >= 0.5


def test_model_latency_cannot_substitute_for_capture_time_span(fake_clock):
    sequence = 0

    def provider():
        nonlocal sequence
        sequence += 1
        # Many unique frames, but only 1 ms of actual capture time apart.
        return {"image_data_url": "data:image/jpeg;base64,AA==", "seq": sequence,
                "stamp": 100.0 + sequence * 0.001}

    def model(*args, **kwargs):
        fake_clock.advance(0.6)
        return {"state": "open", "confidence": 0.99}

    verifier = VisualMLLMVerifier(provider, model, stable_open_s=0.5, timeout_s=2.0,
                                  require_post_start_frames=True, capture_clock=lambda: fake_clock.now)
    assert not verifier.verify(InteractionRequest("compressed-captures"), lambda: False)["success"]


def test_m3_skips_replayed_image_and_requires_new_capture_identity():
    responses = {
        "choices": [{"message": {"content": '{"state":"open","confidence":0.95}'}}]
    }
    # The first two samples are the same ROS frame.  They must not be sent to
    # Qwen twice or extend the stable-open interval; later unique frames can
    # establish the post-action state.
    frame_ids = iter([1, 1, 2, 3, 4, 5])
    calls = []
    events = []

    def image_provider():
        try:
            seq = next(frame_ids)
        except StopIteration:
            seq = 5
        return {"image_data_url": "data:image/jpeg;base64,AA==", "seq": seq}

    def request_json(*args, **kwargs):
        calls.append(kwargs.get("image_data_url"))
        return responses

    verifier = VisualMLLMVerifier(
        image_provider,
        request_json,
        events.append,
        stable_open_s=0.5,
        sample_period_s=0.2,
        timeout_s=2.0,
        require_fresh_frames=True,
    )
    result = verifier.verify(
        InteractionRequest("fresh-frames", target_id="door-1"), lambda: False
    )

    assert result["success"] is True
    # Frame 1 is evaluated once; its replay is explicitly recorded but does
    # not create another model request.
    assert len(calls) >= 4
    assert any(
        event.get("result", {}).get("reason") == "duplicate_frame"
        for event in events
    )
    assert all(sample.get("fresh_frame", True) for sample in result["evidence"] if sample["reason"] != "duplicate_frame")


def test_m3_fresh_frame_mode_rejects_metadata_free_provider():
    calls = []
    verifier = VisualMLLMVerifier(
        lambda: "data:image/jpeg;base64,AA==",
        lambda *args, **kwargs: calls.append(kwargs) or {
            "choices": [{"message": {"content": '{"state":"open","confidence":1.0}'}}]
        },
        stable_open_s=0.5,
        sample_period_s=0.2,
        timeout_s=0.8,
        require_fresh_frames=True,
    )
    result = verifier.verify(InteractionRequest("no-metadata"), lambda: False)

    assert result["success"] is False
    assert result["status"] == "TIMEOUT"
    assert calls == []
    assert any(
        sample.get("reason") == "image_metadata_unavailable"
        for sample in result["evidence"]
    )
