"""D435 device time must survive capture and select the matching body pose."""

from collections import deque
from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_readonly_sensor_bridge import D435iSource, ReadOnlyState, _DeviceClockMapper
import physical_protocol as protocol
from physical_protocol import (
    clock_probe_packet,
    decode_wire_packet,
    encode_wire_packet,
    image_packet,
)


def test_recent_minimum_offset_exposes_later_queue_backlog():
    mapper = _DeviceClockMapper(
        window_s=30., warmup_samples=2, warmup_span_s=0.,
    )
    first = mapper.map(
        1000., "hardware_clock", stream="depth",
        observed_monotonic=11., observed_wall=101.,
    )
    delayed = mapper.map(
        1100., "hardware_clock", stream="depth",
        observed_monotonic=11.5, observed_wall=101.5,
    )

    assert first["valid"] is False
    assert first["warmup_complete"] is False
    assert first["capture_monotonic"] == 11.
    assert delayed["valid"] is True
    assert delayed["warmup_complete"] is True
    assert delayed["capture_monotonic"] == pytest.approx(11.1)
    assert delayed["capture_wall"] == pytest.approx(101.1)
    assert delayed["dequeue_age_s"] == pytest.approx(.4)
    assert delayed["clock_mapping_method"] == "recent_min_offset"


def test_sustained_device_backlog_is_not_absorbed_when_recent_window_expires():
    mapper = _DeviceClockMapper(
        window_s=1., warmup_samples=2, warmup_span_s=0.,
    )
    mapper.map(
        1000., "hardware_clock", stream="depth",
        observed_monotonic=11., observed_wall=101.,
    )
    mapper.map(
        1100., "hardware_clock", stream="depth",
        observed_monotonic=11.1, observed_wall=101.1,
    )

    delayed = mapper.map(
        3100., "hardware_clock", stream="depth",
        observed_monotonic=13.6, observed_wall=103.6,
    )

    assert delayed["offset_s"] == pytest.approx(10.)
    assert delayed["recent_offset_s"] == pytest.approx(10.5)
    assert delayed["capture_monotonic"] == pytest.approx(13.1)
    assert delayed["dequeue_age_s"] == pytest.approx(.5)


def test_clean_device_sample_quickly_removes_biased_startup_backlog():
    mapper = _DeviceClockMapper(
        window_s=30., warmup_samples=2, warmup_span_s=0.,
    )
    # Both warm-up samples were dequeued 200 ms late, so the initial lower
    # envelope cannot know that this latency is avoidable.
    mapper.map(
        1000., "hardware_clock", stream="depth",
        observed_monotonic=11.2, observed_wall=101.2,
    )
    biased = mapper.map(
        1100., "hardware_clock", stream="depth",
        observed_monotonic=11.3, observed_wall=101.3,
    )
    clean = mapper.map(
        2000., "hardware_clock", stream="depth",
        observed_monotonic=12., observed_wall=102.,
    )

    assert biased["offset_s"] == pytest.approx(10.2)
    assert clean["offset_s"] == pytest.approx(10.)
    assert clean["dequeue_age_s"] == pytest.approx(0.)
    assert clean["lower_reanchor_count"] >= 1


@pytest.mark.parametrize("drift_ppm", [-100., 100.])
def test_device_mapper_tracks_one_hour_bidirectional_clock_drift(drift_ppm):
    mapper = _DeviceClockMapper(
        window_s=30., warmup_samples=2, warmup_span_s=0.,
    )
    device_s = 1.
    host_monotonic = 11.
    host_wall = 101.
    mapper.map(
        device_s * 1000., "hardware_clock", stream="depth",
        observed_monotonic=host_monotonic, observed_wall=host_wall,
    )
    device_s += .1
    host_monotonic += .1
    host_wall += .1
    mapper.map(
        device_s * 1000., "hardware_clock", stream="depth",
        observed_monotonic=host_monotonic, observed_wall=host_wall,
    )
    rate = drift_ppm * 1e-6

    for _sample in range(36_000):
        host_monotonic += .1
        host_wall += .1
        device_s += .1 * (1. - rate)
        result = mapper.map(
            device_s * 1000., "hardware_clock", stream="depth",
            observed_monotonic=host_monotonic, observed_wall=host_wall,
        )

    # A positive-drift estimate deliberately follows the 30-second recent
    # lower envelope, so it remains one window (3 ms at 100 ppm) conservative.
    expected_offset = 10. + rate * (
        3600. - (30. if drift_ppm > 0. else 0.)
    )
    assert result["offset_s"] == pytest.approx(expected_offset, abs=1e-8)
    assert result["estimated_rate_ppm"] == pytest.approx(drift_ppm, abs=.01)
    assert abs(result["estimated_rate_ppm"]) <= result["max_adjust_rate_ppm"]


def test_device_mapper_holdover_tracks_drift_without_learning_usb_backlog():
    mapper = _DeviceClockMapper(
        window_s=30., warmup_samples=2, warmup_span_s=0.,
    )
    device_s = 1.
    nominal_host_monotonic = 11.
    nominal_host_wall = 101.
    mapper.map(
        device_s * 1000., "hardware_clock", stream="depth",
        observed_monotonic=nominal_host_monotonic,
        observed_wall=nominal_host_wall,
    )
    device_s += .1
    nominal_host_monotonic += .1
    nominal_host_wall += .1
    mapper.map(
        device_s * 1000., "hardware_clock", stream="depth",
        observed_monotonic=nominal_host_monotonic,
        observed_wall=nominal_host_wall,
    )
    rate = 100e-6

    for _sample in range(1_200):
        nominal_host_monotonic += .1
        nominal_host_wall += .1
        device_s += .1 * (1. - rate)
        mapper.map(
            device_s * 1000., "hardware_clock", stream="depth",
            observed_monotonic=nominal_host_monotonic,
            observed_wall=nominal_host_wall,
        )

    for _sample in range(6_000):
        nominal_host_monotonic += .1
        nominal_host_wall += .1
        device_s += .1 * (1. - rate)
        result = mapper.map(
            device_s * 1000., "hardware_clock", stream="depth",
            observed_monotonic=nominal_host_monotonic + .4,
            observed_wall=nominal_host_wall + .4,
        )

    expected_clock_offset = 10. + rate * (720. - 30.)
    assert result["offset_s"] == pytest.approx(
        expected_clock_offset, abs=1e-4,
    )
    assert result["estimated_queue_excess_s"] > .39
    assert result["rejected_anchor_count"] >= 1
    assert result["holdover_age_s"] > 0.


@pytest.mark.parametrize("domain", ["", "bogus_clock", "not_hardware_clock"])
def test_unknown_device_domain_falls_back_without_claiming_clock_mapping(domain):
    mapper = _DeviceClockMapper()
    result = mapper.map(
        1000., domain, stream="depth",
        observed_monotonic=11., observed_wall=101.,
    )

    assert result["valid"] is False
    assert result["capture_monotonic"] == 11.
    assert result["capture_wall"] == 101.
    assert result["clock_mapping_method"] == "host_read_fallback"
    assert result["failure_reason"] == "unsupported_timestamp_domain"
    assert mapper.snapshot()["sample_counts"] == {}


def test_invalid_sample_replaces_stale_valid_clock_diagnostic():
    mapper = _DeviceClockMapper(warmup_samples=2, warmup_span_s=0.)
    mapper.map(
        1000., "hardware_clock", stream="depth",
        observed_monotonic=11., observed_wall=101.,
    )
    mapper.map(
        1100., "hardware_clock", stream="depth",
        observed_monotonic=11.1, observed_wall=101.1,
    )
    assert mapper.snapshot()["latest"]["valid"] is True

    mapper.map(
        1200., "bogus_clock", stream="depth",
        observed_monotonic=11.2, observed_wall=101.2,
    )

    latest = mapper.snapshot()["latest"]
    assert latest["valid"] is False
    assert latest["failure_reason"] == "unsupported_timestamp_domain"


@pytest.mark.parametrize("domain", [
    "hardware_clock",
    "sensor_hardware_clock",
    "timestamp_domain.hardware_clock",
    "rs2_timestamp_domain.hardware_clock",
])
def test_hardware_clock_aliases_share_one_canonical_domain(domain):
    mapper = _DeviceClockMapper(warmup_samples=2, warmup_span_s=0.)
    result = mapper.map(
        1000., domain, stream="depth",
        observed_monotonic=11., observed_wall=101.,
    )

    assert result["timestamp_domain"] == "hardware_clock"
    assert mapper.snapshot()["sample_counts"] == {"hardware_clock": 1}


def test_sdk_epoch_domain_maps_directly_without_hardware_offset_estimate():
    mapper = _DeviceClockMapper()
    result = mapper.map(
        100_000., "timestamp_domain.global_time", stream="depth",
        observed_monotonic=11., observed_wall=100.025,
    )

    assert result["valid"] is True
    assert result["source_epoch_valid"] is True
    assert result["capture_wall"] == pytest.approx(100.)
    assert result["capture_monotonic"] == pytest.approx(10.975)
    assert result["dequeue_age_s"] == pytest.approx(.025)
    assert result["clock_mapping_method"] == "sdk_global_time"


def test_implausible_future_sdk_epoch_fails_closed():
    mapper = _DeviceClockMapper()
    result = mapper.map(
        101_000., "global_time", stream="depth",
        observed_monotonic=11., observed_wall=100.,
    )

    assert result["valid"] is False
    assert result["failure_reason"] == "epoch_timestamp_out_of_range"


def test_mapper_resets_offset_epoch_on_per_stream_timestamp_regression():
    mapper = _DeviceClockMapper(warmup_samples=2, warmup_span_s=0.)
    mapper.map(
        10000., "hardware_clock", stream="depth",
        observed_monotonic=20., observed_wall=200.,
    )
    reset = mapper.map(
        50., "hardware_clock", stream="depth",
        observed_monotonic=21., observed_wall=201.,
    )

    assert reset["capture_monotonic"] == 21.
    assert reset["dequeue_age_s"] == 0.
    assert mapper.snapshot()["reset_count"] == 1


def _motion_history_source(*entries):
    source = object.__new__(D435iSource)
    source._imu_history = deque(entries, maxlen=512)
    return source


def test_motion_match_uses_mapped_epoch_across_global_and_hardware_domains():
    source = _motion_history_source((
        1_700_000_000_080.,
        "global_time",
        {
            "sample": "global-imu",
            "sample_at": 100.08,
            "sample_time_valid": True,
        },
    ))

    motion = source._motion_at(
        100., "hardware_clock",
        capture_wall=100.1, capture_wall_valid=True,
    )

    assert motion["sample"] == "global-imu"


def test_motion_match_excludes_mapped_imu_sample_after_depth_exposure():
    source = _motion_history_source(
        (
            1_700_000_000_080.,
            "global_time",
            {"sample": "causal", "sample_at": 100.08,
             "sample_time_valid": True},
        ),
        (
            1_700_000_000_110.,
            "global_time",
            {"sample": "future", "sample_at": 100.11,
             "sample_time_valid": True},
        ),
    )

    motion = source._motion_at(
        100., "hardware_clock",
        capture_wall=100.1, capture_wall_valid=True,
    )

    assert motion["sample"] == "causal"


def test_motion_match_rejects_invalid_mapping_and_raw_domain_mismatch():
    source = _motion_history_source((
        1_700_000_000_080.,
        "global_time",
        {
            "sample": "unmapped-global-imu",
            "sample_at": 100.08,
            "sample_time_valid": False,
        },
    ))

    assert source._motion_at(
        100., "hardware_clock",
        capture_wall=100.1, capture_wall_valid=True,
    ) == {}


def test_motion_match_falls_back_to_causal_known_same_raw_domain():
    source = _motion_history_source(
        (95., "timestamp_domain.hardware_clock", {"sample": "causal"}),
        (105., "hardware_clock", {"sample": "future"}),
    )

    motion = source._motion_at(
        100., "sensor_hardware_clock", capture_wall_valid=False,
    )

    assert motion["sample"] == "causal"
    assert source._motion_at(100., "", capture_wall_valid=False) == {}


def test_body_pose_history_selects_capture_time_not_latest_dequeue_pose():
    state = ReadOnlyState(history_size=4)
    state.update({"received_at": 100., "position": [0., 0., 0.], "yaw": 0.})
    state.update({"received_at": 100.1, "position": [1., 0., 0.], "yaw": .2})
    state.update({"received_at": 100.2, "position": [2., 0., 0.], "yaw": .4})

    selected, delta = state.snapshot_at(100.09, max_delta_s=.15)

    assert selected["position"] == [1., 0., 0.]
    assert delta == pytest.approx(.01)
    assert state.snapshot()["position"] == [2., 0., 0.]


def test_body_pose_history_fails_closed_when_nearest_pose_is_too_far():
    state = ReadOnlyState()
    state.update({"received_at": 100., "position": [0., 0., 0.], "yaw": 0.})

    selected, delta = state.snapshot_at(101., max_delta_s=.15)

    assert selected == {}
    assert delta == pytest.approx(-1.)


class _MetadataFrame:
    def __init__(
        self, metadata, *, timestamp_ms=123.5,
        timestamp_domain="timestamp_domain.hardware_clock",
    ):
        self._metadata = metadata
        self._timestamp_ms = timestamp_ms
        self._timestamp_domain = timestamp_domain

    def supports_frame_metadata(self, key):
        return key in self._metadata

    def get_frame_metadata(self, key):
        return self._metadata[key]

    def get_frame_number(self):
        return 7

    def get_timestamp(self):
        return self._timestamp_ms

    def get_frame_timestamp_domain(self):
        return self._timestamp_domain


def _timestamp_source():
    source = object.__new__(D435iSource)
    source.rs = type("RS", (), {
        "frame_metadata_value": type("Metadata", (), {
            "sensor_timestamp": "sensor_timestamp",
            "frame_timestamp": "frame_timestamp",
            "actual_exposure": "actual_exposure",
        })(),
    })()
    return source


def test_sensor_metadata_uses_exposure_midpoint_and_microsecond_units():
    source = _timestamp_source()
    result = source._capture_timestamp_identity(_MetadataFrame({
        "sensor_timestamp": 123456,
        "actual_exposure": 8500,
    }))

    stamp_ms, domain, kind, metadata = result
    assert stamp_ms == pytest.approx(123.456)
    assert domain == "hardware_clock"
    assert kind == "sensor_exposure_midpoint"
    assert metadata == {
        "sdk_frame_timestamp_ms": 123.5,
        "sdk_frame_timestamp_domain": "hardware_clock",
        "sensor_exposure_timestamp_us": 123456.,
        "imu_reference_timestamp_ms": 123.456,
        "imu_reference_timestamp_domain": "hardware_clock",
        "actual_exposure_us": 8500.,
    }


def test_global_frame_and_device_metadata_yield_global_exposure_midpoint():
    source = _timestamp_source()
    stamp_ms, domain, kind, metadata = source._capture_timestamp_identity(
        _MetadataFrame(
            {
                "sensor_timestamp": 9_996_000,
                "frame_timestamp": 10_000_000,
            },
            timestamp_ms=1_700_000_000_100.,
            timestamp_domain="timestamp_domain.global_time",
        )
    )

    assert stamp_ms == pytest.approx(1_700_000_000_096.)
    assert domain == "global_time"
    assert kind == "sensor_exposure_midpoint_global"
    assert metadata["sensor_to_frame_delta_us"] == -4000.
    assert metadata["imu_reference_timestamp_ms"] == 9996.
    assert metadata["imu_reference_timestamp_domain"] == "hardware_clock"


def test_missing_sensor_metadata_preserves_frame_timestamp_semantics():
    source = _timestamp_source()
    stamp_ms, domain, kind, metadata = source._capture_timestamp_identity(
        _MetadataFrame({})
    )

    assert stamp_ms == 123.5
    assert domain == "hardware_clock"
    assert kind == "frame_readout_timestamp"
    assert metadata == {
        "sdk_frame_timestamp_ms": 123.5,
        "sdk_frame_timestamp_domain": "hardware_clock",
        "imu_reference_timestamp_ms": 123.5,
        "imu_reference_timestamp_domain": "hardware_clock",
    }
    assert math.isfinite(stamp_ms)


def test_system_time_without_metadata_is_explicit_host_arrival():
    source = _timestamp_source()
    stamp_ms, domain, kind, _metadata = source._capture_timestamp_identity(
        _MetadataFrame({}, timestamp_domain="timestamp_domain.system_time")
    )

    assert stamp_ms == 123.5
    assert domain == "system_time"
    assert kind == "host_time_of_arrival"


def _wire_packet(capture_timing=None):
    return image_packet(
        seq=1,
        stamp=100.,
        rgb_jpeg=b"rgb",
        depth_png=b"depth",
        width=2,
        height=2,
        camera_frame="camera",
        depth_scale=.001,
        intrinsics={"fx": 1.},
        color_depth_sync_ms=.2,
        capture_timing=capture_timing,
    )


def test_capture_timing_is_optional_for_protocol_v1_legacy_packets():
    packet = _wire_packet()

    assert packet["v"] == 1
    assert "capture_timing" not in packet


def test_capture_timing_survives_compressed_wire_roundtrip():
    timing = {
        "version": 1,
        "timestamp_kind": "sensor_exposure_midpoint",
        "source_epoch_valid": True,
        "dequeue_age_s": .023,
    }

    decoded = decode_wire_packet(encode_wire_packet(_wire_packet(timing)))

    assert decoded["capture_timing"] == timing
    assert decoded["stamp"] == 100.


def test_clock_probe_is_small_read_only_protocol_v1_envelope():
    packet = clock_probe_packet(seq=2)

    assert packet["v"] == 1
    assert packet["type"] == "clock_probe"
    assert packet["seq"] == 2
    assert packet["stamp"] > 0.
    assert math.isfinite(packet["source_monotonic"])
    assert packet["source_monotonic"] > 0.
    assert math.isfinite(packet["source_clock_pair_span_s"])
    assert packet["source_clock_pair_span_s"] >= 0.
    assert packet["read_only"] is True
    assert "telemetry" not in packet


def test_clock_pair_reports_midpoint_and_scheduler_preemption(monkeypatch):
    monotonic_values = iter((10., 10.06))
    monkeypatch.setattr(protocol.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(protocol.time, "time", lambda: 100.)

    wall, monotonic, span_s = protocol.sample_clock_pair()

    assert wall == 100.
    assert monotonic == pytest.approx(10.03)
    assert span_s == pytest.approx(.06)
