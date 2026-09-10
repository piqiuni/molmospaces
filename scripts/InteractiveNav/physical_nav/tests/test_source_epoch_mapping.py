"""Cross-host source time must not hide camera/codec/network backlog."""

from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_sensor_ros_bridge as sensor
from physical_sensor_ros_bridge import _SourceEpochMapper
from test_sensor_stamp_rebase import _bridge


GENERATION = ("sensor-session", 1)


def _observe_offsets(
    mapper,
    offsets,
    *,
    generation=GENERATION,
    source_start=100.,
    monotonic_start=10.,
):
    result = None
    for index, offset in enumerate(offsets):
        source = source_start + index * .01
        result = mapper.observe(
            source,
            source + offset,
            monotonic_start + index * .01,
            generation,
        )
    return result


def _prime(mapper, *, generation=GENERATION, offset=5.012):
    return _observe_offsets(
        mapper,
        [offset + .008, offset, offset + .004],
        generation=generation,
    )


def _prime_with_source_monotonic(mapper, *, offset=5.):
    result = None
    for index in range(3):
        source = 100. + index * .01
        result = mapper.observe(
            source,
            source + offset,
            10. + index * .01,
            GENERATION,
            20. + index * .01,
        )
    return result


def test_three_small_startup_packets_establish_lower_envelope_epoch():
    mapper = _SourceEpochMapper()

    first = _observe_offsets(mapper, [5.020])
    first_projected, first_mapping = mapper.project(101., GENERATION)
    second = _observe_offsets(
        mapper, [5.012], source_start=100.01, monotonic_start=10.01,
    )
    third = _observe_offsets(
        mapper, [5.016], source_start=100.02, monotonic_start=10.02,
    )
    projected, mapping = mapper.project(101., GENERATION)

    assert first["phase"] == "warming"
    assert second["phase"] == "warming"
    assert first_projected == 101.
    assert first_mapping["valid"] is False
    assert third["phase"] == "ready"
    assert third["observation_count"] == 3
    assert third["selected_offset_s"] == pytest.approx(5.012)
    assert projected == pytest.approx(106.012)
    assert mapping["valid"] is True
    assert mapping["method"] == "small_packet_one_way_offset"
    assert mapping["one_way_delay_included"] is True


def test_startup_requires_temporally_distinct_clock_samples():
    mapper = _SourceEpochMapper(min_observation_span_s=.02)

    for index in range(3):
        diagnostics = mapper.observe(
            100. + index * .001,
            105. + index * .001,
            10. + index * .001,
            GENERATION,
        )
    assert diagnostics["phase"] == "warming"
    assert diagnostics["observation_count"] == 3
    assert diagnostics["observation_span_s"] == pytest.approx(.002)

    diagnostics = mapper.observe(100.02, 105.02, 10.02, GENERATION)
    assert diagnostics["phase"] == "ready"
    assert diagnostics["observation_span_s"] == pytest.approx(.02)


def test_preempted_clock_pair_is_rejected_without_defining_generation():
    mapper = _SourceEpochMapper(clock_pair_max_span_s=.005)

    rejected = mapper.observe(
        100., 105., 10., GENERATION, 20., .060, .001,
    )

    assert rejected["clock_pair_rejected_count"] == 1
    assert rejected["clock_pair_suspect"] is True
    assert rejected["generation"] is None
    assert rejected["observation_count"] == 0

    for index in range(3):
        diagnostics = mapper.observe(
            100.1 + index * .01,
            105.1 + index * .01,
            10.1 + index * .01,
            GENERATION,
            20.1 + index * .01,
            .001,
            .001,
        )
    assert diagnostics["valid"] is True


def test_bad_receiver_clock_pair_temporarily_gates_but_cannot_reset_epoch():
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)

    rejected = mapper.observe(
        100.1, 105.1, 10.1, GENERATION, 20.1, .001, .060,
    )
    _, gated = mapper.project(100.1, GENERATION)
    recovered = mapper.observe(
        100.11, 105.11, 10.11, GENERATION, 20.11, .001, .001,
    )

    assert rejected["phase"] == "clock_pair_rejected"
    assert gated["valid"] is False
    assert recovered["valid"] is True
    assert recovered["clock_step_count"] == 0
    assert recovered["mapping_epoch"] == 1


@pytest.mark.parametrize("offset", [-8.25, 6.75])
def test_positive_and_negative_cross_host_offsets_are_projected(offset):
    mapper = _SourceEpochMapper()
    _prime(mapper, offset=offset)

    projected, mapping = mapper.project(101., GENERATION)

    assert projected == pytest.approx(101. + offset)
    assert mapping["selected_offset_s"] == pytest.approx(offset)


def test_sensor_receipt_is_not_an_offset_observation_so_backlog_stays_visible():
    mapper = _SourceEpochMapper()
    _prime(mapper)

    projected, _mapping = mapper.project(101., GENERATION)
    sensor_received_wall = 106.302

    # The 290 ms camera/codec/network age remains measurable. Mapping the
    # sensor receipt itself would incorrectly make this old frame look current.
    assert sensor_received_wall - projected == pytest.approx(.29)
    assert mapper.snapshot()["observation_count"] == 3


def test_sustained_network_backlog_does_not_replace_connection_baseline():
    mapper = _SourceEpochMapper(window_s=1.)
    _prime(mapper, offset=5.)

    diagnostics = mapper.observe(102., 107.5, 12.484, GENERATION)
    projected, _mapping = mapper.project(102.1, GENERATION)

    assert diagnostics["minimum_observed_offset_s"] == pytest.approx(5.)
    assert diagnostics["recent_minimum_observed_offset_s"] == pytest.approx(5.5)
    assert diagnostics["selected_offset_s"] == pytest.approx(5.)
    assert projected == pytest.approx(107.1)


@pytest.mark.parametrize("drift_ppm", [-100., 100.])
def test_source_epoch_tracks_one_hour_bidirectional_clock_drift(drift_ppm):
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)
    source = 100.02
    received = 105.02
    receiver_monotonic = 10.02
    source_monotonic = 20.02
    rate = drift_ppm * 1e-6

    for _minute in range(60):
        receiver_monotonic += 60.
        received += 60.
        source_elapsed = 60. * (1. - rate)
        source += source_elapsed
        source_monotonic += source_elapsed
        diagnostics = mapper.observe(
            source,
            received,
            receiver_monotonic,
            GENERATION,
            source_monotonic,
        )

    expected_offset = 5. + rate * 3600.
    assert diagnostics["selected_offset_s"] == pytest.approx(
        expected_offset, abs=1e-8,
    )
    assert diagnostics["estimated_rate_ppm"] == pytest.approx(
        drift_ppm, abs=.01,
    )
    assert abs(diagnostics["estimated_rate_ppm"]) <= diagnostics[
        "max_adjust_rate_ppm"
    ]


def test_source_epoch_holdover_preserves_drift_without_learning_backlog():
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)
    source = 100.02
    received = 105.02
    receiver_monotonic = 10.02
    source_monotonic = 20.02
    rate = 100e-6

    # Establish a drift estimate before transport becomes persistently slow.
    for _minute in range(2):
        receiver_monotonic += 60.
        received += 60.
        source_elapsed = 60. * (1. - rate)
        source += source_elapsed
        source_monotonic += source_elapsed
        mapper.observe(
            source, received, receiver_monotonic, GENERATION, source_monotonic,
        )

    for backlog_minute in range(10):
        receiver_monotonic += 60.
        received += 60.
        source_elapsed = 60. * (1. - rate)
        source += source_elapsed
        source_monotonic += source_elapsed
        diagnostics = mapper.observe(
            source,
            received + .4,
            receiver_monotonic + .4,
            GENERATION,
            source_monotonic,
        )

    expected_clock_offset = 5. + rate * 720.
    assert diagnostics["selected_offset_s"] == pytest.approx(
        expected_clock_offset, abs=1e-4,
    )
    assert diagnostics["estimated_queue_excess_s"] > .39
    assert diagnostics["rejected_anchor_count"] >= 1
    assert diagnostics["holdover_age_s"] > 0.


def test_source_monotonic_prevents_hol_delay_from_resetting_epoch():
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)

    delayed = mapper.observe(101., 115., 20., GENERATION, 21.)
    recovered = mapper.observe(110.01, 115.01, 20.01, GENERATION, 30.01)

    assert delayed["valid"] is True
    assert recovered["valid"] is True
    assert recovered["clock_step_count"] == 0
    assert recovered["mapping_epoch"] == 1


def test_source_monotonic_detects_wall_step_without_network_timing_guess():
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)

    suspect = mapper.observe(110.1, 105.1, 10.1, GENERATION, 20.1)
    reset = mapper.observe(110.11, 105.11, 10.11, GENERATION, 20.11)

    assert suspect["valid"] is False
    assert suspect["phase"] == "source_phase_suspect"
    assert suspect["clock_step_count"] == 0
    assert reset["valid"] is False
    assert reset["phase"] == "warming"
    assert reset["clock_step_count"] == 1
    assert reset["last_reset_reason"] == "source_wall_step"
    assert reset["observation_count"] == 1


@pytest.mark.parametrize("wall_step_s", [-.4, .4])
def test_paired_source_clock_detects_subsecond_wall_steps(wall_step_s):
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)

    suspect = mapper.observe(
        100.1 + wall_step_s, 105.1, 10.1, GENERATION, 20.1,
    )
    reset = mapper.observe(
        100.11 + wall_step_s, 105.11, 10.11, GENERATION, 20.11,
    )

    assert suspect["valid"] is False
    assert suspect["source_phase_suspect"] is True
    assert reset["valid"] is False
    assert reset["clock_step_count"] == 1
    assert reset["last_reset_reason"] == "source_wall_step"


def test_single_corrupt_paired_source_stamp_pauses_without_resetting_epoch():
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)

    corrupt = mapper.observe(110.1, 105.1, 10.1, GENERATION, 20.1)
    recovered = mapper.observe(100.11, 105.11, 10.11, GENERATION, 20.11)

    assert corrupt["valid"] is False
    assert corrupt["source_phase_suspect"] is True
    assert recovered["valid"] is True
    assert recovered["source_phase_suspect"] is False
    assert recovered["clock_step_count"] == 0
    assert recovered["mapping_epoch"] == 1


def test_just_over_threshold_phase_outlier_cannot_confirm_against_normal_phase():
    mapper = _SourceEpochMapper()
    _prime_with_source_monotonic(mapper)

    corrupt = mapper.observe(100.16, 105.1, 10.1, GENERATION, 20.1)
    recovered = mapper.observe(100.11, 105.11, 10.11, GENERATION, 20.11)

    assert corrupt["valid"] is False
    assert recovered["valid"] is True
    assert recovered["clock_step_count"] == 0
    assert recovered["mapping_epoch"] == 1


def test_legacy_interval_mismatch_gates_but_never_guesses_clock_step():
    mapper = _SourceEpochMapper()
    _prime(mapper, offset=5.)

    suspect = mapper.observe(101., 115., 20., GENERATION)
    recovered = mapper.observe(110., 115., 20., GENERATION)

    assert suspect["valid"] is False
    assert suspect["phase"] == "legacy_interval_suspect"
    assert suspect["clock_step_count"] == 0
    assert recovered["valid"] is True
    assert recovered["clock_step_count"] == 0


def test_project_is_pure_and_lower_delay_probe_removes_startup_bias_immediately():
    mapper = _SourceEpochMapper(max_adjust_rate_s_per_s=.05)
    # Local wall and monotonic are both sampled at receipt and must advance by
    # the same amount even while one-way delay changes.
    mapper.observe(100., 105.5, 10., GENERATION)
    mapper.observe(100.2, 105.6, 10.1, GENERATION)
    mapper.observe(100.4, 105.7, 10.2, GENERATION)
    before = mapper.snapshot()

    assert mapper.project(101., GENERATION)[0] == pytest.approx(106.3)
    assert mapper.project(102., GENERATION)[0] == pytest.approx(107.3)
    assert mapper.snapshot() == before

    # A lower-delay probe is direct evidence that the startup envelope carried
    # 300 ms of avoidable queueing. It must not take minutes of ppm-limited
    # correction to recover; only positive motion remains rate limited.
    diagnostics = mapper.observe(102.7, 107.7, 12.2, GENERATION)
    projected, mapping = mapper.project(103., GENERATION)

    assert diagnostics["minimum_observed_offset_s"] == pytest.approx(5.)
    assert diagnostics["selected_offset_s"] == pytest.approx(5.)
    assert diagnostics["selected_minus_minimum_s"] == pytest.approx(0.)
    assert diagnostics["lower_reanchor_count"] == 1
    assert projected == pytest.approx(108.)
    assert mapping["phase"] == "ready"


def test_new_transport_generation_warms_an_independent_epoch_and_retires_old():
    mapper = _SourceEpochMapper()
    _prime(mapper, offset=5.)
    assert mapper.project(101., GENERATION)[0] == pytest.approx(106.)

    newer = ("sensor-session", 2)
    _observe_offsets(
        mapper, [100.], generation=newer, source_start=10., monotonic_start=20.,
    )
    warming_value, warming = mapper.project(11., newer)
    assert warming_value == 11.
    assert warming["valid"] is False
    _observe_offsets(
        mapper, [100., 100.], generation=newer,
        source_start=10.01, monotonic_start=20.01,
    )

    assert mapper.project(11., newer)[0] == pytest.approx(111.)
    old_projected, old_mapping = mapper.project(102., GENERATION)
    assert old_projected == 102.
    assert old_mapping["valid"] is False

    stale = mapper.observe(103., 108., 21., GENERATION)
    assert stale["generation"] == newer
    assert stale["stale_generation_count"] == 1
    assert mapper.project(11., newer)[0] == pytest.approx(111.)


def test_retired_different_session_cannot_reclaim_mapper_generation():
    mapper = _SourceEpochMapper()
    _prime(mapper)
    newer = ("replacement-session", 1)
    _prime(mapper, generation=newer, offset=7.)

    stale = mapper.observe(110., 115., 20., GENERATION)

    assert stale["generation"] == newer
    assert stale["stale_generation_count"] == 1
    assert mapper.project(101., newer)[0] == pytest.approx(108.)


@pytest.mark.parametrize(
    "source,received,monotonic,reason",
    [
        (90., 106.016, 11., "source_wall_step"),
        (101., 107., 11., "host_wall_step"),
    ],
)
def test_wall_clock_step_reenters_warmup_without_reusing_old_epoch(
    source, received, monotonic, reason,
):
    mapper = _SourceEpochMapper()
    _prime(mapper)

    reset = mapper.observe(source, received, monotonic, GENERATION)
    if reason == "host_wall_step":
        assert reset["valid"] is False
        assert reset["phase"] == "host_phase_suspect"
        assert reset["clock_step_count"] == 0
        source += .01
        received += .01
        monotonic += .01
        reset = mapper.observe(source, received, monotonic, GENERATION)
    projected, mapping = mapper.project(source + .1, GENERATION)

    assert reset["valid"] is False
    assert reset["observation_count"] == 1
    assert reset["clock_step_count"] == 1
    assert reset["last_reset_reason"] == reason
    assert projected == pytest.approx(source + .1)
    assert mapping["valid"] is False

    mapper.observe(source + .01, received + .01, monotonic + .01, GENERATION)
    ready = mapper.observe(
        source + .02, received + .02, monotonic + .02, GENERATION,
    )
    assert ready["valid"] is True
    assert ready["mapping_epoch"] == 2


def test_out_of_order_probe_is_ignored_without_changing_selected_epoch():
    mapper = _SourceEpochMapper()
    _prime(mapper)
    before = mapper.snapshot()

    stale = mapper.observe(100.03, 105.03, 10.01, GENERATION)

    assert stale["out_of_order_count"] == 1
    assert stale["observation_count"] == before["observation_count"]
    assert stale["selected_offset_s"] == before["selected_offset_s"]


def test_ros_stamp_normalizer_uses_ready_epoch_then_preserves_source_age():
    bridge = _bridge()
    bridge._source_epoch_mapper = _SourceEpochMapper()
    packet = {
        "_transport_session": GENERATION[0],
        "_transport_connection": GENERATION[1],
    }
    _prime(bridge._source_epoch_mapper, offset=5.02)

    ros_stamp, rebased = bridge._normalise_ros_capture_stamp(packet, 101.)

    assert ros_stamp == pytest.approx(106.02)
    assert rebased is False
    mapping = bridge._last_stamp_mapping
    assert mapping["source_capture_stamp"] == 101.
    assert mapping["source_epoch_projected_stamp"] == pytest.approx(106.02)
    assert mapping["source_epoch_mapping"]["valid"] is True
    assert mapping["monotonic_floor_applied"] is False


def test_ready_epoch_uses_small_monotonic_floor_instead_of_host_now_anchor():
    bridge = _bridge()
    bridge._source_epoch_mapper = _SourceEpochMapper()
    packet = {
        "_transport_session": GENERATION[0],
        "_transport_connection": GENERATION[1],
    }
    _prime(bridge._source_epoch_mapper, offset=5.)
    first, _ = bridge._normalise_ros_capture_stamp(packet, 101.)

    second, rebased = bridge._normalise_ros_capture_stamp(packet, 100.5)

    assert second == pytest.approx(first + 1e-4)
    assert rebased is False
    assert bridge._last_stamp_mapping["monotonic_floor_applied"] is True
    assert bridge._last_stamp_mapping["source_regression_detected"] is True
    assert bridge._stamp_rebase_count == 0


def test_negative_offset_warmup_does_not_seed_ready_stamp_baselines():
    bridge = _bridge()
    bridge._source_epoch_mapper = _SourceEpochMapper()
    packet = {
        "_transport_session": GENERATION[0],
        "_transport_connection": GENERATION[1],
        "_host_received_wall": 95.1,
    }
    bridge._source_epoch_mapper.observe(100., 95., 10., GENERATION)

    warming, warming_rebased = bridge._normalise_ros_capture_stamp(packet, 100.)

    assert warming == pytest.approx(95.1)
    assert warming_rebased is False
    assert bridge._stamp_generation is None
    assert bridge._last_source_capture_stamp == -float("inf")
    assert bridge._last_ros_capture_stamp == -float("inf")
    assert bridge._stamp_offset_sec == 0.

    bridge._source_epoch_mapper.observe(100.01, 95.01, 10.01, GENERATION)
    bridge._source_epoch_mapper.observe(100.02, 95.02, 10.02, GENERATION)
    ready, ready_rebased = bridge._normalise_ros_capture_stamp(packet, 101.)

    assert ready == pytest.approx(96.)
    assert ready_rebased is False
    assert bridge._last_stamp_mapping["monotonic_floor_applied"] is False
    assert bridge._last_stamp_mapping["generation_changed"] is True


def test_mapper_clock_epoch_reset_never_leaves_legacy_offset_or_floor_storm():
    bridge = _bridge()
    bridge._source_epoch_mapper = _SourceEpochMapper()
    packet = {
        "_transport_session": GENERATION[0],
        "_transport_connection": GENERATION[1],
        "_host_received_wall": 106.1,
    }
    _prime(bridge._source_epoch_mapper, offset=5.)
    first, _ = bridge._normalise_ros_capture_stamp(packet, 101.)
    assert first == pytest.approx(106.)

    reset = bridge._source_epoch_mapper.observe(
        90., 106.004, 11., GENERATION,
    )
    assert reset["valid"] is False
    warming, warming_rebased = bridge._normalise_ros_capture_stamp(packet, 90.)
    assert warming == pytest.approx(106.1)
    assert warming_rebased is False
    assert bridge._stamp_offset_sec == 0.
    assert bridge._last_source_capture_stamp == 101.
    assert bridge._last_ros_capture_stamp == pytest.approx(106.)

    bridge._source_epoch_mapper.observe(90.01, 106.014, 11.01, GENERATION)
    bridge._source_epoch_mapper.observe(90.02, 106.024, 11.02, GENERATION)
    packet["_host_received_wall"] = 106.2
    recovered, recovered_rebased = bridge._normalise_ros_capture_stamp(
        packet, 90.1,
    )

    assert recovered == pytest.approx(106.104)
    assert recovered_rebased is False
    assert bridge._stamp_offset_sec == 0.
    mapping = bridge._last_stamp_mapping
    assert mapping["source_epoch_changed"] is True
    assert mapping["monotonic_floor_applied"] is False
    assert mapping["epoch_warming_receipt_fallback"] is False


def test_invalid_source_uses_local_receipt_without_applying_epoch_twice():
    bridge = _bridge()
    bridge._source_epoch_mapper = _SourceEpochMapper()
    _prime(bridge._source_epoch_mapper, offset=5.)
    packet = {
        "_transport_session": GENERATION[0],
        "_transport_connection": GENERATION[1],
        "_host_received_wall": 200.,
    }

    stamp, rebased = bridge._normalise_ros_capture_stamp(packet, float("nan"))

    assert stamp == 200.
    assert rebased is False
    mapping = bridge._last_stamp_mapping
    assert mapping["source_stamp_fallback"] is True
    assert mapping["source_epoch_projected_stamp"] == 200.
    assert mapping["source_epoch_mapping"]["valid"] is False
    assert mapping["source_epoch_mapping"]["method"] == "host_receipt_fallback"


def _publish_harness(mapper):
    bridge = _bridge()
    bridge.args = SimpleNamespace(camera_frame="camera", telemetry_max_delta_sec=.15)
    bridge.last_stamp = float("-inf")
    bridge.last_seq = -1
    bridge._last_published_bridge_seq = -1
    bridge._last_calibration_key = None
    bridge._packet_lock = threading.Lock()
    bridge._source_epoch_mapper = mapper
    bridge._decode_executor = SimpleNamespace(
        submit=lambda _fn, data, *_args: SimpleNamespace(result=lambda: data)
    )
    bridge._transport_generation_is_current = lambda _packet: True
    bridge._camera_imu_enabled = False
    bridge._camera_imu_use_yaw = False
    bridge.rgb_pub = SimpleNamespace(publish=lambda _msg: None)
    bridge.depth_pub = SimpleNamespace(publish=lambda _msg: None)
    bridge.info_pub = SimpleNamespace(publish=lambda _msg: None)
    bridge.depth_info_pub = SimpleNamespace(publish=lambda _msg: None)
    bridge.calibration_pub = SimpleNamespace(publish=lambda _msg: None)
    bridge.telemetry_at = lambda _stamp: ({"position": [0., 0., 0.], "yaw": 0.}, 0.)
    bridge._camera_transforms = lambda *_args, **_kwargs: []
    telemetry_updates, poses, clouds, contexts = [], [], [], []
    bridge.update_telemetry = lambda *args, **kwargs: telemetry_updates.append((args, kwargs))
    bridge.publish_pose = lambda *args, **kwargs: poses.append((args, kwargs)) or "published"
    bridge._queue_cloud = lambda *args, **kwargs: clouds.append((args, kwargs))
    bridge.publish_capture_context = lambda *args, **kwargs: contexts.append((args, kwargs))
    return bridge, telemetry_updates, poses, clouds, contexts


def _sensor_packet(*, stamp=101., received_wall=106.2):
    intrinsics = {
        "width": 2, "height": 2, "fx": 1., "fy": 1., "cx": .5, "cy": .5,
    }
    return {
        "type": "sensor_frame",
        "seq": 1,
        "_bridge_seq": 1,
        "stamp": stamp,
        "_transport_session": GENERATION[0],
        "_transport_connection": GENERATION[1],
        "_host_received_wall": received_wall,
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth": np.ones((2, 2), dtype=np.uint16),
        "rgb_intrinsics": intrinsics,
        "depth_intrinsics": intrinsics,
        "telemetry": {"position": [0., 0., 0.], "yaw": 0.},
        "depth_scale": .001,
    }


def test_epoch_warmup_publishes_images_but_not_pose_or_world_cloud(monkeypatch):
    mapper = _SourceEpochMapper()
    mapper.observe(100., 105., 10., GENERATION)
    bridge, telemetry_updates, poses, clouds, contexts = _publish_harness(mapper)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *_args: None)
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *_args: None)

    bridge.publish(_sensor_packet())

    assert telemetry_updates == []
    assert poses == []
    assert clouds == []
    timing = contexts[0][1]["capture_timing"]
    assert contexts[0][1]["tf_status"] == "source_epoch_warming"
    assert timing["source_epoch_mapping"]["gate_ready"] is False
    assert timing["source_epoch_mapping"]["gate_reason"] == "warming"


def test_invalid_wire_stamp_uses_receipt_once_and_cannot_publish_world_cloud(monkeypatch):
    mapper = _SourceEpochMapper()
    _prime(mapper, offset=5.)
    bridge, telemetry_updates, poses, clouds, contexts = _publish_harness(mapper)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *_args: None)
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *_args: None)

    bridge.publish(_sensor_packet(stamp=float("nan"), received_wall=200.))

    assert telemetry_updates == []
    assert poses == []
    assert clouds == []
    timing = contexts[0][1]["capture_timing"]
    assert timing["source_packet_stamp"] is None
    assert timing["source_packet_stamp_valid"] is False
    assert timing["ros_capture_stamp"] == 200.
    assert timing["source_to_ros_offset_s"] is None
    assert timing["host_transport_age_s"] == 0.
    assert timing["source_epoch_mapping"]["gate_reason"] == "warming"


def test_implausibly_future_projection_is_fail_closed(monkeypatch):
    mapper = _SourceEpochMapper()
    _prime(mapper, offset=10.)
    bridge, telemetry_updates, poses, clouds, contexts = _publish_harness(mapper)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *_args: None)
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *_args: None)

    bridge.publish(_sensor_packet(received_wall=105.))

    assert telemetry_updates == []
    assert poses == []
    assert clouds == []
    timing = contexts[0][1]["capture_timing"]
    assert contexts[0][1]["tf_status"] == "source_epoch_projected_capture_in_future"
    assert timing["host_transport_age_s"] == pytest.approx(-6.)
    assert timing["source_epoch_mapping"]["gate_reason"] == "projected_capture_in_future"


def test_ready_plausible_epoch_allows_capture_pose_and_cloud(monkeypatch):
    mapper = _SourceEpochMapper()
    _prime(mapper, offset=5.)
    bridge, telemetry_updates, poses, clouds, contexts = _publish_harness(mapper)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *_args: None)
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *_args: None)

    bridge.publish(_sensor_packet())

    assert len(telemetry_updates) == len(poses) == len(clouds) == 1
    assert contexts[0][1]["tf_status"] == "published"
    timing = contexts[0][1]["capture_timing"]
    assert timing["ros_capture_stamp"] == pytest.approx(106.)
    assert timing["host_transport_age_s"] == pytest.approx(.2)
    assert timing["source_epoch_mapping"]["gate_ready"] is True


def test_stamp_rollback_keeps_world_geometry_gated_until_real_clock_catches_up(
    monkeypatch,
):
    mapper = _SourceEpochMapper()
    _prime(mapper, offset=5.)
    bridge, telemetry_updates, poses, clouds, contexts = _publish_harness(mapper)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *_args: None)
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *_args: None)

    for bridge_seq, (stamp, received_wall) in enumerate(
        ((101., 106.2), (100.8, 106.3), (100.9, 106.4), (101.1, 106.5)),
        start=1,
    ):
        packet = _sensor_packet(stamp=stamp, received_wall=received_wall)
        packet["seq"] = bridge_seq
        packet["_bridge_seq"] = bridge_seq
        bridge.publish(packet)

    # Only the initial capture and the first real projected stamp above the
    # previous watermark may reach pose/cloud consumers.  Both intermediate
    # frames remain image-only even though only the first is a raw regression.
    assert len(telemetry_updates) == len(poses) == len(clouds) == 2
    assert [context[1]["tf_status"] for context in contexts] == [
        "published",
        "source_epoch_source_clock_regression",
        "source_epoch_capture_stamp_floor",
        "published",
    ]
    assert contexts[1][1]["capture_timing"]["source_epoch_mapping"][
        "gate_ready"
    ] is False
    assert contexts[2][1]["capture_timing"]["source_epoch_mapping"][
        "gate_reason"
    ] == "capture_stamp_floor"
