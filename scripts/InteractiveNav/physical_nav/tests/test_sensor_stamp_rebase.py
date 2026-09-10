"""Offline regression tests for source-clock rollback at the ROS bridge boundary."""

import pathlib
import sys
import time
import math


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bridge():
    from physical_sensor_ros_bridge import SensorRosBridge

    bridge = object.__new__(SensorRosBridge)
    bridge._stamp_generation = None
    bridge._stamp_offset_sec = 0.0
    bridge._last_source_capture_stamp = float("-inf")
    bridge._last_ros_capture_stamp = float("-inf")
    bridge._stamp_rebase_count = 0
    bridge._stamp_regression_tolerance_sec = 0.05
    return bridge


def test_normal_source_stream_keeps_capture_stamp_exact():
    bridge = _bridge()
    first, rebased = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor", "_transport_connection": 1},
        1000.0,
    )
    second, second_rebased = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor", "_transport_connection": 1},
        1000.1,
    )
    assert first == 1000.0
    assert second == 1000.1
    assert not rebased
    assert not second_rebased
    assert bridge._stamp_rebase_count == 0


def test_restarted_source_clock_is_rebased_for_ros_consumers():
    bridge = _bridge()
    bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor-old", "_transport_connection": 1},
        1000.0,
    )
    rebased, did_rebase = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor-new", "_transport_connection": 1},
        10.0,
    )
    assert did_rebase
    # The exact host anchor is intentionally not asserted; the key invariant
    # is that the new ROS stamp is newer than the old frame and therefore does
    # not fail GMapping's cloud-age/message-filter checks.
    assert rebased > 1000.0
    assert bridge._stamp_rebase_count == 1
    assert bridge._last_stamp_mapping["source_capture_stamp"] == 10.0
    assert bridge._last_stamp_mapping["ros_stamp"] == rebased
    assert bridge._last_stamp_mapping["rebased"] is True

    next_stamp, next_rebased = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor-new", "_transport_connection": 1},
        10.1,
    )
    assert next_rebased is False
    assert next_stamp > rebased
    assert math.isclose(next_stamp - rebased, 0.1, rel_tol=0.0, abs_tol=1e-6)


def test_healthy_reconnect_does_not_carry_old_generation_offset():
    bridge = _bridge()
    bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor-old", "_transport_connection": 1},
        1000.0,
    )
    # Simulate a prior rollback and its offset.
    bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor-new", "_transport_connection": 1},
        10.0,
    )
    healthy_source = time.time() + 1.0
    healthy, rebased = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor-third", "_transport_connection": 1},
        healthy_source,
    )
    assert rebased is False
    assert healthy == healthy_source


def test_small_same_generation_step_also_keeps_ros_stamp_monotonic():
    bridge = _bridge()
    bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor", "_transport_connection": 1},
        1000.0,
    )
    corrected, rebased = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor", "_transport_connection": 1},
        999.99,
    )
    assert rebased
    assert corrected > 1000.0


def test_telemetry_stamp_baseline_is_independent_from_camera_capture_stream():
    bridge = _bridge()
    bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor", "_transport_connection": 1},
        1000.0,
    )
    telemetry_stamp, telemetry_rebased = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor", "_transport_connection": 1},
        1000.2,
        stream="telemetry",
    )
    camera_stamp, camera_rebased = bridge._normalise_ros_capture_stamp(
        {"_transport_session": "sensor", "_transport_connection": 1},
        1000.1,
    )
    assert telemetry_stamp == 1000.2
    assert camera_stamp == 1000.1
    assert not telemetry_rebased
    assert not camera_rebased
