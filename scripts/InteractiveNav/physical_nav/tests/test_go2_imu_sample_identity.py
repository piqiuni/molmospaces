"""Only new hardware samples count toward IMU filtering and calibration."""

import pytest

from test_go2_imu_estimator import _source


def test_repeated_accel_sample_cannot_filter_again_or_finish_calibration():
    source = _source((.2, .1, 0.), reference=None)
    sample = {"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1010.}
    assert source._update_imu_orientation(dict(sample))
    after = list(source._imu_rpy)
    for _ in range(300):
        assert source._update_imu_orientation(dict(sample)) is False
    assert source._imu_rpy == after
    assert source._imu_calibration_samples == 1
    assert source._imu_reference_rpy is None


@pytest.mark.parametrize("repeated_ts", [1020., 1010.])
def test_duplicate_or_late_gyro_cannot_rewind_integration_baseline(repeated_ts):
    source = _source()
    def update(ts):
        return source._update_imu_orientation({"gyroscope": [0., 0., 1.], "gyro_timestamp_ms": ts})
    assert update(1020.)
    assert update(repeated_ts) is False
    assert source._imu_last_gyro_ts == 1020.
    assert update(1030.)
    assert source._imu_rpy[2] == pytest.approx(.03)


def test_accel_and_gyro_have_independent_watermarks_and_counts():
    source = _source(reference=None)
    first = {"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1010.,
             "gyroscope": [0., 0., 1.], "gyro_timestamp_ms": 1010.}
    assert source._update_imu_orientation(dict(first))
    next_gyro = {**first, "gyro_timestamp_ms": 1020.}
    assert source._update_imu_orientation(next_gyro)
    assert source._imu_calibration_samples == 1
    assert next_gyro["imu_valid_updates"] == {"accel": 1, "gyro": 2}
    next_accel = {**first, "accel_timestamp_ms": 1030., "gyro_timestamp_ms": 1020.}
    assert source._update_imu_orientation(next_accel)
    assert source._imu_calibration_samples == 2
    assert next_accel["imu_valid_updates"] == {"accel": 2, "gyro": 2}


@pytest.mark.parametrize("stamp", [None, -1., float("nan"), float("inf"), "bad"])
def test_valid_acceleration_without_valid_sample_time_is_not_a_new_observation(stamp):
    source = _source(reference=None)
    assert not source._update_imu_orientation({"accelerometer": [0., 0., 9.81],
                                              "accel_timestamp_ms": stamp})
    assert source._imu_calibration_samples == 0


def test_late_accel_does_not_contaminate_next_valid_gravity_sample():
    source = _source(reference=None)
    source._update_imu_orientation({"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1010.})
    assert not source._update_imu_orientation({"accelerometer": [0., 9.81, 0.], "accel_timestamp_ms": 1000.})
    assert source._imu_rpy[0] == 0.
    source._update_imu_orientation({"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1020.})
    assert source._imu_rpy[0] == 0.


def test_zero_hardware_timestamp_is_valid_for_first_sample_only():
    source = _source()
    source._imu_last_gyro_ts = None
    sample = {"gyroscope": [0., 0., 1.], "gyro_timestamp_ms": 0.}
    assert source._update_imu_orientation(dict(sample))
    assert not source._update_imu_orientation(dict(sample))
    assert source._update_imu_orientation({**sample, "gyro_timestamp_ms": 10.})
    assert source._imu_rpy[2] == pytest.approx(.01)


def test_duplicate_only_poll_skips_quaternion_and_metadata_work():
    source = _source()
    sample = {"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1010.}
    assert source._update_imu_orientation(dict(sample))
    duplicate = dict(sample)
    assert not source._update_imu_orientation(duplicate)
    assert duplicate == sample
    source._update_imu_orientation({"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1020.})
    assert source._imu_rpy[0] == 0.
