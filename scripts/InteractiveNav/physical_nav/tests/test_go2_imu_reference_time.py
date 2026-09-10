"""Reference capture requires continuous stable elapsed time, not 250 polls."""

import math

import pytest

from test_go2_imu_estimator import _source
from go2_readonly_sensor_bridge import (
    _IMU_REFERENCE_MAX_ACCEL_ERROR_M_S2,
    _IMU_REFERENCE_MAX_GYRO_RAD_S,
    _IMU_REFERENCE_MAX_TILT_SPAN_RAD,
    _STANDARD_GRAVITY_M_S2,
)


def _sample(seconds, roll=1., pitch=.1, scale=9.81, gyro=None):
    motion = {"accel_timestamp_ms": 1000. + seconds*1000.,
              "accelerometer": [-scale*math.sin(pitch),
                                scale*math.sin(roll)*math.cos(pitch),
                                scale*math.cos(roll)*math.cos(pitch)]}
    if gyro is not None:
        motion.update(gyroscope=gyro, gyro_timestamp_ms=motion["accel_timestamp_ms"])
    return motion


@pytest.mark.parametrize("hz", [10, 20, 50, 100, 200])
def test_stable_reference_ready_at_same_elapsed_time_without_filter_startup_bias(hz):
    source = _source(reference=None)
    for index in range(math.ceil(2.5*hz)+1):
        elapsed = index/hz
        motion = _sample(elapsed)
        assert source._update_imu_orientation(motion)
        if elapsed < 2.5:
            assert motion["imu_calibrating"]
    assert not motion["imu_calibrating"]
    assert source._imu_reference_rpy[:2] == pytest.approx([1., .1], abs=1e-10)
    assert motion["correction_rpy"] == pytest.approx([0., 0., 0.], abs=1e-10)
    # A locked reference is not continually recalibrated while walking.
    reference = list(source._imu_reference_rpy)
    source._update_imu_orientation(_sample(2.6, roll=1.2))
    assert source._imu_reference_rpy == reference
    assert source._imu_rpy[0] > 1.


def test_irregular_continuous_reference_window_uses_elapsed_time():
    source = _source(reference=None)
    for stamp in [0., .1, .3, .35, .6, .8, 1., 1.1, 1.3, 1.5, 1.7, 1.8, 2., 2.2, 2.4]:
        source._update_imu_orientation(_sample(stamp))
        assert source._imu_reference_rpy is None
    final = _sample(2.5)
    source._update_imu_orientation(final)
    assert not final["imu_calibrating"]


@pytest.mark.parametrize("kind", ["sway", "rotation", "acceleration"])
def test_motion_does_not_lock_reference_and_quiet_observations_recover(kind):
    source = _source(reference=None)
    for index in range(400):
        motion = _sample(index*.01,
                         roll=1. + (.15*math.sin(index*.2) if kind == "sway" else 0.),
                         scale=12. if kind == "acceleration" else 9.81,
                         gyro=[.4, 0., 0.] if kind == "rotation" else None)
        source._update_imu_orientation(motion)
    assert source._imu_reference_rpy is None
    for index in range(251):
        motion = _sample(4.+index*.01, gyro=[0., 0., 0.])
        source._update_imu_orientation(motion)
    assert not motion["imu_calibrating"]
    assert source._imu_reference_rpy[:2] == pytest.approx([1., .1], abs=1e-10)


def test_missing_interval_is_not_counted_as_stationary_evidence():
    source = _source(reference=None)
    for index in range(201):
        source._update_imu_orientation(_sample(index*.01))
    source._update_imu_orientation(_sample(10.))
    assert source._imu_reference_rpy is None
    for index in range(1, 251):
        motion = _sample(10.+index*.01)
        source._update_imu_orientation(motion)
    assert not motion["imu_calibrating"]


def test_gyro_only_rotation_invalidates_pending_reference_without_advancing_it():
    source = _source(reference=None)
    for index in range(201):
        source._update_imu_orientation(_sample(index*.01))
    source._update_imu_orientation({"gyro_timestamp_ms": 3050., "gyroscope": [.4, 0., 0.]})
    for index in range(206, 455):
        source._update_imu_orientation(_sample(index*.01))
    assert source._imu_reference_rpy is None
    final = _sample(4.56)
    source._update_imu_orientation(final)
    assert not final["imu_calibrating"]


def test_high_frequency_short_burst_is_not_enough_evidence():
    source = _source(reference=None)
    for index in range(1000):
        source._update_imu_orientation(_sample(index*.001))
    assert source._imu_reference_rpy is None


def test_realistic_small_noise_and_timestamp_jitter_can_still_calibrate():
    source = _source(reference=None)
    elapsed = 0.
    index = 0
    while elapsed < 2.51:
        elapsed += (.009 if index % 3 == 0 else .011 if index % 3 == 1 else .010)
        roll = 1. + .004*math.sin(index*.37)
        pitch = .1 + .003*math.cos(index*.21)
        scale = _STANDARD_GRAVITY_M_S2 + .08*math.sin(index*.11)
        motion = _sample(elapsed, roll=roll, pitch=pitch, scale=scale,
                         gyro=[.005, -.004, .003])
        source._update_imu_orientation(motion)
        index += 1
    assert not motion["imu_calibrating"]
    assert motion["imu_calibration_status"] == "calibrated"


@pytest.mark.parametrize("kind,inside", [
    ("accel", True), ("accel", False), ("gyro", True), ("gyro", False),
    ("tilt", True), ("tilt", False),
])
def test_stationary_threshold_boundary_is_explicit(kind, inside):
    source = _source(reference=None)
    epsilon = 1e-4
    for index in range(251):
        scale, gyro, roll = _STANDARD_GRAVITY_M_S2, [0., 0., 0.], 1.
        if kind == "accel":
            scale += _IMU_REFERENCE_MAX_ACCEL_ERROR_M_S2 + (-epsilon if inside else epsilon)
        elif kind == "gyro":
            gyro[0] = _IMU_REFERENCE_MAX_GYRO_RAD_S + (-epsilon if inside else epsilon)
        elif kind == "tilt" and index:
            roll += _IMU_REFERENCE_MAX_TILT_SPAN_RAD + (-epsilon if inside else epsilon)
        motion = _sample(index*.01, roll=roll, pitch=0. if kind == "tilt" else .1,
                         scale=scale, gyro=gyro)
        source._update_imu_orientation(motion)
    assert (not motion["imu_calibrating"]) is inside
