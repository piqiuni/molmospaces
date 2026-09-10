"""Exercise the Go2 IMU estimator without RealSense, SDK, or hardware."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_readonly_sensor_bridge import D435iSource


def _source(rpy=(0., 0., 0.), reference=(0., 0., 0.)):
    source = object.__new__(D435iSource)
    source._imu_rpy = list(rpy)
    source._imu_reference_rpy = list(reference) if reference is not None else None
    source._imu_last_gyro_ts = 1000.
    source._imu_last_accel_ts = None
    source._imu_calibration_samples = 0
    source._imu_calibration_sum = [0., 0., 0.]
    return source


@pytest.mark.parametrize("rpy", [(0., 0., 0.), (.2, -.3, .4), (0., 0., 1.)])
def test_reported_quaternion_is_unit_and_encodes_reported_rpy(rpy):
    source = _source(rpy)
    motion = {"gyroscope": [0., 0., 0.], "gyro_timestamp_ms": 1010.}
    source._update_imu_orientation(motion)
    q = motion["quaternion"]
    assert sum(v*v for v in q) == pytest.approx(1.)
    # Compare the quaternion's rotated X axis with Rz(yaw) Ry(pitch) Rx(roll).
    x, y, z, w = q
    axis = [1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y)]
    assert np.allclose(axis, [math.cos(rpy[2])*math.cos(rpy[1]),
                              math.sin(rpy[2])*math.cos(rpy[1]), -math.sin(rpy[1])])


@pytest.mark.parametrize("accel", [[float("nan"), 0., 1.], [0., float("inf"), 1.],
                                   [0., 0., 0.], ["bad", 0., 1.]])
def test_invalid_gravity_does_not_poison_state_or_advance_calibration(accel):
    source = _source((.2, .1, .3), reference=None)
    motion = {"accelerometer": accel, "accel_timestamp_ms": 1010.}
    assert source._update_imu_orientation(motion) is False
    assert source._imu_rpy == [.2, .1, .3]
    assert source._imu_calibration_samples == 0
    source._update_imu_orientation({"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1020.})
    assert all(math.isfinite(v) for v in source._imu_rpy)
    assert source._imu_calibration_samples == 1


@pytest.mark.parametrize("gyro,timestamp", [
    ([0., 0., float("nan")], 1010.), ([0., 0., float("inf")], 1010.),
    ([0., 0., 1.], float("nan")), ([0., 0., 1.], "bad"),
])
def test_invalid_gyro_or_clock_does_not_poison_following_valid_updates(gyro, timestamp):
    source = _source((.2, .1, .3))
    assert source._update_imu_orientation({"gyroscope": gyro, "gyro_timestamp_ms": timestamp}) is False
    assert source._imu_rpy == [.2, .1, .3]
    assert source._imu_last_gyro_ts == 1000.
    source._update_imu_orientation({"gyroscope": [0., 0., 1.], "gyro_timestamp_ms": 1020.})
    assert source._imu_rpy[2] == pytest.approx(.32)


def test_good_accel_and_gyro_components_remain_independently_usable():
    source = _source((.2, .1, .3), reference=None)
    assert source._update_imu_orientation({"accelerometer": [float("nan"), 0., 1.], "accel_timestamp_ms": 1020.,
                                           "gyroscope": [0., 0., 1.], "gyro_timestamp_ms": 1020.})
    assert source._imu_rpy == pytest.approx([.2, .1, .32])
    assert source._imu_calibration_samples == 0
    assert source._update_imu_orientation({"accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1040.,
                                           "gyroscope": [0., 0., float("nan")],
                                           "gyro_timestamp_ms": 1040.})
    assert source._imu_rpy == pytest.approx([.196, .098, .32])
    assert source._imu_last_gyro_ts == 1020.
    assert source._imu_calibration_samples == 1


def test_invalid_samples_cannot_complete_reference_calibration():
    source = _source(reference=None)
    for index in range(300):
        assert not source._update_imu_orientation({"accelerometer": [0., 0., 0.], "accel_timestamp_ms": float(index)})
    assert source._imu_reference_rpy is None
    assert source._imu_calibration_samples == 0
    for index in range(251):
        motion = {"accelerometer": [0., 0., 9.81],
                  "accel_timestamp_ms": float(1000 + index*10)}
        assert source._update_imu_orientation(motion)
    assert motion["imu_calibrating"] is False
    assert source._imu_reference_rpy == [0., 0., 0.]


def test_correction_angle_wrap_does_not_create_a_full_turn_jump():
    source = _source((-math.pi + .01, 0., 0.), reference=(math.pi - .01, 0., 0.))
    motion = {"gyroscope": [0., 0., 0.], "gyro_timestamp_ms": 1010.}
    source._update_imu_orientation(motion)
    assert motion["correction_rpy"][0] == pytest.approx(.02)


def test_real_read_does_not_replace_last_motion_with_invalid_measurement():
    from types import SimpleNamespace
    source = _source()
    source.motion_enabled = True
    source.align = None
    source.align_to = "none"
    source.rs = SimpleNamespace(stream=SimpleNamespace(gyro="gyro", accel="accel"))
    previous = {"received_at": 1., "correction_rpy": [0., 0., 0.]}
    source.latest_motion = previous
    vector = SimpleNamespace(x=0., y=0., z=float("nan"))
    motion_frame = SimpleNamespace(get_timestamp=lambda: 1020.,
                                   as_motion_frame=lambda: SimpleNamespace(get_motion_data=lambda: vector))
    image_frame = SimpleNamespace(get_timestamp=lambda: 1020.,
                                  get_data=lambda: np.zeros((2, 2), dtype=np.uint16))
    frames = SimpleNamespace(first_or_default=lambda stream: motion_frame if stream == "gyro" else None,
                             get_color_frame=lambda: image_frame, get_depth_frame=lambda: image_frame)
    source.pipeline = SimpleNamespace(wait_for_frames=lambda timeout: frames)
    source.read()
    assert source.latest_motion is previous
    vector.z = 1.
    source.read()
    assert source.latest_motion is not previous
    assert source.latest_motion["rpy"][2] == pytest.approx(.02)
    accepted = source.latest_motion
    source.read()  # The SDK can return the same motion timestamp again.
    assert source.latest_motion is accepted
