"""Gravity smoothing must have a time response, not a frame-count response."""

import math

import pytest

from test_go2_imu_estimator import _source
from go2_readonly_sensor_bridge import D435iSource


def _step_sample(stamp):
    roll, pitch, g = .2, -.1, 9.81
    return {"accel_timestamp_ms": stamp, "accelerometer": [
        -g*math.sin(pitch), g*math.sin(roll)*math.cos(pitch), g*math.cos(roll)*math.cos(pitch),
    ]}


@pytest.mark.parametrize("hz", [10, 20, 50, 100, 200])
def test_one_second_gravity_response_is_independent_of_valid_sample_rate(hz):
    source = _source()
    source._imu_last_accel_ts = 1000.
    for index in range(1, hz+1):
        assert source._update_imu_orientation(_step_sample(1000.+index*1000./hz))
    fraction = 1. - .98**100  # Preserve the existing response at configured 100 Hz.
    assert source._imu_rpy[:2] == pytest.approx([.2*fraction, -.1*fraction], abs=1e-12)


def test_irregular_but_fresh_sample_intervals_keep_same_elapsed_time_response():
    source = _source()
    source._imu_last_accel_ts = 1000.
    for stamp in [1010., 1040., 1190., 1280., 1480., 1510., 1740., 1980., 2000.]:
        assert source._update_imu_orientation(_step_sample(stamp))
    assert source._imu_rpy[0] == pytest.approx(.2*(1.-.98**100), abs=1e-12)


def test_long_sample_gap_does_not_apply_all_missing_time_as_one_gravity_measurement():
    source = _source()
    source._imu_last_accel_ts = 1000.
    motion = _step_sample(11000.)
    assert source._update_imu_orientation(motion)
    assert source._imu_rpy[:2] == pytest.approx([.004, -.002])
    assert motion["imu_gravity_gap"] is True
    assert motion["imu_gravity_dt_s"] == pytest.approx(.01)
    recovered = _step_sample(11100.)
    assert source._update_imu_orientation(recovered)
    assert recovered["imu_gravity_gap"] is False
    assert recovered["imu_gravity_dt_s"] == pytest.approx(.1)
    assert source._imu_rpy[0] > .004


def test_explicit_gravity_time_constant_changes_response_in_seconds():
    source = _source()
    source._imu_gravity_tau_s = .2
    source._imu_last_accel_ts = 1000.
    for stamp in [1100., 1200.]:
        source._update_imu_orientation(_step_sample(stamp))
    assert source._imu_rpy[0] == pytest.approx(.2*(1.-math.exp(-1.)))


@pytest.mark.parametrize("tau", [0., -1., float("nan"), float("inf")])
def test_invalid_time_constant_is_rejected_before_loading_sensor_sdk(tau):
    with pytest.raises(ValueError, match="finite and positive"):
        D435iSource(848, 480, 10, imu_gravity_tau_s=tau)


def test_cli_rejects_invalid_time_constant_before_any_network_or_hardware_start():
    from pathlib import Path
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "go2_readonly_sensor_bridge.py"
    result = subprocess.run([sys.executable, str(script), "--imu-gravity-tau-s", "nan"],
                            text=True, capture_output=True, timeout=5.)
    assert result.returncode == 2
    assert "--imu-gravity-tau-s must be finite and positive" in result.stderr
