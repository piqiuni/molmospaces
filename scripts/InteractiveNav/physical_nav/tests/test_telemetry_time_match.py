"""Both sensor TF and detector use the same non-blocking capture-time policy."""

from pathlib import Path
import sys
import threading
import json
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_sensor_ros_bridge as sensor
import physical_yoloe_bridge as yolo


def _consumers(history):
    bridge = object.__new__(sensor.SensorRosBridge)
    bridge.args = SimpleNamespace(telemetry_max_delta_sec=.15)
    bridge._telemetry_lock = threading.Lock()
    bridge._telemetry_history = list(history)
    bridge.telemetry = {"position": [99., 0., 0.], "yaw": 0.}
    worker = object.__new__(yolo.YoloeWorker)
    worker._telemetry_lock = threading.Lock()
    worker._telemetry_history = list(history)
    worker._telemetry_max_delta_sec = .15
    return bridge, worker


def test_yolo_does_not_accept_pose_already_too_old_for_sensor_tf():
    bridge, worker = _consumers([(10., {"position": [1., 2., .3], "yaw": 0.})])
    assert bridge.telemetry_at(10.3)[0] == {}
    assert worker._telemetry_at(10.3) == {}


def test_empty_history_does_not_bypass_timestamp_check_with_latest_pose():
    bridge, worker = _consumers([])
    assert bridge.telemetry_at(1000.) == ({}, None)
    assert worker._telemetry_at(1000.) == {}


@pytest.mark.parametrize("offset", [0., .1, .2])
def test_sensor_and_yolo_agree_and_report_matching_delta(offset):
    pose = {"position": [1., 2., .3], "yaw": .2}
    bridge, worker = _consumers([(10., pose)])
    selected, delta = bridge.telemetry_at(10. + offset)
    assert worker._telemetry_at(10. + offset) == selected
    assert worker._last_telemetry_match["delta_sec"] == pytest.approx(abs(offset))
    assert delta == pytest.approx(abs(offset))
    assert worker._last_telemetry_match["accepted"] == (abs(offset) <= .15)


@pytest.mark.parametrize("offset", [-.1, -.2])
def test_sensor_and_yolo_never_use_a_future_pose(offset):
    bridge, worker = _consumers([(10., {"position": [1., 2., .3], "yaw": .2})])
    assert bridge.telemetry_at(10. + offset) == ({}, None)
    assert worker._telemetry_at(10. + offset) == {}
    assert worker._last_telemetry_match["delta_sec"] is None
    assert worker._last_telemetry_match["accepted"] is False


def test_latched_sensor_tolerance_overrides_yolo_startup_value():
    bridge, worker = _consumers([(10., {"yaw": 0.})])
    worker._frame_lock = threading.Lock()
    assert worker._telemetry_at(10.1)
    bridge.args.telemetry_max_delta_sec = .05
    worker._calibration_callback(SimpleNamespace(data=json.dumps({
        "depth_scale": .001, "telemetry_max_delta_sec": .05})))
    assert worker._telemetry_max_delta_sec == .05
    assert bridge.telemetry_at(10.1)[0] == worker._telemetry_at(10.1) == {}
    assert worker._last_telemetry_match["max_delta_sec"] == .05


@pytest.mark.parametrize("limit", [-1., float("nan"), float("inf"), "bad"])
def test_invalid_calibration_tolerance_cannot_relax_existing_policy(limit):
    _, worker = _consumers([])
    worker._frame_lock = threading.Lock()
    worker._calibration_callback(SimpleNamespace(data=json.dumps({
        "depth_scale": .001, "telemetry_max_delta_sec": limit})))
    assert worker._telemetry_max_delta_sec == .15


def test_ties_prefer_past_and_latest_receipt_at_identical_stamp():
    past, future = {"yaw": .1}, {"yaw": .3}
    for history in ([(4., future), (2., past)], [(2., past), (4., future)]):
        assert sensor._nearest_capture_telemetry(history, 3., 1.) == (past, 1.)
    assert sensor._nearest_capture_telemetry([(2., past), (2., future)], 2.) == (future, 0.)


def test_closer_future_sample_cannot_overtake_latest_causal_pose():
    past, future = {"yaw": .1}, {"yaw": .3}
    selected, delta = sensor._nearest_capture_telemetry(
        [(2.8, past), (3.01, future)], 3., 1.,
    )
    assert selected == past
    assert delta == pytest.approx(.2)


@pytest.mark.parametrize("stamp,limit", [(0., .15), (float("nan"), .15),
    (float("inf"), .15), (10., float("nan")), (10., -1.)])
def test_invalid_capture_or_limit_cannot_select_a_pose(stamp, limit):
    assert sensor._nearest_capture_telemetry([(10., {"yaw": 0.})], stamp, limit) == ({}, None)


def test_invalid_history_stamps_do_not_hide_valid_samples_or_change_input():
    pose = {"yaw": .1}
    history = [(float("nan"), {"yaw": 9.}), (10., pose), ("bad", {})]
    selected, delta = sensor._nearest_capture_telemetry(history, 10.)
    assert selected == pose and delta == 0.
    selected["yaw"] = 5.
    assert pose["yaw"] == .1
