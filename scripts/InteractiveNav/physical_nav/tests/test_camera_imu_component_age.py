"""A fresh gyro receipt cannot make held gravity tilt fresh."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_sensor_ros_bridge import _camera_imu_capture_match
from test_go2_imu_estimator import _source


def test_source_retains_independent_receipts_and_recovers_on_new_accel():
    source = _source()
    accel = {"received_at": 10., "accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1010.}
    assert source._update_imu_orientation(accel)
    gyro = {"received_at": 10.5, "gyroscope": [0., 0., .1], "gyro_timestamp_ms": 1500.}
    assert source._update_imu_orientation(gyro)
    assert gyro["component_received_at"] == {"accel": 10., "gyro": 10.5}
    result = _camera_imu_capture_match({"camera_imu": gyro}, 10.5)
    assert result["status"] == "stale" and result["accepted"] is False
    assert result["ages_sec"]["accel"] == .5
    previous = copy.deepcopy(gyro)
    recovered = {**accel, "received_at": 10.51, "accel_timestamp_ms": 1510.}
    assert source._update_imu_orientation(recovered)
    assert _camera_imu_capture_match({"camera_imu": recovered}, 10.51)["accepted"] is True
    assert gyro == previous  # Published snapshots must not alias future receipt mutations.


def test_duplicate_accel_cannot_acquire_new_receipt_from_fresh_gyro():
    source = _source()
    initial = {"received_at": 10., "accelerometer": [0., 0., 9.81], "accel_timestamp_ms": 1010.}
    source._update_imu_orientation(initial)
    mixed = {**initial, "received_at": 10.5, "gyroscope": [0., 0., .1], "gyro_timestamp_ms": 1500.}
    assert source._update_imu_orientation(mixed)
    assert mixed["component_received_at"]["accel"] == 10.
    assert not _camera_imu_capture_match({"camera_imu": mixed}, 10.5)["accepted"]


@pytest.mark.parametrize("use_yaw,accepted", [(False, True), (True, False)])
def test_gyro_freshness_is_required_only_when_yaw_correction_is_enabled(use_yaw, accepted):
    motion = {"imu_calibrating": False, "component_received_at": {"accel": 10., "gyro": 9.}}
    result = _camera_imu_capture_match({"camera_imu": motion}, 10., use_yaw=use_yaw)
    assert result["accepted"] is accepted


def test_recent_frame_number_gap_blocks_correction_until_continuity_recovers():
    stream = {
        "healthy": False, "continuous_s": .1, "effective_hz": 100.,
        "missing": 3, "duplicates": 0, "late": 0,
        "timestamp_regressions": 0,
    }
    motion = {
        "imu_calibrating": False,
        "component_received_at": {"accel": 10., "gyro": 10.},
        "imu_ingress": {"streams": {"accel": stream}},
    }
    rejected = _camera_imu_capture_match({"camera_imu": motion}, 10.)
    assert rejected["status"] == "discontinuous"
    assert rejected["accepted"] is False
    assert rejected["continuity"]["accel"]["missing"] == 3

    stream["healthy"] = True
    stream["continuous_s"] = .6
    recovered = _camera_imu_capture_match({"camera_imu": motion}, 10.)
    assert recovered["status"] == "matched"
    assert recovered["accepted"] is True


def test_yaw_mode_requires_both_ingress_streams_to_be_continuous():
    healthy = {"healthy": True, "continuous_s": 1., "effective_hz": 100.}
    motion = {
        "component_received_at": {"accel": 10., "gyro": 10.},
        "imu_ingress": {"streams": {"accel": healthy}},
    }
    result = _camera_imu_capture_match(
        {"camera_imu": motion}, 10., use_yaw=True,
    )
    assert result["status"] == "invalid_continuity"
    assert result["accepted"] is False


@pytest.mark.parametrize("receipt", [None, 0., float("nan"), float("inf"), "bad"])
def test_bad_component_receipt_cannot_be_replaced_with_overall_received_at(receipt):
    motion = {"received_at": 10., "component_received_at": {"accel": receipt, "gyro": 10.}}
    result = _camera_imu_capture_match({"camera_imu": motion}, 10.)
    assert result["status"] == "invalid_time"
    assert not result["accepted"]


def test_future_component_receipt_outside_tolerance_is_rejected():
    motion = {"component_received_at": {"accel": 10.5, "gyro": 10.}}
    result = _camera_imu_capture_match({"camera_imu": motion}, 10.)
    assert not result["accepted"] and result["ages_sec"]["accel"] == -.5


def test_device_mapped_component_time_takes_precedence_over_late_dequeue_receipt():
    motion = {
        "component_received_at": {"accel": 10.5, "gyro": 10.5},
        "component_sample_at": {"accel": 10.02, "gyro": 10.01},
        "component_sample_time_valid": {"accel": True, "gyro": True},
    }

    result = _camera_imu_capture_match(
        {"camera_imu": motion}, 10., use_yaw=True,
    )

    assert result["accepted"] is True
    assert result["status"] == "matched"
    assert result["time_basis"] == "device_clock_mapped_sample"
    assert result["ages_sec"] == pytest.approx({"accel": -.02, "gyro": -.01})


def test_invalid_device_mapping_falls_back_to_component_receipt_times():
    motion = {
        "component_received_at": {"accel": 10., "gyro": 10.},
        "component_sample_at": {"accel": 20., "gyro": 20.},
        "component_sample_time_valid": {"accel": False, "gyro": False},
    }

    result = _camera_imu_capture_match({"camera_imu": motion}, 10.)

    assert result["accepted"] is True
    assert result["time_basis"] == "source_receipt"
    assert result["ages_sec"]["accel"] == 0.


def test_disabled_and_legacy_correction_are_explicit_not_claimed_fresh():
    assert _camera_imu_capture_match({}, 10., enabled=False)["status"] == "disabled"
    assert _camera_imu_capture_match({}, 10.)["status"] == "unavailable"
    old = {"camera_imu": {"correction_rpy": [0., 0., 0.]}}
    assert _camera_imu_capture_match(old, 10.)["status"] == "legacy_unverified"
    calibration = {"camera_imu": {"imu_calibrating": True}}
    assert _camera_imu_capture_match(calibration, 10.)["status"] == "calibrating"


def test_stale_gravity_isolated_in_actual_sensor_publish_and_yolo_context(monkeypatch):
    import threading
    from types import SimpleNamespace
    import numpy as np
    import physical_sensor_ros_bridge as sensor
    import physical_yoloe_bridge as yolo
    from test_body_pose_consistency import _bridge
    from test_capture_context import _context_worker
    from test_yolo_capture_pose import _worker_and_raw
    bridge, odom, transforms = _bridge(monkeypatch)
    bridge._camera_imu_enabled = True
    bridge.args.telemetry_max_delta_sec = .15
    bridge._last_published_bridge_seq = bridge.last_seq = -1
    bridge.last_stamp = float("-inf")
    bridge._last_calibration_key = None
    bridge._packet_lock = threading.Lock()
    bridge._decode_executor = SimpleNamespace(submit=lambda fn, data, *a: SimpleNamespace(result=lambda: data))
    bridge._transport_generation_is_current = lambda packet: True
    bridge._normalise_ros_capture_stamp = lambda packet, stamp: (stamp, False)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *a: None)
    pose = {"position": [1., 2., .3], "yaw": .2, "camera_imu": {
        "received_at": 10., "correction_rpy": [.1, .1, 0.], "imu_calibrating": False,
        "component_received_at": {"accel": 9., "gyro": 10.},
    }}
    bridge.telemetry_at = lambda stamp: (pose, 0.)
    worker = _context_worker()
    bridge.rgb_pub = SimpleNamespace(publish=worker._rgb_callback)
    bridge.depth_pub = SimpleNamespace(publish=worker._depth_callback)
    bridge.info_pub = bridge.depth_info_pub = bridge.calibration_pub = SimpleNamespace(publish=lambda m: None)
    bridge.capture_context_pub = SimpleNamespace(publish=worker._capture_context_callback)
    clouds = []
    bridge._queue_cloud = lambda *args: clouds.append(args)
    intr = dict(width=2, height=2, fx=100., fy=100., cx=.5, cy=.5)
    bridge.publish(dict(seq=1, _bridge_seq=1, stamp=10., rgb=np.ones((2, 2, 3), dtype=np.uint8),
                        depth=np.ones((2, 2), dtype=np.uint16), rgb_intrinsics=intr, depth_intrinsics=intr))
    raw = worker._latest_raw_frame()
    assert raw["capture_tf_status"] == "stale_camera_imu"
    assert not odom and not transforms and not clouds
    detector, detector_raw = _worker_and_raw()
    detector_raw.update(telemetry=pose, capture_tf_status=raw["capture_tf_status"],
                        camera_imu_match=raw["camera_imu_match"])
    monkeypatch.setattr(yolo, "_geometry_lift_candidate", lambda *a: pytest.fail("stale IMU lifted"))
    report = detector.infer(detector_raw)
    assert report["capture_pose_status"] == "stale_camera_imu"
    assert report["overlay_jpeg"] and report["detections"][0]["geometry_skipped"]
    assert report["camera_imu_match"]["ages_sec"]["accel"] == 1.
    pose["camera_imu"]["component_received_at"]["accel"] = 10.1
    bridge.publish(dict(seq=2, _bridge_seq=2, stamp=10.1, rgb=np.ones((2, 2, 3), dtype=np.uint8),
                        depth=np.ones((2, 2), dtype=np.uint16), rgb_intrinsics=intr, depth_intrinsics=intr))
    recovered = worker._latest_raw_frame()
    assert recovered["capture_tf_status"] == "published"
    assert recovered["camera_imu_match"]["accepted"] is True
    assert len(clouds) == len(odom) == 1


@pytest.mark.parametrize("accel_receipt,expected_camera", [(9., False), (10., True), (None, False)])
def test_periodic_tf_checks_source_component_age_without_stopping_body(monkeypatch, accel_receipt, expected_camera):
    from test_body_pose_consistency import _bridge
    bridge, odom, transforms = _bridge(monkeypatch)
    bridge._camera_imu_enabled = True
    bridge.telemetry = {"position": [1., 2., .3], "yaw": .2, "received_at": 10.,
                        "camera_imu": {"correction_rpy": [.1, .1, 0.],
                                       "component_received_at": {"accel": accel_receipt, "gyro": 10.}}}
    # The normalization offset must not be counted as IMU age.
    bridge._telemetry_ros_stamp = 1000.
    bridge.refresh_tf(None)
    assert len(odom) == 1 and odom[0].header.stamp.to_sec() == 1000.
    assert transforms[0].child_frame_id == "tf_frame_base_link"
    assert any(isinstance(t, list) for t in transforms) is expected_camera
    bridge.refresh_tf(None)
    assert len(odom) == 1
    # Fresh gravity at the next body sample restores the camera branch.
    bridge.telemetry["received_at"] = 10.1
    bridge.telemetry["camera_imu"]["component_received_at"]["accel"] = 10.1
    bridge._telemetry_ros_stamp = 1000.1
    bridge.refresh_tf(None)
    assert len(odom) == 2
    assert isinstance(transforms[-1], list)
    assert transforms[-1][0].header.stamp.to_sec() == 1000.1
