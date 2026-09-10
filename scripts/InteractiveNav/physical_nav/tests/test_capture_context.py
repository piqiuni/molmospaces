"""Per-capture pose/calibration survives independent ROS callback ordering."""

from itertools import permutations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import rospy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_sensor_ros_bridge as sensor
from physical_yoloe_bridge import _ExactRgbdPairBuffer
from test_yolo_rgbd_pairing import _worker, _wire_image


def _context_worker():
    worker = _worker()
    worker._rgbd_pairs = _ExactRgbdPairBuffer(require_context=True)
    worker._camera_info = worker._depth_camera_info = None
    def unrelated_history(stamp):
        raise AssertionError("live captures must not rematch YOLO's independent history")
    worker._telemetry_at = unrelated_history
    return worker


def _publish_context(publisher, capture=1, stamp=10., telemetry=None, fx=100.):
    bridge = object.__new__(sensor.SensorRosBridge)
    _mount(bridge)
    bridge.capture_context_pub = SimpleNamespace(publish=publisher)
    intrinsics = dict(width=2, height=2, fx=fx, fy=100., cx=.5, cy=.5)
    calibration = dict(depth_scale=.002, telemetry_max_delta_sec=.15,
                       depth_to_color_extrinsics={"translation": [.02, 0., 0.]},
                       rgb_frame="rgb_capture", depth_frame="depth_capture")
    bridge.publish_capture_context(
        rospy.Time.from_sec(stamp), capture,
        {"position": [1., 2., .3], "yaw": .2} if telemetry is None else telemetry,
        .02, intrinsics, dict(intrinsics), calibration,
    )


def _mount(bridge):
    bridge.args = SimpleNamespace(camera_frame="rgb_capture", camera_parent="tf_frame_base_link",
                                  camera_x=0., camera_y=0., camera_z=1.,
                                  camera_roll=0., camera_pitch=0., camera_yaw=0.,
                                  telemetry_max_delta_sec=.15)
    bridge._camera_imu_enabled = False
    bridge._camera_imu_use_yaw = False
    bridge._camera_imu_max_roll_rad = bridge._camera_imu_max_pitch_rad = .7
    bridge._camera_imu_max_yaw_rad = .35


@pytest.mark.parametrize("order", list(permutations(("rgb", "depth", "context"))))
def test_all_callback_orders_use_sensor_pose_and_capture_calibration(order):
    worker = _context_worker()
    callbacks = {
        "rgb": lambda: worker._rgb_callback(_wire_image(1, 51, 10.)),
        "depth": lambda: worker._depth_callback(_wire_image(1, 1, 10., depth=True)),
        "context": lambda: _publish_context(worker._capture_context_callback),
    }
    for kind in order[:-1]:
        callbacks[kind]()
        assert worker._latest_raw_frame() is None
    callbacks[order[-1]]()
    raw = worker._latest_raw_frame()
    assert raw["telemetry"]["yaw"] == .2
    assert raw["seq"] == 1  # Actual receipt ID from metadata, not either topic counter.
    assert (raw["rgb_seq"], raw["depth_seq"]) == (51, 1)
    assert raw["depth_scale"] == .002
    assert raw["rgb_intrinsics"]["fx"] == 100.
    assert raw["camera_frame"] == "rgb_capture"
    assert raw["depth_frame"] == "depth_capture"
    assert raw["capture_pose_match"]["source"] == "sensor_capture_context"
    assert worker._latest_raw_frame() is None


def test_context_skew_preserves_latest_complete_capture_without_old_frame_queue():
    worker = _context_worker()
    for capture in range(1, 6):
        stamp = 10. + capture * .1
        worker._rgb_callback(_wire_image(capture, capture, stamp))
        worker._depth_callback(_wire_image(capture, capture + 10, stamp, depth=True))
        if capture > 1:
            _publish_context(worker._capture_context_callback, capture - 1,
                             10. + (capture - 1) * .1, fx=100. + capture - 1)
            raw = worker._latest_raw_frame()
            assert raw["seq"] == capture - 1
            assert np.all(raw["rgb_array"] == capture - 1)
            assert raw["rgb_intrinsics"]["fx"] == 100. + capture - 1
    # The next frame's absent metadata does not erase the already complete one.
    assert worker._rgbd_pairs.latest[6]["seq"] == 4


def test_missing_pose_is_authoritative_and_does_not_fall_back_to_newer_telemetry():
    worker = _context_worker()
    worker._rgb_callback(_wire_image(1, 1, 10.))
    worker._depth_callback(_wire_image(1, 1, 10., depth=True))
    _publish_context(worker._capture_context_callback, telemetry={})
    raw = worker._latest_raw_frame()
    assert raw["telemetry"] == {}
    assert raw["capture_pose_match"]["accepted"] is False
    # Existing infer pose gate keeps this frame 2D-only, rather than changing
    # the sensor decision based on which history arrived later at YOLO.


@pytest.mark.parametrize("field,value", [("stamp", 9.9), ("version", 2),
                                        ("depth_scale", -1.), ("telemetry", None)])
def test_wrong_or_invalid_context_cannot_release_capture(field, value):
    worker = _context_worker()
    worker._rgb_callback(_wire_image(1, 1, 10.))
    worker._depth_callback(_wire_image(1, 1, 10., depth=True))
    def corrupt(msg):
        payload = json.loads(msg.data)
        payload[field] = value
        worker._capture_context_callback(SimpleNamespace(data=json.dumps(payload)))
    _publish_context(corrupt)
    assert worker._latest_raw_frame() is None
    _publish_context(worker._capture_context_callback)
    assert worker._latest_raw_frame()["seq"] == 1


def test_context_and_image_pending_storage_is_bounded():
    pairs = _ExactRgbdPairBuffer(require_context=True)
    for stamp in range(1, 31):
        for kind in ("rgb", "context"):
            pairs.push(kind, {}, float(stamp), stamp)
    assert pairs.latest is None
    assert len(pairs.pending["rgb"]) == len(pairs.pending["context"]) == 4
    pairs.push("depth", {}, 30., 0)
    assert pairs.latest[1] == 30.
    assert all(not pending for pending in pairs.pending.values())


def test_worker_uses_only_newest_complete_context_without_copying_images():
    worker = _context_worker()
    for capture in range(1, 4):
        stamp = 10. + capture * .1
        worker._rgb_callback(_wire_image(capture, capture, stamp))
        worker._depth_callback(_wire_image(capture, capture, stamp, depth=True))
        _publish_context(worker._capture_context_callback, capture, stamp)
    rgb, depth = worker._rgb, worker._depth
    raw = worker._latest_raw_frame()
    assert raw["seq"] == 3
    assert raw["rgb_array"] is rgb
    assert raw["depth_array"] is depth
    _publish_context(worker._capture_context_callback, 1, 10.1)
    assert worker._latest_raw_frame() is None


def test_capture_metadata_cannot_describe_different_resolution(monkeypatch):
    worker = _context_worker()
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *a: None)
    worker._rgb_callback(_wire_image(1, 1, 10.))
    worker._depth_callback(_wire_image(1, 1, 10., depth=True))
    def wrong_dimensions(msg):
        payload = json.loads(msg.data)
        payload["depth_intrinsics"]["width"] = 848
        worker._capture_context_callback(SimpleNamespace(data=json.dumps(payload)))
    _publish_context(wrong_dimensions)
    assert worker._latest_raw_frame() is None
    worker._rgb_callback(_wire_image(2, 2, 10.1))
    worker._depth_callback(_wire_image(2, 2, 10.1, depth=True))
    _publish_context(worker._capture_context_callback, 2, 10.1)
    assert worker._latest_raw_frame()["seq"] == 2


def test_sensor_publish_uses_same_matched_snapshot_for_context_and_tf(monkeypatch):
    worker = _context_worker()
    bridge = object.__new__(sensor.SensorRosBridge)
    bridge._last_published_bridge_seq = -1
    bridge.last_seq = -1
    bridge.last_stamp = float("-inf")
    bridge._last_calibration_key = None
    bridge._packet_lock = threading.Lock()
    _mount(bridge)
    bridge._decode_executor = SimpleNamespace(submit=lambda fn, data, *args:
                                             SimpleNamespace(result=lambda: data))
    bridge._transport_generation_is_current = lambda packet: True
    bridge._normalise_ros_capture_stamp = lambda packet, stamp: (stamp, False)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *a: None)
    pose = {"position": [1., 2., .3], "yaw": .7}
    bridge.telemetry_at = lambda stamp: (pose, .01)
    seen = []
    bridge.publish_pose = lambda telemetry, *args, **kwargs: seen.append(telemetry) or "published"
    bridge._queue_cloud = lambda *args: None
    bridge.rgb_pub = SimpleNamespace(publish=worker._rgb_callback)
    bridge.depth_pub = SimpleNamespace(publish=worker._depth_callback)
    bridge.info_pub = bridge.depth_info_pub = bridge.calibration_pub = SimpleNamespace(publish=lambda m: None)
    bridge.capture_context_pub = SimpleNamespace(publish=worker._capture_context_callback)
    intrinsics = dict(width=2, height=2, fx=100., fy=100., cx=.5, cy=.5)
    bridge.publish(dict(seq=1, _bridge_seq=1, stamp=10.1,
                        rgb=np.ones((2, 2, 3), dtype=np.uint8),
                        depth=np.ones((2, 2), dtype=np.uint16),
                        rgb_intrinsics=intrinsics, depth_intrinsics=intrinsics))
    raw = worker._latest_raw_frame()
    assert raw["stamp"] == rospy.Time.from_sec(10.1).to_sec()
    assert raw["telemetry"] == seen[0] == pose
    assert raw["capture_timing"]["source_packet_stamp"] == 10.1
    assert raw["capture_timing"]["ros_capture_stamp"] == 10.1
    assert raw["capture_timing"]["source_to_ros_offset_s"] == 0.
    assert raw["capture_timing"]["source_clock_rebased"] is False
