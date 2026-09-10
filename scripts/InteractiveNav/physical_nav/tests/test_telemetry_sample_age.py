"""A fresh camera/envelope must not make cached sport-state pose fresh."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_telemetry_receipt_identity import _chain


@pytest.mark.parametrize("ros_reference,source_reference,pose_source,expected", [
    (10.5, 10.5, 10., 10.),
    (1000.5, 10.5, 10., 1000.),
    (10.1, 10.1, 10., 10.),
])
def test_history_and_ros_envelope_keep_sample_time(
    ros_reference, source_reference, pose_source, expected,
):
    bridge, worker, published = _chain()
    bridge.update_telemetry({"received_at": pose_source, "position": [1., 2., .3], "yaw": .2},
                            ros_reference, capture_seq=1,
                            source_reference_stamp=source_reference)
    assert bridge._telemetry_history[-1][0] == pytest.approx(expected)
    assert published[-1]["stamp"] == pytest.approx(expected)
    assert bridge._telemetry_ros_stamp == pytest.approx(expected)
    accepted = abs(ros_reference - expected) <= .15
    assert bool(bridge.telemetry_at(ros_reference)[0]) is accepted
    assert bool(worker._telemetry_at(ros_reference)) is accepted


def test_repeated_camera_frames_cannot_rejuvenate_stopped_pose_stream():
    bridge, worker, published = _chain()
    pose = {"received_at": 10., "position": [1., 2., .3], "yaw": .2}
    for seq, stamp in enumerate((10., 10.1, 10.2, 10.5), 1):
        bridge.update_telemetry(pose, stamp, capture_seq=seq, source_reference_stamp=stamp)
    assert {entry[0] for entry in bridge._telemetry_history} == {10.}
    assert {entry["stamp"] for entry in published} == {10.}
    assert bridge.telemetry_at(10.5)[0] == worker._telemetry_at(10.5) == {}
    assert bridge.telemetry_at(10.5)[1] == pytest.approx(.5)


@pytest.mark.parametrize("received_at", [None, float("nan"), 0., 1000.])
def test_missing_or_impossibly_future_sample_time_cannot_create_pose_history(received_at):
    bridge, worker, published = _chain()
    bridge.update_telemetry({"received_at": received_at, "position": [1., 2., .3], "yaw": .2},
                            10., capture_seq=1, source_reference_stamp=10.)
    assert not bridge._telemetry_history
    assert not published
    assert worker._telemetry_at(10.) == {}


def test_new_camera_sequence_is_not_proof_that_an_older_pose_is_new():
    bridge, _, published = _chain()
    bridge.update_telemetry({"received_at": 10.2, "position": [2., 0., .3], "yaw": .2},
                            10.2, source_reference_stamp=10.2)
    bridge.update_telemetry({"received_at": 10.1, "position": [1., 0., .3], "yaw": .1},
                            10.3, source_reference_stamp=10.3, capture_seq=100)
    assert bridge.telemetry["yaw"] == .2
    assert bridge._telemetry_source_stamp == 10.2
    assert len(published) == 1
    assert bridge.telemetry_at(10.3)[0]["yaw"] == .2


def test_confirmed_clock_rebase_preserves_age_and_allows_new_pose_epoch():
    bridge, _, published = _chain()
    for pose_stamp, source_ref, ros_ref, seq, rebased in [
        (20., 20., 20., 1, False), (10., 10.1, 21.1, 2, True),
    ]:
        bridge.update_telemetry({"received_at": pose_stamp, "position": [1., 2., .3], "yaw": .2},
                                ros_ref, source_reference_stamp=source_ref, capture_seq=seq,
                                source_clock_rebased=rebased)
    assert bridge._telemetry_source_stamp == 10.
    assert published[-1]["stamp"] == pytest.approx(21.)
    assert bridge.telemetry_at(21.1)[1] == pytest.approx(.1)
    assert len(bridge._telemetry_history) == 1


def test_sensor_publish_applies_actual_pose_age_before_world_cloud_admission(monkeypatch):
    from types import SimpleNamespace
    import threading
    import numpy as np
    import physical_sensor_ros_bridge as sensor
    from test_capture_context import _mount, _context_worker
    bridge, _, _ = _chain()
    _mount(bridge)
    worker = _context_worker()
    bridge._last_published_bridge_seq = bridge.last_seq = -1
    bridge.last_stamp = float("-inf")
    bridge._last_calibration_key = None
    bridge._packet_lock = threading.Lock()
    bridge._decode_executor = SimpleNamespace(submit=lambda fn, data, *a:
                                             SimpleNamespace(result=lambda: data))
    bridge._transport_generation_is_current = lambda packet: True
    bridge._normalise_ros_capture_stamp = lambda packet, stamp: (stamp, False)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *a: None)
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *a: None)
    bridge.rgb_pub = SimpleNamespace(publish=worker._rgb_callback)
    bridge.depth_pub = SimpleNamespace(publish=worker._depth_callback)
    bridge.info_pub = bridge.depth_info_pub = bridge.calibration_pub = SimpleNamespace(publish=lambda m: None)
    bridge.capture_context_pub = SimpleNamespace(publish=worker._capture_context_callback)
    admitted = []
    bridge.publish_pose = lambda *a, **kw: "published"
    bridge._queue_cloud = lambda *args: admitted.append(args)
    intrinsics = dict(width=2, height=2, fx=100., fy=100., cx=.5, cy=.5)
    for seq, capture_stamp in enumerate((10.1, 10.5), 1):
        bridge.publish(dict(seq=seq, _bridge_seq=seq, stamp=capture_stamp,
                            rgb=np.ones((2, 2, 3), dtype=np.uint8),
                            depth=np.ones((2, 2), dtype=np.uint16),
                            rgb_intrinsics=intrinsics, depth_intrinsics=intrinsics,
                            telemetry={"received_at": 10., "position": [1., 2., .3], "yaw": .2}))
        raw = worker._latest_raw_frame()
        assert raw["capture_pose_match"]["delta_sec"] == pytest.approx(capture_stamp - 10.)
        assert raw["capture_pose_match"]["accepted"] is (seq == 1)
        assert bool(raw["telemetry"]) is (seq == 1)
    assert len(admitted) == 1
