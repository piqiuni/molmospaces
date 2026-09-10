"""Exercise the actual TF2 first-writer cache without a ROS master."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import tf2_ros

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_sensor_ros_bridge as sensor
import physical_yoloe_bridge as yolo
from test_body_pose_consistency import _bridge
from test_capture_context import _context_worker
from test_yolo_capture_pose import _worker_and_raw


def _chain(monkeypatch):
    bridge, odom, _ = _bridge(monkeypatch)
    bridge._camera_imu_enabled = True
    buffer = tf2_ros.Buffer(debug=False)
    messages = []
    def send(messages_or_one):
        batch = messages_or_one if isinstance(messages_or_one, list) else [messages_or_one]
        for message in batch:
            buffer.set_transform(message, "offline_sensor")
            messages.append(message)
    bridge.tf_broadcaster = SimpleNamespace(sendTransform=send)
    return bridge, buffer, odom, messages


@pytest.mark.parametrize("difference", ["body", "camera"])
def test_conflicting_capture_is_rejected_before_tf2_ignores_its_correction(monkeypatch, difference):
    bridge, buffer, odom, sent = _chain(monkeypatch)
    stamp = sensor.rospy.Time.from_sec(10.)
    original = {"position": [1., 2., .3], "yaw": 0.}
    bridge.telemetry = original
    bridge._telemetry_ros_stamp = 10.
    bridge.refresh_tf(None)
    baseline = buffer.lookup_transform("tf_frame_odom", "camera", stamp)
    before = len(sent)
    changed = {**original, "yaw": .5} if difference == "body" else {
        **original, "camera_imu": {"imu_calibrating": False, "correction_rpy": [.2, 0., 0.]}}
    assert bridge.publish_pose(changed, stamp) == "conflicting_capture_tf_stamp"
    assert len(sent) == before
    assert len(odom) == 1
    assert buffer.lookup_transform("tf_frame_odom", "camera", stamp) == baseline
    assert bridge._tf_stamp_conflict_count == 1
    # A rejected capture cannot freeze valid later frames.
    later = sensor.rospy.Time.from_sec(10.1)
    assert bridge.publish_pose(changed, later) == "published"
    assert buffer.lookup_transform("tf_frame_odom", "camera", later) != baseline


def test_duplicate_same_transform_does_not_rebroadcast_and_new_child_can_be_added(monkeypatch):
    bridge, buffer, _, sent = _chain(monkeypatch)
    pose = {"position": [1., 2., .3], "yaw": .2}
    stamp = sensor.rospy.Time.from_sec(10.)
    assert bridge.publish_pose(pose, stamp) == "published"
    before = len(sent)
    assert bridge.publish_pose(pose, stamp) == "duplicate"
    assert len(sent) == before
    assert bridge.publish_pose(pose, stamp, "depth", {}) == "published"
    assert [t.child_frame_id for t in sent[before:]] == ["depth"]
    buffer.lookup_transform("tf_frame_odom", "depth", stamp)


@pytest.mark.parametrize("same_rotation", [[-1., 0., 0., 0.], [1., 0., 0., -1e-12]])
def test_quaternion_sign_is_not_a_transform_conflict(monkeypatch, same_rotation):
    bridge, _, _, _ = _chain(monkeypatch)
    pose = {"position": [1., 2., .3], "imu": {"quaternion": [1., 0., 0., 0.]}}
    stamp = sensor.rospy.Time.from_sec(10.)
    assert bridge.publish_pose(pose, stamp) == "published"
    pose["imu"]["quaternion"] = same_rotation
    assert bridge.publish_pose(pose, stamp) == "duplicate"


def test_tf_binding_history_is_bounded_and_evicted_times_cannot_be_rewritten(monkeypatch):
    bridge, _, _, _ = _chain(monkeypatch)
    pose = {"position": [1., 2., .3], "yaw": 0.}
    for index in range(520):
        bridge.publish_pose(pose, sensor.rospy.Time.from_sec(10. + index * .01))
    assert len(bridge._published_tf_samples) == 512
    assert bridge.publish_pose(pose, sensor.rospy.Time.from_sec(10.)) == "expired_capture_tf_stamp"


@pytest.mark.parametrize("capture_pose,expected_status", [
    ({"position": [1., 2., .3], "yaw": .5}, "conflicting_capture_tf_stamp"),
    ({}, "invalid_capture_pose"),
    ({"position": [1., 2., .3]}, "invalid_capture_pose"),
])
def test_rejected_pose_keeps_rgbd_context_but_never_queues_wrong_world_cloud(
    monkeypatch, capture_pose, expected_status,
):
    bridge, _, _, _ = _chain(monkeypatch)
    original = {"position": [1., 2., .3], "yaw": 0.}
    bridge.publish_pose(original, sensor.rospy.Time.from_sec(10.))
    worker = _context_worker()
    bridge._last_published_bridge_seq = bridge.last_seq = -1
    bridge.last_stamp = float("-inf")
    bridge._last_calibration_key = None
    bridge._packet_lock = threading.Lock()
    bridge.args.telemetry_max_delta_sec = .15
    bridge._decode_executor = SimpleNamespace(submit=lambda fn, data, *a:
                                             SimpleNamespace(result=lambda: data))
    bridge._transport_generation_is_current = lambda packet: True
    bridge._normalise_ros_capture_stamp = lambda packet, stamp: (stamp, False)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *a: None)
    bridge.telemetry_at = lambda stamp: (capture_pose, .01)
    queued = []
    bridge._queue_cloud = lambda *args: queued.append(args)
    bridge.rgb_pub = SimpleNamespace(publish=worker._rgb_callback)
    bridge.depth_pub = SimpleNamespace(publish=worker._depth_callback)
    bridge.info_pub = bridge.depth_info_pub = bridge.calibration_pub = SimpleNamespace(publish=lambda m: None)
    bridge.capture_context_pub = SimpleNamespace(publish=worker._capture_context_callback)
    intrinsics = dict(width=2, height=2, fx=100., fy=100., cx=.5, cy=.5)
    bridge.publish(dict(seq=1, _bridge_seq=1, stamp=10.,
                        rgb=np.ones((2, 2, 3), dtype=np.uint8),
                        depth=np.ones((2, 2), dtype=np.uint16),
                        rgb_intrinsics=intrinsics, depth_intrinsics=intrinsics))
    raw = worker._latest_raw_frame()
    assert raw is not None and raw["capture_tf_status"] == expected_status
    assert not queued
    detector, detector_raw = _worker_and_raw()
    detector_raw.update(telemetry=raw["telemetry"], capture_tf_status=raw["capture_tf_status"])
    def unexpected(*a, **kw):
        pytest.fail("Conflicting TF cannot generate world geometry")
    monkeypatch.setattr(yolo, "_geometry_lift_candidate", unexpected)
    report = detector.infer(detector_raw)
    assert report["overlay_jpeg"]
    assert report["capture_pose_status"] == expected_status
    assert report["detections"][0]["geometry_skipped"]
