"""Sensor odom/TF and YOLO must agree on the body, not camera, orientation."""

from pathlib import Path
import math
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_sensor_ros_bridge as sensor
import physical_yoloe_bridge as yolo


def _bridge(monkeypatch):
    bridge = object.__new__(sensor.SensorRosBridge)
    bridge.args = SimpleNamespace(camera_frame="camera", camera_parent="tf_frame_base_link",
                                  camera_x=0., camera_y=0., camera_z=1.,
                                  camera_roll=0., camera_pitch=0., camera_yaw=0.)
    bridge._depth_to_color_extrinsics = None
    bridge._last_tf_source_stamp = -float("inf")
    bridge._last_depth_frame = "camera"
    bridge._telemetry_lock = threading.Lock()
    bridge._camera_imu_enabled = False
    bridge._camera_imu_use_yaw = False
    bridge._camera_imu_max_roll_rad = bridge._camera_imu_max_pitch_rad = .7
    bridge._camera_imu_max_yaw_rad = .35
    bridge.static_frames = set()
    bridge._static_transforms = {}
    odom, transforms = [], []
    bridge.odom_pub = SimpleNamespace(publish=odom.append)
    bridge.tf_broadcaster = SimpleNamespace(sendTransform=transforms.append)
    bridge.static_broadcaster = SimpleNamespace(sendTransform=lambda *args: None)
    monkeypatch.setattr(sensor.rospy, "logwarn_throttle", lambda *args: None)
    return bridge, odom, transforms


def test_camera_hint_does_not_override_explicit_body_yaw():
    pose = {"position": [2., 3., .3], "yaw": 0.,
            "camera_pose": {"quaternion": [0., 0., math.sin(.5), math.cos(.5)]}}
    rotation, _ = yolo._world_transform(pose, np.zeros(3), (0., 0., 0.))
    assert np.allclose(rotation, np.eye(3))


@pytest.mark.parametrize("orientation", [
    {"imu": {"quaternion": [2., 0., 0., 2.]}},
    {"imu": {"orientation": {"x": 0., "y": 0., "z": 1., "w": 1.}}},
    {"base_pose": {"quaternion": [0., 0., 1., 1.]}},
    {"imu": {"rpy": [0., 0., .7]}},
])
def test_sensor_published_body_rotation_matches_yolo(monkeypatch, orientation):
    bridge, odom, transforms = _bridge(monkeypatch)
    pose = {"position": [2., 3., .3], **orientation}
    bridge.publish_pose(pose, sensor.rospy.Time.from_sec(10.))
    assert len(odom) == 1
    q = odom[0].pose.pose.orientation
    quaternion = (q.x, q.y, q.z, q.w)
    assert sum(value * value for value in quaternion) == pytest.approx(1.)
    expected, offset = yolo._world_transform(pose, np.zeros(3), (0., 0., 0.))
    assert np.allclose(yolo._quaternion_matrix(quaternion), expected)
    assert np.allclose(offset, [2., 3., .3])
    assert transforms[0].transform.rotation == q
    assert odom[0].header.stamp.to_sec() == 10.


@pytest.mark.parametrize("pose", [
    {}, {"position": [1., 2., 3.]},
    {"yaw": 0.}, {"position": [1., 2., float("nan")], "yaw": 0.},
    {"position": [1., 2., 3.], "imu": {"quaternion": [0., 0., 0., 0.]}},
    {"position": [1., 2., 3.], "camera_pose": {"quaternion": [0., 0., 0., 1.]}},
    {"position": [1., 2., 3.], "d435i_pose": {"quaternion": [0., 0., 0., 1.]}},
])
def test_invalid_or_camera_only_pose_cannot_publish_body_tf(monkeypatch, pose):
    bridge, odom, transforms = _bridge(monkeypatch)
    bridge.publish_pose(pose, sensor.rospy.Time.from_sec(10.))
    assert odom == [] and transforms == []
    assert bridge._last_tf_source_stamp == -float("inf")
    assert yolo._capture_pose_error(pose) is not None
    # Rejecting a bad sample does not inhibit recovery or re-date the bad pose.
    recovered_stamp = sensor.rospy.Time.from_sec(10.1)
    bridge.publish_pose({"position": [0., 0., .3], "yaw": 0.}, recovered_stamp)
    assert len(odom) == 1 and odom[0].header.stamp == recovered_stamp


def test_scaled_multi_axis_body_quaternion_matches_expected_rotation(monkeypatch):
    bridge, odom, _ = _bridge(monkeypatch)
    x, y, z, w = sensor._quat_rpy(.2, -.1, .4)
    pose = {"position": [2., 3., .3], "imu": {"quaternion": [2*w, 2*x, 2*y, 2*z]},
            "camera_pose": {"quaternion": [0., 0., 0., 1.]}}
    bridge.publish_pose(pose, sensor.rospy.Time.from_sec(10.))
    q = odom[0].pose.pose.orientation
    expected = yolo._rotation(.2, -.1, .4)
    assert np.allclose(yolo._quaternion_matrix((q.x, q.y, q.z, q.w)), expected)
    assert np.allclose(yolo._world_transform(pose, np.zeros(3), (0., 0., 0.))[0], expected)


def test_bad_quaternion_can_use_explicit_valid_orientation_field():
    pose = {"position": [1., 2., 3.], "imu": {
        "quaternion": [float("nan"), 0., 0., 0.],
        "orientation": {"x": 0., "y": 0., "z": 0., "w": 1.}}}
    assert sensor._telemetry_body_quaternion(pose) == (0., 0., 0., 1.)
    assert yolo._capture_pose_error(pose) is None


def test_periodic_tf_uses_normalized_ros_receipt_not_source_clock(monkeypatch):
    bridge, odom, _ = _bridge(monkeypatch)
    bridge.telemetry = {"position": [1., 2., .3], "yaw": .1, "received_at": 999.}
    bridge._telemetry_ros_stamp = 1001.
    bridge.refresh_tf(None)
    assert odom[0].header.stamp.to_sec() == 1001.
    bridge.refresh_tf(None)
    assert len(odom) == 1


def test_capture_clock_rollback_does_not_freeze_later_periodic_tf(monkeypatch):
    bridge, odom, _ = _bridge(monkeypatch)
    bridge.publish_pose({"position": [1., 2., .3], "yaw": .1, "received_at": 1000.},
                        sensor.rospy.Time.from_sec(1000.))
    bridge.publish_pose({"position": [1., 2., .3], "yaw": .2, "received_at": 999.},
                        sensor.rospy.Time.from_sec(1001.))
    bridge.telemetry = {"position": [1., 2., .3], "yaw": .3, "received_at": 999.2}
    bridge._telemetry_ros_stamp = 1001.2
    bridge.refresh_tf(None)
    assert len(odom) == 3
    assert odom[-1].header.stamp == sensor.rospy.Time.from_sec(1001.2)


def test_historical_capture_keeps_tf_history_without_regressing_live_odom(monkeypatch):
    bridge, odom, transforms = _bridge(monkeypatch)
    pose = {"position": [1., 2., .3], "yaw": .1, "received_at": 10.}
    bridge.publish_pose(pose, sensor.rospy.Time.from_sec(12.))
    bridge.publish_pose(pose, sensor.rospy.Time.from_sec(11.))
    assert [msg.header.stamp.to_sec() for msg in odom] == [12.]
    assert [msg.header.stamp.to_sec() for msg in transforms] == [12., 11.]


def test_periodic_tf_without_normalized_receipt_cannot_invent_current_time(monkeypatch):
    bridge, odom, _ = _bridge(monkeypatch)
    bridge.telemetry = {"position": [1., 2., .3], "yaw": .1, "received_at": 10.}
    bridge.refresh_tf(None)
    assert odom == []


@pytest.mark.parametrize("capture_stamp,expected", [(10., [10., 11.]), (12., [12.])])
def test_capture_and_timer_serialize_watermark_and_live_odom(monkeypatch, capture_stamp, expected):
    bridge, odom, _ = _bridge(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    timer_attempted = threading.Event()
    errors = []

    class ObservedLock:
        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "periodic-test":
                timer_attempted.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    bridge._pose_publish_lock = ObservedLock()
    bridge.telemetry = {"position": [1., 2., .3], "yaw": .1, "received_at": 1.}
    bridge._telemetry_ros_stamp = 11.

    def publish(msg):
        odom.append(msg)
        if threading.current_thread().name == "capture-test":
            entered.set()
            assert release.wait(2.)

    def capture():
        try:
            bridge.publish_pose(bridge.telemetry, sensor.rospy.Time.from_sec(capture_stamp))
        except BaseException as exc:
            errors.append(exc)

    def timer():
        try:
            bridge.refresh_tf(None)
        except BaseException as exc:
            errors.append(exc)

    bridge.odom_pub = SimpleNamespace(publish=publish)
    capture_thread = threading.Thread(target=capture, name="capture-test")
    timer_thread = threading.Thread(target=timer, name="periodic-test")
    capture_thread.start()
    try:
        assert entered.wait(1.)
        timer_thread.start()
        assert timer_attempted.wait(1.)
        assert len(odom) == 1
    finally:
        release.set()
        capture_thread.join(2.)
        if timer_thread.ident is not None:
            timer_thread.join(2.)
    assert not errors
    assert not capture_thread.is_alive() and not timer_thread.is_alive()
    assert [msg.header.stamp.to_sec() for msg in odom] == expected
    assert bridge._last_tf_ros_stamp_ns == sensor.rospy.Time.from_sec(max(capture_stamp, 11.)).to_nsec()
