"""Cloud progress under sustained input; no ROS master or hardware needed."""

import pathlib
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import physical_sensor_ros_bridge as sensor


def _bridge():
    bridge = object.__new__(sensor.SensorRosBridge)
    bridge.args = SimpleNamespace(point_stride=6, max_depth_m=8., no_return_depth_m=8.)
    bridge._packet_lock = threading.Lock()
    bridge._cloud_lock = threading.Lock()
    bridge._cloud_event = threading.Event()
    bridge._active_transport_session = "camera"
    bridge._active_transport_connection = 1
    bridge._last_published_bridge_seq = 1
    bridge._latest_packet = None
    bridge._latest_cloud = None
    bridge._cloud_superseded_count = 0
    bridge._cloud_projection_cache = None
    messages = []
    bridge.cloud_pub = SimpleNamespace(publish=messages.append)
    return bridge, messages


def test_projection_finishes_even_when_newer_frames_arrive_each_time(monkeypatch):
    bridge, messages = _bridge()

    def project(*args, **kwargs):
        # Every projection spans the arrival of another frame. Cancelling
        # here would starve GMapping indefinitely, not make its input fresher.
        bridge._last_published_bridge_seq += 1
        bridge._latest_packet = {"_bridge_seq": bridge._last_published_bridge_seq}
        return np.array([[1., 2., 3.]], dtype=np.float32), None

    monkeypatch.setattr(sensor, "_sample_depth_points", project)
    monkeypatch.setattr(sensor.rospy, "loginfo_throttle", lambda *args: None)
    for seq in range(1, 5):
        bridge.publish_cloud(np.ones((2, 2), dtype=np.uint16), {},
                             sensor.rospy.Time.from_sec(1000. + seq), "depth", .001, seq)
    assert [msg.header.seq for msg in messages] == [1, 2, 3, 4]
    assert [msg.header.stamp.to_sec() for msg in messages] == [1001., 1002., 1003., 1004.]
    assert all(msg.header.frame_id == "depth" for msg in messages)


@pytest.mark.parametrize("generation", [("camera", 2), ("new-camera", 1)])
def test_reconnect_during_projection_still_rejects_retired_result(monkeypatch, generation):
    bridge, messages = _bridge()

    def project(*args, **kwargs):
        bridge._active_transport_session, bridge._active_transport_connection = generation
        return np.array([[1., 2., 3.]], dtype=np.float32), None

    monkeypatch.setattr(sensor, "_sample_depth_points", project)
    bridge.publish_cloud(np.ones((2, 2), dtype=np.uint16), {},
                         sensor.rospy.Time.from_sec(1001.), "depth", .001, 1,
                         ("camera", 1))
    assert messages == []
    assert bridge._cloud_superseded_count == 1


def test_retired_queued_cloud_is_rejected_before_projection(monkeypatch):
    bridge, messages = _bridge()
    bridge._active_transport_connection = 2

    def unexpected_projection(*args, **kwargs):
        raise AssertionError("Retired queued work must not spend projection time")

    monkeypatch.setattr(sensor, "_sample_depth_points", unexpected_projection)
    bridge.publish_cloud(np.ones((2, 2), dtype=np.uint16), {},
                         sensor.rospy.Time.from_sec(1001.), "depth", .001, 1,
                         ("camera", 1))
    assert messages == []
    assert bridge._cloud_superseded_count == 1


def test_legacy_generation_is_valid_only_before_a_tagged_connection():
    bridge, _ = _bridge()
    bridge._active_transport_session = None
    bridge._active_transport_connection = -1
    assert bridge._cloud_generation_is_current((None, 0))
    bridge._active_transport_session = "camera"
    assert not bridge._cloud_generation_is_current((None, 0))


def test_worker_finishes_inflight_then_takes_only_latest_waiting_cloud(monkeypatch):
    bridge, messages = _bridge()
    depth = np.ones((2, 2), dtype=np.uint16)
    projected = []

    def enqueue(seq):
        bridge._queue_cloud(depth, {}, sensor.rospy.Time.from_sec(1000. + seq),
                            "depth", .001, seq, ("camera", 1))

    def project(*args, **kwargs):
        projected.append(True)
        if len(projected) == 1:
            enqueue(2)
            enqueue(3)
            bridge._last_published_bridge_seq = 3
            bridge._latest_packet = {"_bridge_seq": 3}
        return np.array([[1., 2., 3.]], dtype=np.float32), None

    monkeypatch.setattr(sensor, "_sample_depth_points", project)
    iterations = iter([False, False, True])
    monkeypatch.setattr(sensor.rospy, "is_shutdown", lambda: next(iterations))
    enqueue(1)
    bridge._cloud_loop()
    assert len(projected) == 2
    assert [msg.header.seq for msg in messages] == [1, 3]
    assert [msg.header.stamp.to_sec() for msg in messages] == [1001., 1003.]
    assert bridge._latest_cloud is None
