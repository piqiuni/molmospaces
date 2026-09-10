"""Late telemetry must not relabel a newer pose with an older capture stamp."""

from collections import deque
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physical_sensor_ros_bridge as sensor
import physical_yoloe_bridge as yolo


def _chain():
    bridge = object.__new__(sensor.SensorRosBridge)
    bridge._telemetry_lock = threading.Lock()
    bridge._telemetry_history = deque(maxlen=64)
    bridge._telemetry_source_stamp = -float("inf")
    bridge._telemetry_capture_seq = -1
    bridge._telemetry_transport_session = None
    bridge._telemetry_transport_connection = -1
    bridge._telemetry_retired_sessions = set()
    bridge.telemetry = {}
    bridge.args = SimpleNamespace(telemetry_max_delta_sec=.15)
    bridge._mirror_enabled = False
    worker = object.__new__(yolo.YoloeWorker)
    worker._telemetry_lock = threading.Lock()
    worker._telemetry_history = []
    published = []

    def publish(msg):
        published.append(json.loads(msg.data))
        worker._telemetry_callback(msg)

    bridge.telemetry_pub = SimpleNamespace(publish=publish)
    return bridge, worker, published


def _update(bridge, source_stamp, ros_stamp, yaw, *, seq=None, session=None, connection=1, **extra):
    bridge.update_telemetry({"received_at": source_stamp, "position": [yaw, 0., .3],
                             "yaw": yaw, **extra}, ros_stamp, capture_seq=seq,
                            transport_session=session, transport_connection=connection)


def test_late_packet_does_not_publish_new_pose_at_old_stamp():
    bridge, worker, published = _chain()
    _update(bridge, 20., 20., 2., seq=20)
    _update(bridge, 10., 10., 1., seq=10, battery=50,
            base_pose={"quaternion": [0., 0., 1., 0.]})
    assert len(published) == 1
    assert bridge.telemetry["yaw"] == 2.
    assert bridge.telemetry["battery"] == 50
    assert bridge._telemetry_ros_stamp == 20.
    assert "base_pose" not in bridge.telemetry
    assert bridge.telemetry_at(10.)[0] == worker._telemetry_at(10.) == {}
    assert bridge.telemetry_at(20.)[0]["yaw"] == worker._telemetry_at(20.)["yaw"] == 2.


def test_untagged_packet_cannot_impersonate_active_transport():
    bridge, worker, published = _chain()
    _update(bridge, 20., 20., 2., session="active")
    _update(bridge, 21., 21., 9.)
    assert len(published) == 1
    assert bridge.telemetry["yaw"] == 2.
    assert worker._telemetry_at(21.) == {}


@pytest.mark.parametrize("new_session,new_connection", [("old", 2), ("new", 1)])
def test_transport_change_resets_capture_sequence_watermark(new_session, new_connection):
    bridge, worker, published = _chain()
    _update(bridge, 100., 100., 1., seq=1000, session="old")
    _update(bridge, 10., 101., 2., seq=1, session=new_session, connection=new_connection)
    _update(bridge, 9., 102., 3., seq=2, session=new_session, connection=new_connection)
    assert bridge._telemetry_capture_seq == 2
    assert len(published) == 3
    assert bridge.telemetry_at(102.)[0]["yaw"] == worker._telemetry_at(102.)["yaw"] == 3.


def test_fresh_capture_clock_rollback_does_not_poison_following_telemetry():
    bridge, worker, published = _chain()
    _update(bridge, 1000., 1000., .1, seq=1)
    _update(bridge, 999., 1001., .2, seq=2)
    _update(bridge, 999.1, 1001.1, .3)
    assert bridge._telemetry_source_stamp == 999.1
    assert bridge.telemetry["yaw"] == .3
    assert published[-1]["telemetry"]["yaw"] == .3
    assert worker._telemetry_at(1001.1)["yaw"] == .3


def test_older_capture_cannot_win_using_a_larger_pre_rollback_source_stamp():
    bridge, worker, published = _chain()
    _update(bridge, 1000., 1000., .1, seq=1)
    _update(bridge, 999., 1001., .2, seq=2)
    _update(bridge, 1000., 1002., 9., seq=1)
    assert len(published) == 2
    assert bridge.telemetry["yaw"] == .2
    assert worker._telemetry_at(1002.) == {}


def test_overtaken_capture_pose_is_history_only_and_remains_causal():
    bridge, _, published = _chain()
    bridge._mirror_enabled = True
    bridge._mirror_lock = threading.Lock()
    bridge._latest_mirror_telemetry = None
    bridge._mirror_event = threading.Event()
    bridge.update_telemetry(
        {"received_at": 10.18, "position": [2., 0., .3], "yaw": .2},
        10.18,
        source_reference_stamp=10.18,
    )
    latest_mirror = bridge._latest_mirror_telemetry
    bridge.update_telemetry(
        {"received_at": 10.08, "position": [1., 0., .3], "yaw": .1},
        10.10,
        capture_seq=7,
        source_reference_stamp=10.10,
    )

    assert bridge.telemetry["yaw"] == .2
    assert bridge._telemetry_source_stamp == pytest.approx(10.18)
    assert bridge._telemetry_ros_stamp == pytest.approx(10.18)
    assert bridge._telemetry_capture_seq == 7
    assert len(published) == 1
    assert bridge._latest_mirror_telemetry is latest_mirror
    assert [entry[0] for entry in bridge._telemetry_history] == pytest.approx(
        [10.08, 10.18]
    )
    assert bridge.telemetry_at(10.10)[0]["yaw"] == .1
    assert bridge.telemetry_at(10.20)[0]["yaw"] == .2


def test_sorted_history_replaces_duplicate_and_evicts_oldest_by_timestamp():
    history = deque(maxlen=3)
    sensor._insert_capture_telemetry(history, 3., {"yaw": 3.})
    sensor._insert_capture_telemetry(history, 1., {"yaw": 1.})
    sensor._insert_capture_telemetry(history, 2., {"yaw": 2.})
    assert [stamp for stamp, _ in history] == [1., 2., 3.]

    sensor._insert_capture_telemetry(history, 2., {"yaw": 20.})
    assert len(history) == 3
    assert sensor._nearest_capture_telemetry(history, 2.) == ({"yaw": 20.}, 0.)

    sensor._insert_capture_telemetry(history, 4., {"yaw": 4.})
    sensor._insert_capture_telemetry(history, .5, {"yaw": .5})
    assert [stamp for stamp, _ in history] == [2., 3., 4.]
