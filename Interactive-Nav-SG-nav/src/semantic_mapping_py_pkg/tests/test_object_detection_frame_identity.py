import json
import threading
from collections import deque
from types import SimpleNamespace

import pytest

rospy = pytest.importorskip("rospy")
from object_detection_node import ObjectDetectionNode


def test_detector_resolves_task_step_by_stamp_not_ros_sequence():
    node = object.__new__(ObjectDetectionNode)
    node.lock = threading.Lock()
    node.step_identity_history = deque(maxlen=64)
    stamp = rospy.Time(1789304037, 594208001)
    for step, value in [(924, 1789304038.0), (923, stamp.to_sec())]:
        node.step_sync_callback(SimpleNamespace(data=json.dumps({
            "step_index": step, "stamp_sec": value,
        })))
    assert node._capture_step_for_stamp(stamp) == 923
    assert node._capture_step_for_stamp(rospy.Time(1789304040, 0)) is None


def test_detector_does_not_guess_step_when_identity_is_ambiguous():
    node = object.__new__(ObjectDetectionNode)
    node.lock = threading.Lock()
    node.step_identity_history = deque(maxlen=64)
    for step in (923, 924):
        node.step_sync_callback(SimpleNamespace(data=json.dumps({
            "step_index": step, "stamp_sec": 100, "stamp_nsec": 10,
        })))
    assert node._capture_step_for_stamp(rospy.Time(100, 10)) is None
