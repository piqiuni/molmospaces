"""Exercise real image callbacks without constructing YOLO or starting ROS."""

from pathlib import Path
import io
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_yoloe_bridge import YoloeWorker, _ExactRgbdPairBuffer
from physical_sensor_ros_bridge import SensorRosBridge
from rospy.msg import serialize_message
from sensor_msgs.msg import Image
import rospy


def _worker():
    worker = object.__new__(YoloeWorker)
    worker._frame_lock = threading.Lock()
    worker._rgb = worker._depth = None
    worker._rgb_stamp = worker._depth_stamp = 0.
    worker._rgb_seq = worker._depth_seq = 0
    worker._last_frame_key = None
    worker._depth_scale = .001
    worker._depth_to_color_extrinsics = {}
    info = SimpleNamespace(width=2, height=2, K=[1., 0., .5, 0., 1., .5, 0., 0., 1.],
                           D=[], distortion_model="", header=SimpleNamespace(frame_id="camera"))
    worker._camera_info = worker._depth_camera_info = info
    worker._telemetry_at = lambda stamp: {}
    return worker


def _message(seq, *, depth=False, stamp=None):
    array = np.full((2, 2) if depth else (2, 2, 3), seq,
                    dtype=np.uint16 if depth else np.uint8)
    return SimpleNamespace(
        encoding="16UC1" if depth else "bgr8", height=2, width=2,
        step=array.strides[0], data=array.tobytes(),
        header=SimpleNamespace(seq=seq, stamp=SimpleNamespace(
            to_sec=lambda: 1000. + seq * .1 if stamp is None else stamp)),
    )


@pytest.mark.parametrize("depth_leads", [False, True])
def test_one_frame_callback_skew_does_not_starve_complete_pairs(depth_leads):
    worker = _worker()
    leading = worker._depth_callback if depth_leads else worker._rgb_callback
    following = worker._rgb_callback if depth_leads else worker._depth_callback
    leading(_message(1, depth=depth_leads))
    outputs = []
    for seq in range(1, 5):
        leading(_message(seq + 1, depth=depth_leads))
        following(_message(seq, depth=not depth_leads))
        raw = worker._latest_raw_frame()
        assert raw is not None
        outputs.append(raw["seq"])
        assert np.all(raw["rgb_array"] == seq)
        assert np.all(raw["depth_array"] == seq)
        assert raw["rgb_seq"] == raw["depth_seq"] == seq
        assert worker._latest_raw_frame() is None
    assert outputs == [1, 2, 3, 4]


def test_inference_uses_latest_complete_pair_even_if_next_rgb_is_incomplete():
    worker = _worker()
    for seq in (1, 2):
        worker._rgb_callback(_message(seq))
        worker._depth_callback(_message(seq, depth=True))
    worker._rgb_callback(_message(3))
    assert worker._latest_raw_frame()["seq"] == 2
    assert worker._latest_raw_frame() is None
    worker._depth_callback(_message(3, depth=True))
    assert worker._latest_raw_frame()["seq"] == 3


def test_matching_sequence_alone_cannot_pair_different_captures():
    worker = _worker()
    worker._rgb_callback(_message(1, stamp=10.))
    worker._depth_callback(_message(1, depth=True, stamp=10.05))
    assert worker._latest_raw_frame() is None
    worker._depth_callback(_message(1, depth=True, stamp=10.))
    assert worker._latest_raw_frame()["stamp"] == 10.


def test_bridge_sequence_reuse_with_new_capture_stamp_is_not_a_duplicate():
    worker = _worker()
    for stamp in (10., 20.):
        worker._rgb_callback(_message(1, stamp=stamp))
        worker._depth_callback(_message(1, depth=True, stamp=stamp))
        assert worker._latest_raw_frame()["stamp"] == stamp


def test_calibration_arrival_retries_complete_pair_without_losing_it():
    worker = _worker()
    info = worker._camera_info
    worker._camera_info = None
    worker._rgb_callback(_message(1))
    worker._depth_callback(_message(1, depth=True))
    assert worker._latest_raw_frame() is None
    worker._camera_info_callback(info)
    assert worker._latest_raw_frame()["seq"] == 1


def test_exact_pair_buffer_is_bounded_and_does_not_replace_new_pair_with_old():
    pairs = _ExactRgbdPairBuffer(capacity=4)
    image = np.zeros((1, 1))
    for seq in range(1, 21):
        pairs.push("rgb", image, float(seq), seq)
    assert len(pairs.pending["rgb"]) == 4
    pairs.push("depth", image, 1., 1)
    assert pairs.latest is None
    pairs.push("depth", image, 20., 20)
    assert pairs.latest[:2] == (20, 20.)
    pairs.push("rgb", image, 19., 19)
    pairs.push("depth", image, 19., 19)
    assert pairs.latest[:2] == (20, 20.)
    assert all(not entries for entries in pairs.pending.values())


def test_legacy_zero_sequence_keeps_timestamp_tolerance_path():
    worker = _worker()
    worker._rgb_callback(_message(0, stamp=10.))
    worker._depth_callback(_message(0, depth=True, stamp=10.05))
    assert worker._latest_raw_frame()["stamp"] == 10.05


def _wire_image(capture_seq, publisher_seq, stamp, *, depth=False):
    array = np.full((2, 2) if depth else (2, 2, 3), capture_seq,
                    dtype=np.uint16 if depth else np.uint8)
    msg = SensorRosBridge._image_msg(
        array, "16UC1" if depth else "bgr8", rospy.Time.from_sec(stamp),
        "camera", capture_seq,
    )
    wire = io.BytesIO()
    serialize_message(wire, publisher_seq, msg)
    return Image().deserialize(wire.getvalue()[4:])


def test_real_ros_serialization_rewrites_capture_sequence_per_topic():
    worker = _worker()
    rgb = _wire_image(100, 51, 10.)
    depth = _wire_image(100, 1, 10., depth=True)
    assert (rgb.header.seq, depth.header.seq) == (51, 1)
    worker._rgb_callback(rgb)
    worker._depth_callback(depth)
    raw = worker._latest_raw_frame()
    assert raw is not None
    assert raw["stamp"] == rgb.header.stamp.to_sec() == depth.header.stamp.to_sec()
    assert (raw["rgb_seq"], raw["depth_seq"]) == (51, 1)
    assert np.all(raw["rgb_array"] == 100)
    assert np.all(raw["depth_array"] == 100)


@pytest.mark.parametrize("depth_leads", [False, True])
def test_topic_sequence_offset_and_callback_skew_do_not_stop_output(depth_leads):
    worker = _worker()
    leading = worker._depth_callback if depth_leads else worker._rgb_callback
    following = worker._rgb_callback if depth_leads else worker._depth_callback
    def msg(capture, depth):
        return _wire_image(capture, capture + (50 if depth else 0),
                           10. + capture * .1, depth=depth)
    leading(msg(1, depth_leads))
    for capture in range(1, 5):
        leading(msg(capture + 1, depth_leads))
        following(msg(capture, not depth_leads))
        raw = worker._latest_raw_frame()
        assert raw is not None
        assert np.all(raw["rgb_array"] == capture)
        assert np.all(raw["depth_array"] == capture)


def test_equal_ros_topic_counters_cannot_pair_different_capture_stamps():
    worker = _worker()
    worker._rgb_callback(_wire_image(100, 1, 10.))
    worker._depth_callback(_wire_image(101, 1, 10.05, depth=True))
    assert worker._latest_raw_frame() is None
    worker._depth_callback(_wire_image(100, 2, 10., depth=True))
    raw = worker._latest_raw_frame()
    assert np.all(raw["rgb_array"] == 100)
    assert np.all(raw["depth_array"] == 100)


def test_republished_capture_with_new_topic_counters_is_not_new_observation():
    worker = _worker()
    worker._rgb_callback(_wire_image(100, 1, 10.))
    worker._depth_callback(_wire_image(100, 1, 10., depth=True))
    assert worker._latest_raw_frame() is not None
    worker._rgb_callback(_wire_image(100, 21, 10.))
    worker._depth_callback(_wire_image(100, 11, 10., depth=True))
    assert worker._latest_raw_frame() is None
    # Independent publisher counters can restart; normalized capture time
    # still advances, and must allow the next real frame to reach inference.
    worker._rgb_callback(_wire_image(101, 1, 10.1))
    worker._depth_callback(_wire_image(101, 2, 10.1, depth=True))
    raw = worker._latest_raw_frame()
    assert np.all(raw["rgb_array"] == 101)
    assert np.all(raw["depth_array"] == 101)
