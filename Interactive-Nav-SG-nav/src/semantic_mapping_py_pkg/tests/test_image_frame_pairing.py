from pathlib import Path
import sys

import pytest

PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))

from semantic_mapping_py_pkg.image_frame_pairing import (
    payload_capture_step,
    payload_image_size,
    select_image_record,
)


def _record(name, *, step, stamp):
    return {
        "image": name,
        "capture_step": step,
        "header_seq": step,
        "image_sequence": step,
        "stamp_key": stamp,
    }


def test_pairing_uses_detection_identity_instead_of_latest_image():
    records = [
        _record("step_10", step=10, stamp=(100, 10)),
        _record("step_11", step=11, stamp=(100, 11)),
    ]

    selected, reason = select_image_record(
        {"capture_step": 10, "stamp_sec": 100, "stamp_nsec": 10}, records
    )

    assert reason == "matched"
    assert selected["image"] == "step_10"


def test_pairing_rejects_conflicting_step_and_stamp():
    records = [
        _record("step_10", step=10, stamp=(100, 10)),
        _record("step_11", step=11, stamp=(100, 11)),
    ]

    selected, reason = select_image_record(
        {"capture_step": 10, "stamp_sec": 100, "stamp_nsec": 11}, records
    )

    assert selected is None
    assert reason == "capture_step_image_not_found"


def test_payload_step_and_image_size_are_public_envelope_fields():
    assert payload_capture_step({"capture_step": 516, "frame_index": 181}) == 516
    assert payload_image_size({"image_size": [1024, 576]}) == (1024, 576)


def test_task_step_is_not_ros_publication_sequence():
    record = _record("rgb_923", step=923, stamp=(1789304037, 594208001))
    record.pop("capture_step")
    record.update(header_seq=925, image_sequence=925)
    selected, reason = select_image_record(
        {"capture_step": 923, "stamp_sec": 1789304037, "stamp_nsec": 594208001},
        [record],
    )
    assert reason == "matched"
    assert selected is record


def test_real_rospy_serialization_rewrites_seq_without_breaking_pairing():
    from io import BytesIO

    rospy = pytest.importorskip("rospy")
    from rospy.msg import serialize_message
    from sensor_msgs.msg import Image

    message = Image()
    message.header.seq = 923
    message.header.stamp = rospy.Time(1789304037, 594208001)
    serialize_message(BytesIO(), 925, message)
    assert message.header.seq == 925
    record = {
        "image_sequence": message.header.seq,
        "header_seq": message.header.seq,
        "stamp_key": (message.header.stamp.secs, message.header.stamp.nsecs),
    }
    assert select_image_record(
        {"capture_step": 923, "stamp_sec": 1789304037, "stamp_nsec": 594208001},
        [record],
    ) == (record, "matched")


def test_legacy_float_stamp_accepts_only_rounding_not_frame_delay():
    record = _record("correct", step=923, stamp=(1789304037, 594208001))
    next_frame = _record("latest", step=924, stamp=(1789304037, 600000000))
    selected, reason = select_image_record(
        {"capture_step": 923, "stamp_sec": 1789304037.594208}, [record, next_frame]
    )
    assert reason == "matched"
    assert selected is record
    assert select_image_record(
        {"capture_step": 923, "stamp_sec": 1789304037.594208}, [next_frame]
    ) == (None, "image_stamp_mismatch")


def test_float_rounding_with_two_distinct_frames_is_rejected_as_ambiguous():
    records = [
        _record("a", step=1, stamp=(1789304037, 594208001)),
        _record("b", step=2, stamp=(1789304037, 594208002)),
    ]
    assert select_image_record(
        {"stamp_sec": 1789304037.594208}, records
    ) == (None, "image_frame_identity_ambiguous")


def test_exact_integer_stamp_does_not_allow_rounding_tolerance():
    record = _record("wrong", step=1, stamp=(1789304037, 594208001))
    assert select_image_record(
        {"stamp_sec": 1789304037, "stamp_nsec": 594208000}, [record]
    ) == (None, "image_stamp_mismatch")


def test_explicit_source_sequence_must_agree_with_stamp():
    record = _record("a", step=1, stamp=(100, 0))
    assert select_image_record(
        {"image_sequence": 2, "stamp_sec": 100, "stamp_nsec": 0}, [record]
    ) == (None, "image_sequence_mismatch")


def test_step_only_cannot_be_inferred_from_matching_ros_header():
    record = {"header_seq": 923, "image_sequence": 923, "stamp_key": (100, 0)}
    assert select_image_record({"capture_step": 923}, [record]) == (
        None, "capture_step_image_identity_unavailable"
    )


def test_zero_capture_step_is_valid_with_explicit_frame_identity():
    record = _record("initial", step=0, stamp=(100, 0))
    assert select_image_record({"capture_step": 0}, [record]) == (record, "matched")
