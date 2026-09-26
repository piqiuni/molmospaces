"""The ROS perception adapter must not drop M1's capture-time evidence."""

import json
import threading
from types import SimpleNamespace

import pytest

import physical_ros_gateway as gateway
from physical_yoloe_bridge import _capture_observation_metadata
from semantic_mapping_py_pkg.image_frame_pairing import select_image_record
from semantic_mapping_py_pkg.portal_state_consensus import PortalStateConsensus


def _publish_report(monkeypatch, report):
    adapter = object.__new__(gateway.PhysicalRosGateway)
    adapter._last_detection_receipt = None
    adapter.world_frame = "map"
    adapter._debug_cloud_lock = threading.Lock()
    adapter._stable_world_boxes = lambda detections: detections
    adapter._apply_associated_portal_geometry = lambda *args: None
    adapter._queue_detection_overlay = lambda state: None
    adapter._post_mapped_state = lambda state: None
    # A later pose must not replace the detector's capture-time pose.
    adapter._telemetry = {"position": [99, 99, 0], "yaw": 2.5}
    messages = {"mapping": [], "attributes": []}
    for name, channel in (("detection_pub", "mapping"),
                          ("attribute_detection_pub", "attributes")):
        setattr(adapter, name, SimpleNamespace(
            publish=lambda data, key=channel: messages[key].append(json.loads(data)),
            get_num_connections=lambda: 1,
        ))
    monkeypatch.setattr(gateway.rospy, "logwarn_throttle",
                        lambda *args: pytest.fail(str(args)))
    adapter._publish_state(report)
    assert len(messages["mapping"]) == len(messages["attributes"]) == 1
    return messages


@pytest.mark.parametrize("legacy", [False, True])
def test_gateway_preserves_capture_pose_and_exact_image_identity(monkeypatch, legacy):
    raw = {"seq": 45, "stamp": 1788583565.125,
           "telemetry": {"position": [3., 4., .305], "yaw": .5}}
    meta = {"seq": raw["seq"], "stamp": raw["stamp"],
            **_capture_observation_metadata(raw, (720, 1280, 3)),
            "stamp_nsec": 125000000, "source_image_sequence": 45}
    report = {"detection_meta": meta, "detections": [],
              "observation_pose_xyyaw": [99, 99, 2.5]} if legacy else {
                  **meta, "detections": []}
    messages = _publish_report(monkeypatch, report)
    for channel in messages.values():
        envelope = channel[0]
        for key, value in meta.items():
            assert envelope[key] == value
        allowed, reason = PortalStateConsensus(confirmation_count=1).can_request(
            "door", capture_step=envelope["capture_step"],
            observation_pose_xyyaw=envelope["observation_pose_xyyaw"],
        )
        assert allowed and reason == "initial_evidence"
        record = {"stamp_key": (1788583565, 125000000),
                  "image_sequence": 45, "header_seq": 888}
        assert select_image_record(envelope, [record]) == (record, "matched")


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_capture_pose_stays_missing(monkeypatch, legacy):
    meta = {"seq": 45, "stamp": 10.0}
    report = {"detection_meta": meta, "detections": [],
              "observation_pose_xyyaw": [99, 99, 2.5]} if legacy else {
                  **meta, "detections": []}
    messages = _publish_report(monkeypatch, report)
    for channel in messages.values():
        envelope = channel[0]
        assert "observation_pose_xyyaw" not in envelope
        assert envelope["stamp"] == 10.0


def test_gateway_preserves_only_budget_deferred_2d_evidence_separately(monkeypatch):
    report = {"seq": 9, "stamp": 12.0, "detections": [
        {"semantic_class": "cabinet", "bbox": [10, 20, 30, 40],
         "geometry_skipped": True, "geometry_skip_reason": reason,
         "segmentation": {"unnecessary": "payload"}}
        for reason in ["max_geometry_instances", "geometry_budget_ms", "geometry_workers_busy", "insufficient_depth"]
    ]}
    messages = _publish_report(monkeypatch, report)
    envelope = messages["mapping"][0]
    assert envelope["detections"] == []
    deferred = envelope["geometry_deferred_detections"]
    assert len(deferred) == 3
    assert all("segmentation" not in d for d in deferred)
    assert all(d["bbox"] == [10, 20, 30, 40] for d in deferred)
