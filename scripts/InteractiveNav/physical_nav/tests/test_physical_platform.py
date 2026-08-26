import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_protocol import command_blocked_packet, hello_packet, image_packet, validate_packet
from physical_consistency import bbox_iou, evaluate_detection, evaluate_frame
from safety_gate import ReadOnlySafetyGate


class PhysicalPlatformTests(unittest.TestCase):
  def test_sensor_packet_contract(self):
    packet = image_packet(seq=3, stamp=1.0, rgb_jpeg=b"rgb", depth_png=b"depth", width=2, height=2,
                          camera_frame="d435i_color_optical_frame", depth_scale=.001,
                          intrinsics={"fx": 1, "fy": 1, "cx": 1, "cy": 1}, color_depth_sync_ms=0.3)
    validate_packet(packet)
    assert packet["type"] == "sensor_frame"
    assert hello_packet(host="go2", streams={})["actuation_enabled"] is False


  def test_safety_gate_never_accepts_motion(self):
    gate = ReadOnlySafetyGate()
    response = gate.handle_intent({"action": "MOVE_FORWARD", "vx": 1.0})
    assert response["accepted"] is False
    assert gate.actuation_enabled is False
    assert gate.snapshot()["status"] == "READ_ONLY_BLOCKED"
    assert command_blocked_packet(seq=1, command={})["accepted"] is False


  def test_consistency_metrics(self):
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    report = evaluate_detection({"instance_id": "chair_1", "bbox": [0, 0, 10, 10], "confidence": .9}, projected_bbox=[0, 0, 10, 10])
    assert report["status"] == "pass"
    frame = evaluate_frame([{"instance_id": "chair_1", "semantic_class": "chair", "confidence": .9}], graph={"nodes": []})
    assert frame["counts"]["pass"] == 1
