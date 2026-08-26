import pathlib
import sys
import unittest
import ast

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_protocol import command_blocked_packet, hello_packet, image_packet, validate_packet
from physical_consistency import bbox_iou, evaluate_detection, evaluate_frame
from safety_gate import ReadOnlySafetyGate
from physical_yoloe_bridge import _label, _rotation, _world_points


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

  def test_go2_bridge_is_websocket_only_and_read_only(self):
    source = (ROOT / "go2_readonly_sensor_bridge.py").read_text()
    assert "websocket.create_connection" in source
    tree = ast.parse(source)
    imported = {alias.name.rsplit(".", 1)[-1] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert not imported.intersection({"rospy", "SportClient", "ObstaclesAvoidClient", "ChannelPublisher"})
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"Publisher", "publish"} for node in ast.walk(tree))


  def test_consistency_metrics(self):
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    report = evaluate_detection({"instance_id": "chair_1", "bbox": [0, 0, 10, 10], "confidence": .9}, projected_bbox=[0, 0, 10, 10])
    assert report["status"] == "pass"
    spatial = evaluate_detection({"instance_id": "chair_1", "world_position": {"x": 1., "y": 2., "z": .5}, "confidence": .9}, map_node={"centroid": [1., 2., .5]})
    assert spatial["metrics"]["map_distance_m"] == 0.0
    assert spatial["metrics"]["map_z_abs_m"] == 0.0
    lifted = evaluate_detection({"world_position": {"x": 0., "y": 0., "z": 1.}, "camera_position": {"x": 0., "y": 0., "z": 1.}, "depth_median_m": 1., "depth_valid_points": 20, "confidence": .9})
    assert lifted["metrics"]["rgbd_depth_lift_abs_m"] == 0.0
    frame = evaluate_frame([{"instance_id": "chair_1", "semantic_class": "chair", "confidence": .9}], graph={"nodes": []})
    assert frame["counts"]["pass"] == 1

  def test_yoloe_label_and_pose_lift(self):
    assert _label("refrigerator_door") == "fridge"
    points = _world_points(__import__("numpy").array([[0., 0., 1.]], dtype="float32"), {"position": [2., 3., 0.], "yaw": 0.0}, __import__("numpy").zeros(3), (0., 0., 0.))
    assert points[0].tolist() == [2.0, 3.0, 1.0]
    assert _rotation(0., 0., 0.).shape == (3, 3)
