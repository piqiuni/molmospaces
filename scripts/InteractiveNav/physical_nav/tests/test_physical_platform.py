import pathlib
import sys
import unittest
import ast

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_protocol import command_blocked_packet, hello_packet, image_packet, validate_packet
from physical_consistency import bbox_iou, evaluate_detection, evaluate_frame, project_map_node
from safety_gate import ReadOnlySafetyGate
from physical_yoloe_bridge import _label, _rotation, _world_points
from runtime_state import RuntimeState
from physical_six_panel_server import _compact_qwen_context


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
    assert frame["counts"]["warn"] == 1

  def test_consistency_reprojects_graph_node_into_rgbd_frame(self):
    node = {"label": "chair", "aabb_center": [0., 0., 2.], "aabb_size": [.4, .4, .4]}
    camera = {"fx": 100., "fy": 100., "cx": 50., "cy": 50., "width": 100, "height": 100}
    projection = project_map_node(node, intrinsics=camera, telemetry={"position": [0., 0., 0.], "yaw": 0.})
    assert projection is not None
    detection = {"instance_id": "chair_1", "semantic_class": "chair", "bbox": projection["bbox"], "world_position": [0., 0., 2.], "camera_position": [0., 0., 2.], "depth_median_m": projection["depth_m"], "confidence": .9}
    frame = evaluate_frame([detection], graph={"nodes": [node]}, projection_context={"intrinsics": camera, "telemetry": {"position": [0., 0., 0.], "yaw": 0.}, "image_size": (100, 100)})
    assert frame["counts"]["pass"] == 1
    assert frame["detections"][0]["metrics"]["bbox_iou"] > .99
    assert frame["detections"][0]["metrics"]["map_projected_depth_abs_m"] == 0.0

  def test_yoloe_label_and_pose_lift(self):
    assert _label("refrigerator_door") == "fridge"
    points = _world_points(__import__("numpy").array([[0., 0., 1.]], dtype="float32"), {"position": [2., 3., 0.], "yaw": 0.0}, __import__("numpy").zeros(3), (0., 0., 0.))
    assert points[0].tolist() == [2.0, 3.0, 1.0]
    assert _rotation(0., 0., 0.).shape == (3, 3)

  def test_websocket_link_state_is_observable(self):
    state = RuntimeState()
    hello = hello_packet(host="go2", streams={"camera": "d435i"})
    state.link_connected(hello)
    state.link_packet("sensor_frame")
    snapshot = state.snapshot()
    assert snapshot["link"]["connected"] is True
    assert snapshot["link"]["last_hello"]["role"] == "go2_readonly_sensor"
    assert snapshot["link"]["last_packet_type"] == "sensor_frame"
    state.link_disconnected()
    assert state.snapshot()["link"]["connected"] is False

  def test_raw_and_map_aligned_detection_views_are_separate(self):
    state = RuntimeState()
    state.update_topic("detections", {"seq": 4, "detections": [{"semantic_class": "chair", "mask": {"rows": [1]}}]})
    state.update_topic("mapped_detections", {"seq": 4, "map_frame": "tf_frame_map", "detections": [{"semantic_class": "chair", "map_transform_status": "tf"}]})
    snapshot = state.snapshot()
    assert snapshot["detections"][0]["mask"]
    assert snapshot["mapped_detections"][0]["map_transform_status"] == "tf"
    assert snapshot["mapped_detection_meta"]["map_frame"] == "tf_frame_map"

  def test_qwen_context_compacts_sparse_masks_and_graph_interaction(self):
    snapshot = {
      "frame_seq": 7,
      "telemetry": {"position": [0., 0., 0.]},
      "detections": [{"semantic_class": "chair", "mask": {"rows": list(range(1000)), "cols": list(range(1000))}, "bbox": [1, 2, 3, 4]}],
      "graph": {"nodes": [{"id": "chair_1", "interaction": {"state": "unknown", "operation_history": list(range(1000))}}], "edges": []},
      "consistency": {"status": "warn", "detections": []},
    }
    compact = _compact_qwen_context(snapshot)
    assert "mask" not in compact["detections"][0]
    assert compact["detections"][0]["mask_summary"]["rows"] == 1000
    assert "operation_history" not in compact["graph"]["nodes"][0]["interaction"]
    assert len(__import__("json").dumps(compact)) < 2000
