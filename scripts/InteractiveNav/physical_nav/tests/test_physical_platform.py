import pathlib
import sys
import unittest
import ast

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_protocol import command_blocked_packet, hello_packet, image_packet, validate_packet
from physical_consistency import bbox_iou, evaluate_detection, evaluate_frame, project_map_node
from safety_gate import ReadOnlySafetyGate
from physical_yoloe_bridge import (
  _is_excluded_scene_label,
  _is_ground_like_box,
  _is_implausibly_large_object,
  _label,
  _rotation,
  _world_points,
)
from runtime_state import RuntimeState
from physical_six_panel_server import SixPanelRenderer, _HTML, _compact_qwen_context


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

  def test_physical_filter_excludes_lighting_and_scene_regions(self):
    import yaml
    from semantic_mapping_py_pkg.detection_filter import DetectionFilter

    config = yaml.safe_load((ROOT / "config" / "physical_nav.yaml").read_text())
    detection_filter = DetectionFilter(config["object_detection"]["detection_filter"])
    for label in ("lighting", "atrium", "floor", "server room"):
      assert detection_filter.apply_one({"semantic_class_raw": label, "semantic_class": label}) is None
    assert detection_filter.apply_one({"semantic_class_raw": "door", "semantic_class": "door"}) is not None
    object_config = config["object_detection"]
    for label in ("wood_wall", "airport_terminal", "train interior", "elevator_lobby"):
      assert _is_excluded_scene_label(label, object_config)
    assert not _is_excluded_scene_label("door", object_config)

  def test_ground_like_box_gate_preserves_compact_floor_level_objects(self):
    import numpy as np

    config = {"ground_reject_min_xy_span": 1.2, "ground_reject_max_height": .4, "ground_reject_max_center_z": .25}
    assert _is_ground_like_box(np.array([2., 0., -.1]), np.array([2.3, 2.1, .3]), config)
    assert not _is_ground_like_box(np.array([2., 0., .05]), np.array([.3, .2, .2]), config)
    assert _is_implausibly_large_object("wall_art", np.array([3., 2., 1.]), config)
    assert not _is_implausibly_large_object("door", np.array([3., 2., 1.]), config)

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

  def test_navigation_step_is_local_to_policy_session(self):
    state = RuntimeState()
    state.update_frame(frame_seq=90001)
    assert state.advance_navigation_step() == 0
    assert state.advance_navigation_step() == 1
    snapshot = state.snapshot()
    assert snapshot["frame_seq"] == 90001
    assert snapshot["navigation_step"] == 1
    assert SixPanelRenderer._physical_step(snapshot)["step_index"] == 1

  def test_dashboard_polls_snapshots_instead_of_opening_mjpeg_streams(self):
    assert "fetch('/snapshot.jpg?ts='" in _HTML
    assert "src='/stream.mjpg'" not in _HTML
    assert "src='/panel1.mjpg'" not in _HTML
    assert "src='/panel5.mjpg'" not in _HTML

  def test_raw_and_map_aligned_detection_views_are_separate(self):
    state = RuntimeState()
    state.update_topic("detections", {"seq": 4, "detections": [{"semantic_class": "chair", "mask": {"rows": [1]}}]})
    state.update_topic("mapped_detections", {"seq": 4, "map_frame": "tf_frame_map", "detections": [{"semantic_class": "chair", "map_transform_status": "tf"}]})
    snapshot = state.snapshot()
    assert snapshot["detections"][0]["mask"]
    assert snapshot["mapped_detections"][0]["map_transform_status"] == "tf"
    assert snapshot["mapped_detection_meta"]["map_frame"] == "tf_frame_map"

  def test_live_renderer_uses_canonical_offline_six_panel_layout(self):
    import cv2
    import numpy as np

    state = RuntimeState()
    state.rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    state.frame_seq = 7
    state.frame_stamp = 1.0
    state.telemetry = {"position": [0.0, 0.0, 0.0], "yaw": 0.0}
    state.occupancy = {
      "width": 20, "height": 20, "resolution": 0.1,
      "frame_id": "tf_frame_map",
      "origin": {"x": -1.0, "y": -1.0, "qw": 1.0},
      "data": [-1] * 400,
    }
    state.graph = {
      "graph_revision": 1,
      "nodes": [{"id": "chair_1", "type": "object", "label": "chair",
                 "aabb_center": [0.0, 0.0, 1.0], "aabb_size": [0.2, 0.2, 0.4],
                 "attributes": {"instance_id": "chair_1"},
                 "is_currently_visible": True}],
      "edges": [],
    }
    payload = SixPanelRenderer(state).render()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[:2] == (540, 1440)

  def test_live_panel_one_draws_segmentation_and_readable_track_label(self):
    import numpy as np

    panel = np.zeros((270, 480, 3), dtype=np.uint8)
    SixPanelRenderer._draw_live_detections(
      panel,
      [{"semantic_class": "door", "confidence": .9, "bbox": [10, 10, 40, 40],
        "mask": {"rows": [20, 20, 21, 21], "cols": [20, 21, 20, 21]}, "mask_area": 4}],
      (100, 100),
    )
    assert np.any(panel[50:65, 95:110] != 0)
    display = SixPanelRenderer._display_candidate({
      "target_id": "portal_track_0048", "target_name": "track_0048",
      "metadata": {"semantic_name": "portal"},
    })
    assert display["target_name"] == "door #0048"

  def test_six_panel_camera_is_box_only_and_enlargement_is_source_resolution(self):
    import cv2
    import numpy as np

    detection = {"semantic_class": "door", "confidence": .9, "bbox": [5, 5, 45, 45],
                 "mask": {"rows": [25], "cols": [25]}, "mask_area": 1}
    panel = np.zeros((50, 50, 3), dtype=np.uint8)
    SixPanelRenderer._draw_live_detections(panel, [detection], (50, 50), include_masks=False)
    assert np.all(panel[25, 25] == 0)
    assert np.any(panel[5, 5] != 0)

    state = RuntimeState()
    state.rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    state.detections = [detection]
    renderer = SixPanelRenderer(state)
    encoded = renderer.render_camera_overlay()
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (480, 640)
    assert renderer._canonical.transforms.odom_frame == "tf_frame_odom"
    assert "/camera-overlay.jpg?ts=" in _HTML

  def test_stale_global_costmap_display_falls_back_to_aligned_occ(self):
    import numpy as np
    from offline_semantic_renderer import RawGrid

    values = np.zeros((10, 10), dtype=np.int32)
    values[4, 6] = 100
    planning = RawGrid(values, 10, 10, .1, "tf_frame_map", -1., -1., 0.)
    stale = RawGrid(np.zeros((10, 10), dtype=np.int32), 10, 10, .1, "tf_frame_map", -1., -1., 0.)
    display = SixPanelRenderer._global_costmap_for_display(planning, stale)
    assert display.values[4, 6] == 100
    assert np.count_nonzero(display.values >= 100) == 1

  def test_physical_tracker_merges_only_strict_2d_3d_duplicates(self):
    from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore

    store = ObjectMapStore(
      match_distance=.5,
      duplicate_bbox_iou_threshold=.85,
      duplicate_3d_overlap_threshold=.15,
    )
    base = {
      "semantic_name": "chair", "aabb_center": [1., 2., .5],
      "aabb_size": [.4, .4, .4], "bbox_2d": [100, 100, 200, 240],
      "observation_count": 20, "conf": .9, "object_id": 1,
      "label_votes": {"chair": 10.}, "last_seen": 2., "is_confirmed": True,
    }
    duplicate = dict(base, object_id=2, aabb_center=[1.12, 2.02, .5],
                     bbox_2d=[102, 101, 201, 239], observation_count=10)
    distinct = dict(base, object_id=3, aabb_center=[1.2, 2.1, .5],
                    bbox_2d=[210, 100, 310, 240], observation_count=12)
    store.objects = [base, duplicate, distinct]
    store._merge_duplicate_tracks()
    assert {item["object_id"] for item in store.objects} == {1, 3}

  def test_physical_launch_contains_m1_and_shadow_costmaps(self):
    launch = (ROOT / "launch" / "physical_nav_readonly.launch").read_text()
    start = (ROOT / "start_physical_nav.sh").read_text()
    assert 'name="interaction_attribute_inference"' in launch
    assert 'file="$(find nav_pkg)/launch/nav.launch"' in launch
    assert '<param name="scan_filter_tolerance_sec" value="0.15"/>' in launch
    assert '<param name="max_odom_cloud_time_diff" value="0.15"/>' in launch
    assert '<param name="pointcloud_scan_min_support_neighbors" value="2"/>' in launch
    assert '<arg name="override_config_file" value="$(arg move_base_override_config)"/>' in launch
    assert '<remap from="/cmd_vel" to="/physical_nav/shadow_cmd_vel"/>' in launch
    assert '<remap from="/semantic_mapping/attribute_refresh_requests" to="/physical_nav/attribute_refresh_requests"/>' in launch
    assert 'semantic_override_config:="${ROOT_DIR}/config/semantic_shadow_override.yaml"' in start
    assert 'move_base_override_config:="${ROOT_DIR}/config/physical_move_base_override.yaml"' in start
    import yaml
    local_override = yaml.safe_load((ROOT / "config" / "physical_move_base_override.yaml").read_text())
    assert local_override["local_costmap"]["width"] == 8.0
    assert local_override["local_costmap"]["height"] == 8.0
    assert local_override["local_costmap"]["obstacle_range"] == 8.0

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
