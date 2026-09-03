import pathlib
import sys
import unittest
import ast

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[2] / "Interactive-Nav-SG-nav" / "src" / "semantic_mllm_py_pkg" / "scripts"))

from physical_protocol import (
  command_blocked_packet,
  decode_wire_packet,
  encode_wire_packet,
  hello_packet,
  image_packet,
  validate_packet,
)
from physical_consistency import bbox_iou, evaluate_detection, evaluate_frame, project_map_node
from safety_gate import ReadOnlySafetyGate
from physical_yoloe_bridge import (
  YoloeWorker,
  _encode_detection_overlay,
  _encode_point_rows,
  _is_excluded_scene_label,
  _is_ground_like_box,
  _is_implausibly_large_object,
  _passes_class_plausibility,
  _label,
  _prepare_instance_points,
  _oriented_bounds,
  _rotation,
  _supported_euclidean_cluster_mask,
  _world_points,
)
from runtime_state import RuntimeState
from physical_nav_watchdog import HealthLimits, health_errors
from velocity_safety import VelocitySafetyConfig, VelocitySafetyLimiter
from interaction_policy import (
  InteractionPolicy,
  InteractionRequest,
  HumanAssistActuator,
  VisualMLLMVerifier,
)
from physical_six_panel_server import SixPanelRenderer, _HTML, _SHOWCASE_HTML, _WebHandler, _compact_qwen_context, _phone_stream_page
from showcase_pages import ACADEMIC_SHOWCASE_HTML, DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML


class PhysicalPlatformTests(unittest.TestCase):
  def test_compressed_sensor_wire_packet_round_trips_without_sample_changes(self):
    import os

    rgb = os.urandom(4096)
    depth = os.urandom(16384)
    packet = image_packet(
      seq=7,
      stamp=12.5,
      rgb_jpeg=rgb,
      depth_png=depth,
      width=640,
      height=480,
      camera_frame="d435i_color_optical_frame",
      depth_scale=.001,
      intrinsics={"fx": 1.0},
      color_depth_sync_ms=.2,
    )
    encoded = encode_wire_packet(packet)
    decoded = decode_wire_packet(encoded)
    assert decoded == packet
    assert len(encoded) < len(__import__("json").dumps(packet).encode("utf-8"))

  def test_unified_segmentation_keeps_multiple_depth_components(self):
    import numpy as np

    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    mask[60:90, 80:110] = 1
    depth = np.zeros(mask.shape, dtype=np.uint16)
    depth[10:40, 10:40] = 1000
    depth[60:90, 80:110] = 2000
    config = {
      "mask_component_min_area": 20,
      "point_stride": 1,
      "depth_scale": .001,
      "max_depth_m": 8.0,
      "min_valid_points": 8,
      "depth_band_lower_quantile": .02,
      "depth_band_upper_quantile": .98,
      "enable_euclidean_cluster": False,
    }
    filtered_mask, points, depths = _prepare_instance_points(
      mask, depth, (100., 100., 60., 50.), (0, 0, 120, 100), config
    )
    assert int(np.count_nonzero(filtered_mask[10:40, 10:40])) == 900
    assert int(np.count_nonzero(filtered_mask[60:90, 80:110])) == 900
    assert points.shape[0] == depths.shape[0] == 1800
    assert np.isclose(depths.min(), 1.0)
    assert np.isclose(depths.max(), 2.0)

  def test_supported_3d_cluster_does_not_crop_elongated_surface(self):
    import numpy as np

    points = np.stack(
      [np.linspace(0.0, 2.0, 101), np.zeros(101), np.ones(101)], axis=1
    ).astype(np.float32)
    keep = _supported_euclidean_cluster_mask(points, eps=.05, min_points=5)
    retained = points[keep]
    assert retained.shape[0] == points.shape[0]
    assert np.isclose(np.ptp(retained[:, 0]), 2.0)

  def test_oriented_bounds_follow_instance_point_cloud(self):
    import numpy as np

    line = np.stack([np.linspace(-1.0, 1.0, 101), np.zeros(101), np.zeros(101)], axis=1)
    center, size, quaternion = _oriented_bounds(line.astype(np.float32), {"bbox_quantile_lower": 0.0, "bbox_quantile_upper": 1.0})
    assert size[0] > 1.9
    assert size[1] >= 0.01 and size[2] >= 0.01
    assert np.isclose(np.linalg.norm(np.asarray(quaternion)), 1.0)
    assert np.isclose(quaternion[0], 0.0)
    assert np.isclose(quaternion[1], 0.0)

  def test_physics_laboratory_is_filtered_as_scene_label(self):
    assert _is_excluded_scene_label("physics_laboratory", {"excluded_label_tokens": ["physics_laboratory"]})

  def test_stormy_is_filtered_as_scene_label(self):
    assert _is_excluded_scene_label("stormy", {"excluded_label_tokens": ["stormy"]})

  def test_segmented_debug_points_use_compact_float32_transport(self):
    import base64
    import numpy as np

    points = np.asarray([[1.0, 2.0, 3.0], [-1.5, .25, 8.0]], dtype=np.float32)
    encoded = _encode_point_rows(points)
    decoded = np.frombuffer(base64.b64decode(encoded), dtype="<f4").reshape(-1, 3)
    assert np.allclose(decoded, points)
    assert len(encoded) < len(str(points.astype(float).tolist()))

  def test_yolo_overlay_uses_exact_rgb_receipt_and_draws_box(self):
    import base64
    import cv2
    import numpy as np

    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    encoded = _encode_detection_overlay(
      rgb,
      [{"semantic_class": "door", "confidence": .91, "bbox": [10, 15, 90, 70]}],
    )
    overlay = cv2.imdecode(
      np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert overlay is not None
    assert overlay.shape == rgb.shape
    assert int(np.count_nonzero(overlay)) > 0

  def test_yolo_worker_accepts_new_timestamps_after_source_sequence_reset(self):
    worker = object.__new__(YoloeWorker)
    worker.last_seq = -1
    worker.last_stamp = float("-inf")
    packet = lambda seq, stamp: {"seq": seq, "stamp": stamp, "rgb": "rgb", "depth": "depth"}

    assert worker._claim_frame(packet(134455, 100.0)) is True
    assert worker._claim_frame(packet(12, 101.0)) is True
    assert worker.last_seq == 12
    assert worker._claim_frame(packet(12, 101.0)) is False
    assert worker._claim_frame(packet(11, 100.5)) is False
    assert worker._claim_frame(packet(13, 102.0)) is True

  def test_physical_velocity_profile_scales_and_slews(self):
    limiter = VelocitySafetyLimiter(VelocitySafetyConfig())
    first = limiter.limit(.46, 0.0, 1.25, now=1.0)
    second = limiter.limit(.46, 0.0, 1.25, now=1.2)
    assert first == (.30, 0.0, .375)
    assert second[0] <= first[0] + .071
    assert second[2] <= first[2] + .101
    assert limiter.stop() == (0.0, 0.0, 0.0)

  def test_physical_velocity_profile_does_not_promote_terminal_linear_noise(self):
    limiter = VelocitySafetyLimiter(VelocitySafetyConfig(
      scale=1.0,
      max_linear_mps=.60,
      min_linear_mps=.40,
      linear_deadband_mps=.05,
    ))
    assert limiter.limit(.02, 0.0, -.35, now=1.0) == (0.0, 0.0, -.35)
    assert limiter.limit(-.02, 0.0, 0.0, now=2.0) == (0.0, 0.0, 0.0)
    assert limiter.limit(.06, 0.0, 0.0, now=3.0) == (.40, 0.0, 0.0)

  def test_external_mllm_verified_mode_is_accepted(self):
    from semantic_mllm_py_pkg.ablation import AblationConfig
    config = AblationConfig(module1="dynamic_mllm", module2="mllm_score", module3="external_mllm_verified")
    assert config.uses_mllm

  def test_physical_policy_request_normalizes_target_kind(self):
    request = InteractionRequest.from_payload({
      "command_id": "c1", "node_type": "portal", "action": "open",
      "object_id": "door_1",
    })
    assert request.target_kind == "door"
    fridge = InteractionRequest.from_payload({
      "command_id": "c2", "container_kind": "refrigerator", "action": "open",
    })
    assert fridge.target_kind == "fridge"

  def test_speech_transport_failure_is_retryable_not_object_failure(self):
    class VerifierMustNotRun:
      name = "must_not_run"

      def verify(self, request, cancel):
        raise AssertionError("verification must wait for successful speech")

    actuator = HumanAssistActuator(
      lambda _text, _wait: {"accepted": False, "reason": "no_speech_subscriber"},
      retry_count=0,
      retry_interval_s=0.0,
    )
    result = InteractionPolicy(actuator, VerifierMustNotRun()).execute(
      InteractionRequest("speech-transport", target_kind="door")
    )
    assert result.success is False
    assert result.detail["retryable"] is True
    assert result.detail["failure_reason"] == "interaction_transport_speech_unavailable"

  def test_m3_requires_three_seconds_of_open_evidence(self):
    calls = []
    response = {"choices": [{"message": {"content": '{"state":"open","confidence":0.9}'}}]}
    verifier = VisualMLLMVerifier(lambda: "data:image/jpeg;base64,AA==", lambda *a, **k: response, calls.append, stable_open_s=3.0, sample_period_s=.2, timeout_s=5.0)
    result = verifier.verify(InteractionRequest("c1", target_id="door_1"), lambda: False)
    assert result["success"] is True
    assert result["verification_source"] == "physical_visual_mllm"
    assert len(calls) >= 3
    assert calls[-1]["phase"] == "SUCCEEDED"

  def test_m3_publishes_target_and_schedule_before_call_then_temporarily_skips(self):
    events = []
    response = {"choices": [{"message": {"content": '{"state":"closed","confidence":0.95}'}}]}
    verifier = VisualMLLMVerifier(
      lambda: "data:image/jpeg;base64,AA==",
      lambda *a, **k: response,
      events.append,
      stable_open_s=.5,
      sample_period_s=.2,
      timeout_s=1.5,
      temporary_skip_s=7.0,
    )
    result = verifier.verify(
      InteractionRequest("c2", target_id="fridge_7", target_kind="fridge"),
      lambda: False,
    )
    assert events[0]["phase"] == "SCHEDULED"
    assert events[0]["target_kind"] == "fridge"
    assert events[0]["next_call_at"] <= events[0]["deadline_at"]
    assert any(event.get("phase") == "EVALUATING" for event in events)
    assert events[-1]["phase"] == "TIMEOUT"
    assert result["status"] == "TIMEOUT"
    assert result["temporary_skip_s"] == 7.0

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

  def test_fridge_plausibility_rejects_small_patches_and_wall_sized_boxes(self):
    import numpy as np

    config = {
      "class_plausibility": {
        "fridge": {
          "min_bbox_area_px": 3000,
          "min_bbox_side_px": 35,
          "min_height_m": .30,
          "max_height_m": 2.60,
          "max_horizontal_span_m": 1.80,
          "max_volume_m3": 4.0,
        }
      }
    }
    assert _passes_class_plausibility("fridge", [10, 10, 110, 180], np.array([.8, .7, 1.7]), config)
    assert not _passes_class_plausibility("fridge", [10, 10, 40, 60], np.array([.3, .2, .4]), config)
    assert not _passes_class_plausibility("fridge", [0, 0, 500, 400], np.array([3.0, .2, 2.0]), config)

  def test_physical_filter_excludes_lighting_and_scene_regions(self):
    import yaml
    from semantic_mapping_py_pkg.detection_filter import DetectionFilter

    config = yaml.safe_load((ROOT / "config" / "physical_nav.yaml").read_text())
    detection_filter = DetectionFilter(config["object_detection"]["detection_filter"])
    for label in ("lighting", "atrium", "floor", "server room", "carpet", "area rug", "doormat"):
      assert detection_filter.apply_one({"semantic_class_raw": label, "semantic_class": label}) is None
    assert detection_filter.apply_one({"semantic_class_raw": "door", "semantic_class": "door"}) is not None
    object_config = config["object_detection"]
    for label in ("wood_wall", "airport_terminal", "train interior", "elevator_lobby", "woven_carpet", "floor_mat"):
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

  def test_watchdog_detects_stale_camera_and_perception_receipts(self):
    import time

    now = time.time()
    state = RuntimeState()
    state.link_connected(hello_packet(host="go2", streams={"camera": "d435i"}))
    state.update_frame(frame_seq=0, frame_stamp=now)
    state.update_topic("detections", {"seq": 0, "stamp": now, "detections": []})
    payload = state.health_snapshot()
    assert health_errors(payload, HealthLimits()) == []

    payload["generated_at"] = now + 20.0
    errors = health_errors(payload, HealthLimits(frame_stale_s=5.0, perception_stale_s=15.0))
    assert any("D435i frame is stale" in error for error in errors)
    assert any("YOLOE receipt is stale" in error for error in errors)

  def test_watchdog_defaults_tolerate_brief_camera_interruptions(self):
    limits = HealthLimits()
    assert limits.frame_stale_s == 15.0

    launcher = (ROOT / "start_physical_nav.sh").read_text()
    assert 'PHYSICAL_NAV_WATCHDOG_FRAME_STALE_S:-15' in launcher
    assert 'PHYSICAL_NAV_WATCHDOG_FAILURE_LIMIT:-5' in launcher

  def test_launcher_supervises_critical_processes_and_service_detaches(self):
    launcher = (ROOT / "start_physical_nav.sh").read_text()
    service = (ROOT / "physical_nav_service.sh").read_text()
    assert "wait -n -p EXITED_PID" in launcher
    assert "physical_nav_watchdog.py" in launcher
    assert "gateway.log" in launcher and "yoloe.log" in launcher
    assert 'nohup setsid bash "${START_SCRIPT}"' in service
    assert 'kill -TERM -- "-${pid}"' in service
    assert "cleanup_pipeline_residuals" in service
    assert '"__name:=slam_gmapping"' in service
    assert '"__name:=move_base"' in service
    assert '"__name:=physical_nav_consistency"' in service

  def test_navigation_step_is_local_to_policy_session(self):
    state = RuntimeState()
    state.update_frame(frame_seq=90001)
    assert state.advance_navigation_step() == 0
    assert state.advance_navigation_step() == 1
    snapshot = state.snapshot()
    assert snapshot["frame_seq"] == 90001
    assert snapshot["navigation_step"] == 1
    assert SixPanelRenderer._physical_step(snapshot)["step_index"] == 1

  def test_panel_two_keeps_pending_post_interaction_subgoal_visible(self):
    snapshot = RuntimeState().snapshot()
    snapshot["navigation"] = {
      "selection": {
        "active": False,
        "terminal": {"candidate_id": "interaction:portal_6:open"},
      },
      "decision_trace": {
        "event": "post_interaction_refresh",
        "phase": "started",
        "pending_post_interaction_traversal": {
          "candidate_id": "traverse:portal_6:event_2",
          "behavior_type": "NAVIGATE",
          "target_id": "portal_6",
          "target_name": "door #6",
          "goal_xyyaw": [1.5, 2.0, 0.0],
        },
      },
    }

    step = SixPanelRenderer._physical_step(snapshot)

    assert step["semantic_selection"]["active"] is True
    assert step["semantic_selection"]["selection_state"] == "started"
    assert step["semantic_selection"]["goal_xyyaw"] == [1.5, 2.0, 0.0]

  def test_panel_two_does_not_revive_terminal_pending_subgoal(self):
    snapshot = RuntimeState().snapshot()
    pending_id = "traverse:portal_6:event_2"
    snapshot["navigation"] = {
      "selection": {
        "active": False,
        "terminal": {"candidate_id": pending_id},
      },
      "decision_trace": {
        "phase": "released",
        "pending_post_interaction_traversal": {
          "candidate_id": pending_id,
          "behavior_type": "NAVIGATE",
          "goal_xyyaw": [1.5, 2.0, 0.0],
        },
      },
    }

    step = SixPanelRenderer._physical_step(snapshot)

    assert step["semantic_selection"]["active"] is False

  def test_state_summary_excludes_large_detection_overlay(self):
    summary = _WebHandler._state_summary({
        "detection_meta": {
            "seq": 42,
            "stamp": 12.5,
            "camera_frame": "camera",
            "overlay_jpeg": "x" * 100_000,
        },
        "detections": [],
    })
    assert summary["detection_meta"] == {
        "seq": 42,
        "stamp": 12.5,
        "camera_frame": "camera",
    }

  def test_ros_state_includes_clouds_but_browser_summary_stays_light(self):
    snapshot = {
      "detections": [{
        "semantic_class": "door",
        "camera_segment_points_f32": "camera-cloud",
        "world_segment_points_f32": "world-cloud",
        "segment_point_count": 123,
      }],
    }
    browser = _WebHandler._state_summary(snapshot)
    ros = _WebHandler._state_summary(snapshot, include_ros_clouds=True)
    assert "camera_segment_points_f32" not in browser["detections"][0]
    assert ros["detections"][0]["camera_segment_points_f32"] == "camera-cloud"
    assert ros["detections"][0]["world_segment_points_f32"] == "world-cloud"
    assert ros["detections"][0]["segment_point_count"] == 123

  def test_dashboard_uses_native_mjpeg_without_canvas_snapshot_decoding(self):
    assert "<img id='overview'" in _HTML
    assert "src='/stream.mjpg'" in _HTML
    assert "<img id='panel1'" in _HTML
    assert "<img id='panel5'" in _HTML
    assert "fetch('/snapshot.jpg?ts='" not in _HTML
    assert "createImageBitmap" not in _HTML

  def test_showcase_keeps_selected_panels_and_agent_summary(self):
    assert "id='view1'" in _SHOWCASE_HTML
    assert "id='view3'" in _SHOWCASE_HTML
    assert "id='view6'" in _SHOWCASE_HTML
    assert '[["view1",0,0],["view3",960,0],["view6",960,270]]' in _SHOWCASE_HTML
    assert "Go2 实时状态" in _SHOWCASE_HTML
    assert "汇总的 MLLM 调用" in _SHOWCASE_HTML
    assert "Agent 行为对话" in _SHOWCASE_HTML
    assert "fetch('/api/state-summary?ts='" in _SHOWCASE_HTML

  def test_dark_and_light_showcases_share_live_information_architecture(self):
    for html in (DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML):
      assert "id='view1'" in html
      assert "id='view3'" in html
      assert "id='view6'" in html
      assert "Go2 当前状态" in html
      assert "MLLM 调用汇总" in html
      assert "Agent 当前行为时间线" in html
      assert "fetch('/snapshot.jpg?t='" in html
      assert "fetch('/api/state-summary?t='" in html
      assert "fetch('/original-panel3.jpg?t='" in html
      assert "fetch('/original-panel6.jpg?t='" in html
      assert "3D" not in html
    assert "class='dark'" in DARK_SHOWCASE_HTML
    assert "class='light'" in LIGHT_SHOWCASE_HTML
    assert "debug-call-thumb" in DARK_SHOWCASE_HTML
    assert "真实 M1 输入" in DARK_SHOWCASE_HTML
    assert "候选 subgoal" in DARK_SHOWCASE_HTML
    assert "/api/m1-input-image?key=" in DARK_SHOWCASE_HTML
    assert "M3 为规则验证，不调用模型" not in DARK_SHOWCASE_HTML
    assert "aspect-ratio:4/3" in DARK_SHOWCASE_HTML
    assert "container.scrollTop=0" in DARK_SHOWCASE_HTML
    assert "request_sequence||0" in DARK_SHOWCASE_HTML
    assert "debug-call-thumb" not in LIGHT_SHOWCASE_HTML
    assert "/camera-box-overlay.jpg" in DARK_SHOWCASE_HTML
    assert "/camera-box-overlay.jpg" not in LIGHT_SHOWCASE_HTML

  def test_academic_showcase_has_four_research_panels_and_live_endpoints(self):
    html = ACADEMIC_SHOWCASE_HTML
    for panel in (">A<", ">B<", ">C<", ">D<"):
      assert panel in html
    assert "开放词汇感知" in html
    assert "在线语义地图" in html
    assert "分层交互 Graph" in html
    assert "Agent 决策" in html
    assert "/snapshot.jpg" in html
    assert "/api/state-summary" in html
    assert "/original-panel3.jpg" in html
    assert "/original-panel6.jpg" in html

  def test_dark_showcase_has_phone_stream_switch_and_navigation_task(self):
    html = DARK_SHOWCASE_HTML
    for value in ("source-phone", "perception-stream-grid", "关闭手机音视频", "phone-qr.png", "/api/phone-status", "/phone-frame.jpg", "/phone-audio.pcm", "/api/recording-to-mp4", "当前导航任务", "交互导航探索", "record-page", "getDisplayMedia", "MediaRecorder", "录制完整页面"):
      assert value in html
    assert "10.100.5.3:8767/phone-stream" in html
    assert "source-d435i" not in html
    assert "source-phone" not in LIGHT_SHOWCASE_HTML
    phone = _phone_stream_page("http://10.100.5.3:8765/phone-stream")
    assert "getUserMedia" in phone
    assert "/api/phone-frame" in phone
    assert "scheduleFrameUploads" in phone
    assert "20 FPS" in phone
    assert "&fps=" in phone
    assert "ideal:960,max:960" in phone
    assert "ideal:540,max:540" in phone
    assert "录制 MP4" in phone
    assert "MediaRecorder" in phone
    assert "video/mp4" in phone
    assert "/api/recording-to-mp4" in phone
    assert "audio/pcm" in phone
    assert "sampleRate:{ideal:48000}" in phone
    assert "echoCancellation:false" in phone
    assert "转换文件不会留在服务器" in phone

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
    assert image.shape[:2] == (720, 1920)

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

  def test_live_segmentation_draws_all_retained_components(self):
    import numpy as np

    panel = np.zeros((100, 120, 3), dtype=np.uint8)
    detection = {
      "semantic_class": "fridge",
      "confidence": .8,
      "bbox": [5, 5, 115, 95],
      "mask_polygons": [
        [[10, 10], [30, 10], [30, 30], [10, 30]],
        [[80, 60], [105, 60], [105, 85], [80, 85]],
      ],
    }
    SixPanelRenderer._draw_live_detections(panel, [detection], (100, 120))
    assert np.any(panel[20, 20] != 0)
    assert np.any(panel[70, 90] != 0)
    assert np.all(panel[45, 60] == 0)

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
    assert "refreshStill('#panel1','/camera-overlay.jpg')" in _HTML

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

  def test_room_panel_draws_portal_with_graph_obb_yaw(self):
    import math
    from unittest.mock import patch

    import cv2
    import numpy as np
    from offline_semantic_renderer import OfflineSixPanelRenderer, RawGrid, TransformResolver

    grid = RawGrid(
      np.zeros((20, 20), dtype=np.int32), 20, 20, .2,
      "tf_frame_map", -2., -2., 0.,
    )
    renderer = OfflineSixPanelRenderer(
      transforms=TransformResolver([], map_frame="tf_frame_map", odom_frame="tf_frame_odom")
    )
    node = {
      "id": "portal_42", "type": "portal", "label": "door",
      "aabb_center": [0., 0., 1.], "aabb_size": [1.2, .2, 2.],
      "yaw": math.pi / 2., "attributes": {"instance_id": "track_0042"},
    }
    with patch("offline_semantic_renderer.cv2.polylines", wraps=cv2.polylines) as draw:
      renderer.render_room_panel(
        grid, None, (200, 200),
        {"unified_graph": {"nodes": [node]}, "observed_instance_ids": ["track_0042"]},
        1, (-2., -2., 2., 2.),
      )

    corners = draw.call_args_list[-1].args[1][0]
    x_span = int(corners[:, 0].max() - corners[:, 0].min())
    y_span = int(corners[:, 1].max() - corners[:, 1].min())
    assert y_span > x_span

  def test_room_panel_falls_back_to_current_physical_obb_yaw(self):
    import math
    from offline_semantic_renderer import _resolved_node_yaw

    node = {
      "id": "portal_track_0001", "type": "portal", "label": "portal",
      "aabb_center": [6.09, -.86, 1.5],
      "attributes": {
        "candidate_labels": ["door"],
        # Legacy missing-orientation placeholder from the current graph.
        "orientation": [0., 0., 0., 1.],
      },
    }
    detection_yaw = -.123
    detection = {
      "semantic_class": "door",
      "world_box3d_center": [6.01, -.84, 1.5],
      "world_box3d_orientation": [
        0., 0., math.sin(detection_yaw / 2.), math.cos(detection_yaw / 2.)
      ],
    }
    assert _resolved_node_yaw(node, [detection]) == __import__("pytest").approx(detection_yaw)

  def test_room_panel_draws_every_semantic_subgoal_with_occ_style(self):
    from unittest.mock import patch

    import numpy as np
    from offline_semantic_renderer import OfflineSixPanelRenderer, RawGrid, TransformResolver

    grid = RawGrid(np.zeros((20, 20), dtype=np.int32), 20, 20, .2,
                   "tf_frame_map", -2., -2., 0.)
    renderer = OfflineSixPanelRenderer(
      transforms=TransformResolver([], map_frame="tf_frame_map", odom_frame="tf_frame_odom")
    )
    step = {"semantic_candidates": {"candidates": [
      {"candidate_id": "frontier:1", "behavior_type": "EXPLORE", "goal_xyyaw": [-1., 0., 0.]},
      {"candidate_id": "interaction:door:open", "behavior_type": "INTERACT", "goal_xyyaw": [1., 0., 1.]},
    ]}}
    with patch("offline_semantic_renderer._draw_subgoal_marker") as marker:
      renderer.render_room_panel(grid, None, (200, 200), step, 1, (-2., -2., 2., 2.))
    assert marker.call_count == 2

  def test_occ_panel_links_interaction_subgoal_to_target_center(self):
    from unittest.mock import patch

    import numpy as np
    from offline_semantic_renderer import OfflineSixPanelRenderer, RawGrid, TransformResolver

    grid = RawGrid(np.zeros((20, 20), dtype=np.int32), 20, 20, .2,
                   "tf_frame_map", -2., -2., 0.)
    renderer = OfflineSixPanelRenderer(
      transforms=TransformResolver([], map_frame="tf_frame_map", odom_frame="tf_frame_odom")
    )
    step = {"semantic_candidates": {"candidates": [{
      "candidate_id": "interaction:door:open",
      "behavior_type": "INTERACT",
      "goal_xyyaw": [1., 0., 0.],
      "metadata": {"portal_aabb_center_xy": [0., 0.]},
    }]}}
    with patch("offline_semantic_renderer._draw_interaction_target_link") as link:
      renderer.render_map_panel(
        grid, (200, 200), step, 1, title="OCC", kind="occupancy",
        world_bounds=(-2., -2., 2., 2.), draw_semantic_candidates=True,
        draw_interaction_target_links=True,
      )
    assert link.call_count == 1

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
    system_ros_python = (ROOT / "physical_ros_python.sh").read_text()
    override = (ROOT / "config" / "semantic_shadow_override.yaml").read_text()
    all_launcher = (ROOT / "physical_nav_all.sh").read_text()
    service = (ROOT / "physical_nav_service.sh").read_text()
    assert 'name="interaction_attribute_inference"' in launch
    assert 'file="$(find nav_pkg)/launch/nav.launch"' in launch
    assert '<param name="scan_filter_tolerance_sec" value="0.15"/>' in launch
    assert '<param name="max_odom_cloud_time_diff" value="0.15"/>' in launch
    assert '<param name="maxUrange" value="7.9"/>' in launch
    assert '<param name="pointcloud_scan_range_max" value="8.0"/>' in launch
    assert '<param name="max_depth_m" value="8.0"/>' in launch
    assert '<param name="no_return_depth_m" value="8.05"/>' in launch
    assert '<param name="pointcloud_scan_min_support_neighbors" value="0"/>' in launch
    assert '<param name="enable_local_overwrite" value="true"/>' in launch
    assert '<param name="local_overwrite_radius" value="7.9"/>' in launch
    assert '<param name="local_overwrite_ttl_sec" value="2.0"/>' in launch
    assert '<arg name="override_config_file" value="$(arg semantic_override_config)"/>' in launch
    assert '<arg name="override_config_file" value="$(arg move_base_override_config)"/>' in launch
    assert '<remap from="/cmd_vel" to="/physical_nav/shadow_cmd_vel"/>' in launch
    assert '<remap from="/semantic_mapping/attribute_refresh_requests" to="/physical_nav/attribute_refresh_requests"/>' in launch
    assert '<param name="topics/object_detections" value="/physical_nav/tracked_detections"/>' in launch
    assert 'launch-prefix="$(arg system_ros_python)"' in launch
    assert 'system_ros_python:="${PHYSICAL_NAV_SYSTEM_ROS_PYTHON:-${ROOT_DIR}/physical_ros_python.sh}"' in start
    assert '${PHYSICAL_NAV_ROSLAUNCH_STDOUT:-${LOG_DIR}/roslaunch.log}' in start
    assert '*/miniconda3/envs/*/lib/python*/site-packages' in system_ros_python
    assert '/home/user/miniconda3/envs/mlspaces/lib/python3.11/site-packages:' not in system_ros_python
    assert '<param name="min_visible_pixels" value="1024"/>' in launch
    assert '<param name="min_bbox_area_px" value="1600"/>' in launch
    assert 'SEMANTIC_OVERRIDE_CONFIG="${PHYSICAL_NAV_SEMANTIC_OVERRIDE_CONFIG:-${ROOT_DIR}/config/semantic_shadow_override.yaml}"' in start
    assert 'semantic_override_config:="${SEMANTIC_OVERRIDE_CONFIG}"' in start
    assert 'nohup setsid "${GATEWAY_PYTHON}"' in start
    assert '*"start_physical_nav.sh"*' in service
    assert service.index('cleanup_pipeline_residuals\n  rm -f "${PID_FILE}"') > service.index('start_service()')
    assert 'interaction_semantic_types: ["door", "fridge"]' in override
    assert 'drawer_cabinet' not in override
    assert 'min_obstacle_clearance_m: 0.80' in override
    assert 'portal_traversal_clearance_margin_m: 0.55' in override
    assert 'rear_goal_prerotate_enabled: false' in override
    assert 'start_go2_control.py ${motion_flag} --no-restart' in all_launcher
    assert 'start_motion_control' in all_launcher
    assert '--bridge-ready-file "${ready_file}"' in all_launcher
    assert 'MOTION_MAX_VX="${PHYSICAL_NAV_MOTION_MAX_VX:-0.6}"' in all_launcher
    assert 'MOTION_MAX_WZ="${PHYSICAL_NAV_MOTION_MAX_WZ:-1.3}"' in all_launcher
    assert '--bridge-arg=--max-vx --bridge-arg="${max_vx}"' in all_launcher
    assert '--bridge-arg=--max-wz --bridge-arg="${max_wz}"' in all_launcher
    assert 'motion_mismatch' in all_launcher
    assert 'Go2 control bridge did not become ready' in all_launcher
    assert 'did not connect to policy port' in all_launcher
    assert 'refusing read-only launch: an existing Go2 bridge has --enable-motion' in all_launcher
    assert 'move_base_override_config:="${ROOT_DIR}/config/physical_move_base_override.yaml"' in start
    import yaml
    local_override = yaml.safe_load((ROOT / "config" / "physical_move_base_override.yaml").read_text())
    semantic_override = yaml.safe_load((ROOT / "config" / "semantic_shadow_override.yaml").read_text())
    physical_config = yaml.safe_load((ROOT / "config" / "physical_nav.yaml").read_text())
    assert local_override["DWAPlannerROS"]["xy_goal_tolerance"] == 0.30
    assert local_override["DWAPlannerROS"]["yaw_goal_tolerance"] == 0.15
    assert local_override["local_costmap"]["width"] == 8.0
    assert local_override["local_costmap"]["height"] == 8.0
    assert local_override["local_costmap"]["obstacle_range"] == 8.0
    assert local_override["local_costmap"]["obstacle_layer"]["local_obstacles"]["observation_persistence"] == 0.0
    assert local_override["local_costmap"]["obstacle_layer"]["local_obstacles"]["max_observation_age"] == 0.5
    assert local_override["local_costmap"]["obstacle_layer"]["local_obstacles"]["clearing"] is True
    assert local_override["local_costmap"]["inflation_layer"]["inflation_radius"] == 0.50
    assert local_override["global_costmap"]["inflation_layer"]["inflation_radius"] == 0.50
    assert local_override["local_costmap"]["obstacle_layer"]["local_obstacles"]["observation_persistence"] == 0.0
    assert local_override["local_costmap"]["obstacle_layer"]["footprint_clearing_enabled"] is True
    assert local_override["local_costmap"]["obstacle_layer"]["obstacle_reset_interval"] == 2.0
    assert local_override["DWAPlannerROS"]["max_vel_x"] == 0.56
    assert local_override["DWAPlannerROS"]["max_vel_trans"] == 0.60
    assert local_override["DWAPlannerROS"]["max_vel_theta"] == 1.30
    assert local_override["DWAPlannerROS"]["min_vel_theta"] == 0.55
    assert local_override["DWAPlannerROS"]["rear_path_rotate_speed"] == 1.30
    assert semantic_override["candidate"]["interaction_ready_yaw_tolerance_rad"] == 0.15
    assert semantic_override["candidate"]["portal_approach_standoff_offsets_m"] == [0.0, 0.20, 0.40]
    assert semantic_override["candidate"]["portal_approach_tangent_offsets_m"] == [0.0, -0.20, 0.20]
    assert semantic_override["candidate"]["portal_approach_yaw_offsets_rad"] == [0.0]
    assert semantic_override["candidate"]["portal_opposite_side_fallback_enabled"] is False
    assert semantic_override["candidate"]["portal_require_reference_yaw"] is True
    assert semantic_override["candidate"]["portal_side_hysteresis_m"] == 0.50
    assert semantic_override["candidate"]["portal_goal_require_known_free"] is True
    assert semantic_override["executor"]["explore_terminal_yaw_enabled"] is True
    assert semantic_override["executor"]["explore_terminal_xy_tolerance_m"] == 0.30
    assert semantic_override["executor"]["explore_terminal_yaw_tolerance_rad"] == 0.15
    assert physical_config["semantic_map"]["object_portal_cross_view_match_enabled"] is True
    assert physical_config["velocity_safety"]["max_linear_mps"] == 0.60
    assert physical_config["velocity_safety"]["min_linear_mps"] == 0.40
    assert physical_config["velocity_safety"]["linear_deadband_mps"] == 0.05
    assert physical_config["velocity_safety"]["max_angular_rps"] == 1.30
    assert physical_config["velocity_safety"]["min_angular_rps"] == 0.55
    assert '--continuous-ttl-ms "${PHYSICAL_NAV_MOTION_TTL_MS:-500}"' in all_launcher
    assert '--ros-command-refresh-hz "${PHYSICAL_NAV_MOTION_REFRESH_HZ:-20}"' in all_launcher
    assert physical_config["interaction_policy"]["speech_subscriber_wait_s"] == 60.0

  def test_runtime_state_extracts_bounded_m1_input_image(self):
    from runtime_state import RuntimeState

    state = RuntimeState()
    encoded = __import__("base64").b64encode(b"jpeg-bytes").decode("ascii")
    state.add_mllm_event({
      "stage": "M1", "episode_id": "ep", "request_sequence": 7,
      "timestamp": 1.25,
      "m1_input_image_data_url": "data:image/jpeg;base64," + encoded,
      "m1_input_bbox": [1, 2, 30, 40],
    })
    event = state.snapshot()["mllm_events"][0]
    assert "m1_input_image_data_url" not in event
    assert state.m1_input_image(event["m1_input_image_key"]) == b"jpeg-bytes"

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
