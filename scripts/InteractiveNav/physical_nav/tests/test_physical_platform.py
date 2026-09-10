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
from physical_ros_gateway import PhysicalRosGateway, _semantic_detection_payload
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
from offline_semantic_renderer import (
  GlobalCostmapReplay,
  OfflineSixPanelRenderer,
  TransformResolver,
)
from showcase_pages import ACADEMIC_SHOWCASE_HTML, DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML


class PhysicalPlatformTests(unittest.TestCase):
  def test_semantic_detection_payload_omits_display_polygons_only(self):
    detection = {
      "semantic_class": "door",
      "bbox": [1, 2, 30, 40],
      "mask": {"rows": [3], "cols": [4]},
      "mask_polygon": [[1, 2], [3, 4]],
      "mask_polygons": [[[1, 2], [3, 4]]],
      "rgb_mask_polygons": [[[5, 6], [7, 8]]],
      "world_box3d_center": [1.0, 2.0, 0.5],
      "world_segment_points_f32": "packed-cloud",
    }
    projected = _semantic_detection_payload(detection)
    assert projected["semantic_class"] == "door"
    assert projected["mask"] == detection["mask"]
    assert projected["world_box3d_center"] == [1.0, 2.0, 0.5]
    for key in (
      "mask_polygon", "mask_polygons", "rgb_mask_polygons",
      "world_segment_points_f32",
    ):
      assert key not in projected
    # The projection is shallow and must not mutate the raw detector record.
    assert "mask_polygons" in detection

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

  def test_world_gateway_obb_follows_transformed_segment_points(self):
    import math
    import numpy as np

    rng = np.random.default_rng(4)
    local = np.column_stack((
      rng.uniform(-1.00, 1.00, 2000),
      rng.uniform(-0.20, 0.20, 2000),
      rng.uniform(-0.90, 0.90, 2000),
    ))
    yaw = 1.0
    c, s = math.cos(yaw), math.sin(yaw)
    axes = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    points = local @ axes.T + np.asarray([3.0, 2.0, 1.0])
    center, size, quaternion = PhysicalRosGateway._fit_world_obb(points)
    fitted_yaw = 2.0 * math.atan2(quaternion[2], quaternion[3])
    axial_error = abs(math.atan2(math.sin(fitted_yaw - yaw), math.cos(fitted_yaw - yaw)))
    axial_error = min(axial_error, abs(math.pi - axial_error))
    assert axial_error < 0.05
    assert np.allclose(center, [3.0, 2.0, 1.0], atol=0.03)
    assert max(size[0], size[1]) > 1.8

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
      min_linear_mps=0.0,
      linear_deadband_mps=.05,
    ))
    assert limiter.limit(.02, 0.0, -.35, now=1.0) == (0.0, 0.0, -.35)
    assert limiter.limit(-.02, 0.0, 0.0, now=2.0) == (0.0, 0.0, 0.0)
    assert limiter.limit(.06, 0.0, 0.0, now=3.0) == (.06, 0.0, 0.0)

  def test_physical_velocity_profile_preserves_moving_curvature(self):
    limiter = VelocitySafetyLimiter(VelocitySafetyConfig(
      scale=1.0,
      max_linear_mps=.60,
      min_linear_mps=.30,
      min_angular_rps=.60,
      max_linear_accel_mps2=10.0,
      max_angular_accel_rps2=10.0,
    ))
    # A gentle DWA arc must not become a tight circle merely because the
    # physical base has a minimum in-place turn speed.
    result = limiter.limit(.2269, 0.0, -.1138, now=1.0)
    assert result[0] == .30 and result[1] == 0.0
    self.assertAlmostEqual(result[2], -.15046, places=4)
    # The minimum angular speed remains applicable to a pure rotation.
    assert limiter.limit(0.0, 0.0, .10, now=2.0)[2] == .60

  def test_physical_velocity_profile_zero_command_stops_immediately(self):
    limiter = VelocitySafetyLimiter(VelocitySafetyConfig(
      scale=1.0,
      max_linear_mps=.60,
      min_linear_mps=0.0,
      linear_deadband_mps=.0,
      min_angular_rps=0.0,
      angular_deadband_rps=.0,
      max_linear_accel_mps2=.15,
    ))
    assert limiter.limit(.50, 0.0, 0.0, now=1.0) == (.50, 0.0, 0.0)
    assert limiter.limit(0.0, 0.0, 0.0, now=1.2) == (0.0, 0.0, 0.0)

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

  def test_m3_fresh_frame_guard_skips_duplicate_frame_without_model_call(self):
    """A latched RGB image must not count as a new post-action observation."""
    events = []
    model_calls = []
    response = {"choices": [{"message": {"content": '{"state":"closed","confidence":0.95}'}}]}
    first = {
      "image_data_url": "data:image/jpeg;base64,AA==",
      "seq": 7,
      "stamp": 101.0,
    }
    second = {**first, "seq": 8, "stamp": 101.1}
    samples = iter([first, first, second])

    def provider():
      return next(samples, second)

    def request_json(*args, **kwargs):
      model_calls.append((args, kwargs))
      return response

    verifier = VisualMLLMVerifier(
      provider,
      request_json,
      events.append,
      stable_open_s=.5,
      sample_period_s=.2,
      timeout_s=1.5,
      require_fresh_frames=True,
    )
    result = verifier.verify(
      InteractionRequest("fresh-frames", target_id="door_1"),
      lambda: False,
    )

    assert result["status"] == "TIMEOUT"
    # seq=7 is evaluated once, its replay is unknown, and seq=8 is evaluated
    # once; subsequent provider polls are the same seq=8 replay.
    assert len(model_calls) == 2
    duplicate_samples = [
      sample for sample in result["evidence"] if sample.get("reason") == "duplicate_frame"
    ]
    assert duplicate_samples
    assert duplicate_samples[0]["state"] == "unknown"
    assert duplicate_samples[0]["frame_seq"] == 7
    assert duplicate_samples[0]["fresh_frame"] is False
    assert any(
      event.get("result", {}).get("reason") == "duplicate_frame"
      and event.get("fresh_frame") is False
      for event in events
    )

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
    payload["frame_age_s"] = 20.0
    payload["perception_age_s"] = 20.0
    errors = health_errors(payload, HealthLimits(frame_stale_s=5.0, perception_stale_s=15.0))
    assert any("D435i frame is stale" in error for error in errors)
    assert any("YOLOE receipt is stale" in error for error in errors)

  def test_watchdog_defaults_tolerate_brief_camera_interruptions(self):
    limits = HealthLimits()
    assert limits.frame_stale_s == 15.0

    launcher = (ROOT / "start_physical_nav.sh").read_text()
    assert 'PHYSICAL_NAV_WATCHDOG_FRAME_STALE_S:-15' in launcher
    assert 'PHYSICAL_NAV_WATCHDOG_FAILURE_LIMIT:-5' in launcher

  def test_watchdog_prefers_gateway_local_age_over_remote_capture_clock(self):
    # Go2 and the policy host may not share a wall clock.  RuntimeState emits
    # frame_age_s computed at the gateway, which must override the legacy
    # frame_stamp subtraction (otherwise a healthy stream can be reported
    # stale solely because the two clocks differ).
    payload = {
      "link_connected": True,
      "frame_seq": 12,
      "frame_stamp": 1.0,
      "frame_age_s": 0.2,
      "perception_seq": 8,
      "perception_stamp": 1.0,
      "perception_age_s": 0.3,
      "generated_at": 1000.0,
    }
    assert health_errors(payload, HealthLimits(frame_stale_s=1.0, perception_stale_s=1.0)) == []

  def test_watchdog_malformed_optional_fields_are_reported_not_raised(self):
    payload = {
      "link_connected": True,
      "frame_seq": "not-a-sequence",
      "frame_stamp": "not-a-stamp",
      "perception_seq": "not-a-sequence",
      "perception_stamp": "not-a-stamp",
      "generated_at": "not-a-clock",
    }
    errors = health_errors(payload, HealthLimits())
    assert any("no D435i frame" in error for error in errors)
    assert any("YOLOE has not" in error for error in errors)

  def test_launcher_supervises_critical_processes_and_service_detaches(self):
    launcher = (ROOT / "start_physical_nav.sh").read_text()
    all_launcher = (ROOT / "physical_nav_all.sh").read_text()
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
    # Owned Go2 transports must be fully stopped before ownership files are
    # removed; otherwise the next start can silently reuse a stale bridge.
    assert "stop_remote_owned()" in all_launcher
    assert 'kill -KILL "${pid}"' in all_launcher
    assert "for _ in $(seq 1 40)" in all_launcher
    assert 'bash -s -- "${pid}" \\\n    >/dev/null 2>&1 <<\'REMOTE\'' in all_launcher

  def test_persistent_gateway_does_not_hold_supervisor_stack_lock(self):
    launcher = (ROOT / "start_physical_nav.sh").read_text()
    assert '>>"${LOG_DIR}/gateway.log" 2>&1 </dev/null 8>&- &' in launcher

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

  def test_runtime_summary_does_not_copy_raw_segmentation_payloads(self):
    """The browser summary must stay small when YOLO carries sparse masks."""
    state = RuntimeState()
    state.update_topic("detections", {
      "seq": 9,
      "detections": [{
        "semantic_class": "door",
        "confidence": .93,
        "bbox": [1, 2, 30, 40],
        "world_position": {"x": 1.0, "y": 2.0, "z": 0.5},
        "mask": {"rows": list(range(2000)), "cols": list(range(2000))},
        "mask_polygons": [[[float(i), float(i + 1)] for i in range(2000)]],
        "world_segment_points_f32": "packed-cloud",
      }],
    })
    summary = state.summary_snapshot()
    item = summary["detections"][0]
    assert item["semantic_class"] == "door"
    assert item["world_position"] == {"x": 1.0, "y": 2.0, "z": 0.5}
    assert "mask" not in item
    assert "mask_polygons" not in item
    assert "world_segment_points_f32" not in item
    # The full diagnostic snapshot remains available to ROS/debug consumers.
    assert len(state.snapshot()["detections"][0]["mask"]["rows"]) == 2000

  def test_navigation_updates_are_revisioned_without_summary_plan_copy(self):
    """Plans stay available to visualization but never block the status poll."""
    state = RuntimeState()
    state.update_navigation("candidates", {
      "candidates": [
        {"candidate_id": "door-1", "target_name": "door", "distance_m": 1.2},
      ],
    })
    before = state.visualization_revisions()["navigation_revision"]
    plan = {"stamp": 4.0, "poses": [{"x": float(i), "y": 0.0} for i in range(12000)]}
    state.update_navigation("global_plan", plan)
    revisions = state.visualization_revisions()
    assert revisions["navigation_revision"] == before + 1

    summary = state.summary_snapshot()
    assert "global_plan" not in summary["navigation"]
    assert summary["navigation"]["candidates"]["candidates"][0]["candidate_id"] == "door-1"

    # The full diagnostic snapshot remains an immutable copy for renderers;
    # mutating the original HTTP-decoded payload cannot alter it.
    full = state.snapshot()
    assert len(full["navigation"]["global_plan"]["poses"]) == 12000
    plan["poses"][0]["x"] = -1.0
    assert full["navigation"]["global_plan"]["poses"][0]["x"] == 0.0

  def test_offline_costmap_replay_copy_on_write_invalidates_raster_cache(self):
    """A local costmap patch must not leave the cached base image stale."""
    import tempfile
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
      root = pathlib.Path(directory)
      full_path = root / "full.png"
      update_path = root / "update.png"
      assert cv2.imwrite(str(full_path), np.ones((2, 2), dtype=np.uint8))
      assert cv2.imwrite(str(update_path), np.asarray([[100]], dtype=np.uint8))
      full = {
        "stage": "global_costmap",
        "receipt_id": "global:0",
        "image": str(full_path), "width": 2, "height": 2,
        "resolution": 1.0, "frame_id": "map",
        "origin": {"x": 0.0, "y": 0.0}, "png_value_offset": 1,
      }
      update = {
        "stage": "global_costmap_update",
        "receipt_id": "global_costmap_update:1",
        "image": str(update_path), "width": 1, "height": 1,
        "x": 0, "y": 0, "resolution": 1.0, "frame_id": "map",
        "origin": {"x": 0.0, "y": 0.0}, "png_value_offset": 1,
      }
      replay = GlobalCostmapReplay({"full": full, "update": update})
      first_grid = replay.grid_for(full, "global:0")
      renderer = OfflineSixPanelRenderer(
        transforms=TransformResolver([], map_frame="map", odom_frame="odom")
      )
      first_values = first_grid.values
      first_image = renderer._grid_base(first_grid, "costmap").copy()
      second_grid = replay.grid_for(full, "global_costmap_update:1")
      second_image = renderer._grid_base(second_grid, "costmap").copy()
      assert second_grid.values is not first_values
      assert not np.array_equal(first_image, second_image)

  def test_web_disabled_grid_callback_drops_before_list_materialization(self):
    """A headless launch must not copy full OccupancyGrid messages."""
    import threading
    from types import SimpleNamespace

    gateway = object.__new__(PhysicalRosGateway)
    gateway.web_state_enabled = False
    gateway.args = SimpleNamespace(occupancy_period=.2, room_grid_period=.5)
    gateway._last_grid_post = {}
    gateway._grid_post_lock = threading.Lock()
    gateway._pending_grid_posts = {}
    gateway._grid_post_event = threading.Event()
    gateway._grid_callback("occupancy")(object())
    assert gateway._pending_grid_posts == {}

  def test_mapped_state_uses_existing_latest_only_queue_without_thread(self):
    gateway = object.__new__(PhysicalRosGateway)
    calls = []
    gateway._post_state = lambda name, value: calls.append((name, value))

    gateway._post_mapped_state({"seq": 7})

    assert calls == [("mapped_detections", {"seq": 7})]

  def test_debug_cloud_gate_requires_configuration_and_subscriber(self):
    from types import SimpleNamespace

    class Publisher:
      def __init__(self, connections):
        self.connections = connections

      def get_num_connections(self):
        return self.connections

    assert not PhysicalRosGateway._debug_cloud_requested(Publisher(3), False)
    assert not PhysicalRosGateway._debug_cloud_requested(Publisher(0), True)
    assert PhysicalRosGateway._debug_cloud_requested(Publisher(1), True)
    # Replay fakes without ROS connection introspection preserve an explicit
    # opt-in instead of silently disabling their debug stream.
    assert PhysicalRosGateway._debug_cloud_requested(SimpleNamespace(), True)

  def test_compatibility_publisher_work_is_skipped_without_subscribers(self):
    from types import SimpleNamespace

    class Publisher:
      def __init__(self, connections):
        self.connections = connections

      def get_num_connections(self):
        return self.connections

    assert not PhysicalRosGateway._publisher_has_subscribers(Publisher(0))
    assert PhysicalRosGateway._publisher_has_subscribers(Publisher(1))
    assert PhysicalRosGateway._publisher_has_subscribers(SimpleNamespace())

  def test_capture_transform_cache_is_nonblocking_and_stamp_exact(self):
    """Historical TF lookup is attempted once without a 50 ms wait."""
    import threading
    from types import SimpleNamespace

    class Stamp:
      def to_sec(self):
        return 12.5

    class Buffer:
      def __init__(self):
        self.can_calls = []
        self.lookup_calls = []
        self.transform = SimpleNamespace(transform=SimpleNamespace())

      def can_transform(self, target, source, stamp, timeout):
        self.can_calls.append((target, source, stamp.to_sec(), timeout.to_sec()))
        return True

      def lookup_transform(self, target, source, stamp, timeout):
        self.lookup_calls.append((target, source, stamp.to_sec(), timeout.to_sec()))
        return self.transform

    gateway = object.__new__(PhysicalRosGateway)
    gateway.world_frame = "tf_frame_map"
    gateway.tf_buffer = Buffer()
    gateway._capture_transform_cache = {}
    gateway._capture_transform_cache_lock = threading.Lock()
    first = gateway._lookup_capture_transform("camera", Stamp())
    second = gateway._lookup_capture_transform("camera", Stamp())
    assert first is second
    assert len(gateway.tf_buffer.can_calls) == 1
    assert len(gateway.tf_buffer.lookup_calls) == 1
    assert gateway.tf_buffer.lookup_calls[0][-1] == 0.0

  def test_persistent_box_track_drops_large_segmentation_payloads(self):
    import numpy as np

    detection = {
      "semantic_class": "door",
      "confidence": .9,
      "bbox": [1, 2, 30, 40],
      "world_box3d_center": [1.0, 2.0, 0.5],
      "world_box3d_size": [1.0, .1, 2.0],
      "mask": {"rows": [1, 2, 3]},
      "camera_segment_points_f32": "packed",
      "_camera_segment_points_array": np.ones((2, 3), dtype=np.float32),
    }
    compact = PhysicalRosGateway._box_track_detection(detection)
    assert compact["semantic_class"] == "door"
    assert "mask" not in compact
    assert "camera_segment_points_f32" not in compact
    assert "_camera_segment_points_array" not in compact

  def test_full_cloud_camera_obb_is_independent_of_debug_point_sampling(self):
    import math
    from types import SimpleNamespace

    import numpy as np

    gateway = object.__new__(PhysicalRosGateway)
    gateway.world_frame = "tf_frame_map"
    transform_yaw = 0.6
    camera_yaw = 0.2
    transform = SimpleNamespace(transform=SimpleNamespace(
      rotation=SimpleNamespace(
        x=0.0, y=0.0,
        z=math.sin(transform_yaw * 0.5),
        w=math.cos(transform_yaw * 0.5),
      ),
      translation=SimpleNamespace(x=10.0, y=-2.0, z=0.5),
    ))
    base = {
      "semantic_class": "door",
      "source_frame": "d435i_depth_optical_frame",
      "camera_obb_center": [1.0, 0.0, 2.0],
      "camera_obb_size": [1.2, 0.15, 2.1],
      "camera_obb_orientation": [
        0.0, 0.0, math.sin(camera_yaw * 0.5), math.cos(camera_yaw * 0.5),
      ],
      "world_box3d_center": [999.0, 999.0, 999.0],
      "world_box3d_size": [9.0, 9.0, 9.0],
      # Full-point class admission extrema must survive the quantile OBB TF.
      "world_aabb_min_z": 0.34,
      "world_aabb_max_z": 2.61,
    }
    sparse = dict(
      base,
      camera_segment_points_f32=_encode_point_rows(
        np.asarray([[0.0, 0.0, 1.0], [20.0, 0.0, 1.0]], dtype=np.float32)
      ),
    )
    dense = dict(
      base,
      camera_segment_points_f32=_encode_point_rows(
        np.linspace(-50.0, 50.0, 1200, dtype=np.float32).reshape(-1, 3)
      ),
    )

    sparse_cache = {}
    first = gateway._map_detection(
      sparse, transform, segment_cache=sparse_cache,
    )
    second = gateway._map_detection(dense, transform, segment_cache={})

    for key in (
      "world_box3d_center", "world_box3d_size", "world_box3d_orientation",
      "aabb_center", "aabb_size",
    ):
      assert np.allclose(first[key], second[key])
    assert first["map_transform_status"] == "tf_camera_obb"
    assert first["world_box3d_size"] == [1.2, 0.15, 2.1]
    assert first["world_box3d_yaw"] == __import__("pytest").approx(0.8)
    assert first["world_aabb_min_z"] == 0.34
    assert first["world_aabb_max_z"] == 2.61
    expected_center = [
      10.0 + math.cos(transform_yaw),
      -2.0 + math.sin(transform_yaw),
      2.5,
    ]
    assert np.allclose(first["world_box3d_center"], expected_center)
    # The authoritative path leaves point decoding to the optional debug
    # worker; the mapping thread only composes the small OBB contract.
    assert sparse_cache[id(first)] == {}

  def test_full_cloud_obb_fallback_does_not_refit_world_debug_sample(self):
    import numpy as np

    gateway = object.__new__(PhysicalRosGateway)
    gateway.world_frame = "tf_frame_map"
    detection = {
      "semantic_class": "door",
      "source_frame": "d435i_depth_optical_frame",
      "camera_obb_center": [1.0, 0.0, 2.0],
      "camera_obb_size": [1.2, 0.15, 2.1],
      "camera_obb_orientation": [0.0, 0.0, 0.0, 1.0],
      "world_box3d_center": [4.0, 5.0, 1.2],
      "world_box3d_size": [1.2, 0.15, 2.1],
      "world_box3d_orientation": [0.0, 0.0, 0.0, 1.0],
      "world_segment_points_f32": _encode_point_rows(
        np.asarray([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]], dtype=np.float32)
      ),
    }

    mapped = gateway._map_detection(detection, None, segment_cache={})

    assert mapped["map_transform_status"] == "telemetry_full_obb_fallback"
    assert mapped["world_box3d_center"] == [4.0, 5.0, 1.2]
    assert mapped["world_box3d_size"] == [1.2, 0.15, 2.1]

  @staticmethod
  def _box_track_gateway(min_confirmations=1):
    gateway = object.__new__(PhysicalRosGateway)
    gateway._world_box_tracks = {}
    gateway._next_world_box_track_id = 1
    gateway._box_smoothing_alpha = 1.0
    gateway._box_match_distance_m = 0.6
    gateway._box_hold_s = 3.0
    gateway._box_min_confirmations = min_confirmations
    return gateway

  @staticmethod
  def _world_detection(label, x):
    return {
      "semantic_class": label,
      "confidence": .9,
      "bbox": [10, 10, 50, 80],
      "world_box3d_center": [x, 0., 1.],
      "world_box3d_size": [1., .1, 2.],
      "world_box3d_orientation": [0., 0., 0., 1.],
    }

  def test_world_box_track_never_associates_portal_with_container(self):
    gateway = self._box_track_gateway()
    door = self._world_detection("door", 0.)
    fridge = self._world_detection("fridge", .05)

    gateway._stable_world_boxes([door])
    gateway._stable_world_boxes([fridge])

    assert len(gateway._world_box_tracks) == 2
    assert door["_associated_visualization_track_id"] != fridge[
      "_associated_visualization_track_id"
    ]

  def test_unconfirmed_far_door_cannot_borrow_held_portal_geometry(self):
    gateway = self._box_track_gateway(min_confirmations=2)
    for _ in range(2):
      old = self._world_detection("door", 0.)
      gateway._stable_world_boxes([old])
    far = self._world_detection("door", 5.)

    stable = gateway._stable_world_boxes([far])
    gateway._apply_associated_portal_geometry([far], stable)

    assert far["world_box3d_center"] == [5., 0., 1.]
    assert "_associated_visualization_track_id" not in far

  def test_two_current_doors_keep_one_to_one_stable_geometry(self):
    gateway = self._box_track_gateway(min_confirmations=1)
    first = self._world_detection("door", 0.)
    second = self._world_detection("door", .4)

    stable = gateway._stable_world_boxes([first, second])
    first_id = first["_associated_visualization_track_id"]
    second_id = second["_associated_visualization_track_id"]
    gateway._apply_associated_portal_geometry([first, second], stable)

    assert first_id != second_id
    assert first["world_box3d_center"] != second["world_box3d_center"]

  def test_grid_mirror_uses_dedicated_http_lane(self):
    """A large map receipt must not enter the 10-Hz state-post queue."""
    from types import SimpleNamespace

    gateway = object.__new__(PhysicalRosGateway)
    gateway._post_state = lambda *_args: (_ for _ in ()).throw(
        AssertionError("map mirror must not use the shared state queue")
    )
    posted = []
    gateway._post_state_http = lambda name, value: posted.append((name, value))
    origin = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    msg = SimpleNamespace(
        header=SimpleNamespace(frame_id="tf_frame_map"),
        info=SimpleNamespace(width=2, height=2, resolution=.05, origin=origin),
        data=[-1, 0, 100, 0],
    )
    gateway._post_grid_state("occupancy", msg)
    assert posted == [("occupancy", {
        "width": 2,
        "height": 2,
        "resolution": .05,
        "frame_id": "tf_frame_map",
        "origin": {"x": 1.0, "y": 2.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        "data": [-1, 0, 100, 0],
    })]

  def test_state_http_post_preserves_name_and_value(self):
    """The isolated HTTP helper sends the same wire envelope as before."""
    import json
    from types import SimpleNamespace
    from unittest.mock import patch

    gateway = object.__new__(PhysicalRosGateway)
    gateway.args = SimpleNamespace(web_url="http://127.0.0.1:8765")
    requests = []

    class _Response:
      def __enter__(self):
        return self

      def __exit__(self, *_args):
        return False

      def read(self):
        return b"{}"

    def fake_urlopen(request, timeout):
      requests.append((request, timeout))
      return _Response()

    with patch("physical_ros_gateway.urllib.request.urlopen", fake_urlopen):
      gateway._post_state_http("telemetry", {"yaw": 1.25, "received_at": 7.0})

    assert requests and requests[0][1] == .3
    assert requests[0][0].full_url == "http://127.0.0.1:8765/api/ros-state"
    assert json.loads(requests[0][0].data.decode()) == {
      "name": "telemetry", "value": {"yaw": 1.25, "received_at": 7.0}
    }

  def test_web_state_deduplicates_identical_graph_json_replays(self):
    """Latched graph replays must not repeat parse + HTTP mirror work."""
    import threading
    from types import SimpleNamespace

    gateway = object.__new__(PhysicalRosGateway)
    gateway.web_state_enabled = True
    gateway._last_state_callback_payloads = {}
    gateway._state_callback_payload_lock = threading.Lock()
    gateway._state_post_event = threading.Event()
    posted = []
    gateway._post_state = lambda name, value: posted.append((name, value))
    callback = gateway._json_callback("graph")
    message = SimpleNamespace(data='{"graph_revision":7,"nodes":[],"edges":[]}')
    callback(message)
    callback(message)
    assert posted == []
    gateway._flush_state_mirror_payloads()
    assert posted == [("graph", {"graph_revision": 7, "nodes": [], "edges": []})]

  def test_web_state_coalesces_graph_heartbeat_and_flushes_latest_payload(self):
    """Presentation graph heartbeats are bounded without losing the newest state."""
    import threading
    from types import SimpleNamespace

    gateway = object.__new__(PhysicalRosGateway)
    gateway.web_state_enabled = True
    gateway._last_state_callback_payloads = {}
    gateway._last_state_mirror_post_mono = {}
    gateway._pending_state_mirror_payloads = {}
    gateway._state_callback_payload_lock = threading.Lock()
    gateway._state_post_event = threading.Event()
    posted = []
    gateway._post_state = lambda name, value: posted.append((name, value))
    callback = gateway._json_callback("graph")
    callback(SimpleNamespace(data='{"graph_revision":1,"nodes":[],"edges":[]}'))
    gateway._flush_state_mirror_payloads()
    callback(SimpleNamespace(data='{"graph_revision":2,"nodes":[],"edges":[]}'))
    assert posted == [("graph", {"graph_revision": 1, "nodes": [], "edges": []})]
    gateway._last_state_mirror_post_mono["graph"] -= 1.1
    gateway._flush_state_mirror_payloads()
    assert posted[-1] == ("graph", {"graph_revision": 2, "nodes": [], "edges": []})
    # A new receipt at the cooldown boundary supersedes the buffered one;
    # the worker must never send 4 and later rewind the browser to 3.
    callback(SimpleNamespace(data='{"graph_revision":3,"nodes":[],"edges":[]}'))
    gateway._last_state_mirror_post_mono["graph"] -= 1.1
    callback(SimpleNamespace(data='{"graph_revision":4,"nodes":[],"edges":[]}'))
    gateway._flush_state_mirror_payloads()
    gateway._last_state_mirror_post_mono["graph"] -= 1.1
    gateway._flush_state_mirror_payloads()
    assert [value["graph_revision"] for _, value in posted] == [1, 2, 4]

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
    # The dashboard uses latest-only still-image refreshes.  A browser-side
    # timer avoids an MJPEG connection monopolizing the gateway and lets the
    # server drop stale frames when the LAN is slow.
    assert "refreshStill('#overview','/snapshot.jpg')" in _HTML
    assert "refreshStill('#overview-camera','/camera-box-overlay.jpg')" in _HTML
    assert "<img id='overview-camera'" in _HTML
    assert "id='panel1'" not in _HTML
    assert "id='panel5'" not in _HTML
    assert "fetch('/snapshot.jpg?ts='" not in _HTML
    assert "createImageBitmap" not in _HTML

  def test_dashboard_keeps_six_panel_light_and_camera_overlay_at_ten_hz(self):
    """Only the lightweight source-resolution overlay polls at 10 Hz."""
    assert "refreshStill('#overview','/snapshot.jpg'),200" in _HTML
    assert "refreshStill('#overview-camera','/camera-box-overlay.jpg'),100" in _HTML
    assert "感知图 10 Hz · 六面板 5 Hz" in _HTML

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
      assert "['/original-panel3.jpg', 'view3']" in html
      assert "['/original-panel6.jpg', 'view6']" in html
      assert "X-Physical-Panel-Revision" in html
      assert "if (response.status === 204) return" in html
      assert "finally { image.close(); }" in html
      assert "3D" not in html
    assert "class='dark'" in DARK_SHOWCASE_HTML
    assert "class='light'" in LIGHT_SHOWCASE_HTML
    assert "debug-call-thumb" in DARK_SHOWCASE_HTML
    assert "真实 M1 输入" in DARK_SHOWCASE_HTML
    assert "候选 subgoal" in DARK_SHOWCASE_HTML
    assert "/api/m1-input-image?key=" in DARK_SHOWCASE_HTML
    assert "M3 为规则验证，不调用模型" not in DARK_SHOWCASE_HTML
    assert "aspect-ratio:16/9" in DARK_SHOWCASE_HTML
    assert "container.scrollTop=0" in DARK_SHOWCASE_HTML
    assert "request_sequence||0" in DARK_SHOWCASE_HTML
    assert "debug-call-thumb" not in LIGHT_SHOWCASE_HTML
    assert "/camera-overlay.jpg" in DARK_SHOWCASE_HTML
    assert "/camera-box-overlay.jpg" not in LIGHT_SHOWCASE_HTML

  def test_showcases_split_camera_overlay_to_ten_hz(self):
    """Native panels stay at 5 Hz while camera perception is 10 Hz."""
    for html in (DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML):
      assert "video=_showcaseSnapshotTick" in html
      assert "setInterval(video,200)" not in html
      assert "setInterval(refreshOriginalRendererPanels, 200)" in html
      assert "setInterval(refreshCameraOverlay,100)" in html
      assert "_showcaseCameraOverlayBusy" in html
    assert "video=_academicSnapshotTick" in ACADEMIC_SHOWCASE_HTML
    assert "setInterval(video,200)" not in ACADEMIC_SHOWCASE_HTML
    assert "setInterval(refreshCameraOverlay,100)" in ACADEMIC_SHOWCASE_HTML
    assert "_academicCameraOverlayBusy" in ACADEMIC_SHOWCASE_HTML

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
    assert "originalRendererPanelRevision" in html
    assert "X-Physical-Panel-Revision" in html
    assert "setInterval(video,200)" not in html
    assert "setInterval(academicVisualization,1000)" not in html
    # Unchanged OCC/graph/telemetry snapshots are answered with 204 by the
    # gateway; the academic page must not download and parse the full JSON at
    # every one-second heartbeat.
    assert "academicVisualizationRevision" in html
    assert "after_map" in html
    assert "after_graph" in html
    assert "after_telemetry" in html
    assert "after_detection" in html
    assert "after_navigation" in html
    assert "X-Physical-Detection-Revision" in html
    assert "X-Physical-Navigation-Revision" in html
    assert "if(r.status===204)return" in html

  def test_native_navigation_panels_do_not_poll_duplicate_raw_grids(self):
    for html in (DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML):
      assert "refreshNavigationOverlayData" not in html
      assert "drawNavigationOverlay" not in html
      assert "setInterval(refreshVisualization,1000)" not in html

  def test_academic_map_uses_live_position_before_legacy_map_position(self):
    # ``map_position`` is an un-stamped legacy mapper field.  The browser map
    # must follow the capture/current pose whenever both fields are present.
    assert "p=v.telemetry?.position||v.telemetry?.map_position" in ACADEMIC_SHOWCASE_HTML
    assert "p=v.telemetry?.map_position||v.telemetry?.position" not in ACADEMIC_SHOWCASE_HTML

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

  def test_high_rate_mapped_detections_do_not_invalidate_graph_revision(self):
    state = RuntimeState()
    state.update_topic("graph", {"nodes": [], "edges": []})
    graph_revision = state.visualization_revisions()["graph_revision"]
    state.update_topic("mapped_detections", {
      "seq": 1,
      "detections": [{"semantic_class": "door", "position": {"x": 1.0, "y": 2.0}}],
    })
    revisions = state.visualization_revisions()
    assert revisions["graph_revision"] == graph_revision
    assert revisions["detection_revision"] == 1
    state.update_topic("consistency", {"status": "ok"})
    assert state.visualization_revisions()["graph_revision"] == graph_revision
    assert state.snapshot()["consistency_revision"] == 1
    delta = state.visualization_delta({"telemetry", "mapped_detections"})
    assert delta["partial"] == ["mapped_detections", "telemetry"]
    assert "occupancy" not in delta
    assert "graph" not in delta
    assert delta["mapped_detections"][0]["semantic_class"] == "door"
    state.update_topic("graph", {"graph_revision": 2, "nodes": [{"id": "door_1"}], "edges": []})
    graph_delta = state.visualization_delta({"graph"})
    assert graph_delta["partial"] == ["graph"]
    assert graph_delta["graph"]["nodes"][0]["id"] == "door_1"
    assert "occupancy" not in graph_delta

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

  def test_heavy_render_key_ignores_camera_only_receipts(self):
    state = RuntimeState()
    renderer = SixPanelRenderer(state)
    state.update_frame(frame_seq=1, frame_stamp=1.0)
    first = renderer.heavy_render_key(state)
    state.update_frame(frame_seq=2, frame_stamp=1.1)
    assert renderer.heavy_render_key(state) == first
    state.update_topic("telemetry", {"position": [0.0, 0.0, 0.0], "yaw": 0.5})
    assert renderer.heavy_render_key(state) != first
    state.update_topic("occupancy", {"width": 1, "height": 1, "data": [0]})
    assert renderer.heavy_render_key(state) != first

  def test_capture_telemetry_updates_live_heading_without_erasing_full_snapshot(self):
    """A sensor-frame pose patch must move the panel arrow at camera rate.

    The direct sensor bridge includes only capture-critical pose fields in
    each RGB-D packet.  The lower-rate telemetry mirror carries battery and
    diagnostics.  RuntimeState must merge the two streams rather than relying
    on the mirror for yaw or replacing the complete snapshot with a partial
    packet.
    """
    state = RuntimeState()
    state.update_topic(
      "telemetry",
      {
        "position": [0.0, 0.0, 0.0],
        "yaw": 0.10,
        "battery": {"soc": 87},
      },
    )
    first_revision = state.snapshot()["telemetry_revision"]
    state.update_frame(
      frame_seq=1,
      frame_stamp=1.0,
      telemetry={"position": [1.0, 0.5, 0.0], "yaw": 1.20},
    )
    snapshot = state.snapshot()
    assert snapshot["telemetry"]["yaw"] == 1.20
    assert snapshot["telemetry"]["position"] == [1.0, 0.5, 0.0]
    assert snapshot["telemetry"]["battery"]["soc"] == 87
    assert snapshot["telemetry_revision"] == first_revision + 1
    physical_step = SixPanelRenderer._physical_step(snapshot)
    assert physical_step["pose"] == [1.0, 0.5, 1.20]
    assert physical_step["pose_frame_id"] == "tf_frame_odom"
    assert physical_step["trajectory_frame_id"] == "tf_frame_odom"
    # A stale map correction must not override the live capture yaw.
    state.update_frame(
      frame_seq=2,
      frame_stamp=1.1,
      telemetry={
        "position": [2.0, 3.0, 0.0],
        "map_position": [99.0, 99.0, 0.0],
        "yaw": -0.70,
        "map_yaw": 2.80,
      },
    )
    assert SixPanelRenderer._physical_step(state.snapshot())["pose"] == [2.0, 3.0, -0.70]

  def test_stale_telemetry_mirror_cannot_rewind_pose_or_drop_battery_fields(self):
    state = RuntimeState()
    state.update_frame(
      frame_seq=10,
      frame_stamp=10.0,
      telemetry={
        "received_at": 10.0,
        "position": [3.0, 4.0, 0.0],
        "yaw": 1.5,
        "battery": {"soc": 80, "voltage": 24.0, "current": 2.0},
      },
    )
    # The mirror packet arrived later but contains an older capture pose and
    # only a partial battery dictionary.  Pose must remain capture-current;
    # nested diagnostics are merged instead of replacing voltage/current.
    state.update_topic(
      "telemetry",
      {
        "received_at": 9.0,
        "position": [-1.0, -2.0, 0.0],
        "yaw": -2.0,
        "battery": {"soc": 79},
      },
    )
    telemetry = state.snapshot()["telemetry"]
    assert telemetry["position"] == [3.0, 4.0, 0.0]
    assert telemetry["yaw"] == 1.5
    assert telemetry["battery"] == {"soc": 79, "voltage": 24.0, "current": 2.0}

  def test_unstamped_telemetry_pose_is_one_shot_and_cannot_rewind(self):
    state = RuntimeState()
    # Legacy callers may omit received_at.  Keep the first pose for
    # compatibility, but reject another un-stamped pose once a newer,
    # capture-stamped pose has been accepted.
    state.update_topic("telemetry", {"position": [1.0, 2.0, 0.0], "yaw": 0.25})
    state.update_frame(
      frame_seq=1,
      frame_stamp=10.0,
      telemetry={"received_at": 10.0, "position": [3.0, 4.0, 0.0], "yaw": 1.25},
    )
    state.update_topic("telemetry", {"position": [-9.0, -8.0, 0.0], "yaw": -2.5})
    telemetry = state.snapshot()["telemetry"]
    assert telemetry["position"] == [3.0, 4.0, 0.0]
    assert telemetry["yaw"] == 1.25

  def test_map_and_room_panels_apply_explicit_pose_frame_once(self):
    """Panel 2/3 must use the live pose's declared odom/map frame.

    A non-identity map<-odom correction makes the old hard-coded odom path
    observable: a pose already labelled as map was transformed a second time,
    while an odom pose could miss the correction. Keep this regression test at
    the renderer boundary so it does not require ROS or a live map.
    """
    import math
    from unittest.mock import patch

    import numpy as np
    from offline_semantic_renderer import (
      OfflineSixPanelRenderer,
      RawGrid,
      TransformResolver,
      TransformSample,
    )

    grid = RawGrid(
      np.zeros((40, 40), dtype=np.int32), 40, 40, .1,
      "tf_frame_map", -2., -2., 0.,
    )
    renderer = OfflineSixPanelRenderer(
      transforms=TransformResolver(
        [TransformSample(step_index=3, x=0.0, y=0.0, yaw=math.pi / 2.)],
        map_frame="tf_frame_map", odom_frame="tf_frame_odom",
      )
    )
    step = {
      "step_index": 3,
      "pose": [0.5, 0.0, 0.0],
      "pose_frame_id": "tf_frame_odom",
      "trajectory_frame_id": "tf_frame_odom",
    }
    with patch("offline_semantic_renderer._draw_robot_arrow", return_value=None) as draw:
      renderer.render_map_panel(
        grid, (240, 180), step, 3, title="OCC", kind="occupancy",
        world_bounds=(-2., -2., 4., 4.), draw_frontiers=False,
      )
      renderer.render_room_panel(
        grid, None, (240, 180), step, 3, (-2., -2., 4., 4.),
      )
    assert draw.call_count == 2
    assert all(math.isclose(call.args[2], math.pi / 2., abs_tol=1e-6) for call in draw.call_args_list)

    # A map-frame pose must not receive the odom correction a second time.
    map_step = dict(step, pose_frame_id="tf_frame_map")
    with patch("offline_semantic_renderer._draw_robot_arrow", return_value=None) as draw_map:
      renderer.render_map_panel(
        grid, (240, 180), map_step, 3, title="OCC", kind="occupancy",
        world_bounds=(-2., -2., 4., 4.), draw_frontiers=False,
      )
    assert draw_map.call_count == 1
    assert math.isclose(draw_map.call_args.args[2], 0.0, abs_tol=1e-6)

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
    assert "refreshStill('#overview-camera','/camera-box-overlay.jpg')" in _HTML

  def test_semantic_xy_cache_refreshes_when_only_heading_changes(self):
    """The cached semantic panel must not freeze the robot arrow."""
    from unittest.mock import patch

    import numpy as np

    state = RuntimeState()
    state.rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    state.frame_seq = 1
    state.frame_stamp = 1.0
    state.telemetry = {"position": [0.0, 0.0, 0.0], "yaw": 0.0}
    state.occupancy = {
      "width": 20, "height": 20, "resolution": 0.1,
      "frame_id": "tf_frame_map",
      "origin": {"x": -1.0, "y": -1.0, "qw": 1.0},
      "data": [-1] * 400,
    }
    state.graph = {"graph_revision": 1, "nodes": [], "edges": []}
    renderer = SixPanelRenderer(state)
    with patch.object(renderer._canonical, "render_semantic_xy", wraps=renderer._canonical.render_semantic_xy) as render_xy:
      renderer.render()
      state.update_frame(
        frame_seq=2,
        frame_stamp=1.1,
        telemetry={"position": [0.0, 0.0, 0.0], "yaw": 1.0},
      )
      renderer.render()
    assert render_xy.call_count == 2

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
    config = (ROOT / "config" / "physical_nav.yaml").read_text()
    assert 'name="interaction_attribute_inference"' in launch
    assert "subscribe_pointcloud: false" in config
    assert 'name="semantic_map/subscribe_pointcloud"' not in launch
    assert 'file="$(find nav_pkg)/launch/nav.launch"' in launch
    assert '<remap from="/odom" to="/physical_nav/odom"/>' in launch
    assert '<param name="scan_filter_tolerance_sec" value="0.0"/>' in launch
    assert '<param name="max_odom_cloud_time_diff" value="0.15"/>' in launch
    assert '<param name="maxUrange" value="7.9"/>' in launch
    assert '<param name="pointcloud_scan_range_max" value="8.0"/>' in launch
    assert '<param name="enable_obstacle_inflation" value="false" type="bool"/>' in launch
    assert '<param name="max_depth_m" value="8.0"/>' in launch
    assert '<param name="no_return_depth_m" value="8.05"/>' in launch
    assert '<param name="pointcloud_scan_min_support_neighbors" value="0"/>' in launch
    assert '<param name="enable_local_overwrite" value="true"/>' in launch
    assert '<param name="local_overwrite_radius" value="7.9"/>' in launch
    assert '<param name="local_overwrite_ttl_sec" value="2.0"/>' in launch
    assert '<arg name="override_config_file" value="$(arg semantic_override_config)"/>' in launch
    assert '<arg name="override_config_file" value="$(arg move_base_override_config)"/>' in launch
    assert '<remap from="/cmd_vel" to="/physical_nav/semantic_cmd_vel"/>' in launch
    assert '<param name="move_base_topic" value="/physical_nav/move_base_cmd_vel"/>' in launch
    assert '<param name="semantic_topic" value="/physical_nav/semantic_cmd_vel"/>' in launch
    assert 'type="velocity_command_mux.py"' in launch
    assert '<arg name="cmd_vel_topic" value="/physical_nav/move_base_cmd_vel"/>' in launch
    assert '<arg name="cmd_vel_stamped_topic" value="/physical_nav/shadow_cmd_vel_stamped"/>' in launch
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
    assert 'GATEWAY_FINGERPRINT_FILE=' in start
    assert 'restarting stale persistent gateway' in start
    assert 'physical_protocol.py' in start
    assert 'offline_semantic_renderer.py' in start
    assert '"${ROOT_DIR}/../offline_semantic_renderer.py"' in start
    assert 'YOLO_FINGERPRINT_FILE=' in start
    assert 'restarting stale persistent yoloe' in start
    assert 'physical_yoloe_bridge.py' in start
    assert 'nohup setsid "${GATEWAY_PYTHON}"' in start
    assert '*"start_physical_nav.sh"*' in service
    assert service.index('cleanup_pipeline_residuals\n  rm -f "${PID_FILE}"') > service.index('start_service()')
    assert 'interaction_semantic_types: ["door", "fridge"]' in override
    assert 'drawer_cabinet' not in override
    assert 'min_obstacle_clearance_m: 0.80' in override
    assert 'portal_traversal_clearance_margin_m: 0.55' in override
    assert 'rear_goal_prerotate_enabled: true' in override
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
    assert '"--publish-fps|$5"' in all_launcher
    assert '--publish-fps "$5"' in all_launcher
    assert 'capture-critical argument' in all_launcher
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
    assert local_override["local_costmap"]["inflation_layer"]["inflation_radius"] == 0.40
    assert local_override["global_costmap"]["inflation_layer"]["inflation_radius"] == 0.40
    assert local_override["local_costmap"]["obstacle_layer"]["local_obstacles"]["observation_persistence"] == 0.0
    assert local_override["local_costmap"]["obstacle_layer"]["footprint_clearing_enabled"] is True
    assert local_override["local_costmap"]["obstacle_layer"]["obstacle_reset_interval"] == 2.0
    assert local_override["DWAPlannerROS"]["max_vel_x"] == 0.56
    assert local_override["DWAPlannerROS"]["min_vel_x"] == 0.05
    assert local_override["DWAPlannerROS"]["max_vel_trans"] == 0.60
    assert local_override["DWAPlannerROS"]["min_vel_trans"] == 0.05
    assert local_override["DWAPlannerROS"]["max_vel_theta"] == 1.30
    assert local_override["DWAPlannerROS"]["min_vel_theta"] == 0.05
    assert local_override["DWAPlannerROS"]["acc_lim_trans"] == 1.0
    assert local_override["DWAPlannerROS"]["rear_path_prerotate_enabled"] is True
    assert local_override["DWAPlannerROS"]["conditional_reverse_enabled"] is False
    assert local_override["DWAPlannerROS"]["reject_degenerate_cmd_enabled"] is True
    assert semantic_override["candidate"]["interaction_ready_yaw_tolerance_rad"] == 0.15
    assert semantic_override["candidate"]["portal_approach_standoff_offsets_m"] == [0.0, -0.10, -0.20]
    assert semantic_override["candidate"]["portal_approach_tangent_offsets_m"] == [0.0]
    assert semantic_override["candidate"]["portal_approach_yaw_offsets_rad"] == [0.0]
    assert semantic_override["candidate"]["portal_opposite_side_fallback_enabled"] is False
    assert semantic_override["candidate"]["portal_require_reference_yaw"] is True
    assert semantic_override["candidate"]["portal_side_hysteresis_m"] == 0.50
    assert semantic_override["candidate"]["portal_goal_require_known_free"] is False
    assert semantic_override["executor"]["explore_terminal_yaw_enabled"] is True
    assert semantic_override["executor"]["explore_terminal_xy_tolerance_m"] == 0.30
    assert semantic_override["executor"]["explore_terminal_yaw_tolerance_rad"] == 0.15
    assert physical_config["semantic_map"]["object_portal_cross_view_match_enabled"] is True
    assert physical_config["velocity_safety"]["max_linear_mps"] == 0.60
    assert physical_config["velocity_safety"]["min_linear_mps"] == 0.30
    assert physical_config["velocity_safety"]["linear_deadband_mps"] == 0.05
    assert physical_config["velocity_safety"]["max_angular_rps"] == 1.30
    assert physical_config["velocity_safety"]["min_angular_rps"] == 0.40
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
