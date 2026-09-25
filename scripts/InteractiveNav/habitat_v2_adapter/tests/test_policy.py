from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from habitat_v2_adapter.policy import (
    FREE,
    OCCUPIED,
    _ActiveGoal,
    _PublicTargetTrack,
    HabitatInteractiveNavM2Policy,
    PolicyConfig,
)
from habitat_v2_adapter.adapter_config import load_adapter_profile
from habitat_v2_adapter.module1_bridge import (
    Module1Detection,
    Module1DetectionResult,
    Module1DetectorSidecarClient,
    goal_label_aliases,
)
from habitat_v2_adapter.mllm import build_behavior_candidates, candidate_history_key_for
from habitat_v2_adapter.evaluate import (
    _fresh_mllm_requests_path,
    _mllm_request_stats,
    _normalize_scene_dataset_config,
    _select_episodes,
)


class _NoNetworkPolicy(HabitatInteractiveNavM2Policy):
    def __init__(self):
        self.config = PolicyConfig()
        self._shape = (240, 240)
        self._origin = np.array([120, 120], dtype=np.float32)
        self._model = None
        self._vision = None
        self._grounding_dino = None
        self._module1_detector = None
        self._yolov7 = None
        self._mobile_sam = None
        self._pointnav = None
        self.reset({"episode_id": "e", "scene_id": "s", "object_category": "chair"})

    def _select_goal(self, records, pose_xy, heading):
        record = records[0]
        return self._goal_from_record(record)

    def _goal_from_record(self, record):
        candidate_id = record["candidate_id"]
        return _ActiveGoal(
            candidate_id,
            np.asarray(record["goal_xyyaw"][:2], dtype=np.float32),
            self._step,
            self._candidate_routes[candidate_id],
        )

    def _visible_goal_detection(self, observations):
        return None


class _VisualGoalPolicy(_NoNetworkPolicy):
    def __init__(self, detections):
        super().__init__()
        self._detections = iter(detections)

    def _visible_goal_detection(self, observations):
        return next(self._detections)


class _GroundingDinoResult:
    def __init__(self, detections=(), error=""):
        self.detections = tuple(detections)
        self.error = error


class _GroundingDinoDetection:
    def __init__(self, confidence, box):
        self.confidence = confidence
        self.bbox_xyxy_normalized = box


class _GroundingDinoStub:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detect(self, image, caption, context):
        self.calls.append((image.shape, caption, context))
        return self.result


class _YoloV7Result:
    def __init__(self, detections=(), error=""):
        self.detections = tuple(detections)
        self.error = error


class _YoloV7Detection:
    def __init__(self, confidence, box):
        self.confidence = confidence
        self.bbox_xyxy_normalized = box


class _YoloV7Stub:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detect(self, image, target_label, context):
        self.calls.append((image.shape, target_label, context))
        return self.result


class _Module1DetectorStub:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detect(self, image, depth_m, hfov_degrees, *, stamp_index, context):
        self.calls.append((image.shape, None if depth_m is None else depth_m.shape, hfov_degrees, stamp_index, context))
        return self.result


class _MobileSamResult:
    def __init__(self, mask=None, error=""):
        self.mask = mask
        self.error = error


class _MobileSamStub:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def segment_bbox(self, image, bbox_xyxy_pixels, context):
        self.calls.append((image.shape, bbox_xyxy_pixels, context))
        return self.result


class _PointNavResult:
    def __init__(self, action_index, error=""):
        self.action_index = action_index
        self.error = error


class _PointNavStub:
    def __init__(self, actions):
        self._actions = iter(actions)
        self.calls = []

    def act(self, depth, rho, theta, *, reset, context):
        self.calls.append((depth.shape, rho, theta, reset, context))
        return _PointNavResult(next(self._actions))


class _ModelSelectionStub:
    last_result_source = "model_stub"

    def __init__(self, selected_index=-1):
        self.selected_index = selected_index
        self.calls = []

    def select(self, candidates, **kwargs):
        self.calls.append((candidates, kwargs))
        return candidates[self.selected_index]


class _VisionResponse:
    def __init__(self, payload, error=""):
        self.payload = payload
        self.error = error


class _VisionStub:
    def __init__(self, payload):
        self.payload = payload

    def request_json(self, **_kwargs):
        return _VisionResponse(self.payload)


class _M3Result:
    def __init__(self, decision, confidence=0.9, reason="test", error=""):
        self.decision = decision
        self.confidence = confidence
        self.reason = reason
        self.error = error


class _M3VerifierStub:
    def __init__(self, *decisions):
        self._decisions = iter(decisions)
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append(kwargs)
        return _M3Result(next(self._decisions))


def test_policy_only_exposes_navigation_actions() -> None:
    policy = _NoNetworkPolicy()
    action = policy.act(
        {
            "gps": np.array([0.0, 0.0], dtype=np.float32),
            "compass": np.array([0.0], dtype=np.float32),
            "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
        }
    )
    policy.assert_navigation_only(action)
    assert action["action"] in {"velocity_control", "velocity_stop"}


def test_stop_is_a_navigation_action() -> None:
    action = HabitatInteractiveNavM2Policy.stop_action()
    HabitatInteractiveNavM2Policy.assert_navigation_only(action)
    assert action["action"] == "velocity_stop"


def test_navigation_contract_rejects_interaction() -> None:
    try:
        HabitatInteractiveNavM2Policy.assert_navigation_only({"action": "interact"})
    except AssertionError:
        return
    raise AssertionError("interaction action unexpectedly passed contract")


def test_decision_stats_distinguishes_model_and_fallback() -> None:
    policy = _NoNetworkPolicy()
    policy._decision_history = [
        {"result_source": "model"},
        {"result_source": "curated_fallback_invalid_response"},
    ]
    assert policy.decision_stats() == {
        "decisions": 2,
        "model_selected": 1,
        "fallback_selected": 1,
    }


def test_visual_goal_needs_two_close_centered_confirmations_before_stop() -> None:
    detection = {
        "center_x": 0.5,
        "center_y": 0.5,
        "depth_m": 0.3,
        "confidence": 0.9,
    }
    policy = _VisualGoalPolicy([detection, detection])
    observations = {
        "gps": np.array([0.0, 0.0], dtype=np.float32),
        "compass": np.array([0.0], dtype=np.float32),
        "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
    }
    first = policy.act(observations)
    second = policy.act(observations)
    assert first["action"] == "velocity_control"
    assert first["action_args"]["linear_velocity"] == -1.0
    assert first["action_args"]["angular_velocity"] == 0.0
    assert second["action"] == "velocity_stop"


def test_visual_confirmation_survives_non_query_control_steps() -> None:
    policy = _VisualGoalPolicy(
        [
            {"center_x": 0.5, "center_y": 0.5, "depth_m": 0.3, "confidence": 0.9},
            None,
        ]
    )
    observations = {
        "gps": np.array([0.0, 0.0], dtype=np.float32),
        "compass": np.array([0.0], dtype=np.float32),
        "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
    }
    policy.act(observations)
    # Emulate an intervening non-query action: ``_last_vision_step`` remains
    # older than the current step, so the prior confirmation must persist.
    policy._step += 1
    policy._last_vision_step = policy._step - 1
    held = policy.act(observations)
    assert held["action_args"]["linear_velocity"] == -1.0
    assert policy._visible_goal_confirmations == 1


def test_first_visual_detection_holds_view_until_next_confirmation_query() -> None:
    # The real vision method returns ``None`` for the first valid RGB-D box
    # after saving it as a pending temporal confirmation.  Model that observable
    # state directly rather than returning an already-confirmed detection.
    policy = _VisualGoalPolicy([None, None])
    policy._awaiting_visual_confirmation = True
    observations = {
        "gps": np.array([0.0, 0.0], dtype=np.float32),
        "compass": np.array([0.0], dtype=np.float32),
        "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
    }
    first = policy.act(observations)
    assert first["action_args"]["linear_velocity"] == -1.0
    assert policy._awaiting_visual_confirmation
    # The next controller tick is not an image query.  It must preserve the
    # public view instead of letting frontier exploration move the target away.
    second = policy.act(observations)
    assert second["action_args"]["linear_velocity"] == -1.0


def test_grounding_dino_verifier_rejects_non_overlapping_mllm_box() -> None:
    policy = _NoNetworkPolicy()
    policy._grounding_dino = _GroundingDinoStub(
        _GroundingDinoResult([_GroundingDinoDetection(0.9, (0.7, 0.7, 0.9, 0.9))])
    )
    result = policy._verify_with_grounding_dino(
        np.zeros((32, 24, 3), dtype=np.uint8),
        "couch",
        {"center_x": 0.1, "center_y": 0.2, "x1": 0.0, "y1": 0.1, "x2": 0.2, "y2": 0.3, "depth_m": 2.0},
    )
    assert result is None
    assert policy._grounding_dino_confirmed == 0


def test_grounding_dino_verifier_uses_an_overlapping_detector_box() -> None:
    policy = _NoNetworkPolicy()
    policy._grounding_dino = _GroundingDinoStub(
        _GroundingDinoResult([_GroundingDinoDetection(0.9, (0.0, 0.1, 0.22, 0.32))])
    )
    result = policy._verify_with_grounding_dino(
        np.zeros((32, 24, 3), dtype=np.uint8),
        "couch",
        {"center_x": 0.1, "center_y": 0.2, "x1": 0.0, "y1": 0.1, "x2": 0.2, "y2": 0.3, "depth_m": 2.0},
    )
    assert result is not None
    assert np.isclose(result["confidence"], 0.9)
    assert np.isclose(result["x2"], 0.22)
    assert policy._grounding_dino_confirmed == 1
    assert policy._grounding_dino.calls[0][1] == "sofa ."


def test_grounding_dino_box_resamples_public_depth_before_tracking() -> None:
    policy = _NoNetworkPolicy()
    policy._step = policy.config.vision_interval_steps
    policy._vision = _VisionStub(
        {
            "goal_visible": True,
            "confidence": 0.9,
            "bbox_xyxy_normalized": [0.1, 0.1, 0.7, 0.9],
        }
    )
    policy._grounding_dino = _GroundingDinoStub(
        _GroundingDinoResult([_GroundingDinoDetection(0.9, (0.5, 0.1, 0.9, 0.9))])
    )
    depth = np.ones((20, 20, 1), dtype=np.float32)
    depth[:, 10:, 0] = 3.0
    HabitatInteractiveNavM2Policy._visible_goal_detection(
        policy,
        {
            "rgb": np.zeros((20, 20, 3), dtype=np.uint8),
            "depth": depth,
        },
    )
    assert policy._latest_public_visual_measurement is not None
    assert np.isclose(policy._latest_public_visual_measurement["depth_m"], 3.0)


def test_yolov7_detector_measurement_is_track_only_and_uses_coco_aliases() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, detector_first_target_tracking=True)
    policy._episode["object_category"] = "plant"
    policy._yolov7 = _YoloV7Stub(
        _YoloV7Result([_YoloV7Detection(0.91, (0.2, 0.2, 0.8, 0.8))])
    )
    result = policy._yolov7_target_detection(
        np.zeros((20, 20, 3), dtype=np.uint8),
        "plant",
        {"depth": np.full((20, 20, 1), 0.5, dtype=np.float32)},
    )
    assert result is not None
    assert result["track_only"] == 1.0
    assert result["stop_eligible"] == 0.0
    assert not policy._visual_goal_is_safe_to_stop({**result, "depth_m": 0.3, "center_x": 0.5})
    assert policy._yolov7_queries == 1
    assert policy._yolov7_positive == 1
    assert policy._yolov7.calls[0][1] == "potted plant"


def test_detector_first_uses_yolo_measurement_without_mllm_visual_takeover() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, detector_first_target_tracking=True)
    policy._vision = _VisionStub(
        {
            "goal_visible": True,
            "confidence": 0.95,
            "bbox_xyxy_normalized": [0.0, 0.0, 0.2, 0.2],
        }
    )
    policy._yolov7 = _YoloV7Stub(
        _YoloV7Result([_YoloV7Detection(0.91, (0.4, 0.4, 0.8, 0.8))])
    )
    result = HabitatInteractiveNavM2Policy._visible_goal_detection(
        policy,
        {
            "rgb": np.zeros((20, 20, 3), dtype=np.uint8),
            "depth": np.full((20, 20, 1), 0.5, dtype=np.float32),
        },
    )
    assert result is None  # The first tracker measurement never directly controls.
    assert policy._latest_public_visual_measurement is not None
    assert np.isclose(policy._latest_public_visual_measurement["center_x"], 0.6)
    assert policy._latest_public_visual_measurement["track_only"] == 1.0
    assert not policy._awaiting_visual_confirmation


def test_mobile_sam_refines_detector_depth_and_bearing_from_public_mask() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, detector_first_target_tracking=True)
    policy._yolov7 = _YoloV7Stub(
        _YoloV7Result([_YoloV7Detection(0.91, (0.2, 0.2, 0.8, 0.8))])
    )
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 12:16] = True
    policy._mobile_sam = _MobileSamStub(_MobileSamResult(mask=mask))
    depth = np.full((20, 20, 1), 0.8, dtype=np.float32)
    depth[8:12, 12:16, 0] = 0.2
    result = policy._yolov7_target_detection(
        np.zeros((20, 20, 3), dtype=np.uint8),
        "chair",
        {"depth": depth},
    )
    assert result is not None
    assert np.isclose(result["depth_m"], 1.4)
    assert result["center_x"] > 0.6
    assert policy._mobile_sam_queries == 1
    assert policy._mobile_sam_confirmed == 1


def test_negative_visual_requery_finishes_a_pending_short_waypoint() -> None:
    policy = _VisualGoalPolicy([None])
    policy._active_goal = _ActiveGoal(
        "visible_goal", np.array([0.7, 0.0], dtype=np.float32), selected_step=0
    )
    # Make the first action a scheduled negative visual query.  The controller
    # must still travel toward the already confirmed short waypoint, not drop it
    # immediately and rotate back to frontier exploration.
    policy._step = policy.config.vision_interval_steps
    policy._last_vision_step = 0
    action = policy.act(
        {
            "gps": np.array([0.0, 0.0], dtype=np.float32),
            "compass": np.array([0.0], dtype=np.float32),
            "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
        }
    )
    assert action["action_args"]["linear_velocity"] == policy.config.linear_velocity


def test_blocked_visible_goal_releases_after_public_depth_confirmations() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        visual_goal_blocked_confirmations=2,
        visual_goal_cooldown_steps=20,
    )
    policy._step = 5
    policy._last_forward_clearance_m = 0.1
    policy._active_goal = _ActiveGoal("visible_goal", np.array([1.0, 0.0], dtype=np.float32), selected_step=0)
    first = policy._drive_to_visible_goal(
        np.zeros(2, dtype=np.float32), 0.0, policy._active_goal
    )
    assert first["action_args"]["linear_velocity"] == -1.0
    assert policy._active_goal is not None
    second = policy._drive_to_visible_goal(
        np.zeros(2, dtype=np.float32), 0.0, policy._active_goal
    )
    assert second["action_args"]["linear_velocity"] == -1.0
    assert policy._active_goal is None
    assert policy.vision_stats()["visual_goal_clearance_blocks"] == 2
    assert policy.vision_stats()["visual_goal_releases"] == 1
    assert policy._visual_goal_cooldown_until_step == 25


def test_visible_goal_budget_releases_an_unconverged_takeover() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        visual_goal_max_contiguous_steps=2,
        visual_goal_cooldown_steps=20,
    )
    policy._step = 5
    policy._last_forward_clearance_m = policy.config.max_depth_m
    policy._active_goal = _ActiveGoal("visible_goal", np.array([1.0, 0.0], dtype=np.float32), selected_step=0)
    first = policy._drive_to_visible_goal(
        np.zeros(2, dtype=np.float32), 0.0, policy._active_goal
    )
    assert first["action_args"]["linear_velocity"] == policy.config.linear_velocity
    second = policy._drive_to_visible_goal(
        np.zeros(2, dtype=np.float32), 0.0, policy._active_goal
    )
    assert second["action_args"]["linear_velocity"] == -1.0
    assert policy._active_goal is None
    assert policy.vision_stats()["visual_goal_budget_releases"] == 1


def test_visual_goal_cooldown_does_not_allow_immediate_retaking() -> None:
    detection = {"center_x": 0.5, "center_y": 0.5, "depth_m": 1.5, "confidence": 0.9}
    policy = _VisualGoalPolicy([detection])
    policy._visual_goal_cooldown_until_step = 20
    action = policy.act(
        {
            "gps": np.array([0.0, 0.0], dtype=np.float32),
            "compass": np.array([0.0], dtype=np.float32),
            "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
        }
    )
    assert action["action"] == "velocity_control"
    assert policy._active_goal is None or policy._active_goal.candidate_id != "visible_goal"
    assert policy.vision_stats()["visual_controller_steps"] == 0


def test_visual_goal_waypoint_uses_public_compass_frame() -> None:
    policy = _NoNetworkPolicy()
    waypoint = policy._visible_goal_waypoint(
        np.array([0.0, 0.0], dtype=np.float32),
        np.pi / 2.0,
        {"center_x": 0.5, "center_y": 0.5, "depth_m": 1.0, "confidence": 0.9},
    )
    # At +pi/2, forward maps to public GPS -y.
    assert np.allclose(waypoint, [0.0, -0.4], atol=1e-5)


def test_visual_goal_waypoint_keeps_edge_target_bearing_at_public_depth_standoff() -> None:
    policy = _NoNetworkPolicy()
    waypoint = policy._visible_goal_waypoint(
        np.array([0.0, 0.0], dtype=np.float32),
        0.0,
        {"center_x": 0.1, "center_y": 0.7, "depth_m": 3.5, "confidence": 0.9},
    )
    # The capped short waypoint preserves the same leftward camera ray.
    expected_forward = min(0.70, 3.5 - policy.config.vision_waypoint_standoff_m)
    expected_lateral = np.tan((0.1 - 0.5) * np.deg2rad(policy.config.hfov_degrees)) * expected_forward
    assert np.allclose(waypoint, [expected_forward, expected_lateral], atol=1e-5)


def test_public_target_track_requires_a_separated_consistent_detection_before_promotion() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, persistent_target_tracking=True)
    detection = {"center_x": 0.5, "center_y": 0.5, "depth_m": 2.0, "confidence": 0.9}
    assert not policy._update_public_target_track(np.array([0.0, 0.0], dtype=np.float32), 0.0, detection)
    assert policy._target_track is not None
    # Repeating the same camera frame can only seed the estimate, not create a
    # persistent navigation target.
    assert not policy._update_public_target_track(np.array([0.02, 0.0], dtype=np.float32), 0.0, detection)
    assert not policy._target_track.promoted
    # After a public baseline, adjust the camera-z depth to keep the projected
    # surface position consistent and allow promotion.
    assert policy._update_public_target_track(
        np.array([0.3, 0.0], dtype=np.float32), 0.0, {**detection, "depth_m": 1.7}
    )
    assert policy._target_track.promoted
    assert policy._target_track_promoted == 1


def test_public_target_track_rejects_a_geometrically_incompatible_box() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, persistent_target_tracking=True)
    detection = {"center_x": 0.5, "center_y": 0.5, "depth_m": 2.0, "confidence": 0.9}
    policy._update_public_target_track(np.array([0.0, 0.0], dtype=np.float32), 0.0, detection)
    policy._update_public_target_track(np.array([0.3, 0.0], dtype=np.float32), 0.0, {**detection, "depth_m": 1.7})
    assert policy._target_track is not None and policy._target_track.promoted
    original = policy._target_track.surface_xy.copy()
    # A later high-confidence box that projects far away is rejected rather
    # than teleporting the persistent public target.
    assert not policy._update_public_target_track(
        np.array([0.3, 0.0], dtype=np.float32), 0.0, {**detection, "depth_m": 4.5}
    )
    assert np.allclose(policy._target_track.surface_xy, original)
    assert policy._target_track_rejected == 1


def test_unpromoted_target_track_reseeds_after_an_incompatible_measurement() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, persistent_target_tracking=True)
    detection = {"center_x": 0.5, "center_y": 0.5, "depth_m": 2.0, "confidence": 0.9}
    policy._update_public_target_track(np.array([0.0, 0.0], dtype=np.float32), 0.0, detection)
    policy._update_public_target_track(
        np.array([0.3, 0.0], dtype=np.float32),
        0.0,
        {**detection, "depth_m": 4.5},
    )
    assert policy._target_track is not None
    assert not policy._target_track.promoted
    assert np.allclose(policy._target_track.seed_pose_xy, [0.3, 0.0])
    assert policy._target_track_seeded == 2
    assert policy._target_track_rejected == 1


def test_public_target_record_uses_only_an_observed_reachable_standoff_cell() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        persistent_target_tracking=True,
        target_track_standoff_m=1.0,
        target_track_standoff_tolerance_m=0.2,
    )
    policy._grid.fill(0)
    policy._grid[120, 120:131] = 1
    policy._target_track = _PublicTargetTrack(
        surface_xy=np.array([1.2, 0.0], dtype=np.float32),
        seed_pose_xy=np.zeros(2, dtype=np.float32),
        last_seen_step=1,
        promoted=True,
    )
    record = policy._public_target_track_record(np.zeros(2, dtype=np.float32))
    assert record is not None
    assert record["behavior_type"] == "NAVIGATE"
    assert record["candidate_id"] == "public_target_track"
    assert record["features"]["distance_m"] >= policy.config.candidate_min_distance_m
    assert record["metadata"]["target_goal"]


def test_compass_heading_matches_public_gps_frame() -> None:
    policy = _NoNetworkPolicy()
    # At compass 0, the forward depth ray is GPS +x.
    assert policy._drive_to_goal(
        np.array([0.0, 0.0], dtype=np.float32),
        0.0,
        _ActiveGoal("visible_goal", np.array([1.0, 0.0], dtype=np.float32), 0),
    )["action_args"]["linear_velocity"] > 0.0
    # At compass +pi/2, the same physical forward direction is GPS -y.
    assert policy._drive_to_goal(
        np.array([0.0, 0.0], dtype=np.float32),
        np.pi / 2.0,
        _ActiveGoal("visible_goal", np.array([0.0, -1.0], dtype=np.float32), 0),
    )["action_args"]["linear_velocity"] > 0.0


def test_stationary_linear_commands_use_negative_one_under_v2_scaling() -> None:
    policy = _NoNetworkPolicy()
    assert policy._hold_action()["action_args"]["linear_velocity"] == -1.0
    assert policy._rotate_recovery_action()["action_args"]["linear_velocity"] == -1.0
    turning = policy._drive_to_goal(
        np.array([0.0, 0.0], dtype=np.float32),
        0.0,
        _ActiveGoal("visible_goal", np.array([0.0, 1.0], dtype=np.float32), 0),
    )
    assert turning["action_args"]["linear_velocity"] == -1.0


def test_empty_frontier_scan_keeps_direction_until_a_full_sweep() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, exploration_scan_steps=3)
    signs = [
        policy._rotate_recovery_action()["action_args"]["angular_velocity"]
        for _ in range(4)
    ]
    # The third action completes the first scan; only the following action may
    # reverse.  This must not depend on the 12-step M2 retry cadence.
    assert signs[:3] == [policy.config.angular_velocity] * 3
    assert signs[3] == -policy.config.angular_velocity
    assert policy._exploration_scan_steps == 1


def test_stagnation_counts_only_after_a_forward_command() -> None:
    policy = _NoNetworkPolicy()
    pose = np.array([0.0, 0.0], dtype=np.float32)
    policy._last_pose = pose.copy()
    policy._last_forward_command = False
    policy._update_stagnation(pose)
    assert policy._stagnant_steps == 0
    policy._last_forward_command = True
    policy._update_stagnation(pose)
    assert policy._stagnant_steps == 1


def test_stagnation_starts_a_navigation_only_recovery_turn() -> None:
    policy = _NoNetworkPolicy()
    pose = np.array([0.0, 0.0], dtype=np.float32)
    policy._last_pose = pose.copy()
    policy._last_forward_command = True
    policy._update_stagnation(pose)
    assert policy._active_goal is None
    assert policy._collision_recovery_steps == policy.config.collision_recovery_turn_steps + 1
    policy._step = 1
    policy._collision_recovery_steps -= 1
    action = policy._rotate_recovery_action()
    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == -1.0


def test_failed_forward_motion_becomes_local_obstacle_evidence() -> None:
    policy = _NoNetworkPolicy()
    policy._last_pose = np.array([0.0, 0.0], dtype=np.float32)
    policy._last_forward_heading = 0.0
    policy._last_forward_command = True
    policy._update_stagnation(policy._last_pose.copy())
    blocked = policy._world_to_grid(np.array([[policy.config.collision_obstacle_distance_m, 0.0]], dtype=np.float32))[0]
    assert policy._motion_obstacles[blocked[1], blocked[0]]
    policy._inflate_obstacles()
    assert policy._grid[blocked[1], blocked[0]] == 2
    assert policy._collision_recovery_steps == policy.config.collision_recovery_turn_steps + 1
    assert policy._collision_probe_steps == policy.config.collision_probe_forward_steps


def test_collision_probe_turns_orthogonally_before_moving() -> None:
    policy = _NoNetworkPolicy()
    policy._collision_probe_heading = np.pi / 2.0
    policy._collision_probe_steps = 1
    turning = policy._drive_collision_probe(np.zeros(2, dtype=np.float32), 0.0)
    assert turning["action_args"]["linear_velocity"] == -1.0
    moving = policy._drive_collision_probe(np.zeros(2, dtype=np.float32), np.pi / 2.0)
    assert moving["action_args"]["linear_velocity"] == policy.config.linear_velocity


def test_collision_recovery_keeps_a_consistent_sweep_direction() -> None:
    policy = _NoNetworkPolicy()
    policy._last_pose = np.zeros(2, dtype=np.float32)
    policy._last_forward_command = True
    policy._last_forward_heading = 0.0
    policy._update_stagnation(np.zeros(2, dtype=np.float32))
    assert np.isclose(policy._collision_probe_heading, np.pi / 2.0)
    # A later block while facing the first probe direction continues to pi,
    # rather than reversing into the original blocked heading.
    policy._collision_recovery_steps = 0
    policy._last_pose = np.zeros(2, dtype=np.float32)
    policy._last_forward_command = True
    policy._last_forward_heading = np.pi / 2.0
    policy._stagnant_steps = 0
    policy._update_stagnation(np.zeros(2, dtype=np.float32))
    assert np.isclose(abs(policy._collision_probe_heading), np.pi)


def test_obstacle_inflation_is_derived_from_raw_evidence_each_frame() -> None:
    policy = _NoNetworkPolicy()
    start = np.array([120, 120], dtype=np.int32)
    end = np.array([126, 120], dtype=np.int32)
    policy._mark_ray(start, end)
    policy._inflate_obstacles()
    first = policy._grid.copy()
    policy._inflate_obstacles()
    assert np.array_equal(policy._grid, first)
    assert policy._raw_obstacles[120, 126]
    assert policy._grid[120, 126] == 2


def test_max_range_depth_ray_is_free_space_not_an_obstacle() -> None:
    policy = _NoNetworkPolicy()
    start = np.array([120, 120], dtype=np.int32)
    end = np.array([126, 120], dtype=np.int32)
    policy._mark_ray(start, end, endpoint_is_obstacle=False)
    policy._inflate_obstacles()
    assert not np.any(policy._raw_obstacles)
    assert policy._observed_free[120, 126]
    assert policy._grid[120, 126] == 1


def test_non_body_free_ray_cannot_erase_obstacle_evidence() -> None:
    policy = _NoNetworkPolicy()
    start = np.array([120, 120], dtype=np.int32)
    end = np.array([126, 120], dtype=np.int32)
    policy._mark_ray(start, end, endpoint_is_obstacle=True)
    policy._mark_ray(start, end, endpoint_is_obstacle=False, clears_obstacle_evidence=False)
    assert policy._raw_obstacles[120, 126]


def test_non_body_depth_ray_does_not_create_a_2d_free_corridor() -> None:
    policy = _NoNetworkPolicy()
    start = np.array([120, 120], dtype=np.int32)
    end = np.array([126, 120], dtype=np.int32)
    # Height-inappropriate floor/ceiling rays must not be projected into the
    # base traversability layer at all.
    policy._mark_ray(start, end, endpoint_is_obstacle=False)
    policy._observed_free.fill(False)
    policy._raw_obstacles.fill(False)
    policy._inflate_obstacles()
    assert policy._grid[120, 126] == 0


def test_normalized_zero_depth_is_not_projected_as_a_near_wall() -> None:
    policy = _NoNetworkPolicy()
    depth = policy._depth_meters({"depth": np.zeros((8, 8, 1), dtype=np.float32)})
    assert np.isnan(depth).all()


def test_depth_mapping_uses_height_filter_and_does_not_mark_max_range_wall() -> None:
    policy = _NoNetworkPolicy()
    observations = {
        "depth": np.ones((100, 50, 1), dtype=np.float32),
    }
    policy._integrate_depth(observations, np.array([0.0, 0.0], dtype=np.float32), 0.0)
    assert np.any(policy._observed_free)
    assert not np.any(policy._raw_obstacles)


def test_depth_mapping_marks_low_body_height_obstacle_but_not_floor() -> None:
    policy = _NoNetworkPolicy()
    # A horizontal center-row hit is at camera height and above the body band.
    # A lower-row hit corresponds to a short obstacle near base height.
    depth = np.ones((100, 50, 1), dtype=np.float32)
    depth[50, :, 0] = (1.0 - policy.config.min_depth_m) / (policy.config.max_depth_m - policy.config.min_depth_m)
    depth[72, :, 0] = (1.0 - policy.config.min_depth_m) / (policy.config.max_depth_m - policy.config.min_depth_m)
    policy._integrate_depth({"depth": depth}, np.array([0.0, 0.0], dtype=np.float32), 0.0)
    assert np.any(policy._raw_obstacles)


def test_reachable_route_avoids_an_obstacle_barrier() -> None:
    policy = _NoNetworkPolicy()
    policy._grid.fill(1)
    policy._grid[90:151, 125] = 2
    # A three-cell opening lets a robot-radius-inflated grid pass without a
    # diagonal corner cut.
    policy._grid[89:92, 125] = 1
    costs, parents, start = policy._reachable_tree(np.array([0.0, 0.0], dtype=np.float32))
    target = (130, 120)
    route = policy._reconstruct_route(parents, start, target)
    assert route
    assert np.isfinite(costs[target[1], target[0]])
    assert all(policy._grid[y, x] == 1 for x, y in route)
    assert any(y <= 91 for _, y in route)


def test_route_rejects_diagonal_corner_cut() -> None:
    policy = _NoNetworkPolicy()
    policy._grid.fill(0)
    policy._grid[120, 120] = 1
    policy._grid[121, 121] = 1
    # Orthogonal neighbors stay unknown, so diagonal travel is forbidden.
    costs, _, _ = policy._reachable_tree(np.array([0.0, 0.0], dtype=np.float32))
    assert not np.isfinite(costs[121, 121])


def test_route_waypoint_does_not_drive_directly_through_unknown_cells() -> None:
    policy = _NoNetworkPolicy()
    policy._grid.fill(0)
    policy._grid[120, 120:123] = 1
    goal = _ActiveGoal(
        "frontier:test",
        np.array([0.2, 0.0], dtype=np.float32),
        0,
        ((120, 120), (121, 120), (122, 120)),
    )
    waypoint = policy._route_waypoint(np.array([0.0, 0.0], dtype=np.float32), goal)
    assert waypoint is not None
    assert np.allclose(waypoint, [0.2, 0.0], atol=1e-6)


def test_candidate_records_accept_a_short_reachable_frontier() -> None:
    policy = _NoNetworkPolicy()
    policy._grid.fill(0)
    # Make a short free corridor that ends beside unknown space.  Its frontier
    # is only 0.3 m away, so the local exploration threshold must not discard
    # the only safe action and leave the policy in a blind scan.
    policy._grid[120, 120:124] = 1
    records = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))
    assert records
    assert records[0]["features"]["distance_m"] >= policy.config.candidate_min_distance_m
    assert records[0]["features"]["distance_m"] < 0.5


def test_candidate_records_keep_spatially_separated_options_from_one_large_frontier() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        frontier_candidates_per_component=3,
        frontier_candidate_separation_m=1.0,
    )
    policy._grid.fill(0)
    ys, xs = np.ogrid[: policy._shape[0], : policy._shape[1]]
    disk = (xs - 120) ** 2 + (ys - 120) ** 2 <= 28**2
    policy._grid[disk] = 1
    records = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))
    assert len(records) == 3
    points = np.asarray([record["goal_xyyaw"][:2] for record in records], dtype=np.float32)
    pairwise = [float(np.linalg.norm(points[left] - points[right])) for left in range(3) for right in range(left + 1, 3)]
    assert min(pairwise) >= 0.99
    assert all(record["candidate_id"] in policy._candidate_routes for record in records)


def test_frontier_cooldown_filters_nearby_retries_but_expires() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        frontier_candidates_per_component=3,
        frontier_candidate_separation_m=1.0,
        frontier_revisit_radius_m=0.75,
        frontier_revisit_cooldown_steps=10,
    )
    policy._grid.fill(0)
    ys, xs = np.ogrid[: policy._shape[0], : policy._shape[1]]
    policy._grid[(xs - 120) ** 2 + (ys - 120) ** 2 <= 28**2] = 1
    original = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))
    anchor = np.asarray(original[0]["goal_xyyaw"][:2], dtype=np.float32)
    policy._recent_frontier_attempts = [{"xy": anchor.copy(), "step": policy._step}]
    cooled = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))
    assert cooled
    assert all(
        float(np.linalg.norm(np.asarray(record["goal_xyyaw"][:2], dtype=np.float32) - anchor))
        >= policy.config.frontier_revisit_radius_m
        for record in cooled
    )
    policy._step += policy.config.frontier_revisit_cooldown_steps
    recovered = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))
    assert any(
        float(np.linalg.norm(np.asarray(record["goal_xyyaw"][:2], dtype=np.float32) - anchor))
        < policy.config.frontier_revisit_radius_m
        for record in recovered
    )


def test_frontier_cooldown_soft_fallback_keeps_m2_candidates_when_every_point_is_deferred() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        frontier_candidates_per_component=3,
        frontier_candidate_separation_m=1.0,
        frontier_revisit_radius_m=10.0,
        frontier_revisit_cooldown_steps=10,
    )
    policy._grid.fill(0)
    ys, xs = np.ogrid[: policy._shape[0], : policy._shape[1]]
    policy._grid[(xs - 120) ** 2 + (ys - 120) ** 2 <= 28**2] = 1
    original = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))
    policy._recent_frontier_attempts = [{"xy": np.asarray(original[0]["goal_xyyaw"][:2], dtype=np.float32), "step": policy._step}]

    recovered = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))

    assert recovered
    assert all(record["metadata"]["revisit_deferred"] for record in recovered)
    assert policy._last_candidate_pool_stats["deferred_fallback_records"] == len(recovered)


def test_clear_space_fallback_keeps_m2_routeable_options_when_frontiers_disconnect() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, planner_max_expansions=5000, clear_space_fallback_candidates=3)
    # With no UNKNOWN boundary there is deliberately no frontier, but a normal
    # public map follower still has multiple observed-free routes to offer M2.
    policy._grid.fill(1)
    records = policy._candidate_records(np.array([0.0, 0.0], dtype=np.float32))
    assert records
    assert all(record["candidate_id"].startswith("clear_space:") for record in records)
    assert all(record["behavior_type"] == "EXPLORE" for record in records)
    assert all(record["candidate_id"] in policy._candidate_routes for record in records)
    assert policy._last_candidate_pool_stats["clear_space_fallback_records"] == len(records)


def test_clear_space_fallback_is_selected_through_unmodified_m2_seam() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, planner_max_expansions=5000, clear_space_fallback_candidates=3)
    policy._grid.fill(1)
    model = _ModelSelectionStub(selected_index=-1)
    policy._model = model
    # `_NoNetworkPolicy` only exists for unit tests; bind the actual adapter seam
    # to prove a fallback still passes through Module-2 rather than direct motion.
    policy._select_goal = HabitatInteractiveNavM2Policy._select_goal.__get__(policy, HabitatInteractiveNavM2Policy)
    policy._try_replan(np.array([0.0, 0.0], dtype=np.float32), 0.0)
    assert model.calls
    offered, _context = model.calls[0]
    assert all(candidate.behavior_type == "EXPLORE" for candidate in offered)
    assert all(candidate.interaction_command is None for candidate in offered)
    assert policy._active_goal is not None
    assert policy._active_goal.candidate_id.startswith("clear_space:")


def test_promoted_target_uses_original_m2_target_goal_guard_context() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, target_goal_lock_enabled=True, target_goal_pre_score=1.0)
    model = _ModelSelectionStub(selected_index=0)
    policy._model = model
    policy._select_goal = HabitatInteractiveNavM2Policy._select_goal.__get__(policy, HabitatInteractiveNavM2Policy)
    records = [
        {
            "candidate_id": "public_target_track",
            "behavior_type": "NAVIGATE",
            "target_id": "public_rgbd_target_standoff",
            "target_name": "chair",
            "goal_xyyaw": [1.0, 0.0, 0.0],
            "features": {"distance_m": 1.0, "exploration_gain": 0.0},
            "metadata": {"target_goal": True, "target_visible_now": True},
        },
        {
            "candidate_id": "frontier:130:120",
            "behavior_type": "EXPLORE",
            "target_id": "frontier",
            "target_name": "frontier",
            "goal_xyyaw": [2.0, 0.0, 0.0],
            "features": {"distance_m": 2.0, "exploration_gain": 20.0},
            "metadata": {},
        },
    ]

    policy._select_goal(records, np.zeros(2, dtype=np.float32), 0.0)

    assert len(model.calls) == 1
    _candidates, kwargs = model.calls[0]
    robot_context = kwargs["robot_context"]
    assert robot_context["candidate_pre_scores"] == {"public_target_track": 1.0}
    assert robot_context["candidate_decision_hints"] == {"public_target_track": "TARGET_GOAL"}
    assert "frontier:130:120" not in robot_context["candidate_pre_scores"]


def test_frontier_arrival_scan_consumes_the_configured_turn_budget() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, frontier_arrival_scan_steps=3)
    policy._step = 7
    policy._active_goal = _ActiveGoal(
        candidate_id="frontier:120:120",
        xy=np.array([0.1, 0.0], dtype=np.float32),
        selected_step=1,
        route_cells_xy=((120, 120), (121, 120)),
    )

    action = policy._drive_to_goal(np.array([0.0, 0.0], dtype=np.float32), 0.0, policy._active_goal)

    assert policy._active_goal is None
    assert policy._frontier_arrival_scan_steps == 2
    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == -1.0
    assert action["action_args"]["angular_velocity"] != 0.0


def test_local_escape_relaxation_reconnects_only_raw_safe_observed_free_cells() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, local_escape_relaxation_m=0.2)
    policy._grid.fill(2)
    policy._observed_free.fill(False)
    policy._raw_obstacles.fill(False)
    policy._motion_obstacles.fill(False)
    policy._grid[120, 120] = 1
    policy._observed_free[120, 120:122] = True

    costs, _, _ = policy._reachable_tree(np.array([0.0, 0.0], dtype=np.float32))

    assert np.isfinite(costs[120, 121])
    assert policy._segment_is_traversable((120, 120), (121, 120))
    policy._raw_obstacles[120, 121] = True
    assert not policy._segment_is_traversable((120, 120), (121, 120))


def test_m2_history_uses_stable_public_frontier_regions_and_dynamic_age() -> None:
    policy = _NoNetworkPolicy()
    record = {
        "candidate_id": "frontier:121:120",
        "behavior_type": "EXPLORE",
        "target_id": "public_frontier_ahead",
        "target_name": "unknown_frontier",
        "goal_xyyaw": [0.1, 0.0, 0.0],
        "features": {"distance_m": 0.1, "exploration_gain": 1.0, "visibility_gain": 1.0, "interaction_cost": 0.0},
        "metadata": {"frontier_point": [0.1, 0.0], "map_resolution": 0.1},
    }
    candidates = build_behavior_candidates([record])
    graph = {"nodes": [], "edges": []}
    region_key = candidate_history_key_for(candidates[0], graph)
    policy._candidate_history[region_key] = {
        "selection_count": 2,
        "last_selected_step": 3,
        "last_result": "BLOCKED",
        "low_gain_repeat_count": 1,
    }
    policy._step = 11
    projected = policy._candidate_history_for_model(candidates, graph)
    assert projected[candidates[0].candidate_id]["last_selected_steps_ago"] == 8
    assert projected[region_key]["last_result"] == "BLOCKED"


def test_frontier_goal_progress_timeout_uses_public_gps_distance() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        frontier_progress_timeout_steps=3,
        frontier_progress_min_delta_m=0.1,
    )
    policy._grid.fill(1)
    route = tuple((x, 120) for x in range(120, 141))
    goal = _ActiveGoal(
        "frontier:test",
        np.array([2.0, 0.0], dtype=np.float32),
        selected_step=1,
        route_cells_xy=route,
        best_goal_distance_m=2.0,
        last_goal_progress_step=1,
    )
    policy._active_goal = goal
    policy._step = 5
    assert policy._need_replan(np.array([0.0, 0.0], dtype=np.float32))
    assert policy._last_replan_reason == "frontier_progress_timeout"

    goal.best_goal_distance_m = 2.0
    goal.last_goal_progress_step = 1
    policy._step = 5
    assert not policy._need_replan(np.array([0.2, 0.0], dtype=np.float32))
    assert goal.last_goal_progress_step == policy._step


def test_opt_in_public_trace_contains_policy_state_but_no_task_metrics() -> None:
    policy = _NoNetworkPolicy()
    with tempfile.TemporaryDirectory(dir="/home/ldl/tmp/habitat-objectnav") as directory:
        trace_path = Path(directory) / "public_policy_trace.jsonl"
        policy._diagnostic_trace_path = trace_path
        policy.act(
            {
                "gps": np.array([0.0, 0.0], dtype=np.float32),
                "compass": np.array([0.0], dtype=np.float32),
                "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
            }
        )
        records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert {record["event"] for record in records} >= {"replan", "action"}
    assert all("distance_to_goal" not in record for record in records)
    assert all(record.get("action") in {None, "velocity_control", "velocity_stop"} for record in records)


def test_selection_is_scene_stratified_and_scene_config_is_normalized() -> None:
    Episode = type("Episode", (), {})
    episodes = []
    for scene_id in ("scene-b", "scene-a", "scene-c"):
        for episode_id in ("1", "0"):
            episode = Episode()
            episode.scene_id = scene_id
            episode.episode_id = episode_id
            episode.scene_dataset_config = "./data/scene_datasets/hm3d_v0.2/config.json"
            episodes.append(episode)
    selected = _select_episodes(episodes, scene_count=2, episodes_per_scene=1)
    assert [(item.scene_id, item.episode_id) for item in selected] == [("scene-a", "0"), ("scene-b", "0")]
    focused = _select_episodes(
        episodes,
        scene_count=1,
        episodes_per_scene=1,
        scene_ids=["scene-c"],
    )
    assert [(item.scene_id, item.episode_id) for item in focused] == [("scene-c", "0")]
    _normalize_scene_dataset_config(selected, Path("/home/ldl/habitat-objectnav/data/scene_datasets/hm3d_v0.2/config.json"))
    assert {item.scene_dataset_config for item in selected} == {
        "/home/ldl/habitat-objectnav/data/scene_datasets/hm3d_v0.2/config.json"
    }


def test_mllm_request_stats_separates_successes_and_failures() -> None:
    with tempfile.TemporaryDirectory(dir="/home/ldl/tmp/habitat-objectnav") as directory:
        path = Path(directory) / "mllm_requests.jsonl"
        path.write_text(
            '{"role": "subgoal_selection", "error": ""}\n'
            '{"role": "objectnav_goal_visibility", "error": "timeout"}\n',
            encoding="utf-8",
        )
        assert _mllm_request_stats(path) == {
            "requests": 2,
            "successful": 1,
            "failed": 1,
            "by_role": {
                "subgoal_selection": {"requests": 1, "successful": 1, "failed": 0},
                "objectnav_goal_visibility": {"requests": 1, "successful": 0, "failed": 1},
            },
        }


def test_fresh_mllm_requests_path_preserves_existing_evidence() -> None:
    with tempfile.TemporaryDirectory(dir="/home/ldl/tmp/habitat-objectnav") as directory:
        output = Path(directory)
        current = output / "mllm_requests.jsonl"
        current.write_text("old evidence\n", encoding="utf-8")
        assert _fresh_mllm_requests_path(output) == current
        assert not current.exists()
        assert (output / "mllm_requests.previous.jsonl").read_text(encoding="utf-8") == "old evidence\n"


def test_pointnav_macro_finishes_before_nearby_route_goal_is_cleared() -> None:
    policy = _NoNetworkPolicy()
    policy._pointnav = _PointNavStub([])
    policy._pointnav_macro_commands = [(1, 1.0)]
    policy._last_forward_clearance_m = policy.config.max_depth_m
    goal = _ActiveGoal("frontier:test", np.array([0.1, 0.0], dtype=np.float32), 0, ())
    policy._active_goal = goal
    action = policy._drive_to_goal(
        np.zeros(2, dtype=np.float32),
        0.0,
        goal,
        observations={"depth": np.full((64, 48, 1), 0.5, dtype=np.float32)},
    )
    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == 1.0
    assert policy._pointnav_macro_commands == []
    assert policy._active_goal is not None


def test_pointnav_local_goal_uses_vlfm_gps_compass_frame() -> None:
    policy = _NoNetworkPolicy()
    stub = _PointNavStub([2])  # LEFT
    policy._pointnav = stub
    policy._last_forward_clearance_m = policy.config.max_depth_m
    goal = _ActiveGoal("frontier:test", np.array([0.0, -1.0], dtype=np.float32), 0, ())
    action = policy._pointnav_local_action(
        {"depth": np.full((64, 48, 1), 0.5, dtype=np.float32)},
        np.array([0.0, 0.0], dtype=np.float32),
        0.0,
        goal,
        np.array([0.0, -1.0], dtype=np.float32),
    )
    assert action is not None
    assert np.isclose(stub.calls[0][1], 1.0)
    assert np.isclose(stub.calls[0][2], np.pi / 2.0)
    assert stub.calls[0][3] is True
    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == -1.0
    assert action["action_args"]["angular_velocity"] == 1.0


def test_pointnav_forward_macro_matches_one_quarter_metre_without_objectnav_stop() -> None:
    policy = _NoNetworkPolicy()
    policy._pointnav = _PointNavStub([1])  # FORWARD
    policy._last_forward_clearance_m = policy.config.max_depth_m
    goal = _ActiveGoal("frontier:test", np.array([1.0, 0.0], dtype=np.float32), 0, ())
    observations = {"depth": np.full((64, 48, 1), 0.5, dtype=np.float32)}
    actions = [
        policy._pointnav_local_action(
            observations,
            np.array([0.0, 0.0], dtype=np.float32),
            0.0,
            goal,
            np.array([1.0, 0.0], dtype=np.float32),
        )
        for _ in range(9)
    ]
    assert all(action is not None and action["action"] == "velocity_control" for action in actions)
    assert [action["action_args"]["linear_velocity"] for action in actions[:8]] == [1.0] * 8
    assert np.isclose(actions[8]["action_args"]["linear_velocity"], -1.0 / 3.0)
    # One PointNav network query expands into the nine public continuous actions;
    # its local STOP ID is never emitted as a Habitat ObjectNav stop action.
    assert len(policy._pointnav.calls) == 1
    assert all(action["action"] != "velocity_stop" for action in actions)


def test_pointnav_stop_prediction_falls_back_without_velocity_stop() -> None:
    policy = _NoNetworkPolicy()
    policy._pointnav = _PointNavStub([0])
    goal = _ActiveGoal("frontier:test", np.array([1.0, 0.0], dtype=np.float32), 0, ())
    result = policy._pointnav_local_action(
        {"depth": np.full((64, 48, 1), 0.5, dtype=np.float32)},
        np.array([0.0, 0.0], dtype=np.float32),
        0.0,
        goal,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert result is None
    assert policy._pointnav_stop_predictions == 1
    # The worker recorded local STOP in its recurrent state, but Habitat did not
    # execute it; the next local query must begin from a reset state.
    assert policy._pointnav_reset_pending
    assert policy._pointnav_anchor_xy is None


def test_habitat_v2_profile_enforces_module1_module2_and_module3_boundary() -> None:
    profile_path = (
        Path(__file__).resolve().parents[2]
        / "configs/habitat_objectnav_v2/module1_detector_m2_navigation_only.yaml"
    )
    profile = load_adapter_profile(profile_path)
    overrides = profile.policy_overrides()
    assert profile.profile == "habitat-objectnav-v2-m1-detector-m2-navigation-only"
    assert overrides["module1_detector_enabled"]
    assert overrides["mllm_visual_enabled"] is False
    assert overrides["module3_enabled"] is False
    assert overrides["module3_fail_closed"] is True
    assert overrides["allowed_behavior_types"] == ("EXPLORE", "NAVIGATE")
    assert overrides["goal_reached_distance_m"] == 0.1
    assert overrides["vision_stop_enabled"] is False


def test_module1_target_aliases_cover_all_public_v2_goal_names() -> None:
    assert goal_label_aliases("chair") == {"chair"}
    assert goal_label_aliases("sofa") == {"sofa", "couch"}
    assert goal_label_aliases("plant") == {"plant", "potted_plant"}
    assert goal_label_aliases("tv_monitor") == {"tv", "television", "tv_monitor", "monitor"}


def test_module1_detector_only_evidence_is_track_only_and_uses_public_depth() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        module1_detector_enabled=True,
        module1_detector_include_depth=True,
        module1_detector_min_confidence=0.35,
    )
    policy._module1_detector = _Module1DetectorStub(
        Module1DetectionResult(
            detections=(
                Module1Detection("chair", "chair", 0.9, (0.25, 0.25, 0.75, 0.75), "test_detector"),
            )
        )
    )
    policy._step = 1
    observations = {
        "rgb": np.zeros((64, 48, 3), dtype=np.uint8),
        "depth": np.full((64, 48, 1), 0.5, dtype=np.float32),
    }
    detection = policy._module1_target_detection(observations["rgb"], "chair", observations)
    assert detection is not None
    assert detection["track_only"] == 1.0
    assert detection["stop_eligible"] == 0.0
    assert np.isclose(detection["depth_m"], 2.75)
    assert policy._module1_detector_queries == 1
    assert policy._module1_detector_positive == 1
    assert len(policy._module1_detector.calls) == 1


def test_module1_bridge_rejects_nonpublic_sidecar_fields() -> None:
    try:
        Module1DetectorSidecarClient._normalize(
            [
                {
                    "semantic_class": "chair",
                    "confidence": 0.9,
                    "bbox": [1, 1, 10, 10],
                    "world_position": {"x": 1.0},
                }
            ],
            width=48,
            height=64,
        )
    except RuntimeError:
        return
    raise AssertionError("Module-1 bridge accepted a forbidden world/goal field")


def test_failed_ros_target_standoff_is_replaced_by_alternatives() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        original_ros_navigation_enabled=True,
        target_standoff_failure_radius_m=0.35,
        target_standoff_failure_cooldown_steps=500,
        target_standoff_retry_angles_deg=(35.0, -35.0, 70.0, -70.0),
    )
    policy._grid.fill(1)
    policy._step = 20
    policy._target_track = _PublicTargetTrack(
        surface_xy=np.array([5.0, 0.0], dtype=np.float32),
        seed_pose_xy=np.array([3.0, 0.0], dtype=np.float32),
        last_seen_step=20,
        observations=3,
        promoted=True,
        confidence=0.9,
    )
    policy._full_ros_candidate_payload = {
        "candidates": [
            {
                "candidate_id": "target:object_track_tv",
                "behavior_type": "NAVIGATE",
                "target_id": "object_track_tv",
                "target_name": "television",
                "goal_xyyaw": [4.0, 0.0, 0.0],
                "features": {"distance_m": 1.0},
                "metadata": {"target_goal": True, "target_visible_now": True},
                "interaction_command": None,
            }
        ]
    }
    failed = _ActiveGoal(
        "target:object_track_tv",
        np.array([4.0, 0.0], dtype=np.float32),
        selected_step=1,
    )
    policy._record_failed_target_standoff(failed, status=4)

    records = policy._full_ros_records(np.array([3.0, 0.0], dtype=np.float32), 0.0)

    assert records
    assert all(record["candidate_id"] != "target:object_track_tv" for record in records)
    assert all(":retry:" in record["candidate_id"] for record in records)
    assert all(record["behavior_type"] == "NAVIGATE" for record in records)
    assert all(record["interaction_command"] is None for record in records)
    assert all(record["metadata"]["target_goal"] for record in records)
    assert all(
        np.linalg.norm(np.asarray(record["goal_xyyaw"][:2]) - failed.xy)
        > policy.config.target_standoff_failure_radius_m
        for record in records
    )


def test_full_ros_target_subgoal_is_snapped_off_occupied_cell() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, original_ros_navigation_enabled=True)
    policy._grid.fill(OCCUPIED)
    pose = np.array([0.0, 0.0], dtype=np.float32)
    for x in np.arange(0.0, 1.21, 0.1):
        gx, gy = policy._world_to_grid(np.array([[x, 0.0]], dtype=np.float32))[0]
        policy._grid[max(0, gy - 1):gy + 2, gx] = FREE
    requested = np.array([0.55, 0.0], dtype=np.float32)
    requested_cell = policy._world_to_grid(requested.reshape(1, 2))[0]
    policy._grid[requested_cell[1], requested_cell[0]] = OCCUPIED
    policy._target_track = _PublicTargetTrack(
        surface_xy=np.array([1.10, 0.0], dtype=np.float32),
        seed_pose_xy=pose.copy(),
        last_seen_step=1,
        observations=3,
        promoted=True,
        confidence=0.9,
    )
    policy._full_ros_candidate_payload = {
        "candidates": [{
            "candidate_id": "target:object_track_tv",
            "behavior_type": "NAVIGATE",
            "target_id": "object_track_tv",
            "target_name": "television",
            "goal_xyyaw": [0.55, 0.0, 0.0],
            "features": {"distance_m": 0.55},
            "metadata": {"target_goal": True},
            "interaction_command": None,
        }]
    }

    records = policy._full_ros_records(pose, 0.0)

    assert len(records) == 1
    goal = np.asarray(records[0]["goal_xyyaw"][:2], dtype=np.float32)
    goal_cell = policy._world_to_grid(goal.reshape(1, 2))[0]
    assert policy._grid[goal_cell[1], goal_cell[0]] == FREE
    assert not np.allclose(goal, requested)
    assert records[0]["metadata"]["collision_free_reachable_subgoal"] is True
    assert policy._candidate_routes[records[0]["candidate_id"]]


def test_stale_move_base_success_does_not_arrive_new_target_candidate() -> None:
    policy = _arrived_m3_policy("STOP")
    policy._move_base_goal_candidate_id = policy._active_goal.candidate_id
    policy._move_base_goal_seen_active = False

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.zeros(2, dtype=np.float32),
        0.0,
        target_measurement={"center_x": 0.5, "confidence": 0.95, "depth_m": 0.60},
    )

    assert action is None
    assert policy._objectgoal_stop_verifier.calls == []


def test_original_ros_navigation_holds_when_cmd_vel_is_stale() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, original_ros_navigation_enabled=True)
    policy._step = 20
    policy._active_goal = _ActiveGoal(
        "target:object_track_tv",
        np.array([1.0, 0.0], dtype=np.float32),
        selected_step=10,
    )
    policy._full_ros_move_base_status = 1
    policy._full_ros_cmd_vel = {"linear_x": 0.3, "angular_z": 0.0}
    policy._full_ros_cmd_vel_age_s = policy.config.ros_cmd_vel_max_age_s + 0.1

    action = policy._act_original_ros_navigation(
        np.zeros(2, dtype=np.float32),
        heading=0.0,
        frame_updated=False,
    )

    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == -1.0


def test_public_no_motion_defers_failed_ros_target_retry() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(policy.config, original_ros_navigation_enabled=True)
    policy._step = 30
    policy._active_goal = _ActiveGoal(
        "target:object_track_tv:retry:1",
        np.array([1.0, 0.5], dtype=np.float32),
        selected_step=20,
    )
    policy._last_pose = np.zeros(2, dtype=np.float32)
    policy._last_forward_command = True
    policy._last_forward_heading = 0.0

    policy._update_stagnation(np.zeros(2, dtype=np.float32))

    assert policy._active_goal is None
    assert len(policy._failed_target_standoffs) == 1
    failure = policy._failed_target_standoffs[0]
    assert failure["root"] == "target:object_track_tv"
    assert failure["reason"] == "public_no_motion"
    assert failure["status"] == -1


def test_new_ros_target_has_no_motion_handoff_grace() -> None:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        original_ros_navigation_enabled=True,
        ros_goal_terminal_grace_steps=3,
    )
    policy._step = 30
    goal = _ActiveGoal(
        "target:object_track_tv:retry:1",
        np.array([1.0, 0.5], dtype=np.float32),
        selected_step=29,
    )
    policy._active_goal = goal
    policy._last_pose = np.zeros(2, dtype=np.float32)
    policy._last_forward_command = True
    policy._last_forward_heading = 0.0

    policy._update_stagnation(np.zeros(2, dtype=np.float32))

    assert policy._active_goal is goal
    assert policy._failed_target_standoffs == []
    assert policy._stagnant_steps == 0


def _arrived_m3_policy(*decisions: str) -> _NoNetworkPolicy:
    policy = _NoNetworkPolicy()
    policy.config = replace(
        policy.config,
        original_ros_navigation_enabled=True,
        module3_enabled=True,
        module3_stop_trigger_distance_m=0.10,
        module3_detector_confirmations=1,
        module3_missing_target_max_steps=2,
        module3_semantic_rejection_limit=2,
    )
    policy._step = 20
    policy._last_full_ros_stack_step = 20
    policy._full_ros_move_base_status = 3
    policy._active_goal = _ActiveGoal(
        "target:object_track_tv",
        np.array([0.05, 0.0], dtype=np.float32),
        selected_step=10,
    )
    policy._move_base_goal_candidate_id = policy._active_goal.candidate_id
    policy._move_base_goal_seen_active = True
    policy._objectgoal_stop_verifier = _M3VerifierStub(*decisions)
    return policy


def test_m3_only_owns_control_after_target_standoff_arrival() -> None:
    policy = _arrived_m3_policy("STOP")
    policy._full_ros_move_base_status = 1
    policy._active_goal.xy = np.array([0.11, 0.0], dtype=np.float32)

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.zeros(2, dtype=np.float32),
        0.0,
        target_measurement={"center_x": 0.5, "confidence": 0.95, "depth_m": 0.60},
    )

    assert action is None
    assert policy._objectgoal_stop_verifier.calls == []


def test_m3_off_center_box_is_sent_directly_to_verifier() -> None:
    policy = _arrived_m3_policy("STOP")

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.zeros(2, dtype=np.float32),
        0.0,
        target_measurement={"center_x": 0.8, "confidence": 0.95, "depth_m": 0.60},
    )

    assert action["action"] == "velocity_stop"
    assert len(policy._objectgoal_stop_verifier.calls) == 1


def test_m3_arrival_rejects_standoff_after_target_stays_invisible() -> None:
    policy = _arrived_m3_policy("STOP")
    observations = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    first = policy._maybe_objectgoal_m3_stop(
        observations, np.zeros(2, dtype=np.float32), 0.0, target_measurement=None
    )
    second = policy._maybe_objectgoal_m3_stop(
        observations, np.zeros(2, dtype=np.float32), 0.0, target_measurement=None
    )

    assert first["action"] == "velocity_control"
    assert first["action_args"]["angular_velocity"] == 1.0
    assert second["action_args"]["linear_velocity"] == -1.0
    assert policy._active_goal is None
    assert policy._failed_target_standoffs[-1]["reason"] == "m3_target_not_visible"


def test_public_no_motion_hands_target_to_m3_without_clearing_goal() -> None:
    policy = _arrived_m3_policy("STOP")
    goal = policy._active_goal
    policy._last_pose = np.zeros(2, dtype=np.float32)
    policy._last_forward_command = True
    policy._last_forward_heading = 0.0

    policy._update_stagnation(np.zeros(2, dtype=np.float32))

    assert policy._active_goal is goal
    assert policy._m3_navigation_terminal_candidate_id == goal.candidate_id
    assert policy._failed_target_standoffs == []
    assert policy._collision_recovery_steps == 0


def test_m3_centered_confirmed_target_can_emit_explicit_stop() -> None:
    policy = _arrived_m3_policy("STOP")

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.zeros(2, dtype=np.float32),
        0.0,
        target_measurement={"center_x": 0.52, "confidence": 0.95, "depth_m": 0.60},
    )

    assert action["action"] == "velocity_stop"
    assert len(policy._objectgoal_stop_verifier.calls) == 1


def test_m3_receives_bbox_center_distance_when_navigation_terminal_is_far() -> None:
    policy = _arrived_m3_policy("CONTINUE")

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.zeros(2, dtype=np.float32),
        0.0,
        target_measurement={"center_x": 0.5, "confidence": 0.95, "depth_m": 1.20},
    )

    assert action["action"] == "velocity_control"
    assert policy._active_goal is not None
    assert len(policy._objectgoal_stop_verifier.calls) == 1
    assert policy._objectgoal_stop_verifier.calls[0]["bbox_center_distance_m"] == 1.20
    assert policy._objectgoal_stop_verifier.calls[0]["max_bbox_center_distance_m"] == 0.85


def test_m3_low_detector_confidence_is_still_sent_to_verifier() -> None:
    policy = _arrived_m3_policy("STOP")

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.zeros(2, dtype=np.float32),
        0.0,
        target_measurement={"center_x": 0.5, "confidence": 0.60, "depth_m": 0.60},
    )

    assert action["action"] == "velocity_stop"
    assert len(policy._objectgoal_stop_verifier.calls) == 1
    assert policy._objectgoal_stop_verifier.calls[0]["detector_confidence"] == 0.60


def test_m3_continue_keeps_exclusive_persistent_scan() -> None:
    policy = _arrived_m3_policy("CONTINUE", "STOP")

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.zeros(2, dtype=np.float32),
        0.0,
        target_measurement={"center_x": 0.5, "confidence": 0.95, "depth_m": 0.60},
    )

    assert action["action"] == "velocity_control"
    assert action["action_args"]["linear_velocity"] == -1.0
    assert action["action_args"]["angular_velocity"] == 1.0


def test_m3_missing_box_turns_shortest_way_toward_public_track() -> None:
    policy = _arrived_m3_policy("STOP")
    policy.config = replace(
        policy.config,
        module3_missing_target_max_steps=80,
        module3_max_verification_steps=120,
    )
    policy._target_track = _PublicTargetTrack(
        surface_xy=np.array([5.123554, -1.174100], dtype=np.float32),
        seed_pose_xy=np.zeros(2, dtype=np.float32),
        last_seen_step=1,
        promoted=True,
    )

    action = policy._maybe_objectgoal_m3_stop(
        {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        np.array([4.080195, -1.264665], dtype=np.float32),
        1.608150,
        target_measurement=None,
    )

    assert action["action_args"]["angular_velocity"] == -1.0
