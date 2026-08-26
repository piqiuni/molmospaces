"""Navigation-only external policy for official Habitat ObjectNav-v2.

The policy maintains a small occupancy grid from *public* RGB-D/GPS/Compass
observations.  An optional original Module-1 detector-only sidecar provides
public 2-D object evidence; the unmodified InteractiveNav Module-2 client
ranks detector-standoff and exploration candidates.  It never creates,
returns, or executes an interaction action.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .mllm import (
    build_behavior_candidates,
    build_model_policy,
    build_objectgoal_stop_verifier,
    build_vision_client,
    candidate_history_key_for,
    local_model_environment,
)
from .grounding_dino import GroundingDinoClient, MobileSamClient, YoloV7Client
from .module1_bridge import FullRosStackClient, Module1DetectorSidecarClient, goal_label_aliases
from .pointnav import PointNavClient


FREE = np.uint8(1)
OCCUPIED = np.uint8(2)


@dataclass(frozen=True)
class PolicyConfig:
    """All policy inputs are public configuration, not task internals."""

    mllm_endpoint: str = "http://127.0.0.1:8000/v1"
    mllm_model: str = "qwen3.6-35b-a3b-fp8"
    mllm_timeout_s: float = 30.0
    mllm_max_tokens: int = 128
    vision_max_tokens: int = 64
    vision_interval_steps: int = 10
    # A Habitat-v2 profile can replace the auxiliary direct MLLM visual-control
    # lane with the original Module-1 detector-only seam.  Module-2 remains the
    # only candidate selector in both cases.
    mllm_visual_enabled: bool = True
    module1_detector_enabled: bool = False
    module1_detector_endpoint: str = ""
    module1_detector_timeout_s: float = 5.0
    module1_detector_interval_steps: int = 5
    module1_detector_min_confidence: float = 0.35
    module1_detector_include_depth: bool = True
    module1_original_scripts: str = (
        "/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/"
        "src/semantic_mapping_py_pkg/scripts"
    )
    module1_detector_require_healthy: bool = True
    full_ros_stack_enabled: bool = False
    full_ros_stack_endpoint: str = ""
    full_ros_stack_timeout_s: float = 20.0
    full_ros_stack_interval_steps: int = 5
    full_ros_stack_require_healthy: bool = True
    # ObjectGoal-v2 uses a navigation-only Module-3 verifier.  It is not the
    # interaction executor: its closed vocabulary is STOP/CONTINUE/RESCAN.
    module3_enabled: bool = False
    module3_fail_closed: bool = True
    module3_mode: str = "disabled"
    module3_stop_trigger_distance_m: float = 0.25
    module3_detector_confirmations: int = 2
    module3_min_confidence: float = 0.80
    allowed_behavior_types: tuple[str, ...] = ("EXPLORE", "NAVIGATE")
    # GroundingDINO is an optional external RGB-only verifier.  It runs in the
    # existing VLFM environment through a loopback worker so the Challenge 2023
    # Python process and the original InteractiveNav sources remain unchanged.
    grounding_dino_endpoint: str = ""
    grounding_dino_timeout_s: float = 5.0
    grounding_dino_interval_steps: int = 10
    grounding_dino_min_confidence: float = 0.55
    # Optional existing VLFM YOLOv7 COCO detector.  In detector-first tracking
    # mode it creates only a public RGB-D measurement; a promoted standoff still
    # goes through Module-2 candidate selection and the detector can never STOP.
    yolov7_endpoint: str = ""
    yolov7_timeout_s: float = 5.0
    yolov7_interval_steps: int = 10
    yolov7_min_confidence: float = 0.35
    detector_first_target_tracking: bool = False
    # Optional loopback MobileSAM mask refinement for a detector box.  It only
    # changes which public depth pixels define a target-surface measurement.
    mobile_sam_endpoint: str = ""
    mobile_sam_timeout_s: float = 5.0
    # Optional released VLFM PointNav checkpoint, served from an isolated local
    # Torch runtime. It receives only normalized public depth and the public
    # relative vector to an M2-selected map-route waypoint; it never receives
    # ObjectNav GT or is allowed to emit a v2 STOP.
    pointnav_endpoint: str = ""
    pointnav_timeout_s: float = 1.0
    pointnav_forward_action_steps: int = 8
    pointnav_turn_action_steps: int = 12
    # The released policy was trained with a wider depth camera than v2 Stretch;
    # restrict it to a short, map-confirmed route segment rather than using it as
    # a global planner through unseen geometry.
    pointnav_goal_lookahead_m: float = 0.75
    vision_min_depth_m: float = 0.20
    # ObjectNav-v2 success is defined by distance to a navigable valid viewpoint,
    # rather than raw camera-to-object range.  Visible pixels alone cannot prove
    # the 0.1 m success condition, so the RGB-D gate remains conservative.
    vision_stop_depth_m: float = 0.60
    vision_stop_enabled: bool = True
    # Keep a separate short-approach standoff.  It must not inherit the stop
    # threshold because target camera depth is not a calibrated global position.
    vision_waypoint_standoff_m: float = 0.60
    vision_stop_center_tolerance: float = 0.18
    vision_stop_confirmations: int = 2
    vision_waypoint_reached_distance_m: float = 0.08
    # A temporally stable RGB box can still be a wrong or unreachable object.
    # Let public depth evidence release a blocked short visual approach and give
    # Module-2 a chance to resume frontier selection before vision retakes it.
    visual_goal_blocked_confirmations: int = 4
    visual_goal_cooldown_steps: int = 20
    # A correct short approach normally needs only a handful of re-localized
    # 0.7 m segments.  Bound any one visual takeover so a stable but wrong box
    # cannot suppress M2 frontier exploration for an entire v2 episode.
    visual_goal_max_contiguous_steps: int = 120
    # A valid v2 target can first appear several metres away.  The controller
    # still advances only a 0.7 m public-RGB-D waypoint at a time and requires
    # later visual confirmation before STOP.
    vision_navigate_max_depth_m: float = 4.0
    vision_min_confidence: float = 0.55
    # A single MLLM frame is not a reliable takeover signal: cluttered HM3D
    # views can yield a confident box on a wall, floor patch, or another object.
    # Require a stable public RGB-D box before it pre-empts frontier following.
    vision_min_box_area: float = 0.002
    vision_max_box_area: float = 0.65
    # Two independently queried frames must agree before vision pre-empts the
    # frontier route.  This is deliberately precision-first: an MLLM box can
    # otherwise turn a chair, wall, or floor patch into a long false approach.
    vision_temporal_iou: float = 0.15
    vision_temporal_depth_delta_m: float = 0.90
    vision_temporal_center_delta: float = 0.28
    # Experimental public-observation target tracker.  A temporally confirmed
    # RGB-D box is projected into the public GPS frame and must be re-observed
    # after the robot has moved before it can become a navigation candidate.
    # Keep this opt-in until focused official-v2 traces establish that it helps.
    persistent_target_tracking: bool = False
    target_track_min_baseline_m: float = 0.25
    # The image query cadence is 10 control actions, or roughly 0.3 m of
    # forward travel.  Accept a small tolerance for the public GPS/action
    # discretization so the second view can actually satisfy the baseline gate.
    target_track_baseline_tolerance_m: float = 0.06
    target_track_association_base_m: float = 0.45
    target_track_association_depth_fraction: float = 0.15
    target_track_standoff_m: float = 1.00
    target_track_standoff_tolerance_m: float = 0.45
    target_track_max_path_m: float = 8.00
    target_track_expire_steps: int = 90
    metrics_path: str = "/home/ldl/outputs/habitat_objectnav_v2_m2/mllm_requests.jsonl"
    map_resolution_m: float = 0.10
    # GPS is relative to the episode start.  Keep a generously sized local
    # window so normal exploration does not silently clamp depth endpoints onto
    # an artificial map boundary.
    map_extent_m: float = 64.0
    obstacle_inflation_m: float = 0.20
    min_depth_m: float = 0.50
    max_depth_m: float = 5.00
    # Project public depth into a height-filtered 2D obstacle layer.  The
    # lower image rows see floor; the middle rows can contain low furniture
    # that blocks the 0.17 m-radius Stretch base but lies below its camera.
    mapping_row_stride: int = 12
    # Keep very low furniture in the obstacle layer too.  Habitat collision is
    # governed by the Stretch base rather than its high-mounted camera.
    obstacle_min_height_m: float = 0.10
    obstacle_max_height_m: float = 2.00
    max_depth_endpoint_margin_m: float = 0.02
    hfov_degrees: float = 42.0
    camera_height_m: float = 1.31
    candidate_count: int = 6
    # A newly observed corridor can initially expose only a short reachable
    # frontier.  Accepting it at 0.2 m lets Module-2 choose a real local action
    # rather than forcing a blind scan until a 0.5 m candidate happens to form;
    # the public depth clearance and obstacle-inflated route checks still govern
    # every forward command.
    candidate_min_distance_m: float = 0.20
    candidate_preferred_distance_m: float = 2.00
    candidate_max_distance_m: float = 7.00
    # A single connected frontier can wrap around the robot.  Preserve several
    # spatially separated, map-reachable points from it so Module-2 can choose a
    # direction instead of inheriting an arbitrary connected-component scan order.
    frontier_candidates_per_component: int = 3
    frontier_candidate_separation_m: float = 1.00
    # Treat a frontier as a small public map region rather than only its exact
    # grid-cell ID.  When a route has been abandoned, nearby replacements are
    # withheld briefly so Module-2 can choose a genuinely different direction.
    # Leave spatial deferral opt-in until it improves a full official episode:
    # a focused 00800 trace showed that an overly broad cooldown can exhaust the
    # reachable candidate pool and turn the robot into a blind scanner.
    frontier_revisit_cooldown_steps: int = 0
    frontier_revisit_radius_m: float = 0.75
    # A route can remain technically free while rotation or map churn prevents
    # it from making geographic progress.  Bound that condition using only the
    # public GPS distance to the M2-selected frontier, not task success metrics.
    frontier_progress_timeout_steps: int = 180
    frontier_progress_min_delta_m: float = 0.15
    # If depth evidence fragments all map-reachable frontier cells, an optional
    # experiment can retain exploration through a few routeable *observed-free*
    # waypoints.  A focused long-horizon run showed that zero-information hops
    # can move away from the task, so keep this opt-in until it shows a full-run
    # benefit.  When enabled, they are still normal EXPLORE candidates ranked by
    # the unchanged M2 policy, never a direct reactive action or simulator query.
    clear_space_fallback_candidates: int = 0
    clear_space_fallback_max_distance_m: float = 3.0
    clear_space_fallback_unknown_radius_m: float = 1.0
    # Reaching a public frontier is an information-gathering event.  Optionally
    # spend a bounded in-place scan there before asking M2 for a replacement;
    # this avoids immediately re-offering the same frontier after only one new
    # depth view.  It is disabled by default pending focused validation.
    frontier_arrival_scan_steps: int = 0
    # Experimental local reconnection for a conservative inflated map.  It may
    # use only already observed-free cells immediately around the public GPS
    # pose, and never crosses raw depth or no-motion obstacle evidence.  Normal
    # path cells remain inflated and every forward command still passes the
    # public depth-clearance gate.
    local_escape_relaxation_m: float = 0.0
    # This controls the recovery scan direction only.  A valid frontier route
    # is retained until reached, blocked by new public depth, or stagnant;
    # periodically discarding it wastes both action and MLLM budgets.
    replan_interval_steps: int = 12
    replan_retry_interval_steps: int = 12
    route_invalid_confirmations: int = 4
    goal_reached_distance_m: float = 0.35
    path_lookahead_m: float = 0.60
    path_waypoint_reached_distance_m: float = 0.18
    planner_max_expansions: int = 24000
    # Consecutive velocity commands normally move about 3 cm per Challenge-v2
    # action.  A smaller threshold tolerates renderer/physics rounding while
    # still detecting an actually blocked forward command promptly.
    stagnation_motion_threshold_m: float = 0.002
    collision_recovery_turn_steps: int = 12
    collision_obstacle_distance_m: float = 0.45
    # After a confirmed no-motion event, scan from the new public RGB-D view and
    # immediately give the next route choice back to Module-2.  A blind forward
    # side probe previously consumed up to ten actions without an M2 decision or
    # visual update and regressed two focused v2 recovery traces.
    collision_probe_forward_steps: int = 0
    # This is a public RGB-D safety gate, not an oracle collision signal.  It
    # prevents a planned forward command when the center of the current view is
    # already too close for the 0.17 m-radius Stretch base plus one action.
    forward_min_clearance_m: float = 0.65
    empty_frontier_forward_steps: int = 4
    # When no reachable frontier exists, keep a camera sweep in one direction
    # long enough to expose genuinely new RGB-D coverage.  Reversing every
    # replan retry (12 actions) only oscillates through a small angle and can
    # leave the map without any candidate indefinitely.  At Challenge-v2's
    # 0.45 rad/s, 96 actions are roughly a 250-degree scan.
    exploration_scan_steps: int = 96
    rotate_threshold_rad: float = 0.25
    # Official ObjectNav-v2 VelocityAction takes normalized inputs in [-1, 1]
    # and maps them to 0..0.3 m/s and -0.45..0.45 rad/s respectively.  Thus
    # a stationary linear command is -1.0, not 0.0.
    linear_velocity: float = 1.0
    angular_velocity: float = 1.0
    max_stagnant_steps: int = 30
    # Disabled by default.  A diagnostic trace records only public policy state
    # and velocity actions; it never contains semantic observations, metrics, or
    # ObjectNav goal geometry.
    diagnostic_trace_path: str = ""


@dataclass
class _ActiveGoal:
    candidate_id: str
    xy: np.ndarray
    selected_step: int
    # Private controller state only.  M2 receives candidate summaries, not a
    # path or any simulator internals.
    route_cells_xy: tuple[tuple[int, int], ...] = ()
    route_index: int = 0
    best_goal_distance_m: float = float("inf")
    last_goal_progress_step: int = 0
    candidate_history_key: str = ""


@dataclass
class _PublicTargetTrack:
    """A public RGB-D surface estimate, never a simulator goal or semantic label."""

    surface_xy: np.ndarray
    seed_pose_xy: np.ndarray
    last_seen_step: int
    observations: int = 1
    promoted: bool = False


class HabitatInteractiveNavM2Policy:
    """External M1-detector/M2 policy with Challenge-v2 velocity output."""

    name = "interactive_nav_m1_detector_m2_navigation_only"
    uses_oracle_gt = False
    permits_interaction = False

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        if not self.config.module3_fail_closed:
            raise ValueError("ObjectGoal Module-3 must remain fail-closed")
        if self.config.module3_enabled and self.config.module3_mode != "objectgoal_stop_verified":
            raise ValueError("Habitat Module-3 may only use objectgoal_stop_verified mode")
        if set(self.config.allowed_behavior_types) != {"EXPLORE", "NAVIGATE"}:
            raise ValueError("Habitat ObjectNav-v2 allows only EXPLORE and NAVIGATE behavior candidates")
        if self.config.module1_detector_enabled and not self.config.module1_detector_endpoint:
            raise ValueError("a Module-1 detector-only sidecar endpoint is required when Module-1 is enabled")
        if self.config.full_ros_stack_enabled and not self.config.full_ros_stack_endpoint:
            raise ValueError("a full ROS navigation stack endpoint is required when enabled")
        if self.config.full_ros_stack_enabled and self.config.module1_detector_enabled:
            raise ValueError("detector-only and full ROS stack modes are mutually exclusive")
        local_model_environment(self.config.mllm_endpoint)
        Path(self.config.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        self._diagnostic_trace_path = (
            Path(self.config.diagnostic_trace_path) if self.config.diagnostic_trace_path else None
        )
        if self._diagnostic_trace_path is not None:
            self._diagnostic_trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._diagnostic_trace_path.write_text("", encoding="utf-8")
        self._model = build_model_policy(
            endpoint=self.config.mllm_endpoint,
            model=self.config.mllm_model,
            timeout_s=self.config.mllm_timeout_s,
            max_tokens=self.config.mllm_max_tokens,
            metrics_path=self.config.metrics_path,
        )
        self._vision = build_vision_client(
            endpoint=self.config.mllm_endpoint,
            model=self.config.mllm_model,
            timeout_s=self.config.mllm_timeout_s,
            max_tokens=self.config.vision_max_tokens,
            metrics_path=self.config.metrics_path,
        )
        self._objectgoal_stop_verifier = (
            build_objectgoal_stop_verifier(
                self._vision,
                min_confidence=self.config.module3_min_confidence,
            )
            if self.config.module3_enabled
            else None
        )
        self._module1_detector = (
            Module1DetectorSidecarClient(
                endpoint=self.config.module1_detector_endpoint,
                timeout_s=self.config.module1_detector_timeout_s,
                include_depth=self.config.module1_detector_include_depth,
                original_module1_scripts=self.config.module1_original_scripts,
                metrics_path=self.config.metrics_path,
            )
            if self.config.module1_detector_enabled
            else None
        )
        self._full_ros_stack = (
            FullRosStackClient(
                endpoint=self.config.full_ros_stack_endpoint,
                timeout_s=self.config.full_ros_stack_timeout_s,
                metrics_path=self.config.metrics_path,
            )
            if self.config.full_ros_stack_enabled
            else None
        )
        self._grounding_dino = (
            GroundingDinoClient(
                self.config.grounding_dino_endpoint,
                timeout_s=self.config.grounding_dino_timeout_s,
                metrics_path=self.config.metrics_path,
            )
            if self.config.grounding_dino_endpoint
            else None
        )
        self._yolov7 = (
            YoloV7Client(
                self.config.yolov7_endpoint,
                timeout_s=self.config.yolov7_timeout_s,
                metrics_path=self.config.metrics_path,
            )
            if self.config.yolov7_endpoint
            else None
        )
        self._mobile_sam = (
            MobileSamClient(
                self.config.mobile_sam_endpoint,
                timeout_s=self.config.mobile_sam_timeout_s,
                metrics_path=self.config.metrics_path,
            )
            if self.config.mobile_sam_endpoint
            else None
        )
        self._pointnav = (
            PointNavClient(
                self.config.pointnav_endpoint,
                timeout_s=self.config.pointnav_timeout_s,
                metrics_path=self.config.metrics_path,
            )
            if self.config.pointnav_endpoint
            else None
        )
        cells = int(round(self.config.map_extent_m / self.config.map_resolution_m))
        self._shape = (cells, cells)
        self._origin = np.array([cells // 2, cells // 2], dtype=np.float32)
        self.reset({})

    def reset(self, episode_public: dict[str, Any]) -> None:
        """Reset from evaluator IDs and the public ObjectGoal category label."""

        self._episode = dict(episode_public)
        self._grid = np.zeros(self._shape, dtype=np.uint8)
        # Keep sensor evidence separate from its inflated planning projection.
        # Re-inflating an already-inflated grid would grow obstacles on every
        # RGB-D frame until the map became unusable.
        self._observed_free = np.zeros(self._shape, dtype=bool)
        self._raw_obstacles = np.zeros(self._shape, dtype=bool)
        # A failed public velocity command is also valid collision evidence.  It
        # is kept separately so a later long depth ray cannot erase it before a
        # recovery scan has a chance to find an alternate route.
        self._motion_obstacles = np.zeros(self._shape, dtype=bool)
        self._active_goal: _ActiveGoal | None = None
        self._step = 0
        self._last_pose: np.ndarray | None = None
        self._stagnant_steps = 0
        self._last_forward_command = False
        self._last_forward_heading: float | None = None
        self._collision_recovery_steps = 0
        # Continue a consistent 90-degree sweep after a real no-motion
        # observation.  After a side probe reaches another obstacle, reversing
        # the turn would point back at the first blocked direction; continuing
        # the sweep instead samples the next side of the local obstacle.
        self._recovery_turn_direction = 1.0
        self._collision_probe_heading: float | None = None
        self._collision_probe_steps = 0
        self._last_forward_clearance_m = self.config.max_depth_m
        self._decision_history: list[dict[str, Any]] = []
        self._candidate_history: dict[str, dict[str, Any]] = {}
        self._candidate_routes: dict[str, tuple[tuple[int, int], ...]] = {}
        self._recent_frontier_attempts: list[dict[str, Any]] = []
        self._last_candidate_pool_stats: dict[str, int] = {}
        self._last_replan_attempt_step = -self.config.replan_retry_interval_steps
        self._last_replan_reason = ""
        self._trace_pose_xy: np.ndarray | None = None
        self._trace_heading: float | None = None
        self._last_vision_step = -self.config.vision_interval_steps
        self._last_module1_detector_step = -self.config.module1_detector_interval_steps
        self._last_full_ros_stack_step = -self.config.full_ros_stack_interval_steps
        self._full_ros_graph: dict[str, Any] = {}
        self._full_ros_candidate_payload: dict[str, Any] = {}
        self._full_ros_detections: dict[str, Any] = {}
        self._full_ros_queries = 0
        self._full_ros_failed = 0
        self._last_public_detection_step = -1
        self._last_grounding_dino_step = -self.config.grounding_dino_interval_steps
        self._last_yolov7_step = -self.config.yolov7_interval_steps
        self._visible_goal_confirmations = 0
        self._pending_visual_detection: dict[str, float] | None = None
        # Keep a fresh single-frame RGB-D measurement separate from the image-IoU
        # gate.  A seeded 3D track needs it to associate a later camera view
        # after the robot has moved and the image box naturally changes position.
        self._latest_public_visual_measurement: dict[str, float] | None = None
        self._visual_goal_blocked_steps = 0
        self._visual_goal_clearance_blocks = 0
        self._visual_goal_releases = 0
        self._visual_goal_budget_releases = 0
        self._visual_goal_contiguous_steps = 0
        self._visual_goal_cooldown_until_step = 0
        self._target_track: _PublicTargetTrack | None = None
        self._target_track_seeded = 0
        self._target_track_promoted = 0
        self._target_track_updated = 0
        self._target_track_rejected = 0
        self._target_track_expired = 0
        self._awaiting_visual_confirmation = False
        self._route_invalid_streak = 0
        self._empty_frontier_forward_count = 0
        self._frontier_arrival_scan_steps = 0
        self._exploration_turn_direction = 1.0
        self._exploration_scan_steps = 0
        self._vision_queries = 0
        self._vision_failed = 0
        self._vision_positive = 0
        self._vision_temporally_confirmed = 0
        self._visual_controller_steps = 0
        self._visual_stop_emitted = 0
        self._grounding_dino_queries = 0
        self._grounding_dino_failed = 0
        self._grounding_dino_confirmed = 0
        self._yolov7_queries = 0
        self._yolov7_failed = 0
        self._yolov7_positive = 0
        self._module1_detector_queries = 0
        self._module1_detector_failed = 0
        self._module1_detector_detections = 0
        self._module1_detector_positive = 0
        self._mobile_sam_queries = 0
        self._mobile_sam_failed = 0
        self._mobile_sam_confirmed = 0
        # Worker macro-action state is private controller bookkeeping. It is
        # reset at each M2 subgoal/episode boundary so the PointNav LSTM never
        # assumes a previous action which the Challenge simulator did not run.
        self._pointnav_macro_commands: list[tuple[int, float]] = []
        self._pointnav_anchor_xy: np.ndarray | None = None
        self._pointnav_reset_pending = True
        self._pointnav_queries = 0
        self._pointnav_failed = 0
        self._pointnav_forward_steps = 0
        self._pointnav_turn_steps = 0
        self._pointnav_stop_predictions = 0
        self._pointnav_safety_blocks = 0
        self._pointnav_macro_aborts = 0
        self._pointnav_aborted_micro_actions = 0
        self._m3_detector_streak = 0
        self._m3_queries = 0
        self._m3_stop_emitted = 0
        self._m3_failed = 0

    def decision_stats(self) -> dict[str, int]:
        """Expose auditable M2 decision provenance without simulator internals."""

        sources = [str(item.get("result_source", "")) for item in self._decision_history]
        model_selected = sum(source.startswith("model") for source in sources)
        return {
            "decisions": len(sources),
            "model_selected": model_selected,
            "fallback_selected": len(sources) - model_selected,
        }

    def vision_stats(self) -> dict[str, int]:
        """Expose visual-controller provenance for the episode report."""

        return {
            "vision_queries": self._vision_queries,
            "vision_failed": self._vision_failed,
            "vision_positive": self._vision_positive,
            "vision_temporally_confirmed": self._vision_temporally_confirmed,
            "visual_controller_steps": self._visual_controller_steps,
            "visual_stop_emitted": self._visual_stop_emitted,
            "visual_goal_clearance_blocks": self._visual_goal_clearance_blocks,
            "visual_goal_releases": self._visual_goal_releases,
            "visual_goal_budget_releases": self._visual_goal_budget_releases,
            "grounding_dino_queries": self._grounding_dino_queries,
            "grounding_dino_failed": self._grounding_dino_failed,
            "grounding_dino_confirmed": self._grounding_dino_confirmed,
            "module1_detector_queries": self._module1_detector_queries,
            "module1_detector_failed": self._module1_detector_failed,
            "module1_detector_detections": self._module1_detector_detections,
            "module1_detector_positive": self._module1_detector_positive,
            "full_ros_queries": self._full_ros_queries,
            "full_ros_failed": self._full_ros_failed,
            "full_ros_graph_nodes": len(self._full_ros_graph.get("nodes") or []),
            "full_ros_graph_edges": len(self._full_ros_graph.get("edges") or []),
            "yolov7_queries": self._yolov7_queries,
            "yolov7_failed": self._yolov7_failed,
            "yolov7_positive": self._yolov7_positive,
            "mobile_sam_queries": self._mobile_sam_queries,
            "mobile_sam_failed": self._mobile_sam_failed,
            "mobile_sam_confirmed": self._mobile_sam_confirmed,
            "target_track_seeded": self._target_track_seeded,
            "target_track_promoted": self._target_track_promoted,
            "target_track_updated": self._target_track_updated,
            "target_track_rejected": self._target_track_rejected,
            "target_track_expired": self._target_track_expired,
            "m3_queries": self._m3_queries,
            "m3_stop_emitted": self._m3_stop_emitted,
            "m3_failed": self._m3_failed,
        }

    def local_control_stats(self) -> dict[str, int]:
        """Expose PointNav worker use separately from MLLM/vision evidence."""

        return {
            "pointnav_queries": self._pointnav_queries,
            "pointnav_failed": self._pointnav_failed,
            "pointnav_forward_steps": self._pointnav_forward_steps,
            "pointnav_turn_steps": self._pointnav_turn_steps,
            "pointnav_stop_predictions": self._pointnav_stop_predictions,
            "pointnav_safety_blocks": self._pointnav_safety_blocks,
            "pointnav_macro_aborts": self._pointnav_macro_aborts,
            "pointnav_aborted_micro_actions": self._pointnav_aborted_micro_actions,
        }

    def diagnostic_snapshot(self) -> dict[str, Any]:
        """Return a copy of public policy state for opt-in visual diagnostics.

        The snapshot deliberately contains no Habitat goal geometry, task
        metrics, semantic sensor, or simulator pose.  Evaluator-only renderers
        may combine it with post-action diagnostics, but nothing in this method
        can affect policy control.
        """

        active = self._active_goal
        track = self._target_track
        return {
            "step": int(self._step),
            "episode": dict(self._episode),
            "pose_xy": self._trace_pose_xy.copy() if self._trace_pose_xy is not None else None,
            "heading": self._trace_heading,
            "grid": self._grid.copy(),
            "map_origin_cell_xy": self._origin.copy(),
            "map_resolution_m": float(self.config.map_resolution_m),
            "active_goal": (
                {
                    "candidate_id": active.candidate_id,
                    "xy": active.xy.copy(),
                    "selected_step": int(active.selected_step),
                    "route_cells_xy": tuple(active.route_cells_xy),
                    "route_index": int(active.route_index),
                }
                if active is not None
                else None
            ),
            "target_track": (
                {
                    "surface_xy": track.surface_xy.copy(),
                    "promoted": bool(track.promoted),
                    "observations": int(track.observations),
                    "last_seen_step": int(track.last_seen_step),
                }
                if track is not None
                else None
            ),
            "detections": copy.deepcopy(self._full_ros_detections),
            "semantic_graph": copy.deepcopy(self._full_ros_graph),
            "candidate_payload": copy.deepcopy(self._full_ros_candidate_payload),
            "last_decision": copy.deepcopy(self._decision_history[-1]) if self._decision_history else None,
            "last_replan_reason": str(self._last_replan_reason),
            "forward_clearance_m": float(self._last_forward_clearance_m),
        }

    def act(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Return a Habitat continuous action; this method cannot return INTERACT."""

        pose_xy, heading = self._public_pose(observations)
        self._update_stagnation(pose_xy)
        self._integrate_depth(observations, pose_xy, heading)
        self._last_forward_clearance_m = self._local_depth_clearance(observations)
        self._step += 1
        self._trace_pose_xy = pose_xy.copy()
        self._trace_heading = float(heading)
        self._expire_public_target_track()
        full_ros_frame_updated = self._update_full_ros_stack(observations, pose_xy, heading)
        if full_ros_frame_updated:
            measurement = self._full_ros_target_detection(observations)
            if measurement is not None:
                self._latest_public_visual_measurement = dict(measurement)
                self._last_public_detection_step = self._step

        m3_action = self._maybe_objectgoal_m3_stop(observations, pose_xy)
        if m3_action is not None:
            return self._emit_action(m3_action)

        if self._collision_recovery_steps > 0:
            self._collision_recovery_steps -= 1
            return self._emit_action(self._rotate_recovery_action())

        if self._collision_probe_steps > 0:
            probe = self._drive_collision_probe(pose_xy, heading)
            if probe["action_args"]["linear_velocity"] > -1.0 + 1e-6:
                self._collision_probe_steps -= 1
            return self._emit_action(probe)

        if self._frontier_arrival_scan_steps > 0:
            self._frontier_arrival_scan_steps -= 1
            return self._emit_action(self._rotate_recovery_action())

        vision_query_due = self._step - self._last_vision_step >= self.config.vision_interval_steps
        visual_goal = self._visible_goal_detection(observations)
        # A detector-first frame may seed a public 3D track before it satisfies
        # the MLLM image-IoU gate.  Once seeded, later fresh RGB-D measurements
        # provide cross-view association evidence even when camera-frame boxes
        # move naturally with the robot.
        track_measurement = visual_goal
        if track_measurement is None and self._last_public_detection_step == self._step:
            track_measurement = self._latest_public_visual_measurement
        track_promoted_now = False
        if track_measurement is not None:
            track_promoted_now = self._update_public_target_track(pose_xy, heading, track_measurement)
        if track_promoted_now:
            # Offer the public target standoff and frontier options to the
            # unmodified M2 selector; never turn it into a direct action.
            self._active_goal = None
            self._try_replan(pose_xy, heading)
            if self._active_goal is not None:
                return self._emit_action(self._drive_to_goal(pose_xy, heading, self._active_goal, observations=observations))
        if visual_goal is not None and bool(visual_goal.get("track_only", 0.0)):
            # Detector-first evidence may only create a public target track.  It
            # does not pre-empt control or emit a stop before M2 has selected a
            # routeable standoff candidate.
            visual_goal = None
        if visual_goal is not None and self._step < self._visual_goal_cooldown_until_step:
            # Do not let the same persistent RGB box immediately recapture the
            # controller after public depth has ruled its short approach blocked.
            # A later image pair may establish a new, genuinely reachable view.
            self._visible_goal_confirmations = 0
            self._pending_visual_detection = None
            self._awaiting_visual_confirmation = False
            visual_goal = None
        if (
            self._awaiting_visual_confirmation
            and (self._active_goal is None or self._active_goal.candidate_id != "visible_goal")
        ):
            # Keep a promising public RGB-D target in the camera until the next
            # scheduled image query can independently confirm it.  Previously
            # the frontier follower moved during this interval, so an initially
            # visible target could leave the frame before the temporal gate ran.
            return self._emit_action(self._hold_action())
        if visual_goal is not None:
            if self._visual_goal_is_safe_to_stop(visual_goal):
                self._visible_goal_confirmations += 1
            else:
                self._visible_goal_confirmations = 0
            if self._visible_goal_confirmations >= self.config.vision_stop_confirmations:
                self._visual_stop_emitted += 1
                return self._emit_action(self.stop_action())
            if self._visual_goal_is_safe_to_stop(visual_goal):
                # Keep the target in view until the next independently queried
                # frame, rather than rotating away while collecting confirmation.
                return self._emit_action(self._hold_action())
            if self._visual_goal_is_navigable(visual_goal):
                proposed_xy = self._visible_goal_waypoint(pose_xy, heading, visual_goal)
                self._active_goal = _ActiveGoal(
                    candidate_id="visible_goal",
                    xy=proposed_xy,
                    selected_step=self._step,
                )
                self._visual_controller_steps += 1
                return self._emit_action(self._drive_to_visible_goal(pose_xy, heading, self._active_goal))
        elif vision_query_due and self._active_goal is not None and self._active_goal.candidate_id == "visible_goal":
            # A re-localization can transiently fail while the robot is turning
            # toward a confirmed target.  Finish the current short (<=0.7 m)
            # public RGB-D waypoint before dropping it; otherwise the controller
            # abandons a valid target after one negative frame.
            distance_to_visual_waypoint = float(np.linalg.norm(self._active_goal.xy - pose_xy))
            if distance_to_visual_waypoint <= self.config.vision_waypoint_reached_distance_m:
                self._active_goal = None
                self._visible_goal_confirmations = 0
                self._visual_goal_contiguous_steps = 0
            else:
                self._visual_controller_steps += 1
                return self._emit_action(self._drive_to_visible_goal(pose_xy, heading, self._active_goal))
        elif self._active_goal is not None and self._active_goal.candidate_id == "visible_goal":
            # A visual waypoint is intentionally short.  Keep following the last
            # temporally confirmed public RGB-D observation between scheduled
            # image queries instead of applying frontier arrival tolerance.
            self._visual_controller_steps += 1
            return self._emit_action(self._drive_to_visible_goal(pose_xy, heading, self._active_goal))
        elif self._visible_goal_confirmations > 0 and self._step - self._last_vision_step < self.config.vision_interval_steps:
            # Preserve a close visual confirmation while waiting for the next
            # scheduled RGB query; do not let frontier exploration rotate away.
            return self._emit_action(self._hold_action())
        elif self._last_vision_step == self._step:
            # A query ran and did not localize the target.  Do not reset a
            # confirmation on intervening control steps where no image query ran.
            self._visible_goal_confirmations = 0

        # Preserve one bounded PointNav macro (at most 0.25 m or 30 degrees)
        # against transient map updates.  Public depth clearance and no-motion
        # recovery still preempt it above, so this never forces travel through an
        # observed obstacle.
        if not self._pointnav_macro_commands and self._need_replan(pose_xy):
            # A newly integrated depth frame can invalidate the local route.
            # Do not turn that into a synchronous M2 request on every action:
            # clear the stale route, sweep for fresh RGB-D coverage, and retry
            # at a bounded cadence.
            self._defer_active_frontier(self._last_replan_reason or "replan")
            self._active_goal = None
            if self._step - self._last_replan_attempt_step >= self.config.replan_retry_interval_steps:
                self._try_replan(pose_xy, heading)

        if self._active_goal is None:
            return self._emit_action(self._empty_frontier_action(heading))
        return self._emit_action(self._drive_to_goal(pose_xy, heading, self._active_goal, observations=observations))

    @staticmethod
    def stop_action() -> dict[str, Any]:
        return {
            "action": "velocity_stop",
            "action_args": {"velocity_stop": np.asarray([1.0], dtype=np.float32)},
        }

    @staticmethod
    def _hold_action() -> dict[str, Any]:
        return {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": -1.0,
                "angular_velocity": 0.0,
                "camera_pitch_angular_velocity": 0.0,
            },
        }

    @staticmethod
    def assert_navigation_only(action: dict[str, Any]) -> None:
        action_name = str(action.get("action") or "")
        if action_name not in {"velocity_control", "velocity_stop"}:
            raise AssertionError(f"forbidden non-navigation action: {action_name}")

    def _public_pose(self, observations: dict[str, Any]) -> tuple[np.ndarray, float]:
        gps = np.asarray(observations["gps"], dtype=np.float32).reshape(-1)
        compass = float(np.asarray(observations["compass"], dtype=np.float32).reshape(-1)[0])
        if gps.size < 2:
            raise ValueError("official v2 policy requires public 2D GPS observation")
        return gps[:2].copy(), compass

    def _update_stagnation(self, pose_xy: np.ndarray) -> None:
        # A commanded in-place turn or a visual-confirmation hold legitimately
        # has zero GPS displacement.  Count stagnation only after a command that
        # actually requested forward motion in the preceding simulator step.
        blocked_forward_motion = (
            self._last_forward_command
            and self._last_pose is not None
            and float(np.linalg.norm(pose_xy - self._last_pose)) < self.config.stagnation_motion_threshold_m
        )
        if blocked_forward_motion:
            self._stagnant_steps += 1
        else:
            self._stagnant_steps = 0
        self._last_pose = pose_xy.copy()
        if blocked_forward_motion and self._stagnant_steps == 1:
            self._record_motion_obstacle()
            if self._active_goal is not None and self._active_goal.candidate_id == "visible_goal":
                self._visual_goal_contiguous_steps = 0
            self._defer_active_frontier("public_no_motion")
            self._active_goal = None
            self._reset_pointnav_controller()
            base_heading = self._last_forward_heading if self._last_forward_heading is not None else 0.0
            self._collision_probe_heading = self._wrap_angle(
                base_heading + self._recovery_turn_direction * (math.pi / 2.0)
            )
            self._collision_probe_steps = self.config.collision_probe_forward_steps
            # Scan first and let the planner select a route through public
            # depth-observed free cells.  A separate "back out" phase would
            # require an additional near-pi rotation under continuous control
            # and spent many actions retrying the blocked approach.
            self._collision_recovery_steps = self.config.collision_recovery_turn_steps + 1

    def _record_motion_obstacle(self) -> None:
        """Convert a public no-motion result into conservative local obstacle evidence."""

        if self._last_pose is None or self._last_forward_heading is None:
            return
        cos_h, sin_h = math.cos(self._last_forward_heading), math.sin(self._last_forward_heading)
        blocked_xy = self._last_pose + np.asarray(
            [
                cos_h * self.config.collision_obstacle_distance_m,
                -sin_h * self.config.collision_obstacle_distance_m,
            ],
            dtype=np.float32,
        )
        cell = self._world_to_grid_unclipped(blocked_xy[None, :])
        if bool(self._cells_in_bounds(cell)[0]):
            x, y = int(cell[0, 0]), int(cell[0, 1])
            self._motion_obstacles[y, x] = True

    def _local_depth_clearance(self, observations: dict[str, Any]) -> float:
        """Robust near-field clearance from public forward RGB-D pixels only."""

        depth = self._depth_meters(observations)
        height, width = depth.shape
        half_width = max(2, width // 12)
        top = max(0, height // 2 - height // 12)
        bottom = min(height, height // 2 + height // 6)
        central = depth[top:bottom, width // 2 - half_width : width // 2 + half_width]
        valid = central[np.isfinite(central) & (central >= self.config.min_depth_m) & (central <= self.config.max_depth_m)]
        return float(np.percentile(valid, 10)) if valid.size else self.config.max_depth_m

    def _emit_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Record whether the next pose transition should count as progress."""

        self.assert_navigation_only(action)
        args = action.get("action_args", {})
        self._last_forward_command = (
            action.get("action") == "velocity_control"
            and float(args.get("linear_velocity", -1.0)) > -1.0 + 1e-6
        )
        if getattr(self, "_diagnostic_trace_path", None) is not None:
            trace_args: dict[str, Any] = {}
            for key, value in args.items():
                array = np.asarray(value)
                trace_args[str(key)] = float(array.reshape(-1)[0]) if array.size == 1 else array.tolist()
            active = self._active_goal
            self._trace_public_event(
                "action",
                pose_xy=(self._trace_pose_xy.tolist() if self._trace_pose_xy is not None else None),
                compass=self._trace_heading,
                action=str(action.get("action")),
                action_args=trace_args,
                forward_clearance_m=float(self._last_forward_clearance_m),
                stagnant_steps=int(self._stagnant_steps),
                collision_recovery_steps=int(self._collision_recovery_steps),
                collision_probe_steps=int(self._collision_probe_steps),
                route_invalid_streak=int(self._route_invalid_streak),
                active_goal=(
                    {
                        "candidate_id": active.candidate_id,
                        "goal_xy": active.xy.tolist(),
                        "selected_step": active.selected_step,
                        "route_cells": len(active.route_cells_xy),
                        "route_index": active.route_index,
                        "best_goal_distance_m": active.best_goal_distance_m,
                        "last_goal_progress_step": active.last_goal_progress_step,
                    }
                    if active is not None
                    else None
                ),
                map_free_cells=int(np.count_nonzero(self._grid == FREE)),
                map_occupied_cells=int(np.count_nonzero(self._grid == OCCUPIED)),
            )
        return action

    def _trace_public_event(self, event: str, **payload: Any) -> None:
        """Append opt-in, public-observation diagnostics without affecting control."""

        trace_path = getattr(self, "_diagnostic_trace_path", None)
        if trace_path is None:
            return
        record = {
            "event": event,
            "episode_id": str(self._episode.get("episode_id") or ""),
            # These fields originate from the evaluator's public episode handle
            # and are used only to join public policy events with post-hoc
            # evaluator metrics.  They are never sent to Module-1 or Module-2.
            "scene_id": str(self._episode.get("scene_id") or ""),
            "object_category": str(self._episode.get("object_category") or ""),
            "step": int(self._step),
            **payload,
        }
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _expire_frontier_cooldowns(self) -> None:
        cooldown_steps = max(0, int(self.config.frontier_revisit_cooldown_steps))
        if cooldown_steps == 0:
            self._recent_frontier_attempts = []
            return
        self._recent_frontier_attempts = [
            attempt
            for attempt in self._recent_frontier_attempts
            if self._step - int(attempt["step"]) < cooldown_steps
        ]

    def _frontier_is_deferred(self, goal_xy: np.ndarray) -> bool:
        self._expire_frontier_cooldowns()
        radius_m = max(0.0, float(self.config.frontier_revisit_radius_m))
        return any(
            float(np.linalg.norm(goal_xy - np.asarray(attempt["xy"], dtype=np.float32))) < radius_m
            for attempt in self._recent_frontier_attempts
        )

    def _defer_active_frontier(self, reason: str) -> None:
        """Remember a public frontier region that a route just abandoned."""

        goal = self._active_goal
        if goal is None or not goal.candidate_id.startswith("frontier:"):
            return
        self._expire_frontier_cooldowns()
        self._recent_frontier_attempts.append(
            {
                "xy": goal.xy.copy(),
                "step": int(self._step),
            }
        )
        history_key = goal.candidate_history_key or goal.candidate_id
        previous = dict(
            self._candidate_history.get(history_key)
            or self._candidate_history.get(goal.candidate_id)
            or {}
        )
        blocked = reason in {
            "public_no_motion",
            "route_invalid",
            "route_waypoint_unavailable",
            "forward_clearance",
            "pointnav_forward_clearance",
            "frontier_progress_timeout",
        }
        previous.update(
            {
                "last_result": "BLOCKED" if blocked else "REACHED_FRONTIER",
                "low_gain_repeat_count": int(previous.get("low_gain_repeat_count", 0)) + int(blocked),
                "last_frontier_shrink_m": 0.0,
            }
        )
        self._candidate_history[goal.candidate_id] = dict(previous)
        self._candidate_history[history_key] = dict(previous)
        self._trace_public_event(
            "frontier_deferred",
            candidate_id=goal.candidate_id,
            goal_xy=goal.xy.tolist(),
            reason=str(reason),
        )

    def _reset_pointnav_controller(self) -> None:
        """Discard an interrupted PointNav macro action and request an LSTM reset."""

        if self._pointnav_macro_commands:
            self._pointnav_macro_aborts += 1
            self._pointnav_aborted_micro_actions += len(self._pointnav_macro_commands)
        self._pointnav_macro_commands = []
        self._pointnav_anchor_xy = None
        self._pointnav_reset_pending = True

    def _drive_collision_probe(self, pose_xy: np.ndarray, heading: float) -> dict[str, Any]:
        """Move a short, orthogonal public-pose probe after an observed block.

        The occupancy map is deliberately conservative and can lack a complete
        route immediately after a low obstacle collision.  A short probe along
        the current sweep direction gives the depth sensor a useful side view
        before asking M2 to choose the next frontier; it uses neither a
        collision sensor nor any scene/task ground truth.
        """

        target_heading = self._collision_probe_heading
        if target_heading is None:
            self._collision_probe_steps = 0
            return self._rotate_recovery_action()
        heading_error = self._wrap_angle(target_heading - heading)
        angular = float(np.clip(heading_error / max(self.config.rotate_threshold_rad, 1e-3), -1.0, 1.0))
        linear = self.config.linear_velocity if abs(heading_error) < self.config.rotate_threshold_rad else -1.0
        if linear > -1.0 + 1e-6 and self._last_forward_clearance_m < self.config.forward_min_clearance_m:
            self._collision_probe_steps = 0
            self._active_goal = None
            return self._rotate_recovery_action()
        if linear > -1.0 + 1e-6:
            self._last_forward_heading = heading
        return {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": float(linear),
                "angular_velocity": float(angular * self.config.angular_velocity),
                "camera_pitch_angular_velocity": 0.0,
            },
        }

    def _need_replan(self, pose_xy: np.ndarray) -> bool:
        if self._active_goal is None:
            self._last_replan_reason = "no_active_goal"
            return True
        if self._active_goal.candidate_id == "visible_goal":
            return False
        if not self._route_is_usable(pose_xy, self._active_goal):
            self._route_invalid_streak += 1
            if self._route_invalid_streak >= self.config.route_invalid_confirmations:
                self._last_replan_reason = "route_invalid"
                return True
            return False
        self._route_invalid_streak = 0
        if self._stagnant_steps >= self.config.max_stagnant_steps:
            self._last_replan_reason = "stagnant"
            return True
        distance_to_goal = float(np.linalg.norm(self._active_goal.xy - pose_xy))
        if self._active_goal.candidate_id.startswith("frontier:"):
            if distance_to_goal + self.config.frontier_progress_min_delta_m <= self._active_goal.best_goal_distance_m:
                self._active_goal.best_goal_distance_m = distance_to_goal
                self._active_goal.last_goal_progress_step = self._step
            elif (
                self._step - self._active_goal.last_goal_progress_step
                >= self.config.frontier_progress_timeout_steps
            ):
                self._last_replan_reason = "frontier_progress_timeout"
                return True
        if distance_to_goal <= self.config.goal_reached_distance_m:
            self._last_replan_reason = "goal_reached"
            return True
        return False

    def _try_replan(self, pose_xy: np.ndarray, heading: float) -> None:
        """Select one fresh reachable frontier, leaving no stale route behind."""

        self._last_replan_attempt_step = self._step
        if getattr(self, "_full_ros_stack", None) is not None:
            records = self._full_ros_records(pose_xy, heading)
            tracked_record = self._public_target_track_record(pose_xy)
            if tracked_record is not None:
                records.append(tracked_record)
        else:
            records = self._candidate_records(pose_xy, heading)
            tracked_record = self._public_target_track_record(pose_xy)
            if tracked_record is not None:
                records.append(tracked_record)
        self._active_goal = self._select_goal(records, pose_xy, heading) if records else None
        if self._active_goal is not None:
            self._active_goal.best_goal_distance_m = float(np.linalg.norm(self._active_goal.xy - pose_xy))
            self._active_goal.last_goal_progress_step = self._step
        # The local PointNav LSTM is conditioned on a point goal. A newly selected
        # M2 candidate changes that public goal, so begin its next request from an
        # explicit recurrent reset instead of replaying an incompatible action.
        self._reset_pointnav_controller()
        if self._active_goal is None or self._active_goal.candidate_id != "visible_goal":
            self._visual_goal_contiguous_steps = 0
        self._route_invalid_streak = 0
        if records:
            self._empty_frontier_forward_count = 0
            self._exploration_scan_steps = 0
        self._trace_public_event(
            "replan",
            reason=self._last_replan_reason,
            candidates=[
                {
                    "candidate_id": str(record["candidate_id"]),
                    "behavior_type": str(record["behavior_type"]),
                    "goal_xy": [float(value) for value in record["goal_xyyaw"][:2]],
                    "distance_m": float(record["features"].get("distance_m", 0.0)),
                    "exploration_gain": float(record["features"].get("exploration_gain", 0.0)),
                    "relative_bearing_rad": float(record.get("metadata", {}).get("relative_bearing_rad", 0.0)),
                }
                for record in records
            ],
            selected_candidate_id=(self._active_goal.candidate_id if self._active_goal is not None else None),
            selected_result_source=(
                self._decision_history[-1].get("result_source")
                if self._decision_history
                else None
            ),
            candidate_pool=dict(self._last_candidate_pool_stats),
        )

    def _update_full_ros_stack(
        self,
        observations: dict[str, Any],
        pose_xy: np.ndarray,
        heading: float,
    ) -> bool:
        full_ros_stack = getattr(self, "_full_ros_stack", None)
        if full_ros_stack is None:
            return False
        if self._step - self._last_full_ros_stack_step < self.config.full_ros_stack_interval_steps:
            return False
        self._last_full_ros_stack_step = self._step
        self._full_ros_queries += 1
        occupancy = np.full(self._shape, -1, dtype=np.int8)
        occupancy[self._grid == FREE] = 0
        occupancy[self._grid == OCCUPIED] = 100
        raw_occupancy = np.full(self._shape, -1, dtype=np.int8)
        raw_occupancy[self._observed_free] = 0
        raw_occupancy[self._raw_obstacles | self._motion_obstacles] = 100
        # Policy y is Habitat GPS y, while ROS y is its reflection.  Reflect the
        # raster together with pose conversion in ros_full_stack_bridge.py.
        occupancy_ros = np.flipud(occupancy)
        raw_occupancy_ros = np.flipud(raw_occupancy)
        global_plan_ros, local_plan_ros = self._ros_plan_rows(pose_xy, heading)
        half_extent = 0.5 * self.config.map_extent_m
        result = full_ros_stack.step(
            image=np.asarray(observations["rgb"], dtype=np.uint8),
            depth_m=self._depth_meters(observations),
            gps_xy=pose_xy,
            compass=heading,
            object_category=str(self._episode.get("object_category") or "object"),
            occupancy_ros=occupancy_ros,
            raw_occupancy_ros=raw_occupancy_ros,
            global_plan_xyyaw_ros=global_plan_ros,
            local_plan_xyyaw_ros=local_plan_ros,
            map_resolution_m=self.config.map_resolution_m,
            map_origin_xy=(-half_extent, -half_extent),
            hfov_degrees=self.config.hfov_degrees,
            context={
                "episode_id": str(self._episode.get("episode_id") or ""),
                "scene_id": str(self._episode.get("scene_id") or ""),
                "step": self._step,
            },
        )
        if result.error:
            self._full_ros_failed += 1
            self._trace_public_event("full_ros_stack_update", error=result.error)
            return False
        self._full_ros_graph = self._graph_ros_to_public(result.graph)
        self._full_ros_candidate_payload = result.candidate_payload
        self._full_ros_detections = result.detections
        self._trace_public_event(
            "full_ros_stack_update",
            candidate_sequence=result.candidate_sequence,
            candidate_count=len(result.candidate_payload.get("candidates") or []),
            graph_nodes=len(result.graph.get("nodes") or []),
            graph_edges=len(result.graph.get("edges") or []),
            detection_count=len(result.detections.get("detections") or []),
        )
        return True

    def _maybe_objectgoal_m3_stop(
        self,
        observations: dict[str, Any],
        pose_xy: np.ndarray,
    ) -> dict[str, Any] | None:
        """Run the fail-closed ObjectGoal M3 only at a detector-backed standoff."""

        verifier = getattr(self, "_objectgoal_stop_verifier", None)
        if verifier is None or self._last_full_ros_stack_step != self._step:
            return None
        active = self._active_goal
        is_target_route = bool(
            active is not None
            and active.candidate_id.startswith(("target:", "public_target_track"))
        )
        distance_m = (
            float(np.linalg.norm(active.xy - pose_xy))
            if active is not None
            else float("inf")
        )
        aliases = {value.replace("_", " ").casefold() for value in goal_label_aliases(
            str(self._episode.get("object_category") or "object")
        )}
        target_rows: list[dict[str, Any]] = []
        for row in self._full_ros_detections.get("detections") or []:
            if not isinstance(row, dict):
                continue
            labels = {
                str(row.get("semantic_class") or "").replace("_", " ").casefold(),
                str(row.get("semantic_class_raw") or "").replace("_", " ").casefold(),
            }
            if aliases.intersection(labels):
                target_rows.append(row)
        # Accumulate detector agreement independently from route arrival.  The
        # target can be seen one frame before M2 selects its final short standoff;
        # coupling the streak to active-goal state would discard that evidence.
        self._m3_detector_streak = self._m3_detector_streak + 1 if target_rows else 0
        gate = bool(
            is_target_route
            and distance_m <= self.config.module3_stop_trigger_distance_m
            and target_rows
        )
        self._trace_public_event(
            "objectgoal_m3_gate",
            eligible=gate,
            detector_streak=self._m3_detector_streak,
            target_detections=len(target_rows),
            active_candidate_id=(active.candidate_id if active is not None else None),
            public_candidate_distance_m=(distance_m if math.isfinite(distance_m) else None),
        )
        if self._m3_detector_streak < max(1, int(self.config.module3_detector_confirmations)):
            return None
        self._m3_queries += 1
        detector_confidence = max(float(row.get("confidence", 0.0) or 0.0) for row in target_rows)
        result = verifier.verify(
            target_name=str(self._episode.get("object_category") or "object"),
            image_data_url=self._rgb_data_url(np.asarray(observations["rgb"], dtype=np.uint8)),
            detector_confidence=detector_confidence,
            public_candidate_distance_m=distance_m,
            metrics_context={
                "episode_id": str(self._episode.get("episode_id") or ""),
                "scene_id": str(self._episode.get("scene_id") or ""),
                "step": self._step,
                "policy": self.name,
            },
        )
        if result.error:
            self._m3_failed += 1
        self._trace_public_event(
            "objectgoal_m3_result",
            decision=result.decision,
            confidence=result.confidence,
            reason=result.reason,
            error=result.error,
        )
        self._m3_detector_streak = 0
        if result.decision == "STOP":
            self._m3_stop_emitted += 1
            return self.stop_action()
        if result.decision == "RESCAN":
            return self._rotate_recovery_action()
        return None

    def _ros_plan_rows(
        self,
        pose_xy: np.ndarray,
        heading: float,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Expose the active public-grid route as ROS Paths for diagnostics.

        The Habitat GPS second axis is reflected into ROS y.  This is telemetry
        only: publishing the path cannot change the local controller or M2 choice.
        """

        goal = self._active_goal
        if goal is None or not goal.route_cells_xy:
            return [], []
        start = min(max(0, int(goal.route_index)), len(goal.route_cells_xy) - 1)
        cells = np.asarray(goal.route_cells_xy[start:], dtype=np.int32)
        if cells.size == 0:
            return [], []
        points = self._grid_to_world(cells)
        points = np.vstack([np.asarray(pose_xy, dtype=np.float32).reshape(1, 2), points])
        rows: list[list[float]] = []
        for index, point in enumerate(points):
            if index + 1 < len(points):
                delta = points[index + 1] - point
                yaw = math.atan2(-float(delta[1]), float(delta[0]))
            else:
                yaw = float(heading)
            rows.append([float(point[0]), -float(point[1]), float(yaw)])
        local_limit = max(2, int(math.ceil(self.config.path_lookahead_m / self.config.map_resolution_m)) + 1)
        return rows, rows[:local_limit]

    @staticmethod
    def _graph_ros_to_public(graph: dict[str, Any]) -> dict[str, Any]:
        """Reflect ROS y coordinates back into the Habitat public GPS frame."""

        converted = copy.deepcopy(graph if isinstance(graph, dict) else {})
        for node in converted.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            for key in ("centroid", "aabb_center"):
                value = node.get(key)
                if isinstance(value, list) and len(value) >= 2:
                    value[1] = -float(value[1])
        for hint in converted.get("navigation_hints") or []:
            if isinstance(hint, dict):
                value = hint.get("position")
                if isinstance(value, list) and len(value) >= 2:
                    value[1] = -float(value[1])
        return converted

    def _full_ros_records(self, pose_xy: np.ndarray, heading: float) -> list[dict[str, Any]]:
        """Route original ROS candidates on the public Habitat occupancy map."""

        rows = self._full_ros_candidate_payload.get("candidates") or []
        if not isinstance(rows, list):
            rows = []
        costs, parents, start_xy = self._reachable_tree(pose_xy)
        ys, xs = np.nonzero((self._grid == FREE) & np.isfinite(costs))
        self._candidate_routes = {}
        records: list[dict[str, Any]] = []
        if not len(xs):
            return records
        cells = np.stack([xs, ys], axis=1).astype(np.int32)
        world = self._grid_to_world(cells)
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            behavior_type = str(raw.get("behavior_type") or "").upper()
            if behavior_type not in set(self.config.allowed_behavior_types):
                continue
            if raw.get("interaction_command") is not None:
                continue
            goal = raw.get("goal_xyyaw")
            if not isinstance(goal, (list, tuple)) or len(goal) < 2:
                continue
            try:
                requested = np.asarray([float(goal[0]), -float(goal[1])], dtype=np.float32)
            except (TypeError, ValueError):
                continue
            errors = np.linalg.norm(world - requested[None, :], axis=1)
            index = min(
                range(len(cells)),
                key=lambda item: (float(errors[item]), float(costs[ys[item], xs[item]]), int(ys[item]), int(xs[item])),
            )
            # A semantic/explorer goal outside known traversable space is not a
            # valid Habitat action yet.  Let later depth/map updates make it
            # executable instead of silently snapping across a wall.
            if float(errors[index]) > 1.0:
                continue
            goal_cell = (int(cells[index, 0]), int(cells[index, 1]))
            route = self._reconstruct_route(parents, start_xy, goal_cell)
            if not route:
                continue
            candidate_id = str(raw.get("candidate_id") or "")
            if not candidate_id:
                continue
            self._candidate_routes[candidate_id] = route
            features = dict(raw.get("features") or {})
            features["distance_m"] = float(costs[goal_cell[1], goal_cell[0]] * self.config.map_resolution_m)
            features.setdefault("exploration_gain", 0.0)
            features.setdefault("visibility_gain", 0.0)
            features.setdefault("interaction_cost", 0.0)
            metadata = dict(raw.get("metadata") or {})
            relative = world[index] - pose_xy
            metadata["relative_bearing_rad"] = self._wrap_angle(
                math.atan2(-float(relative[1]), float(relative[0])) - heading
            )
            metadata["full_ros_graph_candidate"] = True
            record = dict(raw)
            record["candidate_id"] = candidate_id
            record["behavior_type"] = behavior_type
            record["goal_xyyaw"] = [float(world[index, 0]), float(world[index, 1]), float(goal[2]) if len(goal) > 2 else 0.0]
            record["features"] = features
            record["metadata"] = metadata
            record["interaction_command"] = None
            records.append(record)
        self._last_candidate_pool_stats = {
            "source_candidates": len(rows),
            "routeable_records": len(records),
            "graph_nodes": len(self._full_ros_graph.get("nodes") or []),
            "graph_edges": len(self._full_ros_graph.get("edges") or []),
        }
        return records

    def _full_ros_target_detection(self, observations: dict[str, Any]) -> dict[str, float] | None:
        """Convert this frame's ROS detector output into public RGB-D track evidence.

        The ROS graph remains the source for rooms, objects, and frontier
        candidates.  This narrow adapter uses only the current 2-D target box and
        the same public Habitat depth frame, so it can share the conservative
        cross-view tracker used by the detector-only profile.  It cannot issue an
        action or STOP directly.
        """

        rows = self._full_ros_detections.get("detections") or []
        if not isinstance(rows, list):
            rows = []
        goal = str(self._episode.get("object_category") or "object")
        aliases = {value.replace("_", " ").casefold() for value in goal_label_aliases(goal)}
        rgb = np.asarray(observations.get("rgb"))
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        height, width = rgb.shape[:2]
        candidates: list[tuple[dict[str, float], dict[str, Any]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            labels = {
                str(row.get("semantic_class") or "").replace("_", " ").casefold(),
                str(row.get("semantic_class_raw") or "").replace("_", " ").casefold(),
            }
            if not aliases.intersection(labels):
                continue
            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if confidence < self.config.module1_detector_min_confidence:
                continue
            box = row.get("bbox") or row.get("bbox_2d") or row.get("bbox_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in box[:4])
            except (TypeError, ValueError):
                continue
            # Original Module-1 publishes pixel xyxy boxes.  Also accept an
            # already-normalized provider result for interface compatibility.
            if max(abs(x1), abs(x2)) > 1.5 or max(abs(y1), abs(y2)) > 1.5:
                x1, x2 = x1 / max(width, 1), x2 / max(width, 1)
                y1, y2 = y1 / max(height, 1), y2 / max(height, 1)
            x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
            y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
            if x2 - x1 < 0.02 or y2 - y1 < 0.02:
                continue
            depth_m = self._box_depth_meters(observations, height, width, x1, y1, x2, y2)
            if depth_m is None:
                continue
            candidates.append(({
                "center_x": 0.5 * (x1 + x2),
                "center_y": 0.5 * (y1 + y2),
                "depth_m": float(depth_m),
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "stop_eligible": 0.0,
                "track_only": 1.0,
            }, row))
        if not candidates:
            self._trace_public_event(
                "full_ros_target_detection",
                goal_category=goal,
                raw_detection_count=len(rows),
                target_candidate_count=0,
                selected_target=None,
            )
            return None
        selected, source = max(
            candidates,
            key=lambda pair: (pair[0]["confidence"], -abs(pair[0]["center_x"] - 0.5)),
        )
        self._module1_detector_positive += 1
        self._trace_public_event(
            "full_ros_target_detection",
            goal_category=goal,
            raw_detection_count=len(rows),
            target_candidate_count=len(candidates),
            selected_target={
                "label": str(source.get("semantic_class") or ""),
                "raw_label": str(source.get("semantic_class_raw") or ""),
                "confidence": float(selected["confidence"]),
                "bbox_xyxy_normalized": [selected[key] for key in ("x1", "y1", "x2", "y2")],
                "depth_m": float(selected["depth_m"]),
                "source_model": str(source.get("source_model") or "module1_ros_yoloe"),
                "track_only": True,
                "stop_eligible": False,
            },
        )
        return selected

    def _public_target_track_record(self, pose_xy: np.ndarray) -> dict[str, Any] | None:
        """Offer one routeable public target standoff to the unchanged M2 seam.

        The tracker stores only a public RGB-D surface projection.  To prevent
        treating that surface as a privileged final pose, the planner chooses a
        nearby *already observed and reachable* free cell and exposes it as a
        normal NAVIGATE candidate beside the existing EXPLORE frontier records.
        """

        if not self._has_promoted_target_track():
            return None
        track = self._target_track
        assert track is not None
        costs, parents, start_xy = self._reachable_tree(pose_xy)
        free = self._grid == FREE
        ys, xs = np.nonzero(free & np.isfinite(costs))
        if not len(xs):
            return None
        cells = np.stack([xs, ys], axis=1).astype(np.int32)
        world = self._grid_to_world(cells)
        surface_distance = np.linalg.norm(world - track.surface_xy[None, :], axis=1)
        path_distance = costs[ys, xs] * self.config.map_resolution_m
        eligible = (
            (np.abs(surface_distance - self.config.target_track_standoff_m) <= self.config.target_track_standoff_tolerance_m)
            & (path_distance >= self.config.candidate_min_distance_m)
            & (path_distance <= self.config.target_track_max_path_m)
        )
        if not np.any(eligible):
            return None
        eligible_indices = np.flatnonzero(eligible)
        # Prefer the desired surface standoff, then a shorter known free route.
        selected_index = min(
            eligible_indices.tolist(),
            key=lambda index: (
                abs(float(surface_distance[index]) - self.config.target_track_standoff_m),
                float(path_distance[index]),
                int(cells[index, 1]),
                int(cells[index, 0]),
            ),
        )
        goal_cell_xy = (int(cells[selected_index, 0]), int(cells[selected_index, 1]))
        route = self._reconstruct_route(parents, start_xy, goal_cell_xy)
        if not route:
            return None
        candidate_id = "public_target_track"
        self._candidate_routes[candidate_id] = route
        goal_xy = world[selected_index]
        return {
            "candidate_id": candidate_id,
            "behavior_type": "NAVIGATE",
            "target_id": "public_rgbd_target_standoff",
            "target_name": str(self._episode.get("object_category") or "object"),
            "goal_xyyaw": [float(goal_xy[0]), float(goal_xy[1]), 0.0],
            "features": {
                "distance_m": float(path_distance[selected_index]),
                "exploration_gain": 0.0,
                "visibility_gain": 0.0,
                "interaction_cost": 0.0,
            },
            "metadata": {
                "target_goal": True,
                "target_visible_now": track.last_seen_step == self._step,
                "track_observations": track.observations,
                "surface_standoff_m": float(surface_distance[selected_index]),
                "map_resolution": self.config.map_resolution_m,
            },
        }

    def _depth_meters(self, observations: dict[str, Any]) -> np.ndarray:
        raw = np.asarray(observations["depth"], dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        if raw.size == 0:
            raise ValueError("empty public depth observation")
        # Official v2 configuration normalizes [0.5, 5.0] to [0, 1].
        if float(np.nanmax(raw)) <= 1.01:
            # Habitat-Sim emits zero for a ray with no hit.  Habitat-Lab clips
            # and normalizes it to zero, making it otherwise indistinguishable
            # from the sensor's minimum range after de-normalization.  Keep it
            # invalid so it cannot fabricate a near obstacle or free corridor.
            no_hit = raw <= 1e-6
            raw = self.config.min_depth_m + raw * (self.config.max_depth_m - self.config.min_depth_m)
            raw = raw.copy()
            raw[no_hit] = np.nan
        return raw

    def _visible_goal_detection(self, observations: dict[str, Any]) -> dict[str, float] | None:
        """Localize the public ObjectGoal in public RGB and its own depth pixels.

        No Habitat semantic observation, goal position, task metric, or
        interaction state reaches this method.  The MLLM returns a normalized
        target box; the adapter uses only public depth inside that box to steer
        or determine whether a conservative stop is warranted.
        """

        module1_due = bool(
            self._module1_detector is not None
            and self._step - self._last_module1_detector_step >= self.config.module1_detector_interval_steps
        )
        vision_due = bool(
            self.config.mllm_visual_enabled
            and self._step - self._last_vision_step >= self.config.vision_interval_steps
        )
        if not module1_due and not vision_due:
            return None
        self._latest_public_visual_measurement = None
        rgb = observations.get("rgb")
        if rgb is None:
            self._awaiting_visual_confirmation = False
            return None
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            self._awaiting_visual_confirmation = False
            return None
        goal = str(self._episode.get("object_category") or "").strip()
        if not goal:
            self._awaiting_visual_confirmation = False
            return None
        module1_measurement = self._module1_target_detection(image, goal, observations) if module1_due else None
        detector_measurement = (
            module1_measurement
            if self.config.module1_detector_enabled
            else self._yolov7_target_detection(image, goal, observations)
        )
        if not vision_due:
            return self._track_only_detection(detector_measurement)
        self._last_vision_step = self._step
        self._vision_queries += 1
        schema = {
            "name": "objectnav_goal_localization",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "goal_visible": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "bbox_xyxy_normalized": {
                        "type": ["array", "null"],
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["goal_visible", "confidence", "bbox_xyxy_normalized"],
            },
        }
        visual_goal_description = self._visual_goal_description(goal)
        response = self._vision.request_json(
            role="objectnav_goal_localization",
            instruction=(
                "This is the robot's current first-person RGB image. "
                f"ObjectNav target: {visual_goal_description}. "
                "Return true only when one identifiable target instance is clearly visible in the pixels. "
                "Do not box floors, walls, reflections, vague or partial background shapes, or a different "
                "object category. If uncertain or the target is not visible, return false and null for the "
                "box. If yes, return one tight normalized decimal [x1,y1,x2,y2] bounding box, with every "
                "coordinate in [0,1]. Answer only from pixels."
            ),
            context={"goal_category": goal, "policy": self.name},
            images=[self._rgb_data_url(image)],
            response_schema=schema,
            metrics_context={"episode_id": str(self._episode.get("episode_id") or ""), "step": self._step, "policy": self.name},
        )
        payload = response.payload or {}
        box = payload.get("bbox_xyxy_normalized")
        height, width = image.shape[:2]
        mllm_detection: dict[str, float] | None = None
        if response.error:
            self._vision_failed += 1
        elif payload.get("goal_visible") and isinstance(box, list) and len(box) == 4:
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
                confidence = float(payload.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
                x1 = y1 = x2 = y2 = 0.0
            if confidence >= self.config.vision_min_confidence:
                x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
                y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                if (
                    box_width >= 0.02
                    and box_height >= 0.02
                    and self.config.vision_min_box_area <= box_area <= self.config.vision_max_box_area
                ):
                    depth_m = self._box_depth_meters(observations, height, width, x1, y1, x2, y2)
                    if depth_m is not None:
                        mllm_detection = {
                            "center_x": (x1 + x2) / 2.0,
                            "center_y": (y1 + y2) / 2.0,
                            "depth_m": depth_m,
                            "confidence": confidence,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "stop_eligible": 1.0,
                            "track_only": 0.0,
                        }
        # Detector-first mode is deliberately precision-scoped: keep issuing the
        # MLLM visual request for auditable context, but only a COCO detector box
        # may seed the public 3D track.  Otherwise a single MLLM box can hold the
        # robot still for temporal confirmation, preventing the separated public
        # viewpoints that the tracker needs.  Module-2 selection remains the
        # mandatory route decision in both modes.
        detection = (
            detector_measurement
            if self.config.module1_detector_enabled or self.config.detector_first_target_tracking
            else mllm_detection
        )
        if detection is None:
            self._pending_visual_detection = None
            self._awaiting_visual_confirmation = False
            return None
        if mllm_detection is not None and self._grounding_dino is not None:
            verified = self._verify_with_grounding_dino(image, goal, mllm_detection)
            if verified is None:
                self._pending_visual_detection = None
                self._awaiting_visual_confirmation = False
                return None
            verified_depth_m = self._box_depth_meters(
                observations,
                height,
                width,
                verified["x1"],
                verified["y1"],
                verified["x2"],
                verified["y2"],
            )
            if verified_depth_m is None:
                self._pending_visual_detection = None
                self._awaiting_visual_confirmation = False
                return None
            detection = {**verified, "depth_m": verified_depth_m, "stop_eligible": 1.0, "track_only": 0.0}
        # A single valid measurement is not allowed to pre-empt frontier
        # following.  It can, however, be used to associate an already-seeded
        # public target track after the camera has translated.
        self._latest_public_visual_measurement = dict(detection)
        self._last_public_detection_step = self._step
        self._vision_positive += 1
        previous = self._pending_visual_detection
        self._pending_visual_detection = detection
        if previous is None:
            # Detector-first measurements seed public 3D tracking but must not
            # freeze frontier exploration while waiting for a second frame. They
            # can only become a route through the later M2-selected track record.
            self._awaiting_visual_confirmation = not bool(detection.get("track_only", 0.0))
            return None
        iou = self._bbox_iou(previous, detection)
        center_delta = math.hypot(
            previous["center_x"] - detection["center_x"],
            previous["center_y"] - detection["center_y"],
        )
        depth_delta = abs(previous["depth_m"] - detection["depth_m"])
        if (
            iou < self.config.vision_temporal_iou
            or center_delta > self.config.vision_temporal_center_delta
            or depth_delta > self.config.vision_temporal_depth_delta_m
        ):
            self._awaiting_visual_confirmation = True
            return None
        self._awaiting_visual_confirmation = False
        self._vision_temporally_confirmed += 1
        return detection

    def _box_depth_meters(
        self,
        observations: dict[str, Any],
        image_height: int,
        image_width: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float | None:
        """Return robust public depth for one normalized RGB box, if available."""

        left = max(0, min(image_width - 1, int(math.floor(x1 * image_width))))
        right = max(left + 1, min(image_width, int(math.ceil(x2 * image_width))))
        top = max(0, min(image_height - 1, int(math.floor(y1 * image_height))))
        bottom = max(top + 1, min(image_height, int(math.ceil(y2 * image_height))))
        depth = self._depth_meters(observations)[top:bottom, left:right]
        valid_depth = depth[
            np.isfinite(depth)
            & (depth >= self.config.vision_min_depth_m)
            & (depth <= self.config.max_depth_m)
        ]
        if valid_depth.size < max(8, depth.size // 20):
            return None
        return float(np.median(valid_depth))

    def _track_only_detection(self, detection: dict[str, float] | None) -> dict[str, float] | None:
        """Feed detector-only evidence to the public target tracker, never control.

        Image-space IoU is retained only as a short-term quality counter.  The
        actual persistent track associates public RGB-D observations in the GPS
        frame after the robot moves, so a normal cross-view box shift cannot
        prevent tracker promotion.
        """

        if detection is None:
            self._pending_visual_detection = None
            self._awaiting_visual_confirmation = False
            return None
        self._latest_public_visual_measurement = dict(detection)
        self._last_public_detection_step = self._step
        previous = self._pending_visual_detection
        self._pending_visual_detection = dict(detection)
        self._awaiting_visual_confirmation = False
        if previous is None:
            return None
        iou = self._bbox_iou(previous, detection)
        center_delta = math.hypot(
            previous["center_x"] - detection["center_x"],
            previous["center_y"] - detection["center_y"],
        )
        depth_delta = abs(previous["depth_m"] - detection["depth_m"])
        if (
            iou < self.config.vision_temporal_iou
            or center_delta > self.config.vision_temporal_center_delta
            or depth_delta > self.config.vision_temporal_depth_delta_m
        ):
            return None
        self._vision_temporally_confirmed += 1
        return detection

    def _module1_target_detection(
        self,
        image: np.ndarray,
        goal: str,
        observations: dict[str, Any],
    ) -> dict[str, float] | None:
        """Convert original Module-1 detector evidence into a track-only measurement.

        The sidecar gets exactly the original Module-1 external-provider request
        (RGB, optional public depth, and camera intrinsics).  This adapter then
        filters its *predicted* labels against the already-public ObjectGoal and
        estimates depth from the same public Habitat frame.  No detector result
        can directly issue a velocity action or STOP.
        """

        if self._module1_detector is None:
            return None
        self._last_module1_detector_step = self._step
        self._module1_detector_queries += 1
        depth_m = self._depth_meters(observations) if self.config.module1_detector_include_depth else None
        result = self._module1_detector.detect(
            image,
            depth_m,
            self.config.hfov_degrees,
            stamp_index=self._step,
            context={
                "episode_id": str(self._episode.get("episode_id") or ""),
                "step": self._step,
                "policy": self.name,
            },
        )
        if result.error:
            self._module1_detector_failed += 1
            self._trace_public_event(
                "module1_target_detection",
                goal_category=goal,
                query_error=result.error,
                latency_s=float(result.latency_s),
                raw_detection_count=len(result.detections),
                target_candidate_count=0,
                selected_target=None,
            )
            return None
        self._module1_detector_detections += len(result.detections)
        aliases = goal_label_aliases(goal)
        height, width = image.shape[:2]
        candidates: list[tuple[dict[str, float], Any]] = []
        for item in result.detections:
            if item.confidence < self.config.module1_detector_min_confidence:
                continue
            if item.label not in aliases and item.raw_label not in aliases:
                continue
            x1, y1, x2, y2 = item.bbox_xyxy_normalized
            x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
            y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            if (
                box_width < 0.02
                or box_height < 0.02
                or box_area < self.config.vision_min_box_area
                or box_area > self.config.vision_max_box_area
            ):
                continue
            box_depth_m = self._box_depth_meters(observations, height, width, x1, y1, x2, y2)
            if box_depth_m is None:
                continue
            candidates.append(
                ({
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                    "depth_m": box_depth_m,
                    "confidence": float(item.confidence),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "stop_eligible": 0.0,
                    "track_only": 1.0,
                }, item)
            )
        if not candidates:
            self._trace_public_event(
                "module1_target_detection",
                goal_category=goal,
                query_error="",
                latency_s=float(result.latency_s),
                raw_detection_count=len(result.detections),
                target_candidate_count=0,
                selected_target=None,
            )
            return None
        self._module1_detector_positive += 1
        selected, source = max(candidates, key=lambda pair: (pair[0]["confidence"], -abs(pair[0]["center_x"] - 0.5)))
        self._trace_public_event(
            "module1_target_detection",
            goal_category=goal,
            query_error="",
            latency_s=float(result.latency_s),
            raw_detection_count=len(result.detections),
            target_candidate_count=len(candidates),
            selected_target={
                "label": source.label,
                "raw_label": source.raw_label,
                "confidence": float(selected["confidence"]),
                "bbox_xyxy_normalized": [
                    float(selected["x1"]),
                    float(selected["y1"]),
                    float(selected["x2"]),
                    float(selected["y2"]),
                ],
                "depth_m": float(selected["depth_m"]),
                "source_model": source.source_model,
                "track_only": True,
                "stop_eligible": False,
            },
        )
        return selected

    def _mask_depth_meters(self, observations: dict[str, Any], mask: np.ndarray) -> float | None:
        """Return robust public depth restricted to a same-frame segmentation mask."""

        depth = self._depth_meters(observations)
        if mask.ndim != 2 or mask.shape != depth.shape:
            return None
        mask_bool = mask.astype(bool)
        valid_depth = depth[
            mask_bool
            & np.isfinite(depth)
            & (depth >= self.config.vision_min_depth_m)
            & (depth <= self.config.max_depth_m)
        ]
        if valid_depth.size < max(8, int(mask_bool.sum()) // 20):
            return None
        return float(np.median(valid_depth))

    def _yolov7_target_detection(
        self,
        image: np.ndarray,
        goal: str,
        observations: dict[str, Any],
    ) -> dict[str, float] | None:
        """Return one detector-first public RGB-D measurement when enabled.

        YOLO is deliberately a discovery source only: it supplies a COCO box for
        the already-public ObjectGoal class, samples public depth in that box,
        and marks the result as track-only.  The measurement cannot pre-empt
        control or STOP; a later cross-view public track must still yield a
        routeable standoff candidate selected by the unmodified Module-2 client.
        """

        if not self.config.detector_first_target_tracking or self._yolov7 is None:
            return None
        if self._step - self._last_yolov7_step < self.config.yolov7_interval_steps:
            return None
        self._last_yolov7_step = self._step
        self._yolov7_queries += 1
        label = self._yolov7_label(goal)
        result = self._yolov7.detect(
            image,
            label,
            context={
                "episode_id": str(self._episode.get("episode_id") or ""),
                "step": self._step,
                "policy": self.name,
            },
        )
        if result.error:
            self._yolov7_failed += 1
            return None
        height, width = image.shape[:2]
        candidates: list[dict[str, float]] = []
        for item in result.detections:
            if item.confidence < self.config.yolov7_min_confidence:
                continue
            x1, y1, x2, y2 = item.bbox_xyxy_normalized
            x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
            y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            if (
                box_width < 0.02
                or box_height < 0.02
                or box_area < self.config.vision_min_box_area
                or box_area > self.config.vision_max_box_area
            ):
                continue
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            depth_m = self._box_depth_meters(observations, height, width, x1, y1, x2, y2)
            if self._mobile_sam is not None:
                pixel_box = (
                    max(0, min(width - 1, int(math.floor(x1 * width)))),
                    max(0, min(height - 1, int(math.floor(y1 * height)))),
                    max(1, min(width, int(math.ceil(x2 * width)))),
                    max(1, min(height, int(math.ceil(y2 * height)))),
                )
                self._mobile_sam_queries += 1
                mask_result = self._mobile_sam.segment_bbox(
                    image,
                    pixel_box,
                    context={
                        "episode_id": str(self._episode.get("episode_id") or ""),
                        "step": self._step,
                        "policy": self.name,
                    },
                )
                if mask_result.error:
                    # The optional refinement is fail-open: public bounding-box
                    # depth remains valid detector evidence if the isolated worker
                    # is unavailable, and its failure is fully reported.
                    self._mobile_sam_failed += 1
                elif mask_result.mask is not None:
                    mask_depth_m = self._mask_depth_meters(observations, mask_result.mask)
                    if mask_depth_m is None:
                        # A valid segmentation with no valid public depth is not a
                        # useful 3D measurement, so do not silently substitute the
                        # less precise full-box background depth.
                        continue
                    ys, xs = np.nonzero(mask_result.mask)
                    if len(xs):
                        center_x = float((float(np.mean(xs)) + 0.5) / width)
                        center_y = float((float(np.mean(ys)) + 0.5) / height)
                    depth_m = mask_depth_m
                    self._mobile_sam_confirmed += 1
            if depth_m is None:
                continue
            candidates.append(
                {
                    "center_x": center_x,
                    "center_y": center_y,
                    "depth_m": depth_m,
                    "confidence": float(item.confidence),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    # Detector-only evidence is intentionally excluded from the
                    # direct MLLM visual controller and all ObjectNav STOP gates.
                    "stop_eligible": 0.0,
                    "track_only": 1.0,
                }
            )
        if not candidates:
            return None
        self._yolov7_positive += 1
        return max(candidates, key=lambda item: (item["confidence"], -abs(item["center_x"] - 0.5)))

    @staticmethod
    def _yolov7_label(goal: str) -> str:
        """Map public ObjectNav-v2 aliases to the worker's exact COCO labels."""

        aliases = {
            "couch": "couch",
            "sofa": "couch",
            "plant": "potted plant",
            "potted plant": "potted plant",
            "tv_monitor": "tv",
            "television": "tv",
            "tv": "tv",
        }
        return aliases.get(goal.casefold(), goal.casefold())

    def _verify_with_grounding_dino(
        self,
        image: np.ndarray,
        goal: str,
        mllm_detection: dict[str, float],
    ) -> dict[str, float] | None:
        """Require an overlapping open-vocabulary detector box when configured.

        Module-2 still supplies the MLLM observation and remains mandatory for
        subgoal selection.  This optional external adapter only rejects a
        single-frame MLLM box that has no corresponding RGB detector evidence;
        it never reads semantic labels, goal geometry, or Habitat metrics.
        """

        if self._step - self._last_grounding_dino_step < self.config.grounding_dino_interval_steps:
            # A verifier result is tied to the same RGB query, so do not silently
            # reuse an old box for a new MLLM frame.
            return None
        self._last_grounding_dino_step = self._step
        self._grounding_dino_queries += 1
        caption = self._grounding_dino_caption(goal)
        result = self._grounding_dino.detect(
            image,
            caption,
            context={
                "episode_id": str(self._episode.get("episode_id") or ""),
                "step": self._step,
                "policy": self.name,
            },
        )
        if result.error:
            self._grounding_dino_failed += 1
            return None
        candidates = []
        for item in result.detections:
            if item.confidence < self.config.grounding_dino_min_confidence:
                continue
            x1, y1, x2, y2 = item.bbox_xyxy_normalized
            candidate = {
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "confidence": float(item.confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            if self._bbox_iou(mllm_detection, candidate) >= self.config.vision_temporal_iou:
                candidates.append(candidate)
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: (item["confidence"], -abs(item["center_x"] - 0.5)))
        self._grounding_dino_confirmed += 1
        # Keep the MLLM confidence, but use the detector's geometrically tighter
        # box for the public depth sample and bearing calculation.
        return {**mllm_detection, **selected}

    @staticmethod
    def _grounding_dino_caption(goal: str) -> str:
        # The existing VLFM wrapper returns exact phrase matches, so send one
        # canonical category rather than a multi-class caption whose tokens can
        # be merged by GroundingDINO and then filtered out by that wrapper.
        aliases = {
            "couch": "sofa",
            "sofa": "sofa",
            "plant": "plant",
            "tv_monitor": "television",
        }
        return aliases.get(goal.casefold(), goal) + " ."

    @staticmethod
    def _visual_goal_description(goal: str) -> str:
        """Disambiguate the public task labels without exposing extra scene data."""

        if goal.casefold() == "couch":
            return (
                "couch, also called a sofa or loveseat: a continuous upholstered multi-seat "
                "(two-or-more-person) seat with a back; not a single armchair or chair"
            )
        return goal

    @staticmethod
    def _bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
        ix1 = max(left["x1"], right["x1"])
        iy1 = max(left["y1"], right["y1"])
        ix2 = min(left["x2"], right["x2"])
        iy2 = min(left["y2"], right["y2"])
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        left_area = max(0.0, left["x2"] - left["x1"]) * max(0.0, left["y2"] - left["y1"])
        right_area = max(0.0, right["x2"] - right["x1"]) * max(0.0, right["y2"] - right["y1"])
        union = left_area + right_area - intersection
        return float(intersection / union) if union > 1e-8 else 0.0

    def _visual_goal_is_safe_to_stop(self, detection: dict[str, float]) -> bool:
        return (
            self.config.vision_stop_enabled
            and float(detection.get("stop_eligible", 1.0)) >= 1.0
            and detection["depth_m"] <= self.config.vision_stop_depth_m + 1e-3
            and abs(detection["center_x"] - 0.5) <= self.config.vision_stop_center_tolerance
        )

    def _visual_goal_is_navigable(self, detection: dict[str, float]) -> bool:
        return detection["depth_m"] <= self.config.vision_navigate_max_depth_m

    def _visible_goal_waypoint(
        self, pose_xy: np.ndarray, heading: float, detection: dict[str, float]
    ) -> np.ndarray:
        # Estimate a short public-RGB-D waypoint along the detected camera ray.
        # Re-localize after each short segment: an MLLM depth estimate is not a
        # calibrated global target position, and a later visual confirmation is
        # still required before velocity_stop.
        forward = max(
            0.0,
            min(0.70, detection["depth_m"] - self.config.vision_waypoint_standoff_m),
        )
        hfov = math.radians(self.config.hfov_degrees)
        # ``depth_m`` is the camera-z depth, so preserve the detection bearing
        # while applying the same standoff to its forward and lateral components.
        lateral = math.tan((detection["center_x"] - 0.5) * hfov) * forward
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        return np.asarray(
            [
                pose_xy[0] + cos_h * forward + sin_h * lateral,
                pose_xy[1] - sin_h * forward + cos_h * lateral,
            ],
            dtype=np.float32,
        )

    def _public_target_surface_xy(
        self, pose_xy: np.ndarray, heading: float, detection: dict[str, float]
    ) -> np.ndarray:
        """Project one public RGB-D box centre into the public GPS frame.

        This is deliberately only a noisy estimate of an object *surface*.  It
        is not a Habitat goal position and is never passed to the simulator as a
        privileged task field.  A later, spatially separated observation must
        agree with it before the planner may offer a standoff candidate to M2.
        """

        depth = float(detection["depth_m"])
        hfov = math.radians(self.config.hfov_degrees)
        lateral = math.tan((float(detection["center_x"]) - 0.5) * hfov) * depth
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        return np.asarray(
            [
                pose_xy[0] + cos_h * depth + sin_h * lateral,
                pose_xy[1] - sin_h * depth + cos_h * lateral,
            ],
            dtype=np.float32,
        )

    def _expire_public_target_track(self) -> None:
        track = self._target_track
        if (
            track is not None
            and self._step - track.last_seen_step > self.config.target_track_expire_steps
        ):
            self._target_track = None
            if self._active_goal is not None and self._active_goal.candidate_id == "public_target_track":
                self._active_goal = None
            self._target_track_expired += 1
            self._trace_public_event(
                "target_track_expired",
                surface_xy=track.surface_xy.tolist(),
                last_seen_step=int(track.last_seen_step),
                observations=int(track.observations),
                promoted=bool(track.promoted),
            )

    def _has_promoted_target_track(self) -> bool:
        return bool(self.config.persistent_target_tracking and self._target_track and self._target_track.promoted)

    def _update_public_target_track(
        self, pose_xy: np.ndarray, heading: float, detection: dict[str, float]
    ) -> bool:
        """Fuse separated public detections into one conservative target track."""

        if not self.config.persistent_target_tracking:
            return False
        surface_xy = self._public_target_surface_xy(pose_xy, heading, detection)
        track = self._target_track
        if track is None:
            self._target_track = _PublicTargetTrack(surface_xy, pose_xy.copy(), self._step)
            self._target_track_seeded += 1
            self._trace_public_event(
                "target_track_seeded",
                surface_xy=surface_xy.tolist(),
                camera_pose_xy=pose_xy.tolist(),
                depth_m=float(detection["depth_m"]),
                confidence=float(detection["confidence"]),
            )
            return False
        association_limit = max(
            self.config.target_track_association_base_m,
            self.config.target_track_association_depth_fraction * float(detection["depth_m"]),
        )
        association_error = float(np.linalg.norm(surface_xy - track.surface_xy))
        baseline = float(np.linalg.norm(pose_xy - track.seed_pose_xy))
        if association_error > association_limit:
            # A strong but geometrically incompatible box is likely a different
            # instance or a false positive.  Preserve an existing promoted
            # target rather than allowing a single frame to teleport it.  Before
            # promotion there is no trustworthy target yet, so let the newer
            # measurement replace a bad seed instead of suppressing exploration
            # for the entire expiry window.
            self._target_track_rejected += 1
            if not track.promoted:
                self._target_track = _PublicTargetTrack(surface_xy, pose_xy.copy(), self._step)
                self._target_track_seeded += 1
            self._trace_public_event(
                "target_track_rejected",
                surface_xy=surface_xy.tolist(),
                camera_pose_xy=pose_xy.tolist(),
                depth_m=float(detection["depth_m"]),
                confidence=float(detection["confidence"]),
                association_error_m=association_error,
                association_limit_m=association_limit,
                baseline_m=baseline,
                reseeded=not bool(track.promoted),
            )
            return False
        track.last_seen_step = self._step
        if not track.promoted:
            if baseline + self.config.target_track_baseline_tolerance_m < self.config.target_track_min_baseline_m:
                self._trace_public_event(
                    "target_track_pending",
                    surface_xy=surface_xy.tolist(),
                    camera_pose_xy=pose_xy.tolist(),
                    depth_m=float(detection["depth_m"]),
                    confidence=float(detection["confidence"]),
                    association_error_m=association_error,
                    association_limit_m=association_limit,
                    baseline_m=baseline,
                )
                return False
            track.promoted = True
            track.observations += 1
            track.surface_xy = (track.surface_xy + surface_xy) / 2.0
            self._target_track_promoted += 1
            self._trace_public_event(
                "target_track_promoted",
                surface_xy=track.surface_xy.tolist(),
                camera_pose_xy=pose_xy.tolist(),
                depth_m=float(detection["depth_m"]),
                confidence=float(detection["confidence"]),
                association_error_m=association_error,
                association_limit_m=association_limit,
                baseline_m=baseline,
                observations=int(track.observations),
            )
            return True
        # A promoted track only accepts locally consistent, public RGB-D
        # measurements.  The running average is intentionally slow so noisy
        # boxes do not cause the navigation target to oscillate.
        weight = min(4, track.observations)
        track.surface_xy = (track.surface_xy * weight + surface_xy) / float(weight + 1)
        track.observations += 1
        self._target_track_updated += 1
        self._trace_public_event(
            "target_track_updated",
            surface_xy=track.surface_xy.tolist(),
            camera_pose_xy=pose_xy.tolist(),
            depth_m=float(detection["depth_m"]),
            confidence=float(detection["confidence"]),
            association_error_m=association_error,
            association_limit_m=association_limit,
            baseline_m=baseline,
            observations=int(track.observations),
        )
        return False

    @staticmethod
    def _rgb_data_url(image: np.ndarray) -> str:
        import base64
        from io import BytesIO

        from PIL import Image

        if image.shape[2] > 3:
            image = image[..., :3]
        encoded = BytesIO()
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(encoded, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

    def _integrate_depth(self, observations: dict[str, Any], pose_xy: np.ndarray, heading: float) -> None:
        depth = self._depth_meters(observations)
        height, width = depth.shape
        column_stride = max(1, width // 96)
        row_stride = max(1, self.config.mapping_row_stride)
        columns = np.arange(0, width, column_stride, dtype=np.int32)
        rows = np.arange(0, height, row_stride, dtype=np.int32)
        sample_depth = depth[np.ix_(rows, columns)]
        valid = np.isfinite(sample_depth) & (sample_depth >= self.config.min_depth_m) & (sample_depth <= self.config.max_depth_m)
        if not np.any(valid):
            return

        focal = (width / 2.0) / math.tan(math.radians(self.config.hfov_degrees) / 2.0)
        pixel_x = columns[None, :].astype(np.float32)
        pixel_y = rows[:, None].astype(np.float32)
        horizontal = (pixel_x - (width - 1.0) / 2.0) / focal * sample_depth
        vertical = (pixel_y - (height - 1.0) / 2.0) / focal * sample_depth
        # Habitat depth is camera-z depth, so its horizontal/vertical image-plane
        # offsets scale directly with the public depth value.  Convert the point
        # height into an approximate base frame and retain only objects that can
        # obstruct the robot body; floor and ceiling observations stay free rays.
        point_height = self.config.camera_height_m - vertical
        obstacle_height = (point_height >= self.config.obstacle_min_height_m) & (
            point_height <= self.config.obstacle_max_height_m
        )
        forward = sample_depth
        right = horizontal[valid]
        forward = forward[valid]
        # In the public GPS frame, [0] is forward and [1] is right at episode
        # start.  Compass rotates from forward +x toward -y, so
        # forward=[cos, -sin] and right=[sin, cos].  This matches
        # VelocityAction's local -z forward.
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        world_x = pose_xy[0] + cos_h * forward + sin_h * right
        world_y = pose_xy[1] - sin_h * forward + cos_h * right
        endpoints = self._world_to_grid_unclipped(np.stack([world_x, world_y], axis=1))
        terminal_depth = sample_depth[valid]
        terminal_is_obstacle = (
            (terminal_depth < (self.config.max_depth_m - self.config.max_depth_endpoint_margin_m))
            & obstacle_height[valid]
        )
        robot_cell_unclipped = self._world_to_grid_unclipped(pose_xy[None, :])
        if not bool(self._cells_in_bounds(robot_cell_unclipped)[0]):
            # The finite start-relative map window has been exceeded.  Do not
            # clip observations to its edge and fabricate a boundary wall.
            return
        robot_cell = robot_cell_unclipped[0]
        self._observed_free[int(robot_cell[1]), int(robot_cell[0])] = True
        self._raw_obstacles[int(robot_cell[1]), int(robot_cell[0])] = False
        endpoint_in_bounds = self._cells_in_bounds(endpoints)
        body_free_ray = obstacle_height[valid]
        for endpoint, is_obstacle, is_body_free, in_bounds in zip(
            endpoints[::2], terminal_is_obstacle[::2], body_free_ray[::2], endpoint_in_bounds[::2]
        ):
            if not in_bounds or not is_body_free:
                continue
            self._mark_ray(
                robot_cell,
                endpoint,
                endpoint_is_obstacle=bool(is_obstacle),
            )
        self._inflate_obstacles()

    def _world_to_grid(self, points_xy: np.ndarray) -> np.ndarray:
        cells = self._world_to_grid_unclipped(points_xy)
        cells[:, 0] = np.clip(cells[:, 0], 0, self._shape[1] - 1)
        cells[:, 1] = np.clip(cells[:, 1], 0, self._shape[0] - 1)
        return cells

    def _world_to_grid_unclipped(self, points_xy: np.ndarray) -> np.ndarray:
        scale = 1.0 / self.config.map_resolution_m
        cells = np.rint(points_xy * scale + self._origin).astype(np.int32)
        return cells

    def _cells_in_bounds(self, cells_xy: np.ndarray) -> np.ndarray:
        return (
            (cells_xy[:, 0] >= 0)
            & (cells_xy[:, 0] < self._shape[1])
            & (cells_xy[:, 1] >= 0)
            & (cells_xy[:, 1] < self._shape[0])
        )

    def _local_escape_mask(self, start_xy: tuple[int, int]) -> np.ndarray:
        """Return a tiny raw-safe public-free bubble around the current robot cell."""

        radius_m = max(0.0, float(self.config.local_escape_relaxation_m))
        if radius_m <= 0.0:
            return np.zeros(self._shape, dtype=bool)
        radius_cells = radius_m / self.config.map_resolution_m
        yy, xx = np.ogrid[: self._shape[0], : self._shape[1]]
        local = (xx - int(start_xy[0])) ** 2 + (yy - int(start_xy[1])) ** 2 <= radius_cells**2
        return local & self._observed_free & ~self._raw_obstacles & ~self._motion_obstacles

    def _grid_to_world(self, cells_xy: np.ndarray) -> np.ndarray:
        return (cells_xy.astype(np.float32) - self._origin) * self.config.map_resolution_m

    def _mark_ray(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        endpoint_is_obstacle: bool = True,
        clears_obstacle_evidence: bool = True,
    ) -> None:
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1, dtype=np.int32)
        ys = np.linspace(y0, y1, steps + 1, dtype=np.int32)
        # A depth ray observes traversable cells until its terminal obstacle.
        # At the sensor's maximum range it is only a free-space observation, not
        # evidence of a wall.  A later ray that reaches farther may correct stale
        # obstacle evidence along its free portion.
        free_slice = slice(None, -1) if endpoint_is_obstacle else slice(None)
        self._observed_free[ys[free_slice], xs[free_slice]] = True
        if clears_obstacle_evidence:
            self._raw_obstacles[ys[free_slice], xs[free_slice]] = False
        if endpoint_is_obstacle:
            self._observed_free[y1, x1] = False
            self._raw_obstacles[y1, x1] = True

    def _reachable_tree(self, pose_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Compute one deterministic 8-connected shortest-path tree on free map cells."""

        traversable = self._grid == FREE
        start_cell = self._world_to_grid(pose_xy[None, :])[0]
        start_x, start_y = int(start_cell[0]), int(start_cell[1])
        # The robot's public GPS pose is a valid start even when its exact grid
        # cell was not hit by a depth ray this frame.  Unknown cells elsewhere
        # remain non-traversable, so this never creates an unknown shortcut.
        traversable[start_y, start_x] = True
        # A base can be surrounded by a newly inflated halo even though nearby
        # cells have explicit free-ray evidence.  Optionally reconnect just that
        # local raw-safe evidence; unknown and raw-obstacle cells remain closed.
        traversable |= self._local_escape_mask((start_x, start_y))
        height, width = traversable.shape
        costs = np.full((height, width), np.inf, dtype=np.float32)
        parents = np.full((height, width, 2), -1, dtype=np.int32)
        costs[start_y, start_x] = 0.0
        queue: list[tuple[float, int, int]] = [(0.0, start_y, start_x)]
        expansions = 0
        neighbors = (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        )
        while queue and expansions < self.config.planner_max_expansions:
            cost, y, x = heapq.heappop(queue)
            if cost > float(costs[y, x]) + 1e-6:
                continue
            expansions += 1
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or not traversable[ny, nx]:
                    continue
                # Do not let a diagonal route squeeze through two touching
                # obstacles; this is important at the robot-radius-inflated map.
                if dx and dy and (not traversable[y, nx] or not traversable[ny, x]):
                    continue
                candidate_cost = cost + (math.sqrt(2.0) if dx and dy else 1.0)
                if candidate_cost + 1e-6 >= float(costs[ny, nx]):
                    continue
                costs[ny, nx] = candidate_cost
                parents[ny, nx] = (x, y)
                heapq.heappush(queue, (candidate_cost, ny, nx))
        return costs, parents, (start_x, start_y)

    @staticmethod
    def _reconstruct_route(
        parents: np.ndarray, start_xy: tuple[int, int], goal_xy: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        """Return a start-to-goal path encoded as ``(x, y)`` grid cells."""

        route = [goal_xy]
        current = goal_xy
        max_steps = int(parents.shape[0] * parents.shape[1])
        for _ in range(max_steps):
            if current == start_xy:
                return tuple(reversed(route))
            parent = parents[current[1], current[0]]
            if int(parent[0]) < 0 or int(parent[1]) < 0:
                return ()
            current = (int(parent[0]), int(parent[1]))
            route.append(current)
        return ()

    def _candidate_records(self, pose_xy: np.ndarray, heading: float = 0.0) -> list[dict[str, Any]]:
        unknown = self._grid == 0
        free = self._grid == FREE
        if not np.any(free):
            self._last_candidate_pool_stats = {
                "free_cells": 0,
                "frontier_cells": 0,
                "reachable_free_cells": 0,
                "components": 0,
                "in_range_frontier_cells": 0,
                "deferred_frontier_cells": 0,
                "deferred_fallback_records": 0,
                "clear_space_fallback_records": 0,
                "records": 0,
            }
            return []
        costs, parents, start_xy = self._reachable_tree(pose_xy)
        self._candidate_routes = {}
        self._expire_frontier_cooldowns()
        near_unknown = self._dilate(unknown, radius=1)
        frontier = free & near_unknown
        components = self._connected_components(frontier)
        pool_stats = {
            "free_cells": int(np.count_nonzero(free)),
            "frontier_cells": int(np.count_nonzero(frontier)),
            "reachable_free_cells": int(np.count_nonzero(free & np.isfinite(costs))),
            "components": len(components),
            "in_range_frontier_cells": 0,
            "deferred_frontier_cells": 0,
            "deferred_fallback_records": 0,
            "clear_space_fallback_records": 0,
            "records": 0,
        }
        records = []
        deferred_records = []
        for label, component in enumerate(components, start=1):
            cell_count = int(np.count_nonzero(component))
            if cell_count < 3:
                continue
            ys, xs = np.nonzero(component)
            reachable = np.isfinite(costs[ys, xs])
            if not np.any(reachable):
                continue
            reachable_ys, reachable_xs = ys[reachable], xs[reachable]
            reachable_costs = costs[reachable_ys, reachable_xs]
            in_range = (reachable_costs >= self.config.candidate_min_distance_m / self.config.map_resolution_m) & (
                reachable_costs <= self.config.candidate_max_distance_m / self.config.map_resolution_m
            )
            if not np.any(in_range):
                continue
            eligible_ys, eligible_xs, eligible_costs = (
                reachable_ys[in_range],
                reachable_xs[in_range],
                reachable_costs[in_range],
            )
            pool_stats["in_range_frontier_cells"] += int(len(eligible_xs))
            # Apply temporary public-map deferral before spatial NMS.  Filtering
            # only the first three preferred cells would incorrectly discard an
            # otherwise valid component instead of backfilling another point.
            eligible_world = self._grid_to_world(
                np.stack([eligible_xs, eligible_ys], axis=1).astype(np.int32)
            )
            deferred = np.asarray(
                [self._frontier_is_deferred(goal_xy) for goal_xy in eligible_world],
                dtype=bool,
            )
            pool_stats["deferred_frontier_cells"] += int(np.count_nonzero(deferred))
            # Prefer a fresh public frontier whenever there is one.  If every
            # routeable point is temporarily deferred, retain a separate soft
            # fallback pool instead of starving M2 and dropping into blind scan.
            # The fallback candidates keep their public history, so Module-2 can
            # still see that they were already reached or blocked.
            revisit_deferred = bool(np.all(deferred))
            target_records = deferred_records if revisit_deferred else records
            if not revisit_deferred:
                eligible_ys = eligible_ys[~deferred]
                eligible_xs = eligible_xs[~deferred]
                eligible_costs = eligible_costs[~deferred]
            unknown_gain = self._unknown_gain(component, unknown)
            for selected_index in self._spatially_diverse_frontier_indices(
                eligible_xs,
                eligible_ys,
                eligible_costs,
            ):
                goal_cell_xy = (int(eligible_xs[selected_index]), int(eligible_ys[selected_index]))
                route = self._reconstruct_route(parents, start_xy, goal_cell_xy)
                if not route:
                    continue
                grid_xy = np.asarray([goal_cell_xy], dtype=np.int32)
                goal_xy = self._grid_to_world(grid_xy)[0]
                distance = float(costs[goal_cell_xy[1], goal_cell_xy[0]] * self.config.map_resolution_m)
                relative = goal_xy - pose_xy
                relative_heading = self._wrap_angle(math.atan2(-float(relative[1]), float(relative[0])) - heading)
                local_unknown_gain = self._local_unknown_gain(unknown, goal_cell_xy)
                record_id = f"frontier:{goal_cell_xy[0]:03d}:{goal_cell_xy[1]:03d}"
                self._candidate_routes[record_id] = route
                target_records.append(
                    {
                        "candidate_id": record_id,
                        "behavior_type": "EXPLORE",
                        "target_id": f"public_frontier_{self._relative_heading_label(relative_heading)}",
                        "target_name": "unknown_frontier",
                        "goal_xyyaw": [float(goal_xy[0]), float(goal_xy[1]), 0.0],
                        "features": {
                            "distance_m": distance,
                            "exploration_gain": float(local_unknown_gain),
                            "visibility_gain": float(local_unknown_gain),
                            "interaction_cost": 0.0,
                        },
                        "metadata": {
                            "cell_count": cell_count,
                            "map_resolution": self.config.map_resolution_m,
                            "frontier_point": [float(goal_xy[0]), float(goal_xy[1])],
                            "relative_bearing_rad": float(relative_heading),
                            "revisit_deferred": revisit_deferred,
                            "unknown_component_area_m2": float(unknown_gain) * self.config.map_resolution_m**2,
                            "expected_visible_unknown_area_m2": float(unknown_gain) * self.config.map_resolution_m**2,
                        },
                    }
                )
        if not records and deferred_records:
            records = deferred_records
            pool_stats["deferred_fallback_records"] = len(records)
        if not records:
            records = self._clear_space_fallback_records(pose_xy, heading, costs, parents, unknown)
            pool_stats["clear_space_fallback_records"] = len(records)
        records.sort(key=lambda item: (-item["features"]["exploration_gain"], item["features"]["distance_m"], item["candidate_id"]))
        records = records[: self.config.candidate_count]
        self._candidate_routes = {
            str(record["candidate_id"]): self._candidate_routes[str(record["candidate_id"])] for record in records
        }
        pool_stats["records"] = len(records)
        self._last_candidate_pool_stats = pool_stats
        return records

    def _clear_space_fallback_records(
        self,
        pose_xy: np.ndarray,
        heading: float,
        costs: np.ndarray,
        parents: np.ndarray,
        unknown: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Offer M2 safe public waypoints when no known frontier is reachable.

        A conservative depth map can temporarily sever the robot's observed-free
        component from every frontier after a collision.  Continuing a blind
        rotate/forward loop loses the opportunity to select a direction.  These
        records intentionally use only already observed FREE cells and their
        reconstructed public grid routes; moving there can reveal a new frontier
        on a later depth frame.
        """

        if self.config.clear_space_fallback_candidates <= 0:
            return []
        free = self._grid == FREE
        ys, xs = np.nonzero(free & np.isfinite(costs))
        if not len(xs):
            return []
        path_costs = costs[ys, xs]
        min_cells = self.config.candidate_min_distance_m / self.config.map_resolution_m
        max_cells = self.config.clear_space_fallback_max_distance_m / self.config.map_resolution_m
        in_range = (path_costs >= min_cells) & (path_costs <= max_cells)
        if not np.any(in_range):
            return []
        eligible_xs = xs[in_range]
        eligible_ys = ys[in_range]
        eligible_costs = path_costs[in_range]
        # Reuse the spatial NMS used for real frontiers so a broad local region
        # gives M2 several directions instead of a scan-order-biased singleton.
        selected_indices = self._spatially_diverse_frontier_indices(
            eligible_xs,
            eligible_ys,
            eligible_costs,
            limit=self.config.clear_space_fallback_candidates,
        )
        radius = max(1, int(round(self.config.clear_space_fallback_unknown_radius_m / self.config.map_resolution_m)))
        records: list[dict[str, Any]] = []
        start_xy = self._world_to_grid(pose_xy[None, :])[0]
        for selected_index in selected_indices:
            goal_cell_xy = (int(eligible_xs[selected_index]), int(eligible_ys[selected_index]))
            route = self._reconstruct_route(
                parents,
                (int(start_xy[0]), int(start_xy[1])),
                goal_cell_xy,
            )
            if not route:
                continue
            goal_xy = self._grid_to_world(np.asarray([goal_cell_xy], dtype=np.int32))[0]
            local_unknown_gain = self._local_unknown_gain(unknown, goal_cell_xy, radius=radius)
            record_id = f"clear_space:{goal_cell_xy[0]:03d}:{goal_cell_xy[1]:03d}"
            self._candidate_routes[record_id] = route
            relative = goal_xy - pose_xy
            relative_heading = self._wrap_angle(math.atan2(-float(relative[1]), float(relative[0])) - heading)
            records.append(
                {
                        "candidate_id": record_id,
                        "behavior_type": "EXPLORE",
                        "target_id": f"public_clear_space_{self._relative_heading_label(relative_heading)}",
                    "target_name": "observed_free_escape_waypoint",
                    "goal_xyyaw": [float(goal_xy[0]), float(goal_xy[1]), 0.0],
                    "features": {
                        "distance_m": float(costs[goal_cell_xy[1], goal_cell_xy[0]] * self.config.map_resolution_m),
                        "exploration_gain": float(local_unknown_gain),
                        "visibility_gain": float(local_unknown_gain),
                        "interaction_cost": 0.0,
                    },
                    "metadata": {
                        "fallback_clear_space": True,
                        "map_resolution": self.config.map_resolution_m,
                        "frontier_point": [float(goal_xy[0]), float(goal_xy[1])],
                        "relative_bearing_rad": float(relative_heading),
                        "expected_visible_unknown_area_m2": float(local_unknown_gain)
                        * self.config.map_resolution_m**2,
                    },
                }
            )
        return records

    def _local_unknown_gain(
        self,
        unknown: np.ndarray,
        goal_cell_xy: tuple[int, int],
        *,
        radius: int | None = None,
    ) -> int:
        if radius is None:
            radius = max(1, int(round(self.config.clear_space_fallback_unknown_radius_m / self.config.map_resolution_m)))
        y0 = max(0, goal_cell_xy[1] - radius)
        y1 = min(self._shape[0], goal_cell_xy[1] + radius + 1)
        x0 = max(0, goal_cell_xy[0] - radius)
        x1 = min(self._shape[1], goal_cell_xy[0] + radius + 1)
        return int(np.count_nonzero(unknown[y0:y1, x0:x1]))

    @staticmethod
    def _relative_heading_label(relative_heading_rad: float) -> str:
        """Encode a public robot-relative direction in the M2-visible subject ID."""

        degrees = math.degrees(relative_heading_rad)
        magnitude = int(round(abs(degrees) / 15.0) * 15)
        if magnitude <= 15:
            return "ahead"
        if magnitude >= 165:
            return "behind"
        side = "left" if degrees > 0.0 else "right"
        return f"{side}_{magnitude:03d}_deg"

    def _spatially_diverse_frontier_indices(
        self,
        cells_x: np.ndarray,
        cells_y: np.ndarray,
        costs: np.ndarray,
        *,
        limit: int | None = None,
    ) -> list[int]:
        """Choose useful-route frontier points without component scan-order bias."""

        if not len(cells_x):
            return []
        preferred_cells = self.config.candidate_preferred_distance_m / self.config.map_resolution_m
        order = sorted(
            range(len(cells_x)),
            key=lambda index: (
                abs(float(costs[index]) - preferred_cells),
                float(costs[index]),
                int(cells_y[index]),
                int(cells_x[index]),
            ),
        )
        separation_cells = self.config.frontier_candidate_separation_m / self.config.map_resolution_m
        max_candidates = self.config.frontier_candidates_per_component if limit is None else max(0, int(limit))
        if max_candidates <= 0:
            return []
        selected: list[int] = []
        for index in order:
            if any(
                math.hypot(
                    float(cells_x[index] - cells_x[existing]),
                    float(cells_y[index] - cells_y[existing]),
                )
                < separation_cells
                for existing in selected
            ):
                continue
            selected.append(index)
            if len(selected) >= max_candidates:
                break
        return selected

    @staticmethod
    def _unknown_gain(component: np.ndarray, unknown: np.ndarray) -> int:
        border = HabitatInteractiveNavM2Policy._dilate(component, radius=4)
        return int(np.count_nonzero(border & unknown))

    def _inflate_obstacles(self) -> None:
        radius = max(1, int(round(self.config.obstacle_inflation_m / self.config.map_resolution_m)))
        inflated = self._dilate(self._raw_obstacles | self._motion_obstacles, radius=radius)
        # Rebuild the planning projection from raw evidence on every update.
        # In particular, do not use the previous inflated projection as input.
        self._grid.fill(0)
        self._grid[self._observed_free & ~inflated] = FREE
        self._grid[inflated] = OCCUPIED

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        """Boolean square dilation without a heavyweight OpenCV runtime."""

        padded = np.pad(mask.astype(bool), radius, mode="constant")
        height, width = mask.shape
        result = np.zeros_like(mask, dtype=bool)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                result |= padded[dy : dy + height, dx : dx + width]
        return result

    @staticmethod
    def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
        """Return 8-connected boolean components, deterministically."""

        pending = mask.astype(bool).copy()
        height, width = pending.shape
        components: list[np.ndarray] = []
        for y, x in zip(*np.nonzero(pending)):
            if not pending[y, x]:
                continue
            component = np.zeros_like(pending, dtype=bool)
            stack = [(int(y), int(x))]
            pending[y, x] = False
            while stack:
                cy, cx = stack.pop()
                component[cy, cx] = True
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if pending[ny, nx]:
                            pending[ny, nx] = False
                            stack.append((ny, nx))
            components.append(component)
        return components

    def _select_goal(self, records: list[dict[str, Any]], pose_xy: np.ndarray, heading: float) -> _ActiveGoal:
        candidates = build_behavior_candidates(records)
        if any(candidate.interaction_command is not None for candidate in candidates):
            raise AssertionError("unexpected interaction command in M2 candidate pool")
        if any(str(candidate.behavior_type).upper() not in set(self.config.allowed_behavior_types) for candidate in candidates):
            raise AssertionError("unexpected non-navigation behavior in M2 candidate pool")
        target_category = str(self._episode.get("object_category") or "unknown")
        # Scene/episode identifiers are useful to the evaluator's result file,
        # but they are not given to the language model: they are dataset metadata
        # rather than an RGB-D/GPS/Compass observation.
        graph = (
            dict(self._full_ros_graph)
            if getattr(self, "_full_ros_stack", None) is not None
            else {
                "scene_id": "habitat_objectnav_v2",
                "episode_id": "public_episode",
                "nodes": [],
                "edges": [],
            }
        )
        candidate_history = self._candidate_history_for_model(candidates, graph)
        robot_context = {
            "robot_xy": [float(pose_xy[0]), float(pose_xy[1])],
            "heading": float(heading),
            "decision_history": self._decision_history[-8:],
            "candidate_history": candidate_history,
        }
        selected = self._model.select(
            candidates,
            target_context={
                "enabled": True,
                "target_name": target_category,
                "object_labels": [target_category],
                # A historical public surface estimate remains a NAVIGATE
                # candidate, but only a measurement from this control step is
                # represented as currently visible to Module-2.
                "visible": bool(
                    (
                        self._target_track
                        and self._target_track.promoted
                        and self._target_track.last_seen_step == self._step
                    )
                    or any(
                        bool(node.get("is_currently_visible"))
                        and str(node.get("label") or node.get("name") or "").casefold().replace(" ", "_")
                        in goal_label_aliases(target_category)
                        for node in (graph.get("nodes") or [])
                        if isinstance(node, dict)
                    )
                ),
            },
            graph=graph,
            robot_context=robot_context,
            metrics_context={
                "episode_id": str(self._episode.get("episode_id") or ""),
                "step": self._step,
                "policy": self.name,
            },
        )
        # Existing ModelPolicyClient returns None on a transport/model failure.
        # The deterministic fallback keeps an MLLM outage from masquerading as STOP.
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        if selected is None or selected.candidate_id not in candidate_by_id:
            selected = min(candidates, key=lambda item: (item.features.get("distance_m", float("inf")), item.candidate_id))
        if (
            str(selected.behavior_type).upper() not in set(self.config.allowed_behavior_types)
            or selected.interaction_command is not None
        ):
            raise AssertionError("M2 result violated navigation-only external policy contract")
        # Use the currently offered candidate object even if a misbehaving model
        # returned an equivalent object with stale metadata.
        selected = candidate_by_id[selected.candidate_id]
        goal_xy = np.asarray(selected.goal_xyyaw[:2], dtype=np.float32)
        self._decision_history.append(
            {
                "step": self._step,
                "candidate_id": selected.candidate_id,
                "result_source": self._model.last_result_source,
            }
        )
        history_key = candidate_history_key_for(selected, graph)
        previous = self._candidate_history.get(history_key) or self._candidate_history.get(selected.candidate_id, {})
        last_selected_step = int(previous.get("last_selected_step", self._step))
        history_entry = {
            "selection_count": int(previous.get("selection_count", 0)) + 1,
            "last_selected_step": self._step,
            "last_selected_steps_ago": max(0, self._step - last_selected_step),
            "last_result": "PENDING",
            "low_gain_repeat_count": int(previous.get("low_gain_repeat_count", 0)),
            "last_frontier_shrink_m": float(previous.get("last_frontier_shrink_m", 0.0)),
        }
        self._candidate_history[selected.candidate_id] = dict(history_entry)
        self._candidate_history[history_key] = dict(history_entry)
        self._stagnant_steps = 0
        self._trace_public_event(
            "module2_selection",
            candidate_id=selected.candidate_id,
            behavior_type=str(selected.behavior_type),
            result_source=str(self._model.last_result_source),
            candidate_count=len(candidates),
            target_track_offered=any(item.candidate_id == "public_target_track" for item in candidates),
            target_track_selected=selected.candidate_id == "public_target_track",
        )
        if getattr(self, "_full_ros_stack", None) is not None:
            mirror_error = self._full_ros_stack.publish_selection(
                {
                    "step_id": int(self._step),
                    "sequence": int(self._step),
                    "candidate_id": str(selected.candidate_id),
                    "target_id": str(selected.target_id),
                    "target_name": str(selected.target_name),
                    "behavior_type": str(selected.behavior_type).upper(),
                    "goal_xyyaw": [float(value) for value in (selected.goal_xyyaw or [])],
                    "result_source": str(self._model.last_result_source),
                    "interaction_command": None,
                }
            )
            self._trace_public_event(
                "module2_selection_mirror",
                candidate_id=str(selected.candidate_id),
                error=mirror_error,
            )
        return _ActiveGoal(
            selected.candidate_id,
            goal_xy,
            self._step,
            self._candidate_routes.get(selected.candidate_id, ()),
            candidate_history_key=history_key,
        )

    def _candidate_history_for_model(self, candidates: list[Any], graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Refresh M2's stable region history from public adapter outcomes."""

        projected: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            history_key = candidate_history_key_for(candidate, graph)
            history = self._candidate_history.get(candidate.candidate_id) or self._candidate_history.get(history_key)
            if not history:
                continue
            item = dict(history)
            item["last_selected_steps_ago"] = max(
                0,
                self._step - int(item.get("last_selected_step", self._step)),
            )
            # Preserve both forms: the original Module-2 lookup first checks the
            # instantaneous candidate ID and then its stable spatial-region key.
            projected[candidate.candidate_id] = dict(item)
            projected[history_key] = dict(item)
        return projected

    def _route_is_usable(self, pose_xy: np.ndarray, goal: _ActiveGoal) -> bool:
        if goal.candidate_id == "visible_goal":
            return True
        if not goal.route_cells_xy:
            return False
        current_cell = self._world_to_grid(pose_xy[None, :])[0]
        current_xy = (int(current_cell[0]), int(current_cell[1]))
        route_index = goal.route_index
        while route_index < len(goal.route_cells_xy):
            waypoint = self._grid_to_world(np.asarray([goal.route_cells_xy[route_index]], dtype=np.int32))[0]
            if float(np.linalg.norm(waypoint - pose_xy)) > self.config.path_waypoint_reached_distance_m:
                break
            route_index += 1
        if route_index >= len(goal.route_cells_xy):
            return float(np.linalg.norm(goal.xy - pose_xy)) <= self.config.goal_reached_distance_m
        return self._segment_is_traversable(current_xy, goal.route_cells_xy[route_index])

    def _segment_is_traversable(self, start_xy: tuple[int, int], end_xy: tuple[int, int]) -> bool:
        """Check a grid segment without allowing unknown or occupied shortcuts."""

        x0, y0 = start_xy
        x1, y1 = end_xy
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1, dtype=np.int32)
        ys = np.linspace(y0, y1, steps + 1, dtype=np.int32)
        height, width = self._grid.shape
        cells_free = self._grid[ys, xs] == FREE
        cells_free |= self._local_escape_mask(start_xy)[ys, xs]
        # The exact GPS start cell can lie inside a freshly inflated obstacle
        # halo even though the simulator has just confirmed the robot occupies
        # it.  Treat only that first sample as traversable, matching the
        # shortest-path tree; every other segment sample remains observed FREE.
        cells_free[0] = True
        return bool(
            np.all((xs >= 0) & (xs < width))
            and np.all((ys >= 0) & (ys < height))
            and np.all(cells_free)
        )

    def _route_waypoint(
        self, pose_xy: np.ndarray, goal: _ActiveGoal, *, lookahead_m: float | None = None
    ) -> np.ndarray | None:
        """Return the farthest visible route point within a public controller horizon."""

        horizon_m = self.config.path_lookahead_m if lookahead_m is None else max(0.0, float(lookahead_m))

        if not goal.route_cells_xy:
            return None
        current_cell = self._world_to_grid(pose_xy[None, :])[0]
        current_xy = (int(current_cell[0]), int(current_cell[1]))
        while goal.route_index < len(goal.route_cells_xy):
            waypoint = self._grid_to_world(np.asarray([goal.route_cells_xy[goal.route_index]], dtype=np.int32))[0]
            if float(np.linalg.norm(waypoint - pose_xy)) > self.config.path_waypoint_reached_distance_m:
                break
            goal.route_index += 1
        if goal.route_index >= len(goal.route_cells_xy):
            return None
        last_visible = None
        for route_index in range(goal.route_index, len(goal.route_cells_xy)):
            route_cell = goal.route_cells_xy[route_index]
            waypoint = self._grid_to_world(np.asarray([route_cell], dtype=np.int32))[0]
            if float(np.linalg.norm(waypoint - pose_xy)) > horizon_m:
                break
            if not self._segment_is_traversable(current_xy, route_cell):
                break
            last_visible = waypoint
            goal.route_index = route_index
        if last_visible is not None:
            return last_visible
        next_cell = goal.route_cells_xy[goal.route_index]
        if not self._segment_is_traversable(current_xy, next_cell):
            return None
        return self._grid_to_world(np.asarray([next_cell], dtype=np.int32))[0]

    def _pointnav_local_action(
        self,
        observations: dict[str, Any] | None,
        pose_xy: np.ndarray,
        heading: float,
        goal: _ActiveGoal,
        waypoint: np.ndarray | None,
    ) -> dict[str, Any] | None:
        """Ask the optional released PointNav policy to follow one public route.

        Module-2 has already selected ``goal`` from public frontier/track records.
        This local controller receives only the current public normalized depth
        and the relative vector to an already known-free route point.  Its action
        zero is deliberately *not* mapped to ``velocity_stop``: a PointNav local
        arrival prediction cannot establish ObjectNav-v2 valid-viewpoint success.
        """

        client = getattr(self, "_pointnav", None)
        if client is None or observations is None:
            return None
        if self._pointnav_macro_commands:
            action_index, magnitude = self._pointnav_macro_commands.pop(0)
        else:
            if waypoint is None:
                return None
            delta = waypoint - pose_xy
            rho = float(np.linalg.norm(delta))
            target_heading = math.atan2(-float(delta[1]), float(delta[0]))
            theta = self._wrap_angle(target_heading - heading)
            # Match the released controller's recurrent contract: when the
            # local point goal changes materially, start a fresh state instead
            # of pairing old hidden state with a new waypoint.
            anchor_changed = (
                self._pointnav_anchor_xy is not None
                and float(np.linalg.norm(waypoint - self._pointnav_anchor_xy)) > self.config.map_resolution_m
            )
            result = client.act(
                np.asarray(observations["depth"], dtype=np.float32),
                rho,
                theta,
                reset=self._pointnav_reset_pending or anchor_changed,
                context={"step": self._step, "candidate_id": goal.candidate_id},
            )
            self._pointnav_queries += 1
            if result.error or result.action_index is None:
                self._pointnav_failed += 1
                self._reset_pointnav_controller()
                return None
            self._pointnav_reset_pending = False
            action_index = int(result.action_index)
            self._pointnav_anchor_xy = waypoint.copy()
            if action_index == 0:
                # The learned checkpoint has reached its *local* PointNav target
                # estimate. Continue through the public map follower; only the
                # independent ObjectNav visual gate may issue velocity_stop.  The
                # worker has recorded STOP in its recurrent history although
                # Habitat will not execute it, so reset before the next query.
                self._pointnav_stop_predictions += 1
                self._reset_pointnav_controller()
                return None
            # One discrete VLFM action was trained as 0.25 m forward or 30 deg
            # turn. Expand it into individual official 0.1 s VelocityAction
            # calls, including the final fractional control step, while the
            # outer policy still checks map validity and public depth each time.
            if action_index == 1:
                macro = [(1, 1.0)] * 8 + [(1, -1.0 / 3.0)]
            elif action_index == 2:
                macro = [(2, 1.0)] * 11 + [(2, (math.pi / 6.0 - 11.0 * 0.045) / 0.045)]
            elif action_index == 3:
                macro = [(3, 1.0)] * 11 + [(3, (math.pi / 6.0 - 11.0 * 0.045) / 0.045)]
            else:
                self._reset_pointnav_controller()
                return None
            self._pointnav_macro_commands = macro
            action_index, magnitude = self._pointnav_macro_commands.pop(0)

        if action_index == 1:
            if self._last_forward_clearance_m < self.config.forward_min_clearance_m:
                # The worker's depth encoder has a different camera/robot
                # training distribution. Preserve the adapter's public-depth
                # safety boundary and restart its recurrent state after a
                # command that was intentionally not sent to Habitat.  Match the
                # geometric follower's invalid-route accounting so a persistent
                # public obstruction cannot repeatedly recapture this goal.
                self._pointnav_safety_blocks += 1
                self._route_invalid_streak += 1
                if self._route_invalid_streak >= self.config.route_invalid_confirmations:
                    self._defer_active_frontier("pointnav_forward_clearance")
                    self._active_goal = None
                self._reset_pointnav_controller()
                return self._rotate_recovery_action()
            self._pointnav_forward_steps += 1
            self._last_forward_heading = heading
            return {
                "action": "velocity_control",
                "action_args": {
                    "linear_velocity": float(magnitude),
                    "angular_velocity": 0.0,
                    "camera_pitch_angular_velocity": 0.0,
                },
            }
        if action_index in {2, 3}:
            # VLFM's released categorical order is STOP/FORWARD/LEFT/RIGHT.
            # Challenge-v2's positive normalized angular velocity is the same
            # positive Compass-heading direction used by _drive_to_goal.
            self._pointnav_turn_steps += 1
            turn_sign = 1.0 if action_index == 2 else -1.0
            return {
                "action": "velocity_control",
                "action_args": {
                    "linear_velocity": -1.0,
                    "angular_velocity": float(turn_sign * magnitude * self.config.angular_velocity),
                    "camera_pitch_angular_velocity": 0.0,
                },
            }
        self._reset_pointnav_controller()
        return None

    def _drive_to_goal(
        self,
        pose_xy: np.ndarray,
        heading: float,
        goal: _ActiveGoal,
        *,
        observations: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if goal.candidate_id == "visible_goal":
            return self._drive_to_visible_goal(pose_xy, heading, goal)
        pointnav_enabled = observations is not None and getattr(self, "_pointnav", None) is not None
        # Finish a single bounded PointNav macro before frontier-arrival or route
        # maintenance can replace its local target.  This remains interruptible by
        # public depth safety and no-motion recovery in ``act``/``_pointnav_local_action``.
        if pointnav_enabled and self._pointnav_macro_commands:
            pointnav_action = self._pointnav_local_action(observations, pose_xy, heading, goal, None)
            if pointnav_action is not None:
                self.assert_navigation_only(pointnav_action)
                return pointnav_action
        distance_to_goal = float(np.linalg.norm(goal.xy - pose_xy))
        if distance_to_goal <= self.config.goal_reached_distance_m:
            self._defer_active_frontier("goal_reached")
            self._active_goal = None
            self._reset_pointnav_controller()
            # The immediate action below is the first scan turn.  Save only the
            # remaining turns so a configured N-step scan remains exactly N
            # public depth views before another M2 replan is attempted.
            self._frontier_arrival_scan_steps = max(
                self._frontier_arrival_scan_steps,
                max(0, int(self.config.frontier_arrival_scan_steps) - 1),
            )
            return self._rotate_recovery_action()
        waypoint = self._route_waypoint(
            pose_xy,
            goal,
            lookahead_m=(self.config.pointnav_goal_lookahead_m if pointnav_enabled else self.config.path_lookahead_m),
        )
        if waypoint is None:
            # Fresh single-frame depth can temporarily paint an inflated halo
            # across a route segment.  Preserve the route until `_need_replan`
            # has accumulated independent invalid observations; no forward
            # command is emitted here, so this cannot force travel through it.
            # There is no valid public point goal for the optional worker here,
            # so discard any recurrent state rather than resuming a stale macro.
            if pointnav_enabled:
                self._reset_pointnav_controller()
            if self._route_invalid_streak < self.config.route_invalid_confirmations:
                return self._rotate_recovery_action()
            self._defer_active_frontier("route_waypoint_unavailable")
            self._active_goal = None
            return self._rotate_recovery_action()
        if pointnav_enabled:
            pointnav_action = self._pointnav_local_action(observations, pose_xy, heading, goal, waypoint)
            if pointnav_action is not None:
                self.assert_navigation_only(pointnav_action)
                return pointnav_action
        delta = waypoint - pose_xy
        # Compass angles are expressed clockwise in GPS coordinates.
        target_heading = math.atan2(-float(delta[1]), float(delta[0]))
        heading_error = self._wrap_angle(target_heading - heading)
        angular = float(np.clip(heading_error / max(self.config.rotate_threshold_rad, 1e-3), -1.0, 1.0))
        linear = self.config.linear_velocity if abs(heading_error) < self.config.rotate_threshold_rad else -1.0
        if linear > -1.0 + 1e-6 and self._last_forward_clearance_m < self.config.forward_min_clearance_m:
            # Do not wait stationary in front of an observed obstruction.  The
            # next public RGB-D update will replan instead of retrying the same
            # forward segment indefinitely.
            self._route_invalid_streak += 1
            if self._route_invalid_streak >= self.config.route_invalid_confirmations:
                self._defer_active_frontier("forward_clearance")
                self._active_goal = None
            return self._rotate_recovery_action()
        if linear > -1.0 + 1e-6:
            self._last_forward_heading = heading
        action = {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": float(linear),
                "angular_velocity": float(angular * self.config.angular_velocity),
                "camera_pitch_angular_velocity": 0.0,
            },
        }
        self.assert_navigation_only(action)
        return action

    def _empty_frontier_action(self, heading: float) -> dict[str, Any]:
        """Make a short public RGB-D-guarded probe when no frontier is reachable."""

        if (
            self._empty_frontier_forward_count < self.config.empty_frontier_forward_steps
            and self._last_forward_clearance_m >= self.config.forward_min_clearance_m
        ):
            self._empty_frontier_forward_count += 1
            self._last_forward_heading = heading
            action = {
                "action": "velocity_control",
                "action_args": {
                    "linear_velocity": float(self.config.linear_velocity),
                    "angular_velocity": 0.0,
                    "camera_pitch_angular_velocity": 0.0,
                },
            }
            self.assert_navigation_only(action)
            return action
        self._empty_frontier_forward_count = 0
        return self._rotate_recovery_action()

    def _drive_to_visible_goal(
        self, pose_xy: np.ndarray, heading: float, goal: _ActiveGoal
    ) -> dict[str, Any]:
        """Approach a just-localized RGB-D target without frontier tolerance."""

        delta = goal.xy - pose_xy
        target_heading = math.atan2(-float(delta[1]), float(delta[0]))
        heading_error = self._wrap_angle(target_heading - heading)
        distance = float(np.linalg.norm(delta))
        if distance <= 0.03:
            return self._hold_action()
        self._visual_goal_contiguous_steps += 1
        if self._visual_goal_contiguous_steps >= self.config.visual_goal_max_contiguous_steps:
            # A short visual waypoint is an approach aid, not an indefinitely
            # authoritative navigation policy.  Give public-map exploration a
            # bounded turn when repeated re-localizations have not converged.
            self._visual_goal_budget_releases += 1
            self._release_visible_goal()
            return self._rotate_recovery_action()
        angular = float(np.clip(heading_error / max(self.config.rotate_threshold_rad, 1e-3), -1.0, 1.0))
        linear = self.config.linear_velocity if abs(heading_error) < self.config.rotate_threshold_rad else -1.0
        if linear > -1.0 + 1e-6 and self._last_forward_clearance_m < self.config.forward_min_clearance_m:
            # A visually stable box is not privileged over public depth: it can
            # be a false positive, a target behind furniture, or an object the
            # Stretch base cannot safely approach.  Release it only after
            # repeated independent RGB-D observations, then cool it down long
            # enough for Module-2 frontier exploration to make progress.
            self._visual_goal_blocked_steps += 1
            self._visual_goal_clearance_blocks += 1
            if self._visual_goal_blocked_steps >= self.config.visual_goal_blocked_confirmations:
                self._release_visible_goal()
            return self._rotate_recovery_action()
        if linear > -1.0 + 1e-6:
            self._visual_goal_blocked_steps = 0
            self._last_forward_heading = heading
        action = {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": float(linear),
                "angular_velocity": float(angular * self.config.angular_velocity),
                "camera_pitch_angular_velocity": 0.0,
            },
        }
        self.assert_navigation_only(action)
        return action

    def _release_visible_goal(self) -> None:
        """End one blocked visual takeover using only public controller evidence."""

        if self._active_goal is not None and self._active_goal.candidate_id == "visible_goal":
            self._active_goal = None
        self._visible_goal_confirmations = 0
        self._pending_visual_detection = None
        self._awaiting_visual_confirmation = False
        self._visual_goal_blocked_steps = 0
        self._visual_goal_contiguous_steps = 0
        self._visual_goal_cooldown_until_step = max(
            self._visual_goal_cooldown_until_step,
            self._step + self.config.visual_goal_cooldown_steps,
        )
        self._visual_goal_releases += 1

    def _rotate_recovery_action(self) -> dict[str, Any]:
        if self._collision_recovery_steps > 0:
            direction = self._recovery_turn_direction
        else:
            # Keep scanning in a persistent direction rather than tying the
            # turn sign to the M2 retry cadence.  A 12-step turn changes heading
            # by only about 0.54 rad, so alternating there merely sweeps back
            # and forth through the same view and cannot discover side frontiers.
            direction = self._exploration_turn_direction
            self._exploration_scan_steps += 1
            if self._exploration_scan_steps >= self.config.exploration_scan_steps:
                self._exploration_scan_steps = 0
                self._exploration_turn_direction *= -1.0
                # Give the newly exposed direction a few guarded translation
                # probes before beginning the next full camera sweep.
                self._empty_frontier_forward_count = 0
        action = {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": -1.0,
                "angular_velocity": float(direction * self.config.angular_velocity),
                "camera_pitch_angular_velocity": 0.0,
            },
        }
        self.assert_navigation_only(action)
        return action

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
